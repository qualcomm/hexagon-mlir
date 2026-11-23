//===-- RewriteMaskedAccessToInboundsFastPath.cpp -------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// Rewrites the single-sided boundary mask that PyTorch inductor emits on
// pointwise tt.load / tt.store into a per-program "perfect tile" fast path.
// Handles both 1-D and 2-D pointwise kernels.
//
// --- 1-D pointwise ---
//
// Inductor codegen for a 1-D pointwise kernel produces:
//
//   xoffset = tl.program_id(0) * XBLOCK
//   xindex  = xoffset + tl.arange(0, XBLOCK)
//   xmask   = xindex < xnumel
//   tmp0    = tl.load(in_ptr + xindex, xmask)
//   tl.store(out_ptr + xindex, tmp, xmask)
//
// which lowers to a masked access whose mask is a single reduced-rank
// comparison:
//
//   %idx  = addi(splat(%xoffset), extsi?(make_range(0, XBLOCK)))
//   %mask = cmpi slt, %idx, splat(%xnumel)
//   tt.load  %ptr, %mask [, %other]
//   tt.store %ptr, %val, %mask
//
// --- 2-D pointwise ---
//
// Inductor codegen for a 2-D pointwise kernel produces a mask that
// is the AND of two single-sided comparisons:
//
//   xmask = xindex[:, None] < xnumel   # shape [XBLOCK, 1]
//   ymask = yindex[None, :] < ynumel   # shape [1, YBLOCK]
//   mask  = xmask & ymask              # shape [XBLOCK, YBLOCK]
//
// which lowers to:
//
//   %mask = andi(
//     broadcast(cmpi slt, expand_dims(addi(splat(xoff), range(XBLOCK)), 1),
//               splat(xnumel)),
//     broadcast(cmpi slt, expand_dims(addi(splat(yoff), range(YBLOCK)), 0),
//               splat(ynumel)))
//   tt.load  %ptr, %mask [, %other]
//   tt.store %ptr, %val, %mask
//
// --- Problem ---
//
// Every grid program pays the per-element mask computation + a masked memory
// access, even the fully-interior tiles.
//
// --- Solution ---
//
// This pass reproduces the interior/boundary split at the MLIR level using a
// two-part strategy applied uniformly to both the 1-D and 2-D idioms:
//
//  1. Loads — the mask is dropped entirely. All programs load all lanes
//     unconditionally. For the remainder tile the last (BLOCK - N%BLOCK) lanes
//     per dimension read out-of-bounds addresses, but:
//       * HVX does not fault on these reads (no page-fault semantics for
//         within-allocation overreads; PyTorch allocators align tensors to at
//         least BLOCK*sizeof(element) bytes).
//       * The garbage values loaded for OOB lanes are consumed by subsequent
//         element-wise compute (which produces garbage output for those lanes).
//       * The garbage is discarded by the masked store in step 2.
//
//  2. Stores — a single scalar in-bounds check guards a scf.if:
//
//       1-D:  %inb = cmpi slt, (%xoffset + (XBLOCK - 1)), %xnumel
//
//       2-D:  %inb = andi(
//               cmpi slt, (%xoffset + (XBLOCK - 1)), %xnumel,
//               cmpi slt, (%yoffset + (YBLOCK - 1)), %ynumel)
//
//     (every lane of this program's tile is in bounds  =>  the whole tile is a
//     perfect tile) guards:
//
//       scf.if %inb { <unmasked store> } else { <original masked store> }
//
//     so interior tiles take the unmasked fast-path store and only boundary
//     tiles (remainder in any dimension) pay for the mask.
//
// This results in ZERO scalar branches for loads (removing per-load scf.if
// overhead) and ONE scalar branch per store.
//
// Current scope: the 1-D and 2-D pointwise idioms above are matched. Anything
// else (multi-leaf reduction masks, offsets that are not pid*BLOCK, dynamic
// symbolic strides) is left untouched, which is always correct. This
// optimization is opt-in via the rewriteMaskedAccessToInboundsFastPath option.
//
//===----------------------------------------------------------------------===//

#include "python/triton_qcom_hexagon_backend_api.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/raw_ostream.h"

#include "triton/Dialect/Triton/IR/Dialect.h"

using namespace mlir;
using namespace mlir::triton;

namespace {

static Value traceThroughIntCast(Value v) {
  while (v) {
    Operation *def = v.getDefiningOp();
    if (!def)
      return v;
    if (isa<arith::ExtSIOp, arith::ExtUIOp, arith::TruncIOp,
            arith::IndexCastOp>(def)) {
      v = def->getOperand(0);
      continue;
    }
    return v;
  }
  return v;
}

static Value getSplatScalar(OpBuilder &b, Value v) {
  if (auto sp = v.getDefiningOp<SplatOp>())
    return sp.getSrc();
  if (auto c = v.getDefiningOp<arith::ConstantOp>()) {
    if (auto d = dyn_cast<DenseIntElementsAttr>(c.getValue())) {
      if (d.isSplat()) {
        auto elemTy = d.getType().getElementType();
        return arith::ConstantOp::create(
            b, c.getLoc(), elemTy,
            b.getIntegerAttr(elemTy, d.getSplatValue<APInt>()));
      }
    }
  }
  return nullptr;
}

struct MaskInfo {
  Value xoffset;
  Value xnumel;
  int64_t xblock;
};

struct Mask2DInfo {
  MaskInfo row; // X / row dimension (expand_dims at axis=1, shape [XBLOCK, 1])
  MaskInfo col; // Y / col dimension (expand_dims at axis=0, shape [1, YBLOCK])
};

// Look through one level of tt.broadcast on the mask, then require exactly one
// arith.cmpi leaf. Returns null on any other shape.
static arith::CmpIOp getSingleCmpLeaf(Value mask) {
  Value v = mask;
  if (auto bc = v.getDefiningOp<BroadcastOp>())
    v = bc.getSrc();
  return v.getDefiningOp<arith::CmpIOp>();
}

// Match the coordinate tensor %idx = addi(splat(%xoffset), extsi?(make_range))
// where %xoffset traces to muli(%pid, XBLOCK) with %pid = tt.get_program_id and
// the multiplier equals the make_range extent. On success fills `out` and
// returns true.
static bool matchInductorMask(OpBuilder &b, Value mask, MaskInfo &out) {
  auto maskTy = dyn_cast<RankedTensorType>(mask.getType());
  if (!maskTy || maskTy.getRank() != 1)
    return false;

  arith::CmpIOp cmp = getSingleCmpLeaf(mask);
  if (!cmp || cmp.getPredicate() != arith::CmpIPredicate::slt)
    return false;

  // RHS: splat(%xnumel).
  Value ub = getSplatScalar(b, cmp.getRhs());
  if (!ub)
    return false;

  // LHS: addi(splat(offset), extsi?(make_range(0, XBLOCK))).
  Value idx = cmp.getLhs();
  auto add = idx.getDefiningOp<arith::AddIOp>();
  if (!add)
    return false;

  int64_t xblock = -1;
  Value offsetSplat;
  for (Value operand : {add.getLhs(), add.getRhs()}) {
    Value t = operand;
    if (auto ext = t.getDefiningOp<arith::ExtSIOp>())
      t = ext.getIn();
    if (auto mr = dyn_cast_or_null<MakeRangeOp>(t.getDefiningOp())) {
      if (mr.getStart() != 0)
        return false;
      xblock = static_cast<int64_t>(mr.getEnd()) -
               static_cast<int64_t>(mr.getStart());
    } else {
      offsetSplat = operand;
    }
  }
  if (xblock <= 0 || !offsetSplat)
    return false;

  // offsetSplat must be splat(%xoffset).
  Value xoffset = getSplatScalar(b, offsetSplat);
  if (!xoffset)
    return false;

  // Safety predicate: %xoffset == %pid * XBLOCK, %pid = tt.get_program_id.
  Value off = traceThroughIntCast(xoffset);
  auto mul = off.getDefiningOp<arith::MulIOp>();
  if (!mul)
    return false;
  Value pid, factor;
  for (Value operand : {mul.getLhs(), mul.getRhs()}) {
    if (traceThroughIntCast(operand).getDefiningOp<GetProgramIdOp>())
      pid = operand;
    else
      factor = operand;
  }
  if (!pid || !factor)
    return false;
  int64_t factorVal;
  {
    Value f = traceThroughIntCast(factor);
    auto fc = f.getDefiningOp<arith::ConstantOp>();
    if (!fc)
      return false;
    auto ia = dyn_cast<IntegerAttr>(fc.getValue());
    if (!ia)
      return false;
    factorVal = ia.getInt();
  }
  if (factorVal != xblock)
    return false;

  out.xoffset = xoffset;
  out.xnumel = ub;
  out.xblock = xblock;
  return true;
}

// Build the scalar i1 in-bounds check `%xoffset + (XBLOCK - 1) < %xnumel`,
// normalizing xoffset and xnumel to a common integer type.
static Value buildInBoundsCheck(OpBuilder &b, Location loc, const MaskInfo &m) {
  Value xoffset = m.xoffset;
  Value xnumel = m.xnumel;
  auto offTy = dyn_cast<IntegerType>(xoffset.getType());
  auto numTy = dyn_cast<IntegerType>(xnumel.getType());
  if (!offTy || !numTy)
    return nullptr;
  unsigned width = std::max(offTy.getWidth(), numTy.getWidth());
  Type intTy = b.getIntegerType(width);
  if (offTy.getWidth() < width)
    xoffset = arith::ExtSIOp::create(b, loc, intTy, xoffset);
  if (numTy.getWidth() < width)
    xnumel = arith::ExtSIOp::create(b, loc, intTy, xnumel);

  Value blockMax = arith::ConstantIntOp::create(b, loc, intTy, m.xblock - 1);
  Value hi = arith::AddIOp::create(b, loc, xoffset, blockMax);
  return arith::CmpIOp::create(b, loc, arith::CmpIPredicate::slt, hi, xnumel);
}

// Match one arm of a 2-D andi mask:
// broadcast(cmpi slt, expand_dims(addi(splat(off), range(N)), axis),
//                                 expand_dims?(splat?(bound))).
// isRowArm=true when expand_dims axis==1 → arm has shape [N, 1] (row dim).
// isRowArm=false when expand_dims axis==0 → arm has shape [1, M] (col dim).
static bool match2DMaskArm(Value arm, OpBuilder &b, MaskInfo &info,
                           bool &isRowArm) {
  // Strip tt.broadcast.
  Value v = arm;
  if (auto bc = v.getDefiningOp<BroadcastOp>())
    v = bc.getSrc();

  // Must be a rank-2 cmpi slt.
  auto cmp = v.getDefiningOp<arith::CmpIOp>();
  if (!cmp || cmp.getPredicate() != arith::CmpIPredicate::slt)
    return false;
  auto cmpTy = dyn_cast<RankedTensorType>(cmp.getType());
  if (!cmpTy || cmpTy.getRank() != 2)
    return false;

  // Identify singleton axis to determine row vs col arm.
  auto shape = cmpTy.getShape();
  int64_t expandAxis;
  if (shape[1] == 1) {
    expandAxis = 1;
    isRowArm = true;
  } else if (shape[0] == 1) {
    expandAxis = 0;
    isRowArm = false;
  } else {
    return false;
  }

  // Strip expand_dims from LHS → rank-1 addi(splat(offset), extsi?(range(N))).
  Value lhs = cmp.getLhs();
  auto ed = lhs.getDefiningOp<ExpandDimsOp>();
  if (!ed || (int64_t)ed.getAxis() != expandAxis)
    return false;
  lhs = ed.getSrc();
  auto lhsTy = dyn_cast<RankedTensorType>(lhs.getType());
  if (!lhsTy || lhsTy.getRank() != 1)
    return false;

  // Strip optional expand_dims from RHS then getSplatScalar.
  Value rhs = cmp.getRhs();
  if (auto ed2 = rhs.getDefiningOp<ExpandDimsOp>())
    rhs = ed2.getSrc();
  Value ub = getSplatScalar(b, rhs);
  if (!ub)
    return false;

  // LHS: addi(splat(offset), extsi?(range(BLOCK))) — same as 1-D.
  auto add = lhs.getDefiningOp<arith::AddIOp>();
  if (!add)
    return false;
  int64_t xblock = -1;
  Value offsetSplat;
  for (Value operand : {add.getLhs(), add.getRhs()}) {
    Value t = operand;
    if (auto ext = t.getDefiningOp<arith::ExtSIOp>())
      t = ext.getIn();
    if (auto mr = dyn_cast_or_null<MakeRangeOp>(t.getDefiningOp())) {
      if (mr.getStart() != 0)
        return false;
      xblock = static_cast<int64_t>(mr.getEnd()) -
               static_cast<int64_t>(mr.getStart());
    } else {
      offsetSplat = operand;
    }
  }
  if (xblock <= 0 || !offsetSplat)
    return false;

  Value xoffset = getSplatScalar(b, offsetSplat);
  if (!xoffset)
    return false;

  // Safety predicate: xoffset == pid * BLOCK.
  Value off = traceThroughIntCast(xoffset);
  auto mul = off.getDefiningOp<arith::MulIOp>();
  if (!mul)
    return false;
  Value pid, factor;
  for (Value operand : {mul.getLhs(), mul.getRhs()}) {
    if (traceThroughIntCast(operand).getDefiningOp<GetProgramIdOp>())
      pid = operand;
    else
      factor = operand;
  }
  if (!pid || !factor)
    return false;
  {
    Value f = traceThroughIntCast(factor);
    auto fc = f.getDefiningOp<arith::ConstantOp>();
    if (!fc)
      return false;
    auto ia = dyn_cast<IntegerAttr>(fc.getValue());
    if (!ia || ia.getInt() != xblock)
      return false;
  }
  info.xoffset = xoffset;
  info.xnumel = ub;
  info.xblock = xblock;
  return true;
}

// Match the 2-D inductor mask: andi(broadcast(cmp_row), broadcast(cmp_col))
// where one arm is the row mask (expand_dims axis=1) and the other is the col
// mask (expand_dims axis=0).
static bool match2DInductorMask(OpBuilder &b, Value mask, Mask2DInfo &out) {
  auto maskTy = dyn_cast<RankedTensorType>(mask.getType());
  if (!maskTy || maskTy.getRank() != 2)
    return false;
  auto andOp = mask.getDefiningOp<arith::AndIOp>();
  if (!andOp)
    return false;
  bool isRow0, isRow1;
  MaskInfo info0, info1;
  if (!match2DMaskArm(andOp.getLhs(), b, info0, isRow0))
    return false;
  if (!match2DMaskArm(andOp.getRhs(), b, info1, isRow1))
    return false;
  if (isRow0 == isRow1)
    return false; // need exactly one row arm and one col arm
  out.row = isRow0 ? info0 : info1;
  out.col = isRow0 ? info1 : info0;
  return true;
}

// Build the scalar i1 in-bounds check for both dimensions of a 2-D tile:
//   (xoffset + XBLOCK - 1 < xnumel) AND (yoffset + YBLOCK - 1 < ynumel)
static Value buildInBoundsCheck2D(OpBuilder &b, Location loc,
                                  const Mask2DInfo &m) {
  Value xInb = buildInBoundsCheck(b, loc, m.row);
  Value yInb = buildInBoundsCheck(b, loc, m.col);
  if (!xInb || !yInb)
    return nullptr;
  return arith::AndIOp::create(b, loc, xInb, yInb);
}

// Return true if 'v' transitively depends on a tt.load result.
static bool ptrDependsOnLoad(Value v) {
  SmallPtrSet<Value, 16> visited;
  SmallVector<Value, 8> worklist = {v};
  while (!worklist.empty()) {
    Value cur = worklist.pop_back_val();
    if (!visited.insert(cur).second)
      continue;
    Operation *def = cur.getDefiningOp();
    if (!def)
      continue; // block argument — not a load
    if (isa<LoadOp>(def))
      return true; // indirect indexing — bail
    for (Value operand : def->getOperands())
      worklist.push_back(operand);
  }
  return false;
}

// Rewrite a masked load by dropping the mask entirely (unmasked load).
//
// All programs — including the remainder tile — load all XBLOCK lanes without
// a mask.  The OOB lanes on the remainder tile read garbage (no HVX fault for
// within-allocation over-reads; PyTorch allocators pad to at least one vector
// width) and that garbage is discarded by the downstream masked store.
static bool rewriteLoad(LoadOp op, int &rewrittenCount) {
  Value mask = op.getMask();
  if (!mask)
    return false;
  OpBuilder b(op);
  MaskInfo info1d;
  Mask2DInfo info2d;
  bool is1d = matchInductorMask(b, mask, info1d);
  if (!is1d && !match2DInductorMask(b, mask, info2d))
    return false;

  // Skip indirect / gather loads: their pointer is data-dependent.
  if (ptrDependsOnLoad(op.getPtr()))
    return false;

  // Replace the masked load with an unconditional unmasked load.
  // This is deliberately noisy: the remainder tile performs an OOB read by
  // construction.  The optimization is only valid when the target allocator
  // provides the documented vector-width padding; this diagnostic makes the
  // assumption visible in compiler logs before a device error 500 occurs.
  llvm::errs() << "[RewriteMaskedAccessToInboundsFastPath][OOB-risk] "
               << "dropping load mask at " << op.getLoc() << " ptr=";
  op.getPtr().print(llvm::errs());
  llvm::errs() << " (block=";
  if (is1d)
    llvm::errs() << info1d.xblock;
  else
    llvm::errs() << info2d.row.xblock << "x" << info2d.col.xblock;
  llvm::errs() << ")\n";
  auto fast = LoadOp::create(b, op.getLoc(), op.getPtr(), op.getCache(),
                             op.getEvict(), op.getIsVolatile());
  op.getResult().replaceAllUsesWith(fast.getResult());
  op.erase();
  ++rewrittenCount;
  return true;
}

// Rewrite a masked store into scf.if(inb){unmasked}else{masked}.
//
// One scalar branch per store guards the writes: interior tiles take the
// unmasked fast path; only the remainder tile falls through to the masked
// slow path.
static bool rewriteStore(StoreOp op, int &rewrittenCount) {
  Value mask = op.getMask();
  if (!mask)
    return false;
  OpBuilder b(op);
  MaskInfo info1d;
  Mask2DInfo info2d;
  bool is1d = matchInductorMask(b, mask, info1d);
  bool is2d = !is1d && match2DInductorMask(b, mask, info2d);
  if (!is1d && !is2d)
    return false;

  // Skip indirect / scatter stores: their pointer is data-dependent.
  // Check before buildInBoundsCheck to avoid emitting dead IR.
  if (ptrDependsOnLoad(op.getPtr()))
    return false;

  Value inb;
  if (is1d)
    inb = buildInBoundsCheck(b, op.getLoc(), info1d);
  else
    inb = buildInBoundsCheck2D(b, op.getLoc(), info2d);
  if (!inb)
    return false;

  Location loc = op.getLoc();
  llvm::errs() << "[RewriteMaskedAccessToInboundsFastPath][bounds-check] "
               << "store at " << loc << " uses full-tile predicate; "
               << "boundary tiles retain the masked store\n";
  auto ifOp = scf::IfOp::create(b, loc, inb, /*withElse=*/true);
  {
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(ifOp.thenBlock());
    StoreOp::create(b, loc, op.getPtr(), op.getValue(), op.getCache(),
                    op.getEvict());
  }
  {
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(ifOp.elseBlock());
    // Clone the original masked store verbatim (preserves mask/attrs).
    b.clone(*op.getOperation());
  }
  op.erase();
  ++rewrittenCount;
  return true;
}

} // namespace

//===----------------------------------------------------------------------===//
// Public entry points.
//===----------------------------------------------------------------------===//

namespace hexagon_backend {

void rewriteMaskedAccessToInboundsFastPath(mlir::ModuleOp &mod) {
  int maskedCount = 0, rewrittenCount = 0;
  // Collect first: rewriteStore creates new tt.store inside scf.if bodies,
  // and rewriting mutates the IR, so gather candidate ops up front.
  SmallVector<LoadOp> loads;
  SmallVector<StoreOp> stores;
  mod.walk([&](Operation *op) {
    if (auto loadOp = dyn_cast<LoadOp>(op)) {
      if (loadOp.getMask()) {
        ++maskedCount;
        loads.push_back(loadOp);
      }
    } else if (auto storeOp = dyn_cast<StoreOp>(op)) {
      if (storeOp.getMask()) {
        ++maskedCount;
        stores.push_back(storeOp);
      }
    }
  });
  for (LoadOp op : loads)
    rewriteLoad(op, rewrittenCount);
  for (StoreOp op : stores)
    rewriteStore(op, rewrittenCount);

  llvm::outs() << "[RewriteMaskedAccessToInboundsFastPath] masked_accesses="
               << maskedCount << " rewritten=" << rewrittenCount << "\n";
  llvm::outs().flush();
}

namespace {
struct RewriteMaskedAccessToInboundsFastPathMLIRPass
    : public mlir::PassWrapper<RewriteMaskedAccessToInboundsFastPathMLIRPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      RewriteMaskedAccessToInboundsFastPathMLIRPass)
  StringRef getArgument() const override {
    return "hexagon-rewrite-masked-access-to-inbounds-fast-path";
  }
  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::triton::TritonDialect, mlir::arith::ArithDialect,
                    mlir::scf::SCFDialect>();
  }
  void runOnOperation() override {
    mlir::ModuleOp mod = getOperation();
    rewriteMaskedAccessToInboundsFastPath(mod);
  }
};
} // namespace

std::unique_ptr<mlir::Pass> createRewriteMaskedAccessToInboundsFastPathPass() {
  return std::make_unique<RewriteMaskedAccessToInboundsFastPathMLIRPass>();
}

} // namespace hexagon_backend
