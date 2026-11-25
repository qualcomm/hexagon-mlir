//===-- EliminateSafeBoundaryChecks.cpp -----------------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// Eliminates redundant boundary masks on tt.load / tt.store when safe at
// compile time.
//
// tl.make_tensor_descriptor loads/stores are lowered by Triton's
// RewriteTensorDescriptorToPointer pass into a masked pointer access:
//
//   coord_d = expand_dims(offset_d + tt.make_range(0, blockSize_d)) // reduced
//   rank valid_d = tt.broadcast((coord_d >= 0) & (coord_d < tensorSize_d)) mask
//   = valid_0 & valid_1 & ... tt.load %ptrs, %mask [, %other]
//
// This pass decomposes the mask into per-dimension boundary terms (looking
// through the per-dim tt.broadcast) and removes those where tensorSize_d is a
// compile-time constant divisible by blockSize_d. When all dimension terms are
// safe, the mask (and optional other) operand is dropped entirely.
//
// ASSUMPTION: the caller launches with grid[dim] = cdiv(tensorSize[dim],
// blockSize[dim]) (the standard Triton pattern). This ensures no tile exceeds
// bounds when tensorSize[dim] % blockSize[dim] == 0. This optimization is
// opt-in via the eliminateSafeBoundaryChecks option.
//
//===----------------------------------------------------------------------===//

#include "python/triton_qcom_hexagon_backend_api.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

#include "triton/Dialect/Triton/IR/Dialect.h"

using namespace mlir;
using namespace mlir::triton;

namespace {

// Trace a scalar integer constant through value-preserving casts.
static bool getConstantIntThroughCast(Value v, int64_t &out) {
  while (v) {
    if (auto cst = v.getDefiningOp<arith::ConstantOp>()) {
      if (auto ia = dyn_cast<IntegerAttr>(cst.getValue())) {
        out = ia.getInt();
        return true;
      }
      return false;
    }
    Operation *def = v.getDefiningOp();
    if (!def)
      return false;
    if (isa<arith::ExtSIOp, arith::ExtUIOp, arith::TruncIOp,
            arith::IndexCastOp>(def)) {
      v = def->getOperand(0);
      continue;
    }
    return false;
  }
  return false;
}

// Given a splat or dense constant tensor, extract its scalar value.
static bool getSplatConstant(Value v, int64_t &out) {
  if (auto sp = v.getDefiningOp<SplatOp>())
    return getConstantIntThroughCast(sp.getSrc(), out);
  if (auto c = v.getDefiningOp<arith::ConstantOp>()) {
    if (auto d = dyn_cast<DenseIntElementsAttr>(c.getValue())) {
      if (d.isSplat()) {
        out = d.getSplatValue<APInt>().getSExtValue();
        return true;
      }
    }
  }
  return false;
}

// Return true if the tensor value is a zero constant (dense<0> or splat(0)).
static bool isZeroTensor(Value v) {
  int64_t val;
  return getSplatConstant(v, val) && val == 0;
}

// Given a coordinate tensor (the LHS of a boundary comparison), trace through
// expand_dims <- addi(splat(offset), extsi?(make_range(0, E))) and return the
// per-dim block size E (= make_range end - start).  Returns -1 on failure.
// The coordinate is compared at reduced rank (the broadcast sits on the i1
// result, not on the coordinate), so no BroadcastOp appears on this chain.
static int64_t getCoordExtent(Value coord) {
  while (coord) {
    Operation *def = coord.getDefiningOp();
    if (!def)
      return -1;
    if (auto ed = dyn_cast<ExpandDimsOp>(def)) {
      coord = ed.getSrc();
      continue;
    }
    if (auto add = dyn_cast<arith::AddIOp>(def)) {
      // One operand is splat(offset), the other traces to make_range.
      for (Value o : {add.getLhs(), add.getRhs()}) {
        Value t = o;
        if (auto ext = t.getDefiningOp<arith::ExtSIOp>())
          t = ext.getIn();
        if (auto mr = dyn_cast_or_null<MakeRangeOp>(t.getDefiningOp()))
          return static_cast<int64_t>(mr.getEnd()) -
                 static_cast<int64_t>(mr.getStart());
      }
      return -1;
    }
    if (auto ext = dyn_cast<arith::ExtSIOp>(def)) {
      coord = ext.getIn();
      continue;
    }
    if (auto mr = dyn_cast<MakeRangeOp>(def))
      return static_cast<int64_t>(mr.getEnd()) -
             static_cast<int64_t>(mr.getStart());
    return -1;
  }
  return -1;
}

// Flatten an andi-tree into its leaf values, looking through tt.broadcast.
static void collectAndLeaves(Value v, SmallVectorImpl<Value> &leaves) {
  if (auto a = v.getDefiningOp<arith::AndIOp>()) {
    collectAndLeaves(a.getLhs(), leaves);
    collectAndLeaves(a.getRhs(), leaves);
  } else if (auto bc = v.getDefiningOp<BroadcastOp>()) {
    // The descriptor->pointer rewrite emits per-dimension validity as
    //   andi(broadcast(andi(sge, slt)), broadcast(andi(sge, slt)))
    // Look through the broadcast so the cmpi leaves remain reachable.
    collectAndLeaves(bc.getSrc(), leaves);
  } else {
    leaves.push_back(v);
  }
}

// Build a left-associative andi tree from a list of i1-tensor values.
static Value buildAndTree(OpBuilder &b, Location loc, ArrayRef<Value> vals) {
  assert(!vals.empty());
  Value acc = vals.front();
  for (size_t i = 1; i < vals.size(); ++i)
    acc = arith::AndIOp::create(b, loc, acc, vals[i]);
  return acc;
}

// Per-coordinate dimension info used for boundary-check classification.
struct DimInfo {
  Value lowerLeaf;        // cmpi sge, coord, 0
  Value upperLeaf;        // cmpi slt, coord, splat(tensorSize)
  int64_t tensorSize = 0; // shape extent for this dimension
  int64_t blockSize = 0;  // tile extent (make_range end - start)
};

// Try to eliminate safe boundary mask terms on a masked load or store.
// Returns true if the op was modified or replaced.
template <typename OpTy>
static bool tryEliminateMask(OpTy op, int &eliminatedCount) {
  Value mask = op.getMask();
  if (!mask)
    return false;

  // Flatten the andi tree.
  SmallVector<Value> leaves;
  collectAndLeaves(mask, leaves);
  if (leaves.empty())
    return false;

  // Classify leaves into per-coordinate DimInfo entries.
  llvm::DenseMap<Value, DimInfo> dims;
  for (Value leaf : leaves) {
    auto cmp = leaf.getDefiningOp<arith::CmpIOp>();
    if (!cmp)
      continue;
    auto pred = cmp.getPredicate();
    Value lhs = cmp.getLhs();
    Value rhs = cmp.getRhs();

    if (pred == arith::CmpIPredicate::sge && isZeroTensor(rhs)) {
      dims[lhs].lowerLeaf = leaf;
    } else if (pred == arith::CmpIPredicate::slt) {
      int64_t shapeVal;
      if (getSplatConstant(rhs, shapeVal)) {
        dims[lhs].upperLeaf = leaf;
        dims[lhs].tensorSize = shapeVal;
        dims[lhs].blockSize = getCoordExtent(lhs);
      }
    }
  }

  // A coord is safe if both bounds are present, blockSize is known, and
  // tensorSize is divisible by blockSize.
  llvm::DenseSet<Value> removableLeaves;
  for (auto &[coord, d] : dims) {
    if (d.lowerLeaf && d.upperLeaf && d.blockSize > 0 &&
        d.tensorSize % d.blockSize == 0) {
      removableLeaves.insert(d.lowerLeaf);
      removableLeaves.insert(d.upperLeaf);
    }
  }

  if (removableLeaves.empty())
    return false;

  // Collect remaining (unsafe) leaves.
  SmallVector<Value> remaining;
  for (Value leaf : leaves) {
    if (!removableLeaves.contains(leaf))
      remaining.push_back(leaf);
  }

  if (remaining.empty()) {
    // All boundary terms are safe — drop mask (and other) entirely.
    // Emit the proof inputs as a diagnostic.  If the launch grid does not
    // actually satisfy cdiv(tensorSize, blockSize), this rewrite can turn a
    // masked access into an OOB access and eventually surface as device error
    // 500.  The pass cannot validate the host launch, so record the assumption
    // at the point where it is made.
    llvm::errs() << "[EliminateSafeBoundaryChecks][OOB-risk] "
                 << "removing all boundary checks at " << op.getLoc() << " for "
                 << (std::is_same_v<OpTy, LoadOp> ? "load" : "store") << ":";
    for (auto &[coord, d] : dims) {
      if (d.lowerLeaf && d.upperLeaf)
        llvm::errs() << " tensor_size=" << d.tensorSize
                     << " block_size=" << d.blockSize;
    }
    llvm::errs() << " (requires grid=cdiv(tensor_size, block_size))\n";
    OpBuilder b(op);
    if constexpr (std::is_same_v<OpTy, LoadOp>) {
      auto newLoad = LoadOp::create(b, op.getLoc(), op.getPtr(), op.getCache(),
                                    op.getEvict(), op.getIsVolatile());
      op.getResult().replaceAllUsesWith(newLoad.getResult());
      op.erase();
    } else {
      StoreOp::create(b, op.getLoc(), op.getPtr(), op.getValue(), op.getCache(),
                      op.getEvict());
      op.erase();
    }
    ++eliminatedCount;
    return true;
  }

  // Partial elimination: rebuild mask from remaining leaves.
  // After looking through tt.broadcast a remaining leaf may have lower rank
  // than the mask; only rebuild when all types match, otherwise bail.
  for (Value leaf : remaining)
    if (leaf.getType() != mask.getType())
      return false;
  OpBuilder b(op);
  Value newMask = buildAndTree(b, op.getLoc(), remaining);
  op.getMaskMutable().assign(newMask);
  return true;
}

} // namespace

//===----------------------------------------------------------------------===//
// Public entry points.
//===----------------------------------------------------------------------===//

namespace hexagon_backend {

void eliminateSafeBoundaryChecks(mlir::ModuleOp &mod) {
  int loadCount = 0, eliminatedCount = 0;
  mod.walk([&](Operation *op) {
    if (auto loadOp = dyn_cast<LoadOp>(op)) {
      if (loadOp.getMask()) {
        ++loadCount;
        tryEliminateMask(loadOp, eliminatedCount);
      }
    } else if (auto storeOp = dyn_cast<StoreOp>(op)) {
      if (storeOp.getMask()) {
        int dummy = 0;
        tryEliminateMask(storeOp, dummy);
      }
    }
  });
  llvm::outs() << "[EliminateSafeBoundaryChecks] masked_loads=" << loadCount
               << " fully_eliminated=" << eliminatedCount << "\n";
  llvm::outs().flush();
}

namespace {
struct EliminateSafeBoundaryChecksMLIRPass
    : public mlir::PassWrapper<EliminateSafeBoundaryChecksMLIRPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      EliminateSafeBoundaryChecksMLIRPass)
  StringRef getArgument() const override {
    return "hexagon-eliminate-safe-boundary-checks";
  }
  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::triton::TritonDialect, mlir::arith::ArithDialect>();
  }
  void runOnOperation() override {
    mlir::ModuleOp mod = getOperation();
    eliminateSafeBoundaryChecks(mod);
  }
};
} // namespace

std::unique_ptr<mlir::Pass> createEliminateSafeBoundaryChecksPass() {
  return std::make_unique<EliminateSafeBoundaryChecksMLIRPass>();
}

} // namespace hexagon_backend
