//===- VectorRowReducePass.cpp - 2-D row reduce to vector fold + butterfly -===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// A 2-D last-dim `linalg.reduce` (f16/f32 max/add) that reaches the Hexagon
// backend is scalarized: ConvertLinalgToLoops turns it into a per-lane
// scf.for chain, and HexagonISelLowering only gives VECREDUCE_ADD an HVX DAG
// combine -- a `llvm.vector.reduce.fmax` becomes a serial
// memw/sfcmp/mux/vinsert chain per lane, which was 55% of the FA softmax
// kernel body.
//
// This pass replaces the reduce before that point with explicit vector IR:
// each row's 128-byte chunks are folded elementwise (one vector max/add per
// chunk), then a vror butterfly -- rotate the accumulator by half the
// register and fold, halving the lane span each step, llama.cpp
// hvx-reduce.h style -- leaves every lane of one HVX vector holding the
// row's reduction. The original outs init is folded in and lane 0 is stored
// back, so the scalar contract of the reduce (init folded in, one value per
// row) is preserved; the butterfly's all-lanes-redundant result is what a
// later consumer optimization (splats instead of extract) can exploit.
//
// Everything it cannot prove static, contiguous and whole-vector is left for
// the scalar path. Off by default (`enable-vector-row-reduce`): the knob is
// the device A/B switch, not a correctness guard.
//
//===----------------------------------------------------------------------===//

#include "hexagon/Conversion/LinalgToLLVM/Passes.h"
#include "hexagon/Dialect/Hvx/IR/HvxDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"

#include <functional>
#include <optional>

using namespace mlir;

// At global scope, like the rest of this family: the generated pass base lands
// in ::impl.
#define GEN_PASS_DEF_VECTORROWREDUCE
#include "hexagon/Conversion/LinalgToLLVM/Passes.h.inc"

namespace {

/// One HVX vector register, in bytes. The kernels this pipeline serves run
/// with HVX length 128B (v79); a row chunk of this width is what the butterfly
/// needs.
static constexpr int kHvxVectorBytes = 128;

/// The row byte width the pass requires: a whole number of HVX vectors.
struct RowShape {
  int64_t rows;      // dim 0
  int64_t cols;      // dim 1
  int64_t elemBytes; // f16 -> 2, f32 -> 4
};

static std::optional<RowShape> rowShapeOf(Value memref) {
  auto type = dyn_cast<MemRefType>(memref.getType());
  if (!type || !type.hasStaticShape() || type.getRank() != 2)
    return std::nullopt;
  Type elemTy = type.getElementType();
  int64_t elemBytes;
  if (elemTy.isF32())
    elemBytes = 4;
  else if (elemTy.isF16())
    elemBytes = 2;
  else
    return std::nullopt;
  // A row must be contiguous (the vector reads walk it with unit stride); the
  // outer stride may be anything -- each row is addressed through its own
  // subview.
  SmallVector<int64_t> strides;
  int64_t offset;
  if (failed(type.getStridesAndOffset(strides, offset)) || strides.empty() ||
      strides.back() != 1)
    return std::nullopt;
  int64_t cols = type.getDimSize(1);
  if ((cols * elemBytes) % kHvxVectorBytes != 0)
    return std::nullopt;
  return RowShape{type.getDimSize(0), cols, elemBytes};
}

/// The body contract: exactly one binary maxnumf/addf over the two block
/// args, yielded directly. Returns the fastmath flags it must be rebuilt
/// with, or nothing when the body does not match.
static std::optional<arith::FastMathFlags>
matchFoldBody(linalg::ReduceOp op,
              std::function<Value(OpBuilder &, Location, arith::FastMathFlags,
                                  Value, Value)> &emit) {
  Block &block = op.getRegion().front();
  if (block.getOperations().size() != 2)
    return std::nullopt;
  auto yield = dyn_cast<linalg::YieldOp>(block.getTerminator());
  if (!yield || yield.getNumOperands() != 1)
    return std::nullopt;
  auto bin = yield->getOperand(0).getDefiningOp();
  if (!bin)
    return std::nullopt;
  Value blockIn = block.getArgument(0), blockInit = block.getArgument(1);
  auto isArgs = [&](Value v) {
    return v == blockIn || v == blockInit;
  };

  if (auto maxnumf = dyn_cast<arith::MaxNumFOp>(bin)) {
    if (!isArgs(maxnumf.getLhs()) || !isArgs(maxnumf.getRhs()))
      return std::nullopt;
    emit = [](OpBuilder &b, Location loc, arith::FastMathFlags fm, Value l,
              Value r) {
      return arith::MaxNumFOp::create(b, loc, l, r, fm).getResult();
    };
    return maxnumf.getFastmath();
  }
  if (auto addf = dyn_cast<arith::AddFOp>(bin)) {
    if (!isArgs(addf.getLhs()) || !isArgs(addf.getRhs()))
      return std::nullopt;
    emit = [](OpBuilder &b, Location loc, arith::FastMathFlags fm, Value l,
              Value r) {
      return arith::AddFOp::create(b, loc, l, r, fm).getResult();
    };
    return addf.getFastmath();
  }
  return std::nullopt;
}

struct VectorRowReducePass
    : public ::impl::VectorRowReduceBase<VectorRowReducePass> {
public:
  explicit VectorRowReducePass() = default;

  void getDependentDialects(DialectRegistry &registry) const override {
    // The rewrite emits vector/scf/memref/arith ops and hvx.vror; without
    // this a kernel whose input never mentions the vector dialect would die
    // on "created with unregistered dialect".
    registry.insert<arith::ArithDialect, memref::MemRefDialect,
                    scf::SCFDialect, vector::VectorDialect, hvx::HvxDialect>();
  }

  void runOnOperation() override {
    auto fn = getOperation();
    // Collect first: rewriting invalidates the walk.
    SmallVector<linalg::ReduceOp> candidates;
    fn.walk([&](linalg::ReduceOp op) {
      std::function<Value(OpBuilder &, Location, arith::FastMathFlags, Value,
                          Value)>
          emit;
      if (matches(op, emit))
        candidates.push_back(op);
    });
    for (linalg::ReduceOp op : candidates) {
      if (failed(rewrite(op)))
        return signalPassFailure();
    }
  }

private:
  static bool matches(linalg::ReduceOp op,
                      std::function<Value(OpBuilder &, Location,
                                          arith::FastMathFlags, Value, Value)>
                          &emit) {
    // Bufferized form only: the vector rewrite walks memrefs.
    if (op.getNumResults() != 0)
      return false;
    if (op.getInputs().size() != 1 || op.getInits().size() != 1)
      return false;
    ArrayRef<int64_t> dims = op.getDimensions();
    if (dims.size() != 1 || dims[0] != 1)
      return false;
    auto src = rowShapeOf(op.getInputs()[0]);
    if (!src || src->rows == 0)
      return false;
    auto dstTy = dyn_cast<MemRefType>(op.getInits()[0].getType());
    if (!dstTy || !dstTy.hasStaticShape() || dstTy.getRank() != 1 ||
        dstTy.getDimSize(0) != src->rows ||
        dstTy.getElementType() !=
            cast<MemRefType>(op.getInputs()[0].getType()).getElementType())
      return false;
    return matchFoldBody(op, emit).has_value();
  }

  LogicalResult rewrite(linalg::ReduceOp op) const {
    Value src = op.getInputs()[0];
    Value dst = op.getInits()[0];
    auto srcTy = cast<MemRefType>(src.getType());
    Type elemTy = srcTy.getElementType();
    RowShape shape = *rowShapeOf(src);

    std::function<Value(OpBuilder &, Location, arith::FastMathFlags, Value,
                        Value)>
        emit;
    arith::FastMathFlags fm = *matchFoldBody(op, emit);

    int64_t lanes = kHvxVectorBytes / shape.elemBytes; // f32: 32, f16: 64
    int64_t chunks = shape.cols / lanes;
    auto vecTy = VectorType::get({lanes}, elemTy);

    Location loc = op.getLoc();
    OpBuilder b(op);

    Value c0 = arith::ConstantIndexOp::create(b, loc, 0);
    Value c1 = arith::ConstantIndexOp::create(b, loc, 1);
    Value cRows = arith::ConstantIndexOp::create(b, loc, shape.rows);
    Value pad = arith::ConstantOp::create(
        b, loc, elemTy, b.getZeroAttr(elemTy));

    // One row per iteration. `in_bounds` is true: the row is a full,
    // statically known extent.
    auto loop = scf::ForOp::create(
        b, loc, c0, cRows, c1, ValueRange{},
        [](OpBuilder &, Location, Value, ValueRange) {});
    loop->moveBefore(op);
    b.setInsertionPointToStart(loop.getBody());
    Value r = loop.getInductionVar();

    // The row as a contiguous 1-D view (rank reduced by the size-1 row dim);
    // the outer stride stays in the subview. Static sizes/strides: the
    // rank-reduced verifier check needs the dropped dim to be statically 1.
    // The memory space carries over: the reduce source may live in VTCM
    // (hexagonmem, space 1), and the reads must stay in it.
    auto rowTy = MemRefType::get(
        {shape.cols}, elemTy,
        StridedLayoutAttr::get(b.getContext(), ShapedType::kDynamic, {1}),
        srcTy.getMemorySpace());
    SmallVector<OpFoldResult> rowOffsets{r, b.getIndexAttr(0)};
    SmallVector<OpFoldResult> rowSizes{b.getIndexAttr(1),
                                       b.getIndexAttr(shape.cols)};
    SmallVector<OpFoldResult> rowStrides{b.getIndexAttr(1),
                                         b.getIndexAttr(1)};
    auto row = memref::SubViewOp::create(b, loc, rowTy, src, rowOffsets,
                                         rowSizes, rowStrides);

    // Cross-chunk elementwise fold: lane i holds the fold over the same lane
    // of every 128-byte chunk of the row.
    static constexpr bool kInBounds[] = {true};
    Value acc;
    for (int64_t k = 0; k < chunks; ++k) {
      Value col = k == 0 ? c0 : arith::ConstantIndexOp::create(b, loc, k * lanes);
      Value chunk = vector::TransferReadOp::create(
          b, loc, vecTy, row, ValueRange{col}, pad,
          llvm::ArrayRef<bool>(kInBounds));
      acc = k == 0 ? chunk : emit(b, loc, fm, acc, chunk);
    }

    // Butterfly: rotate the register right by half the remaining span and
    // fold; after log2(128/elemBytes) steps every lane holds the row value.
    for (int64_t bytes = kHvxVectorBytes / 2; bytes >= shape.elemBytes;
         bytes /= 2) {
      Value rotated = hvx::VrorOp::create(b, loc, vecTy, acc,
                                          b.getI32IntegerAttr(bytes));
      acc = emit(b, loc, fm, rotated, acc);
    }

    // Fold in the reduce's own init (the outs element) and keep the scalar
    // contract: one reduced value per row, stored where the reduce wrote it.
    Value init = memref::LoadOp::create(b, loc, dst, ValueRange{r});
    Value initVec = vector::BroadcastOp::create(b, loc, vecTy, init);
    acc = emit(b, loc, fm, acc, initVec);
    Value scalar = vector::ExtractOp::create(b, loc, acc, 0);
    memref::StoreOp::create(b, loc, scalar, dst, ValueRange{r});
    scf::YieldOp::create(b, loc);

    op.erase();
    return success();
  }
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::hexagon::createVectorRowReducePass() {
  return std::make_unique<VectorRowReducePass>();
}
