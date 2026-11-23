//===- ShardToLLVMPass.cpp - Lower shard ops to arith/tensor
//---------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// Pass for lowering shard dialect ops produced by the NSP sharding pipeline.
//
// Scope / intent
// --------------
// The pass currently supports only a small, well-defined subset of shard ops
// emitted by the NSP sharding/materialization flow and rewrites them into
// standard MLIR (arith/tensor) so the downstream linalg-to-llvm pipeline can
// complete lowering.
//
// Supported shard ops
// -------------------
//   - shard.process_linear_index
//   - shard.all_slice  (ranked 1-D tensors, static slice size)
//   - shard.shard      (treated as an annotation carrier)
//   - shard.sharding   (erased once unused)
//
// ABI notes
// ---------
// The lowering of shard.process_linear_index is tied to the Hexagon entry-point
// ABI used by the runtime: we recover {coreId, threadId, numThreadsPerCore}
// from the tail of the function arguments and compute:
//   linearIdx = coreId * numThreadsPerCore + threadId
//
// Robustness with chained producer/consumer graphs
// -----------------------------------------------
// Some frontends emit a *chain* of elementwise tensor producers (often multiple
// linalg.generic ops) rather than a single fused kernel. In such cases sharding
// annotations frequently appear on multiple intermediate tensors, typically by
// wrapping SSA values with shard.shard while reusing a shared !shard.sharding.
//
// Dialect conversion treats the whole shard dialect as illegal, therefore *all*
// shard ops must be rewritten. A common failure mode is erasing shard.sharding
// while it still has uses (e.g. intermediate shard.shard ops): the conversion
// driver may visit shard.sharding before rewriting all of its users.
//
// To keep legalization robust for chained patterns, this pass follows a simple
// discipline:
//   (1) rewrite the *users* first (shard.shard and shard.all_slice),
//   (2) erase shard.sharding and shard.grid only once they become dead.
//
// Current semantic restrictions
// -----------------------------
// This lowering assumes that sharding is used for SPMD-style DOALL execution on
// independent slices. In particular, it does not currently materialize any
// cross-shard communication.
//
// Consequences:
//   - Elementwise chains are fine (all ops operate on per-shard slices).
//   - Reductions are only safe if the reduction does *not* need to combine
//     partial results across shards.
//       * If the reduced dimension is fully local to each shard (i.e. the value
//         being reduced is not partitioned across shards), the reduction can
//         remain local.
//       * If the reduced dimension is partitioned across shards, the lowering
//         must introduce a collective (e.g. shard.all_reduce) to combine
//         partial reductions, followed by any necessary broadcasts.
//   - Collective ops such as shard.all_reduce are not lowered yet.
//
//===----------------------------------------------------------------------===//

#include "hexagon/Common/Common.h"
#include "hexagon/Conversion/ShardToLLVM/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Shard/IR/ShardDialect.h"
#include "mlir/Dialect/Shard/IR/ShardOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "shard-to-llvm"
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define DBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;

// Declare the pass
#define GEN_PASS_DECL_SHARDTOLLVM
#include "hexagon/Conversion/ShardToLLVM/Passes.h.inc"

// Define the pass
#define GEN_PASS_DEF_SHARDTOLLVM
#include "hexagon/Conversion/ShardToLLVM/Passes.h.inc"

namespace {

// Resolve linear-index components from the entry-point signature.
//
// NOTE: This is ABI-specific by construction.
// The current Hexagon runtime ABI appends execution IDs at the tail of the
// entry-point signature. Right-to-left (last -> first), the relevant slots look
// like:
//   [..., ntpc, num_cores, <reserved>, tid, cid, <reserved>]
//
// Therefore, with last = args[N-1]:
//   cid  = args[N-2]
//   tid  = args[N-3]
//   ntpc = args[N-6]
//===----------------------------------------------------------------------===//

struct LinearIdxABI {
  Value cid;
  Value tid;
  Value ntpc;
  Value numCores; // optional
};

static Value castToIndexIfNeeded(Value v, OpBuilder &b, Location loc) {
  if (v.getType().isIndex())
    return v;
  if (isa<IntegerType>(v.getType()))
    return arith::IndexCastOp::create(b, loc, b.getIndexType(), v);
  return Value(); // Unexpected type.
}

static FailureOr<LinearIdxABI> resolveLinearIdxABIFromTail(func::FuncOp func) {
  auto args = func.getArguments();
  int64_t n = (int64_t)args.size();

  // Need at least 6 tail slots to access ntpc at N-6.
  if (n < 6)
    return failure();

  LinearIdxABI abi;
  abi.cid = args[n - 2];
  abi.tid = args[n - 3];
  abi.numCores =
      args[n - 5]; // not required for linearIdx, but kept for future use
  abi.ntpc = args[n - 6];

  return abi;
}

/// Compute linear process id:
///   linearIdx = coreId * numThreadsPerCore + threadId
static FailureOr<Value>
computeLinearIdxFromFuncArgs(func::FuncOp func, OpBuilder &b, Location loc) {
  auto abiOrFail = resolveLinearIdxABIFromTail(func);
  if (failed(abiOrFail))
    return failure();

  LinearIdxABI abi = *abiOrFail;

  Value cid = castToIndexIfNeeded(abi.cid, b, loc);
  Value tid = castToIndexIfNeeded(abi.tid, b, loc);
  Value ntpc = castToIndexIfNeeded(abi.ntpc, b, loc);

  if (!cid || !tid || !ntpc)
    return failure();

  Value mul = arith::MulIOp::create(b, loc, cid, ntpc);
  Value add = arith::AddIOp::create(b, loc, mul, tid);
  return add;
}

static FailureOr<Value>
computeProcessIndexForGridAxes(func::FuncOp func, OpBuilder &b, Location loc,
                               ArrayRef<int16_t> gridAxes) {
  if (gridAxes.empty())
    return failure();

  auto abiOrFail = resolveLinearIdxABIFromTail(func);
  if (failed(abiOrFail))
    return failure();

  LinearIdxABI abi = *abiOrFail;
  Value cid = castToIndexIfNeeded(abi.cid, b, loc);
  Value tid = castToIndexIfNeeded(abi.tid, b, loc);
  Value ntpc = castToIndexIfNeeded(abi.ntpc, b, loc);
  Value numCores = castToIndexIfNeeded(abi.numCores, b, loc);
  if (!cid || !tid || !ntpc)
    return failure();

  if (gridAxes.size() == 1) {
    if (gridAxes[0] == 0)
      return cid;
    if (gridAxes[0] == 1)
      return tid;
    return failure();
  }

  if (gridAxes.size() == 2) {
    if (gridAxes[0] == 0 && gridAxes[1] == 1) {
      Value mul = arith::MulIOp::create(b, loc, cid, ntpc);
      return arith::AddIOp::create(b, loc, mul, tid).getResult();
    }
    if (gridAxes[0] == 1 && gridAxes[1] == 0) {
      if (!numCores)
        return failure();
      Value mul = arith::MulIOp::create(b, loc, tid, numCores);
      return arith::AddIOp::create(b, loc, mul, cid).getResult();
    }
  }

  return failure();
}

static SmallVector<int16_t, 2> getGridAxesFromOp(Operation *op) {
  SmallVector<int16_t, 2> axes;
  Attribute attr = op->getAttr("grid_axes");
  if (!attr)
    return axes;

  if (auto dense = dyn_cast<DenseI16ArrayAttr>(attr)) {
    for (int16_t axis : dense.asArrayRef())
      axes.push_back(axis);
    return axes;
  }

  if (auto dense = dyn_cast<DenseI64ArrayAttr>(attr)) {
    for (int64_t axis : dense.asArrayRef())
      axes.push_back(static_cast<int16_t>(axis));
    return axes;
  }

  if (auto array = dyn_cast<ArrayAttr>(attr)) {
    for (Attribute elem : array)
      if (auto intAttr = dyn_cast<IntegerAttr>(elem))
        axes.push_back(static_cast<int16_t>(intAttr.getInt()));
  }

  return axes;
}

/// Lower shard.process_linear_index to: linearIdx = cid * ntpc + tid
struct LowerProcessLinearIndex final : public RewritePattern {
  explicit LowerProcessLinearIndex(MLIRContext *ctx)
      : RewritePattern("shard.process_linear_index", /*benefit=*/1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    auto func = op->getParentOfType<func::FuncOp>();
    if (!func)
      return failure();

    Location loc = op->getLoc();
    auto linearIdxOrFail = computeLinearIdxFromFuncArgs(func, rewriter, loc);
    if (failed(linearIdxOrFail))
      return failure();

    rewriter.replaceOp(op, *linearIdxOrFail);
    return success();
  }
};

/// Lower shard.all_slice into tensor.extract_slice.
///
/// The slice offset is derived from the linear process id:
///   offset[slice_axis] = linearIdx * sliceSize[slice_axis]
///   offset[d] = 0, for d != slice_axis
///
/// Restrictions (current implementation):
///   - ranked tensors only
///   - static slice sizes only
///   - slicing is along the op's slice_axis attribute
///   - no bounds/padding handling (assumes exact partitioning)
struct LowerAllSlice final : public RewritePattern {
  explicit LowerAllSlice(MLIRContext *ctx)
      : RewritePattern("shard.all_slice", /*benefit=*/1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (op->getNumOperands() < 1 || op->getNumResults() != 1)
      return failure();

    auto resultTy = dyn_cast<RankedTensorType>(op->getResult(0).getType());
    auto inputTy = dyn_cast<RankedTensorType>(op->getOperand(0).getType());
    if (!resultTy || !inputTy)
      return failure();

    if (resultTy.getRank() != inputTy.getRank() || resultTy.getRank() < 1)
      return failure();

    const int64_t rank = resultTy.getRank();

    SmallVector<int64_t> sliceShape(resultTy.getShape().begin(),
                                    resultTy.getShape().end());
    for (int64_t dim : sliceShape) {
      if (dim <= 0)
        return failure();
    }

    auto func = op->getParentOfType<func::FuncOp>();
    if (!func)
      return failure();

    Location loc = op->getLoc();

    SmallVector<int16_t, 2> gridAxes = getGridAxesFromOp(op);
    FailureOr<Value> procIdxOrFail =
        gridAxes.empty()
            ? computeLinearIdxFromFuncArgs(func, rewriter, loc)
            : computeProcessIndexForGridAxes(func, rewriter, loc, gridAxes);
    if (failed(procIdxOrFail))
      return failure();

    Value procIdx = *procIdxOrFail;

    auto sliceAxisAttr = op->getAttrOfType<IntegerAttr>("slice_axis");
    if (!sliceAxisAttr)
      return failure();

    int64_t sliceAxis = sliceAxisAttr.getInt();
    if (sliceAxis < 0 || sliceAxis >= rank)
      return failure();

    Value sliceSize =
        arith::ConstantIndexOp::create(rewriter, loc, sliceShape[sliceAxis]);
    Value sliceOffset =
        arith::MulIOp::create(rewriter, loc, procIdx, sliceSize);

    SmallVector<OpFoldResult> offsets;
    SmallVector<OpFoldResult> sizes;
    SmallVector<OpFoldResult> strides;
    offsets.reserve(rank);
    sizes.reserve(rank);
    strides.reserve(rank);

    for (int64_t d = 0; d < rank; ++d) {
      offsets.push_back(d == sliceAxis
                            ? OpFoldResult(sliceOffset)
                            : OpFoldResult(rewriter.getIndexAttr(0)));
      sizes.push_back(rewriter.getIndexAttr(sliceShape[d]));
      strides.push_back(rewriter.getIndexAttr(1));
    }

    Value input = op->getOperand(0);
    auto slice = tensor::ExtractSliceOp::create(rewriter, loc, resultTy, input,
                                                offsets, sizes, strides);

    rewriter.replaceOp(op, slice.getResult());
    return success();
  }
};

/// Lower shard.shard.
///
/// In the current NSP sharding flow, shard.shard is used as an *annotation
/// carrier* on SSA tensor values ("this value is sharded like X") and does not
/// materialize any data movement.
///
/// Therefore lowering can conservatively drop the wrapper and keep the tensor
/// operand (operand 0).
///
/// IMPORTANT: if shard.shard gains semantic meaning in the future (e.g.
/// materializing layout/resharding or changing buffer placement), this lowering
/// must be revisited.
struct LowerShardOp final : public RewritePattern {
  explicit LowerShardOp(MLIRContext *ctx)
      : RewritePattern("shard.shard", /*benefit=*/1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    // shard.shard %tensor to %sharding ...
    // Some pipelines may also produce shard.shard with extra operands/attrs.
    if (op->getNumOperands() < 1 || op->getNumResults() != 1)
      return failure();

    rewriter.replaceOp(op, op->getOperand(0));
    return success();
  }
};

/// Erase dead shard.sharding values.
///
/// shard.sharding produces a !shard.sharding SSA value that is typically used
/// by multiple shard.shard and/or shard.all_slice ops.
///
/// During dialect conversion, erasing shard.sharding too early can be brittle
/// when there are chained producers/consumers: the conversion driver may visit
/// the producer before rewriting all users.
///
/// This pattern is intentionally conservative: only erase shard.sharding once
/// the result has no remaining uses.
struct EraseDeadShardingValue final : public RewritePattern {
  explicit EraseDeadShardingValue(MLIRContext *ctx)
      : RewritePattern("shard.sharding", /*benefit=*/1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (op->getNumResults() != 1)
      return failure();

    if (!op->getResult(0).use_empty())
      return failure();

    rewriter.eraseOp(op);
    return success();
  }
};

/// Erase shard.grid once unused.
///
/// shard.grid is currently treated as metadata that becomes redundant after all
/// shard ops are lowered.
struct EraseGrid final : public RewritePattern {
  explicit EraseGrid(MLIRContext *ctx)
      : RewritePattern("shard.grid", /*benefit=*/1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    // Be conservative if it unexpectedly has results that are still used.
    for (Value r : op->getResults())
      if (!r.use_empty())
        return failure();
    rewriter.eraseOp(op);
    return success();
  }
};

struct ShardToLLVMPass : public ::impl::ShardToLLVMBase<ShardToLLVMPass> {

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, tensor::TensorDialect,
                    func::FuncDialect, shard::ShardDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *ctx = module.getContext();

    // Step 1: greedily rewrite shard consumers so shard.sharding becomes dead.
    {
      RewritePatternSet pre(ctx);
      pre.add<LowerShardOp>(ctx);
      pre.add<LowerAllSlice>(ctx);
      pre.add<LowerProcessLinearIndex>(ctx);
      pre.add<EraseGrid>(ctx);
      (void)applyPatternsGreedily(module, std::move(pre));
    }

    // Check if still have uses of shard.sharding
    module.walk([&](Operation *op) {
      if (op->getName().getStringRef() == "shard.sharding") {
        Value sh = op->getResult(0);
        if (!sh.use_empty()) {
          op->emitRemark() << "shard.sharding still has uses:";
          for (Operation *u : sh.getUsers())
            op->emitRemark() << "  user: " << u->getName();
        }
      }
    });

    // Safe to cleanup: erase now-dead sharding values.
    {
      RewritePatternSet dce(ctx);
      dce.add<EraseDeadShardingValue>(ctx);
      (void)applyPatternsGreedily(module, std::move(dce));
    }

    // Step 2: enforce that no shard ops remain.
    ConversionTarget target(*ctx);
    target.addIllegalDialect<shard::ShardDialect>();
    target.addLegalDialect<arith::ArithDialect, tensor::TensorDialect,
                           func::FuncDialect>();
    target.markUnknownOpDynamicallyLegal([](Operation *op) {
      return op->getName().getDialectNamespace() != StringRef("shard");
    });

    // No more patterns needed; we're just validating legality now.
    RewritePatternSet empty(ctx);
    if (failed(applyPartialConversion(module, target, std::move(empty))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> mlir::hexagon::createShardToLLVMPass() {
  return std::make_unique<ShardToLLVMPass>();
}
