//===- FormCompositeVirtualThreadsPass.cpp - MT for composite ops --------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// This pass tiles composite ops (softmax, rms_norm, layer_norm) to
// scf.forall for multi-threading. It uses TilingInterface and the shared
// thread-distribution helper from ThreadDistribution.h.
//
// Must run before bufferization (at tensor level).
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Interfaces/TilingInterface.h"
#include "mlir/Pass/Pass.h"

#include "llvm/Support/Debug.h"

#include "hexagon/Conversion/LinalgToLLVM/LinalgToLLVM.h"
#include "hexagon/Conversion/LinalgToLLVM/ThreadDistribution.h"
#include "hexagon/Dialect/Composite/IR/CompositeOps.h"

#define DEBUG_TYPE "form-composite-virtual-threads"

using namespace mlir;
using namespace mlir::utils;
using namespace hexagon;

#define GEN_PASS_DEF_FORMCOMPOSITEVIRTUALTHREADS
#include "hexagon/Conversion/LinalgToLLVM/Passes.h.inc"

namespace {

/// Extract static ranges from TilingInterface::getIterationDomain().
/// Returns false if any dimension is dynamic.
static bool getStaticRanges(TilingInterface tilingOp,
                            SmallVectorImpl<int64_t> &ranges) {
  OpBuilder b(tilingOp.getContext());
  b.setInsertionPoint(tilingOp);
  SmallVector<Range> domain = tilingOp.getIterationDomain(b);

  ranges.clear();
  ranges.reserve(domain.size());
  for (const Range &r : domain) {
    std::optional<int64_t> size = getConstantIntValue(r.size);
    if (!size)
      return false;
    ranges.push_back(*size);
  }

  return true;
}

/// Bypass the polytope-size heuristic in computeTLPTilingOptions — composite
/// ops are always worth threading (each iteration has 2-3 fused vector loops).
static constexpr int64_t kAlwaysThread = hexagon::kMinPolytopeSize;

struct FormCompositeVirtualThreadsPass
    : public ::impl::FormCompositeVirtualThreadsBase<
          FormCompositeVirtualThreadsPass> {
public:
  explicit FormCompositeVirtualThreadsPass(
      const FormCompositeVirtualThreadsOptions &options)
      : Base(options) {}

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<affine::AffineDialect, scf::SCFDialect,
                    tensor::TensorDialect>();
  }

  void runOnOperation() override;
};

void FormCompositeVirtualThreadsPass::runOnOperation() {
  auto funcOp = getOperation();
  if (preferredNumThreads <= 0) {
    funcOp.emitError() << "preferred-num-threads must be greater than zero";
    signalPassFailure();
    return;
  }

  funcOp.walk([&](Operation *op) {
    auto tilingOp = dyn_cast<TilingInterface>(op);
    if (!tilingOp)
      return WalkResult::advance();

    // Only tile composite ops.
    if (!isa<composite::SoftmaxOp, composite::RmsNormOp,
             composite::LayerNormOp>(op))
      return WalkResult::advance();

    auto iteratorTypes = tilingOp.getLoopIteratorTypes();
    SmallVector<int64_t> staticRanges;
    if (!getStaticRanges(tilingOp, staticRanges))
      return WalkResult::advance();

    int64_t polytopeSize = kAlwaysThread;

    IRRewriter rewriter(op->getContext());
    scf::SCFTilingOptions tilingOptions;

    if (!hexagon::computeTLPTilingOptions(rewriter, iteratorTypes, staticRanges,
                                          polytopeSize, blockSize,
                                          preferredNumThreads, tilingOptions))
      return WalkResult::advance();

    rewriter.setInsertionPoint(op);
    FailureOr<scf::SCFTilingResult> tiledResults =
        scf::tileUsingSCF(rewriter, tilingOp, tilingOptions);
    if (failed(tiledResults))
      return WalkResult::advance();

    rewriter.replaceOp(op, tiledResults->replacements);
    return WalkResult::advance();
  });
}

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
hexagon::createFormCompositeVirtualThreadsPass(
    const FormCompositeVirtualThreadsOptions &options) {
  return std::make_unique<FormCompositeVirtualThreadsPass>(options);
}
