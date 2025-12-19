//===-- ConvertZeroSizeMemrefPass.cpp - Convert zero size memref ----------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// During the LowerDeallocations pass, as part of the bufferization
// deallocation pipeline, a zero size memref allocation is generated for the
// flash attention triton kernel (and similar kernels), which is then lowered to
// a malloc 0 in LLVM IR. Due to improper handling of malloc 0 in the runtime,
// the test execution crashes on device. So, this pass replaces the zero size
// memref alloc operations with memref alloc of unit size, so that malloc 0 is
// not generated, and hence the crash is not observed.
//
//===----------------------------------------------------------------------===//

#include "hexagon/Transforms/Passes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace hexagon;

#define GEN_PASS_DEF_CONVERTZEROSIZEMEMREF
#include "hexagon/Transforms/Passes.h.inc"

namespace {

struct ZeroSizeMemrefPattern final : public OpRewritePattern<memref::AllocOp> {
  ZeroSizeMemrefPattern(MLIRContext *ctx) : OpRewritePattern(ctx) {}

  LogicalResult matchAndRewrite(memref::AllocOp op,
                                PatternRewriter &rewriter) const override {
    auto shape = op.getType().getShape();
    auto result = shape.size() == 1 && shape[0] == 0;
    if (result) {
      rewriter.replaceOpWithNewOp<memref::AllocOp>(
          op, MemRefType::get({1}, op.getType().getElementType()));
    }
    return success(result);
  }
};

struct ConvertZeroSizeMemrefPass
    : public ::impl::ConvertZeroSizeMemrefBase<ConvertZeroSizeMemrefPass> {

  void runOnOperation() override {
    auto moduleOp = getOperation();
    RewritePatternSet patterns(moduleOp.getContext());
    patterns.add<ZeroSizeMemrefPattern>(patterns.getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};
} // end anonymous namespace

std::unique_ptr<OperationPass<func::FuncOp>>
hexagon::createConvertZeroSizeMemrefPass() {
  return std::make_unique<ConvertZeroSizeMemrefPass>();
}
