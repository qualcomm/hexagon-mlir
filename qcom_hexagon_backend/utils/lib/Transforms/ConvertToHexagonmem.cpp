//===- ConvertToHexagonmem.cpp: lower memref ops to hexagonmem      ------====//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// Lower memref ops (alloc, copy, dealloc) to hexagonmem equivalent ops.
//
//===----------------------------------------------------------------------===//

#include "hexagon/Conversion/LinalgToLLVM/Common.h"
#include "hexagon/Conversion/LinalgToLLVM/LinalgToLLVM.h"
#include "hexagon/Dialect/HexagonMem/IR/HexagonMemDialect.h"
#include "hexagon/Transforms/Passes.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"
#include <algorithm>
#include <vector>

#define DEBUG_TYPE "convert-to-hexagonmem"

#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define DBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
using namespace hexagon;

#define GEN_PASS_DEF_CONVERTTOHEXAGONMEM
#include "hexagon/Transforms/Passes.h.inc"

namespace {

/// Returns `true` if type provided is ranked memref type with no dynamic dim.
bool isSafe(Type type) {
  auto baseMemRefType = dyn_cast<BaseMemRefType>(type);
  if (!baseMemRefType || !baseMemRefType.hasRank())
    return false;
  return llvm::count_if(baseMemRefType.getShape(), ShapedType::isDynamic) == 0;
}

/// Returns memory address assuming provided type is memref.
unsigned getMemorySpace(Type type) {
  auto baseMemRefType = dyn_cast<BaseMemRefType>(type);
  assert(baseMemRefType && "expected base memref type");
  return baseMemRefType.getMemorySpaceAsInt();
}

struct AllocConverter : public OpRewritePattern<memref::AllocOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(memref::AllocOp op,
                                PatternRewriter &rewriter) const final {
    Type type = op.getType();
    if (!isSafe(type) || getMemorySpace(type) != VTCM_ADDRESS_SPACE)
      return failure();
    rewriter.replaceOpWithNewOp<hexagonmem::AllocOp>(op, type,
                                                     op.getDynamicSizes());
    return success();
  }
};

struct DeAllocConverter : public OpRewritePattern<memref::DeallocOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(memref::DeallocOp op,
                                PatternRewriter &rewriter) const final {
    Type type = op.getMemref().getType();
    if (!isSafe(type) || getMemorySpace(type) != VTCM_ADDRESS_SPACE)
      return failure();
    rewriter.replaceOpWithNewOp<hexagonmem::DeallocOp>(op, op.getMemref());
    return success();
  }
};

struct CopyConverter : public OpRewritePattern<memref::CopyOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(memref::CopyOp op,
                                PatternRewriter &rewriter) const final {
    Type srcType = op.getSource().getType();
    Type tgtType = op.getTarget().getType();

    if (!isSafe(srcType) || !isSafe(tgtType))
      return failure();
    if (getMemorySpace(srcType) == getMemorySpace(tgtType))
      return failure();
    rewriter.replaceOpWithNewOp<hexagonmem::CopyOp>(op, op.getSource(),
                                                    op.getTarget());
    return success();
  }
};

struct ConvertToHexagonmemPass
    : public ::impl::ConvertToHexagonmemBase<ConvertToHexagonmemPass> {

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<hexagonmem::HexagonMemDialect>();
  }
  void runOnOperation() override;
};

void ConvertToHexagonmemPass::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  patterns.add<AllocConverter, DeAllocConverter, CopyConverter>(
      patterns.getContext());
  if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
    signalPassFailure();
}
} // namespace

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
hexagon::createConvertToHexagonmemPass() {
  return std::make_unique<ConvertToHexagonmemPass>();
}
