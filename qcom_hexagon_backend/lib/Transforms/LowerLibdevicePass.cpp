//===- LowerLibdevicePass.cpp - lower libdev triton calls -----------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// Patterns to transform a external library call, specifically a libdevice
// call which lowers to a tt.extern_elementwise op, to a QHL qhmath call.
//
//===----------------------------------------------------------------------===//

#include "hexagon/Transforms/Transforms.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/TransformOps/TensorTransformOps.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "triton/Conversion/TritonGPUToLLVM/ElementwiseOpToLLVMBase.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

using namespace mlir;
using namespace hexagon;

#define GEN_PASS_DEF_LOWERLIBDEVICE
#include "hexagon/Transforms/Passes.h.inc"

namespace {

struct LowerLibdevicePattern final
    : public OpRewritePattern<triton::ExternElementwiseOp> {
  LowerLibdevicePattern(MLIRContext *ctx) : OpRewritePattern(ctx) {}

  LogicalResult matchAndRewrite(triton::ExternElementwiseOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value inputTensor = op.getSrcs()[0];
    Value outputTensor = op.getResult();

    auto inputTy = inputTensor.getType();
    auto outputTy = outputTensor.getType();
    auto llvmPtrTy = mlir::LLVM::LLVMPointerType::get(rewriter.getContext());
    auto inTensorType = mlir::dyn_cast<RankedTensorType>(inputTy);
    auto outTensorType = mlir::dyn_cast<RankedTensorType>(outputTy);

    if (!inTensorType || !outTensorType)
      return rewriter.notifyMatchFailure(
          op, "only ranked tensor types are supported");

    if (inTensorType.getShape() != outTensorType.getShape())
      return rewriter.notifyMatchFailure(
          op, "input and output tensor shapes must match");

    if (!inTensorType.hasStaticShape() || !outTensorType.hasStaticShape())
      return rewriter.notifyMatchFailure(
          op, "only static shaped tensors are supported");

    int64_t numElems = inTensorType.getNumElements();

    auto memrefType = MemRefType::get(outTensorType.getShape(),
                                      outTensorType.getElementType());
    auto i64Ty = IntegerType::get(rewriter.getContext(), 64);

    Value outputMemref = memref::AllocOp::create(rewriter, loc, memrefType);
    Value outputPtrIndex = memref::ExtractAlignedPointerAsIndexOp::create(
        rewriter, loc, outputMemref);
    Value outputPtrInt =
        arith::IndexCastOp::create(rewriter, loc, i64Ty, outputPtrIndex);
    Value outputPtr =
        LLVM::IntToPtrOp::create(rewriter, loc, llvmPtrTy, outputPtrInt)
            .getResult();

    Value inputMemref = bufferization::ToBufferOp::create(
        rewriter, loc, memrefType, inputTensor);
    Value inputPtrIndex = memref::ExtractAlignedPointerAsIndexOp::create(
        rewriter, loc, inputMemref);
    Value inputPtrInt =
        arith::IndexCastOp::create(rewriter, loc, i64Ty, inputPtrIndex);
    Value inputPtr =
        LLVM::IntToPtrOp::create(rewriter, loc, llvmPtrTy, inputPtrInt)
            .getResult();

    Value size = arith::ConstantIntOp::create(rewriter, loc, numElems, 32);
    // Size should be uint32_t for libdevice functions (QHL)
    auto i32Ty = rewriter.getI32Type();
    if (size.getType() != i32Ty)
      size = arith::IndexCastUIOp::create(rewriter, loc, i32Ty, size);

    std::vector<Value> inputs = {inputPtr, outputPtr, size};
    mlir::ValueRange callOperands(inputs);

    ModuleOp module = op->getParentOfType<ModuleOp>();
    StringRef fName = op.getSymbol();

    auto llvmFnTy = LLVM::LLVMFunctionType::get(
        i32Ty, {llvmPtrTy, outputPtr.getType(), size.getType()},
        /*isVarArg=*/false);

    auto llvmFunc = module.lookupSymbol<LLVM::LLVMFuncOp>(fName);
    if (!llvmFunc) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(module.getBody());

      llvmFunc = LLVM::LLVMFuncOp::create(rewriter, loc, fName, llvmFnTy);
      llvmFunc.setVisibility(SymbolTable::Visibility::Private);
    }
    rewriter.setInsertionPoint(op);
    LLVM::CallOp::create(rewriter, loc, TypeRange{i32Ty},
                         SymbolRefAttr::get(llvmFunc), callOperands);
    Value res = bufferization::ToTensorOp::create(
        rewriter, loc, outTensorType, outputMemref, true /* restrict */,
        true /* writable */);

    rewriter.replaceOp(op, res);
    return success();
  }
};

struct LowerLibdevicePass
    : public ::impl::LowerLibdeviceBase<LowerLibdevicePass> {
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add<LowerLibdevicePattern>(context);

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>> hexagon::createLowerLibdevicePass() {
  return std::make_unique<LowerLibdevicePass>();
}
