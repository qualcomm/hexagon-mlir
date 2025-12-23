//===- DMAToLLVMPass.cpp - Lowering DMA ops to LLVM -----------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to convert DMA dialect to LLVM dialect.
//===----------------------------------------------------------------------===//

#include "hexagon/Common/Common.h"
#include "hexagon/Conversion/DMAToLLVM/DMAExternalFnNames.h"
#include "hexagon/Conversion/DMAToLLVM/Passes.h"
#include "hexagon/Dialect/HexagonMem/IR/HexagonMemDialect.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/FunctionCallUtils.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/Support/Debug.h"
#define DEBUG_TYPE "dma-to-llvm"

#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define DBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
using namespace hexagon;
using namespace mlir::hexagonmem;

#define GEN_PASS_CLASSES
#include "hexagon/Conversion/DMAToLLVM/Passes.h.inc"

namespace {

// Function to get an LLVM pointer type
static LLVM::LLVMPointerType getPtrTy(MLIRContext *context) {
  return LLVM::LLVMPointerType::get(context);
}

// Function to get an LLVM void type
static LLVM::LLVMVoidType getVoidTy(MLIRContext *context) {
  return LLVM::LLVMVoidType::get(context);
}

// Function to create an LLVM constant of i32 type
static LLVM::ConstantOp getI32Constant(ConversionPatternRewriter &rewriter,
                                       Location loc, int64_t val) {
  // Get the context from the rewriter
  MLIRContext *context = rewriter.getContext();
  Type paramTy = rewriter.getI32Type();
  // Create and return an LLVM constant operation with the given value
  return rewriter.create<LLVM::ConstantOp>(loc, paramTy, val);
}

// Function to truncate a value to i32 type
static Value truncToI32(ConversionPatternRewriter &rewriter, Location loc,
                        Value val) {
  // Ensure the value is of an integer type
  assert(llvm::isa<IntegerType>(val.getType()) && "Expected an integer type");
  MLIRContext *context = rewriter.getContext();
  // Create and return a truncation operation to i32 type
  return rewriter.create<LLVM::TruncOp>(loc, rewriter.getI32Type(), val);
}

//===----------------------------------------------------------------------===//
// Lower DMAStart
//===----------------------------------------------------------------------===//

// Structure to lower DMA start operation to LLVM pattern
struct LowerDMAStart : public ConvertOpToLLVMPattern<memref::DmaStartOp> {
  using ConvertOpToLLVMPattern<memref::DmaStartOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(memref::DmaStartOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

// Function to get or create the DMA start function in the module
static FailureOr<LLVM::LLVMFuncOp>
getDMAStartFn(ModuleOp module, StringRef fnName,
              ConversionPatternRewriter &rewriter) {
  MLIRContext *context = module->getContext();
  return LLVM::lookupOrCreateFn(rewriter, module, fnName,
                                {getPtrTy(context), rewriter.getI32Type(),
                                 getPtrTy(context), rewriter.getI32Type(),
                                 rewriter.getI32Type(), rewriter.getI32Type(),
                                 rewriter.getI32Type(), getPtrTy(context)},
                                rewriter.getI32Type());
}

// Function to get or create the 2D DMA start function in the module
static FailureOr<LLVM::LLVMFuncOp>
getDMA2DStartFn(ModuleOp module, StringRef fnName,
                ConversionPatternRewriter &rewriter) {
  MLIRContext *context = module->getContext();
  return LLVM::lookupOrCreateFn(
      rewriter, module, fnName,
      {getPtrTy(context), rewriter.getI32Type(), getPtrTy(context),
       rewriter.getI32Type(), rewriter.getI32Type(), rewriter.getI32Type(),
       rewriter.getI32Type(), rewriter.getI32Type(), rewriter.getI32Type(),
       rewriter.getI32Type(), rewriter.getI32Type(), rewriter.getI32Type(),
       getPtrTy(context)},
      rewriter.getI32Type());
}

// Function to convert a value to i32 type
Value convertToI32Type(mlir::Location loc, Value value,
                       ConversionPatternRewriter &rewriter,
                       mlir::MLIRContext *context) {
  // Get the type of the value
  mlir::Type type = value.getType();
  assert(mlir::LLVM::isCompatibleType(type));
  return rewriter.create<LLVM::TruncOp>(loc, rewriter.getI32Type(), value);
}

// Function to cast a pointer to a different address space
static Value AddrSpaceCast(mlir::Location loc, Value inputPtr,
                           ConversionPatternRewriter &rewriter,
                           mlir::MLIRContext *context) {
  auto i32Type = rewriter.getI32Type();
  auto asI32 = rewriter.create<LLVM::PtrToIntOp>(loc, i32Type, inputPtr);
  return rewriter.create<LLVM::IntToPtrOp>(loc, getPtrTy(context),
                                           asI32->getResult(0));
}

// Function to create DMAStart operation
LogicalResult
LowerDMAStart::matchAndRewrite(memref::DmaStartOp op, OpAdaptor adaptor,
                               ConversionPatternRewriter &rewriter) const {
  // Get the context and location of the operation
  auto context = op->getContext();
  auto loc = op->getLoc();

  // Get the parent module of the operation
  auto module = op->getParentOfType<ModuleOp>();

  // Get the name of the DMA start function
  auto dmaStartFnName = mlir::hexagon::getDMA2DStartFnName();
  FailureOr<LLVM::LLVMFuncOp> funcOp =
      getDMA2DStartFn(module, dmaStartFnName, rewriter);
  if (failed(funcOp))
    return failure();

  auto isVTCM = [](Value memref) {
    assert(llvm::isa<MemRefType>(memref.getType()) && "Unexpected type passed");
    auto memrefType = cast<MemRefType>(memref.getType());
    return isInVTCMAddressSpace(memrefType);
  };

  // Get the LLVM pointer to the memory reference
  auto getMemRefPtr = [this, loc,
                       context](Value memref, int rank, signed offset,
                                OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) {
    auto memrefType = cast<MemRefType>(memref.getType());
    auto llvmMemref = *(adaptor.getOperands().begin() + offset);
    ValueRange indices = {
        std::next(adaptor.getOperands().begin(), offset + 1),
        std::next(adaptor.getOperands().begin(), offset + rank + 1)};
    auto llvmPtr =
        getStridedElementPtr(rewriter, loc, memrefType, llvmMemref, indices);
    return llvmPtr;
  };

  // Get the source memory reference and its rank
  auto srcMemref = op.getSrcMemRef();
  auto srcRank = op.getSrcMemRefRank();
  auto tagRank = op.getTagMemRefRank();
  // Get the LLVM pointer to the source memory reference
  auto llvmSrcPtr = getMemRefPtr(srcMemref, srcRank, 0, adaptor, rewriter);

  // Get the destination memory reference and its rank
  auto dstMemref = op.getDstMemRef();
  auto dstRank = op.getDstMemRefRank();
  // Get the LLVM pointer to the destination memory reference
  auto llvmDstPtr =
      getMemRefPtr(dstMemref, dstRank, 1 + srcRank, adaptor, rewriter);

  // Get the tag memory reference and its LLVM pointer
  // This is used to synchronize between dma_start and dma_wait operation
  auto tagMemref = op.getTagMemRef();
  auto llvmTagPtr = getMemRefPtr(tagMemref, op.getTagMemRefRank(),
                                 3 + srcRank + dstRank, adaptor, rewriter);

  // Convert the number of elements to an LLVM integer type
  auto numElements = convertToI32Type(
      loc, *(adaptor.getOperands().begin() + 2 + srcRank + dstRank), rewriter,
      context);
  // Get the size of the element type in bytes
  auto typeSizeInBytes = getI32Constant(rewriter, loc,
                                        cast<MemRefType>(srcMemref.getType())
                                                .getElementType()
                                                .getIntOrFloatBitWidth() /
                                            8);
  // Calculate the transfer size
  auto transferSize =
      rewriter.create<LLVM::MulOp>(loc, numElements, typeSizeInBytes);

  // Get the source and destination memory spaces
  auto srcMemSpace = getI32Constant(rewriter, loc, isVTCM(srcMemref) ? 1 : 0);
  auto dstMemSpace = getI32Constant(rewriter, loc, isVTCM(dstMemref) ? 1 : 0);
  // Set the bypass cache flag to 0
  auto bypassCache = getI32Constant(rewriter, loc, 0);
  SmallVector<Value, 16> operands;

  if (op.isStrided()) {
    Value width = convertToI32Type(
        loc, *(adaptor.getOperands().begin() + srcRank + dstRank + tagRank + 5),
        rewriter, context);
    Value widthInBytes =
        rewriter.create<LLVM::MulOp>(loc, width, typeSizeInBytes);
    Value height =
        (rewriter.create<LLVM::SDivOp>(loc, transferSize, widthInBytes))
            ->getResult(0);
    // Stride arg is optional, and if present, it is only slow address space.
    // source stride iff stride is present, and dest is faster.
    // If no stride, use stride = width
    auto srcStride =
        (isVTCM(dstMemref) && (!(isVTCM(srcMemref))))
            ? convertToI32Type(loc,
                               (*(adaptor.getOperands().begin() + srcRank +
                                  dstRank + tagRank + 4)),
                               rewriter, context)
            : width;
    auto srcStrideInBytes =
        rewriter.create<LLVM::MulOp>(loc, srcStride, typeSizeInBytes);
    // dest stride iff stride is present, and src is faster.
    auto dstStride =
        (isVTCM(srcMemref) && (!(isVTCM(dstMemref))))
            ? convertToI32Type(loc,
                               (*(adaptor.getOperands().begin() + srcRank +
                                  dstRank + tagRank + 4)),
                               rewriter, context)
            : width;
    auto dstStrideInBytes =
        rewriter.create<LLVM::MulOp>(loc, dstStride, typeSizeInBytes);

    // Create a vector of operands for the DMA copy operation
    operands = {llvmSrcPtr,
                srcMemSpace,
                llvmDstPtr,
                dstMemSpace,
                widthInBytes,
                height,
                srcStrideInBytes,
                dstStrideInBytes,
                bypassCache,
                bypassCache,
                getI32Constant(rewriter, loc, 0), // isOrdered
                getI32Constant(rewriter, loc, 0), // Cache Allocation Policy
                llvmTagPtr};
  } else {
    dmaStartFnName = mlir::hexagon::getDMAStartFnName();
    funcOp = getDMAStartFn(module, dmaStartFnName, rewriter);
    operands = {llvmSrcPtr,   srcMemSpace, llvmDstPtr,  dstMemSpace,
                transferSize, bypassCache, bypassCache, llvmTagPtr};
  }
  // Erase the original operation
  rewriter.eraseOp(op);

  // Create the DMA copy operation
  auto dmaCopyOp = rewriter.create<LLVM::CallOp>(loc, funcOp.value(), operands);

  // Get the token from the DMA copy operation
  auto token = *dmaCopyOp.getODSResults(0).begin();

  // Store the token in the tag memory reference
  rewriter.create<LLVM::StoreOp>(loc, token, llvmTagPtr);

  return success();
}

//===----------------------------------------------------------------------===//
// Lower DMAWait
//===----------------------------------------------------------------------===//

// Structure to lower DMA wait operation to LLVM pattern
struct LowerDMAWait : public ConvertOpToLLVMPattern<memref::DmaWaitOp> {
  // Inherit the constructor from the base class
  using ConvertOpToLLVMPattern<memref::DmaWaitOp>::ConvertOpToLLVMPattern;

  // Override the matchAndRewrite function to perform the conversion
  LogicalResult
  matchAndRewrite(memref::DmaWaitOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;
};

// Function to get or create the DMA wait function in the module
static FailureOr<LLVM::LLVMFuncOp>
getDMAWaitFn(ModuleOp module, StringRef fnName,
             ConversionPatternRewriter &rewriter) {
  // Get the context from the module
  MLIRContext *context = module->getContext();
  // Lookup or create the DMA wait function with the specified name and types
  return LLVM::lookupOrCreateFn(rewriter, module, fnName,
                                {rewriter.getI32Type()}, getVoidTy(context));
}

// Override the matchAndRewrite function to perform the conversion
LogicalResult
LowerDMAWait::matchAndRewrite(memref::DmaWaitOp op, OpAdaptor adaptor,
                              ConversionPatternRewriter &rewriter) const {
  // Get the context and location of the operation
  auto context = op->getContext();
  auto loc = op->getLoc();
  // Get the parent module of the operation
  auto module = op->getParentOfType<ModuleOp>();

  // Get the name of the DMA wait function
  auto dmaWaitFnName = mlir::hexagon::getDMAWaitFnName();
  // Get the LLVM function for DMA wait
  FailureOr<LLVM::LLVMFuncOp> funcOp =
      getDMAWaitFn(module, dmaWaitFnName, rewriter);
  if (failed(funcOp))
    return failure();

  // Get the tag memory reference and its type
  auto tagMemref = op.getTagMemRef();
  auto tagMemrefType = cast<MemRefType>(tagMemref.getType());
  // Get the LLVM tag memory reference
  auto llvmTagMemref = adaptor.getTagMemRef();
  // Get the pointer to the tag memory reference
  auto llvmTagPtr = getStridedElementPtr(
      rewriter, loc, tagMemrefType, llvmTagMemref, adaptor.getTagIndices());
  // Erase the original operation
  rewriter.eraseOp(op);
  // Create a load operation to get the token
  auto tokenLoadOp =
      rewriter.create<LLVM::LoadOp>(loc, rewriter.getI32Type(), llvmTagPtr);
  // Create a vector of operands for the DMA wait operation
  SmallVector<Value, 10> operands = {tokenLoadOp.getResult()};
  // Create the DMA wait operation
  auto DMAWaitOp = rewriter.create<LLVM::CallOp>(loc, funcOp.value(), operands);
  // Return success
  return success();
}

//===----------------------------------------------------------------------===//
// Setup the Lowering Pass and patterns
//===----------------------------------------------------------------------===//

void populateDMAToLLVMConversionPatterns(LLVMTypeConverter &converter,
                                         RewritePatternSet &patterns) {
  patterns.add<LowerDMAStart, LowerDMAWait>(converter);
}

struct DMAToLLVMPass : public DMAToLLVMBase<DMAToLLVMPass> {
  using Base::Base;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<memref::MemRefDialect, LLVM::LLVMDialect>();
  }

  void runOnOperation() override {
    auto moduleOp = getOperation();
    MLIRContext *context = moduleOp->getContext();

    LLVMConversionTarget target(*context);
    RewritePatternSet patterns(context);
    LowerToLLVMOptions options(context);
    LLVMTypeConverter typeConverter(context, options);

    hexagon::addTypeConversions(context, typeConverter);
    populateDMAToLLVMConversionPatterns(typeConverter, patterns);

    if (failed(applyPartialConversion(moduleOp, target, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>> hexagon::createDMAToLLVMPass() {
  return std::make_unique<DMAToLLVMPass>();
}
