//===- HexKLToLLVMPass.cpp - HexKL to LLVM  conversion      ---------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// This pass implements lowering of HexKL ops to LLVM.
//
//===----------------------------------------------------------------------===//

#include "hexagon/Common/Common.h"
#include "hexagon/Conversion/HexKLToLLVM/HexKLExternalFnNames.h"
#include "hexagon/Conversion/HexKLToLLVM/HexKLToLLVM.h"
#include "hexagon/Conversion/HexKLToLLVM/Passes.h"
#include "hexagon/Conversion/LinalgToLLVM/LinalgToLLVM.h"
#include "hexagon/Dialect/HexKL/IR/HexKLDialect.h"
#include "mlir/Analysis/DataLayoutAnalysis.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/MemRefBuilder.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/FunctionCallUtils.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "hexkl-to-llvm"

#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define DBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
using namespace mlir::hexkl;

#define GEN_PASS_CLASSES
#include "hexagon/Conversion/HexKLToLLVM/Passes.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// Existing lowering: hexkl::MatmulOp -> extern C matmul
// Keep this implementation intact per project guidance.
//===----------------------------------------------------------------------===//
static FailureOr<LLVM::LLVMFuncOp>
getMatmulRefFn(ModuleOp module, const LLVMTypeConverter &typeConverter,
               ConversionPatternRewriter &rewriter) {
  MLIRContext *ctx = module->getContext();

  // size_t → use the target index type (i32 on 32-bit, i64 on 64-bit)
  Type idxTy = typeConverter.getIndexType();

  // Pointers are opaque in modern LLVM dialect: use !llvm.ptr
  auto ptrTy = LLVM::LLVMPointerType::get(ctx);
  auto voidTy = LLVM::LLVMVoidType::get(ctx);

  SmallVector<Type, 6> argTys = {idxTy, idxTy, idxTy, ptrTy, ptrTy, ptrTy};
  return LLVM::lookupOrCreateFn(rewriter, module, hexkl::getMatmulMicroFnName(),
                                argTys, voidTy);
}

// Common helper: extract aligned pointer from lowered memref descriptor.
static Value alignedPtr(ConversionPatternRewriter &rewriter, Location loc,
                        Value memrefDesc) {
  MemRefDescriptor desc(memrefDesc);
  return desc.alignedPtr(rewriter, loc);
}

struct LowerMatmul : public ConvertOpToLLVMPattern<hexkl::MatmulOp> {
  using ConvertOpToLLVMPattern<hexkl::MatmulOp>::ConvertOpToLLVMPattern;

  LogicalResult matchAndRewrite(hexkl::MatmulOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const {
    Location loc = op->getLoc();

    // Extract memref types
    auto lhsType = cast<mlir::MemRefType>(op.getLhs().getType());
    auto rhsType = cast<mlir::MemRefType>(op.getRhs().getType());
    auto resType = cast<mlir::MemRefType>(op.getOuts().getType());

    // --- Ensure the callee is declared ---
    ModuleOp module = op->getParentOfType<ModuleOp>();
    const auto *typeConverter = getTypeConverter();
    auto fn = getMatmulRefFn(module, *typeConverter, rewriter);
    if (failed(fn))
      return failure();

    // --- Extract aligned data pointers using helper function ---
    Value outPtr = alignedPtr(rewriter, loc, adaptor.getOuts());
    Value lhsPtr = alignedPtr(rewriter, loc, adaptor.getLhs());
    Value rhsPtr = alignedPtr(rewriter, loc, adaptor.getRhs());

    // --- Extract matrix extents as index-typed values ---
    // lhs: [n_row, n_inner], rhs: [n_inner, n_col], out: [n_row, n_col]
    MemRefDescriptor lhsDesc(adaptor.getLhs());
    MemRefDescriptor outDesc(adaptor.getOuts());
    Value nRow = lhsDesc.size(rewriter, loc, /*dim=*/0);
    Value nInner = lhsDesc.size(rewriter, loc, /*dim=*/1);
    Value nCol = outDesc.size(rewriter, loc, /*dim=*/1);

    // --- Build argument list and emit the call ---
    SmallVector<Value, 6> operands = {nRow,   nCol,   nInner,
                                      outPtr, lhsPtr, rhsPtr};
    auto fnCallee = FlatSymbolRefAttr::get((*fn).getOperation());
    rewriter.replaceOpWithNewOp<LLVM::CallOp>(op, TypeRange{}, fnCallee,
                                              operands);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Helpers: extern declaration and common utilities
//===----------------------------------------------------------------------===//
static Value castToI32(ConversionPatternRewriter &rewriter, Location loc,
                       Value v) {
  auto i32Ty = rewriter.getI32Type();
  if (v.getType() == i32Ty)
    return v;
  auto intTy = dyn_cast<IntegerType>(v.getType());
  assert(intTy && "castToI32 expects an integer-typed value");
  unsigned w = intTy.getWidth();
  if (w > 32)
    return rewriter.create<LLVM::TruncOp>(loc, i32Ty, v);
  if (w < 32)
    return rewriter.create<LLVM::ZExtOp>(loc, i32Ty, v);
  return v;
}
static FailureOr<LLVM::LLVMFuncOp>
getNoArgI32RetFn(ModuleOp module, ConversionPatternRewriter &rewriter,
                 StringRef name) {
  auto i32Ty = rewriter.getI32Type();
  SmallVector<Type, 0> argTys;
  auto fn = LLVM::lookupOrCreateFn(rewriter, module, name, argTys, i32Ty);
  if (succeeded(fn)) {
    auto funcOp = *fn;
    auto ctx = rewriter.getContext();
    funcOp->setAttr(StringAttr::get(ctx, "llvm.readnone"),
                    rewriter.getUnitAttr());
    funcOp->setAttr(StringAttr::get(ctx, "llvm.nounwind"),
                    rewriter.getUnitAttr());
  }
  return fn;
}

static FailureOr<LLVM::LLVMFuncOp>
getSetupAccReadF16Fn(ModuleOp module, ConversionPatternRewriter &rewriter) {
  auto ctx = module->getContext();
  auto ptrTy = LLVM::LLVMPointerType::get(ctx);
  auto i32Ty = rewriter.getI32Type();
  auto i32Vec = SmallVector<Type, 2>{ptrTy, i32Ty};
  auto voidTy = LLVM::LLVMVoidType::get(ctx);
  return LLVM::lookupOrCreateFn(
      rewriter, module, hexkl::getHmxSetupAccReadF16FnName(), i32Vec, voidTy);
}

static FailureOr<LLVM::LLVMFuncOp>
getAccClearF16Fn(ModuleOp module, ConversionPatternRewriter &rewriter) {
  auto ctx = module->getContext();
  auto voidTy = LLVM::LLVMVoidType::get(ctx);
  SmallVector<Type, 0> argTys;
  return LLVM::lookupOrCreateFn(
      rewriter, module, hexkl::getHmxAccClearF16FnName(), argTys, voidTy);
}

static FailureOr<LLVM::LLVMFuncOp>
getAccReadF16Fn(ModuleOp module, ConversionPatternRewriter &rewriter) {
  auto ctx = module->getContext();
  auto ptrTy = LLVM::LLVMPointerType::get(ctx);
  auto i32Ty = rewriter.getI32Type();
  SmallVector<Type, 3> argTys{ptrTy, i32Ty, i32Ty};
  auto voidTy = LLVM::LLVMVoidType::get(ctx);
  return LLVM::lookupOrCreateFn(
      rewriter, module, hexkl::getHmxAccReadF16FnName(), argTys, voidTy);
}

static FailureOr<LLVM::LLVMFuncOp>
getCopySubmatrixToF16Fn(ModuleOp module, ConversionPatternRewriter &rewriter) {
  auto ctx = module->getContext();
  auto ptrTy = LLVM::LLVMPointerType::get(ctx);
  auto i32Ty = rewriter.getI32Type();
  SmallVector<Type, 7> argTys{ptrTy, i32Ty, ptrTy, i32Ty, i32Ty, i32Ty, i32Ty};
  auto voidTy = LLVM::LLVMVoidType::get(ctx);
  return LLVM::lookupOrCreateFn(rewriter, module,
                                hexkl::getHmxCopySubmatrixToF16FnName(), argTys,
                                voidTy);
}

static FailureOr<LLVM::LLVMFuncOp>
getRmToAhF16Fn(ModuleOp module, ConversionPatternRewriter &rewriter) {
  auto ctx = module->getContext();
  auto ptrTy = LLVM::LLVMPointerType::get(ctx);
  auto i32Ty = rewriter.getI32Type();
  SmallVector<Type, 3> argTys{ptrTy, i32Ty, i32Ty};
  auto voidTy = LLVM::LLVMVoidType::get(ctx);
  return LLVM::lookupOrCreateFn(rewriter, module,
                                hexkl::getHmxRmToAhF16FnName(), argTys, voidTy);
}

static FailureOr<LLVM::LLVMFuncOp>
getRmToWhF16Fn(ModuleOp module, ConversionPatternRewriter &rewriter) {
  auto ctx = module->getContext();
  auto ptrTy = LLVM::LLVMPointerType::get(ctx);
  auto i32Ty = rewriter.getI32Type();
  SmallVector<Type, 6> argTys{ptrTy, i32Ty, ptrTy, i32Ty, i32Ty, i32Ty};
  auto voidTy = LLVM::LLVMVoidType::get(ctx);
  return LLVM::lookupOrCreateFn(rewriter, module,
                                hexkl::getHmxRmToWhF16FnName(), argTys, voidTy);
}

static FailureOr<LLVM::LLVMFuncOp>
getMmF16Fn(ModuleOp module, ConversionPatternRewriter &rewriter) {
  auto ctx = module->getContext();
  auto ptrTy = LLVM::LLVMPointerType::get(ctx);
  auto i32Ty = rewriter.getI32Type();
  SmallVector<Type, 3> argTys{ptrTy, i32Ty, i32Ty};
  auto voidTy = LLVM::LLVMVoidType::get(ctx);
  return LLVM::lookupOrCreateFn(rewriter, module, hexkl::getHmxMmF16FnName(),
                                argTys, voidTy);
}

static FailureOr<LLVM::LLVMFuncOp>
getAhToRmF16Fn(ModuleOp module, ConversionPatternRewriter &rewriter) {
  auto ctx = module->getContext();
  auto ptrTy = LLVM::LLVMPointerType::get(ctx);
  auto i32Ty = rewriter.getI32Type();
  SmallVector<Type, 3> argTys{ptrTy, i32Ty, i32Ty};
  auto voidTy = LLVM::LLVMVoidType::get(ctx);
  return LLVM::lookupOrCreateFn(rewriter, module,
                                hexkl::getHmxAhToRmF16FnName(), argTys, voidTy);
}

static FailureOr<LLVM::LLVMFuncOp>
getCopyF16ToF32SubmatrixFn(ModuleOp module,
                           ConversionPatternRewriter &rewriter) {
  auto ctx = module->getContext();
  auto ptrTy = LLVM::LLVMPointerType::get(ctx);
  auto i32Ty = rewriter.getI32Type();
  SmallVector<Type, 7> argTys{ptrTy, i32Ty, ptrTy, i32Ty, i32Ty, i32Ty, i32Ty};
  auto voidTy = LLVM::LLVMVoidType::get(ctx);
  return LLVM::lookupOrCreateFn(rewriter, module,
                                hexkl::getHmxCopyF16ToF32SubmatrixFnName(),
                                argTys, voidTy);
}

//===----------------------------------------------------------------------===//
// Lower micro ops to extern calls
//===----------------------------------------------------------------------===//

struct LowerSetupAccReadF16
    : public ConvertOpToLLVMPattern<hexkl::MicroHMXSetupAccReadF16Op> {
  using ConvertOpToLLVMPattern<
      hexkl::MicroHMXSetupAccReadF16Op>::ConvertOpToLLVMPattern;
  LogicalResult matchAndRewrite(hexkl::MicroHMXSetupAccReadF16Op op,
                                OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    // no-op

    // vtcm_base
    Value vtcmBase = alignedPtr(rewriter, loc, adaptor.getHmxBlock());
    MemRefDescriptor hmxDesc(adaptor.getHmxBlock());

    // vtcm_size (element count of memref<?xi8>)
    Value vtcmSizeIdx = hmxDesc.size(rewriter, loc, 0);
    Value vtcmSizeI32 = castToI32(rewriter, loc, vtcmSizeIdx);

    auto cfgFn =
        getNoArgI32RetFn(module, rewriter, hexkl::getHmxConfigSizeFnName());
    if (failed(cfgFn))
      return failure();
    auto cfgCallee = FlatSymbolRefAttr::get((*cfgFn).getOperation());
    Value cfgSize = rewriter
                        .create<LLVM::CallOp>(loc, rewriter.getI32Type(),
                                              cfgCallee, ValueRange{})
                        .getResult();

    // hmx_config_offset = vtcm_size - config_size
    Value cfgOffset = rewriter.create<LLVM::SubOp>(loc, vtcmSizeI32, cfgSize);

    // call setup_acc_read_f16(vtcm_base, hmx_config_offset)
    auto setupFn = getSetupAccReadF16Fn(module, rewriter);
    if (failed(setupFn))
      return failure();
    auto setupCallee = FlatSymbolRefAttr::get((*setupFn).getOperation());
    SmallVector<Value> args{vtcmBase, cfgOffset};
    rewriter.replaceOpWithNewOp<LLVM::CallOp>(op, TypeRange{}, setupCallee,
                                              args);
    return success();
  }
};

struct LowerAccClearF16
    : public ConvertOpToLLVMPattern<hexkl::MicroHMXAccClearF16Op> {
  using ConvertOpToLLVMPattern<
      hexkl::MicroHMXAccClearF16Op>::ConvertOpToLLVMPattern;
  LogicalResult matchAndRewrite(hexkl::MicroHMXAccClearF16Op op,
                                OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    // no-op
    auto fn = getAccClearF16Fn(module, rewriter);
    if (failed(fn))
      return failure();
    auto callee = FlatSymbolRefAttr::get((*fn).getOperation());
    rewriter.replaceOpWithNewOp<LLVM::CallOp>(op, TypeRange{}, callee,
                                              ValueRange{});
    return success();
  }
};

struct LowerAccReadF16
    : public ConvertOpToLLVMPattern<hexkl::MicroHMXAccReadF16Op> {
  using ConvertOpToLLVMPattern<
      hexkl::MicroHMXAccReadF16Op>::ConvertOpToLLVMPattern;
  LogicalResult matchAndRewrite(hexkl::MicroHMXAccReadF16Op op,
                                OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    // no-op

    Value vtcmBase = alignedPtr(rewriter, loc, adaptor.getHmxBlock());
    MemRefDescriptor hmxDesc(adaptor.getHmxBlock());
    Value vtcmSizeIdx = hmxDesc.size(rewriter, loc, 0);
    Value vtcmSizeI32 = castToI32(rewriter, loc, vtcmSizeIdx);

    auto cfgFn =
        getNoArgI32RetFn(module, rewriter, hexkl::getHmxConfigSizeFnName());
    if (failed(cfgFn))
      return failure();
    auto cfgCallee = FlatSymbolRefAttr::get((*cfgFn).getOperation());
    Value cfgSize = rewriter
                        .create<LLVM::CallOp>(loc, rewriter.getI32Type(),
                                              cfgCallee, ValueRange{})
                        .getResult();

    Value cfgOffset = rewriter.create<LLVM::SubOp>(loc, vtcmSizeI32, cfgSize);

    Value outOffset = adaptor.getOutOffset();
    auto fn = getAccReadF16Fn(module, rewriter);
    if (failed(fn))
      return failure();
    auto callee = FlatSymbolRefAttr::get((*fn).getOperation());
    SmallVector<Value> args{vtcmBase, cfgOffset, outOffset};
    rewriter.replaceOpWithNewOp<LLVM::CallOp>(op, TypeRange{}, callee, args);
    return success();
  }
};

struct LowerCopySubmatrixToF16
    : public ConvertOpToLLVMPattern<hexkl::MicroHMXCopySubmatrixToF16Op> {
  using ConvertOpToLLVMPattern<
      hexkl::MicroHMXCopySubmatrixToF16Op>::ConvertOpToLLVMPattern;
  LogicalResult matchAndRewrite(hexkl::MicroHMXCopySubmatrixToF16Op op,
                                OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    // no-op

    Value vtcmBase = alignedPtr(rewriter, loc, adaptor.getHmxBlock());
    Value srcPtr = alignedPtr(rewriter, loc, adaptor.getSrc());

    auto fn = getCopySubmatrixToF16Fn(module, rewriter);
    if (failed(fn))
      return failure();
    auto callee = FlatSymbolRefAttr::get((*fn).getOperation());
    SmallVector<Value> args{vtcmBase,
                            adaptor.getOutOffset(),
                            srcPtr,
                            adaptor.getTileRow(),
                            adaptor.getTileCol(),
                            adaptor.getInputRows(),
                            adaptor.getInputCols()};
    rewriter.replaceOpWithNewOp<LLVM::CallOp>(op, TypeRange{}, callee, args);
    return success();
  }
};

struct LowerRmToAhF16
    : public ConvertOpToLLVMPattern<hexkl::MicroHMXRmToAhF16Op> {
  using ConvertOpToLLVMPattern<
      hexkl::MicroHMXRmToAhF16Op>::ConvertOpToLLVMPattern;
  LogicalResult matchAndRewrite(hexkl::MicroHMXRmToAhF16Op op,
                                OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    // no-op
    Value vtcmBase = alignedPtr(rewriter, loc, adaptor.getHmxBlock());
    auto fn = getRmToAhF16Fn(module, rewriter);
    if (failed(fn))
      return failure();
    auto callee = FlatSymbolRefAttr::get((*fn).getOperation());
    SmallVector<Value> args{vtcmBase, adaptor.getActivationOutOffset(),
                            adaptor.getFlatInOffset()};
    rewriter.replaceOpWithNewOp<LLVM::CallOp>(op, TypeRange{}, callee, args);
    return success();
  }
};

struct LowerRmToWhF16
    : public ConvertOpToLLVMPattern<hexkl::MicroHMXRmToWhF16Op> {
  using ConvertOpToLLVMPattern<
      hexkl::MicroHMXRmToWhF16Op>::ConvertOpToLLVMPattern;
  LogicalResult matchAndRewrite(hexkl::MicroHMXRmToWhF16Op op,
                                OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    Value vtcmBase = alignedPtr(rewriter, loc, adaptor.getHmxBlock());
    Value srcPtr = alignedPtr(rewriter, loc, adaptor.getSrc());
    auto fn = getRmToWhF16Fn(module, rewriter);
    if (failed(fn))
      return failure();
    auto callee = FlatSymbolRefAttr::get((*fn).getOperation());
    SmallVector<Value> args{vtcmBase,
                            adaptor.getWeightOffset(),
                            srcPtr,
                            adaptor.getTileRow(),
                            adaptor.getTileCol(),
                            adaptor.getWtCols()};
    rewriter.replaceOpWithNewOp<LLVM::CallOp>(op, TypeRange{}, callee, args);
    return success();
  }
};

struct LowerMmF16 : public ConvertOpToLLVMPattern<hexkl::MicroHMXMmF16Op> {
  using ConvertOpToLLVMPattern<hexkl::MicroHMXMmF16Op>::ConvertOpToLLVMPattern;
  LogicalResult matchAndRewrite(hexkl::MicroHMXMmF16Op op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    Value vtcmBase = alignedPtr(rewriter, loc, adaptor.getHmxBlock());
    auto fn = getMmF16Fn(module, rewriter);
    if (failed(fn))
      return failure();
    auto callee = FlatSymbolRefAttr::get((*fn).getOperation());
    SmallVector<Value> args{vtcmBase, adaptor.getActivationOffset(),
                            adaptor.getWeightOffset()};
    rewriter.replaceOpWithNewOp<LLVM::CallOp>(op, TypeRange{}, callee, args);
    return success();
  }
};

struct LowerAhToRmF16
    : public ConvertOpToLLVMPattern<hexkl::MicroHMXAhToRmF16Op> {
  using ConvertOpToLLVMPattern<
      hexkl::MicroHMXAhToRmF16Op>::ConvertOpToLLVMPattern;
  LogicalResult matchAndRewrite(hexkl::MicroHMXAhToRmF16Op op,
                                OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    Value vtcmBase = alignedPtr(rewriter, loc, adaptor.getHmxBlock());
    auto fn = getAhToRmF16Fn(module, rewriter);
    if (failed(fn))
      return failure();
    auto callee = FlatSymbolRefAttr::get((*fn).getOperation());
    SmallVector<Value> args{vtcmBase, adaptor.getFlatOutOffset(),
                            adaptor.getActivationInOffset()};
    rewriter.replaceOpWithNewOp<LLVM::CallOp>(op, TypeRange{}, callee, args);
    return success();
  }
};

struct LowerCopyF16ToF32Submatrix
    : public ConvertOpToLLVMPattern<hexkl::MicroHMXCopyF16ToF32SubmatrixOp> {
  using ConvertOpToLLVMPattern<
      hexkl::MicroHMXCopyF16ToF32SubmatrixOp>::ConvertOpToLLVMPattern;
  LogicalResult matchAndRewrite(hexkl::MicroHMXCopyF16ToF32SubmatrixOp op,
                                OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    Value vtcmBase = alignedPtr(rewriter, loc, adaptor.getHmxBlock());
    Value dstPtr = alignedPtr(rewriter, loc, adaptor.getDst());
    auto fn = getCopyF16ToF32SubmatrixFn(module, rewriter);
    if (failed(fn))
      return failure();
    auto callee = FlatSymbolRefAttr::get((*fn).getOperation());
    SmallVector<Value> args{vtcmBase,
                            adaptor.getInOffset(),
                            dstPtr,
                            adaptor.getTileRow(),
                            adaptor.getTileCol(),
                            adaptor.getOutputRows(),
                            adaptor.getOutputCols()};
    rewriter.replaceOpWithNewOp<LLVM::CallOp>(op, TypeRange{}, callee, args);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Setup the Lowering Pass and patterns
//===----------------------------------------------------------------------===//

void populateHexKLToLLVMConversionPatterns(LLVMTypeConverter &converter,
                                           RewritePatternSet &patterns) {
  // Keep existing matmul lowering.
  patterns.add<LowerMatmul>(converter);
  // Lower micro ops to externs.
  patterns.add<LowerSetupAccReadF16, LowerAccClearF16, LowerAccReadF16,
               LowerCopySubmatrixToF16, LowerRmToAhF16, LowerRmToWhF16,
               LowerMmF16, LowerAhToRmF16, LowerCopyF16ToF32Submatrix>(
      converter);
}

struct HexKLToLLVMPass : public HexKLToLLVMBase<HexKLToLLVMPass> {
  using Base::Base;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<memref::MemRefDialect, LLVM::LLVMDialect, HexKLDialect>();
  }

  void runOnOperation() override {
    auto moduleOp = getOperation();
    MLIRContext *context = moduleOp->getContext();
    const auto &dataLayoutAnalysis = getAnalysis<DataLayoutAnalysis>();

    LLVMConversionTarget target(*context);
    RewritePatternSet patterns(context);
    LowerToLLVMOptions options(context,
                               dataLayoutAnalysis.getAtOrAbove(moduleOp));
    LLVMTypeConverter typeConverter(context, options);

    target.addLegalDialect<memref::MemRefDialect>();
    target.addIllegalDialect<HexKLDialect>();

    hexagon::addTypeConversions(context, typeConverter);
    populateHexKLToLLVMConversionPatterns(typeConverter, patterns);

    if (failed(applyPartialConversion(moduleOp, target, std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

std::unique_ptr<OperationPass<ModuleOp>> hexkl::createHexKLToLLVMPass() {
  return std::make_unique<HexKLToLLVMPass>();
}
