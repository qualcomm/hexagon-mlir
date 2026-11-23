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

#define GEN_PASS_DEF_HEXKLTOLLVM
#include "hexagon/Conversion/HexKLToLLVM/Passes.h.inc"
#undef GEN_PASS_DEF_HEXKLTOLLVM

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

// Common helper: lower  memref descriptor and extract buffer pointer
static Value bufferPtr(ConversionPatternRewriter &rewriter,
                       const LLVMTypeConverter &converter, Location loc,
                       Value memrefDesc, Type memrefType) {
  auto memref = cast<MemRefType>(memrefType);
  MemRefDescriptor desc(memrefDesc);
  return desc.bufferPtr(rewriter, loc, converter, memref);
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
    auto &typeConverter = *getTypeConverter();
    auto fn = getMatmulRefFn(module, typeConverter, rewriter);
    if (failed(fn))
      return failure();

    // --- Extract aligned data pointers using helper function ---
    Value outPtr =
        bufferPtr(rewriter, typeConverter, loc, adaptor.getOuts(), resType);
    Value lhsPtr =
        bufferPtr(rewriter, typeConverter, loc, adaptor.getLhs(), lhsType);
    Value rhsPtr =
        bufferPtr(rewriter, typeConverter, loc, adaptor.getRhs(), rhsType);

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
    return LLVM::TruncOp::create(rewriter, loc, i32Ty, v);
  if (w < 32)
    return LLVM::ZExtOp::create(rewriter, loc, i32Ty, v);
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

// The config block occupies the VTCM tail: size - config size. Keep this shared
// by all accumulator operations and consistent with the launcher and daemon.
static FailureOr<Value>
computeHmxConfigOffset(ModuleOp module, ConversionPatternRewriter &rewriter,
                       Location loc, MemRefDescriptor hmxDesc) {
  // vtcm_size (element count of memref<?xi8>, i.e. bytes)
  Value vtcmSizeIdx = hmxDesc.size(rewriter, loc, 0);
  Value vtcmSizeI32 = castToI32(rewriter, loc, vtcmSizeIdx);

  auto cfgFn =
      getNoArgI32RetFn(module, rewriter, hexkl::getHmxConfigSizeFnName());
  if (failed(cfgFn))
    return failure();
  auto cfgCallee = FlatSymbolRefAttr::get((*cfgFn).getOperation());
  Value cfgSize = LLVM::CallOp::create(rewriter, loc, rewriter.getI32Type(),
                                       cfgCallee, ValueRange{})
                      .getResult();

  return Value(LLVM::SubOp::create(rewriter, loc, vtcmSizeI32, cfgSize));
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
getCopySubmatrixTo8bActivationFn(ModuleOp module,
                                 ConversionPatternRewriter &rewriter) {
  auto ctx = module->getContext();
  auto ptrTy = LLVM::LLVMPointerType::get(ctx);
  auto i32Ty = rewriter.getI32Type();
  SmallVector<Type, 7> argTys{ptrTy, i32Ty, ptrTy, i32Ty, i32Ty, i32Ty, i32Ty};
  auto voidTy = LLVM::LLVMVoidType::get(ctx);
  return LLVM::lookupOrCreateFn(
      rewriter, module, hexkl::getHmxCopySubmatrixTo8bActivationFnName(),
      argTys, voidTy);
}

static FailureOr<LLVM::LLVMFuncOp>
getRmToWhI8Fn(ModuleOp module, ConversionPatternRewriter &rewriter) {
  auto ctx = module->getContext();
  auto ptrTy = LLVM::LLVMPointerType::get(ctx);
  auto i32Ty = rewriter.getI32Type();
  SmallVector<Type, 6> argTys{ptrTy, i32Ty, ptrTy, i32Ty, i32Ty, i32Ty};
  auto voidTy = LLVM::LLVMVoidType::get(ctx);
  return LLVM::lookupOrCreateFn(rewriter, module, hexkl::getHmxRmToWhI8FnName(),
                                argTys, voidTy);
}

static FailureOr<LLVM::LLVMFuncOp>
getCopy32bToSubmatrixFn(ModuleOp module, ConversionPatternRewriter &rewriter) {
  auto ctx = module->getContext();
  auto ptrTy = LLVM::LLVMPointerType::get(ctx);
  auto i32Ty = rewriter.getI32Type();
  SmallVector<Type, 7> argTys{ptrTy, i32Ty, ptrTy, i32Ty, i32Ty, i32Ty, i32Ty};
  auto voidTy = LLVM::LLVMVoidType::get(ctx);
  return LLVM::lookupOrCreateFn(rewriter, module,
                                hexkl::getHmxCopy32bToSubmatrixFnName(), argTys,
                                voidTy);
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

static FailureOr<LLVM::LLVMFuncOp>
getHmxMicroDispatchFn(ModuleOp module, ConversionPatternRewriter &rewriter) {
  auto ctx = module->getContext();
  auto ptrTy = LLVM::LLVMPointerType::get(ctx);
  auto i32Ty = rewriter.getI32Type();
  auto i1Ty = rewriter.getI1Type();
  // void hmx_micro_dispatch(queue, worker_id, act, wgt, out, clear_acc,
  //                         write_out, job_kind)
  SmallVector<Type, 8> argTys{ptrTy, i32Ty, i32Ty, i32Ty,
                              i32Ty, i1Ty,  i1Ty,  i32Ty};
  auto voidTy = LLVM::LLVMVoidType::get(ctx);
  return LLVM::lookupOrCreateFn(
      rewriter, module, hexkl::getHmxMicroDispatchFnName(), argTys, voidTy);
}

static FailureOr<LLVM::LLVMFuncOp>
getHmxMicroWaitIdleFn(ModuleOp module, ConversionPatternRewriter &rewriter) {
  auto ctx = module->getContext();
  auto ptrTy = LLVM::LLVMPointerType::get(ctx);
  auto i32Ty = rewriter.getI32Type();
  // void hmx_micro_wait_idle(queue, worker_id)
  SmallVector<Type, 2> argTys{ptrTy, i32Ty};
  auto voidTy = LLVM::LLVMVoidType::get(ctx);
  return LLVM::lookupOrCreateFn(
      rewriter, module, hexkl::getHmxMicroWaitIdleFnName(), argTys, voidTy);
}

static FailureOr<LLVM::LLVMFuncOp>
getMacroNoArgI32Fn(ModuleOp module, ConversionPatternRewriter &rewriter,
                   StringRef name) {
  SmallVector<Type, 0> argTys;
  auto i32Ty = rewriter.getI32Type();
  return LLVM::lookupOrCreateFn(rewriter, module, name, argTys, i32Ty);
}

static FailureOr<LLVM::LLVMFuncOp>
getMacroRmToAhF16InplaceFn(ModuleOp module,
                           ConversionPatternRewriter &rewriter) {
  auto ctx = module->getContext();
  auto ptrTy = LLVM::LLVMPointerType::get(ctx);
  auto i32Ty = rewriter.getI32Type();
  SmallVector<Type, 3> argTys{i32Ty, i32Ty, ptrTy};
  return LLVM::lookupOrCreateFn(
      rewriter, module, hexkl::getMacroRmToAhF16InplaceFnName(), argTys, i32Ty);
}

static FailureOr<LLVM::LLVMFuncOp>
getMacroAhToRmF16InplaceFn(ModuleOp module,
                           ConversionPatternRewriter &rewriter) {
  auto ctx = module->getContext();
  auto ptrTy = LLVM::LLVMPointerType::get(ctx);
  auto i32Ty = rewriter.getI32Type();
  SmallVector<Type, 3> argTys{i32Ty, i32Ty, ptrTy};
  return LLVM::lookupOrCreateFn(
      rewriter, module, hexkl::getMacroAhToRmF16InplaceFnName(), argTys, i32Ty);
}

static FailureOr<LLVM::LLVMFuncOp>
getMacroMmF16Fn(ModuleOp module, ConversionPatternRewriter &rewriter) {
  auto ctx = module->getContext();
  auto ptrTy = LLVM::LLVMPointerType::get(ctx);
  auto i32Ty = rewriter.getI32Type();
  SmallVector<Type, 6> argTys{i32Ty, i32Ty, i32Ty, ptrTy, ptrTy, ptrTy};
  return LLVM::lookupOrCreateFn(rewriter, module, hexkl::getMacroMmF16FnName(),
                                argTys, i32Ty);
}

static FailureOr<LLVM::LLVMFuncOp>
getQurtMemCacheCleanFn(ModuleOp module, ConversionPatternRewriter &rewriter) {
  auto i32Ty = rewriter.getI32Type();
  SmallVector<Type, 4> argTys{i32Ty, i32Ty, i32Ty, i32Ty};
  return LLVM::lookupOrCreateFn(
      rewriter, module, hexkl::getQurtMemCacheCleanFnName(), argTys, i32Ty);
}

//===----------------------------------------------------------------------===//
// Lower macro ops to extern calls
//===----------------------------------------------------------------------===//

struct LowerMacroInitialize
    : public ConvertOpToLLVMPattern<hexkl::MacroInitializeOp> {
  using ConvertOpToLLVMPattern<
      hexkl::MacroInitializeOp>::ConvertOpToLLVMPattern;
  LogicalResult matchAndRewrite(hexkl::MacroInitializeOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const {
    ModuleOp module = op->getParentOfType<ModuleOp>();
    auto fn =
        getMacroNoArgI32Fn(module, rewriter, hexkl::getMacroInitializeFnName());
    if (failed(fn))
      return failure();

    auto callee = FlatSymbolRefAttr::get((*fn).getOperation());
    LLVM::CallOp::create(rewriter, op.getLoc(), rewriter.getI32Type(), callee,
                         ValueRange{});
    rewriter.eraseOp(op);
    return success();
  }
};

struct LowerMacroFinalize
    : public ConvertOpToLLVMPattern<hexkl::MacroFinalizeOp> {
  using ConvertOpToLLVMPattern<hexkl::MacroFinalizeOp>::ConvertOpToLLVMPattern;
  LogicalResult matchAndRewrite(hexkl::MacroFinalizeOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const {
    ModuleOp module = op->getParentOfType<ModuleOp>();
    auto fn =
        getMacroNoArgI32Fn(module, rewriter, hexkl::getMacroFinalizeFnName());
    if (failed(fn))
      return failure();

    auto callee = FlatSymbolRefAttr::get((*fn).getOperation());
    LLVM::CallOp::create(rewriter, op.getLoc(), rewriter.getI32Type(), callee,
                         ValueRange{});
    rewriter.eraseOp(op);
    return success();
  }
};

struct LowerMacroLockHmx
    : public ConvertOpToLLVMPattern<hexkl::MacroLockHmxOp> {
  using ConvertOpToLLVMPattern<hexkl::MacroLockHmxOp>::ConvertOpToLLVMPattern;
  LogicalResult matchAndRewrite(hexkl::MacroLockHmxOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const {
    ModuleOp module = op->getParentOfType<ModuleOp>();
    auto fn =
        getMacroNoArgI32Fn(module, rewriter, hexkl::getMacroLockHmxFnName());
    if (failed(fn))
      return failure();

    auto callee = FlatSymbolRefAttr::get((*fn).getOperation());
    LLVM::CallOp::create(rewriter, op.getLoc(), rewriter.getI32Type(), callee,
                         ValueRange{});
    rewriter.eraseOp(op);
    return success();
  }
};

struct LowerMacroUnlockHmx
    : public ConvertOpToLLVMPattern<hexkl::MacroUnlockHmxOp> {
  using ConvertOpToLLVMPattern<hexkl::MacroUnlockHmxOp>::ConvertOpToLLVMPattern;
  LogicalResult matchAndRewrite(hexkl::MacroUnlockHmxOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const {
    ModuleOp module = op->getParentOfType<ModuleOp>();
    auto fn =
        getMacroNoArgI32Fn(module, rewriter, hexkl::getMacroUnlockHmxFnName());
    if (failed(fn))
      return failure();

    auto callee = FlatSymbolRefAttr::get((*fn).getOperation());
    LLVM::CallOp::create(rewriter, op.getLoc(), rewriter.getI32Type(), callee,
                         ValueRange{});
    rewriter.eraseOp(op);
    return success();
  }
};

struct LowerMacroRmToAhF16Inplace
    : public ConvertOpToLLVMPattern<hexkl::MacroRmToAhF16InplaceOp> {
  using ConvertOpToLLVMPattern<
      hexkl::MacroRmToAhF16InplaceOp>::ConvertOpToLLVMPattern;
  LogicalResult matchAndRewrite(hexkl::MacroRmToAhF16InplaceOp op,
                                OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const {
    ModuleOp module = op->getParentOfType<ModuleOp>();
    auto fn = getMacroRmToAhF16InplaceFn(module, rewriter);
    if (failed(fn))
      return failure();
    auto callee = FlatSymbolRefAttr::get((*fn).getOperation());
    SmallVector<Value, 3> args{adaptor.getM(), adaptor.getK(),
                               adaptor.getPtr()};
    LLVM::CallOp::create(rewriter, op.getLoc(), rewriter.getI32Type(), callee,
                         args);
    rewriter.eraseOp(op);
    return success();
  }
};

struct LowerMacroAhToRmF16Inplace
    : public ConvertOpToLLVMPattern<hexkl::MacroAhToRmF16InplaceOp> {
  using ConvertOpToLLVMPattern<
      hexkl::MacroAhToRmF16InplaceOp>::ConvertOpToLLVMPattern;
  LogicalResult matchAndRewrite(hexkl::MacroAhToRmF16InplaceOp op,
                                OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const {
    ModuleOp module = op->getParentOfType<ModuleOp>();
    auto fn = getMacroAhToRmF16InplaceFn(module, rewriter);
    if (failed(fn))
      return failure();
    auto callee = FlatSymbolRefAttr::get((*fn).getOperation());
    SmallVector<Value, 3> args{adaptor.getM(), adaptor.getN(),
                               adaptor.getPtr()};
    LLVM::CallOp::create(rewriter, op.getLoc(), rewriter.getI32Type(), callee,
                         args);
    rewriter.eraseOp(op);
    return success();
  }
};

struct LowerMacroMmF16 : public ConvertOpToLLVMPattern<hexkl::MacroMmF16Op> {
  using ConvertOpToLLVMPattern<hexkl::MacroMmF16Op>::ConvertOpToLLVMPattern;
  LogicalResult matchAndRewrite(hexkl::MacroMmF16Op op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const {
    ModuleOp module = op->getParentOfType<ModuleOp>();
    auto fn = getMacroMmF16Fn(module, rewriter);
    if (failed(fn))
      return failure();
    auto callee = FlatSymbolRefAttr::get((*fn).getOperation());
    SmallVector<Value, 6> args{adaptor.getM(),      adaptor.getN(),
                               adaptor.getK(),      adaptor.getOutPtr(),
                               adaptor.getActPtr(), adaptor.getWgtPtr()};
    LLVM::CallOp::create(rewriter, op.getLoc(), rewriter.getI32Type(), callee,
                         args);
    rewriter.eraseOp(op);
    return success();
  }
};

struct LowerQurtMemCacheClean
    : public ConvertOpToLLVMPattern<hexkl::QurtMemCacheCleanOp> {
  using ConvertOpToLLVMPattern<
      hexkl::QurtMemCacheCleanOp>::ConvertOpToLLVMPattern;
  LogicalResult matchAndRewrite(hexkl::QurtMemCacheCleanOp op,
                                OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const {
    ModuleOp module = op->getParentOfType<ModuleOp>();
    auto fn = getQurtMemCacheCleanFn(module, rewriter);
    if (failed(fn))
      return failure();
    auto callee = FlatSymbolRefAttr::get((*fn).getOperation());
    SmallVector<Value, 4> args{adaptor.getPtrI32(), adaptor.getSizeBytesI32(),
                               adaptor.getOp(), adaptor.getCache()};
    LLVM::CallOp::create(rewriter, op.getLoc(), rewriter.getI32Type(), callee,
                         args);
    rewriter.eraseOp(op);
    return success();
  }
};

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
    auto &converter = *getTypeConverter();
    Value vtcmBase = bufferPtr(rewriter, converter, loc, adaptor.getHmxBlock(),
                               op.getHmxBlock().getType());
    MemRefDescriptor hmxDesc(adaptor.getHmxBlock());
    auto cfgOffset = computeHmxConfigOffset(module, rewriter, loc, hmxDesc);
    if (failed(cfgOffset))
      return failure();

    // call setup_acc_read_f16(vtcm_base, hmx_config_offset)
    auto setupFn = getSetupAccReadF16Fn(module, rewriter);
    if (failed(setupFn))
      return failure();
    auto setupCallee = FlatSymbolRefAttr::get((*setupFn).getOperation());
    SmallVector<Value> args{vtcmBase, *cfgOffset};
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

    auto &converter = *getTypeConverter();
    Value vtcmBase = bufferPtr(rewriter, converter, loc, adaptor.getHmxBlock(),
                               op.getHmxBlock().getType());
    MemRefDescriptor hmxDesc(adaptor.getHmxBlock());
    auto cfgOffset = computeHmxConfigOffset(module, rewriter, loc, hmxDesc);
    if (failed(cfgOffset))
      return failure();

    Value outOffset = adaptor.getOutOffset();
    auto fn = getAccReadF16Fn(module, rewriter);
    if (failed(fn))
      return failure();
    auto callee = FlatSymbolRefAttr::get((*fn).getOperation());
    SmallVector<Value> args{vtcmBase, *cfgOffset, outOffset};
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

    auto &converter = *getTypeConverter();
    Value vtcmBase = bufferPtr(rewriter, converter, loc, adaptor.getHmxBlock(),
                               op.getHmxBlock().getType());
    Value srcPtr = bufferPtr(rewriter, converter, loc, adaptor.getSrc(),
                             op.getSrc().getType());

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
    auto &converter = *getTypeConverter();
    Value vtcmBase = bufferPtr(rewriter, converter, loc, adaptor.getHmxBlock(),
                               op.getHmxBlock().getType());
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
    auto &converter = *getTypeConverter();
    Value vtcmBase = bufferPtr(rewriter, converter, loc, adaptor.getHmxBlock(),
                               op.getHmxBlock().getType());
    Value srcPtr = bufferPtr(rewriter, converter, loc, adaptor.getSrc(),
                             op.getSrc().getType());
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
    auto &converter = *getTypeConverter();
    Value vtcmBase = bufferPtr(rewriter, converter, loc, adaptor.getHmxBlock(),
                               op.getHmxBlock().getType());
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
    auto &converter = *getTypeConverter();
    Value vtcmBase = bufferPtr(rewriter, converter, loc, adaptor.getHmxBlock(),
                               op.getHmxBlock().getType());
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

//===----------------------------------------------------------------------===//
// Lower HMX quantized data-movement micro ops to extern calls
//===----------------------------------------------------------------------===//

struct LowerCopySubmatrixTo8bActivation
    : public ConvertOpToLLVMPattern<
          hexkl::MicroHMXCopySubmatrixTo8bActivationOp> {
  using ConvertOpToLLVMPattern<
      hexkl::MicroHMXCopySubmatrixTo8bActivationOp>::ConvertOpToLLVMPattern;
  LogicalResult matchAndRewrite(hexkl::MicroHMXCopySubmatrixTo8bActivationOp op,
                                OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    auto &converter = *getTypeConverter();
    Value vtcmBase = bufferPtr(rewriter, converter, loc, adaptor.getHmxBlock(),
                               op.getHmxBlock().getType());
    Value srcPtr = bufferPtr(rewriter, converter, loc, adaptor.getSrc(),
                             op.getSrc().getType());

    auto fn = getCopySubmatrixTo8bActivationFn(module, rewriter);
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

struct LowerRmToWhI8
    : public ConvertOpToLLVMPattern<hexkl::MicroHMXRmToWhI8Op> {
  using ConvertOpToLLVMPattern<
      hexkl::MicroHMXRmToWhI8Op>::ConvertOpToLLVMPattern;
  LogicalResult matchAndRewrite(hexkl::MicroHMXRmToWhI8Op op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    auto &converter = *getTypeConverter();
    Value vtcmBase = bufferPtr(rewriter, converter, loc, adaptor.getHmxBlock(),
                               op.getHmxBlock().getType());
    Value srcPtr = bufferPtr(rewriter, converter, loc, adaptor.getSrc(),
                             op.getSrc().getType());
    auto fn = getRmToWhI8Fn(module, rewriter);
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

struct LowerCopy32bToSubmatrix
    : public ConvertOpToLLVMPattern<hexkl::MicroHMXCopy32bToSubmatrixOp> {
  using ConvertOpToLLVMPattern<
      hexkl::MicroHMXCopy32bToSubmatrixOp>::ConvertOpToLLVMPattern;
  LogicalResult matchAndRewrite(hexkl::MicroHMXCopy32bToSubmatrixOp op,
                                OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    auto &converter = *getTypeConverter();
    Value vtcmBase = bufferPtr(rewriter, converter, loc, adaptor.getHmxBlock(),
                               op.getHmxBlock().getType());
    Value dstPtr = bufferPtr(rewriter, converter, loc, adaptor.getDst(),
                             op.getDst().getType());
    auto fn = getCopy32bToSubmatrixFn(module, rewriter);
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

struct LowerCopyF16ToF32Submatrix
    : public ConvertOpToLLVMPattern<hexkl::MicroHMXCopyF16ToF32SubmatrixOp> {
  using ConvertOpToLLVMPattern<
      hexkl::MicroHMXCopyF16ToF32SubmatrixOp>::ConvertOpToLLVMPattern;
  LogicalResult matchAndRewrite(hexkl::MicroHMXCopyF16ToF32SubmatrixOp op,
                                OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    auto &converter = *getTypeConverter();
    Value vtcmBase = bufferPtr(rewriter, converter, loc, adaptor.getHmxBlock(),
                               op.getHmxBlock().getType());
    Value dstPtr = bufferPtr(rewriter, converter, loc, adaptor.getDst(),
                             op.getDst().getType());
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
// Lower HMX micro daemon queue ops to extern calls
//===----------------------------------------------------------------------===//

struct LowerMicroHmxDispatch
    : public ConvertOpToLLVMPattern<hexkl::MicroHMXDispatchOp> {
  using ConvertOpToLLVMPattern<
      hexkl::MicroHMXDispatchOp>::ConvertOpToLLVMPattern;
  LogicalResult matchAndRewrite(hexkl::MicroHMXDispatchOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    auto fn = getHmxMicroDispatchFn(module, rewriter);
    if (failed(fn))
      return failure();

    // job_kind is a compile-time string on the op; materialise it as the
    // HMXMicroJobKind enum value the runtime expects. The op verifier already
    // restricted it to the two accepted spellings.
    int64_t kindVal = op.getJobKind() == "u8i8" ? 1 : 0;
    Value jobKind =
        LLVM::ConstantOp::create(rewriter, loc, rewriter.getI32Type(),
                                 rewriter.getI32IntegerAttr(kindVal));

    // The queue arrives as the injected !llvm.ptr kernel argument; every other
    // operand is a scalar (offsets are already-computed i32s). No memref
    // descriptor extraction is needed -- the daemon holds vtcm_base itself.
    auto callee = FlatSymbolRefAttr::get((*fn).getOperation());
    SmallVector<Value> args{adaptor.getQueue(),     adaptor.getWorkerId(),
                            adaptor.getActOffset(), adaptor.getWgtOffset(),
                            adaptor.getOutOffset(), adaptor.getClearAcc(),
                            adaptor.getWriteOut(),  jobKind};
    rewriter.replaceOpWithNewOp<LLVM::CallOp>(op, TypeRange{}, callee, args);
    return success();
  }
};

struct LowerMicroHmxWaitIdle
    : public ConvertOpToLLVMPattern<hexkl::MicroHMXWaitIdleOp> {
  using ConvertOpToLLVMPattern<
      hexkl::MicroHMXWaitIdleOp>::ConvertOpToLLVMPattern;
  LogicalResult matchAndRewrite(hexkl::MicroHMXWaitIdleOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const {
    ModuleOp module = op->getParentOfType<ModuleOp>();
    auto fn = getHmxMicroWaitIdleFn(module, rewriter);
    if (failed(fn))
      return failure();
    auto callee = FlatSymbolRefAttr::get((*fn).getOperation());
    SmallVector<Value> args{adaptor.getQueue(), adaptor.getWorkerId()};
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

  // Lower macro ops to externs.
  patterns
      .add<LowerMacroInitialize, LowerMacroFinalize, LowerMacroLockHmx,
           LowerMacroUnlockHmx, LowerMacroRmToAhF16Inplace,
           LowerMacroAhToRmF16Inplace, LowerMacroMmF16, LowerQurtMemCacheClean>(
          converter);

  // Lower micro ops to externs.
  patterns.add<LowerSetupAccReadF16, LowerAccClearF16, LowerAccReadF16,
               LowerCopySubmatrixToF16, LowerRmToAhF16, LowerRmToWhF16,
               LowerMmF16, LowerAhToRmF16, LowerCopyF16ToF32Submatrix>(
      converter);

  // Lower quantized data-movement micro ops to externs.
  patterns.add<LowerCopySubmatrixTo8bActivation, LowerRmToWhI8,
               LowerCopy32bToSubmatrix>(converter);

  // Lower HMX micro daemon queue ops to externs.
  patterns.add<LowerMicroHmxDispatch, LowerMicroHmxWaitIdle>(converter);
}

struct HexKLToLLVMPass : public ::impl::HexKLToLLVMBase<HexKLToLLVMPass> {
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
