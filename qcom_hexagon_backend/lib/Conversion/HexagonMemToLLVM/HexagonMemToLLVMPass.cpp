//===- HexagonMemToLLVMPass.cpp - HexagonMem to LLVM Pass -----------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to convert HexagonMem dialect to LLVM dialect.
//===----------------------------------------------------------------------===//

#include "hexagon/Common/Common.h"
#include "hexagon/Conversion/HexagonMemToLLVM/HexagonMemExternalFnNames.h"
#include "hexagon/Conversion/HexagonMemToLLVM/HexagonMemToLLVM.h"
#include "hexagon/Conversion/HexagonMemToLLVM/Passes.h"
#include "hexagon/Conversion/LinalgToLLVM/LinalgToLLVM.h"
#include "hexagon/Dialect/HexagonMem/IR/HexagonMemDialect.h"
#include "mlir/Analysis/DataLayoutAnalysis.h"
#include "mlir/Conversion/ConvertToLLVM/ToLLVMInterface.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/MemRefBuilder.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/LLVMIR/FunctionCallUtils.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/MemRef/Utils/MemRefUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "hexagonmem-to-llvm"

#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define DBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
using namespace mlir::hexagonmem;

#define GEN_PASS_CLASSES
#include "hexagon/Conversion/HexagonMemToLLVM/Passes.h.inc"

namespace {

static LLVM::LLVMPointerType getPtrTy(MLIRContext *context) {
  return LLVM::LLVMPointerType::get(context);
}

static LLVM::LLVMVoidType getVoidTy(MLIRContext *context) {
  return LLVM::LLVMVoidType::get(context);
}

static LLVM::ConstantOp getI32Constant(ConversionPatternRewriter &rewriter,
                                       Location loc, int64_t val) {
  Type paramTy = rewriter.getI32Type();
  return rewriter.create<LLVM::ConstantOp>(loc, paramTy, val);
}

static int64_t getAllocationSize(MemRefType cTy) {
  return cTy.getNumElements() * cTy.getElementTypeBitWidth() / 8;
}

static Value computeAllocationSize(MemRefType type,
                                   ConversionPatternRewriter &rewriter,
                                   MemRefDescriptor desc, Location loc,
                                   Type indexType, Value sizeInBytes) {

  if (type.hasStaticShape()) {
    auto size = getAllocationSize(type);
    return getI32Constant(rewriter, loc, size);
  }

  // Compute number of elements.
  Value numElements = rewriter.create<LLVM::ConstantOp>(
      loc, indexType, rewriter.getIndexAttr(1));
  for (int pos = 0; pos < type.getRank(); ++pos) {
    auto size = desc.size(rewriter, loc, pos);
    numElements = rewriter.create<LLVM::MulOp>(loc, numElements, size);
  }
  Value totalSize = rewriter.create<LLVM::MulOp>(loc, numElements, sizeInBytes);
  Value sizeI32 =
      rewriter.create<LLVM::TruncOp>(loc, rewriter.getI32Type(), totalSize);
  return sizeI32;
}

/// Runtime call prototype for memref type:
/// void* (size_t bytes, size_t alignment, bool isVtcm)
static FailureOr<LLVM::LLVMFuncOp>
getAllocFn(Operation *module, StringRef fnName,
           ConversionPatternRewriter &rewriter) {
  MLIRContext *context = module->getContext();
  return LLVM::lookupOrCreateFn(
      rewriter, module, fnName,
      {rewriter.getI32Type(), rewriter.getI64Type(), rewriter.getI1Type()},
      getPtrTy(context));
}

/// Dealloc just needs the pointer: void(void *ptr)
static FailureOr<LLVM::LLVMFuncOp>
getDeallocFn(ModuleOp module, StringRef fnName,
             ConversionPatternRewriter &rewriter) {
  MLIRContext *context = module->getContext();
  return LLVM::lookupOrCreateFn(rewriter, module, fnName, {getPtrTy(context)},
                                getVoidTy(context));
}

/// Common code to determine the type used and its memory space
template <typename AllocDeallocOp>
std::tuple<LogicalResult, bool>
computeTypeInfo(AllocDeallocOp op, ConversionPatternRewriter &rewriter) {
  auto type = op.getBuffer().getType();
  bool isInVtcm = false;
  if (auto memrefType = mlir::dyn_cast<MemRefType>(type)) {
    isInVtcm = hexagon::isInVTCMAddressSpace(memrefType);
  } else {
    llvm::errs() << "Invalid type passed to hexagonmem operation\n";
    return {failure(), isInVtcm};
  }

  return {success(), isInVtcm};
}

//===----------------------------------------------------------------------===//
// Lower hexagonmem::AllocOp
//===----------------------------------------------------------------------===//

struct LowerAlloc : public ConvertOpToLLVMPattern<hexagonmem::AllocOp> {
  using ConvertOpToLLVMPattern<hexagonmem::AllocOp>::ConvertOpToLLVMPattern;

  /// Lower `hexagonmem.alloc` to `llvm.call @hexagon_runtime_alloc_1d` call
  LogicalResult matchAndRewrite(hexagonmem::AllocOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const {
    auto [result, isInVtcm] =
        computeTypeInfo<hexagonmem::AllocOp>(op, rewriter);

    if (failed(result))
      return result;

    auto loc = op->getLoc();
    auto type = op.getBuffer().getType();

    // Replace "hexagonmem.alloc" with "memref.alloc" for flat DDR allocations
    if (!isInVtcm) {
      rewriter.replaceOpWithNewOp<memref::AllocOp>(
          op, mlir::cast<MemRefType>(type));
      return success();
    }

    Value alignmentValue;
    auto alignmentAttr = op.getAlignment();
    Type indexType = getIndexType();
    alignmentValue =
        createIndexAttrConstant(rewriter, loc, indexType, alignmentAttr);

    auto allocFnName = getAllocFnName();
    FailureOr<LLVM::LLVMFuncOp> funcOp = getAllocFn(
        op->getParentWithTrait<OpTrait::SymbolTable>(), allocFnName, rewriter);
    if (failed(funcOp))
      return failure();

    Value isInVtcmValue =
        rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI1Type(), isInVtcm);

    auto origMemRefType = mlir::cast<MemRefType>(type);
    auto memRefType = mlir::affine::normalizeMemRefType(origMemRefType);
    Value size;

    // Get actual sizes of the memref as values: static sizes are constant
    // values and dynamic sizes are passed to 'alloc' as operands.  In case of
    // zero-dimensional memref, assume a scalar (size 1).
    SmallVector<Value, 4> sizes;
    SmallVector<Value, 4> strides;
    Value sizeAsI32;
    this->getMemRefDescriptorSizes(loc, memRefType, adaptor.getOperands(),
                                   rewriter, sizes, strides, size,
                                   /* sizeInBytes */ true);
    sizeAsI32 =
        rewriter.create<LLVM::TruncOp>(loc, rewriter.getIntegerType(32), size);
    mlir::LLVM::CallOp callOp = rewriter.create<LLVM::CallOp>(
        loc, funcOp.value(),
        ValueRange({sizeAsI32, alignmentValue, isInVtcmValue}));
    auto memRefDescriptor = this->createMemRefDescriptor(
        loc, memRefType, callOp.getResult(), callOp.getResult(), sizes, strides,
        rewriter);
    rewriter.replaceOp(op, {memRefDescriptor});

    return success();
  }
};

//===----------------------------------------------------------------------===//
// Lower hexagonmem::DeallocOp
//===----------------------------------------------------------------------===//

struct LowerDealloc : public ConvertOpToLLVMPattern<hexagonmem::DeallocOp> {
  using ConvertOpToLLVMPattern<hexagonmem::DeallocOp>::ConvertOpToLLVMPattern;
  /// Lower `hexagonmem.dealloc` to `llvm.call @hexagon_runtime_free_1d` call
  LogicalResult matchAndRewrite(hexagonmem::DeallocOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const {
    auto [result, isInVtcm] =
        computeTypeInfo<hexagonmem::DeallocOp>(op, rewriter);

    if (failed(result))
      return result;

    auto loc = op->getLoc();
    auto module = op->getParentOfType<ModuleOp>();

    // Replace "hexagonmem.dealloc" with "memref.dealloc" for flat DDR
    // allocations
    if (!isInVtcm) {
      rewriter.replaceOpWithNewOp<memref::DeallocOp>(op, op.getBuffer());
      return success();
    }

    auto deallocFnName = getDeallocFnName();
    FailureOr<LLVM::LLVMFuncOp> funcOp =
        getDeallocFn(module, deallocFnName, rewriter);
    if (failed(funcOp))
      return failure();

    MemRefDescriptor bufferDesc(adaptor.getBuffer());
    auto bufferPtr = bufferDesc.alignedPtr(rewriter, loc);

    rewriter.replaceOpWithNewOp<LLVM::CallOp>(op, funcOp.value(),
                                              ValueRange({bufferPtr}));
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Lower hexagonmem::CopyOp
//===----------------------------------------------------------------------===//

static FailureOr<LLVM::LLVMFuncOp>
getCopyFn(ModuleOp module, StringRef fnName,
          ConversionPatternRewriter &rewriter) {
  MLIRContext *context = module->getContext();
  return LLVM::lookupOrCreateFn(rewriter, module, fnName,
                                {getPtrTy(context), getPtrTy(context),
                                 rewriter.getI32Type(), rewriter.getI1Type(),
                                 rewriter.getI1Type()},
                                getVoidTy(context));
}

struct LowerCopy : public ConvertOpToLLVMPattern<hexagonmem::CopyOp> {
  using ConvertOpToLLVMPattern<hexagonmem::CopyOp>::ConvertOpToLLVMPattern;

  // This method is an exact replica of the one in MemRefToLLVM lowering.
  // TODO: Expose the upstream method and use that instead of a local copy
  LogicalResult
  lowerToMemCopyFunctionCall(hexagonmem::CopyOp op, OpAdaptor adaptor,
                             ConversionPatternRewriter &rewriter) const {
    auto loc = op.getLoc();
    auto srcType = cast<BaseMemRefType>(op.getSource().getType());
    auto targetType = cast<BaseMemRefType>(op.getTarget().getType());

    // First make sure we have an unranked memref descriptor representation.
    auto makeUnranked = [&, this](Value ranked, MemRefType type) {
      auto rank = rewriter.create<LLVM::ConstantOp>(loc, getIndexType(),
                                                    type.getRank());
      auto *typeConverter = getTypeConverter();
      auto ptr =
          typeConverter->promoteOneMemRefDescriptor(loc, ranked, rewriter);

      auto unrankedType =
          UnrankedMemRefType::get(type.getElementType(), type.getMemorySpace());
      return UnrankedMemRefDescriptor::pack(
          rewriter, loc, *typeConverter, unrankedType, ValueRange{rank, ptr});
    };

    // Save stack position before promoting descriptors
    auto stackSaveOp = rewriter.create<LLVM::StackSaveOp>(loc, getPtrType());

    auto srcMemRefType = dyn_cast<MemRefType>(srcType);
    Value unrankedSource =
        srcMemRefType ? makeUnranked(adaptor.getSource(), srcMemRefType)
                      : adaptor.getSource();
    auto targetMemRefType = dyn_cast<MemRefType>(targetType);
    Value unrankedTarget =
        targetMemRefType ? makeUnranked(adaptor.getTarget(), targetMemRefType)
                         : adaptor.getTarget();

    // Now promote the unranked descriptors to the stack.
    auto one = rewriter.create<LLVM::ConstantOp>(loc, getIndexType(),
                                                 rewriter.getIndexAttr(1));
    auto promote = [&](Value desc) {
      auto ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());
      auto allocated =
          rewriter.create<LLVM::AllocaOp>(loc, ptrType, desc.getType(), one);
      rewriter.create<LLVM::StoreOp>(loc, desc, allocated);
      return allocated;
    };

    auto sourcePtr = promote(unrankedSource);
    auto targetPtr = promote(unrankedTarget);

    // Derive size from llvm.getelementptr which will account for any
    // potential alignment
    auto elemSize = getSizeInBytes(loc, srcType.getElementType(), rewriter);
    auto copyFn = LLVM::lookupOrCreateMemRefCopyFn(
        rewriter, op->getParentOfType<ModuleOp>(), getIndexType(),
        sourcePtr.getType());
    if (failed(copyFn))
      return failure();
    rewriter.create<LLVM::CallOp>(loc, copyFn.value(),
                                  ValueRange{elemSize, sourcePtr, targetPtr});

    // Restore stack used for descriptors
    rewriter.create<LLVM::StackRestoreOp>(loc, stackSaveOp);

    rewriter.eraseOp(op);

    return success();
  }

  /// Lower `hexagonmem.copy` to `llvm.call @hexagon_runtime_copy` call
  LogicalResult matchAndRewrite(hexagonmem::CopyOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const {
    auto loc = op->getLoc();
    auto module = op->getParentOfType<ModuleOp>();

    auto sourceType = op.getSource().getType();
    auto targetType = op.getTarget().getType();

    // Verify that both source and target are memref types
    if (!mlir::isa<MemRefType>(sourceType) ||
        !mlir::isa<MemRefType>(targetType)) {
      llvm::errs() << "hexagonmem.copy op expects memref types\n";
      return failure();
    }

    auto srcMemrefType = mlir::cast<MemRefType>(sourceType);
    auto tgtMemrefType = mlir::cast<MemRefType>(targetType);

    bool sourceIsVTCM = hexagon::isInVTCMAddressSpace(srcMemrefType);
    bool targetIsVTCM = hexagon::isInVTCMAddressSpace(tgtMemrefType);

    if (!sourceIsVTCM && !targetIsVTCM) {
      rewriter.replaceOpWithNewOp<memref::CopyOp>(op, op.getSource(),
                                                  op.getTarget());
      return success();
    }

    if (!hexagon::isContiguousMemrefType(srcMemrefType) ||
        !hexagon::isContiguousMemrefType(tgtMemrefType)) {
      return lowerToMemCopyFunctionCall(op, adaptor, rewriter);
    }

    auto copyFnName = getCopyFnName();
    FailureOr<LLVM::LLVMFuncOp> funcOp =
        getCopyFn(module, copyFnName, rewriter);
    if (failed(funcOp))
      return failure();

    auto getDescAndPtr =
        [&](MemRefType memRefType,
            Value value) -> std::tuple<MemRefDescriptor, Value> {
      Type elementType =
          typeConverter->convertType(memRefType.getElementType());
      MemRefDescriptor desc(value);
      Value basePtr = desc.alignedPtr(rewriter, loc);
      Value offset = desc.offset(rewriter, loc);
      return {desc, rewriter.create<LLVM::GEPOp>(loc, basePtr.getType(),
                                                 elementType, basePtr, offset)};
    };

    auto [sourceDesc, sourcePtr] =
        getDescAndPtr(srcMemrefType, adaptor.getSource());
    auto [targetDesc, targetPtr] =
        getDescAndPtr(tgtMemrefType, adaptor.getTarget());

    Value elementSizeInBytes =
        getSizeInBytes(loc, srcMemrefType.getElementType(), rewriter);
    Value copySize =
        computeAllocationSize(srcMemrefType, rewriter, sourceDesc, loc,
                              getIndexType(), elementSizeInBytes);

    Value sourceIsVTCMValue = rewriter.create<LLVM::ConstantOp>(
        loc, rewriter.getI1Type(), sourceIsVTCM);
    Value targetIsVTCMValue = rewriter.create<LLVM::ConstantOp>(
        loc, rewriter.getI1Type(), targetIsVTCM);
    rewriter.replaceOpWithNewOp<LLVM::CallOp>(
        op, funcOp.value(),
        ValueRange({targetPtr, sourcePtr, copySize, targetIsVTCMValue,
                    sourceIsVTCMValue}));

    return success();
  }
};

//===----------------------------------------------------------------------===//
// Setup the Lowering Pass and patterns
//===----------------------------------------------------------------------===//

void populateHexagonMemToLLVMConversionPatterns(LLVMTypeConverter &converter,
                                                RewritePatternSet &patterns) {
  patterns.add<LowerAlloc, LowerDealloc, LowerCopy>(converter);
}

struct HexagonMemToLLVMPass
    : public HexagonMemToLLVMBase<HexagonMemToLLVMPass> {
  using Base::Base;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<memref::MemRefDialect, LLVM::LLVMDialect, HexagonMemDialect>();
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
    target.addIllegalDialect<HexagonMemDialect>();

    hexagon::addTypeConversions(context, typeConverter);
    populateHexagonMemToLLVMConversionPatterns(typeConverter, patterns);

    if (failed(applyPartialConversion(moduleOp, target, std::move(patterns))))
      signalPassFailure();
  }
};

/// Implement the interface to convert HexagonMem to LLVM.
struct HexagonMemToLLVMDialectInterface : public ConvertToLLVMPatternInterface {
  using ConvertToLLVMPatternInterface::ConvertToLLVMPatternInterface;
  void loadDependentDialects(MLIRContext *context) const final {
    context->loadDialect<LLVM::LLVMDialect>();
  }

  /// Hook for derived dialect interface to provide conversion patterns
  /// and mark dialect legal for the conversion target.
  void populateConvertToLLVMConversionPatterns(
      ConversionTarget &target, LLVMTypeConverter &typeConverter,
      RewritePatternSet &patterns) const final {
    hexagon::addTypeConversions(getContext(), typeConverter);
    populateHexagonMemToLLVMConversionPatterns(typeConverter, patterns);
  }
};

} // namespace

void mlir::hexagonmem::registerConvertHexagonMemToLLVMInterface(
    DialectRegistry &registry) {
  registry.addExtension(
      +[](MLIRContext *ctx, hexagonmem::HexagonMemDialect *dialect) {
        dialect->addInterfaces<HexagonMemToLLVMDialectInterface>();
      });
}

std::unique_ptr<OperationPass<ModuleOp>>
hexagonmem::createHexagonMemToLLVMPass() {
  return std::make_unique<HexagonMemToLLVMPass>();
}
