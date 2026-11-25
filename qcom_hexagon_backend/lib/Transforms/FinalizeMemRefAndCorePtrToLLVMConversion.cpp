//===- FinalizeMemRefAndCorePtrToLLVMConversion.cpp - Convert ptr and
// memref dialect ops to LLVM ---------------------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// Converts `ptr` and `memref` dialect ops to LLVM, then collapses non-zero
// address spaces (VTCM) to address space 0.
//
// These three tasks (ptr lowering, memref lowering, address space collapsing)
// are combined into a single pass because of ordering constraints:
//
//   1) ptr and memref must be lowered together: memrefs of ptrs
//      (e.g. memref<4x!ptr.ptr<...>>) require both conversions in the same
//      pattern set — upstream FinalizeMemRefToLLVM rejects ptr element types.
//
//   2) ptr must be lowered before collapsing address spaces: ptr ops carry
//      MemorySpaceAttributeInterface attributes that the collapse logic
//      cannot process.
//
//   3) Address space collapsing must happen in the same pass as memref
//      lowering: memref-to-LLVM emits `llvm.call @free` with the source
//      memref's address space (e.g. !llvm.ptr<1>), which is illegal against
//      the space-0 @free declaration unless immediately collapsed.
//
//===----------------------------------------------------------------------===//

#include "hexagon/Transforms/Transforms.h"

#include "hexagon/Common/Common.h"
#include "hexagon/Dialect/HexagonTPtr/IR/HexagonTPtrDialect.h"

#include "mlir/Analysis/DataLayoutAnalysis.h"
#include "mlir/Conversion/ConvertToLLVM/ToLLVMInterface.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/MemRefBuilder.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/PtrToLLVM/PtrToLLVM.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/Ptr/IR/PtrDialect.h"
#include "mlir/Dialect/Ptr/IR/PtrOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

#define DEBUG_TYPE "finalize-memref-and-core-ptr-to-llvm-conversion"
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define DBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
using namespace hexagon;

namespace mlir {
namespace hexagon {
#define GEN_PASS_DEF_FINALIZEMEMREFANDCOREPTRTOLLVMCONVERSION
#include "hexagon/Transforms/Passes.h.inc"
} // namespace hexagon
} // namespace mlir

namespace {

// If FromPtrOp has sufficient metadata to be lowered, lower it
struct FromPtrNoMetadataOpConversion
    : public ConvertOpToLLVMPattern<ptr::FromPtrOp> {
  explicit FromPtrNoMetadataOpConversion(const LLVMTypeConverter &converter,
                                         PatternBenefit benefit = 2)
      : ConvertOpToLLVMPattern<ptr::FromPtrOp>(converter, benefit) {}

  LogicalResult
  matchAndRewrite(ptr::FromPtrOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    auto memrefType = dyn_cast<MemRefType>(op.getType());
    if (!memrefType)
      return rewriter.notifyMatchFailure(
          op, "can only convert ptr.from_ptr to ranked memref result");

    // Let the upstream pattern handle the metadata-carrying case.
    if (op.getMetadata())
      return rewriter.notifyMatchFailure(
          op, "metadata present; use normal FromPtrOpConversion");

    if (!memrefType.hasStaticShape())
      return rewriter.notifyMatchFailure(
          op, "cannot reconstruct dynamic-shape memref without metadata");

    SmallVector<int64_t> strides;
    int64_t offset = 0;
    if (failed(memrefType.getStridesAndOffset(strides, offset)))
      return rewriter.notifyMatchFailure(
          op, "cannot compute static strides/offset");

    if (ShapedType::isDynamic(offset))
      return rewriter.notifyMatchFailure(
          op, "cannot reconstruct dynamic offset without metadata");

    if (llvm::any_of(strides, ShapedType::isDynamic))
      return rewriter.notifyMatchFailure(
          op, "cannot reconstruct dynamic strides without metadata");

    // At this point the descriptor can be synthesized from the pointer.
    Value ptr = adaptor.getPtr();

    MemRefDescriptor descriptor = MemRefDescriptor::fromStaticShape(
        rewriter, loc, *getTypeConverter(), memrefType, ptr);

    Value descriptorValue = descriptor;
    rewriter.replaceOp(op, descriptorValue);
    return success();
  }
};

/// Lower `ptr.type_offset` to an `arith.constant` equal to sizeof(elementType).
/// Mirrors LowerTypeOffset for tptr.type_offset. Registered with higher
/// benefit than the upstream generic pattern (which lowers via a
/// getelementptr-on-null-pointer + ptrtoint sequence): when the element type's
/// size is statically known via the DataLayout, folding directly to a
/// constant here is simpler and avoids emitting dead pointer arithmetic.
struct PtrTypeOffsetOpConversion : public OpRewritePattern<ptr::TypeOffsetOp> {
  PtrTypeOffsetOpConversion(MLIRContext *context, DataLayoutAnalysis &dla,
                            PatternBenefit benefit = 2)
      : OpRewritePattern<ptr::TypeOffsetOp>(context, benefit), dataLayout(dla) {
  }

  LogicalResult matchAndRewrite(ptr::TypeOffsetOp op,
                                PatternRewriter &rewriter) const final {
    Type elementType = op.getElementType();
    Type resultType = op.getResult().getType();

    const DataLayout &layout = dataLayout.getAtOrAbove(op);
    uint64_t sizeInBytes = layout.getTypeSize(elementType).getFixedValue();

    Attribute constAttr;
    if (resultType.isIndex()) {
      constAttr = rewriter.getIndexAttr(sizeInBytes);
    } else if (isa<IntegerType>(resultType)) {
      constAttr = rewriter.getIntegerAttr(resultType, sizeInBytes);
    } else {
      return rewriter.notifyMatchFailure(
          op, "Unsupported result type for TypeOffset");
    }
    rewriter.replaceOpWithNewOp<arith::ConstantOp>(op, resultType,
                                                   cast<TypedAttr>(constAttr));
    return success();
  }

private:
  DataLayoutAnalysis &dataLayout;
};

// Collapses non-zero LLVM address spaces (namely TCM) down to DDR (space 0):
// the real Hexagon LLVM backend has no address-space distinction. Must run
// after ptr/memref-to-LLVM conversion completes, since the `ptr` dialect
// still carries memory-space identifiers this logic doesn't understand.
Type getCollapsedType(Type type, llvm::DenseMap<Type, Type> &visited) {
  if (auto it = visited.find(type); it != visited.end())
    return it->second; // return cached result

  if (!(isa<LLVM::LLVMStructType>(type) || isa<LLVM::LLVMPointerType>(type) ||
        isa<MemRefType>(type)))
    return type; // return unchanged

  // Recursive collapse.
  return llvm::TypeSwitch<Type, Type>(type)
      .Case<LLVM::LLVMStructType>([&](auto structTy) {
        MLIRContext *ctx = structTy.getContext();
        visited[structTy] = structTy; // cache before recursion

        SmallVector<Type> memberTys;
        for (Type elemTy : structTy.getBody()) {
          Type collapsed = getCollapsedType(elemTy, visited);
          memberTys.push_back(collapsed);
        }

        auto newStruct = LLVM::LLVMStructType::getLiteral(ctx, memberTys);
        visited[structTy] = newStruct; // update cache
        return newStruct;
      })
      .Case<LLVM::LLVMPointerType>([&](auto ptrTy) {
        MLIRContext *ctx = ptrTy.getContext();
        return LLVM::LLVMPointerType::get(ctx, DEFAULT_DDR_ADDRESS_SPACE);
      })
      .Case<MemRefType>(
          [&](auto memrefTy) { // This case is needed despite lowering memrefs
                               // for leftover unrealized conversion cast pairs
            MLIRContext *ctx = memrefTy.getContext();
            Type elemTy = getCollapsedType(memrefTy.getElementType(), visited);
            Attribute newSpace = IntegerAttr::get(IntegerType::get(ctx, 64),
                                                  DEFAULT_DDR_ADDRESS_SPACE);
            return MemRefType::get(memrefTy.getShape(), elemTy,
                                   memrefTy.getLayout(), newSpace);
          })
      .Default([](Type ty) -> Type { return ty; });
}

// Collapses address spaces across `funcOp`'s signature, block arguments, and
// op results. `funcOp` must NOT be the pass's own current operation (i.e.
// the pass driving this call must be scoped above `funcOp`, e.g. a ModuleOp
// pass walking its LLVM::LLVMFuncOp children): this rebuilds the function
// with a new type and erases the original via IRRewriter::replaceOp, which
// is only safe when `funcOp` isn't the operation the surrounding
// OpToOpPassAdaptor is currently iterating/bookkeeping on.
void collapseFuncAddressSpaces(LLVM::LLVMFuncOp funcOp) {
  LLVM::LLVMFunctionType oldFuncType = funcOp.getFunctionType();
  llvm::DenseMap<Type, Type> visited; // single map cache reused below.

  // Build collapsed input and result types
  SmallVector<Type> newInputs;
  bool changed = false;
  for (Type inputType : oldFuncType.getParams()) {
    Type collapsed = getCollapsedType(inputType, visited);
    newInputs.push_back(collapsed);
    changed |= (collapsed != inputType);
  }

  Type oldResultType = oldFuncType.getReturnType();
  Type newResultType = getCollapsedType(oldResultType, visited);
  changed |= (newResultType != oldResultType);

  auto updateBlockTypes = [&](Region &region) {
    for (Block &block : region.getBlocks()) {
      for (BlockArgument arg : block.getArguments())
        arg.setType(getCollapsedType(arg.getType(), visited));

      block.walk([&](Operation *op) {
        for (unsigned i = 0; i < op->getNumResults(); ++i)
          op->getResult(i).setType(
              getCollapsedType(op->getResult(i).getType(), visited));
      });
    }
  };

  if (!changed) {
    updateBlockTypes(funcOp.getBody());
    return;
  }

  auto newFuncType = LLVM::LLVMFunctionType::get(newResultType, newInputs,
                                                 oldFuncType.isVarArg());

  IRRewriter rewriter(funcOp.getContext());
  rewriter.setInsertionPoint(funcOp);
  auto newFuncOp = LLVM::LLVMFuncOp::create(
      rewriter, funcOp.getLoc(), funcOp.getName(), newFuncType,
      funcOp.getLinkage(), funcOp.getDsoLocal(), funcOp.getCConv());
  // Copy over all other attributes (e.g. `passthrough`), but not
  // `function_type`/`sym_name` -- those were just set above from
  // `newFuncType`/`funcOp.getName()` and must not be clobbered back to the
  // old (uncollapsed) function type.
  for (NamedAttribute attr : funcOp->getAttrs()) {
    if (attr.getName() == funcOp.getFunctionTypeAttrName() ||
        attr.getName() == funcOp.getSymNameAttrName())
      continue;
    newFuncOp->setAttr(attr.getName(), attr.getValue());
  }
  rewriter.inlineRegionBefore(funcOp.getBody(), newFuncOp.getBody(),
                              newFuncOp.getBody().end());
  updateBlockTypes(newFuncOp.getBody());
  rewriter.replaceOp(funcOp, newFuncOp);
}

// Both convertGlobals and convertFunc need a type converter that resolves
// the `tptr` dialect's default memory-space attr to a concrete LLVM address
// space; without this, a memref/ptr type carrying that attr (e.g. a
// `memref.global` with no explicit memory space) fails to convert.
static void addHexagonMemorySpaceConversions(LLVMTypeConverter &typeConverter,
                                             MLIRContext *ctx) {
  typeConverter.addTypeAttributeConversion(
      [ctx](BaseMemRefType type,
            tptr::HexagonDefaultMemorySpaceAttr defaultSpaceAttr) {
        return IntegerAttr::get(IntegerType::get(ctx, 64),
                                hexagon::DEFAULT_DDR_ADDRESS_SPACE);
      });

  typeConverter.addTypeAttributeConversion(
      [ctx](ptr::PtrType type,
            tptr::HexagonDefaultMemorySpaceAttr defaultSpaceAttr) {
        return IntegerAttr::get(IntegerType::get(ctx, 64),
                                hexagon::DEFAULT_DDR_ADDRESS_SPACE);
      });
}

struct FinalizeMemRefAndCorePtrToLLVMConversionPass
    : public hexagon::impl::FinalizeMemRefAndCorePtrToLLVMConversionBase<
          FinalizeMemRefAndCorePtrToLLVMConversionPass> {
  FinalizeMemRefAndCorePtrToLLVMConversionPass() = default;
  explicit FinalizeMemRefAndCorePtrToLLVMConversionPass(
      const FinalizeMemRefAndCorePtrToLLVMConversionOptions &options)
      : FinalizeMemRefAndCorePtrToLLVMConversionBase(options) {}

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();

    // `memref.global` ops live directly in the module body, as siblings of
    // the functions -- they are never visited by convertFunc's per-function
    // applyPartialConversion(funcOp, ...) below, since that conversion's
    // root is the function, not the module. Lower them here explicitly so
    // that in-function `llvm.mlir.addressof` ops (converted from
    // `memref.get_global`) resolve to an actual `llvm.mlir.global` rather
    // than a stale `memref.global`.
    if (failed(convertGlobals(moduleOp))) {
      signalPassFailure();
      return;
    }

    for (auto funcOp :
         llvm::make_early_inc_range(moduleOp.getOps<FunctionOpInterface>())) {
      if (failed(convertFunc(funcOp))) {
        signalPassFailure();
        return;
      }
    }
  }

  LogicalResult convertGlobals(ModuleOp moduleOp) {
    MLIRContext *ctx = &getContext();

    SmallVector<Operation *> globals;
    for (auto globalOp : moduleOp.getOps<memref::GlobalOp>())
      globals.push_back(globalOp);
    if (globals.empty())
      return success();

    LowerToLLVMOptions options(ctx);
    LLVMTypeConverter typeConverter(ctx, options);
    addHexagonMemorySpaceConversions(typeConverter, ctx);

    RewritePatternSet patterns(ctx);
    LLVMConversionTarget target(*ctx);
    target.addLegalDialect<LLVM::LLVMDialect>();
    target.addIllegalOp<memref::GlobalOp>();
    target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });

    populateFinalizeMemRefToLLVMConversionPatterns(typeConverter, patterns);

    return applyPartialConversion(globals, target, std::move(patterns));
  }

  LogicalResult convertFunc(FunctionOpInterface funcOp) {
    if (failed(convertFuncPtrDialect(funcOp)))
      return failure();

    if (failed(convertFuncMemRefDialect(funcOp)))
      return failure();

    if (enableCollapseAddressSpace) {
      // Collapsing the address space currently only works for llvm.func not
      // func.func
      auto llvmFuncOp = cast<LLVM::LLVMFuncOp>(funcOp.getOperation());
      collapseFuncAddressSpaces(llvmFuncOp);
    }
    return success();
  }

  // Phase 1: lower the `ptr` dialect to LLVM
  LogicalResult convertFuncPtrDialect(FunctionOpInterface funcOp) {
    MLIRContext *ctx = &getContext();

    LowerToLLVMOptions options(ctx);
    LLVMTypeConverter typeConverter(ctx, options);
    addHexagonMemorySpaceConversions(typeConverter, ctx);

    RewritePatternSet patterns(ctx);
    LLVMConversionTarget target(*ctx);

    target.addLegalDialect<LLVM::LLVMDialect>();
    target.addLegalDialect<memref::MemRefDialect>();
    target.addIllegalDialect<ptr::PtrDialect>();

    target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });

    auto *ptrDialect = ctx->getLoadedDialect<ptr::PtrDialect>();
    if (!ptrDialect) {
      funcOp.emitError("ptr dialect is not loaded");
      return failure();
    }

    auto *iface =
        ptrDialect->getRegisteredInterface<ConvertToLLVMPatternInterface>();

    if (!iface) {
      funcOp.emitError("No PtrToLLVM ConvertToLLVM interface");
      return failure();
    }

    // Reuse upstream PtrToLLVM patterns to lower the Ptr dialect
    //  Ptr dialect needs lowering because certain vllm Triton kernels and
    //  helion kernels fall back to a scalar pointer path
    iface->populateConvertToLLVMConversionPatterns(target, typeConverter,
                                                   patterns);
    patterns.add<FromPtrNoMetadataOpConversion>(typeConverter);
    patterns.add<PtrTypeOffsetOpConversion>(ctx,
                                            getAnalysis<DataLayoutAnalysis>());

    return applyPartialConversion(funcOp, target, std::move(patterns));
  }

  // Phase 2: finalize the `memref` dialect to LLVM
  LogicalResult convertFuncMemRefDialect(FunctionOpInterface funcOp) {
    MLIRContext *ctx = &getContext();

    LowerToLLVMOptions options(ctx);
    LLVMTypeConverter typeConverter(ctx, options);
    addHexagonMemorySpaceConversions(typeConverter, ctx);

    RewritePatternSet patterns(ctx);
    LLVMConversionTarget target(*ctx);

    target.addLegalDialect<LLVM::LLVMDialect>();
    target.addIllegalDialect<memref::MemRefDialect>();

    target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });

    // Ptr patterns are also needed to convert memrefs with ptr element types.
    ptr::populatePtrToLLVMConversionPatterns(typeConverter, patterns);
    populateFinalizeMemRefToLLVMConversionPatterns(typeConverter, patterns);

    return applyPartialConversion(funcOp, target, std::move(patterns));
  }
};
} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::hexagon::createFinalizeMemRefAndCorePtrToLLVMConversionPass(
    const FinalizeMemRefAndCorePtrToLLVMConversionOptions &options) {
  return std::make_unique<FinalizeMemRefAndCorePtrToLLVMConversionPass>(
      options);
}
