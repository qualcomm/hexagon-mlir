//===- LowerTPtr.cpp - Lower Tptr to LLVM ---------------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/DataLayoutAnalysis.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Ptr/IR/PtrDialect.h"
#include "mlir/Dialect/Ptr/IR/PtrOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

#include "mlir/Conversion/LLVMCommon/MemRefBuilder.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinTypes.h"

#include "llvm/Support/Debug.h"
#include <algorithm>
#include <numeric>
#include <vector>

#include "hexagon/Conversion/LinalgToLLVM/Common.h"
#include "hexagon/Dialect/HexagonTPtr/IR/HexagonTPtrDialect.h"
#include "hexagon/Transforms/Transforms.h"

#define DEBUG_TYPE "lower-tptr"

#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define DBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
using namespace hexagon;

#define GEN_PASS_DEF_LOWERTPTR
#include "hexagon/Transforms/Passes.h.inc"

namespace {

struct LowerTPtrPass : public ::impl::LowerTPtrBase<LowerTPtrPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<tptr::HexagonTPtrDialect, ptr::PtrDialect, LLVM::LLVMDialect>();
  }

  void runOnOperation() override;
};

/// Lower `tptr.type_offset`
struct LowerTypeOffset : public OpRewritePattern<tptr::HexagonTypeOffsetOp> {
  using OpRewritePattern::OpRewritePattern;

  LowerTypeOffset(MLIRContext *context, DataLayoutAnalysis &dla)
      : OpRewritePattern<tptr::HexagonTypeOffsetOp>(context), dataLayout(dla) {}

  LogicalResult matchAndRewrite(tptr::HexagonTypeOffsetOp op,
                                PatternRewriter &rewriter) const final {

    auto typeOffsetOp = cast<tptr::HexagonTypeOffsetOp>(op);
    Type baseType = typeOffsetOp.getBaseType();
    Type resultType = typeOffsetOp.getType();
    assert(baseType && "LowerTypeOffset: Base type cannot be null\n");

    const DataLayout &layout = dataLayout.getAtOrAbove(op);
    auto size = layout.getTypeSize(baseType);
    uint64_t sizeInBytes = size.getFixedValue();

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

// Lower `tptr.from_memref`
struct LowerFromMemref : public OpRewritePattern<tptr::HexagonFromMemrefOp> {
  LowerFromMemref(MLIRContext *context)
      : OpRewritePattern<tptr::HexagonFromMemrefOp>(context) {}

  LogicalResult matchAndRewrite(tptr::HexagonFromMemrefOp op,
                                PatternRewriter &rewriter) const final {
    Value asIdx = rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(
        op->getLoc(), op.getInput());

    auto cast = rewriter.create<UnrealizedConversionCastOp>(
        op.getLoc(), op.getType(), asIdx);
    rewriter.replaceAllUsesWith(op.getResult(), cast.getResult(0));
    return success();
  }
};

// Lower `tptr.ptradd` by converting to index-add. That will lower
// to `llvm.add _ : i64` and intoptr after convert-to-llvm.
struct LowerTPtrAdd : public OpRewritePattern<tptr::HexagonPtrAddOp> {

  LowerTPtrAdd(MLIRContext *context)
      : OpRewritePattern<tptr::HexagonPtrAddOp>(context) {}

  LogicalResult matchAndRewrite(tptr::HexagonPtrAddOp op,
                                PatternRewriter &rewriter) const final {
    auto loc = op.getLoc();
    auto cast = rewriter.create<UnrealizedConversionCastOp>(
        loc, IndexType::get(op.getContext()), op.getBase());

    Value offset = op.getOffset();
    if (!offset.getType().isIndex()) {
      offset = rewriter.create<arith::IndexCastOp>(loc, rewriter.getIndexType(),
                                                   offset);
    }
    auto addI = rewriter.create<arith::AddIOp>(loc, cast.getResult(0), offset);
    auto recast = rewriter.create<UnrealizedConversionCastOp>(
        loc, op.getBase().getType(), addI.getResult());

    rewriter.replaceAllUsesWith(op.getResult(), recast.getResult(0));
    rewriter.eraseOp(op);
    return success();
  }
};

// Lower `tptr.to_memref` by `inttoptr` of !ptr.ptr to `llvm.ptr`, and
// then building llvm.struct that is equivalent to result memref-type.
struct LowerToMemRef : public OpRewritePattern<tptr::HexagonToMemrefOp> {
  LowerToMemRef(MLIRContext *context)
      : OpRewritePattern<tptr::HexagonToMemrefOp>(context) {}

  LogicalResult matchAndRewrite(tptr::HexagonToMemrefOp op,
                                PatternRewriter &rewriter) const final {
    auto loc = op.getLoc();
    // ptrtoint : cast `!ptr.ptr` type to llvm opaque `!llvm.ptr` type.
    Value ptrAsIdx = rewriter
                         .create<UnrealizedConversionCastOp>(
                             loc, IndexType::get(op.getContext()), op.getArg())
                         .getResult(0);
    mlir::Type i64Type = rewriter.getIntegerType(64);
    Value ptrAsI64 =
        rewriter.create<arith::IndexCastOp>(loc, i64Type, ptrAsIdx).getResult();
    Type opaquePtrType = LLVM::LLVMPointerType::get(rewriter.getContext());
    Value intToPtr =
        rewriter.create<mlir::LLVM::IntToPtrOp>(loc, opaquePtrType, ptrAsI64);

    // construct `!llvm.struct` from the `memref` result type of `to_memref`.
    auto memrefType = cast<MemRefType>(op.getRes().getType());
    assert(memrefType.hasStaticShape() && "expect static shape as result type");

    auto rank = memrefType.getRank();
    auto shape = memrefType.getShape();
    auto [strides, offset] = memrefType.getStridesAndOffset();
    assert(offset == 0 && "dynamic offset is not materializable");

    auto arrayType = LLVM::LLVMArrayType::get(i64Type, rank);
    auto structType = LLVM::LLVMStructType::getLiteral(
        op.getContext(),
        {opaquePtrType, opaquePtrType, i64Type, arrayType, arrayType});

    // populate `!llvm.struct` with ptr and memref descriptor contents.
    Value undefStruct = rewriter.create<LLVM::UndefOp>(loc, structType);
    Value s0 =
        rewriter.create<LLVM::InsertValueOp>(loc, undefStruct, intToPtr, 0);
    Value s1 = rewriter.create<LLVM::InsertValueOp>(loc, s0, intToPtr, 1);
    Value zero =
        rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI64Type(), 0);
    Value s2 = rewriter.create<LLVM::InsertValueOp>(loc, s1, zero, 2);

    SmallVector<int64_t> size_idxs{3, 0};
    SmallVector<int64_t> stride_idxs{4, 0};
    Value structVal = s2;
    for (int i = 0; i < rank; ++i) {
      size_idxs[1] = i;
      Value size = rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI64Type(),
                                                     shape[i]);
      Value structI =
          rewriter.create<LLVM::InsertValueOp>(loc, structVal, size, size_idxs);

      stride_idxs[1] = i;
      Value stride = rewriter.create<LLVM::ConstantOp>(
          loc, rewriter.getI64Type(), strides[i]);
      structVal = rewriter.create<LLVM::InsertValueOp>(loc, structI, stride,
                                                       stride_idxs);
    }
    // cast back struct to memref type.
    Value toMemref = rewriter
                         .create<UnrealizedConversionCastOp>(
                             op.getLoc(), op.getRes().getType(), structVal)
                         .getResult(0);
    rewriter.replaceAllUsesWith(op.getResult(), toMemref);
    rewriter.eraseOp(op);
    return success();
  }
};

void LowerTPtrPass::runOnOperation() {
  auto &dla = getAnalysis<DataLayoutAnalysis>();
  MLIRContext *context = &getContext();

  RewritePatternSet patterns(context);
  patterns.add<LowerTypeOffset>(context, dla);
  patterns.add<LowerFromMemref, LowerTPtrAdd, LowerToMemRef>(context);

  if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
    signalPassFailure();
  }
}
} // namespace

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
hexagon::createLowerTPtrPass() {
  return std::make_unique<LowerTPtrPass>();
}
