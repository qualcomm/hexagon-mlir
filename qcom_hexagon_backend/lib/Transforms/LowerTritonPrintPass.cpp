//===- LowerTritonPrintPass.cpp - lower tt.print to runtime calls ---------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// Lowers triton::PrintOp (tl.device_print) to calls into the hexagon runtime
// print shims (hexagon_print_scalar_* / hexagon_print_tensor_*). The emitted
// symbol carries a _dsp suffix. The hex and isSigned print attributes are
// forwarded to the shim.
//
//===----------------------------------------------------------------------===//

#include "hexagon/Transforms/Transforms.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "triton/Dialect/Triton/IR/Dialect.h"

#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"

using namespace mlir;
using namespace hexagon;

#define GEN_PASS_DEF_LOWERTRITONPRINT
#include "hexagon/Transforms/Passes.h.inc"

namespace {

// Return a !llvm.ptr to the first byte of a private, null-terminated global
// string holding `content`, deduplicated on the symbol name. The symbol encodes
// a hash of the exact content, so distinct prefixes never share a global even
// when their sanitized spellings coincide.
static Value getOrCreatePrefixGlobal(Location loc, PatternRewriter &rewriter,
                                     ModuleOp module, StringRef content) {
  auto ctx = rewriter.getContext();
  auto ptrTy = LLVM::LLVMPointerType::get(ctx);

  llvm::SmallString<64> sym("triton_print_prefix_");
  for (char c : content.take_front(32))
    sym.push_back(llvm::isAlnum(c) ? c : '_');
  sym += "_";
  sym += llvm::utohexstr(llvm::hash_value(content), /*LowerCase=*/true);

  if (!module.lookupSymbol<LLVM::GlobalOp>(sym)) {
    llvm::SmallString<64> buf(content);
    buf.push_back('\0');
    auto arrTy = LLVM::LLVMArrayType::get(IntegerType::get(ctx, 8),
                                          static_cast<unsigned>(buf.size()));
    auto strAttr = StringAttr::get(ctx, StringRef(buf.data(), buf.size()));
    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPointToStart(module.getBody());
    LLVM::GlobalOp::create(rewriter, loc, arrTy, /*isConstant=*/true,
                           LLVM::Linkage::Private, sym, strAttr);
  }

  auto addr = LLVM::AddressOfOp::create(rewriter, loc, ptrTy, sym);
  auto zero = LLVM::ConstantOp::create(rewriter, loc, rewriter.getI64Type(), 0);
  return LLVM::GEPOp::create(rewriter, loc, ptrTy, IntegerType::get(ctx, 8),
                             addr, ValueRange{zero});
}

// Look up, or declare at module start, a private extern LLVM function.
static LLVM::LLVMFuncOp getOrDeclareFn(Location loc, PatternRewriter &rewriter,
                                       ModuleOp module, StringRef name,
                                       LLVM::LLVMFunctionType fnTy) {
  auto fn = module.lookupSymbol<LLVM::LLVMFuncOp>(name);
  if (!fn) {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(module.getBody());
    fn = LLVM::LLVMFuncOp::create(rewriter, loc, name, fnTy);
    fn.setVisibility(SymbolTable::Visibility::Private);
  }
  return fn;
}

struct LowerTritonPrintPattern final
    : public OpRewritePattern<triton::PrintOp> {
  std::string suffix;
  LowerTritonPrintPattern(MLIRContext *ctx, StringRef suffix)
      : OpRewritePattern(ctx), suffix(suffix) {}

  // Map a print element type to its shim type token, or nullopt if unsupported.
  static std::optional<StringRef> shimTypeSuffix(Type elemTy) {
    if (elemTy.isF32())
      return StringRef("f32");
    if (elemTy.isInteger(32))
      return StringRef("i32");
    return std::nullopt;
  }

  LogicalResult matchAndRewrite(triton::PrintOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    MLIRContext *ctx = rewriter.getContext();
    ModuleOp module = op->getParentOfType<ModuleOp>();

    auto ptrTy = LLVM::LLVMPointerType::get(ctx);
    auto i32Ty = rewriter.getI32Type();
    auto i64Ty = rewriter.getIntegerType(64);
    auto f32Ty = Float32Type::get(ctx);

    StringRef prefix = op.getPrefix();
    bool hex = op.getHex();
    ArrayRef<int32_t> isSigned = op.getIsSigned();

    if (op.getArgs().empty()) {
      rewriter.eraseOp(op);
      return success();
    }

    Value hexVal =
        arith::ConstantIntOp::create(rewriter, loc, hex ? 1 : 0, /*width=*/32);

    // Validate all operands up front so we don't partially rewrite.
    for (auto operand : op.getArgs()) {
      Type ty = operand.getType();
      Type elemTy = ty;
      if (auto tensorTy = dyn_cast<RankedTensorType>(ty)) {
        if (!tensorTy.hasStaticShape())
          return rewriter.notifyMatchFailure(
              op, "only static-shaped tensor operands are supported");
        elemTy = tensorTy.getElementType();
      }
      if (!shimTypeSuffix(elemTy).has_value())
        return rewriter.notifyMatchFailure(op,
                                           "unsupported print element type");
    }

    Value prefixPtr = getOrCreatePrefixGlobal(loc, rewriter, module, prefix);

    for (auto it : llvm::enumerate(op.getArgs())) {
      unsigned idx = it.index();
      Value operand = it.value();
      Type ty = operand.getType();

      bool argSigned = idx < isSigned.size() ? (isSigned[idx] != 0) : true;
      Value signedVal =
          arith::ConstantIntOp::create(rewriter, loc, argSigned ? 1 : 0,
                                       /*width=*/32);

      if (auto tensorTy = dyn_cast<RankedTensorType>(ty)) {
        Type elemTy = tensorTy.getElementType();
        StringRef tySuffix = *shimTypeSuffix(elemTy);

        auto memrefType =
            MemRefType::get(tensorTy.getShape(), tensorTy.getElementType());
        Value memref = bufferization::ToBufferOp::create(rewriter, loc,
                                                         memrefType, operand);
        Value ptrIndex = memref::ExtractAlignedPointerAsIndexOp::create(
            rewriter, loc, memref);
        Value ptrInt =
            arith::IndexCastOp::create(rewriter, loc, i64Ty, ptrIndex);
        Value dataPtr =
            LLVM::IntToPtrOp::create(rewriter, loc, ptrTy, ptrInt).getResult();

        Value numElems = arith::ConstantIntOp::create(
            rewriter, loc, tensorTy.getNumElements(), /*width=*/64);

        llvm::SmallString<48> fnName("hexagon_print_tensor_");
        fnName += tySuffix;
        fnName += suffix;
        SmallVector<Type> argTys{ptrTy, i32Ty, ptrTy, i64Ty};
        SmallVector<Value> args{prefixPtr, hexVal, dataPtr, numElems};
        if (elemTy.isInteger(32)) {
          argTys.push_back(i32Ty);
          args.push_back(signedVal);
        }
        auto fnTy = LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(ctx),
                                                argTys, /*isVarArg=*/false);
        auto fn = getOrDeclareFn(loc, rewriter, module, fnName, fnTy);
        LLVM::CallOp::create(rewriter, loc, fn, args);
      } else {
        StringRef tySuffix = *shimTypeSuffix(ty);

        llvm::SmallString<48> fnName("hexagon_print_scalar_");
        fnName += tySuffix;
        fnName += suffix;
        SmallVector<Type> argTys{ptrTy, i32Ty};
        SmallVector<Value> args{prefixPtr, hexVal};
        if (ty.isF32()) {
          argTys.push_back(f32Ty);
          args.push_back(operand);
        } else {
          argTys.push_back(i32Ty);
          args.push_back(operand);
          argTys.push_back(i32Ty);
          args.push_back(signedVal);
        }
        auto fnTy = LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(ctx),
                                                argTys, /*isVarArg=*/false);
        auto fn = getOrDeclareFn(loc, rewriter, module, fnName, fnTy);
        LLVM::CallOp::create(rewriter, loc, fn, args);
      }
    }

    rewriter.eraseOp(op);
    return success();
  }
};

struct LowerTritonPrintPass
    : public ::impl::LowerTritonPrintBase<LowerTritonPrintPass> {
  using LowerTritonPrintBase::LowerTritonPrintBase;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    std::string suffix = "_dsp";
    RewritePatternSet patterns(context);
    patterns.add<LowerTritonPrintPattern>(context, suffix);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, bufferization::BufferizationDialect,
                    memref::MemRefDialect, LLVM::LLVMDialect,
                    tensor::TensorDialect>();
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
hexagon::createLowerTritonPrintPass(const LowerTritonPrintOptions &options) {
  return std::make_unique<LowerTritonPrintPass>(options);
}
