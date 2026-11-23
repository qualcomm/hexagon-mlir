//===- LegalizeFP8CastsToLLVMCallPass.cpp -------------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// Legalizes residual fp8<->f16 lowering fragments that still contain
// UnrealizedConversionCast/arith ops and are not directly LLVM-lowerable.
//
// The pass rewrites exactly two vector<128> patterns:
//
// 1. i8 -> fp8 -> f16:
// ```
//   %fp8_vec = builtin.unrealized_conversion_cast %x
//       : vector<128xi8> to vector<128xf8E4M3FNUZ>
//   %fp16_vec = arith.extf %fp8_vec fastmath<fast>
//       : vector<128xf8E4M3FNUZ> to vector<128xf16>
// ```
// is replaced by:
// ```
//   %arg_vec = llvm.bitcast %x : vector<128xi8> to vector<32xi32>
//   %conv_vec = llvm.call @convert_fp8_fp16_fnuz(%arg_vec)
//       : (vector<32xi32>) -> vector<64xi32>
//   %fp16_vec = llvm.bitcast %conv_vec : vector<64xi32> to vector<128xf16>
// ```
//
// 2. f16 -> fp8 -> i8:
// ```
//   %fp8_vec = arith.truncf %x fastmath<fast>
//       : vector<128xf16> to vector<128xf8E4M3FNUZ>
//   %i8_vec = builtin.unrealized_conversion_cast %fp8_vec
//       : vector<128xf8E4M3FNUZ> to vector<128xi8>
// ```
// is replaced by:
// ```
//   %arg_vec = llvm.bitcast %x : vector<128xf16> to vector<64xi32>
//   %conv_vec = llvm.call @convert_fp16_fp8_fnuz(%arg_vec)
//       : (vector<64xi32>) -> vector<32xi32>
//   %i8_vec = llvm.bitcast %conv_vec : vector<32xi32> to vector<128xi8>
// ```
//
// Runtime callee selection is type-driven (E4M3FN vs E4M3FNUZ). Helper LLVM
// function declarations are created once per module as private symbols.
// i32-vector arguments/results are used as a bit-preserving ABI transport.
//
//===----------------------------------------------------------------------===//

#include <iterator>

#include "hexagon/Transforms/Transforms.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;
using namespace hexagon;

#define GEN_PASS_DEF_LEGALIZEFP8CASTSTOLLVMCALL
#include "hexagon/Transforms/Passes.h.inc"

namespace {

static FlatSymbolRefAttr getOrCreateLLVMFunc(ModuleOp module,
                                             OpBuilder &builder, StringRef name,
                                             Type retTy,
                                             ArrayRef<Type> argTys) {
  if (auto fn = module.lookupSymbol<LLVM::LLVMFuncOp>(name))
    return FlatSymbolRefAttr::get(builder.getContext(), fn.getSymName());

  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPoint(module.getBody(),
                            std::prev(module.getBody()->end()));
  auto fnTy = LLVM::LLVMFunctionType::get(retTy, argTys);
  auto fn = LLVM::LLVMFuncOp::create(builder, module.getLoc(), name, fnTy);
  fn.setPrivate();
  return FlatSymbolRefAttr::get(builder.getContext(), fn.getSymName());
}

static bool isVecI8(Type t, int64_t n) {
  auto v = dyn_cast<VectorType>(t);
  return v && v.getRank() == 1 && v.getShape()[0] == n &&
         v.getElementType().isInteger(8);
}

static bool isVecF16(Type t, int64_t n) {
  auto v = dyn_cast<VectorType>(t);
  return v && v.getRank() == 1 && v.getShape()[0] == n &&
         v.getElementType().isF16();
}

enum class FP8Kind { None, E4M3FNUZ, E4M3FN };

static FP8Kind getVecFP8Kind(Type t, int64_t n) {
  auto v = dyn_cast<VectorType>(t);
  if (!v || v.getRank() != 1 || v.getShape()[0] != n)
    return FP8Kind::None;

  Type elemTy = v.getElementType();
  if (isa<Float8E4M3FNUZType>(elemTy))
    return FP8Kind::E4M3FNUZ;
  if (isa<Float8E4M3FNType>(elemTy))
    return FP8Kind::E4M3FN;
  return FP8Kind::None;
}

// Rewrite A: (i8->fp8 cast) + extf(fp8->f16)
//   => bitcast(i8x128 -> i32x32) + llvm.call @convert_fp8_fp16_* +
//   bitcast(i32x64 -> f16x128)
static bool tryRewriteExtF(arith::ExtFOp extf, IRRewriter &rewriter) {
  auto cast = extf.getIn().getDefiningOp<UnrealizedConversionCastOp>();
  if (!cast || cast.getInputs().size() != 1 || cast->getNumResults() != 1)
    return false;

  Value inI8 = cast.getInputs()[0];
  Type castResTy = cast->getResult(0).getType();
  Type outTy = extf.getType();
  FP8Kind fp8Kind = getVecFP8Kind(castResTy, 128);

  if (!isVecI8(inI8.getType(), 128) || fp8Kind == FP8Kind::None ||
      !isVecF16(outTy, 128))
    return false;

  auto module = extf->getParentOfType<ModuleOp>();
  auto vec32i32Ty = VectorType::get({32}, rewriter.getI32Type());
  auto vec64i32Ty = VectorType::get({64}, rewriter.getI32Type());
  StringRef fp8ToFp16Fn = (fp8Kind == FP8Kind::E4M3FN)
                              ? "convert_fp8_fp16_fn"
                              : "convert_fp8_fp16_fnuz";

  rewriter.setInsertionPoint(extf);
  auto inBitcast =
      LLVM::BitcastOp::create(rewriter, extf.getLoc(), vec32i32Ty, inI8);
  auto callee = getOrCreateLLVMFunc(module, rewriter, fp8ToFp16Fn, vec64i32Ty,
                                    {vec32i32Ty});

  auto call =
      LLVM::CallOp::create(rewriter, extf.getLoc(), TypeRange{vec64i32Ty},
                           callee, ValueRange{inBitcast.getResult()});

  rewriter.replaceOpWithNewOp<LLVM::BitcastOp>(extf, outTy, call.getResult());
  return true;
}

// Rewrite B: truncf(f16->fp8) + (fp8->i8 cast)
//   => bitcast(f16x128 -> i32x64) + llvm.call @convert_fp16_fp8_* +
//   bitcast(i32x32 -> i8x128)
static bool tryRewriteTruncCast(UnrealizedConversionCastOp cast,
                                IRRewriter &rewriter) {
  if (cast.getInputs().size() != 1 || cast->getNumResults() != 1)
    return false;

  auto trunc = cast.getInputs()[0].getDefiningOp<arith::TruncFOp>();
  if (!trunc)
    return false;

  Type inTy = trunc.getIn().getType();
  Type truncOutTy = trunc.getType();
  Type castOutTy = cast->getResult(0).getType();
  FP8Kind fp8Kind = getVecFP8Kind(truncOutTy, 128);

  if (!isVecF16(inTy, 128) || fp8Kind == FP8Kind::None ||
      !isVecI8(castOutTy, 128))
    return false;

  auto module = cast->getParentOfType<ModuleOp>();
  auto vec32i32Ty = VectorType::get({32}, rewriter.getI32Type());
  auto vec64i32Ty = VectorType::get({64}, rewriter.getI32Type());
  StringRef fp16ToFp8Fn = (fp8Kind == FP8Kind::E4M3FN)
                              ? "convert_fp16_fp8_fn"
                              : "convert_fp16_fp8_fnuz";

  rewriter.setInsertionPoint(cast);
  auto inBitcast = LLVM::BitcastOp::create(rewriter, cast.getLoc(), vec64i32Ty,
                                           trunc.getIn());
  auto callee = getOrCreateLLVMFunc(module, rewriter, fp16ToFp8Fn, vec32i32Ty,
                                    {vec64i32Ty});

  auto call =
      LLVM::CallOp::create(rewriter, cast.getLoc(), TypeRange{vec32i32Ty},
                           callee, ValueRange{inBitcast.getResult()});

  rewriter.replaceOpWithNewOp<LLVM::BitcastOp>(cast, castOutTy,
                                               call.getResult());
  return true;
}

struct LegalizeFP8CastsToLLVMCallPass
    : public ::impl::LegalizeFP8CastsToLLVMCallBase<
          LegalizeFP8CastsToLLVMCallPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, LLVM::LLVMDialect>();
  }

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    IRRewriter rewriter(&getContext());

    SmallVector<arith::ExtFOp> extfOps;
    moduleOp.walk([&](arith::ExtFOp op) { extfOps.push_back(op); });
    for (arith::ExtFOp op : extfOps) {
      if (op->getBlock())
        (void)tryRewriteExtF(op, rewriter);
    }

    SmallVector<UnrealizedConversionCastOp> castOps;
    moduleOp.walk(
        [&](UnrealizedConversionCastOp op) { castOps.push_back(op); });
    for (UnrealizedConversionCastOp op : castOps) {
      if (op->getBlock())
        (void)tryRewriteTruncCast(op, rewriter);
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
hexagon::createLegalizeFP8CastsToLLVMCallPass() {
  return std::make_unique<LegalizeFP8CastsToLLVMCallPass>();
}
