//===- LegalizeLateFP8CastsToLLVMCallPassRefactor.cpp -------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// This pass rewrites FP8 casts that survive past arith-to-LLVM lowering.
//
// fp8 -> f16 path:
// ```
//   %fp8_vec = builtin.unrealized_conversion_cast %x
//       : vector<128xi8> to vector<128xf8E4M3FNUZ>
//   %f16_vec = arith.extf %fp8_vec
//       : vector<128xf8E4M3FNUZ> to vector<128xf16>
// ```
// is replaced by:
// ```
//   %arg_vec = llvm.bitcast %x : vector<128xi8> to vector<32xi32>
//   %conv_vec = llvm.call @convert_fp8_fp16_fnuz(%arg_vec)
//       : (vector<32xi32>) -> vector<64xi32>
//   %f16_vec = llvm.bitcast %conv_vec : vector<64xi32> to vector<128xf16>
// ```
//
// f16/f32 -> fp8 path:
// ```
//   %fp8_vec = arith.truncf %arg0 fastmath<fast>
//       : vector<128xf32> to vector<128xf8E4M3FNUZ>
//   %i8_vec = builtin.unrealized_conversion_cast %fp8_vec
//       : vector<128xf8E4M3FNUZ> to vector<128xi8>
// ```
// is replaced by:
// ```
//   // Present only when trunc input is f32.
//   %f16_vec = llvm.fptrunc %arg0 : vector<128xf32> to vector<128xf16>
//   %arg_vec = llvm.bitcast %f16_vec : vector<128xf16> to vector<64xi32>
//   %conv_vec = llvm.call @convert_fp16_fp8_fnuz(%arg_vec)
//       : (vector<64xi32>) -> vector<32xi32>
//   %i8_vec = llvm.bitcast %conv_vec : vector<32xi32> to vector<128xi8>
// ```
// ABI notes:
//   * vector path is specialized to 1-D 128-lane payloads.
//   * scalar path is routed through the same vector ABI by splatting to
//     vector<128 x T>, calling the runtime converter, then extracting lane 0.
//   * supported fp8 element types are E4M3FN and E4M3FNUZ.
//
//===----------------------------------------------------------------------===//

#include "hexagon/Transforms/Transforms.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include <iterator>

using namespace mlir;
using namespace hexagon;

#define GEN_PASS_DEF_LEGALIZELATEFP8CASTSTOLLVMCALL
#include "hexagon/Transforms/Passes.h.inc"

namespace {

constexpr int64_t kAbiLanes = 128;
constexpr int64_t kAbiInWords = 32;
constexpr int64_t kAbiOutWords = 64;
#define DEBUG_TYPE "legalize-late-fp8-casts"

enum class FragmentShape { Unsupported, Scalar, Vector128 };

static FlatSymbolRefAttr getOrCreateLLVMFunc(ModuleOp module,
                                             PatternRewriter &rewriter,
                                             StringRef name, Type retTy,
                                             ArrayRef<Type> argTys) {
  if (auto fn = module.lookupSymbol<LLVM::LLVMFuncOp>(name))
    return FlatSymbolRefAttr::get(rewriter.getContext(), fn.getSymName());

  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(module.getBody(),
                             std::prev(module.getBody()->end()));
  auto fnTy = LLVM::LLVMFunctionType::get(retTy, argTys);
  auto fn = LLVM::LLVMFuncOp::create(rewriter, module.getLoc(), name, fnTy);
  fn.setPrivate();
  return FlatSymbolRefAttr::get(rewriter.getContext(), fn.getSymName());
}

template <typename ElemPred>
static bool isVec1DOf(Type t, int64_t n, ElemPred elemPred) {
  auto v = dyn_cast<VectorType>(t);
  return v && v.getRank() == 1 && v.getShape()[0] == n &&
         elemPred(v.getElementType());
}

template <typename ElemPred>
static bool isTypeOfShape(Type t, FragmentShape shape, ElemPred elemPred) {
  switch (shape) {
  case FragmentShape::Scalar:
    return elemPred(t);
  case FragmentShape::Vector128:
    return isVec1DOf(t, kAbiLanes, elemPred);
  case FragmentShape::Unsupported:
    return false;
  }
  return false;
}

static bool isI8(Type t, FragmentShape shape) {
  return isTypeOfShape(t, shape,
                       [](Type elemTy) { return elemTy.isInteger(8); });
}

static bool isF16(Type t, FragmentShape shape) {
  return isTypeOfShape(t, shape, [](Type elemTy) { return elemTy.isF16(); });
}

static bool isF32(Type t, FragmentShape shape) {
  return isTypeOfShape(t, shape, [](Type elemTy) { return elemTy.isF32(); });
}

static bool isFP8E4M3FNUZOrFN(Type t, FragmentShape shape) {
  return isTypeOfShape(t, shape, [](Type elemTy) {
    return isa<Float8E4M3FNUZType, Float8E4M3FNType>(elemTy);
  });
}

static Type getFP8ElementType(Type t) {
  if (auto v = dyn_cast<VectorType>(t))
    return v.getElementType();
  return t;
}

enum class FP8Kind { None, E4M3FNUZ, E4M3FN };
enum class FP8ConvertDirection { Fp8ToFp16, Fp16ToFp8 };

static FP8Kind getFP8Kind(Type fp8Ty) {
  Type elemTy = getFP8ElementType(fp8Ty);
  if (isa<Float8E4M3FNUZType>(elemTy))
    return FP8Kind::E4M3FNUZ;
  if (isa<Float8E4M3FNType>(elemTy))
    return FP8Kind::E4M3FN;
  return FP8Kind::None;
}

static StringRef getConvertFuncName(Type fp8Ty, FP8ConvertDirection direction) {
  switch (getFP8Kind(fp8Ty)) {
  case FP8Kind::E4M3FNUZ:
    return direction == FP8ConvertDirection::Fp8ToFp16
               ? "convert_fp8_fp16_fnuz"
               : "convert_fp16_fp8_fnuz";
  case FP8Kind::E4M3FN:
    return direction == FP8ConvertDirection::Fp8ToFp16 ? "convert_fp8_fp16_fn"
                                                       : "convert_fp16_fp8_fn";
  case FP8Kind::None:
    return {};
  }
  return {};
}

static StringRef getFp8ToFp16ConvertFuncName(Type fp8Ty) {
  return getConvertFuncName(fp8Ty, FP8ConvertDirection::Fp8ToFp16);
}

static StringRef getFp16ToFp8ConvertFuncName(Type fp8Ty) {
  return getConvertFuncName(fp8Ty, FP8ConvertDirection::Fp16ToFp8);
}

static Value splatScalarToVec128(PatternRewriter &rewriter, Location loc,
                                 Value scalar, Type elemTy) {
  auto vec1Ty = VectorType::get({1}, elemTy);
  auto vec128Ty = VectorType::get({kAbiLanes}, elemTy);
  auto scalarAsVec1 =
      LLVM::BitcastOp::create(rewriter, loc, vec1Ty, scalar).getResult();
  auto undefVec1 = LLVM::UndefOp::create(rewriter, loc, vec1Ty).getResult();
  llvm::SmallVector<int32_t, kAbiLanes> splatMask(kAbiLanes, 0);
  return LLVM::ShuffleVectorOp::create(rewriter, loc, vec128Ty, scalarAsVec1,
                                       undefVec1, splatMask)
      .getRes();
}

static Value extractLane0Scalar(PatternRewriter &rewriter, Location loc,
                                Value vec128, Type scalarTy) {
  auto vec128Ty = cast<VectorType>(vec128.getType());
  auto vec1Ty = VectorType::get({1}, vec128Ty.getElementType());
  auto lane0Vec = LLVM::vector_extract::create(rewriter, loc, vec1Ty, vec128, 0)
                      .getResult();
  return LLVM::BitcastOp::create(rewriter, loc, scalarTy, lane0Vec).getResult();
}

template <typename InPred, typename OutPred>
static Value emitABIConversionForShape(PatternRewriter &rewriter, Location loc,
                                       ModuleOp module, Value input, Type outTy,
                                       StringRef convertFuncName,
                                       int64_t abiInWords, int64_t abiOutWords,
                                       Type scalarInputElemTy,
                                       Type scalarOutputElemTy, InPred inPred,
                                       OutPred outPred) {
  auto abiInTy = VectorType::get({abiInWords}, rewriter.getI32Type());
  auto abiOutTy = VectorType::get({abiOutWords}, rewriter.getI32Type());
  auto callee = getOrCreateLLVMFunc(module, rewriter, convertFuncName, abiOutTy,
                                    {abiInTy});

  auto emitVectorCall = [&](Value vec128Payload) {
    auto packedIn =
        LLVM::BitcastOp::create(rewriter, loc, abiInTy, vec128Payload)
            .getResult();
    auto call = LLVM::CallOp::create(rewriter, loc, TypeRange{abiOutTy}, callee,
                                     ValueRange{packedIn});
    auto vec128OutTy = VectorType::get({kAbiLanes}, scalarOutputElemTy);
    return LLVM::BitcastOp::create(rewriter, loc, vec128OutTy, call.getResult())
        .getResult();
  };

  if (inPred(input.getType(), FragmentShape::Vector128) &&
      outPred(outTy, FragmentShape::Vector128)) {
    Value vecOut = emitVectorCall(input);
    if (vecOut.getType() == outTy)
      return vecOut;
    return LLVM::BitcastOp::create(rewriter, loc, outTy, vecOut).getResult();
  }

  if (inPred(input.getType(), FragmentShape::Scalar) &&
      outPred(outTy, FragmentShape::Scalar)) {
    auto vec128Input =
        splatScalarToVec128(rewriter, loc, input, scalarInputElemTy);
    Value vecOut = emitVectorCall(vec128Input);
    return extractLane0Scalar(rewriter, loc, vecOut, outTy);
  }

  return Value();
}

// Convert i8 payload (vector or scalar) to f16 via convert_fp8_fp16_fn/fnuz.
static Value emitConvertFp8ToF16(PatternRewriter &rewriter, Location loc,
                                 ModuleOp module, Value inI8, Type outTy,
                                 StringRef convertFuncName) {
  return emitABIConversionForShape(
      rewriter, loc, module, inI8, outTy, convertFuncName, kAbiInWords,
      kAbiOutWords, rewriter.getI8Type(), rewriter.getF16Type(),
      [](Type ty, FragmentShape shape) { return isI8(ty, shape); },
      [](Type ty, FragmentShape shape) { return isF16(ty, shape); });
}

// Convert i8 payload (vector or scalar) to f32 via convert_fp8_fp16_fn/fnuz.
static Value emitConvertFp8ToF32(PatternRewriter &rewriter, Location loc,
                                 ModuleOp module, Value inI8, Type outTy,
                                 FragmentShape shape,
                                 StringRef convertFuncName) {
  Type f16Ty = shape == FragmentShape::Vector128
                   ? Type(VectorType::get({kAbiLanes}, rewriter.getF16Type()))
                   : Type(rewriter.getF16Type());
  Value f16Val =
      emitConvertFp8ToF16(rewriter, loc, module, inI8, f16Ty, convertFuncName);
  if (!f16Val)
    return Value();
  return LLVM::FPExtOp::create(rewriter, loc, outTy, f16Val).getResult();
}

// Convert f16 payload (vector or scalar) to i8 payload via
// convert_fp16_fp8_fn/fnuz.
static Value emitConvertF16ToFp8I8(PatternRewriter &rewriter, Location loc,
                                   ModuleOp module, Value inF16, Type outTy,
                                   StringRef convertFuncName) {
  return emitABIConversionForShape(
      rewriter, loc, module, inF16, outTy, convertFuncName, kAbiOutWords,
      kAbiInWords, rewriter.getF16Type(), rewriter.getI8Type(),
      [](Type ty, FragmentShape shape) { return isF16(ty, shape); },
      [](Type ty, FragmentShape shape) { return isI8(ty, shape); });
}

// Normalizes trunc input to f16 for a given lane shape.
// For f16 input, accept direct f16 values as-is.
static Value getNormalizedF16Input(PatternRewriter &rewriter, Location loc,
                                   Value truncIn, FragmentShape shape) {
  Type truncInTy = truncIn.getType();
  if (isF16(truncInTy, shape))
    return truncIn;

  if (!isF32(truncInTy, shape))
    return Value();

  Type normalizedF16Ty =
      shape == FragmentShape::Vector128
          ? Type(VectorType::get({kAbiLanes}, rewriter.getF16Type()))
          : rewriter.getF16Type();
  return LLVM::FPTruncOp::create(rewriter, loc, normalizedF16Ty, truncIn)
      .getResult();
}

template <typename InPred, typename MidPred, typename OutPred>
static FragmentShape classifyFragmentByShape(Type inTy, Type midTy, Type outTy,
                                             InPred inPred, MidPred midPred,
                                             OutPred outPred) {
  if (inPred(inTy, FragmentShape::Vector128) &&
      midPred(midTy, FragmentShape::Vector128) &&
      outPred(outTy, FragmentShape::Vector128))
    return FragmentShape::Vector128;

  if (inPred(inTy, FragmentShape::Scalar) &&
      midPred(midTy, FragmentShape::Scalar) &&
      outPred(outTy, FragmentShape::Scalar))
    return FragmentShape::Scalar;

  return FragmentShape::Unsupported;
}

template <typename OutPred>
static FragmentShape classifyFp8ToFloatFragment(Type inTy, Type castResTy,
                                                Type outTy, OutPred outPred) {
  return classifyFragmentByShape(
      inTy, castResTy, outTy,
      [](Type ty, FragmentShape shape) { return isI8(ty, shape); },
      [](Type ty, FragmentShape shape) { return isFP8E4M3FNUZOrFN(ty, shape); },
      outPred);
}

static FragmentShape classifyFp8ToF16Fragment(Type inTy, Type castResTy,
                                              Type outTy) {
  return classifyFp8ToFloatFragment(
      inTy, castResTy, outTy,
      [](Type ty, FragmentShape shape) { return isF16(ty, shape); });
}

static FragmentShape classifyFp8ToF32Fragment(Type inTy, Type castResTy,
                                              Type outTy) {
  return classifyFp8ToFloatFragment(
      inTy, castResTy, outTy,
      [](Type ty, FragmentShape shape) { return isF32(ty, shape); });
}

static FragmentShape classifyF16ToFp8I8Fragment(Type truncInTy, Type truncOutTy,
                                                Type castOutTy) {
  return classifyFragmentByShape(
      truncInTy, truncOutTy, castOutTy,
      [](Type ty, FragmentShape shape) {
        return isF16(ty, shape) || isF32(ty, shape);
      },
      [](Type ty, FragmentShape shape) { return isFP8E4M3FNUZOrFN(ty, shape); },
      [](Type ty, FragmentShape shape) { return isI8(ty, shape); });
}

static void replaceIfValid(PatternRewriter &rewriter, Operation *op,
                           Value replacement) {
  if (replacement)
    rewriter.replaceOp(op, replacement);
}

static void collectCandidateOps(
    ModuleOp moduleOp, llvm::SmallVectorImpl<arith::ExtFOp> &extfOps,
    llvm::SmallVectorImpl<UnrealizedConversionCastOp> &castOps) {
  moduleOp.walk([&](arith::ExtFOp extf) { extfOps.push_back(extf); });
  moduleOp.walk(
      [&](UnrealizedConversionCastOp cast) { castOps.push_back(cast); });
}

static void rewriteFp8ToF16ExtFOp(PatternRewriter &rewriter, ModuleOp moduleOp,
                                  arith::ExtFOp extf) {
  auto cast = extf.getIn().getDefiningOp<UnrealizedConversionCastOp>();
  if (!cast || cast.getInputs().size() != 1 || cast->getNumResults() != 1)
    return;

  Value inI8 = cast.getInputs()[0];
  Type castResTy = cast->getResult(0).getType();
  Type outTy = extf.getType();
  rewriter.setInsertionPoint(extf);

  StringRef convertFuncName = getFp8ToFp16ConvertFuncName(castResTy);
  if (convertFuncName.empty()) {
    LLVM_DEBUG(llvm::dbgs()
               << "Skipping extf fragment with unknown fp8 kind at "
               << extf.getLoc() << "\n");
    return;
  }

  FragmentShape shape =
      classifyFp8ToF16Fragment(inI8.getType(), castResTy, outTy);
  if (shape != FragmentShape::Unsupported) {
    Value replacement = emitConvertFp8ToF16(rewriter, extf.getLoc(), moduleOp,
                                            inI8, outTy, convertFuncName);
    replaceIfValid(rewriter, extf, replacement);
    return;
  }

  shape = classifyFp8ToF32Fragment(inI8.getType(), castResTy, outTy);
  if (shape != FragmentShape::Unsupported) {
    Value replacement = emitConvertFp8ToF32(
        rewriter, extf.getLoc(), moduleOp, inI8, outTy, shape, convertFuncName);
    replaceIfValid(rewriter, extf, replacement);
    return;
  }

  LLVM_DEBUG(llvm::dbgs() << "Skipping unsupported extf fragment at "
                          << extf.getLoc() << "\n");
}

static void rewriteF16ToFp8I8CastOp(PatternRewriter &rewriter,
                                    ModuleOp moduleOp,
                                    UnrealizedConversionCastOp cast) {
  if (cast.getInputs().size() != 1 || cast->getNumResults() != 1)
    return;

  auto trunc = cast.getInputs()[0].getDefiningOp<arith::TruncFOp>();
  if (!trunc)
    return;

  Value truncIn = trunc.getIn();
  Type truncInTy = truncIn.getType();
  Type truncOutTy = trunc.getType();
  Type castOutTy = cast->getResult(0).getType();
  rewriter.setInsertionPoint(cast);

  FragmentShape shape =
      classifyF16ToFp8I8Fragment(truncInTy, truncOutTy, castOutTy);
  if (shape == FragmentShape::Unsupported) {
    LLVM_DEBUG(llvm::dbgs() << "Skipping unsupported trunc/cast fragment at "
                            << cast.getLoc() << "\n");
    return;
  }

  Value f16Input =
      getNormalizedF16Input(rewriter, cast.getLoc(), truncIn, shape);
  if (!f16Input) {
    LLVM_DEBUG(llvm::dbgs()
               << "Skipping fragment with non-f16/f32 trunc input at "
               << cast.getLoc() << "\n");
    return;
  }

  StringRef convertFuncName = getFp16ToFp8ConvertFuncName(truncOutTy);
  if (convertFuncName.empty()) {
    LLVM_DEBUG(llvm::dbgs()
               << "Skipping trunc/cast fragment with unknown fp8 kind at "
               << cast.getLoc() << "\n");
    return;
  }

  Value replacement = emitConvertF16ToFp8I8(
      rewriter, cast.getLoc(), moduleOp, f16Input, castOutTy, convertFuncName);
  replaceIfValid(rewriter, cast, replacement);
}

struct LegalizeLateFP8CastsToLLVMCallPass
    : public ::impl::LegalizeLateFP8CastsToLLVMCallBase<
          LegalizeLateFP8CastsToLLVMCallPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, LLVM::LLVMDialect>();
  }

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    PatternRewriter rewriter(&getContext());
    llvm::SmallVector<arith::ExtFOp> extfOps;
    llvm::SmallVector<UnrealizedConversionCastOp> castOps;

    collectCandidateOps(moduleOp, extfOps, castOps);

    for (arith::ExtFOp extf : extfOps)
      rewriteFp8ToF16ExtFOp(rewriter, moduleOp, extf);

    for (UnrealizedConversionCastOp cast : castOps)
      rewriteF16ToFp8I8CastOp(rewriter, moduleOp, cast);
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
hexagon::createLegalizeLateFP8CastsToLLVMCallPass() {
  return std::make_unique<LegalizeLateFP8CastsToLLVMCallPass>();
}
