//===- LinalgToCompositePass.cpp - Recognize composite ops in linalg IR ---===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// Walks the function looking for composite patterns expressed as DAGs of
// linalg.generic ops and replaces them with single composite ops.
//
// Currently recognized patterns:
//   - softmax: max-reduce -> sub -> exp -> sum-reduce -> reciprocal -> mul
//   - rms_norm: square -> sum-reduce -> scalar(mean, eps, rsqrt) -> fill -> mul
//   - layer_norm: sum(mean) -> sub -> select -> square -> sum(var) -> rsqrt ->
//                 normalize -> mul(weights) -> add(bias)
//
// A matched set ensures ops already claimed by one pattern aren't re-examined.
//
//===----------------------------------------------------------------------===//

#include "hexagon/Conversion/LinalgToLLVM/LinalgToLLVM.h"
#include "hexagon/Dialect/Composite/IR/CompositeDialect.h"
#include "hexagon/Dialect/Composite/IR/CompositeOps.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "linalg-to-composite"
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")

#define GEN_PASS_DEF_LINALGTOCOMPOSITE
#include "hexagon/Conversion/LinalgToLLVM/Passes.h.inc"

using namespace mlir;
using namespace hexagon;

namespace {

/// Return the single block body of a linalg.generic, or nullptr.
static Block *getBodyBlock(linalg::GenericOp op) {
  if (op.getBody()->empty())
    return nullptr;
  return op.getBody();
}

/// Check whether the body of a linalg.generic contains exactly the given
/// sequence of operation types (ignoring the trailing linalg.yield).
template <typename... OpTys> static bool bodyContainsOps(linalg::GenericOp op) {
  Block *body = getBodyBlock(op);
  if (!body)
    return false;
  SmallVector<Operation *> ops;
  for (Operation &o : body->getOperations()) {
    if (!isa<linalg::YieldOp>(o))
      ops.push_back(&o);
  }
  constexpr size_t expected = sizeof...(OpTys);
  if (ops.size() != expected)
    return false;
  size_t idx = 0;
  return ((isa<OpTys>(ops[idx++])) && ...);
}

/// Return true if the generic is a reduction along the innermost dim
/// (last iterator type is "reduction", all others "parallel").
static bool isRowReduction(linalg::GenericOp op) {
  auto iterTypes = op.getIteratorTypesArray();
  if (iterTypes.empty())
    return false;
  if (iterTypes.back() != utils::IteratorType::reduction)
    return false;
  for (size_t i = 0; i + 1 < iterTypes.size(); ++i) {
    if (iterTypes[i] != utils::IteratorType::parallel)
      return false;
  }
  return true;
}

/// Return true if all iterator types are parallel.
static bool isAllParallel(linalg::GenericOp op) {
  for (auto it : op.getIteratorTypesArray()) {
    if (it != utils::IteratorType::parallel)
      return false;
  }
  return true;
}

/// Return the linalg.generic that defines the given value, or nullptr.
static linalg::GenericOp getDefiningGeneric(Value v) {
  if (!v)
    return nullptr;
  return dyn_cast_or_null<linalg::GenericOp>(v.getDefiningOp());
}

/// Return true if the generic is a pure broadcast (body just yields the input).
static bool isBroadcast(linalg::GenericOp op) {
  Block *body = getBodyBlock(op);
  if (!body)
    return false;
  auto &ops = body->getOperations();
  if (ops.size() != 1)
    return false;
  auto yieldOp = dyn_cast<linalg::YieldOp>(&ops.front());
  if (!yieldOp)
    return false;
  return isa<BlockArgument>(yieldOp.getOperand(0));
}

/// Holds all ops in a recognized softmax pattern.
struct SoftmaxPattern {
  linalg::GenericOp maxReduce;
  linalg::GenericOp broadcastMax;
  linalg::GenericOp sub;
  linalg::GenericOp exp;
  linalg::GenericOp sumReduce;
  linalg::GenericOp reciprocal;
  linalg::GenericOp broadcastRcp;
  linalg::GenericOp mul;
  Value input;
};

/// Match the softmax DAG starting from exp:
///
/// input -> max -> [bcast] -> sub -> exp -> sum -> 1/sum -> [bcast] -> mul
///                          ^           |                             ^
///                          +-- input   +-----------------------------+
///
static bool matchSoftmax(linalg::GenericOp expGeneric, SoftmaxPattern &pat) {
  // exp
  if (!bodyContainsOps<math::ExpOp>(expGeneric))
    return false;
  if (!isAllParallel(expGeneric))
    return false;

  // sub (backward from exp)
  if (expGeneric.getNumDpsInputs() != 1)
    return false;
  Value subResult = expGeneric.getDpsInputOperand(0)->get();
  auto subGeneric = getDefiningGeneric(subResult);
  if (!subGeneric)
    return false;
  if (!bodyContainsOps<arith::SubFOp>(subGeneric))
    return false;
  if (!isAllParallel(subGeneric))
    return false;

  // max-reduce (backward from sub, through optional broadcast)
  if (subGeneric.getNumDpsInputs() != 2)
    return false;
  Value input = subGeneric.getDpsInputOperand(0)->get();
  Value maxOrBroadcast = subGeneric.getDpsInputOperand(1)->get();

  linalg::GenericOp broadcastMax;
  Value maxResult = maxOrBroadcast;
  if (auto bcast = getDefiningGeneric(maxOrBroadcast)) {
    if (isBroadcast(bcast)) {
      broadcastMax = bcast;
      if (bcast.getNumDpsInputs() != 1)
        return false;
      maxResult = bcast.getDpsInputOperand(0)->get();
    }
  }

  auto maxReduce = getDefiningGeneric(maxResult);
  if (!maxReduce)
    return false;
  if (!bodyContainsOps<arith::MaxNumFOp>(maxReduce))
    return false;
  if (!isRowReduction(maxReduce))
    return false;

  // Verify max-reduce and sub share the same input.
  if (maxReduce.getNumDpsInputs() != 1)
    return false;
  if (maxReduce.getDpsInputOperand(0)->get() != input)
    return false;

  // sum-reduce and final mul (forward from exp)
  Value expResult = expGeneric.getResult(0);

  linalg::GenericOp sumReduce;
  linalg::GenericOp finalMul;

  for (Operation *user : expResult.getUsers()) {
    auto generic = dyn_cast<linalg::GenericOp>(user);
    if (!generic)
      continue;
    if (bodyContainsOps<arith::AddFOp>(generic) && isRowReduction(generic))
      sumReduce = generic;
    else if (bodyContainsOps<arith::MulFOp>(generic) && isAllParallel(generic))
      finalMul = generic;
  }

  if (!sumReduce || !finalMul)
    return false;

  // reciprocal (forward from sum-reduce)
  Value sumResult = sumReduce.getResult(0);
  linalg::GenericOp reciprocal;
  for (Operation *user : sumResult.getUsers()) {
    auto generic = dyn_cast<linalg::GenericOp>(user);
    if (!generic)
      continue;
    if (bodyContainsOps<arith::DivFOp>(generic) && isAllParallel(generic))
      reciprocal = generic;
  }
  if (!reciprocal)
    return false;

  // broadcast (optional, forward from reciprocal)
  Value rcpResult = reciprocal.getResult(0);
  linalg::GenericOp broadcastRcp;

  for (Operation *user : rcpResult.getUsers()) {
    auto generic = dyn_cast<linalg::GenericOp>(user);
    if (!generic)
      continue;
    if (isBroadcast(generic)) {
      broadcastRcp = generic;
      rcpResult = generic.getResult(0);
      break;
    }
  }

  // final mul: verify it takes exp result and (broadcasted) reciprocal.
  if (finalMul.getNumDpsInputs() != 2)
    return false;
  Value mulIn0 = finalMul.getDpsInputOperand(0)->get();
  Value mulIn1 = finalMul.getDpsInputOperand(1)->get();
  bool validInputs = (mulIn0 == expResult && mulIn1 == rcpResult) ||
                     (mulIn1 == expResult && mulIn0 == rcpResult);
  if (!validInputs)
    return false;

  LLVM_DEBUG(DBGS() << "  Full softmax pattern matched!\n");

  pat.maxReduce = maxReduce;
  pat.broadcastMax = broadcastMax;
  pat.sub = subGeneric;
  pat.exp = expGeneric;
  pat.sumReduce = sumReduce;
  pat.reciprocal = reciprocal;
  pat.broadcastRcp = broadcastRcp;
  pat.mul = finalMul;
  pat.input = input;
  return true;
}

/// Return true if all iterator types are reduction (full reduction to scalar).
static bool isFullReduction(linalg::GenericOp op) {
  for (auto it : op.getIteratorTypesArray()) {
    if (it != utils::IteratorType::reduction)
      return false;
  }
  return true;
}

/// Return true if the generic is a self-multiply (both inputs are the same
/// value), i.e. x * x.
static bool isSelfMultiply(linalg::GenericOp op) {
  if (!bodyContainsOps<arith::MulFOp>(op))
    return false;
  if (!isAllParallel(op))
    return false;
  if (op.getNumDpsInputs() != 2)
    return false;
  return op.getDpsInputOperand(0)->get() == op.getDpsInputOperand(1)->get();
}

/// Holds all ops in a recognized RMS norm pattern.
struct RmsNormPattern {
  linalg::GenericOp square;
  linalg::GenericOp sumReduce;
  tensor::ExtractOp extract;
  arith::DivFOp divByN;
  arith::AddFOp addEps;
  math::SqrtOp sqrt;
  arith::DivFOp rsqrt;
  linalg::FillOp broadcastFill;
  linalg::GenericOp normalize;
  linalg::GenericOp weightMul;
  Value input;
  Value weights;
  float epsilon;
  float normSize;
};

/// Match the RMS norm DAG starting from a full-reduction sum:
///
/// input -> square(x*x) -> sum-reduce -> extract -> divf(N) -> addf(eps) ->
///   sqrt -> divf(1,_) -> fill(broadcast) -> normalize(x*rsqrt) -> mul(weights)
///
static bool matchRmsNorm(linalg::GenericOp sumGeneric, RmsNormPattern &pat) {
  // sumReduce: full reduction with addf, output must be 0-d (scalar).
  if (!bodyContainsOps<arith::AddFOp>(sumGeneric))
    return false;
  if (!isFullReduction(sumGeneric))
    return false;
  auto outTy = dyn_cast<RankedTensorType>(sumGeneric.getResult(0).getType());
  if (!outTy || outTy.getRank() != 0)
    return false;

  // square (backward from sumReduce): single input, must be x*x.
  if (sumGeneric.getNumDpsInputs() != 1)
    return false;
  auto squareGeneric =
      getDefiningGeneric(sumGeneric.getDpsInputOperand(0)->get());
  if (!squareGeneric || !isSelfMultiply(squareGeneric))
    return false;

  Value input = squareGeneric.getDpsInputOperand(0)->get();

  // Scalar chain (forward from sumReduce):
  //   extract -> divf(N) -> addf(eps) -> sqrt -> divf(1.0, _)
  Value sumResult = sumGeneric.getResult(0);
  if (!sumResult.hasOneUse())
    return false;

  auto extractOp = dyn_cast<tensor::ExtractOp>(*sumResult.getUsers().begin());
  if (!extractOp)
    return false;

  Value extracted = extractOp.getResult();
  if (!extracted.hasOneUse())
    return false;
  auto divByN = dyn_cast<arith::DivFOp>(*extracted.getUsers().begin());
  if (!divByN || divByN.getLhs() != extracted)
    return false;

  // The divisor (N) must be a constant — this is the actual element count
  // which may differ from the tensor dimension when padded.
  float normSize = 0.0f;
  if (auto cst = divByN.getRhs().getDefiningOp<arith::ConstantFloatOp>()) {
    normSize = cst.value().convertToFloat();
  } else {
    return false;
  }

  Value mean = divByN.getResult();
  if (!mean.hasOneUse())
    return false;
  auto addEps = dyn_cast<arith::AddFOp>(*mean.getUsers().begin());
  if (!addEps)
    return false;

  // Epsilon can be on either side of the addf.
  float epsilon = 0.0f;
  if (auto cst = addEps.getRhs().getDefiningOp<arith::ConstantFloatOp>()) {
    epsilon = cst.value().convertToFloat();
  } else if (auto cst =
                 addEps.getLhs().getDefiningOp<arith::ConstantFloatOp>()) {
    epsilon = cst.value().convertToFloat();
  } else {
    return false;
  }

  Value shifted = addEps.getResult();
  if (!shifted.hasOneUse())
    return false;
  auto sqrtOp = dyn_cast<math::SqrtOp>(*shifted.getUsers().begin());
  if (!sqrtOp)
    return false;

  Value sqrtResult = sqrtOp.getResult();
  if (!sqrtResult.hasOneUse())
    return false;
  auto rsqrtDiv = dyn_cast<arith::DivFOp>(*sqrtResult.getUsers().begin());
  if (!rsqrtDiv)
    return false;
  if (auto cst = rsqrtDiv.getLhs().getDefiningOp<arith::ConstantFloatOp>()) {
    if (!cst.value().isExactlyValue(1.0))
      return false;
  } else {
    return false;
  }

  // Broadcast: linalg.fill of rsqrt scalar into a tensor.
  Value rsqrtScalar = rsqrtDiv.getResult();
  if (!rsqrtScalar.hasOneUse())
    return false;
  auto fillOp = dyn_cast<linalg::FillOp>(*rsqrtScalar.getUsers().begin());
  if (!fillOp)
    return false;

  // Normalize: mulf(x, rsqrt_broadcast).
  Value fillResult = fillOp.getResult(0);
  if (!fillResult.hasOneUse())
    return false;
  auto normalizeOp =
      dyn_cast<linalg::GenericOp>(*fillResult.getUsers().begin());
  if (!normalizeOp)
    return false;
  if (!bodyContainsOps<arith::MulFOp>(normalizeOp))
    return false;
  if (!isAllParallel(normalizeOp))
    return false;
  if (normalizeOp.getNumDpsInputs() != 2)
    return false;

  Value normIn0 = normalizeOp.getDpsInputOperand(0)->get();
  Value normIn1 = normalizeOp.getDpsInputOperand(1)->get();
  bool normValid = (normIn0 == input && normIn1 == fillResult) ||
                   (normIn1 == input && normIn0 == fillResult);
  if (!normValid)
    return false;

  // Weight multiply: mulf(normalized, weights).
  Value normResult = normalizeOp.getResult(0);
  if (!normResult.hasOneUse())
    return false;
  auto weightMulOp =
      dyn_cast<linalg::GenericOp>(*normResult.getUsers().begin());
  if (!weightMulOp)
    return false;
  if (!bodyContainsOps<arith::MulFOp>(weightMulOp))
    return false;
  if (!isAllParallel(weightMulOp))
    return false;
  if (weightMulOp.getNumDpsInputs() != 2)
    return false;

  Value wmIn0 = weightMulOp.getDpsInputOperand(0)->get();
  Value wmIn1 = weightMulOp.getDpsInputOperand(1)->get();
  Value weights;
  if (wmIn0 == normResult)
    weights = wmIn1;
  else if (wmIn1 == normResult)
    weights = wmIn0;
  else
    return false;

  LLVM_DEBUG(DBGS() << "  Full RMS norm pattern matched!\n");

  pat.square = squareGeneric;
  pat.sumReduce = sumGeneric;
  pat.extract = extractOp;
  pat.divByN = divByN;
  pat.addEps = addEps;
  pat.sqrt = sqrtOp;
  pat.rsqrt = rsqrtDiv;
  pat.broadcastFill = fillOp;
  pat.normalize = normalizeOp;
  pat.weightMul = weightMulOp;
  pat.input = input;
  pat.weights = weights;
  pat.epsilon = epsilon;
  pat.normSize = normSize;
  return true;
}

/// Holds all ops in a recognized layer norm pattern.
struct LayerNormPattern {
  // Mean path.
  linalg::GenericOp meanSumReduce;
  tensor::ExtractOp meanExtract;
  arith::DivFOp meanDivByN;
  linalg::FillOp meanBroadcastFill;
  linalg::GenericOp subGeneric;

  // Select (tl.where for padding, may be null).
  linalg::GenericOp selectGeneric;

  // Variance path.
  linalg::GenericOp squareGeneric;
  linalg::GenericOp varSumReduce;
  tensor::ExtractOp varExtract;
  arith::DivFOp varDivByN;
  arith::AddFOp addEps;
  math::SqrtOp sqrtOp;
  arith::DivFOp rsqrtDiv;

  // Normalize path.
  linalg::FillOp rstdBroadcastFill;
  linalg::GenericOp normalizeGeneric;
  linalg::GenericOp weightMulGeneric;
  linalg::GenericOp biasAddGeneric;

  // Side outputs (optional affine.store for mean/rstd).
  Operation *meanStore = nullptr;
  Operation *rstdStore = nullptr;
  Value meanOutMemref;
  Value rstdOutMemref;

  // Values.
  Value input;
  Value weights;
  Value bias;
  float epsilon = 0.0f;
  float normSize = 0.0f;
};

/// Match the layer norm DAG starting from the mean sum-reduce:
///
/// input -> sum(mean) -> /N -> fill -> sub(input, mean) -> [select] ->
///   square -> sum(var) -> /N -> +eps -> sqrt -> 1/sqrt -> fill ->
///   normalize -> mul(weights) -> add(bias)
///
/// Side outputs: affine.store of mean and rstd scalars (optional).
///
static bool matchLayerNorm(linalg::GenericOp meanSumGeneric,
                           LayerNormPattern &pat) {
  // Mean sum-reduce: full reduction with addf, 0-d output.
  if (!bodyContainsOps<arith::AddFOp>(meanSumGeneric))
    return false;
  if (!isFullReduction(meanSumGeneric))
    return false;
  auto outTy =
      dyn_cast<RankedTensorType>(meanSumGeneric.getResult(0).getType());
  if (!outTy || outTy.getRank() != 0)
    return false;

  // Input must NOT be a self-multiply (discriminates from rms_norm).
  if (meanSumGeneric.getNumDpsInputs() != 1)
    return false;
  Value input = meanSumGeneric.getDpsInputOperand(0)->get();
  auto inputGeneric = getDefiningGeneric(input);
  if (inputGeneric && isSelfMultiply(inputGeneric)) {
    return false;
  }

  // Forward: extract -> divf(N) -> mean scalar.
  Value sumResult = meanSumGeneric.getResult(0);
  if (!sumResult.hasOneUse()) {
    return false;
  }
  auto meanExtract = dyn_cast<tensor::ExtractOp>(*sumResult.getUsers().begin());
  if (!meanExtract) {
    return false;
  }

  Value extracted = meanExtract.getResult();
  if (!extracted.hasOneUse()) {
    return false;
  }
  auto meanDivByN = dyn_cast<arith::DivFOp>(*extracted.getUsers().begin());
  if (!meanDivByN || meanDivByN.getLhs() != extracted) {
    return false;
  }

  float normSize = 0.0f;
  if (auto cst = meanDivByN.getRhs().getDefiningOp<arith::ConstantFloatOp>())
    normSize = cst.value().convertToFloat();
  else {
    return false;
  }

  // Mean scalar users: fill (broadcast) and optionally affine.store.
  Value meanScalar = meanDivByN.getResult();
  linalg::FillOp meanFill;
  Operation *meanStoreOp = nullptr;
  Value meanOutMemref;

  for (Operation *user : meanScalar.getUsers()) {
    if (auto fill = dyn_cast<linalg::FillOp>(user)) {
      if (meanFill && meanFill != fill)
        return false;
      meanFill = fill;
    } else if (isa<affine::AffineStoreOp>(user)) {
      if (meanStoreOp && meanStoreOp != user)
        return false;
      meanStoreOp = user;
      meanOutMemref = cast<affine::AffineStoreOp>(user).getMemRef();
    } else {
      return false;
    }
  }
  if (!meanFill)
    return false;

  // From fill: sub(input, mean_broadcast) -> centered.
  Value meanBcast = meanFill.getResult(0);
  if (!meanBcast.hasOneUse())
    return false;
  auto subGeneric = dyn_cast<linalg::GenericOp>(*meanBcast.getUsers().begin());
  if (!subGeneric)
    return false;
  if (!bodyContainsOps<arith::SubFOp>(subGeneric))
    return false;
  if (!isAllParallel(subGeneric))
    return false;
  if (subGeneric.getNumDpsInputs() != 2)
    return false;

  Value subIn0 = subGeneric.getDpsInputOperand(0)->get();
  Value subIn1 = subGeneric.getDpsInputOperand(1)->get();
  if (!(subIn0 == input && subIn1 == meanBcast) &&
      !(subIn1 == input && subIn0 == meanBcast)) {
    return false;
  }

  // Optional select(mask, centered, zeros) from tl.where for padding.
  Value centeredValue = subGeneric.getResult(0);
  linalg::GenericOp selectGeneric;

  for (Operation *user : centeredValue.getUsers()) {
    auto generic = dyn_cast<linalg::GenericOp>(user);
    if (!generic)
      continue;
    if (bodyContainsOps<arith::SelectOp>(generic) && isAllParallel(generic)) {
      selectGeneric = generic;
      break;
    }
  }

  if (selectGeneric)
    centeredValue = selectGeneric.getResult(0);

  // centeredValue users: self-multiply (square) and mulf (normalize).
  linalg::GenericOp squareGeneric;
  linalg::GenericOp normalizeGeneric;

  for (Operation *user : centeredValue.getUsers()) {
    auto generic = dyn_cast<linalg::GenericOp>(user);
    if (!generic)
      continue;
    if (!bodyContainsOps<arith::MulFOp>(generic) || !isAllParallel(generic))
      continue;
    if (generic.getNumDpsInputs() != 2)
      continue;
    if (isSelfMultiply(generic)) {
      if (squareGeneric && squareGeneric != generic)
        return false;
      squareGeneric = generic;
    } else {
      if (normalizeGeneric && normalizeGeneric != generic)
        return false;
      normalizeGeneric = generic;
    }
  }

  if (!squareGeneric || !normalizeGeneric)
    return false;

  // From square: var sum-reduce -> extract -> divf(N) -> addf(eps) ->
  //   sqrt -> divf(1,_) -> rstd scalar.
  Value squareResult = squareGeneric.getResult(0);
  if (!squareResult.hasOneUse())
    return false;
  auto varSumGeneric =
      dyn_cast<linalg::GenericOp>(*squareResult.getUsers().begin());
  if (!varSumGeneric) {
    return false;
  }
  if (!bodyContainsOps<arith::AddFOp>(varSumGeneric)) {
    return false;
  }
  if (!isFullReduction(varSumGeneric)) {
    return false;
  }
  auto varOutTy =
      dyn_cast<RankedTensorType>(varSumGeneric.getResult(0).getType());
  if (!varOutTy || varOutTy.getRank() != 0) {
    return false;
  }

  Value varSumResult = varSumGeneric.getResult(0);
  if (!varSumResult.hasOneUse()) {
    return false;
  }
  auto varExtract =
      dyn_cast<tensor::ExtractOp>(*varSumResult.getUsers().begin());
  if (!varExtract) {
    return false;
  }

  Value varExtracted = varExtract.getResult();
  if (!varExtracted.hasOneUse()) {
    return false;
  }
  auto varDivByN = dyn_cast<arith::DivFOp>(*varExtracted.getUsers().begin());
  if (!varDivByN || varDivByN.getLhs() != varExtracted) {
    return false;
  }

  Value variance = varDivByN.getResult();
  if (!variance.hasOneUse()) {
    return false;
  }
  auto addEps = dyn_cast<arith::AddFOp>(*variance.getUsers().begin());
  if (!addEps) {
    return false;
  }

  float epsilon = 0.0f;
  if (auto cst = addEps.getRhs().getDefiningOp<arith::ConstantFloatOp>())
    epsilon = cst.value().convertToFloat();
  else if (auto cst = addEps.getLhs().getDefiningOp<arith::ConstantFloatOp>())
    epsilon = cst.value().convertToFloat();
  else
    return false;

  Value shifted = addEps.getResult();
  if (!shifted.hasOneUse())
    return false;
  auto sqrtOp = dyn_cast<math::SqrtOp>(*shifted.getUsers().begin());
  if (!sqrtOp)
    return false;

  Value sqrtResult = sqrtOp.getResult();
  if (!sqrtResult.hasOneUse())
    return false;
  auto rsqrtDiv = dyn_cast<arith::DivFOp>(*sqrtResult.getUsers().begin());
  if (!rsqrtDiv)
    return false;
  if (auto cst = rsqrtDiv.getLhs().getDefiningOp<arith::ConstantFloatOp>()) {
    if (!cst.value().isExactlyValue(1.0))
      return false;
  } else {
    return false;
  }

  // Rstd scalar users: fill (broadcast) and optionally affine.store.
  Value rstdScalar = rsqrtDiv.getResult();
  linalg::FillOp rstdFill;
  Operation *rstdStoreOp = nullptr;
  Value rstdOutMemref;

  for (Operation *user : rstdScalar.getUsers()) {
    if (auto fill = dyn_cast<linalg::FillOp>(user)) {
      if (rstdFill && rstdFill != fill)
        return false;
      rstdFill = fill;
    } else if (isa<affine::AffineStoreOp>(user)) {
      if (rstdStoreOp && rstdStoreOp != user)
        return false;
      rstdStoreOp = user;
      rstdOutMemref = cast<affine::AffineStoreOp>(user).getMemRef();
    } else {
      return false;
    }
  }
  if (!rstdFill)
    return false;

  // Verify normalize: mulf(centeredValue, rstd_broadcast).
  Value rstdBcast = rstdFill.getResult(0);
  Value normIn0 = normalizeGeneric.getDpsInputOperand(0)->get();
  Value normIn1 = normalizeGeneric.getDpsInputOperand(1)->get();
  if (!(normIn0 == centeredValue && normIn1 == rstdBcast) &&
      !(normIn1 == centeredValue && normIn0 == rstdBcast)) {
    return false;
  }

  // From normalize: weightMul (mulf with weights).
  // The normalize result may be used as both input and DPS output init of the
  // next generic, so find the single unique user op (skipping self).
  Value normResult = normalizeGeneric.getResult(0);
  linalg::GenericOp weightMulGeneric;
  for (Operation *user : normResult.getUsers()) {
    auto generic = dyn_cast<linalg::GenericOp>(user);
    if (!generic || generic == normalizeGeneric)
      continue;
    if (weightMulGeneric && weightMulGeneric != generic)
      return false;
    weightMulGeneric = generic;
  }
  if (!weightMulGeneric) {
    return false;
  }
  if (!bodyContainsOps<arith::MulFOp>(weightMulGeneric)) {
    return false;
  }
  if (!isAllParallel(weightMulGeneric)) {
    return false;
  }
  if (weightMulGeneric.getNumDpsInputs() != 2)
    return false;

  Value wmIn0 = weightMulGeneric.getDpsInputOperand(0)->get();
  Value wmIn1 = weightMulGeneric.getDpsInputOperand(1)->get();
  Value weights;
  if (wmIn0 == normResult)
    weights = wmIn1;
  else if (wmIn1 == normResult)
    weights = wmIn0;
  else {
    return false;
  }

  // From weightMul: biasAdd (addf with bias).
  Value wmResult = weightMulGeneric.getResult(0);
  linalg::GenericOp biasAddGeneric;
  for (Operation *user : wmResult.getUsers()) {
    auto generic = dyn_cast<linalg::GenericOp>(user);
    if (!generic || generic == weightMulGeneric)
      continue;
    if (biasAddGeneric && biasAddGeneric != generic)
      return false;
    biasAddGeneric = generic;
  }
  if (!biasAddGeneric) {
    return false;
  }
  if (!bodyContainsOps<arith::AddFOp>(biasAddGeneric)) {
    return false;
  }
  if (!isAllParallel(biasAddGeneric)) {
    return false;
  }
  if (biasAddGeneric.getNumDpsInputs() != 2)
    return false;

  Value baIn0 = biasAddGeneric.getDpsInputOperand(0)->get();
  Value baIn1 = biasAddGeneric.getDpsInputOperand(1)->get();
  Value bias;
  if (baIn0 == wmResult)
    bias = baIn1;
  else if (baIn1 == wmResult)
    bias = baIn0;
  else {
    return false;
  }

  LLVM_DEBUG(DBGS() << "  Full layer norm pattern matched!\n");

  pat.meanSumReduce = meanSumGeneric;
  pat.meanExtract = meanExtract;
  pat.meanDivByN = meanDivByN;
  pat.meanBroadcastFill = meanFill;
  pat.subGeneric = subGeneric;
  pat.selectGeneric = selectGeneric;
  pat.squareGeneric = squareGeneric;
  pat.varSumReduce = varSumGeneric;
  pat.varExtract = varExtract;
  pat.varDivByN = varDivByN;
  pat.addEps = addEps;
  pat.sqrtOp = sqrtOp;
  pat.rsqrtDiv = rsqrtDiv;
  pat.rstdBroadcastFill = rstdFill;
  pat.normalizeGeneric = normalizeGeneric;
  pat.weightMulGeneric = weightMulGeneric;
  pat.biasAddGeneric = biasAddGeneric;
  pat.meanStore = meanStoreOp;
  pat.rstdStore = rstdStoreOp;
  pat.meanOutMemref = meanOutMemref;
  pat.rstdOutMemref = rstdOutMemref;
  pat.input = input;
  pat.weights = weights;
  pat.bias = bias;
  pat.epsilon = epsilon;
  pat.normSize = normSize;
  return true;
}

/// Collect ops to erase for a matched LayerNorm pattern.
static void collectOpsToErase(const LayerNormPattern &pat,
                              SmallVectorImpl<Operation *> &toErase) {
  auto addIfValid = [&](Operation *op) {
    if (op)
      toErase.push_back(op);
  };
  // Reverse topological order.
  addIfValid(pat.biasAddGeneric);
  addIfValid(pat.weightMulGeneric);
  addIfValid(pat.normalizeGeneric);
  addIfValid(pat.rstdBroadcastFill);
  addIfValid(pat.rsqrtDiv);
  addIfValid(pat.sqrtOp);
  addIfValid(pat.addEps);
  addIfValid(pat.varDivByN);
  addIfValid(pat.varExtract);
  addIfValid(pat.varSumReduce);
  addIfValid(pat.squareGeneric);
  addIfValid(pat.selectGeneric);
  addIfValid(pat.subGeneric);
  addIfValid(pat.meanBroadcastFill);
  addIfValid(pat.meanDivByN);
  addIfValid(pat.meanExtract);
  addIfValid(pat.meanSumReduce);

  // Side output stores.
  addIfValid(pat.meanStore);
  addIfValid(pat.rstdStore);

  // tensor.insert ops feeding reduction inits.
  auto eraseInsertIfDead = [&](linalg::GenericOp reduceOp) {
    for (OpOperand &init : reduceOp.getDpsInitsMutable()) {
      if (auto insertOp = init.get().getDefiningOp<tensor::InsertOp>()) {
        if (insertOp->hasOneUse())
          toErase.push_back(insertOp);
      }
    }
  };
  eraseInsertIfDead(pat.meanSumReduce);
  eraseInsertIfDead(pat.varSumReduce);
}

/// Holds all ops in a recognized 2D RMS norm pattern (from block-pointer
/// kernels).  The scalar chain (divByN, addEps, sqrt, rsqrt) becomes a
/// sequence of 1D linalg.generic ops, and the sum-reduce is a partial
/// (row-wise) reduction rather than a full reduction to scalar.
struct RmsNormPattern2D {
  linalg::GenericOp square;
  linalg::GenericOp sumReduce;
  linalg::GenericOp divByN;    // 1D generic: mulf(sum, 1/N)
  linalg::GenericOp addEps;    // 1D generic: addf(mean, eps)
  linalg::GenericOp sqrtOp;    // 1D generic: math.sqrt
  linalg::GenericOp rsqrtOp;   // 1D generic: divf(1.0, sqrt)
  linalg::GenericOp broadcast; // 1D->2D broadcast of rsqrt
  linalg::GenericOp normalize;
  linalg::GenericOp weightMul;
  Value input;
  Value weights;
  float epsilon;
  float normSize;
};

/// Match the 2D RMS norm DAG starting from a row-wise sum-reduce:
///
/// input -> square(x*x) -> row-sum -> generic(mean) -> generic(+eps) ->
///   generic(sqrt) -> generic(1/sqrt) -> broadcast -> normalize(x*rsqrt) ->
///   mul(weights)
///
/// Unlike the 1D (scalar-chain) variant, here the reduction output is a
/// 1D tensor and the intermediate ops are 1D linalg.generics.
static bool matchRmsNorm2D(linalg::GenericOp sumGeneric,
                           RmsNormPattern2D &pat) {
  // sumReduce: row reduction with addf.  Output is rank >= 1 (not scalar).
  if (!bodyContainsOps<arith::AddFOp>(sumGeneric))
    return false;
  if (!isRowReduction(sumGeneric))
    return false;
  auto outTy = dyn_cast<RankedTensorType>(sumGeneric.getResult(0).getType());
  if (!outTy || outTy.getRank() == 0)
    return false;

  // square (backward from sumReduce): single input, must be x*x.
  if (sumGeneric.getNumDpsInputs() != 1)
    return false;
  auto squareGeneric =
      getDefiningGeneric(sumGeneric.getDpsInputOperand(0)->get());
  if (!squareGeneric || !isSelfMultiply(squareGeneric))
    return false;

  Value input = squareGeneric.getDpsInputOperand(0)->get();

  // Forward chain from sumReduce: all 1D generics.
  //   generic(mulf by 1/N) -> generic(addf eps) -> generic(sqrt) ->
  //   generic(divf 1.0) -> broadcast -> normalize -> weightMul

  // divByN: 1D generic with mulf, single use of sumReduce result.
  Value sumResult = sumGeneric.getResult(0);
  if (!sumResult.hasOneUse())
    return false;
  auto divByNOp = dyn_cast<linalg::GenericOp>(*sumResult.getUsers().begin());
  if (!divByNOp || !bodyContainsOps<arith::MulFOp>(divByNOp))
    return false;
  if (!isAllParallel(divByNOp) || divByNOp.getNumDpsInputs() != 2)
    return false;

  // Extract normSize from the reciprocal constant (1/N).
  float normSize = 0.0f;
  for (int i = 0; i < 2; ++i) {
    Value operand = divByNOp.getDpsInputOperand(i)->get();
    if (auto fillOp = operand.getDefiningOp<linalg::FillOp>()) {
      if (auto cst =
              fillOp.getInputs()[0].getDefiningOp<arith::ConstantFloatOp>()) {
        float val = cst.value().convertToFloat();
        if (val > 0.0f && val < 1.0f) {
          normSize = 1.0f / val;
          break;
        }
      }
    }
  }
  if (normSize == 0.0f)
    return false;

  // addEps: 1D generic with addf.
  Value divResult = divByNOp.getResult(0);
  if (!divResult.hasOneUse())
    return false;
  auto addEpsOp = dyn_cast<linalg::GenericOp>(*divResult.getUsers().begin());
  if (!addEpsOp || !bodyContainsOps<arith::AddFOp>(addEpsOp))
    return false;
  if (!isAllParallel(addEpsOp) || addEpsOp.getNumDpsInputs() != 2)
    return false;

  // Extract epsilon from the constant fill input.
  float epsilon = 0.0f;
  for (int i = 0; i < 2; ++i) {
    Value operand = addEpsOp.getDpsInputOperand(i)->get();
    if (auto fillOp = operand.getDefiningOp<linalg::FillOp>()) {
      if (auto cst =
              fillOp.getInputs()[0].getDefiningOp<arith::ConstantFloatOp>()) {
        epsilon = cst.value().convertToFloat();
        break;
      }
    }
  }
  if (epsilon == 0.0f)
    return false;

  // sqrt: 1D generic with math.sqrt.
  Value epsResult = addEpsOp.getResult(0);
  if (!epsResult.hasOneUse())
    return false;
  auto sqrtGeneric = dyn_cast<linalg::GenericOp>(*epsResult.getUsers().begin());
  if (!sqrtGeneric || !bodyContainsOps<math::SqrtOp>(sqrtGeneric))
    return false;
  if (!isAllParallel(sqrtGeneric))
    return false;

  // rsqrt: 1D generic with divf(1.0, sqrt).
  Value sqrtResult = sqrtGeneric.getResult(0);
  if (!sqrtResult.hasOneUse())
    return false;
  auto rsqrtGeneric =
      dyn_cast<linalg::GenericOp>(*sqrtResult.getUsers().begin());
  if (!rsqrtGeneric || !bodyContainsOps<arith::DivFOp>(rsqrtGeneric))
    return false;
  if (!isAllParallel(rsqrtGeneric))
    return false;

  // Verify the divf is 1.0 / sqrt (not sqrt / something).
  Block *rsqrtBody = getBodyBlock(rsqrtGeneric);
  if (!rsqrtBody)
    return false;
  auto rsqrtDivOp = dyn_cast<arith::DivFOp>(&rsqrtBody->front());
  if (!rsqrtDivOp)
    return false;
  // Simpler check: just verify one input is sqrtResult.
  bool hasSqrtInput = false;
  for (int i = 0; i < (int)rsqrtGeneric.getNumDpsInputs(); ++i) {
    if (rsqrtGeneric.getDpsInputOperand(i)->get() == sqrtResult)
      hasSqrtInput = true;
  }
  if (!hasSqrtInput)
    return false;

  // broadcast: generic that broadcasts 1D rsqrt to 2D.
  Value rsqrtResult = rsqrtGeneric.getResult(0);
  if (!rsqrtResult.hasOneUse())
    return false;
  auto broadcastGeneric =
      dyn_cast<linalg::GenericOp>(*rsqrtResult.getUsers().begin());
  if (!broadcastGeneric || !isBroadcast(broadcastGeneric))
    return false;

  // normalize: mulf(x, rsqrt_broadcast).
  Value broadcastResult = broadcastGeneric.getResult(0);
  if (!broadcastResult.hasOneUse())
    return false;
  auto normalizeOp =
      dyn_cast<linalg::GenericOp>(*broadcastResult.getUsers().begin());
  if (!normalizeOp || !bodyContainsOps<arith::MulFOp>(normalizeOp))
    return false;
  if (!isAllParallel(normalizeOp) || normalizeOp.getNumDpsInputs() != 2)
    return false;

  Value normIn0 = normalizeOp.getDpsInputOperand(0)->get();
  Value normIn1 = normalizeOp.getDpsInputOperand(1)->get();
  bool normValid = (normIn0 == input && normIn1 == broadcastResult) ||
                   (normIn1 == input && normIn0 == broadcastResult);
  if (!normValid)
    return false;

  // Weight multiply: mulf(normalized, weights).
  Value normResult = normalizeOp.getResult(0);
  if (!normResult.hasOneUse())
    return false;
  auto weightMulOp =
      dyn_cast<linalg::GenericOp>(*normResult.getUsers().begin());
  if (!weightMulOp || !bodyContainsOps<arith::MulFOp>(weightMulOp))
    return false;
  if (!isAllParallel(weightMulOp) || weightMulOp.getNumDpsInputs() != 2)
    return false;

  Value wmIn0 = weightMulOp.getDpsInputOperand(0)->get();
  Value wmIn1 = weightMulOp.getDpsInputOperand(1)->get();
  Value weights;
  if (wmIn0 == normResult)
    weights = wmIn1;
  else if (wmIn1 == normResult)
    weights = wmIn0;
  else
    return false;

  LLVM_DEBUG(DBGS() << "  Full 2D RMS norm pattern matched!\n");

  pat.square = squareGeneric;
  pat.sumReduce = sumGeneric;
  pat.divByN = divByNOp;
  pat.addEps = addEpsOp;
  pat.sqrtOp = sqrtGeneric;
  pat.rsqrtOp = rsqrtGeneric;
  pat.broadcast = broadcastGeneric;
  pat.normalize = normalizeOp;
  pat.weightMul = weightMulOp;
  pat.input = input;
  pat.weights = weights;
  pat.epsilon = epsilon;
  pat.normSize = normSize;
  return true;
}

/// Collect ops to erase for a matched 2D RmsNorm pattern.
static void collectOpsToErase(RmsNormPattern2D &pat,
                              SmallVectorImpl<Operation *> &toErase) {
  auto addIfValid = [&](Operation *op) {
    if (op)
      toErase.push_back(op);
  };
  addIfValid(pat.weightMul);
  addIfValid(pat.normalize);
  addIfValid(pat.broadcast);
  addIfValid(pat.rsqrtOp);
  addIfValid(pat.sqrtOp);
  addIfValid(pat.addEps);
  addIfValid(pat.divByN);
  addIfValid(pat.sumReduce);
  addIfValid(pat.square);

  // Erase linalg.fill ops that feed constants into the 1D generics.
  auto eraseDefiningFillIfDead = [&](linalg::GenericOp op) {
    if (!op)
      return;
    for (int i = 0, e = op.getNumDpsInputs(); i < e; ++i) {
      if (auto fillOp =
              op.getDpsInputOperand(i)->get().getDefiningOp<linalg::FillOp>()) {
        if (fillOp->hasOneUse())
          toErase.push_back(fillOp);
      }
    }
    for (OpOperand &init : op.getDpsInitsMutable()) {
      if (auto fillOp = init.get().getDefiningOp<linalg::FillOp>()) {
        if (fillOp->hasOneUse())
          toErase.push_back(fillOp);
      }
    }
  };
  eraseDefiningFillIfDead(pat.sumReduce);
  eraseDefiningFillIfDead(pat.divByN);
  eraseDefiningFillIfDead(pat.addEps);
}

/// Collect ops to erase for a matched RmsNorm pattern.
static void collectOpsToErase(RmsNormPattern &pat,
                              SmallVectorImpl<Operation *> &toErase) {
  auto addIfValid = [&](Operation *op) {
    if (op)
      toErase.push_back(op);
  };
  addIfValid(pat.weightMul);
  addIfValid(pat.normalize);
  addIfValid(pat.broadcastFill);
  addIfValid(pat.rsqrt);
  addIfValid(pat.sqrt);
  addIfValid(pat.addEps);
  addIfValid(pat.divByN);
  addIfValid(pat.extract);
  addIfValid(pat.sumReduce);
  addIfValid(pat.square);

  // Erase the tensor.insert that feeds the reduction init (0-d tensor), if
  // dead.
  for (OpOperand &init : pat.sumReduce.getDpsInitsMutable()) {
    if (auto insertOp = init.get().getDefiningOp<tensor::InsertOp>()) {
      if (insertOp->hasOneUse())
        toErase.push_back(insertOp);
    }
  }
}

/// Collect all ops that should be erased for a matched pattern, in reverse
/// topological order.  Includes the linalg.fill ops that initialize the
/// reduction accumulators.
static void collectOpsToErase(const SoftmaxPattern &pat,
                              SmallVectorImpl<Operation *> &toErase) {
  auto addIfValid = [&](Operation *op) {
    if (op)
      toErase.push_back(op);
  };
  addIfValid(pat.mul);
  addIfValid(pat.broadcastRcp);
  addIfValid(pat.reciprocal);
  addIfValid(pat.sumReduce);
  addIfValid(pat.exp);
  addIfValid(pat.sub);
  addIfValid(pat.broadcastMax);
  addIfValid(pat.maxReduce);

  // Also erase linalg.fill ops that feed into the reductions as init values,
  // if they have no other users.
  auto eraseDefiningFillIfDead = [&](linalg::GenericOp reduceOp) {
    if (!reduceOp)
      return;
    for (OpOperand &init : reduceOp.getDpsInitsMutable()) {
      if (auto fillOp = init.get().getDefiningOp<linalg::FillOp>()) {
        if (fillOp->hasOneUse())
          toErase.push_back(fillOp);
      }
    }
  };
  eraseDefiningFillIfDead(pat.maxReduce);
  eraseDefiningFillIfDead(pat.sumReduce);
}

/// The pass that walks the IR to replace composite patterns with single ops.
struct LinalgToCompositePass
    : public ::impl::LinalgToCompositeBase<LinalgToCompositePass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, bufferization::BufferizationDialect,
                    composite::CompositeDialect, func::FuncDialect,
                    linalg::LinalgDialect, math::MathDialect, scf::SCFDialect,
                    tensor::TensorDialect>();
  }

  void runOnOperation() override {
    FunctionOpInterface funcOp = getOperation();
    llvm::SmallPtrSet<Operation *, 8> matched;
    SmallVector<std::pair<SoftmaxPattern, Location>> softmaxToRewrite;
    SmallVector<std::pair<LayerNormPattern, Location>> layerNormToRewrite;
    SmallVector<std::pair<RmsNormPattern, Location>> rmsNormToRewrite;
    SmallVector<std::pair<RmsNormPattern2D, Location>> rmsNorm2DToRewrite;

    // Collect all matches.
    funcOp.walk([&](linalg::GenericOp generic) {
      if (matched.contains(generic))
        return;

      // Try softmax.
      SoftmaxPattern smPat;
      if (matchSoftmax(generic, smPat)) {
        auto markMatched = [&](linalg::GenericOp op) {
          if (op)
            matched.insert(op.getOperation());
        };
        markMatched(smPat.maxReduce);
        markMatched(smPat.broadcastMax);
        markMatched(smPat.sub);
        markMatched(smPat.exp);
        markMatched(smPat.sumReduce);
        markMatched(smPat.reciprocal);
        markMatched(smPat.broadcastRcp);
        markMatched(smPat.mul);
        softmaxToRewrite.emplace_back(smPat, smPat.mul.getLoc());
        return;
      }

      // Try layer norm (before rms_norm: the variance sub-pattern inside
      // layer_norm could falsely match rms_norm if tried first).
      LayerNormPattern lnPat;
      if (matchLayerNorm(generic, lnPat)) {
        auto markMatched = [&](Operation *op) {
          if (op)
            matched.insert(op);
        };
        markMatched(lnPat.meanSumReduce);
        markMatched(lnPat.subGeneric);
        markMatched(lnPat.selectGeneric);
        markMatched(lnPat.squareGeneric);
        markMatched(lnPat.varSumReduce);
        markMatched(lnPat.normalizeGeneric);
        markMatched(lnPat.weightMulGeneric);
        markMatched(lnPat.biasAddGeneric);
        layerNormToRewrite.emplace_back(lnPat, lnPat.biasAddGeneric.getLoc());
        return;
      }

      // Try RMS norm.
      RmsNormPattern rmPat;
      if (matchRmsNorm(generic, rmPat)) {
        auto markMatched = [&](Operation *op) {
          if (op)
            matched.insert(op);
        };
        markMatched(rmPat.square);
        markMatched(rmPat.sumReduce);
        markMatched(rmPat.normalize);
        markMatched(rmPat.weightMul);
        rmsNormToRewrite.emplace_back(rmPat, rmPat.weightMul.getLoc());
        return;
      }

      // Try RMS norm (2D row-reduction variant from block-pointer kernels).
      RmsNormPattern2D rm2DPat;
      if (matchRmsNorm2D(generic, rm2DPat)) {
        auto markMatched = [&](Operation *op) {
          if (op)
            matched.insert(op);
        };
        markMatched(rm2DPat.square);
        markMatched(rm2DPat.sumReduce);
        markMatched(rm2DPat.divByN);
        markMatched(rm2DPat.addEps);
        markMatched(rm2DPat.sqrtOp);
        markMatched(rm2DPat.rsqrtOp);
        markMatched(rm2DPat.broadcast);
        markMatched(rm2DPat.normalize);
        markMatched(rm2DPat.weightMul);
        rmsNorm2DToRewrite.emplace_back(rm2DPat, rm2DPat.weightMul.getLoc());
        return;
      }
    });

    // Rewrite softmax matches.
    for (auto &[pat, loc] : softmaxToRewrite) {
      OpBuilder builder(pat.mul);

      auto inputType = cast<RankedTensorType>(pat.input.getType());
      auto resultType = cast<RankedTensorType>(pat.mul.getResult(0).getType());
      int64_t rank = inputType.getRank();

      // Create the init tensor (destination-passing style).
      Value init = tensor::EmptyOp::create(builder, loc, resultType.getShape(),
                                           resultType.getElementType());

      // Create composite.softmax with axis = innermost dim.
      auto axisAttr = builder.getI64IntegerAttr(rank - 1);
      auto softmaxOp = composite::SoftmaxOp::create(builder, loc, resultType,
                                                    axisAttr, pat.input, init);

      // Replace all uses of the final mul result.
      pat.mul.getResult(0).replaceAllUsesWith(softmaxOp->getResult(0));

      // Erase matched ops.
      SmallVector<Operation *> toErase;
      collectOpsToErase(pat, toErase);
      for (Operation *op : toErase) {
        if (op->use_empty())
          op->erase();
      }
    }

    // Rewrite layer norm matches.
    for (auto &[pat, loc] : layerNormToRewrite) {
      OpBuilder builder(pat.biasAddGeneric);

      auto resultType =
          cast<RankedTensorType>(pat.biasAddGeneric.getResult(0).getType());
      int64_t rank = cast<RankedTensorType>(pat.input.getType()).getRank();

      Value init = tensor::EmptyOp::create(builder, loc, resultType.getShape(),
                                           resultType.getElementType());

      auto axisAttr = builder.getI64IntegerAttr(rank - 1);
      auto epsAttr = builder.getF32FloatAttr(pat.epsilon);
      auto normSizeAttr = builder.getF32FloatAttr(pat.normSize);
      auto layerNormOp = composite::LayerNormOp::create(
          builder, loc, resultType, axisAttr, epsAttr, normSizeAttr, pat.input,
          pat.weights, pat.bias, init, pat.meanOutMemref, pat.rstdOutMemref);

      pat.biasAddGeneric.getResult(0).replaceAllUsesWith(
          layerNormOp->getResult(0));

      SmallVector<Operation *> toErase;
      collectOpsToErase(pat, toErase);
      for (Operation *op : toErase) {
        if (op->use_empty())
          op->erase();
      }
    }

    // Rewrite RMS norm matches.
    for (auto &[pat, loc] : rmsNormToRewrite) {
      OpBuilder builder(pat.weightMul);

      auto resultType =
          cast<RankedTensorType>(pat.weightMul.getResult(0).getType());
      int64_t rank = cast<RankedTensorType>(pat.input.getType()).getRank();

      Value init = tensor::EmptyOp::create(builder, loc, resultType.getShape(),
                                           resultType.getElementType());

      auto axisAttr = builder.getI64IntegerAttr(rank - 1);
      auto epsAttr = builder.getF32FloatAttr(pat.epsilon);
      auto normSizeAttr = builder.getF32FloatAttr(pat.normSize);
      auto rmsNormOp = composite::RmsNormOp::create(
          builder, loc, resultType, axisAttr, epsAttr, normSizeAttr, pat.input,
          pat.weights, init);

      pat.weightMul.getResult(0).replaceAllUsesWith(rmsNormOp->getResult(0));

      SmallVector<Operation *> toErase;
      collectOpsToErase(pat, toErase);
      for (Operation *op : toErase) {
        if (op->use_empty())
          op->erase();
      }
    }

    // Rewrite 2D RMS norm matches.
    for (auto &[pat, loc] : rmsNorm2DToRewrite) {
      OpBuilder builder(pat.weightMul);

      auto resultType =
          cast<RankedTensorType>(pat.weightMul.getResult(0).getType());
      int64_t rank = cast<RankedTensorType>(pat.input.getType()).getRank();

      Value init = tensor::EmptyOp::create(builder, loc, resultType.getShape(),
                                           resultType.getElementType());

      auto axisAttr = builder.getI64IntegerAttr(rank - 1);
      auto epsAttr = builder.getF32FloatAttr(pat.epsilon);
      auto normSizeAttr = builder.getF32FloatAttr(pat.normSize);
      auto rmsNormOp = composite::RmsNormOp::create(
          builder, loc, resultType, axisAttr, epsAttr, normSizeAttr, pat.input,
          pat.weights, init);

      pat.weightMul.getResult(0).replaceAllUsesWith(rmsNormOp->getResult(0));

      SmallVector<Operation *> toErase;
      collectOpsToErase(pat, toErase);
      for (Operation *op : toErase) {
        if (op->use_empty())
          op->erase();
      }
    }
  }
};

} // namespace

std::unique_ptr<InterfacePass<FunctionOpInterface>>
hexagon::createLinalgToCompositePass() {
  return std::make_unique<LinalgToCompositePass>();
}
