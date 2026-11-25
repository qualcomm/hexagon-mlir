//===- HexagonVectorizePackedGenericPass.cpp ------------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc.
// SPDX-License-Identifier: BSD-3-Clause.
//
//===----------------------------------------------------------------------===//
//
// General-purpose vectorization for linalg.generic ops with packed layouts.
//
// This pass vectorizes linalg.generic operations where some inputs/outputs
// have last 2 dimensions as 32x2 (packed layout). The pass applies heuristics:
//
// 1) Prioritize vectorizing loads/stores of tensors with most elements
//    (usually inputs and outputs, so last 2 iterators remain intact)
// 2) Perform reductions in registers by permuting loops to bring reduction
//    dims close together (without changing positions of last 2 iterators)
// 3) Vectorize loads/stores of smaller tensors via loop permutations and
//    splitting (without changing positions of last 2 iterators)
// 4) Use shuffles on loaded values to make them usable in vector computation
// 5) Use shuffles to store results back to memory (if necessary)
//
// The shuffle maps are derived from the affine maps for tensor accesses.
//
//===----------------------------------------------------------------------===//

#include "hexagon/Conversion/LinalgToLLVM/LinalgToLLVM.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <optional>

using mlir::hexagon::HexagonVectorizePackedGenericOptions;

#define GEN_PASS_DEF_HEXAGONVECTORIZEPACKEDGENERIC
#include "hexagon/Conversion/LinalgToLLVM/Passes.h.inc"

using namespace mlir;
using namespace mlir::linalg;

namespace {

struct TensorInfo;

// Represents information about a tensor operand
struct TensorInfo {
  Value tensor;
  RankedTensorType type;
  AffineMap indexingMap;
  int64_t numElements;
  bool isOutput;
  int operandIndex;

  // Check if last 2 dims are 32x2 (packed layout)
  bool hasPackedLayout() const {
    if (type.getRank() < 2)
      return false;
    ArrayRef<int64_t> shape = type.getShape();
    return shape[shape.size() - 2] == 32 && shape[shape.size() - 1] == 2;
  }

  // Check if can be vectorized as 64-wide vector (collapse 32x2 -> 64)
  bool canVectorizeAs64() const {
    return hasPackedLayout() && indexingMap.getNumResults() >= 2;
  }
};

// Support tensors with last dim 32 accessed by packedOuterDim, where the
// tensor's last-1 dim is divisible by 2. These can be repacked to enable
// vector<64> loads (like the tricky/simple tests).
// Shape should be (Other, 2*X, 32)
// Map should be (?, AffineDimExpr, packedOuterDim)
static bool isNx32PackedOuterAccess(const TensorInfo &info,
                                    unsigned packedOuterDim) {
  if (info.hasPackedLayout())
    return false;
  if (info.type.getRank() < 2)
    return false;

  ArrayRef<int64_t> shape = info.type.getShape();
  int64_t last = shape.back();
  int64_t prev = shape[shape.size() - 2];
  if (ShapedType::isDynamic(last) || ShapedType::isDynamic(prev))
    return false;
  if (last != 32 || (prev % 2) != 0)
    return false;

  auto results = info.indexingMap.getResults();
  if (static_cast<int64_t>(results.size()) != info.type.getRank())
    return false;
  auto lastExpr = dyn_cast<AffineDimExpr>(results.back());
  // Check if the last index expression used in accessing the tensor
  // is packedOuterDim
  if (!lastExpr || lastExpr.getPosition() != packedOuterDim)
    return false;
  // Check that the last but one index expression is an AffineDimExpr
  auto prevExpr = dyn_cast<AffineDimExpr>(results[results.size() - 2]);
  if (!prevExpr)
    return false;
  return true;
}

// Analyzes a generic op to determine vectorization strategy
struct VectorizationStrategy {
  struct SplitPlan {
    int inputIndex = -1;
    unsigned loopDim = 0;
    int64_t splitFactor = 1;
  }; // Which input to split, which loop to split it on, and by what factor

  linalg::GenericOp op;
  SmallVector<TensorInfo> inputs;
  SmallVector<TensorInfo> outputs;
  SmallVector<utils::IteratorType> iteratorTypes;
  int numLoops;

  // Indices of parallel and reduction iterators
  SmallVector<unsigned> parallelDims;
  SmallVector<unsigned> reductionDims;

  // Primary tensors to vectorize (highest priority)
  // Indices into inputs and outputs vectors. Non-negative indices index into
  // inputs vector, negative indices index into outputs vector (-ve i refers
  // to outputs[-(i+1)])
  SmallVector<int> primaryTensors;

  // Loop permutation to optimize vectorization
  SmallVector<unsigned> loopPermutation;

  // Inverse of loopPermutation for index reconstruction.
  SmallVector<unsigned> inverseLoopPermutation;

  // Optional split plans for secondary accesses.
  SmallVector<SplitPlan> splitPlans;

  VectorizationStrategy(linalg::GenericOp op) : op(op) { analyze(); }

  void analyze() {
    numLoops = op.getNumLoops();
    iteratorTypes = op.getIteratorTypesArray();

    // Classify iterators
    for (unsigned i = 0; i < iteratorTypes.size(); ++i) {
      if (linalg::isParallelIterator(iteratorTypes[i]))
        parallelDims.push_back(i);
      else if (linalg::isReductionIterator(iteratorTypes[i]))
        reductionDims.push_back(i);
      // TODO: Bail out if iterator is neither parallel nor reduction?
    }

    // Collect input tensor info
    for (unsigned i = 0; i < op.getNumDpsInputs(); ++i) {
      OpOperand *operand = op.getDpsInputOperand(i);
      auto tensorType = dyn_cast<RankedTensorType>(operand->get().getType());
      if (!tensorType)
        continue;

      TensorInfo info;
      info.tensor = operand->get();
      info.type = tensorType;
      info.indexingMap = op.getMatchingIndexingMap(operand);
      // Use -1 as a sentinel for dynamic shapes; getNumElements() asserts on
      // dynamic shapes, so guard it conservatively.
      info.numElements =
          tensorType.hasStaticShape() ? tensorType.getNumElements() : -1;
      info.isOutput = false;
      info.operandIndex = i;
      inputs.push_back(info);
    }

    // Collect output tensor info
    // TODO: Refacto
    for (unsigned i = 0; i < op.getNumDpsInits(); ++i) {
      OpOperand *operand = op.getDpsInitOperand(i);
      auto tensorType = dyn_cast<RankedTensorType>(operand->get().getType());
      if (!tensorType)
        continue;

      TensorInfo info;
      info.tensor = operand->get();
      info.type = tensorType;
      info.indexingMap = op.getMatchingIndexingMap(operand);
      // Use -1 as a sentinel for dynamic shapes; getNumElements() asserts on
      // dynamic shapes, so guard it conservatively.
      info.numElements =
          tensorType.hasStaticShape() ? tensorType.getNumElements() : -1;
      info.isOutput = true;
      info.operandIndex = i;
      outputs.push_back(info);
    }

    // Heuristic 1: Identify primary tensors (largest with packed layout)
    identifyPrimaryTensors();

    // Heuristic 2: Compute optimal loop permutation
    // Identifies loops to be split for vectorizing accesses involving
    // non-packed tensors.
    identifySplitCandidates();
    computeLoopPermutation();
    computeInversePermutation();
  }

  void identifySplitCandidates() {
    if (numLoops <= 2)
      return;

    for (const auto &input : inputs) {
      if (input.hasPackedLayout())
        continue;
      if (input.type.getRank() < 2)
        continue;

      ArrayRef<int64_t> shape = input.type.getShape();
      if (ShapedType::isDynamic(shape[shape.size() - 2]) ||
          ShapedType::isDynamic(shape[shape.size() - 1]))
        continue;

      auto results = input.indexingMap.getResults();
      if (results.size() < 2)
        continue;

      auto penultimate = dyn_cast<AffineDimExpr>(results[results.size() - 2]);
      auto trailing = dyn_cast<AffineDimExpr>(results[results.size() - 1]);
      // trailing is used only to check if the last index is AffineDimExpr
      if (!penultimate || !trailing)
        continue;

      // Conservative heuristic: if the access ends with (..., loopDim, dN)
      // and the trailing tensor dim is 32 or 2, record a split plan for the
      // preceding loop dim so future vector loads can fuse the split-inner part
      // with the trailing packed dimension. Restrict to exact divisibility.
      int64_t trailingSize = shape.back();
      int64_t splitFactor = 1;
      if (trailingSize == 32)
        splitFactor = 2;
      else if (trailingSize == 2)
        splitFactor = 32;
      else
        continue;

      unsigned loopDim = penultimate.getPosition();
      // Skip if the penultimate index to access the tensor is an
      // "inner packed dim" (last 2 dims - usually 32x2 for FP16).
      // The trailing index could actually be numLoops - 2.
      if (loopDim >= static_cast<unsigned>(numLoops - 2))
        continue;

      int64_t loopExtent = -1;
      // Make sure we can split the loop whose iterator indexes penultimate
      // dim of input, if we want to create a split plan for the tensor
      if (!outputs.empty() &&
          static_cast<int64_t>(loopDim) < outputs[0].type.getRank())
        loopExtent = outputs[0].type.getShape()[loopDim];
      if (ShapedType::isDynamic(loopExtent) || loopExtent <= 0 ||
          loopExtent % splitFactor != 0)
        continue;

      splitPlans.push_back(SplitPlan{input.operandIndex, loopDim, splitFactor});
    }
  }

  void identifyPrimaryTensors() {
    // Find tensors with most elements that have packed layout.
    // Tensors with dynamic shapes have numElements == -1 and are skipped
    // conservatively (we cannot vectorize them).
    int64_t maxElements = 0;

    for (size_t i = 0; i < inputs.size(); ++i) {
      if (inputs[i].hasPackedLayout() && inputs[i].numElements > maxElements) {
        maxElements = inputs[i].numElements;
      }
    }

    for (size_t i = 0; i < outputs.size(); ++i) {
      if (outputs[i].hasPackedLayout() &&
          outputs[i].numElements > maxElements) {
        maxElements = outputs[i].numElements;
      }
    }

    // maxElements == 0 means no static packed tensor was found; bail out.
    if (maxElements <= 0)
      return;

    // Mark all tensors with max elements as primary
    for (size_t i = 0; i < inputs.size(); ++i) {
      if (inputs[i].hasPackedLayout() && inputs[i].numElements == maxElements) {
        primaryTensors.push_back(i);
      }
    }

    for (size_t i = 0; i < outputs.size(); ++i) {
      if (outputs[i].hasPackedLayout() &&
          outputs[i].numElements == maxElements) {
        primaryTensors.push_back(-(i + 1)); // Negative for outputs
      }
    }
  }

  // Reorders loops so the chosen dimension that is split is moved before the
  // packed dims. If there are multiple candidate split dims, only the first
  // parallel dim that is split (or reduction dim, if split) is reordered. If
  // no dim is split, then identity permutation.
  void computeLoopPermutation() {
    // Start with identity permutation
    loopPermutation.clear();
    for (unsigned i = 0; i < numLoops; ++i) {
      loopPermutation.push_back(i);
    }

    // Heuristic 2: Keep last 2 iterators intact, but permute others
    // to bring reduction dims together
    if (numLoops <= 2)
      return;

    // Separate into: [other dims] [last-2 dims]
    unsigned numOtherDims = numLoops - 2;
    SmallVector<unsigned> otherDims;

    for (unsigned i = 0; i < numOtherDims; ++i) {
      otherDims.push_back(i);
    }

    // For each loop dim, fix the split factor - max of the factor in
    // splitPlans
    DenseMap<unsigned, int64_t> splitPriority;
    // Since lookup for an absent dim returns 0 (default for int64_t type),
    // no need to initialize splitPriority before using it in std::max
    for (const SplitPlan &plan : splitPlans)
      splitPriority[plan.loopDim] =
          std::max(splitPriority.lookup(plan.loopDim), plan.splitFactor);

    // Sort other dims: keep split candidates closest to the packed tail while
    // still clustering reductions for register accumulation.
    std::stable_sort(
        otherDims.begin(), otherDims.end(), [&](unsigned a, unsigned b) {
          bool aIsSplit = splitPriority.count(a);
          bool bIsSplit = splitPriority.count(b);
          bool aIsReduction = linalg::isReductionIterator(iteratorTypes[a]);
          bool bIsReduction = linalg::isReductionIterator(iteratorTypes[b]);
          if (aIsSplit != bIsSplit)
            return !aIsSplit; // Non-split comes first
          // Parallel dims come first
          if (aIsReduction != bIsReduction)
            return !aIsReduction; // Non-reduction comes first
          return a < b; // Otherwise, stable sort based on original order
        });

    // If we found a split candidate, place the most profitable one immediately
    // before the trailing packed iterators. This preserves the last two
    // iterators while making loop splitting possible after permutation.
    // Most profitable definition: first parallel dim that is split (if no
    // parallel dim is split, returns the first reduction dim that is split)
    // For any dim that is split, split factor is the largest among inputs.
    if (!otherDims.empty() && !splitPlans.empty()) {
      // Find the first dim that is split
      auto bestIt = llvm::find_if(otherDims, [&](unsigned dim) {
        return llvm::any_of(splitPlans, [&](const SplitPlan &plan) {
          return plan.loopDim == dim;
        });
      });
      if (bestIt != otherDims.end()) {
        unsigned splitDim = *bestIt;
        otherDims.erase(bestIt);
        otherDims.push_back(splitDim);
      }
    }

    // Rebuild permutation: [sorted other dims] [last-2 dims unchanged]
    for (unsigned i = 0; i < numOtherDims; ++i) {
      loopPermutation[i] = otherDims[i];
    }
    // Last 2 dims stay in place
    loopPermutation[numLoops - 2] = numLoops - 2;
    loopPermutation[numLoops - 1] = numLoops - 1;
  }

  void computeInversePermutation() {
    inverseLoopPermutation.assign(numLoops, 0);
    // Def: I[P[i]] = i. That is, if we apply inv on the permutation,
    // we should get identity.
    for (unsigned i = 0; i < loopPermutation.size(); ++i)
      inverseLoopPermutation[loopPermutation[i]] = i;
  }

  bool supportsNonPackedAccess(const TensorInfo &tensorInfo) const {
    if (tensorInfo.hasPackedLayout())
      return true;

    bool usesPackedOuter = false;
    bool usesPackedInner = false;
    for (AffineExpr expr : tensorInfo.indexingMap.getResults()) {
      auto dimExpr = dyn_cast<AffineDimExpr>(expr);
      if (!dimExpr)
        return false;
      if (dimExpr.getPosition() == static_cast<unsigned>(numLoops - 2))
        usesPackedOuter = true;
      if (dimExpr.getPosition() == static_cast<unsigned>(numLoops - 1))
        usesPackedInner = true;
    }

    // Supported cases:
    //  * scalars
    //  * the non-packed tensor is invariant across the packed tail, or
    //  * it depends only on the innermost packed dim (size 2), which we can
    //    materialize as an alternating 64-lane vector, or
    //  * it is a rank >= 2 tensor with trailing dim 32 accessed as (dX,
    //  packedOuter)
    //    (like the tricky/simple tests), which we can repack to expose a
    //    64-lane vector load after splitting dX by 2.
    if (!usesPackedOuter) // Scalars pass this test
      return true;
    // usesPackedOuter, shape: (Y, 2*X, 32); Map: (*, AffineDimExpr, numLoops-2)
    // Y could be empty
    if (usesPackedOuter && !usesPackedInner &&
        isNx32PackedOuterAccess(tensorInfo, numLoops - 2))
      return true;

    return false;
  }

  bool canVectorize() const {
    // Check if we have at least one primary tensor with packed layout
    if (primaryTensors.empty())
      return false;

    // Check packed tail iterator types.
    if (numLoops < 2)
      return false;

    // The innermost packed dim (size 2) must remain parallel.
    if (!linalg::isParallelIterator(iteratorTypes[numLoops - 1]))
      return false;

    // The packed-outer dim (size 32) can be either parallel (elementwise case)
    // or reduction (reduction case).
    if (!linalg::isParallelIterator(iteratorTypes[numLoops - 2]) &&
        !linalg::isReductionIterator(iteratorTypes[numLoops - 2]))
      return false;

    // scalar arguments to generic pass supportsNonPackedAccess
    return llvm::all_of(inputs,
                        [&](const TensorInfo &tensorInfo) {
                          return supportsNonPackedAccess(tensorInfo);
                        }) &&
           llvm::all_of(outputs, [&](const TensorInfo &tensorInfo) {
             return supportsNonPackedAccess(tensorInfo);
           });
  }
};

// Helper to create shuffle mask for deinterleaving or interleaving
static SmallVector<int64_t> createDeinterleaveMask(int64_t size, bool even) {
  SmallVector<int64_t> mask;
  mask.reserve(size);
  for (int64_t i = 0; i < size; ++i) {
    mask.push_back(2 * i + (even ? 0 : 1));
  }
  return mask;
}

static SmallVector<int64_t> createHalfMask(int64_t size, bool upper) {
  SmallVector<int64_t> mask;
  mask.reserve(size);
  int64_t offset = upper ? size : 0;
  for (int64_t i = 0; i < size; ++i)
    mask.push_back(offset + i);
  return mask;
}

static Value selectLowerOrUpper32(OpBuilder &builder, Location loc, Value vec64,
                                  Type elementType, Value selectUpper) {
  auto vec32Ty = VectorType::get({32}, elementType);
  Value lower = vector::ShuffleOp::create(builder, loc, vec64, vec64,
                                          createHalfMask(32, /*upper=*/false));
  Value upper = vector::ShuffleOp::create(builder, loc, vec64, vec64,
                                          createHalfMask(32, /*upper=*/true));
  return arith::SelectOp::create(builder, loc, selectUpper, upper, lower);
}

static SmallVector<int64_t> createDuplicateHalfMask64(bool upperHalf) {
  SmallVector<int64_t> mask;
  mask.reserve(64);
  int64_t base = upperHalf ? 32 : 0;
  for (int64_t i = 0; i < 32; ++i) {
    mask.push_back(base + i);
    mask.push_back(base + i);
  }
  return mask;
}

static Value duplicateLowerOrUpperHalf64(OpBuilder &builder, Location loc,
                                         Value vec64, Value selectUpper) {
  Value lowerDup = vector::ShuffleOp::create(builder, loc, vec64, vec64,
                                             createDuplicateHalfMask64(false));
  Value upperDup = vector::ShuffleOp::create(builder, loc, vec64, vec64,
                                             createDuplicateHalfMask64(true));
  return arith::SelectOp::create(builder, loc, selectUpper, upperDup, lowerDup);
}

static SmallVector<int64_t> createInterleaveMask(int64_t size) {
  SmallVector<int64_t> mask;
  mask.reserve(2 * size);
  for (int64_t i = 0; i < size; ++i) {
    mask.push_back(i);
    mask.push_back(size + i);
  }
  return mask;
}

static SmallVector<int64_t> createAlternateMask(int64_t size) {
  SmallVector<int64_t> mask;
  mask.reserve(size);
  // vector.shuffle mask indices range over concatenation of the two inputs:
  //   first vector: 0..(N-1)
  //   second vector: N..(2N-1)
  // For scalar pair materialization we want alternating picks from the first
  // and second inputs, reusing the same lane from each source vector:
  //   [lhs[0], rhs[0], lhs[0], rhs[0], ...]
  // This matches the packed 32x2 lane layout after collapsing to 64.
  int64_t half = size / 2;
  for (int64_t i = 0; i < half; ++i) {
    mask.push_back(0);
    mask.push_back(half);
  }
  return mask;
}

// Helper to duplicate each lane (for broadcasting across packed dim)
static SmallVector<int64_t> createDuplicateMask(int64_t size) {
  SmallVector<int64_t> mask;
  mask.reserve(2 * size);
  for (int64_t i = 0; i < size; ++i) {
    mask.push_back(i);
    mask.push_back(i);
  }
  return mask;
}

static bool usesLoopDim(AffineMap map, unsigned loopDim) {
  return llvm::any_of(map.getResults(), [&](AffineExpr expr) {
    auto dimExpr = dyn_cast<AffineDimExpr>(expr);
    return dimExpr && dimExpr.getPosition() == loopDim;
  });
}

static bool allValuesValid(ValueRange values) {
  return llvm::all_of(values, [](Value v) { return static_cast<bool>(v); });
}

struct ValueRep {
  Value scalar;
  Value even;
  Value odd;
  Value vector;

  bool hasAny() const { return scalar || even || odd || vector; }
  bool hasScalar() const { return static_cast<bool>(scalar); }
  bool hasPair() const { return even && odd; }
  bool hasVector() const { return static_cast<bool>(vector); }
};

static Value createAlternatingPairVector(OpBuilder &builder, Location loc,
                                         Type elementType, Value even,
                                         Value odd) {
  auto vec32Ty = VectorType::get({32}, elementType);
  Value even32 = vector::BroadcastOp::create(builder, loc, vec32Ty, even);
  Value odd32 = vector::BroadcastOp::create(builder, loc, vec32Ty, odd);
  return vector::ShuffleOp::create(builder, loc, even32, odd32,
                                   createAlternateMask(64));
}

static Value materializeVector(ValueRep &rep, OpBuilder &builder, Location loc,
                               VectorType vectorType, Type elementType) {
  if (rep.vector)
    return rep.vector;
  if (rep.scalar) {
    rep.vector =
        vector::BroadcastOp::create(builder, loc, vectorType, rep.scalar);
    return rep.vector;
  }
  if (rep.hasPair()) {
    rep.vector = createAlternatingPairVector(builder, loc, elementType,
                                             rep.even, rep.odd);
    return rep.vector;
  }
  return Value();
}

static Value getScalarLane(ValueRep &rep, bool useOdd) {
  if (rep.scalar)
    return rep.scalar;
  return useOdd ? rep.odd : rep.even;
}

// Given an op, create a new op using values mapped by valueMap.
// Scalar [prefered over] > Scalar Pair [prefered over] > Vector.
// Returns true if new op is created at the current insertion point of builder,
//         false otherwise (can't handle some cases - eg. op's result type is
//                          a vector, can't materialize a necessary vector
//                          representation, a result's type is not elementType,
//                          etc.)
// When it returns true, valueMap is updated with the entry oldRes -> newRes
/// Clone op with the given operands and result types, copying all attributes.
static Operation *cloneOpWithOperands(OpBuilder &b, Operation &op,
                                      ArrayRef<Value> operands,
                                      ArrayRef<Type> resultTypes) {
  OperationState st(op.getLoc(), op.getName().getStringRef());
  st.addOperands(operands);
  st.addTypes(resultTypes);
  st.addAttributes(op.getAttrs()); // TODO: Is it correct to copy attributes?
                                   // TODO: Do we need vector attributes?
  return b.create(st);
}

static bool cloneOpWithRepresentations(Operation &op,
                                       DenseMap<Value, ValueRep> &valueMap,
                                       OpBuilder &builder,
                                       VectorType vectorType, Type elementType,
                                       bool eagerVectorization = false) {
  auto getRep = [&](Value v) -> ValueRep * {
    auto it = valueMap.find(v);
    return it == valueMap.end() ? nullptr : &it->second;
  };

  auto hasReadyRep = [&](Value v) {
    if (auto *rep = getRep(v))
      return rep->hasAny();
    // Not defined in map => to be used directly (defined outside generic op
    // body?)
    return true;
  };

  if (!llvm::all_of(op.getOperands(), hasReadyRep)) {
    llvm_unreachable("An operand in valueMap, but no ready representation!");
  }

  bool scalarResults = llvm::all_of(op.getResultTypes(), [](Type t) {
    return isa<IntegerType, FloatType, IndexType>(t);
  });
  // A generic op's body contains code to compute one element of
  // result. Can't handle case when an element is a non-scalar.
  if (!scalarResults)
    return false;

  // Track whether every operand that has a vector rep also has a scalar or
  // pair rep available.  When true we can keep the computation scalar/pair
  // and defer the broadcast/shuffle to the point where a true vector value
  // (one that only exists as a vector, e.g. a packed tensor load) forces
  // vectorization.  This avoids unnecessary broadcast→vector-op sequences
  // when scalar-op→broadcast achieves the same result more efficiently.
  bool anyVectorOperand = false;
  bool anyPairOperand = false;
  bool anyScalarOperand = false;
  bool allVectorOpsHaveScalarOrPair = true;
  for (Value operand : op.getOperands()) {
    if (auto *rep = getRep(operand)) {
      anyVectorOperand |= rep->hasVector();
      anyPairOperand |= rep->hasPair();
      anyScalarOperand |= rep->hasScalar();
      if (rep->hasVector() && !rep->hasScalar() && !rep->hasPair())
        allVectorOpsHaveScalarOrPair = false;
    }
  }

  SmallVector<Type> scalarResultTypes(op.getResultTypes().begin(),
                                      op.getResultTypes().end());

  // Prefer scalar/pair computation over vector when all vector operands also
  // have scalar or pair representations.  This implements the "don't vectorize
  // for the sake of vectorization" principle: if we can compute the result as
  // scalar (or pair) and broadcast/shuffle later, that is both more power- and
  // performance-efficient than broadcasting first and doing vector arithmetic.

  // At least one operand has vector rep, and at least one vector rep doesn't
  // have scalar or pair reps, so we must vectorize.
  // In eager mode, always take the vector path when possible.
  bool forceVector = eagerVectorization &&
                     (anyVectorOperand || anyPairOperand || anyScalarOperand);
  if (forceVector || (anyVectorOperand && !allVectorOpsHaveScalarOrPair)) {
    SmallVector<Value> vectorOperands;
    vectorOperands.reserve(op.getNumOperands());
    for (Value operand : op.getOperands()) {
      if (auto *rep = getRep(operand)) {
        Value vec = materializeVector(*rep, builder, op.getLoc(), vectorType,
                                      elementType);
        if (!vec)
          return false;
        vectorOperands.push_back(vec);
      } else {
        Value vecOperand = operand;
        // TODO: Check for incompatible operand type and bail out if so?
        if (vecOperand && vecOperand.getType() == elementType)
          vecOperand = vector::BroadcastOp::create(builder, op.getLoc(),
                                                   vectorType, vecOperand);
        vectorOperands.push_back(vecOperand);
      }
    }

    bool canVectorizeResults = llvm::all_of(
        op.getResultTypes(), [&](Type t) { return t == elementType; });
    if (!canVectorizeResults)
      return false;

    Operation *newOp =
        cloneOpWithOperands(builder, op, vectorOperands,
                            SmallVector<Type>(op.getNumResults(), vectorType));
    for (auto [oldRes, newRes] :
         llvm::zip(op.getResults(), newOp->getResults()))
      valueMap[oldRes].vector = newRes;
    return true;
  }
  // No vector rep, or every vector op has scalar/pair reps.
  // When all vector operands also have scalar/pair reps, fall through to the
  // pair or scalar path below.  This keeps the computation narrow and defers
  // vector materialization to the consumer that truly needs it.
  //
  // Determine whether all operands can be resolved to pair (even/odd) form.
  // An operand qualifies if it has an explicit pair rep, or if it has a scalar
  // rep (which can serve as both even and odd lanes).
  bool allCanResolveToPair = true;
  bool allCanResolveToScalar = true;
  for (Value operand : op.getOperands()) {
    if (auto *rep = getRep(operand)) {
      bool canPair = rep->hasPair() || rep->hasScalar();
      bool canScalar = rep->hasScalar();
      allCanResolveToPair &= canPair;
      allCanResolveToScalar &= canScalar;
    }
    // Operands not in the map (e.g. constants defined outside the body) are
    // usable in any form.
  }

  if (anyPairOperand && allCanResolveToPair) {
    SmallVector<Value> evenOperands;
    SmallVector<Value> oddOperands;
    evenOperands.reserve(op.getNumOperands());
    oddOperands.reserve(op.getNumOperands());
    for (Value operand : op.getOperands()) {
      if (auto *rep = getRep(operand)) {
        Value even = getScalarLane(*rep, /*useOdd=*/false);
        Value odd = getScalarLane(*rep, /*useOdd=*/true);
        if (!even || !odd)
          return false;
        evenOperands.push_back(even);
        oddOperands.push_back(odd);
      } else {
        evenOperands.push_back(operand);
        oddOperands.push_back(operand);
      }
    }
    Operation *evenOp =
        cloneOpWithOperands(builder, op, evenOperands, scalarResultTypes);
    Operation *oddOp =
        cloneOpWithOperands(builder, op, oddOperands, scalarResultTypes);
    for (auto [idx, oldRes] : llvm::enumerate(op.getResults())) {
      valueMap[oldRes].even = evenOp->getResult(idx);
      valueMap[oldRes].odd = oddOp->getResult(idx);
    }
    return true;
  }

  SmallVector<Value> scalarOperands(op.getOperands().begin(),
                                    op.getOperands().end());
  for (auto &operand : scalarOperands) {
    if (auto *rep = getRep(operand))
      operand = rep->scalar;
  }
  Operation *newOp =
      cloneOpWithOperands(builder, op, scalarOperands, scalarResultTypes);
  for (auto [idx, oldRes] : llvm::enumerate(op.getResults()))
    valueMap[oldRes].scalar = newOp->getResult(idx);
  return true;
}

// For a given rank, return the map {0, 1, ... rank-1} -> {rank-1}
static AffineMap getTrailingVectorPermutationMap(Builder &builder,
                                                 unsigned rank) {
  return AffineMap::get(rank, 0 /* symbol count */,
                        {builder.getAffineDimExpr(rank - 1)},
                        builder.getContext());
}

// Given a tensor, return the permutation map to project the last dimension
// to get a vector.
static AffineMap getTrailingVectorPermutationMapForTensor(OpBuilder &builder,
                                                          Value tensor) {
  auto ty = dyn_cast<RankedTensorType>(tensor.getType());
  assert(ty && "expected ranked tensor");
  return getTrailingVectorPermutationMap(builder, ty.getRank());
}

// Given a 64-element interleaved vector of two 32-element vectors, return
// the reduced value of 32 elements.
static Value reduceInterleavedVectorHalf(OpBuilder &builder, Location loc,
                                         Value vec64, Type elementType,
                                         vector::CombiningKind kind,
                                         bool isEven) {
  auto vec32Ty = VectorType::get({32}, elementType);
  SmallVector<int64_t> mask = createDeinterleaveMask(32, isEven);
  Value vec32 = vector::ShuffleOp::create(builder, loc, vec64, vec64, mask);
  return vector::ReductionOp::create(builder, loc, kind, vec32);
}

// Given an AffineMap and loopDim, return the position of the loopDim in the
// map's results. If not found, return nullopt.
static std::optional<unsigned> getPackedInnerResultPosition(AffineMap map,
                                                            unsigned loopDim) {
  for (auto [idx, expr] : llvm::enumerate(map.getResults())) {
    auto dimExpr = dyn_cast<AffineDimExpr>(expr);
    if (dimExpr && dimExpr.getPosition() == loopDim)
      return idx;
  }
  return std::nullopt;
}

// For a tensor, returns the loop iterator indexing into the last-but-one
// dimension.
static std::optional<unsigned> getNx32RowLoopDim(const TensorInfo &info) {
  auto results = info.indexingMap.getResults();
  if (static_cast<int64_t>(results.size()) != info.type.getRank())
    return std::nullopt;
  auto prevExpr = dyn_cast<AffineDimExpr>(results[results.size() - 2]);
  if (!prevExpr)
    return std::nullopt;
  return prevExpr.getPosition();
}

struct VectorizePackedGenericPattern
    : public OpRewritePattern<linalg::GenericOp> {
  bool enableLoopSplitting;
  bool eagerVectorization;

  VectorizePackedGenericPattern(MLIRContext *ctx, bool enableLoopSplitting,
                                bool eagerVectorization)
      : OpRewritePattern(ctx), enableLoopSplitting(enableLoopSplitting),
        eagerVectorization(eagerVectorization) {}

private:
  struct RepackedNx32Info {
    Value tensor;
    unsigned rowLoopDim = 0;
  };

  static std::optional<vector::CombiningKind>
  getCombiningKindFromYield(linalg::GenericOp op) {
    auto yield = dyn_cast<linalg::YieldOp>(op.getBody()->getTerminator());
    if (!yield || yield.getNumOperands() != 1)
      return std::nullopt;
    Operation *def = yield.getOperand(0).getDefiningOp();
    if (!def)
      return std::nullopt;
    if (isa<arith::AddFOp>(def) || isa<arith::AddIOp>(def))
      return vector::CombiningKind::ADD;
    if (isa<arith::MulFOp>(def) || isa<arith::MulIOp>(def))
      return vector::CombiningKind::MUL;
    if (isa<arith::AndIOp>(def))
      return vector::CombiningKind::AND;
    if (isa<arith::OrIOp>(def))
      return vector::CombiningKind::OR;
    if (isa<arith::XOrIOp>(def))
      return vector::CombiningKind::XOR;
    if (isa<arith::MaximumFOp>(def))
      return vector::CombiningKind::MAXIMUMF;
    if (isa<arith::MinimumFOp>(def))
      return vector::CombiningKind::MINIMUMF;
    return std::nullopt;
  }

  // Checks if yield transitively uses output block argument - true
  // for reductions (in which output is read and modified and yielded)
  static bool yieldUsesOutputBlockArgument(linalg::GenericOp op) {
    Block &body = op.getRegion().front();
    unsigned outArgIdx = op.getNumDpsInputs();
    if (outArgIdx >= body.getNumArguments() || (op.getNumDpsInits() != 1))
      return false;
    Value outArg = body.getArgument(outArgIdx);
    auto yield = dyn_cast<linalg::YieldOp>(body.getTerminator());
    if (!yield || yield.getNumOperands() != 1)
      return false;

    SmallVector<Value> worklist{yield.getOperand(0)}; // Yielded value
    DenseSet<Value> visited;
    while (!worklist.empty()) {
      Value value = worklist.pop_back_val();
      if (!visited.insert(value).second) // Already visited?
        continue;
      if (value == outArg) // Found a use of the first output argument
        return true;
      Operation *def = value.getDefiningOp();
      if (!def ||
          def->getBlock() != &body) // Not an op, or not defined in this block?
        continue;
      worklist.append(def->operand_begin(), def->operand_end());
    }
    return false;
  }

  /// Collapse all packed inputs (...x32x2 -> ...x64) by merging the last two
  /// dimensions. Returns the collapsed tensor values and their permutation maps
  /// for vector transfer ops.
  /// Collapse the trailing two dimensions of a packed tensor (e.g. [..., 32,
  /// 2]) into a single dimension of size 64, producing [..., 64]. Returns the
  /// collapsed Value; also populates \p reassociation with the indices used
  /// (useful for the caller to later expand back).
  static Value collapseTail2To64(
      OpBuilder &b, Location loc, Value tensor, RankedTensorType tensorTy,
      SmallVectorImpl<ReassociationIndices> *reassociation = nullptr) {
    int64_t rank = tensorTy.getRank();
    SmallVector<ReassociationIndices> reassoc;
    reassoc.reserve(rank - 1);
    for (int64_t r = 0; r < rank - 2; ++r)
      reassoc.push_back({r});
    reassoc.push_back({rank - 2, rank - 1});

    SmallVector<int64_t> collapsedShape(tensorTy.getShape().begin(),
                                        tensorTy.getShape().end() - 2);
    collapsedShape.push_back(64);
    auto collapsedTy =
        RankedTensorType::get(collapsedShape, tensorTy.getElementType());

    Value collapsed =
        tensor::CollapseShapeOp::create(b, loc, collapsedTy, tensor, reassoc);
    if (reassociation) {
      reassociation->clear();
      *reassociation = std::move(reassoc);
    }
    return collapsed;
  }

  static void
  collapsePackedInputs(linalg::GenericOp op,
                       const VectorizationStrategy &strategy, Type elementType,
                       PatternRewriter &rewriter, Location loc,
                       SmallVectorImpl<Value> &collapsedInputs,
                       SmallVectorImpl<AffineMap> &collapsedInputPermMaps) {
    collapsedInputs.resize(strategy.inputs.size());
    collapsedInputPermMaps.resize(strategy.inputs.size());
    for (int i = 0; i < (int)strategy.inputs.size(); ++i) {
      const TensorInfo &info = strategy.inputs[i];
      if (!info.hasPackedLayout()) {
        // Non-packed input: keep as-is (will be scalar-extracted + broadcast).
        collapsedInputs[i] = op.getDpsInputOperand(i)->get();
        collapsedInputPermMaps[i] = AffineMap();
        continue;
      }
      RankedTensorType inTy = info.type;
      collapsedInputs[i] = collapseTail2To64(
          rewriter, loc, op.getDpsInputOperand(i)->get(), inTy);
      collapsedInputPermMaps[i] = getTrailingVectorPermutationMap(
          rewriter,
          cast<RankedTensorType>(collapsedInputs[i].getType()).getRank());
    }
  }

  /// Extract output loop dimensions from the output indexing map, validate that
  /// the last dim is packedInnerDim (which will be unrolled, not looped), pop
  /// it off, and compute the loop extents from the output tensor shape. Returns
  /// failure if the output map is not a simple dim projection or if the last
  /// dim is not packedInnerDim.
  static LogicalResult
  buildOutLoopDimsAndExtents(const TensorInfo &outputInfo,
                             unsigned packedInnerDim,
                             SmallVectorImpl<unsigned> &outLoopDims,
                             SmallVectorImpl<int64_t> &outLoopExtents) {
    RankedTensorType outTy = outputInfo.type;
    if (!outTy.hasStaticShape())
      return failure();
    ArrayRef<int64_t> outShape = outTy.getShape();

    AffineMap outMap = outputInfo.indexingMap;
    for (AffineExpr expr : outMap.getResults()) {
      auto dimExpr = dyn_cast<AffineDimExpr>(expr);
      if (!dimExpr)
        return failure();
      outLoopDims.push_back(dimExpr.getPosition());
    }
    if (outLoopDims.empty())
      return failure();

    // The last output dim must be the packed-inner dim (size 2), which we
    // handle by unrolling (reducing even/odd halves separately).
    if (outLoopDims.back() != packedInnerDim)
      return failure();

    // Remove packedInnerDim from outLoopDims — it's handled by unrolling,
    // not by a loop.
    outLoopDims.pop_back();

    // Build the loop extents for the output loops (excluding the packed-inner
    // dim which is unrolled).
    for (unsigned i = 0; i < outLoopDims.size(); ++i)
      outLoopExtents.push_back(outShape[i]);

    return success();
  }

  /// Derive loop extents for the explicit reduction dimensions (excluding the
  /// packed-outer dim which is handled by lane deinterleaving) from the primary
  /// input's indexing map and shape.
  static SmallVector<int64_t>
  buildRedLoopExtents(const TensorInfo &primaryInputInfo,
                      ArrayRef<unsigned> redDims) {
    RankedTensorType primaryInputTy = primaryInputInfo.type;
    SmallVector<int64_t> redLoopExtents;
    for (unsigned rd : redDims) {
      auto results = primaryInputInfo.indexingMap.getResults();
      int64_t ext = -1;
      for (auto [idx, expr] : llvm::enumerate(results)) {
        auto dimExpr = dyn_cast<AffineDimExpr>(expr);
        if (dimExpr && dimExpr.getPosition() == rd) {
          ext = primaryInputTy.getShape()[idx];
          break;
        }
      }
      if (ext < 0)
        llvm_unreachable("Expecting static shape for input");
      redLoopExtents.push_back(ext);
    }
    return redLoopExtents;
  }

  /// Create a splat vector<64> constant filled with the identity element for
  /// the given combining kind:
  ///   ADD:      0.0
  ///   MUL:      1.0
  ///   MAXIMUMF: -inf
  ///   MINIMUMF: +inf
  static Value createIdentityVector(OpBuilder &b, Location loc,
                                    VectorType vec64Ty, Type elementType,
                                    vector::CombiningKind kind) {
    TypedAttr identityAttr;
    if (auto floatTy = dyn_cast<FloatType>(elementType)) {
      switch (kind) {
      case vector::CombiningKind::ADD:
        identityAttr = b.getFloatAttr(elementType, 0.0);
        break;
      case vector::CombiningKind::MUL:
        identityAttr = b.getFloatAttr(elementType, 1.0);
        break;
      case vector::CombiningKind::MAXIMUMF:
        identityAttr = b.getFloatAttr(
            elementType,
            APFloat::getInf(floatTy.getFloatSemantics(), /*Negative=*/true));
        break;
      case vector::CombiningKind::MINIMUMF:
        identityAttr = b.getFloatAttr(
            elementType,
            APFloat::getInf(floatTy.getFloatSemantics(), /*Negative=*/false));
        break;
      default:
        llvm_unreachable("unsupported CombiningKind for createIdentityVector");
      }
    } else {
      llvm_unreachable(
          "unsupported non-float element type for createIdentityVector");
    }
    auto identityVecAttr = DenseElementsAttr::get(vec64Ty, identityAttr);
    return arith::ConstantOp::create(b, loc, vec64Ty, identityVecAttr);
  }

  /// Resolve a single affine map result (at position \p r) to its loop IV,
  /// searching \p outLoopDims / \p currentOuterIVs first, then
  /// \p redDims / \p currentRedIVs. Returns a null Value on failure (e.g. the
  /// result is not a plain AffineDimExpr, or the dim is not found in either
  /// loop-dim list).
  static Value resolveMapResultToIV(AffineMap map, unsigned r,
                                    ArrayRef<unsigned> outLoopDims,
                                    ArrayRef<Value> currentOuterIVs,
                                    ArrayRef<unsigned> redDims,
                                    ArrayRef<Value> currentRedIVs) {
    auto dimExpr = dyn_cast<AffineDimExpr>(map.getResult(r));
    if (!dimExpr)
      return {};
    unsigned pos = dimExpr.getPosition();
    for (unsigned j = 0; j < outLoopDims.size(); ++j)
      if (outLoopDims[j] == pos)
        return currentOuterIVs[j];
    for (unsigned j = 0; j < redDims.size(); ++j)
      if (redDims[j] == pos)
        return currentRedIVs[j];
    return {};
  }

  /// Build indices into a packed input's collapsed view given outer loop IVs
  /// and reduction IVs. The collapsed input has shape [..., 64] where the last
  /// dim is the collapsed 32x2 packed tail.
  /// Returns an empty vector on failure.
  static SmallVector<Value> buildCollapsedInputIndices(
      const TensorInfo &info, ArrayRef<unsigned> outLoopDims,
      ArrayRef<Value> currentOuterIVs, ArrayRef<unsigned> redDims,
      ArrayRef<Value> currentRedIVs, Value c0) {
    AffineMap map = info.indexingMap;
    SmallVector<Value> indices;
    for (int64_t r = 0; r < map.getNumResults() - 2; ++r) {
      Value iv = resolveMapResultToIV(map, r, outLoopDims, currentOuterIVs,
                                      redDims, currentRedIVs);
      if (!iv)
        return {};
      indices.push_back(iv);
    }
    indices.push_back(c0); // vectorized dim
    return indices;
  }

  /// Shared context for the three recursive loop-building helpers below.
  struct ReductionContext {
    linalg::GenericOp op;
    const VectorizationStrategy &strategy;
    Location loc;
    Type elementType;
    VectorType vec64Ty;
    vector::CombiningKind kind;
    Value c0, c1;
    int primaryInputIdx;
    SmallVector<unsigned> outLoopDims;
    SmallVector<int64_t> outLoopExtents;
    SmallVector<unsigned> redDims;
    SmallVector<int64_t> redLoopExtents;
    SmallVector<Value> collapsedInputs;
    SmallVector<AffineMap> collapsedInputPermMaps;
    bool eagerVectorization = false;
  };

  /// Innermost reduction body: loads all packed input vectors, clones the
  /// generic body ops, deinterleaves even/odd lanes, reduces each to a scalar,
  /// and inserts both scalars into the output tensor.
  /// outerIVs[i] corresponds to ctx.outLoopDims[i].
  /// redIVs[i] corresponds to ctx.redDims[i].
  static Value buildReductionBody(ReductionContext &ctx, OpBuilder &b,
                                  Location loc, ArrayRef<Value> outerIVs,
                                  MutableArrayRef<Value> redIVs,
                                  Value currentOutTensor) {
    // Initialize the vector accumulator with the identity element.
    Value acc0 =
        createIdentityVector(b, loc, ctx.vec64Ty, ctx.elementType, ctx.kind);
    Value finalVec = buildRedLoops(ctx, b, loc, 0, acc0, outerIVs, redIVs);
    if (!finalVec)
      return Value();

    // Unroll the packed-inner dim (size 2): reduce even half and store at
    // output index [..., 0], then reduce odd half and store at [..., 1].
    Value updatedTensor = currentOutTensor;
    for (int isOdd = 0; isOdd < 2; ++isOdd) {
      SmallVector<Value> outIndices(outerIVs.begin(), outerIVs.end());
      outIndices.push_back(isOdd == 0 ? ctx.c0 : ctx.c1);
      Value reduced = reduceInterleavedVectorHalf(b, loc, finalVec,
                                                  ctx.elementType, ctx.kind,
                                                  /*isEven=*/isOdd == 0);
      // Insert the reduced scalar directly — no read-modify-write needed.
      updatedTensor =
          tensor::InsertOp::create(b, loc, reduced, updatedTensor, outIndices);
    }
    return updatedTensor;
  }

  /// Clone body ops whose operands are all available in the value mapping.
  ///
  /// Iterates until no more progress can be made.  This enables partial
  /// body evaluation after each input-loading pass so that ops depending
  /// only on scalar inputs (e.g. math.rsqrt) are emitted before
  /// repacked/packed tensor reads.
  /// Clone all body ops whose operands are ready, in data-flow order.
  /// Returns false if any op fails to clone.
  static bool cloneReadyBodyOps(Block *body,
                                DenseSet<Operation *> &clonedBodyOps,
                                DenseMap<Value, ValueRep> &valueMap,
                                OpBuilder &b, VectorType vectorType,
                                Type elementType,
                                bool eagerVectorization = false) {
    bool progress = true;
    while (progress) {
      progress = false;
      for (Operation &bodyOp : body->without_terminator()) {
        if (clonedBodyOps.count(&bodyOp))
          continue;
        // Check if all operands are available.
        bool ready = true;
        for (Value operand : bodyOp.getOperands()) {
          if (auto blockArg = dyn_cast<BlockArgument>(operand)) {
            auto it = valueMap.find(blockArg);
            if (it == valueMap.end() || !it->second.hasAny()) {
              ready = false;
              break;
            }
          } else if (Operation *defOp = operand.getDefiningOp()) {
            if (defOp->getBlock() == body && !clonedBodyOps.count(defOp)) {
              ready = false;
              break;
            }
          }
        }
        if (!ready)
          continue;
        if (!cloneOpWithRepresentations(bodyOp, valueMap, b, vectorType,
                                        elementType, eagerVectorization))
          return false;
        clonedBodyOps.insert(&bodyOp);
        progress = true;
      }
    }
    return true;
  }

  /// Recursively builds the explicit reduction loop nest starting at
  /// ctx.redDims[redIdx]. vecAcc is the vector<64> accumulator carried into
  /// this level. outerIVs are the already-bound output loop IVs; redIVs is a
  /// pre-allocated array that is filled slot-by-slot as recursion descends.
  /// At the base case (redIdx == ctx.redDims.size()) loads all packed input
  /// vectors, clones the generic body ops, and returns the updated accumulator.
  /// Returns a null Value on failure.
  static Value buildRedLoops(ReductionContext &ctx, OpBuilder &b, Location loc,
                             int redIdx, Value vecAcc, ArrayRef<Value> outerIVs,
                             MutableArrayRef<Value> redIVs) {
    if (redIdx == (int)ctx.redDims.size()) {
      // Innermost: load all packed input vectors and clone body ops,
      // and return the result vector.
      SmallVector<Value> primaryIdx = buildCollapsedInputIndices(
          ctx.strategy.inputs[ctx.primaryInputIdx], ctx.outLoopDims, outerIVs,
          ctx.redDims, redIVs, ctx.c0);
      if (primaryIdx.empty())
        return Value();

      Value vec = vector::TransferReadOp::create(
          b, loc, ctx.vec64Ty, ctx.collapsedInputs[ctx.primaryInputIdx],
          primaryIdx, /*padding=*/std::nullopt,
          ctx.collapsedInputPermMaps[ctx.primaryInputIdx],
          /*inBounds=*/ArrayRef<bool>{true});

      Block *body = ctx.op.getBody();
      DenseMap<Value, ValueRep> valueMap;
      DenseSet<Operation *> clonedBodyOps;
      valueMap[body->getArgument(ctx.primaryInputIdx)].vector = vec;

      for (int i = 0; i < (int)ctx.strategy.inputs.size(); ++i) {
        if (i == ctx.primaryInputIdx)
          continue;
        const TensorInfo &info = ctx.strategy.inputs[i];
        if (info.hasPackedLayout()) {
          // Packed input: load as vector<64> from collapsed tensor.
          SmallVector<Value> idx = buildCollapsedInputIndices(
              info, ctx.outLoopDims, outerIVs, ctx.redDims, redIVs, ctx.c0);
          if (idx.empty())
            return Value();
          Value otherVec = vector::TransferReadOp::create(
              b, loc, ctx.vec64Ty, ctx.collapsedInputs[i], idx,
              /*padding=*/std::nullopt, ctx.collapsedInputPermMaps[i],
              /*inBounds=*/ArrayRef<bool>{true});
          valueMap[body->getArgument(i)].vector = otherVec;
        } else {
          // Non-packed input: extract scalar, broadcast to vector<64>.
          AffineMap map = info.indexingMap;
          SmallVector<Value> scalarIdx;
          for (unsigned r = 0; r < map.getNumResults(); ++r) {
            Value iv = resolveMapResultToIV(map, r, ctx.outLoopDims, outerIVs,
                                            ctx.redDims, redIVs);
            if (!iv)
              return Value();
            scalarIdx.push_back(iv);
          }
          Value scalar = tensor::ExtractOp::create(
              b, loc, ctx.collapsedInputs[i], scalarIdx);
          Value broadcasted =
              vector::BroadcastOp::create(b, loc, ctx.vec64Ty, scalar);
          valueMap[body->getArgument(i)].vector = broadcasted;
        }
      }

      unsigned outArgIdx = ctx.op.getNumDpsInputs();
      valueMap[body->getArgument(outArgIdx)].vector = vecAcc;

      // Clone body ops following the actual data flow.
      if (!cloneReadyBodyOps(body, clonedBodyOps, valueMap, b, ctx.vec64Ty,
                             ctx.elementType, ctx.eagerVectorization))
        return Value();

      auto yieldOp = cast<linalg::YieldOp>(body->getTerminator());
      auto yieldIt = valueMap.find(yieldOp.getOperand(0));
      if (yieldIt == valueMap.end() || !yieldIt->second.hasAny())
        return Value();
      return materializeVector(yieldIt->second, b, loc, ctx.vec64Ty,
                               ctx.elementType);
    }

    // Build a for loop for redDims[redIdx] and recurse.
    // Recursive case. Create a loop, and call buildRelLoops again.
    int64_t ext = ctx.redLoopExtents[redIdx];
    Value ub = arith::ConstantIndexOp::create(b, loc, ext);
    bool redFailed = false;
    auto forOp = scf::ForOp::create(
        b, loc, ctx.c0, ub, ctx.c1, ValueRange{vecAcc},
        [&](OpBuilder &innerB, Location innerLoc, Value iv,
            ValueRange innerArgs) {
          redIVs[redIdx] = iv;
          Value result = buildRedLoops(ctx, innerB, innerLoc, redIdx + 1,
                                       innerArgs[0], outerIVs, redIVs);
          if (!result) {
            result = innerArgs[0];
            redFailed = true;
          }
          scf::YieldOp::create(innerB, innerLoc, result);
        });
    if (redFailed)
      return Value();
    // return the value produced by the for op.
    return forOp.getResult(0);
  }

  /// Recursively builds the outer (output) loop nest over ctx.outLoopDims,
  /// starting at ctx.outLoopDims[dimIdx]. currentTensor is the output tensor
  /// carried as an scf.for iter-arg through each loop level. outerIVs is
  /// appended with each new IV as recursion descends and popped on the way
  /// back up. At the base case (dimIdx == ctx.outLoopDims.size()) all output
  /// IVs are bound and buildReductionBody is called to emit the vector
  /// accumulation and scalar insert. Returns a null Value on failure.
  static Value buildOuterLoops(ReductionContext &ctx, OpBuilder &b,
                               Location loc, int dimIdx, Value currentTensor,
                               SmallVectorImpl<Value> &outerIVs) {
    if (dimIdx == (int)ctx.outLoopDims.size()) {
      // All outer IVs collected; emit the reduction body.
      // Base case.
      SmallVector<Value> redIVs(ctx.redDims.size());
      return buildReductionBody(ctx, b, loc, outerIVs, redIVs, currentTensor);
    }

    int64_t ext = ctx.outLoopExtents[dimIdx];
    Value ub = arith::ConstantIndexOp::create(b, loc, ext);
    bool outerFailed = false;
    auto forOp = scf::ForOp::create(
        b, loc, ctx.c0, ub, ctx.c1, ValueRange{currentTensor},
        [&](OpBuilder &innerB, Location innerLoc, Value iv,
            ValueRange innerArgs) {
          outerIVs.push_back(iv);
          Value result = buildOuterLoops(ctx, innerB, innerLoc, dimIdx + 1,
                                         innerArgs[0], outerIVs);
          outerIVs.pop_back();
          if (!result) {
            result = innerArgs[0];
            outerFailed = true;
          }
          scf::YieldOp::create(innerB, innerLoc, result);
        });
    if (outerFailed)
      return Value();
    return forOp.getResult(0);
  }

  // Vectorize reductions from packed [...,32,2] inputs into a non-packed output
  // by accumulating in a vector<64> register and then reducing even/odd lanes
  // to scalar outputs.
  // The body computation is derived from the actual generic op body via
  // cloneOpWithRepresentations, so this handles any reduction body.
  // Scalar-only subexpressions (inputs that don't vary across the
  // packed dimensions) are kept scalar throughout to avoid unnecessary
  // broadcast/vector computation.

  static LogicalResult vectorizePackedReductionToScalarOutput(
      linalg::GenericOp op, const VectorizationStrategy &strategy,
      PatternRewriter &rewriter, bool eagerVectorization) {
    // Only handle single output for now.
    if (op.getNumDpsInits() != 1)
      return failure();

    auto kindOpt = getCombiningKindFromYield(op);
    if (!kindOpt)
      return failure();
    if (!yieldUsesOutputBlockArgument(op))
      return failure();

    const TensorInfo &outputInfo = strategy.outputs.front();
    if (outputInfo.hasPackedLayout())
      return failure();

    // Find the primary packed input.
    int primaryInputIdx = -1;
    for (int i = 0; i < (int)strategy.inputs.size(); ++i) {
      if (strategy.inputs[i].hasPackedLayout()) {
        primaryInputIdx = i;
        break;
      }
    }
    if (primaryInputIdx < 0)
      return failure();

    const TensorInfo &primaryInputInfo = strategy.inputs[primaryInputIdx];
    RankedTensorType primaryInputTy = primaryInputInfo.type;
    RankedTensorType outTy = outputInfo.type;
    if (primaryInputTy.getRank() < 2 || outTy.getRank() < 1)
      return failure();

    // Currently assume the packed tail is exactly 32x2.
    ArrayRef<int64_t> primaryInShape = primaryInputTy.getShape();
    if (primaryInShape[primaryInputTy.getRank() - 2] != 32 ||
        primaryInShape.back() != 2)
      return failure();

    Location loc = op.getLoc();
    Type elementType = primaryInputTy.getElementType();
    if (!elementType.isF16())
      return failure();

    // Identify reduction dims; packed-outer (size 32) is handled by lane
    // deinterleaving, so remove it from the explicit reduction loop dims.
    SmallVector<unsigned> redDims = strategy.reductionDims;
    if (redDims.empty())
      return failure();
    unsigned packedOuterDim = strategy.numLoops - 2;
    unsigned packedInnerDim = strategy.numLoops - 1;
    if (!llvm::is_contained(redDims, packedOuterDim))
      return failure();
    if (llvm::is_contained(redDims, packedInnerDim))
      return failure();
    llvm::erase(redDims, packedOuterDim);
    if (!primaryInputTy.hasStaticShape())
      return failure();

    // Build output loop dims/extents (packedInnerDim is unrolled, not looped).
    SmallVector<unsigned> outLoopDims;
    SmallVector<int64_t> outLoopExtents;
    if (failed(buildOutLoopDimsAndExtents(outputInfo, packedInnerDim,
                                          outLoopDims, outLoopExtents)))
      return failure();

    // Derive reduction loop extents before moving redDims into the context.
    SmallVector<int64_t> redLoopExtents =
        buildRedLoopExtents(primaryInputInfo, redDims);

    // Fill the shared context.
    ReductionContext ctx{op,
                         strategy,
                         loc,
                         elementType,
                         VectorType::get({64}, elementType),
                         *kindOpt,
                         arith::ConstantIndexOp::create(rewriter, loc, 0),
                         arith::ConstantIndexOp::create(rewriter, loc, 1),
                         primaryInputIdx,
                         std::move(outLoopDims),
                         std::move(outLoopExtents),
                         std::move(redDims),
                         std::move(redLoopExtents),
                         /*collapsedInputs=*/{},
                         /*collapsedInputPermMaps=*/{},
                         eagerVectorization};

    collapsePackedInputs(op, strategy, elementType, rewriter, loc,
                         ctx.collapsedInputs, ctx.collapsedInputPermMaps);

    Value initOut = op.getDpsInitOperand(0)->get();
    SmallVector<Value> outerIVs;
    outerIVs.reserve(ctx.outLoopDims.size());
    Value finalResult =
        buildOuterLoops(ctx, rewriter, loc, 0, initOut, outerIVs);
    if (!finalResult)
      return failure();

    if (auto *defOp = finalResult.getDefiningOp())
      defOp->setAttr("hexagon.packed_vectorized",
                     UnitAttr::get(op->getContext()));
    rewriter.replaceOp(op, finalResult);
    return success();
  }

  /// Collapse or repack each input tensor for the vectorized op.
  /// Packed inputs have their trailing 2 dims collapsed into 64.
  /// Non-packed Nx32 inputs accessed via packedOuterDim are repacked
  /// (expand + collapse) into (N/2)x64.
  /// Other non-packed inputs are kept as-is.
  /// Returns failure() if any Nx32 repack cannot determine the row loop dim.
  static LogicalResult collapseOrRepackInputs(
      OpBuilder &rewriter, Location loc, const VectorizationStrategy &strategy,
      SmallVectorImpl<Value> &collapsedInputs,
      SmallVectorImpl<bool> &isInputPacked,
      SmallVectorImpl<std::optional<RepackedNx32Info>> &repackedInputs) {
    unsigned packedOuterDim = strategy.numLoops - 2;
    for (const auto &inputInfo : strategy.inputs) {
      if (inputInfo.hasPackedLayout()) {
        // Collapse this input
        Value collapsed =
            collapseTail2To64(rewriter, loc, inputInfo.tensor, inputInfo.type);
        collapsedInputs.push_back(collapsed);
        isInputPacked.push_back(true);
        repackedInputs.push_back(std::nullopt);
      } else {
        // Repack Nx32 tensors indexed by packedOuterDim into (N/2)x64.
        if (isNx32PackedOuterAccess(inputInfo, packedOuterDim)) {
          ArrayRef<int64_t> shape = inputInfo.type.getShape();
          int64_t srcRank = inputInfo.type.getRank();
          int64_t prev = shape[shape.size() - 2];

          // Expand: [..., prev, 32] -> [..., prev/2, 2, 32]
          // tensor.expand_shape reassociation has size == source rank.
          // Source dim (srcRank-2) expands to (srcRank-2, srcRank-1).
          // Source dim (srcRank-1) maps to last result dim (srcRank).

          SmallVector<ReassociationIndices> expandReassoc;
          expandReassoc.reserve(srcRank);
          for (int64_t i = 0; i < srcRank - 2; ++i)
            expandReassoc.push_back({static_cast<int64_t>(i)});
          expandReassoc.push_back({srcRank - 2, srcRank - 1});
          expandReassoc.push_back({srcRank});

          SmallVector<int64_t> expandedShape(shape.begin(), shape.end() - 2);
          expandedShape.push_back(prev / 2);
          expandedShape.push_back(2);
          expandedShape.push_back(32);
          auto expandedTy = RankedTensorType::get(
              expandedShape, inputInfo.type.getElementType());
          Value expanded = tensor::ExpandShapeOp::create(
              rewriter, loc, expandedTy, inputInfo.tensor, expandReassoc);

          // Collapse: [..., prev/2, 2, 32] -> [..., prev/2, 64]
          // tensor.collapse_shape reassociation has size == result rank.
          SmallVector<ReassociationIndices> repackCollapse;
          repackCollapse.reserve(srcRank);
          for (int64_t i = 0; i < srcRank - 1; ++i)
            repackCollapse.push_back({static_cast<int64_t>(i)});
          repackCollapse.push_back({srcRank - 1, srcRank});

          SmallVector<int64_t> collapsedShape2(expandedShape.begin(),
                                               expandedShape.end() - 2);
          collapsedShape2.push_back(64);
          auto repackedTy = RankedTensorType::get(
              collapsedShape2, inputInfo.type.getElementType());
          Value repacked = tensor::CollapseShapeOp::create(
              rewriter, loc, repackedTy, expanded, repackCollapse);

          collapsedInputs.push_back(repacked);
          isInputPacked.push_back(false);
          auto rowLoopDim = getNx32RowLoopDim(inputInfo);
          if (!rowLoopDim)
            return failure();
          repackedInputs.push_back(RepackedNx32Info{repacked, *rowLoopDim});
        } else {
          // Keep other non-packed inputs as-is.
          collapsedInputs.push_back(inputInfo.tensor);
          isInputPacked.push_back(false);
          repackedInputs.push_back(std::nullopt);
        }
      }
    }
    return success();
  }

  /// Shared context for vectorization helper functions.
  ///
  /// Bundles the op, strategy, collapsed inputs, and constants that are
  /// threaded through every helper so that function signatures stay concise.
  struct VectorizeContext {
    linalg::GenericOp op;
    const VectorizationStrategy &strategy;
    RankedTensorType outputType;
    int64_t rank;
    Value c0;
    Value c1;
    ArrayRef<Value> collapsedInputs;
    ArrayRef<bool> isInputPacked;
    ArrayRef<std::optional<RepackedNx32Info>> repackedInputs;
    bool eagerVectorization = false;
  };

  /// Load a single non-packed, non-repacked (scalar/broadcast) input,
  /// indexed by 'i'. Indices for accessing the input can be obtained
  /// from originalIVs and the input map.
  /// Uniform scalar inputs stay scalar so scalar-only subexpressions
  /// can also stay scalar.  Inputs that depend on packedInnerDim still
  /// materialise a lane-alternating even/odd pair because they vary
  /// across the collapsed 64-lane packed layout.
  ///
  /// Returns true on success, false on failure.
  static bool loadScalarInput(const VectorizeContext &ctx, OpBuilder &b,
                              Location loc, size_t i,
                              ArrayRef<Value> originalIVs,
                              SmallVectorImpl<ValueRep> &loadedInputs) {
    const auto &strategy = ctx.strategy;
    Value c0 = ctx.c0;
    Value c1 = ctx.c1;
    Value input = ctx.collapsedInputs[i];
    const TensorInfo &inputInfo = strategy.inputs[i];
    unsigned packedOuterDim = strategy.numLoops - 2;
    unsigned packedInnerDim = strategy.numLoops - 1;

    bool usesPackedOuter = usesLoopDim(inputInfo.indexingMap, packedOuterDim);
    bool usesPackedInner = usesLoopDim(inputInfo.indexingMap, packedInnerDim);
    if (usesPackedOuter) // This isn't really scalar!
      return false;      // signals failure

    SmallVector<Value> inputIndices;
    AffineMap inputMap = inputInfo.indexingMap;
    std::optional<unsigned> packedInnerResultPos;
    if (usesPackedInner) // This input can be treated as a scalar pair
      packedInnerResultPos =
          getPackedInnerResultPosition(inputMap, packedInnerDim);

    for (int64_t dim = 0; dim < inputMap.getNumResults(); ++dim) {
      auto expr = inputMap.getResult(dim);
      if (auto dimExpr = dyn_cast<AffineDimExpr>(expr)) {
        if (dimExpr.getPosition() == packedInnerDim)
          continue;
        inputIndices.push_back(originalIVs[dimExpr.getPosition()]);
      }
    }

    if (usesPackedInner) {
      if (!packedInnerResultPos)
        return false;
      SmallVector<Value> evenIndices(inputIndices.begin(), inputIndices.end());
      evenIndices.insert(evenIndices.begin() + *packedInnerResultPos, c0);
      Value scalar0 = tensor::ExtractOp::create(b, loc, input, evenIndices);
      SmallVector<Value> oddIndices(inputIndices.begin(), inputIndices.end());
      oddIndices.insert(oddIndices.begin() + *packedInnerResultPos, c1);
      Value scalar1 = tensor::ExtractOp::create(b, loc, input, oddIndices);
      loadedInputs[i].even = scalar0;
      loadedInputs[i].odd = scalar1;
    } else {
      Value scalar = tensor::ExtractOp::create(b, loc, input, inputIndices);
      loadedInputs[i].scalar = scalar;
    }
    return true;
  }

  /// Load all repacked Nx32 inputs (used in Pass 2 of vectorizeOp below).
  ///
  /// If preloadedRepackedVecs is non-empty, uses the pre-loaded vectors
  /// directly (split-plan unrolling).  Otherwise computes div/rem on the
  /// row loop IV to index the repacked tensor and selects the lower or
  /// upper half.
  ///
  /// Returns false on failure (e.g. non-AffineDimExpr in the indexing map).
  static bool loadRepackedInputs(const VectorizeContext &ctx, OpBuilder &b,
                                 Location loc, ArrayRef<Value> originalIVs,
                                 VectorType vectorType,
                                 ArrayRef<Value> preloadedRepackedVecs,
                                 SmallVectorImpl<ValueRep> &loadedInputs) {
    const auto &strategy = ctx.strategy;
    Value c0 = ctx.c0;
    ArrayRef<std::optional<RepackedNx32Info>> repackedInputs =
        ctx.repackedInputs;

    for (size_t i = 0; i < ctx.collapsedInputs.size(); ++i) {
      if (!repackedInputs[i]) // Not a repacked input, skip
        continue;
      Value dup64;
      if (!preloadedRepackedVecs.empty() && preloadedRepackedVecs[i]) {
        // Use the pre-loaded vector (already selected even/odd half).
        dup64 = preloadedRepackedVecs[i];
      } else {
        const TensorInfo &inputInfo = strategy.inputs[i];
        unsigned rowDim = repackedInputs[i]->rowLoopDim;
        Value row = originalIVs[rowDim];
        Value c2 = arith::ConstantIndexOp::create(b, loc, 2);
        Value rowOuter = arith::DivUIOp::create(b, loc, row, c2);
        Value rowInner = arith::RemUIOp::create(b, loc, row, c2);

        // Indices for repacked tensor [..., prev/2, 64].
        SmallVector<Value> repackedIdx;
        AffineMap map = inputInfo.indexingMap;
        for (int64_t r = 0; r + 2 < map.getNumResults(); ++r) {
          auto dimExpr = dyn_cast<AffineDimExpr>(map.getResult(r));
          if (!dimExpr)
            return false;
          repackedIdx.push_back(originalIVs[dimExpr.getPosition()]);
        }
        repackedIdx.push_back(rowOuter);
        repackedIdx.push_back(c0);

        AffineMap repackedPermMap = getTrailingVectorPermutationMapForTensor(
            b, repackedInputs[i]->tensor);
        Value vec64 = vector::TransferReadOp::create(
            b, loc, vectorType, repackedInputs[i]->tensor, repackedIdx,
            /*padding=*/std::nullopt, repackedPermMap,
            /*inBounds=*/ArrayRef<bool>{true});
        Value selectUpper = arith::CmpIOp::create(
            b, loc, arith::CmpIPredicate::ne, rowInner, c0);
        dup64 = duplicateLowerOrUpperHalf64(b, loc, vec64, selectUpper);
      }
      loadedInputs[i].vector = dup64;
    }
    return true;
  }

  /// Load all packed inputs (Pass 3).
  ///
  /// Packed tensors are collapsed from [..., 32, 2] to [..., 64].
  /// Builds indices by dropping the original trailing two map results
  /// and replacing them with a single zero index into the vectorised
  /// dimension.
  static void loadPackedInputs(const VectorizeContext &ctx, OpBuilder &b,
                               Location loc, ArrayRef<Value> originalIVs,
                               VectorType vectorType,
                               SmallVectorImpl<ValueRep> &loadedInputs) {
    const auto &strategy = ctx.strategy;
    Value c0 = ctx.c0;
    ArrayRef<Value> collapsedInputs = ctx.collapsedInputs;
    ArrayRef<bool> isInputPacked = ctx.isInputPacked;

    for (size_t i = 0; i < collapsedInputs.size(); ++i) {
      if (!isInputPacked[i])
        continue;
      Value input = collapsedInputs[i];
      const TensorInfo &inputInfo = strategy.inputs[i];
      SmallVector<Value> inputIndices;
      AffineMap inputMap = inputInfo.indexingMap;

      for (int64_t dim = 0; dim < inputMap.getNumResults() - 2; ++dim) {
        auto expr = inputMap.getResult(dim);
        if (auto dimExpr = dyn_cast<AffineDimExpr>(expr))
          inputIndices.push_back(originalIVs[dimExpr.getPosition()]);
      }
      inputIndices.push_back(c0);

      AffineMap inputPermMap =
          getTrailingVectorPermutationMapForTensor(b, input);
      Value vec = vector::TransferReadOp::create(
          b, loc, vectorType, input, inputIndices, /*padding=*/std::nullopt,
          inputPermMap, /*inBounds=*/ArrayRef<bool>{true});
      loadedInputs[i].vector = vec;
    }
  }

  /// Emit the vectorized body for one set of loop IVs.
  ///
  /// Loads every input (scalar / repacked-Nx32 / packed), clones the original
  /// body ops in vectorised form, and writes the result back to the
  /// accumulator tensor.
  ///
  /// Returns the updated accumulator tensor, or a null Value on failure.
  static Value emitVectorBody(const VectorizeContext &ctx, OpBuilder &b,
                              Location loc, ValueRange permutedIVs,
                              Value accumulatorTensor,
                              ArrayRef<Value> preloadedRepackedVecs = {}) {
    const auto &strategy = ctx.strategy;
    linalg::GenericOp op = ctx.op;
    RankedTensorType outputType = ctx.outputType;
    int64_t rank = ctx.rank;
    Value c0 = ctx.c0;
    Value c1 = ctx.c1;
    ArrayRef<Value> collapsedInputs = ctx.collapsedInputs;
    ArrayRef<bool> isInputPacked = ctx.isInputPacked;
    ArrayRef<std::optional<RepackedNx32Info>> repackedInputs =
        ctx.repackedInputs;
    SmallVector<Value> originalIVs(strategy.numLoops);
    for (unsigned permutedPos = 0; permutedPos < permutedIVs.size();
         ++permutedPos) {
      unsigned originalDim = strategy.loopPermutation[permutedPos];
      originalIVs[originalDim] = permutedIVs[permutedPos];
    }

    SmallVector<Value> outputIndices;
    for (int64_t dim = 0; dim < rank - 2; ++dim)
      outputIndices.push_back(originalIVs[dim]);
    outputIndices.push_back(c0);

    // Pre-allocate input representations so we can fill by index in
    // any order while delaying vector materialization until required.
    SmallVector<ValueRep> loadedInputs(collapsedInputs.size());
    Type elementType = outputType.getElementType();
    auto vectorType = VectorType::get({64}, elementType);
    AffineMap vectorPermMap = getTrailingVectorPermutationMap(b, rank - 1);

    // Set up body mapping and partial-evaluation infrastructure.
    // We interleave input loading with body cloning so that body ops
    // depending only on scalar inputs (e.g. math.rsqrt) are emitted
    // before repacked/packed tensor reads.
    Block *body = op.getBody();
    DenseMap<Value, ValueRep> valueMap;
    DenseSet<Operation *> clonedBodyOps;

    // Pass 1: Load scalar/broadcast (non-packed, non-repacked) inputs
    // first, then immediately clone any body ops that are ready.
    // This ensures body ops depending only on scalar inputs (e.g.
    // math.rsqrt) are emitted before repacked/packed tensor reads.
    for (size_t i = 0; i < collapsedInputs.size(); ++i) {
      if (isInputPacked[i] || repackedInputs[i])
        continue;
      if (!loadScalarInput(ctx, b, loc, i, originalIVs, loadedInputs))
        return Value();
      valueMap[body->getArgument(i)] = loadedInputs[i];
    }
    cloneReadyBodyOps(body, clonedBodyOps, valueMap, b, vectorType, elementType,
                      ctx.eagerVectorization);

    // Pass 2: Load repacked Nx32 inputs, then clone newly-ready body ops.
    if (!loadRepackedInputs(ctx, b, loc, originalIVs, vectorType,
                            preloadedRepackedVecs, loadedInputs))
      return Value();
    for (size_t i = 0; i < collapsedInputs.size(); ++i) {
      if (repackedInputs[i])
        valueMap[body->getArgument(i)] = loadedInputs[i];
    }
    cloneReadyBodyOps(body, clonedBodyOps, valueMap, b, vectorType, elementType,
                      ctx.eagerVectorization);

    // Pass 3: Load packed inputs last, then clone all remaining body ops.
    loadPackedInputs(ctx, b, loc, originalIVs, vectorType, loadedInputs);
    if (llvm::any_of(loadedInputs,
                     [](const ValueRep &rep) { return !rep.hasAny(); }))
      return Value();
    for (size_t i = 0; i < collapsedInputs.size(); ++i) {
      if (isInputPacked[i])
        valueMap[body->getArgument(i)] = loadedInputs[i];
    }
    if (!cloneReadyBodyOps(body, clonedBodyOps, valueMap, b, vectorType,
                           elementType, ctx.eagerVectorization))
      return Value();

    auto yieldOp = cast<linalg::YieldOp>(body->getTerminator());
    auto yieldIt = valueMap.find(yieldOp.getOperand(0));
    // Find the entry corresponding to the op's result in valueMap
    if (yieldIt == valueMap.end() || !yieldIt->second.hasAny())
      return Value();
    Value resultVec =
        materializeVector(yieldIt->second, b, loc, vectorType, elementType);
    if (!resultVec)
      return Value();
    auto write = vector::TransferWriteOp::create(
        b, loc, resultVec, accumulatorTensor, outputIndices, vectorPermMap);
    return write.getResult();
  }

  /// Pre-load all repacked Nx32 inputs for a split plan.
  ///
  /// For each repacked input, loads the full vector<64> row at the
  /// outermost split index (ivs.back()) and splits it into even
  /// (lower-32-duplicated) and odd (upper-32-duplicated) halves.
  ///
  /// \p preloadedHalves[i]  Filled with {evenHalf, oddHalf} for each
  ///   repacked input; left empty for non-repacked inputs.
  ///
  /// Returns true on success, false if any index expression cannot be
  /// resolved (the caller should fall back to an scf.for loop).
  static bool preloadRepackedInputs(
      const VectorizeContext &ctx, OpBuilder &b, Location loc, ValueRange ivs,
      VectorType vectorType,
      SmallVectorImpl<SmallVector<Value, 2>> &preloadedHalves) {
    const auto &strategy = ctx.strategy;
    Value c0 = ctx.c0;
    ArrayRef<Value> collapsedInputs = ctx.collapsedInputs;
    ArrayRef<std::optional<RepackedNx32Info>> repackedInputs =
        ctx.repackedInputs;
    preloadedHalves.resize(collapsedInputs.size());
    for (size_t i = 0; i < collapsedInputs.size(); ++i) {
      if (!repackedInputs[i])
        continue;
      const TensorInfo &inputInfo = strategy.inputs[i];

      // Build indices for the repacked tensor at rowOuter = ivs.back().
      SmallVector<Value> repackedIdx;
      AffineMap map = inputInfo.indexingMap;
      for (int64_t r = 0; r + 2 < map.getNumResults(); ++r) {
        auto dimExpr = dyn_cast<AffineDimExpr>(map.getResult(r));
        if (!dimExpr)
          return false;
        // Find which permuted IV corresponds to this original dim.
        unsigned pos = dimExpr.getPosition();
        Value iv;
        for (unsigned p = 0; p < ivs.size(); ++p) {
          if (strategy.loopPermutation[p] == pos) {
            iv = ivs[p];
            break;
          }
        }
        if (!iv)
          return false;
        repackedIdx.push_back(iv);
      }
      repackedIdx.push_back(ivs.back()); // rowOuter = ivs.back()
      repackedIdx.push_back(c0);

      AffineMap repackedPermMap = getTrailingVectorPermutationMapForTensor(
          b, repackedInputs[i]->tensor);
      Value vec64 = vector::TransferReadOp::create(
          b, loc, vectorType, repackedInputs[i]->tensor, repackedIdx,
          /*padding=*/std::nullopt, repackedPermMap,
          /*inBounds=*/ArrayRef<bool>{true});
      // Even half: lower 32 lanes duplicated to 64.
      Value falseVal = arith::ConstantOp::create(b, loc, b.getI1Type(),
                                                 b.getBoolAttr(false));
      Value evenHalf = duplicateLowerOrUpperHalf64(b, loc, vec64, falseVal);
      // Odd half: upper 32 lanes duplicated to 64.
      Value trueVal =
          arith::ConstantOp::create(b, loc, b.getI1Type(), b.getBoolAttr(true));
      Value oddHalf = duplicateLowerOrUpperHalf64(b, loc, vec64, trueVal);
      preloadedHalves[i] = {evenHalf, oddHalf};
    }
    return true;
  }

  /// Emit an unrolled split-plan body (splitFactor sequential iterations).
  ///
  /// Pre-loaded even/odd halves are routed to the corresponding iteration.
  /// Returns the final accumulator value, or a null Value on failure.
  static Value
  emitUnrolledSplitBody(const VectorizeContext &ctx, OpBuilder &b, Location loc,
                        ValueRange ivs, Value accumulator, Value splitFactorVal,
                        int64_t splitFactor,
                        ArrayRef<SmallVector<Value, 2>> preloadedHalves) {
    ArrayRef<Value> collapsedInputs = ctx.collapsedInputs;
    ArrayRef<std::optional<RepackedNx32Info>> repackedInputs =
        ctx.repackedInputs;
    Value currentAccumulator = accumulator;
    for (int64_t splitInner = 0; splitInner < splitFactor; ++splitInner) {
      // Adjust the split dim IV: adjustedIVs.back() = ivs.back() *
      // splitFactor + splitInner.
      SmallVector<Value> adjustedIVs(ivs.begin(), ivs.end());
      Value splitInnerVal = arith::ConstantIndexOp::create(b, loc, splitInner);
      Value scaledOuter =
          arith::MulIOp::create(b, loc, ivs.back(), splitFactorVal);
      adjustedIVs.back() =
          arith::AddIOp::create(b, loc, scaledOuter, splitInnerVal);

      // Build the pre-loaded repacked vector map for this split offset.
      SmallVector<Value> preloadedForThisIter(collapsedInputs.size());
      for (size_t i = 0; i < collapsedInputs.size(); ++i) {
        if (!repackedInputs[i] || preloadedHalves[i].empty())
          continue;
        preloadedForThisIter[i] = preloadedHalves[i][splitInner];
      }

      Value updated = emitVectorBody(ctx, b, loc, adjustedIVs,
                                     currentAccumulator, preloadedForThisIter);
      if (!updated)
        return Value();
      currentAccumulator = updated;
    }
    return currentAccumulator;
  }

  static LogicalResult vectorizeOp(linalg::GenericOp op,
                                   const VectorizationStrategy &strategy,
                                   PatternRewriter &rewriter,
                                   bool eagerVectorization) {
    Location loc = op.getLoc();
    Value c0 = arith::ConstantIndexOp::create(rewriter, loc, 0);
    Value c1 = arith::ConstantIndexOp::create(rewriter, loc, 1);

    // Step 1: Collapse last 2 dims (32x2 -> 64) for output tensor
    const TensorInfo &outputInfo = strategy.outputs[0];
    auto outputType = outputInfo.type;
    int64_t rank = outputType.getRank();

    SmallVector<ReassociationIndices> reassociation;
    Value collapsedOutput =
        collapseTail2To64(rewriter, loc, op.getDpsInitOperand(0)->get(),
                          outputType, &reassociation);
    auto collapsedOutputType =
        cast<RankedTensorType>(collapsedOutput.getType());
    ArrayRef<int64_t> collapsedShape = collapsedOutputType.getShape();

    // Step 2: Collapse input tensors that have packed layout
    SmallVector<Value> collapsedInputs;
    SmallVector<bool> isInputPacked;
    SmallVector<std::optional<RepackedNx32Info>> repackedInputs;

    if (failed(collapseOrRepackInputs(rewriter, loc, strategy, collapsedInputs,
                                      isInputPacked, repackedInputs)))
      return failure();

    // Step 3: Build nested loops for all dimensions except the packed vector
    // dimension. Apply the computed loop permutation, and split a single
    // secondary loop if that exposes a better position next to the packed tail.
    int64_t numLoops = collapsedShape.size() - 1; // Exclude vectorized dim
    SmallVector<Value> lowerBounds, upperBounds, steps;
    std::optional<VectorizationStrategy::SplitPlan> activeSplitPlan;
    for (const auto &plan : strategy.splitPlans) {
      if (plan.loopDim == strategy.loopPermutation[numLoops - 1]) {
        activeSplitPlan = plan;
        break;
      }
    }

    for (int64_t i = 0; i < numLoops; ++i) {
      unsigned originalDim = strategy.loopPermutation[i];
      int64_t extent = outputType.getShape()[originalDim];
      if (activeSplitPlan && originalDim == activeSplitPlan->loopDim)
        extent /= activeSplitPlan->splitFactor;
      lowerBounds.push_back(arith::ConstantIndexOp::create(rewriter, loc, 0));
      upperBounds.push_back(
          arith::ConstantIndexOp::create(rewriter, loc, extent));
      steps.push_back(c1);
    }

    // Step 4: Create the nested loop structure
    VectorizeContext ctx{
        op, strategy,        outputType,    rank,           c0,
        c1, collapsedInputs, isInputPacked, repackedInputs, eagerVectorization};
    bool vectorizationFailed = false;
    auto loopNest = scf::buildLoopNest(
        rewriter, loc, lowerBounds, upperBounds, steps, collapsedOutput,
        [&](OpBuilder &b, Location loc, ValueRange ivs,
            ValueRange iterArgs) -> scf::ValueVector {
          Value accumulator = iterArgs[0];

          // No split plan — emit a single vectorized body.
          if (!activeSplitPlan) {
            Value updated = emitVectorBody(ctx, b, loc, ivs, accumulator);
            if (!updated) {
              vectorizationFailed = true;
              return {accumulator};
            }
            return {updated};
          }

          // Split plan — pre-load repacked Nx32 inputs once and unroll.
          int64_t splitFactor = activeSplitPlan->splitFactor;
          Value splitFactorVal =
              arith::ConstantIndexOp::create(b, loc, splitFactor);

          Type elementType = outputType.getElementType();
          auto vectorType = VectorType::get({64}, elementType);

          SmallVector<SmallVector<Value, 2>> preloadedHalves;
          if (!preloadRepackedInputs(ctx, b, loc, ivs, vectorType,
                                     preloadedHalves)) {
            vectorizationFailed = true;
            return {accumulator};
          }

          Value result = emitUnrolledSplitBody(ctx, b, loc, ivs, accumulator,
                                               splitFactorVal, splitFactor,
                                               preloadedHalves);
          if (!result)
            vectorizationFailed = true;
          return {result ? result : accumulator};
        });

    // If vectorization failed inside the loop body, undo all generated IR
    // and report failure so the original generic op is preserved.
    if (vectorizationFailed)
      return failure();

    // Step 8: Expand the result back to original shape
    Value result = loopNest.loops.empty() ? loopNest.results[0]
                                          : loopNest.loops[0].getResult(0);

    Value expandedResult = tensor::ExpandShapeOp::create(
        rewriter, loc, outputType, result, reassociation);

    expandedResult.getDefiningOp()->setAttr("hexagon.packed_vectorized",
                                            UnitAttr::get(op->getContext()));
    rewriter.replaceOp(op, expandedResult);

    return success();
  }

public:
  LogicalResult matchAndRewrite(linalg::GenericOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.hasPureTensorSemantics())
      return failure();

    VectorizationStrategy strategy(op);

    // Gate loop splitting: clear split plans and recompute permutation.
    if (!enableLoopSplitting) {
      strategy.splitPlans.clear();
      strategy.computeLoopPermutation();
      strategy.computeInversePermutation();
    }

    if (!strategy.canVectorize())
      return failure();

    // Handle reductions into non-packed outputs.
    if (!strategy.reductionDims.empty() && !strategy.outputs.empty() &&
        !strategy.outputs[0].hasPackedLayout())
      return vectorizePackedReductionToScalarOutput(op, strategy, rewriter,
                                                    eagerVectorization);

    // Reductions into packed outputs are not yet supported by vectorizeOp
    // (it assumes all-parallel iteration and indexes the output shape with
    // loop-dim indices that may exceed the output tensor rank).
    if (!strategy.reductionDims.empty())
      return failure();

    // For now, only handle simple cases similar to existing passes
    // This is a framework that can be extended

    // Check for specific patterns we can handle
    if (strategy.outputs.size() != 1)
      return failure();

    if (strategy.outputs[0].type.getRank() < 2)
      return failure();

    // Check if output has packed layout
    if (!strategy.outputs[0].hasPackedLayout())
      return failure();

    // Attempt vectorization
    return vectorizeOp(op, strategy, rewriter, eagerVectorization);
  }
};

struct HexagonVectorizePackedGenericPass
    : public ::impl::HexagonVectorizePackedGenericBase<
          HexagonVectorizePackedGenericPass> {
  HexagonVectorizePackedGenericPass() = default;
  HexagonVectorizePackedGenericPass(bool enableLoopSplittingOpt,
                                    bool eagerVectorizationOpt) {
    enablePackedGenericLoopSplitting = enableLoopSplittingOpt;
    enablePackedGenericEagerVectorization = eagerVectorizationOpt;
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, func::FuncDialect,
                    linalg::LinalgDialect, math::MathDialect, scf::SCFDialect,
                    tensor::TensorDialect, vector::VectorDialect>();
  }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<VectorizePackedGenericPattern>(
        &getContext(), enablePackedGenericLoopSplitting,
        enablePackedGenericEagerVectorization);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
hexagon::createHexagonVectorizePackedGenericPass(
    const HexagonVectorizePackedGenericOptions &options) {
  return std::make_unique<HexagonVectorizePackedGenericPass>(
      options.enablePackedGenericLoopSplitting,
      options.enablePackedGenericEagerVectorization);
}
