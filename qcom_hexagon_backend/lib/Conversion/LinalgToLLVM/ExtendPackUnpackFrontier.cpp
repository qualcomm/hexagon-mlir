//===- ExtendPackFrontier.cpp:  extend pack/unpack frontier           ----====//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// Note: compared to upstream changes in this file are marked with a
// comment saying
// "Different than upstream begin" and "Different than upstream end"
// The main differences are, 1) the linalg.pack support with padding
// 2) support for lowerFrontier, upperFrontier, parallelsOnly flags,
// 3) documentation of pass limitations below
//
// This pass extends packing/unpacking frontier over non-conv ops.
// The supported types of propagation are:
// 1) linalg.pack and linalg.unpack through linalg.generic when all iterator
// types
//    are "parallel". linalg.pack through linalg.generic contains "reduction"
//    iterator types. There are several limitations to propagation
//    through linalg.generic:
//    -**PACK_MULTIRESULT_SUPPORT** no support for linalg.pack and linalg.unpack
//    propagation through multi-result linalg.generic
//    -**GENERIC_PAD_SUPPORT** no support for linalg.pack propagation
//    through linalg.generic if the linalg.pack has a padding of 0 and the
//    linalg.generic contains a division (this is better than the upstream where
//    paddings of 0 are not supported in any case).
//    -**WOULD_BREAK_DOMINANCE** this prevents layout propagation from being
//    effective when the result of the linalg.pack is not stored into a
//    linalg.empty. This case is covered with upstream test with upstream test
// 3) linalg.pack and linalg.unpack through tensor.pad is supported. These are
//    the limitations for this case:
//    -**PAD_OF_PACKED_DIM** linalg.unpack layout propagation is not supported
//    if we're trying to pad a packed dimension. Only constant padding values
//    are supported for tensor.pad.
//    -**PACK_WITH_PAD_THROUGH_PAD** There is no support when we want to
//    propagate linalg.pack with padding through tensor.pad.
//    -**PACK_OUTPUT_TO_EMPTY_4PAD** linalg.pack needs to output its result into
//    tensor.empty for layout propagation support.
// 4) linalg.pack and linalg.unpack through tensor.expand_shape and
//    tensor.collapse_shape:
//    -**UNPACK_INNER_SIZES_DIV_PROJECTED_DIMS** linalg.unpack through
//    tensor.expand_shape is possible only when the inner tile
//    sizes can divide the projected dims.
//    -**NO_UNPACK_THROUGH_COLLAPSE** linalg.unpack through
//    tensor.collapse_shape is not supported.
// 5) linalg.pack through tensor.expand_shape and tensor.collapse_shape.
// Propagation
//    through tensor.expand_shape has several limitation: A) outer dimension
//    permutation is not supported, B) multiple expanded dimensions cannot be
//    packed, just at most one, C) only the inner-most expanded dimension can be
//    packed if we want layout propagation to be effective.
//===----------------------------------------------------------------------===//

//  Different than upstream begin
//  Differences wrt upstream: The upstream code includes AffineOps.h
//  Tensor.h, IndexUtils.h whereas we do not include these
//  since they were either redundant or they were already included
//  by some of the headers we do include here. The headers we include
//  here and were not in upstream are TileUsingInterface.h, PatternMatch.h
//  and Pass.h and LinalgToLLVM.h
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/ADT/STLForwardCompat.h"
#include "llvm/ADT/SetOperations.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

#include "hexagon/Conversion/LinalgToLLVM/LinalgToLLVM.h"
//  Different than upstream end

using namespace mlir;
using namespace hexagon;
using namespace mlir::linalg;
#define DEBUG_TYPE "extend-pack-frontier"

#define GEN_PASS_DEF_HEXAGONEXTENDPACK
#include "hexagon/Conversion/LinalgToLLVM/Passes.h.inc"

namespace {

static bool hasGatherSemantics(linalg::GenericOp genericOp) {
  for (Operation &op : genericOp.getBody()->getOperations())
    if (isa<tensor::ExtractOp, linalg::IndexOp>(op))
      return true;
  return false;
}

// The struct contains the infomation about mapping packing information to
// the iteration domain of Linalg ops.
struct PackInfo {
  int64_t getNumTiledLoops() const { return tileToPointMapping.size(); };
  // InnerDimsPos on iteration domain, which follows the order in pack ops.
  SmallVector<int64_t> tiledDimsPos;
  // The sizes of tiling data dimensions on iteration domain.
  llvm::DenseMap<int64_t, OpFoldResult> domainDimAndTileMapping;
  // The mapping from a dimension of iteration domain to the corresponding inner
  // tiling dimension on iteration domain.
  llvm::DenseMap<int64_t, int64_t> tileToPointMapping;
  // The permutation of outer dims (on domain).
  SmallVector<int64_t> outerDimsOnDomainPerm;
  // Different than upstream begin
  TypedAttr paddingValue;
  llvm::SmallSet<int64_t, 10> dimPosToReduce;
  // Different than upstream end
};

// Different than upstream begin
// This struct captures sufficient information to propagate
// ExpandShape/CollapseShape across generic ops.
struct ExpandShapeInfo {
  // For each old iterator, stores the expansion factors.
  // If an iterator is not expanded, the expansion factors vector contains a
  // single element equal to loop bound for that iterator.
  llvm::SmallVector<llvm::SmallVector<int64_t>> iterToExpandedFactors;
  llvm::SmallVector<int64_t>
      newIterPos; // When a generic op is expanded, an iterator's position in
  // the new generic op could change. For each old iterator, stores the
  // new iterator position
  llvm::SmallVector<ReassociationIndices>
      newIterReassocInfo; // reassoc info for applying inverse ops
  int64_t numNewIters;    // Number of new iterators for the generic op after
                          // a generic op is expanded (either by pushing
                          // CollapseShape below or ExpandShape above)
};

// Eg. The following IR

//    %collapsed = tensor.collapse_shape %arg0 [[0], [1, 2], [3]] :
//                          tensor<1x7x1x768xf16> into tensor<1x7x768xf16>
//    %13 = tensor.empty() : tensor<1x7x768xf32>
//    %14 = tensor.empty() : tensor<1x7x768xf64>
//    %15:2 = linalg.generic {indexing_maps = [#map1, #map1, #map1],
//            iterator_types = ["parallel", "parallel", "parallel"]}
//            ins(%collapsed : tensor<1x7x768xf16>) outs(%13, %14 :
//            tensor<1x7x768xf32>,  tensor<1x7x768xf64>)
//     { ^bb0(%in: f16, %out: f32, %out_27: f64):
//       %38 = arith.extf %in : f16 to f32
//       %39 = arith.extf %38 : f32 to f64
//       linalg.yield %38, %39 : f32, f64
//     } -> (tensor<1x7x768xf32>, tensor<1x7x768xf64>)

// becomes

//     %0 = tensor.empty(): tensor<1x7x768xf32>
//     %1 = tensor.empty(): tensor<1x7x768xf64>
//     %expanded = tensor.expand_shape %0 [[0], [1, 2], [3]] output_shape
//        [1, 7, 1, 768] : tensor<1x7x768xf32> into tensor<1x7x1x768xf32>
//     %expanded_0 = tensor.expand_shape %1 [[0], [1, 2], [3]]
//     			output_shape [1, 7, 1, 768] :
//     			tensor<1x7x768xf64> into tensor<1x7x1x768xf64>
//     %2:2 = linalg.generic {indexing_maps = [#map, #map, #map],
//     		iterator_types = ["parallel", "parallel", "parallel",
//     		"parallel"]}
//     		ins(%arg0 : tensor<1x7x1x768xf16>)
//     		outs(%expanded, %expanded_0 :
//     		tensor<1x7x1x768xf32>, tensor<1x7x1x768xf64>)
//     {
//     	 ^bb0(%in: f16, %out: f32, %out_1: f64):
//       %3 = arith.extf %in : f16 to f32
//       %4 = arith.extf %3 : f32 to f64
//       linalg.yield %3, %4 : f32, f64
//     } -> (tensor<1x7x1x768xf32>, tensor<1x7x1x768xf64>)

// Here, iterToExpandedFactors is [0 -> [1], 1 -> [7, 1], 2-> [768]]
// newIterPos is [0, 1, 3]
// newIterReassocInfo is [[0], [1,2], [3]]
// numNewIters is 4

// All of them are expressed in terms of iterators of the generic op.
// Using this, old indexing map and old iterator types, we can recompute
// the new indexing maps and iterator types.

template <typename OpTy>
static FailureOr<ExpandShapeInfo>
getExpandShapeInfoFromOperand(OpOperand *opOperand, linalg::GenericOp genericOp,
                              OpTy expandOrCollapseShapeOp) {
  static_assert(llvm::is_one_of<OpTy, tensor::CollapseShapeOp,
                                tensor::ExpandShapeOp>::value,
                "applies to only collapse shape or expand shape operations");
  ExpandShapeInfo expandShapeInfo;

  LLVM_DEBUG({
    llvm::dbgs() << "--- Construct ExpandShapeInfo From an operand ---\n";
  });
  SmallVector<int64_t> expandedShape;
  if (isa<tensor::CollapseShapeOp>(expandOrCollapseShapeOp)) {
    // Get the type of source of CollapseShapeOp
    auto srcType = cast<ShapedType>(expandOrCollapseShapeOp.getSrc().getType());
    if (!srcType.hasStaticShape()) {
      return failure(); // Not supporting dynamic shapes for now
    }
    expandedShape.append(srcType.getShape().begin(), srcType.getShape().end());
  } else if (isa<tensor::ExpandShapeOp>(expandOrCollapseShapeOp)) {
    auto dstType =
        cast<ShapedType>(expandOrCollapseShapeOp.getResult().getType());
    if (!dstType.hasStaticShape()) {
      return failure();
    }
    expandedShape.append(dstType.getShape().begin(), dstType.getShape().end());
  } else
    return failure();

  AffineMap indexingMap = genericOp.getMatchingIndexingMap(opOperand);
  SmallVector<AffineMap> indexingMaps = genericOp.getIndexingMapsArray();
  int64_t origNumDims = indexingMap.getNumDims();
  SmallVector<AffineExpr> exprs(indexingMap.getResults());

  // get reassociation for collapseShape op - this is stored as indices
  // of dimensions (not iterators of generic op)
  SmallVector<ReassociationIndices> reassociation =
      expandOrCollapseShapeOp.getReassociationIndices();

  llvm::SmallVector<int64_t> staticRanges = genericOp.getStaticLoopRanges();
  // Initialize iterToExpandedFactors to the loop bounds of the original
  // generic op
  for (int i = 0; i < origNumDims; ++i) {
    int64_t loopBound = staticRanges[i];
    if (loopBound == mlir::ShapedType::kDynamic) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Loop bound for dim " << i << " is dynamic for this op\n"
                 << genericOp << "\n");
      return failure();
    }
    // Initialize this to a single factor = loopbound, for all iters
    expandShapeInfo.iterToExpandedFactors.push_back({loopBound});
  }

  // Now construct reassociation in terms of iterators of the
  // generic op. We can't handle the case where an iterator
  // is used to index multiple dimensions of opOperand
  llvm::DenseMap<int64_t, bool> encounteredIters;
  for (int i = 0; i < exprs.size(); ++i) {
    // There are two cases where AffineConstantExpr can occur.
    // 1. truly constant indexing (e.g., A[3][j])
    // 2. reduction iterator.
    // We can ignore constant expressions: no iterator is used, so the original
    // loop bound remains unchanged.
    if (auto dimExpr = dyn_cast<AffineDimExpr>(exprs[i])) {
      int64_t iterPos = dimExpr.getPosition();
      if (encounteredIters.contains(iterPos)) {
        LLVM_DEBUG(llvm::dbgs() << "Encountered iter " << iterPos
                                << " twice in " << indexingMap << "\n");
        return failure();
      }
      encounteredIters[iterPos] = true;
      // iterator 'iterPos' is used to index dimension 'i'
      ReassociationIndices reassoc = reassociation[i];
      llvm::SmallVector<int64_t> expandedFactors;
      // Dim 'i' is collapsed from dims in reassoc.
      // Record the expansion factors for each of these collapsed dims.
      for (auto idx : reassoc) {
        expandedFactors.push_back(expandedShape[idx]);
      }
      // Bounds for new iterators can be obtained from expandedFactors
      expandShapeInfo.iterToExpandedFactors[iterPos] = expandedFactors;
    } else if (!isa<AffineConstantExpr>(exprs[i])) {
      LLVM_DEBUG(llvm::dbgs() << "Encountered this map " << indexingMap
                              << "\n for this operand\n"
                              << opOperand->get() << "\n");
      return failure();
    }
  }

  // For now, don't handle cases when an iterator that is expanded
  // is used in a non-AffineDimExpr
  auto areAllAffineDimExpr = [&](int dim) {
    for (AffineMap map : indexingMaps) {
      if (llvm::any_of(map.getResults(), [dim](AffineExpr expr) {
            return expr.isFunctionOfDim(dim) && !isa<AffineDimExpr>(expr);
          })) {
        return false;
      }
    }
    return true;
  };
  // Check that any iterator that is expanded is only used in AffineDimExpr
  for (int i = 0; i < origNumDims; ++i) {
    if (expandShapeInfo.iterToExpandedFactors[i].size() > 1) {
      if (!areAllAffineDimExpr(i)) {
        LLVM_DEBUG(llvm::dbgs() << "Iterator " << i
                                << " is expanded but not used "
                                   "only in AffineDimExpr\n");
        return failure();
      }
    }
  }

  // Now compute newIterPos and reassoc info
  int64_t newPos = 0;
  for (int i = 0; i < expandShapeInfo.iterToExpandedFactors.size(); ++i) {
    expandShapeInfo.newIterPos.push_back(newPos);
    // iterator 'i' is expanded into multiple iterators
    // (iterToExpandedFactors[i].size)
    // Record position of iterators in the new generic op
    ReassociationIndices reassoc;
    for (int j = 0; j < expandShapeInfo.iterToExpandedFactors[i].size(); ++j)
      reassoc.push_back(newPos + j);
    expandShapeInfo.newIterReassocInfo.push_back(reassoc);
    newPos += (expandShapeInfo.iterToExpandedFactors[i].size());
  }
  if (newPos != expandedShape.size()) {
    // Missing expansion info for some iterators
    LLVM_DEBUG(llvm::dbgs()
               << "New pos " << newPos << " != expanded shape size "
               << expandedShape.size() << "\n");
    return failure();
  }
  expandShapeInfo.numNewIters = newPos;

  return expandShapeInfo;
}

/// Given a GenericOp, ExpandShapeInfo struct and an operand, returns a
// tuple for expanded-shaped operand and indexing_map in the (to-be) expanded
// generic op
static FailureOr<std::tuple<Value, AffineMap>>
getOrCreateExpandedViewOfOperand(OpBuilder &b, Location loc,
                                 const ExpandShapeInfo &expandShapeInfo,
                                 GenericOp genericOp, OpOperand *opOperand) {
  int64_t numOrigLoops = genericOp.getNumLoops();
  int64_t numNewLoops = expandShapeInfo.numNewIters;

  AffineMap origIndexingMap = genericOp.getMatchingIndexingMap(opOperand);
  auto oldRankedTensorType = cast<RankedTensorType>(opOperand->get().getType());
  auto oldShape = oldRankedTensorType.getShape();

  llvm::DenseMap<int64_t, int64_t> domainDimToOperandDim;
  SmallVector<AffineExpr> newExprs;
  auto oldExprs = origIndexingMap.getResults();

  // If the OpOperand is a scalar or a zero-rank tensor, no need to expand.
  if (genericOp.isScalar(opOperand) || oldExprs.empty())
    return std::make_tuple(
        opOperand->get(),
        AffineMap::get(numNewLoops, 0, oldExprs, b.getContext()));

  // Step 1. Map iter dims to indices of the operand (which dimension, an
  // iterator indexes) iterators used to index 'n' dims (m[0], m[1], ...,
  // m[n-1]) m[i] is mapped to newIterPos[m[i]], and is expanded to
  // iterToExpandedFactors[m[i]].size() iters 'i' is expanded to
  // iterToExpandedFactors[m[i]]. Assumption: An iter dim is used to index at
  // most one dimension
  int cumIndex = 0;
  llvm::SmallVector<ReassociationIndices> reassocIndicesVec;
  SmallVector<int64_t> newShape;
  for (auto [oldIndex, expr] :
       llvm::zip_equal(llvm::seq<unsigned>(0, oldExprs.size()), oldExprs)) {
    ReassociationIndices reassocIndices;
    if (auto dimExpr = dyn_cast<AffineDimExpr>(expr)) {
      int64_t dimPos = dimExpr.getPosition();
      if ((expandShapeInfo.iterToExpandedFactors[dimPos]).size() > 1) {
        // iterator is expanded;
        int newIterCount = expandShapeInfo.iterToExpandedFactors[dimPos].size();
        for (int i = 0; i < newIterCount; ++i) {
          newExprs.push_back(
              b.getAffineDimExpr(expandShapeInfo.newIterPos[dimPos] + i));
          reassocIndices.push_back(cumIndex + i);
          newShape.push_back(expandShapeInfo.iterToExpandedFactors[dimPos][i]);
        }
        cumIndex += newIterCount;
      } else {
        newExprs.push_back(
            b.getAffineDimExpr(expandShapeInfo.newIterPos[dimPos]));
        reassocIndices.push_back(cumIndex++);
        newShape.push_back(oldShape[oldIndex]);
      }
    } else if (auto constExpr = dyn_cast<AffineConstantExpr>(expr)) {
      newExprs.push_back(constExpr);
      reassocIndices.push_back(cumIndex++);
      newShape.push_back(oldShape[oldIndex]);
    } else {
      LLVM_DEBUG(llvm::dbgs() << "Unexpected expression type in map "
                              << origIndexingMap << "\n");
      return failure();
    }
    reassocIndicesVec.push_back(reassocIndices);
  }

  AffineMap newMap = AffineMap::get(numNewLoops, 0, newExprs, b.getContext());
  Value expandedOperand = tensor::ExpandShapeOp::create(
      b, loc,
      RankedTensorType::get(newShape, oldRankedTensorType.getElementType()),
      opOperand->get(), reassocIndicesVec);
  return std::make_tuple(expandedOperand, newMap);
}

/// Return the shape collapsed operand, if present, for the current generic op.
static FailureOr<OpOperand *> getShapeCollapsedOperand(GenericOp genericOp) {
  for (OpOperand &operand : genericOp->getOpOperands()) {
    auto collapseShapeOp =
        operand.get().getDefiningOp<tensor::CollapseShapeOp>();
    if (!collapseShapeOp)
      continue;
    // pick the first encountered CollapseShapeOp
    return &operand;
  }
  return failure();
}

bool isScalarOrZeroRankTensor(Value value) {
  Type type = value.getType();

  // 1. Check if it is a built-in scalar type (like IntegerType, FloatType)
  if (type.isIntOrFloat()) {
    return true;
  }

  // 2. Check if it is a ShapedType (like TensorType or VectorType)
  if (auto shapedType = dyn_cast<ShapedType>(type)) {
    if (shapedType.hasRank() && shapedType.getRank() == 0) {
      return true;
    }
  }

  return false;
}

/// This function is a helper subroutine to expand a genericOp and return it. It
/// will create a new generic op with the expanded operand and the expanded
/// output according to ExpandShapeInfo when we attempt to push down
/// collapseshape or bubble up expandshape around it. Implicitly this will only
/// work when a ExpandShapeInfo can be obtained.
static FailureOr<GenericOp>
expandGenericOp(RewriterBase &rewriter, GenericOp genericOp,
                const ExpandShapeInfo &expandShapeInfo,
                bool isFoldableCollapseExpand) {
  Location loc = genericOp.getLoc();
  SmallVector<Value> inputOperands;
  SmallVector<Value> inputOperandsFromExpandedSource;
  SmallVector<AffineMap> indexingMaps;
  auto isInverse = [](tensor::ExpandShapeOp expandShapeOp,
                      tensor::CollapseShapeOp collapseShapeOp) {
    return expandShapeOp.getReassociationIndices() ==
           collapseShapeOp.getReassociationIndices();
  };

  // Get expanded operand and indexing map info for each init operand
  SmallVector<Value> destInitVals;
  AffineMap expandedOutIndexingMap;
  for (auto i = 0; i < genericOp.getNumDpsInits(); ++i) {
    auto initOperand = genericOp.getDpsInitOperand(i);
    auto maybeExpandedOut = getOrCreateExpandedViewOfOperand(
        rewriter, genericOp.getLoc(), expandShapeInfo, genericOp, initOperand);
    if (failed(maybeExpandedOut))
      return failure();
    auto [expandedOutOperand, expandedOutIndexingMapCopy] = *maybeExpandedOut;
    destInitVals.push_back(expandedOutOperand);
    if (i > 0) {
      if (expandedOutIndexingMapCopy != expandedOutIndexingMap)
        return failure();
    } else {
      expandedOutIndexingMap = expandedOutIndexingMapCopy;
    }
  }

  for (OpOperand *inputOperand : genericOp.getDpsInputOperands()) {
    auto maybeExpandedIn = getOrCreateExpandedViewOfOperand(
        rewriter, loc, expandShapeInfo, genericOp, inputOperand);
    if (failed(maybeExpandedIn))
      return failure();
    auto [expandedOperand, expandedIndexingMap] = *maybeExpandedIn;
    auto collapseShapeOp =
        inputOperand->get().getDefiningOp<tensor::CollapseShapeOp>();
    auto expandShapeOp = expandedOperand.getDefiningOp<tensor::ExpandShapeOp>();
    // inputOperand is defined by CollapseShapeOp, and is expanded immediately.
    // In this case simply forward the input of CollapseShapeOp
    if (expandShapeOp && collapseShapeOp &&
        isInverse(expandShapeOp, collapseShapeOp)) {
      inputOperandsFromExpandedSource.push_back(collapseShapeOp.getSrc());
    } else {
      inputOperandsFromExpandedSource.push_back(expandedOperand);
    }
    inputOperands.push_back(expandedOperand);
    indexingMaps.push_back(expandedIndexingMap);
  }

  // If the expand->collapse sequences can be folded, replace use the sources of
  // the collapseshape ops in any collapse->expand chains on the generic op
  // operands.
  if (isFoldableCollapseExpand) {
    inputOperands = inputOperandsFromExpandedSource;
    // Check the same for destInitVals as well
    for (auto i = 0; i < destInitVals.size(); ++i) {
      auto dest = destInitVals[i];
      if (auto destExpand = dest.getDefiningOp<tensor::ExpandShapeOp>()) {
        auto destCollapse =
            destExpand.getSrc().getDefiningOp<tensor::CollapseShapeOp>();
        if (destCollapse && isInverse(destExpand, destCollapse))
          destInitVals[i] = destCollapse.getSrc();
      }
    }
  }

  SmallVector<utils::IteratorType> iterTypes =
      genericOp.getIteratorTypesArray();

  SmallVector<utils::IteratorType> expandedIterTypes;
  for (int64_t i = 0; i < genericOp.getNumLoops(); ++i) {
    expandedIterTypes.append(expandShapeInfo.iterToExpandedFactors[i].size(),
                             iterTypes[i]);
  }
  for (auto i = 0; i < genericOp.getNumResults(); ++i)
    indexingMaps.push_back(expandedOutIndexingMap);

  SmallVector<Type> destTypes;
  for (auto destInitVal : destInitVals) {
    destTypes.push_back(destInitVal.getType());
  }
  auto newGenericOp = linalg::GenericOp::create(
      rewriter, loc, destTypes, inputOperands, destInitVals, indexingMaps,
      expandedIterTypes,
      /*bodyBuild=*/nullptr, linalg::getPrunedAttributeList(genericOp));
  rewriter.cloneRegionBefore(genericOp.getRegion(), newGenericOp.getRegion(),
                             newGenericOp.getRegion().begin());
  return newGenericOp;
}

mlir::AffineMap getUniqueOutputAffineMap(GenericOp genericOp) {

  // Get the number of inputs and outputs (inits)
  int64_t numInputs = genericOp.getNumDpsInputs();
  int64_t numOutputs = genericOp.getNumDpsInits();

  // If there are no outputs, there is no "unique map" to return
  if (numOutputs == 0) {
    return mlir::AffineMap(); // Null map
  }

  // Retrieve all indexing maps
  // The array contains [input_maps..., output_maps...]
  llvm::SmallVector<mlir::AffineMap> maps = genericOp.getIndexingMapsArray();

  // Get the map of the first output operand
  // Output maps start at index 'numInputs'
  mlir::AffineMap firstOutMap = maps[numInputs];
  LLVM_DEBUG(llvm::dbgs() << "First output map: " << firstOutMap << "\n");

  // Check if subsequent output maps match the first one
  for (int64_t i = 1; i < numOutputs; ++i) {
    LLVM_DEBUG(llvm::dbgs()
               << "Next output map: " << maps[numInputs + i] << "\n");
    if (maps[numInputs + i] != firstOutMap) {
      LLVM_DEBUG(llvm::dbgs() << "Mismatch in maps\n");
      return mlir::AffineMap(); // Return null object on mismatch
    }
  }

  // All output maps are identical, return the map
  return firstOutMap;
}

static FailureOr<std::tuple<GenericOp, SmallVector<Value>>>
pushDownCollapseShapeOpThroughGenericOp(RewriterBase &rewriter,
                                        GenericOp genericOp,
                                        ControlPropagationFn controlFn) {
  AffineMap outMap = getUniqueOutputAffineMap(genericOp);
  // Don't propagate when maps for outputs differ
  if (!outMap)
    return failure();
  if (outMap.isEmpty())
    return failure();

  if (hasGatherSemantics(genericOp))
    return failure();

  // Collect the shapeCollapsed operand, if present.
  auto maybeShapeCollapsedOperand = getShapeCollapsedOperand(genericOp);
  if (failed(maybeShapeCollapsedOperand)) {
    LLVM_DEBUG(llvm::dbgs()
               << "Failed to get CollapseShapeOp for " << genericOp << "\n");
    return failure();
  }
  OpOperand *shapeCollapsedOperand = *(maybeShapeCollapsedOperand);

  LLVM_DEBUG(llvm::dbgs() << "Trying to push " << shapeCollapsedOperand->get()
                          << "\n below " << genericOp << "\n");

  tensor::CollapseShapeOp producerCollapseShapeOp =
      shapeCollapsedOperand->get().getDefiningOp<tensor::CollapseShapeOp>();
  if (!producerCollapseShapeOp)
    return failure();

  if (!controlFn(shapeCollapsedOperand))
    return failure();

  auto expandShapeInfo = getExpandShapeInfoFromOperand(
      shapeCollapsedOperand, genericOp, producerCollapseShapeOp);

  if (failed(expandShapeInfo)) {
    LLVM_DEBUG(llvm::dbgs()
               << "Failed to get ExpandShapeInfo for " << genericOp << "\n");
    return failure();
  }

  auto newGenericOp = expandGenericOp(rewriter, genericOp, *expandShapeInfo,
                                      /*isFoldableCollapseExpand=*/true);
  if (failed(newGenericOp))
    return failure();

  SmallVector<Value> newResults;
  for (int i = 0; i < newGenericOp->getNumDpsInits(); ++i) {
    auto initOperand = newGenericOp->getDpsInitOperand(i);
    Value newResult = newGenericOp->getTiedOpResult(initOperand);

    // If the output is unaffected, no need to collapse.
    if (isScalarOrZeroRankTensor(newResult))
      newResults.push_back(newResult);
    else {
      auto destExpandOp =
          (initOperand->get()).getDefiningOp<tensor::ExpandShapeOp>();
      if (!destExpandOp)
        return failure();
      // Create a new CollapseShapeOp
      Value collapsedOutput = tensor::CollapseShapeOp::create(
          rewriter, genericOp.getLoc(), genericOp.getResultTypes()[i],
          newResult, destExpandOp.getReassociationIndices());
      newResults.push_back(collapsedOutput);
    }
  }

  return std::make_tuple(*newGenericOp, newResults);
}

// Wrapper pattern that applies pushDownCollapseShapeOpThroughGenericOp method.
struct PushDownCollapseShapeOpThroughGenericOp
    : public OpRewritePattern<GenericOp> {
public:
  PushDownCollapseShapeOpThroughGenericOp(MLIRContext *context,
                                          ControlPropagationFn fun)
      : OpRewritePattern<GenericOp>(context), controlFn(std::move(fun)) {}

  LogicalResult matchAndRewrite(GenericOp genericOp,
                                PatternRewriter &rewriter) const override {

    // Don't propagate through generics that don't produce results
    if (genericOp.getNumResults() < 1)
      return failure();

    auto genericAndRepl =
        pushDownCollapseShapeOpThroughGenericOp(rewriter, genericOp, controlFn);

    if (failed(genericAndRepl)) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Failed to push CollapseShape for " << genericOp << "\n");
      return failure();
    }
    auto results = std::get<1>(*genericAndRepl);
    LLVM_DEBUG(llvm::dbgs() << "Successfully pushed CollapseShape below \n"
                            << std::get<0>(*genericAndRepl) << "\n");
    rewriter.replaceOp(genericOp, results);
    return success();
  }

private:
  ControlPropagationFn controlFn;
};

// Wrapper pattern that applies bubbleUpExpandShapeOpThroughGenericOp method.
static FailureOr<GenericOp>
bubbleUpExpandShapeOpThroughGenericOp(RewriterBase &rewriter,
                                      tensor::ExpandShapeOp expandShapeOp,
                                      const ControlPropagationFn &controlFn) {
  // Check if the source of ExpandShapeOp is a GenericOp
  auto genericOp = expandShapeOp.getSrc().getDefiningOp<GenericOp>();
  if (!genericOp)
    return failure();

  // User controlled propagation function.
  if (!controlFn(&expandShapeOp.getSrcMutable()))
    return failure();

  if (hasGatherSemantics(genericOp))
    return failure();

  // TODO: Relax the restriction. We are able to bubble up the expand shape op
  // through multi-result generic op. It just needs more work.
  if (genericOp.getNumResults() != 1)
    return failure();

  // Bail-out if the result of the generic has multiple uses, as bubbling up
  // creates recomputation if the generic has multiple users.
  // TODO: Enable the case where every use is an identical expand shape op as no
  // recomputation is needed in that case.
  if (!genericOp->getResult(0).hasOneUse())
    return failure();

  OpOperand *opOperand = genericOp.getDpsInitOperand(0);
  auto expandShapeInfo =
      getExpandShapeInfoFromOperand(opOperand, genericOp, expandShapeOp);
  if (failed(expandShapeInfo))
    return failure();

  // We want to move the expand shape not the generic.
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(genericOp);

  return expandGenericOp(rewriter, genericOp, *expandShapeInfo,
                         /*isFoldableCollapseExpand=*/true);
}

// Wrapper pattern that applies bubbleUpExpandShapeOpThroughGenericOp method.
struct BubbleUpExpandShapeOpThroughGenericOp
    : public OpRewritePattern<tensor::ExpandShapeOp> {
public:
  BubbleUpExpandShapeOpThroughGenericOp(MLIRContext *context,
                                        ControlPropagationFn fun)
      : OpRewritePattern<tensor::ExpandShapeOp>(context),
        controlFn(std::move(fun)) {}

  LogicalResult matchAndRewrite(tensor::ExpandShapeOp expandShapeOp,
                                PatternRewriter &rewriter) const override {
    auto genericOp = bubbleUpExpandShapeOpThroughGenericOp(
        rewriter, expandShapeOp, controlFn);
    if (failed(genericOp))
      return failure();
    rewriter.replaceOp(expandShapeOp, genericOp->getResults());
    return success();
  }

private:
  ControlPropagationFn controlFn;
};
// Different than upstream end

template <typename OpTy>
static FailureOr<PackInfo>
getPackingInfoFromOperand(OpOperand *opOperand, linalg::GenericOp genericOp,
                          OpTy packOrUnPackOp) {
  static_assert(llvm::is_one_of<OpTy, linalg::PackOp, linalg::UnPackOp>::value,
                "applies to only pack or unpack operations");
  LLVM_DEBUG(
      { llvm::dbgs() << "--- Construct PackInfo From an operand ---\n"; });

  AffineMap indexingMap = genericOp.getMatchingIndexingMap(opOperand);
  SmallVector<AffineMap> indexingMaps = genericOp.getIndexingMapsArray();
  SmallVector<utils::IteratorType> iterators =
      genericOp.getIteratorTypesArray();

  PackInfo packInfo;
  // Different than upstream begin
  // Only accept constant padding values for PackOp
  if constexpr (std::is_same<linalg::PackOp, OpTy>::value) {
    if (Value paddingValue =
            static_cast<linalg::PackOp>(packOrUnPackOp).getPaddingValue()) {
      auto constantOp = paddingValue.getDefiningOp<arith::ConstantOp>();
      if (!constantOp || !(packInfo.paddingValue =
                               dyn_cast<TypedAttr>(constantOp.getValueAttr())))
        return failure();
    }
  }
  // Different than upstream end
  int64_t origNumDims = indexingMap.getNumDims();
  SmallVector<AffineExpr> exprs(indexingMap.getResults());

  // Different than upstream begin
  DenseMap<int64_t, int64_t> nonAffineDimExprsCount;
  int k = 0;
  SmallVector<int64_t> unusedDimsPos;
  SmallVector<int64_t> usedDims;
  for (int i = 0; i < exprs.size(); i++) {
    auto expr = exprs[i];
    nonAffineDimExprsCount.insert({i, k});
    if (auto dimExpr = dyn_cast<AffineDimExpr>(expr)) {
      usedDims.push_back(dimExpr.getPosition());
    } else {
      k++;
    }
  }
  for (auto idx : llvm::seq<size_t>(0, iterators.size())) {
    if (!llvm::is_contained(usedDims, idx)) {
      nonAffineDimExprsCount.insert({idx, k});
      k++;
      unusedDimsPos.push_back(idx);
    }
  }

  // Different than upstream end

  ArrayRef<int64_t> innerDimsPos = packOrUnPackOp.getInnerDimsPos();
  for (auto [index, innerDimPos, tileSize] :
       llvm::zip_equal(llvm::seq<unsigned>(0, innerDimsPos.size()),
                       innerDimsPos, packOrUnPackOp.getMixedTiles())) {
    auto expr = exprs[innerDimPos];
    // Different than upstream begin
    // Map the inner dimension position to the corresponding iteration domain
    // dimension position. This handles two cases:
    // 1. If the expression is an AffineDimExpr (e.g., d0, d1), directly extract
    //    the dimension position from the expression.
    // 2. If the expression is NOT an AffineDimExpr (e.g., constant or complex
    //    affine expression), we need to map it to an unused dimension in the
    //    iteration domain. We count how many non-AffineDimExpr expressions
    //    appear before this position (k), then use that count to index into the
    //    unusedDimsPos array to find the corresponding unused domain dimension.
    //    This allows packing/unpacking to work with generic ops that have
    //    non-trivial indexing maps (e.g., broadcasting, constant dimensions).
    int64_t domainDimPos;
    if (auto dimExpr = dyn_cast<AffineDimExpr>(expr)) {
      domainDimPos = dimExpr.getPosition();
    } else {
      domainDimPos = unusedDimsPos[nonAffineDimExprsCount[innerDimPos]];
    }
    // Different than upstream end
    // Upstream bails out on reduction of a dimension in inner
    // dims pos but now we just keep track of the overlapping
    // dimensions. We have tested this approach with
    // generic reductions that permute dimensions that overlap
    // with the unpack inner dims and outer dims in their
    // affine maps. For more details see end-to-end example
    // in test/python/mlir/layout_propagation and reduction lit
    // tests in qcom_hexagon_backend/test/Transforms/
    // LayoutPropagation
    if (!isParallelIterator(iterators[domainDimPos])) {
      if (isReductionIterator(iterators[domainDimPos]))
        packInfo.dimPosToReduce.insert(index);
      else {
        return failure();
      }
    }
    packInfo.tiledDimsPos.push_back(domainDimPos);
    packInfo.domainDimAndTileMapping[domainDimPos] = tileSize;
    packInfo.tileToPointMapping[domainDimPos] = origNumDims + index;
    LLVM_DEBUG({
      llvm::dbgs() << "map innerDimPos=" << innerDimPos
                   << " to iteration dimension (d" << domainDimPos << ", d"
                   << packInfo.tileToPointMapping[domainDimPos]
                   << "), which has size=("
                   << packInfo.domainDimAndTileMapping[domainDimPos] << ")\n";
    });
  }

  // Bail out if a tiled dimension is present in a map but not as an affine dim
  // expression.
  auto areAllAffineDimExpr = [&](int dim) {
    for (AffineMap map : indexingMaps) {
      if (llvm::any_of(map.getResults(), [dim](AffineExpr expr) {
            return expr.isFunctionOfDim(dim) && !isa<AffineDimExpr>(expr);
          })) {
        return false;
      }
    }
    return true;
  };
  for (int64_t i : packInfo.tiledDimsPos)
    if (!areAllAffineDimExpr(i))
      return failure();

  // Get the outer dims perm on the iteration domain. Start by identifying the
  // set of domain dims affected by the outer permutation along with the
  // permuted ordering for those dims. Then the full outer dims permutation can
  // be constructed by replacing the affected dims with the permuted result in a
  // numLoops-rank identity. e.g.
  //   outerDimsPerm = [1, 2, 0]
  //   indexingMap = (d0, d1, d2, d3, d4) -> (d1, d4, d3)
  //
  //   permutedOuterDims =        [4,    3, 1]
  //   outerDimsOnDomainPerm = [0, 4, 2, 3, 1]
  //
  // Non-affine dim expressions must not be permuted by the outer dims
  // permutation.
  SmallVector<int64_t> permutedOuterDims;
  for (auto [index, dim] : llvm::enumerate(packOrUnPackOp.getOuterDimsPerm())) {
    auto permutedExpr = indexingMap.getResult(dim);
    if (auto dimExpr = dyn_cast<AffineDimExpr>(permutedExpr)) {
      permutedOuterDims.push_back(dimExpr.getPosition());
      continue;
    }

    // TODO: Allow propagation with transposes on non affine dim expressions,
    // e.g. d0 + d1 which implies transposing both dims simultaneously while
    // maintaining the relative position between them.
    if (static_cast<int64_t>(index) != dim)
      return failure();
  }
  if (!permutedOuterDims.empty()) {
    int64_t outerDimIndex = 0;
    llvm::DenseSet<int64_t> permutedDomainDims(permutedOuterDims.begin(),
                                               permutedOuterDims.end());
    for (int i = 0, e = indexingMap.getNumDims(); i < e; i++)
      packInfo.outerDimsOnDomainPerm.push_back(
          permutedDomainDims.contains(i) ? permutedOuterDims[outerDimIndex++]
                                         : i);
    LLVM_DEBUG({
      llvm::dbgs() << "map outer dimsDimsPerm to ";
      for (auto dim : packInfo.outerDimsOnDomainPerm)
        llvm::dbgs() << dim << " ";
      llvm::dbgs() << "\n";
    });
  }

  return packInfo;
}

static SmallVector<int64_t> computeOuterDims(ArrayRef<int64_t> perm,
                                             ArrayRef<AffineExpr> exprs) {
  // Compute `outer_dims_perm`. See example:
  // current exprs      : (d0, d1, d2, d3) -> (d2, d3)
  // perm               : [0, 3, 1, 2]
  // First map d2, d3 with their position in the array as:
  // currentPositionTileLoops: dim | pos
  //                           d2  | 0
  //                           d3  | 1
  // then scan `perm` in order and get the `outer_dims_perm`
  // to be used, here it would be [1, 0].
  assert(!perm.empty() && "expect perm not to be empty");
  assert(!exprs.empty() && "expect exprs not to be empty");
  if (exprs.size() == 1)
    return {};
  SmallVector<int64_t> outerDimsPerm;
  DenseMap<int64_t, int64_t> currentPositionTileLoops;
  for (auto [pos, expr] : llvm::enumerate(exprs)) {
    // Here we rely on the assumption that the outer dims permutation
    // when propagating currently requires that non-affine dim expressions
    // are not permuted, thus allowing the identity assignment below.
    if (auto dimExpr = dyn_cast<AffineDimExpr>(expr))
      currentPositionTileLoops[dimExpr.getPosition()] = pos;
    else
      currentPositionTileLoops[pos] = pos;
  }
  for (int64_t loopIdx : perm) {
    if (currentPositionTileLoops.count(loopIdx))
      outerDimsPerm.push_back(currentPositionTileLoops.lookup(loopIdx));
  }
  return outerDimsPerm;
}

/// Returns a tuple for packed operand and indexing_map with the assumptions:
///   1) The generic op is the producer of the pack op.
///   2) The generic op has only one result.
/// If the operand is a scalar or packing dimensions are all irrelevant to the
/// operand, the operand and the updated indexing map will be returned.
/// Otherwise, it returns the packed operand and the updated indexing map. E.g.,
///
///   #map0 = affine_map<(d0, d1) -> (d0, d1)>
///   #map1 = affine_map<(d0, d1) -> (d0)>
///   #map2 = affine_map<(d0, d1) -> (d1)>
///   %0 = linalg.generic {indexing_maps = [#map1, #map2, #map0],
///                        iterator_types = ["parallel", "parallel"]}
///      ins(%arg0, %arg1 : tensor<?xf32>, tensor<?xf32>)
///      outs(%init : tensor<?x?xf32>) {
///    ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):
///      %4 = arith.addf %arg3, %arg4 : f32
///      linalg.yield %4 : f32
///  } -> tensor<?x?xf32>
///  %1 = linalg.pack %0
///    inner_dims_pos = [0, 1]
///    inner_tiles = [8, 2]
///    into %dest : tensor<?x?xf32> -> tensor<?x?x8x2xf32>
///
///  Taking the first input operand as an example, the inner tile size of d1 is
///  8. Thus, the below operation and `affine_map<(d0, d1, d2, d3)> ->
///  affine_map<(d1, d3)>` will be returned.
///
///  %pack = linalg.pack %arg0
///    inner_dims_pos = [0]
///    inner_tiles = [8]
///    into %init : tensor<?xf32> -> tensor<?x8xf32>

/// Helper function to create a constant padding value of the specified type.
/// Supports FloatType and IntegerType with the given numeric value.
static Value createConstantPaddingValue(OpBuilder &b, Location loc,
                                        Type elemType, int64_t intValue) {
  if (isa<FloatType>(elemType)) {
    return arith::ConstantOp::create(
               b, loc, b.getFloatAttr(elemType, static_cast<double>(intValue)))
        .getResult();
  } else if (isa<IntegerType>(elemType)) {
    return arith::ConstantOp::create(b, loc,
                                     b.getIntegerAttr(elemType, intValue))
        .getResult();
  }
  return Value();
}

/// Helper function to cast a padding value to the target element type.
/// Returns the casted value if successful, or the original value if casting
/// is not needed or not possible.
static Value castPaddingValueToType(OpBuilder &b, Location loc,
                                    Value paddingValue, Type targetElemType) {
  if (!paddingValue || paddingValue.getType() == targetElemType) {
    return paddingValue;
  }

  auto constantOp = paddingValue.getDefiningOp<arith::ConstantOp>();
  if (!constantOp) {
    return paddingValue;
  }

  if (auto floatAttr = dyn_cast<FloatAttr>(constantOp.getValueAttr())) {
    return arith::ConstantOp::create(
               b, loc,
               b.getFloatAttr(targetElemType, floatAttr.getValueAsDouble()))
        .getResult();
  } else if (auto intAttr = dyn_cast<IntegerAttr>(constantOp.getValueAttr())) {
    return arith::ConstantOp::create(
               b, loc, b.getIntegerAttr(targetElemType, intAttr.getInt()))
        .getResult();
  }

  return paddingValue;
}

/// Helper function to compute padding value and create a packed operand.
/// This function handles:
/// 1. Checking if perfect tiling is possible
/// 2. Computing or inferring the padding value if needed
/// 3. Casting the padding value to the correct type
/// 4. Creating the linalg.PackOp
static Value createPackedOperand(OpBuilder &b, Location loc, Value source,
                                 Value dest, ArrayRef<int64_t> innerDimsPos,
                                 ArrayRef<OpFoldResult> innerTileSizes,
                                 ArrayRef<int64_t> outerDimsPerm,
                                 TypedAttr initialPaddingAttr,
                                 GenericOp genericOp, OpOperand *opOperand) {
  Value paddingValue;

  // Convert the TypedAttr to a Value if it exists
  if (initialPaddingAttr) {
    paddingValue =
        arith::ConstantOp::create(b, loc, initialPaddingAttr).getResult();
  }

  // Check if perfect tiling is possible (all dimensions divisible by tile
  // sizes)
  bool isPerfectTiling = true;
  ShapedType sourceType = cast<ShapedType>(source.getType());

  // If source has dynamic shape, we generally assume padding might be needed
  if (!sourceType.hasStaticShape()) {
    isPerfectTiling = false;
  } else {
    for (auto [pos, tileSize] : llvm::zip(innerDimsPos, innerTileSizes)) {
      int64_t dimSize = sourceType.getDimSize(pos);
      std::optional<int64_t> cstTile = getConstantIntValue(tileSize);

      // If tile size is dynamic or dimension is not divisible, padding is
      // needed
      if (!cstTile || (dimSize % *cstTile != 0)) {
        isPerfectTiling = false;
        break;
      }
    }
  }

  // If perfect tiling, force paddingValue to null (even if packInfo provided
  // one) or keep it null if it wasn't provided.
  if (isPerfectTiling) {
    paddingValue = Value();
  }

  Type elemType = getElementTypeOrSelf(source.getType());

  // Logic to compute padding if missing AND strictly needed
  if (!paddingValue && !isPerfectTiling) {
    paddingValue = createConstantPaddingValue(b, loc, elemType, 0);
  }

  // Cast padding if it exists and type mismatch
  paddingValue = castPaddingValueToType(b, loc, paddingValue, elemType);

  return linalg::PackOp::create(b, loc, source, dest, innerDimsPos,
                                innerTileSizes, paddingValue, outerDimsPerm)
      .getResult();
}

static FailureOr<std::tuple<Value, AffineMap>>
getOrCreatePackedViewOfOperand(OpBuilder &b, Location loc, PackInfo packInfo,
                               GenericOp genericOp, OpOperand *opOperand) {
  int64_t numOrigLoops = genericOp.getNumLoops();
  int64_t numInnerLoops = packInfo.getNumTiledLoops();
  int64_t numLoops = numOrigLoops + numInnerLoops;
  AffineMap origIndexingMap = genericOp.getMatchingIndexingMap(opOperand);
  llvm::DenseMap<int64_t, int64_t> domainDimToOperandDim;
  SmallVector<AffineExpr> exprs(origIndexingMap.getResults());

  // If the OpOperand is a scalar or a zero-rank tensor, no need to pack.
  if (genericOp.isScalar(opOperand) || exprs.empty())
    return std::make_tuple(opOperand->get(),
                           AffineMap::get(numLoops, 0, exprs, b.getContext()));

  // Step 1. Construct the information of packing data dimensions; append inner
  // dimensions to the indexing maps for the operand.

  // Different than upstream begin
  SmallVector<int64_t> unusedDimsPos;
  SmallVector<int64_t> usedDims;
  for (auto expr : origIndexingMap.getResults()) {
    if (auto dimExpr = dyn_cast<AffineDimExpr>(expr))
      usedDims.push_back(dimExpr.getPosition());
  }
  SmallVector<utils::IteratorType> iterators =
      genericOp.getIteratorTypesArray();
  for (auto idx : llvm::seq<size_t>(0, iterators.size())) {
    if (!llvm::is_contained(usedDims, idx)) {
      unusedDimsPos.push_back(idx);
    }
  }

  size_t k = 0;
  // Different than upstream end

  for (auto [index, expr] : llvm::enumerate(exprs)) {
    if (auto dimExpr = dyn_cast<AffineDimExpr>(expr)) {
      int64_t dimPos = dimExpr.getPosition();
      domainDimToOperandDim[dimPos] = index;
      continue;
    }
    // Different than upstream begin
    // For non-AffineDimExpr expressions (e.g., constants, complex affine
    // exprs), map them to unused dimensions in the iteration domain. This
    // handles cases like broadcasting where an operand dimension doesn't
    // directly correspond to an iteration domain dimension. We sequentially
    // assign unused domain dimensions to these non-affine expressions to
    // maintain proper mapping between domain dimensions and operand dimensions.
    if (k >= unusedDimsPos.size())
      return failure();
    domainDimToOperandDim[unusedDimsPos[k]] = index;
    k++;
    // Different than upstream end
  }
  SmallVector<int64_t> innerDimsPos;
  SmallVector<OpFoldResult> innerTileSizes;
  for (auto dimPos : packInfo.tiledDimsPos) {
    if (!domainDimToOperandDim.count(dimPos))
      continue;
    int64_t index = domainDimToOperandDim[dimPos];
    innerTileSizes.push_back(packInfo.domainDimAndTileMapping[dimPos]);
    innerDimsPos.push_back(index);
    exprs.push_back(b.getAffineDimExpr(packInfo.tileToPointMapping[dimPos]));
  }

  // Step 2. Handle outer dim permutations.
  SmallVector<int64_t> outerDimsPerm;
  if (!packInfo.outerDimsOnDomainPerm.empty()) {
    outerDimsPerm = computeOuterDims(packInfo.outerDimsOnDomainPerm, exprs);

    // Step 2.1: Fold transpose into the linalg.generic.
    SmallVector<int64_t> inversedOuterPerm =
        invertPermutationVector(packInfo.outerDimsOnDomainPerm);
    for (auto i : llvm::seq<unsigned>(0, origIndexingMap.getNumResults())) {
      if (auto dimExpr = dyn_cast<AffineDimExpr>(exprs[i])) {
        int64_t dimPos = dimExpr.getPosition();
        exprs[i] = b.getAffineDimExpr(inversedOuterPerm[dimPos]);
        continue;
      }
      assert(isa<AffineConstantExpr>(exprs[i]) &&
             "Attempted to permute non-constant and non-affine dim expression");
    }
    // Step 2.2: Undo the transposition on `exprs` and propagate the
    // transposition on the pack using outerDimsPerm.
    if (!outerDimsPerm.empty()) {
      SmallVector<AffineExpr> auxVec = exprs;
      for (const auto &en : enumerate(outerDimsPerm))
        auxVec[en.index()] = exprs[en.value()];
      exprs = auxVec;
    }
  }
  auto indexingMap = AffineMap::get(numLoops, 0, exprs, b.getContext());

  // The operand does not have dimensions that relates to pack op.
  if (innerDimsPos.empty() && outerDimsPerm.empty())
    return std::make_tuple(opOperand->get(), indexingMap);

  auto empty = linalg::PackOp::createDestinationTensor(
      b, loc, opOperand->get(), innerTileSizes, innerDimsPos, outerDimsPerm);
  // Different than upstream begin
  auto packedOperand = createPackedOperand(
      b, loc, opOperand->get(), empty, innerDimsPos, innerTileSizes,
      outerDimsPerm, packInfo.paddingValue, genericOp, opOperand);
  // Different than upstream end
  return std::make_tuple(packedOperand, indexingMap);
}

/// This function is a helper subroutine to pack a genericOp and return it. It
/// will create a new generic op with the packed operand and the packed output
/// according to packInfo when we attempt to push down unpack or bubble up pack
/// around it. Implicitly this will only work when a packInfo can be obtained.
/// This make sure that we are only using this function on parallel permuted
/// dimensions.
static FailureOr<GenericOp>
packGenericOp(RewriterBase &rewriter, GenericOp genericOp,
              SmallVector<Value> dests,
              SmallVector<AffineMap> packedOutIndexingMaps,
              const PackInfo &packInfo, bool isFoldableUnpackPack) {
  Location loc = genericOp.getLoc();
  SmallVector<Value> inputOperands;
  SmallVector<Value> inputOperandsFromUnpackedSource;
  SmallVector<AffineMap> indexingMaps;
  auto hasEquivalentTiles = [](PackOp packOp, UnPackOp unPackOp) {
    return packOp.getOuterDimsPerm() == unPackOp.getOuterDimsPerm() &&
           packOp.getInnerDimsPos() == unPackOp.getInnerDimsPos() &&
           llvm::equal(packOp.getMixedTiles(), unPackOp.getMixedTiles());
  };
  for (OpOperand *inputOperand : genericOp.getDpsInputOperands()) {
    auto packedViewOrFailure = getOrCreatePackedViewOfOperand(
        rewriter, loc, packInfo, genericOp, inputOperand);
    if (failed(packedViewOrFailure))
      return failure();
    auto [packedOperand, packedIndexingMap] = *packedViewOrFailure;
    auto unpackOp = inputOperand->get().getDefiningOp<linalg::UnPackOp>();
    auto packOp = packedOperand.getDefiningOp<linalg::PackOp>();
    if (packOp && unpackOp && hasEquivalentTiles(packOp, unpackOp)) {
      inputOperandsFromUnpackedSource.push_back(unpackOp.getSource());
    } else {
      inputOperandsFromUnpackedSource.push_back(packedOperand);
    }
    inputOperands.push_back(packedOperand);
    indexingMaps.push_back(packedIndexingMap);
  }

  // If the unpack->pack sequences can be folded, replace use the sources of
  // the unpack ops in any unpack->pack chains on the generic op operands.
  if (isFoldableUnpackPack) {
    inputOperands = inputOperandsFromUnpackedSource;
    for (int i = 0; i < dests.size(); i++) {
      auto dest = dests[i];
      if (auto destPack = dest.getDefiningOp<linalg::PackOp>()) {
        auto destUnPack =
            destPack.getSource().getDefiningOp<linalg::UnPackOp>();
        if (destUnPack && hasEquivalentTiles(destPack, destUnPack)) {
          dests[i] = destUnPack.getSource();
        }
      }
    }
  }

  int64_t numInnerLoops = packInfo.getNumTiledLoops();
  SmallVector<utils::IteratorType> iterTypes =
      genericOp.getIteratorTypesArray();

  for (int64_t i = 0; i < numInnerLoops; ++i)
    if (packInfo.dimPosToReduce.contains(i))
      iterTypes.append(1, utils::IteratorType::reduction);
    else {
      iterTypes.append(1, utils::IteratorType::parallel);
    }

  indexingMaps.append(packedOutIndexingMaps);

  SmallVector<Type> types;
  for (int i = 0; i < dests.size(); i++) {
    types.push_back(dests[i].getType());
  }
  auto newGenericOp = linalg::GenericOp::create(
      rewriter, loc, types, inputOperands, dests, indexingMaps, iterTypes,
      /*bodyBuild=*/nullptr, linalg::getPrunedAttributeList(genericOp));
  rewriter.cloneRegionBefore(genericOp.getRegion(), newGenericOp.getRegion(),
                             newGenericOp.getRegion().begin());
  return newGenericOp;
}

// Different than upstream begin
bool containsDivision(GenericOp genericOp) {
  for (auto &block : genericOp.getRegion()) {
    for (auto &op : block) {
      if (isa<arith::DivFOp>(op) || isa<arith::DivSIOp>(op) ||
          isa<arith::DivUIOp>(op)) {
        return true;
      }
    }
  }
  return false;
}
// Different than upstream end

/// Bubbles up linalg.pack op through a producer generic op. This
/// swap pack(generic) to generic(pack). The new generic op works on packed
/// domain; pack ops are created for input and output operands. E.g.,
///
///     #map0 = affine_map<(d0, d1) -> (d0, d1)>
///     %0 = tensor.dim %arg0, %c0 : tensor<?x?xf32>
///     %1 = tensor.dim %arg0, %c1 : tensor<?x?xf32>
///     %2 = tensor.empty(%0, %1) : tensor<?x?xf32>
///     %3 = linalg.generic {indexing_maps = [#map0, #map0],
///                          iterator_types = ["parallel", "parallel"]}
///         ins(%arg0 : tensor<?x?xf32>)
///         outs(%2 : tensor<?x?xf32>) {
///       ^bb0(%arg3: f32, %arg4: f32):
///         %4 = arith.addf %arg3, %arg3 : f32
///         linalg.yield %4 : f32
///     } -> tensor<?x?xf32>
///     %4 = linalg.pack %3
///       inner_dims_pos = [0, 1]
///       inner_tiles = [8, 2]
///       into %dest : tensor<?x?xf32> -> tensor<?x?x8x2xf32>
///
/// will be converted to
///
///     #map = affine_map<()[s0] -> (s0 ceildiv 8)>
///     #map1 = affine_map<()[s0] -> (s0 ceildiv 2)>
///     #map2 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
///     %dim = tensor.dim %arg0, %c0 : tensor<?x?xf32>
///     %dim_0 = tensor.dim %arg0, %c1 : tensor<?x?xf32>
///     %0 = affine.apply #map()[%dim]
///     %1 = affine.apply #map1()[%dim_0]
///     %2 = tensor.empty(%0, %1) : tensor<?x?x8x2xf32>
///     %pack = linalg.pack %arg0
///       inner_dims_pos = [0, 1]
///       inner_tiles = [8, 2]
///       into %2 : tensor<?x?xf32> -> tensor<?x?x8x2xf32>
///     %3 = linalg.generic {indexing_maps = [#map2, #map2],
///       iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
///       ins(%pack : tensor<?x?x8x2xf32>)
///       outs(%arg1 : tensor<?x?x8x2xf32>) {
///     ^bb0(%in: f32, %out: f32):
///       %4 = arith.addf %in, %in : f32
///       linalg.yield %4 : f32
///     } -> tensor<?x?x8x2xf32>
static FailureOr<
    std::tuple<SmallVector<SmallVector<linalg::PackOp>>, GenericOp>>
bubbleUpPackOpThroughGenericOp(RewriterBase &rewriter, linalg::PackOp packOp,
                               const ControlPropagationFn &controlFn) {
  auto genericOp = packOp.getSource().getDefiningOp<GenericOp>();
  if (!genericOp)
    return failure();

  // User controlled propagation function.
  if (!controlFn(&packOp.getSourceMutable()))
    return failure();

  // TODO: Enable propagation in the presence of linalg.index and
  // tensor.extract, likely as a separate pattern as the pack information and
  // propagation decision needs to be inferred from the region of the generic.
  if (hasGatherSemantics(genericOp))
    return failure();

  // Check for matching types for 2 packops.
  auto haveMatchingTypes = [](linalg::PackOp packOp1,
                              linalg::PackOp packOp2) -> bool {
    RankedTensorType ty1 =
        dyn_cast_or_null<RankedTensorType>(packOp1.getResult().getType());
    RankedTensorType ty2 =
        dyn_cast_or_null<RankedTensorType>(packOp2.getResult().getType());

    if (!ty1 || !ty2)
      return false;

    if (ty1.getShape() != ty2.getShape() ||
        ty1.getEncoding() != ty2.getEncoding())
      return false;

    return true;
  };

  // Bail-out if the result of the generic has multiple uses, as bubbling up
  // creates recomputation if the generic has multiple users.
  // TODO: Enable the case where every use is an identical pack op as no
  // recomputation is needed in that case.
  // For a multi-result generic op,
  // 1. Bail-out when any of the result has more than 1 use.
  // 2. Bail-out when not all the results have packOp users.
  // 3. Bail-out when not all the results have same packed type.
  // Stores the packOp users for all the results.
  SmallVector<SmallVector<linalg::PackOp>> userPackOpsForRes;
  int numResults = genericOp.getNumResults();
  for (int i = 0; i < numResults; i++) {
    auto resValue = genericOp.getResult(i);
    if (resValue.hasNUses(0))
      return failure();

    SmallVector<linalg::PackOp> userPackOps;
    for (OpOperand &use : resValue.getUses()) {
      if (auto packOp = dyn_cast<linalg::PackOp>(use.getOwner())) {
        userPackOps.push_back(packOp);

        // Check for matching pack types
        if (!haveMatchingTypes(packOp, userPackOps[0]))
          return failure();
      } else
        return failure();
    }
    userPackOpsForRes.push_back(userPackOps);
  }

  if (numResults != genericOp.getNumDpsInits())
    return failure();

  SmallVector<Value> dests;
  SmallVector<AffineMap> packedOutIndexingMaps;
  PackInfo packInfo;
  for (int i = 0; i < numResults; i++) {
    linalg::PackOp userPackOp = userPackOpsForRes[i][0];
    // TODO: Add an option for allowing padding values. It could introduce
    // undefined behavior if we unconditionally propagate pack op through all
    // the ops. E.g., if the padding value is zero and there are division ops in
    // a generic op. Some values of padding area could be NaN (0/0).
    // Different than upstream begin
    if (userPackOp.getPaddingValue() and containsDivision(genericOp))
      return failure();
    // Different than upstream end

    OpOperand *initOp = genericOp.getDpsInitOperand(i);
    if (i == 0) {
      auto packingInfo =
          getPackingInfoFromOperand(initOp, genericOp, userPackOp);
      if (failed(packingInfo))
        return failure();
      packInfo = *packingInfo;
    }

    // We want to move the pack not the generic.
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(genericOp);

    // We need to handle two cases:
    // 1) The linalg.pack destination is a tensor.empty. If this is the case, we
    // create a new tensor.empty to avoid breaking dominance, as we are moving
    // the linalg.pack above the linalg.generic. 2) The destination is not a
    // tensor.empty. In this case we can replace only if the destination of the
    // linalg.pack dominates the linalg.generic.
    Value packOpDest = userPackOp.getDest();
    if (!packOpDest.hasOneUse())
      return failure();
    if (auto emptyOp = packOpDest.getDefiningOp<tensor::EmptyOp>()) {
      packOpDest = tensor::EmptyOp::create(rewriter, genericOp->getLoc(),
                                           emptyOp.getMixedSizes(),
                                           emptyOp.getType().getElementType());
    } else {
      DominanceInfo dom(genericOp);
      if (!dom.properlyDominates(packOpDest, genericOp))
        return failure();
    }

    // Rebuild the indexing map for the corresponding init operand.
    auto packedOutViewOrFailure = getOrCreatePackedViewOfOperand(
        rewriter, genericOp.getLoc(), packInfo, genericOp, initOp);
    if (failed(packedOutViewOrFailure))
      return failure();
    auto [packedOutOperand, packedOutIndexingMap] = *packedOutViewOrFailure;

    // If the dps init operand of the generic is a tensor.empty forward the pack
    // op destination.
    Value dest = packedOutOperand;
    if (auto initTensor = genericOp.getDpsInitOperand(i)
                              ->get()
                              .getDefiningOp<tensor::EmptyOp>()) {
      dest = packOpDest;
    }

    dests.push_back(dest);

    packedOutIndexingMaps.push_back(packedOutIndexingMap);
  }

  // pack(unpack) isn't naively foldable because the unpack op can be from
  // an arbitrary domain so we need to keep both.
  auto newGenericOpOrFailure =
      packGenericOp(rewriter, genericOp, dests, packedOutIndexingMaps, packInfo,
                    /*isFoldableUnpackPack=*/false);
  if (failed(newGenericOpOrFailure))
    return failure();
  return std::make_tuple(userPackOpsForRes, *newGenericOpOrFailure);
}

/// Wrapper pattern that applies bubbleUpPackOpThroughGenericOp method.
struct BubbleUpPackOpThroughGenericOpPattern
    : public OpRewritePattern<linalg::PackOp> {
public:
  BubbleUpPackOpThroughGenericOpPattern(MLIRContext *context,
                                        ControlPropagationFn fun)
      : OpRewritePattern<linalg::PackOp>(context), controlFn(std::move(fun)) {}

  LogicalResult matchAndRewrite(linalg::PackOp packOp,
                                PatternRewriter &rewriter) const override {
    auto packOpsAndGeneric =
        bubbleUpPackOpThroughGenericOp(rewriter, packOp, controlFn);
    if (failed(packOpsAndGeneric))
      return failure();

    auto packOps = std::get<0>(*packOpsAndGeneric);
    auto genericOp = std::get<1>(*packOpsAndGeneric);
    for (int i = 0; i < genericOp.getNumResults(); i++) {
      for (int j = 0; j < packOps[i].size(); j++)
        rewriter.replaceOp(packOps[i][j], genericOp->getResult(i));
    }
    return success();
  }

private:
  ControlPropagationFn controlFn;
};

/// Propagate a linalg.pack operation up through a tensor.pad. The idea is to
/// add as many zero padding dimensions in `high` and `low` based on the number
/// of point loops.
class BubbleUpPackThroughPadOp final : public OpRewritePattern<linalg::PackOp> {
public:
  BubbleUpPackThroughPadOp(MLIRContext *context, ControlPropagationFn fun)
      : OpRewritePattern<linalg::PackOp>(context), controlFn(std::move(fun)) {}

  LogicalResult matchAndRewrite(linalg::PackOp packOp,
                                PatternRewriter &rewriter) const override {
    auto padOp = packOp.getSource().getDefiningOp<tensor::PadOp>();
    if (!padOp)
      return failure();

    // User controlled propagation function.
    if (!controlFn(&packOp.getSourceMutable()))
      return failure();

    // TODO: Enable padding when the padding values are the same.
    if (packOp.getPaddingValue())
      return failure();

    // Fail for non-constant padding values. The body of the pad could
    // depend on the padding indices and/or properties of the padded
    // tensor so for now we fail.
    // TODO: Support non-constant padding values.
    Value paddingVal = padOp.getConstantPaddingValue();
    if (!paddingVal)
      return failure();

    if (!packOp.getDest().getDefiningOp<tensor::EmptyOp>())
      return failure();

    ArrayRef<int64_t> innerDimsPos = packOp.getInnerDimsPos();

    // Bail out if one of the padded dimension is a tiled one.
    llvm::SmallBitVector paddedDims = padOp.getPaddedDims();
    llvm::SmallBitVector innerDims(paddedDims.size());
    for (int64_t dim : innerDimsPos)
      innerDims.flip(dim);
    if (paddedDims.anyCommon(innerDims))
      return failure();

    Location loc = padOp->getLoc();
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(padOp);

    ArrayRef<int64_t> outerDimsPerm = packOp.getOuterDimsPerm();
    SmallVector<OpFoldResult> mixedTiles = packOp.getMixedTiles();
    auto empty = linalg::PackOp::createDestinationTensor(
        rewriter, loc, padOp.getSource(), mixedTiles, innerDimsPos,
        outerDimsPerm);
    auto sourcePack = linalg::PackOp::create(
        rewriter, loc, padOp.getSource(), empty, innerDimsPos, mixedTiles,
        /*padding=*/std::nullopt, outerDimsPerm);

    // If we have `outer_dims_perms` we need to adjust the padded dimensions.
    SmallVector<OpFoldResult> lowPad = padOp.getMixedLowPad();
    SmallVector<OpFoldResult> highPad = padOp.getMixedHighPad();
    if (!outerDimsPerm.empty()) {
      applyPermutationToVector<OpFoldResult>(lowPad, outerDimsPerm);
      applyPermutationToVector<OpFoldResult>(highPad, outerDimsPerm);
    }
    // The tiled dimensions were verified to be unpadded above, so here we
    // just append 0 for the inner tile dimensions.
    size_t pointLoopsSize = innerDimsPos.size();
    lowPad.append(pointLoopsSize, rewriter.getIndexAttr(0));
    highPad.append(pointLoopsSize, rewriter.getIndexAttr(0));

    auto newPadOp =
        tensor::PadOp::create(rewriter, loc, Type(), sourcePack.getResult(),
                              lowPad, highPad, paddingVal, padOp.getNofold());

    // If the pad has more than one user, create an unpack on the new pad to
    // replace the other uses.
    if (!padOp->hasOneUse()) {
      auto unpackEmpty = linalg::UnPackOp::createDestinationTensor(
          rewriter, loc, newPadOp, mixedTiles, innerDimsPos, outerDimsPerm);
      UnPackOp unpackedPad =
          linalg::UnPackOp::create(rewriter, loc, newPadOp, unpackEmpty,
                                   innerDimsPos, mixedTiles, outerDimsPerm);
      rewriter.replaceAllUsesExcept(padOp, unpackedPad.getResult(), sourcePack);
    }

    // Replace the pack with the new pad.
    rewriter.replaceOp(packOp, newPadOp.getResult());

    return success();
  }

private:
  ControlPropagationFn controlFn;
};

/// Project dimsPos to the inner-most non-unit dim pos with reassocIndices.
///
/// For example, given dimsPos [0, 2], reassocIndices [[0, 1], [2, 3]], and
/// targetShape [16, 16, 32, 1], it returns [1, 2]. Because for pos 0, the
/// inner-most projected dim in pos [0, 1] is 1. And for pos 2, the inner-most
/// non-unit projected dims in pos [2, 3] is 2.
///
/// If all candidates in a reassociation are unit dims, it chooses the
/// inner-most dim pos.
static SmallVector<int64_t>
projectToInnerMostNonUnitDimsPos(ArrayRef<int64_t> dimsPos,
                                 ArrayRef<ReassociationIndices> reassocIndices,
                                 ArrayRef<int64_t> targetShape) {
  SmallVector<int64_t> projectedDimsPos;
  for (auto pos : dimsPos) {
    // In the case all dims are unit, this will return the inner-most one.
    int64_t projectedPos = reassocIndices[pos].back();
    for (auto i : llvm::reverse(reassocIndices[pos])) {
      int64_t dim = targetShape[i];
      if (dim > 1 || ShapedType::isDynamic(dim)) {
        projectedPos = i;
        break;
      }
    }
    projectedDimsPos.push_back(projectedPos);
  }
  return projectedDimsPos;
}

/// Check if all dims in dimsPos are divisible by the corresponding tile sizes.
static bool isDimsDivisibleByTileSizes(ArrayRef<int64_t> dimsPos,
                                       ArrayRef<int64_t> shape,
                                       ArrayRef<int64_t> tileSizes) {
  for (auto [pos, tileSize] : llvm::zip_equal(dimsPos, tileSizes)) {
    int64_t dim = shape[pos];
    if (ShapedType::isDynamic(dim) || (dim % tileSize) != 0)
      return false;
  }
  return true;
}

/// Permutate the reassociation indices and reindex them in the sequence order.
/// Returns the next dim pos in the sequence.
///
/// For example, given reassocIndices [[0, 1], [2]] and permutation [1, 0], it
/// applies the permutation to get [[2], [0, 1]] and reindexes the indices into
/// [[0], [1, 2]].
static int64_t applyPermutationAndReindexReassoc(
    SmallVector<ReassociationIndices> &reassocIndices,
    ArrayRef<int64_t> permutation) {
  if (!permutation.empty())
    applyPermutationToVector<ReassociationIndices>(reassocIndices, permutation);
  int64_t nextPos = 0;
  for (ReassociationIndices &indices : reassocIndices) {
    for (auto &index : indices) {
      index = nextPos;
      nextPos += 1;
    }
  }
  return nextPos;
}

/// Bubble up pack op through collapse shape op when the packed dims can be
/// projected to the dims before collapsing. This is possible when the inner
/// tile sizes can divide the projected dims.
///
/// For example:
///
/// %collapsed = tensor.collapse_shape %in [[0, 1], 2]
///     : tensor<?x16x4xf32> into tensor<?x4xf32>
/// %pack = linalg.pack %collapsed outer_dims_perm = [0, 1]
///     inner_dims_pos = [0, 1] inner_tiles = [8, 1] into %empty
///     : tensor<?x4xf32> -> tensor<?x4x8x1xf32>
///
/// can be transformed into:
///
/// %pack = linalg.pack %in outer_dims_perm = [1, 2]
///     inner_dims_pos = [1, 2] inner_tiles = [8, 1] into %empty
///     : tensor<?x16x4xf32> -> tensor<?x2x4x8x1xf32>
/// %collapsed = tensor.collapse_shape %pack [[0, 1], 2, 3, 4]
///     : tensor<?x2x4x8x1xf32> into tensor<?x4x8x1>
static LogicalResult
bubbleUpPackOpThroughCollapseShape(tensor::CollapseShapeOp collapseOp,
                                   linalg::PackOp packOp,
                                   PatternRewriter &rewriter) {
  SmallVector<int64_t> innerTileSizes = packOp.getStaticTiles();
  ArrayRef<int64_t> innerDimsPos = packOp.getInnerDimsPos();
  ArrayRef<int64_t> outerDimsPerm = packOp.getOuterDimsPerm();

  ArrayRef<int64_t> srcShape = collapseOp.getSrcType().getShape();
  SmallVector<ReassociationIndices> reassocIndices =
      collapseOp.getReassociationIndices();
  // Project inner tile pos to the dim pos before collapsing. For example, if
  // dims [x, y] is collapsed into [z], packing on dim z can be projected back
  // to pack on dim y.
  //
  // Project to inner-most non-unit dims to increase the chance that they can be
  // divided by the inner tile sizes. This is correct because for [..., x, 1],
  // packing on dim 1 is equivalent to packing on dim x.
  SmallVector<int64_t> projectedInnerDimsPos =
      projectToInnerMostNonUnitDimsPos(innerDimsPos, reassocIndices, srcShape);

  if (!isDimsDivisibleByTileSizes(projectedInnerDimsPos, srcShape,
                                  innerTileSizes)) {
    return failure();
  }
  // Expand the outer dims permutation with the associated source dims for the
  // new permutation after bubbling. This is because moving a collapsed dim is
  // equivalent to moving the associated source dims together.
  SmallVector<int64_t> newOuterDimsPerm;
  for (auto outerPos : outerDimsPerm)
    llvm::append_range(newOuterDimsPerm, reassocIndices[outerPos]);

  auto emptyOp = linalg::PackOp::createDestinationTensor(
      rewriter, packOp.getLoc(), collapseOp.getSrc(), packOp.getMixedTiles(),
      projectedInnerDimsPos, newOuterDimsPerm);
  auto newPackOp = linalg::PackOp::create(
      rewriter, packOp.getLoc(), collapseOp.getSrc(), emptyOp,
      projectedInnerDimsPos, packOp.getMixedTiles(), packOp.getPaddingValue(),
      newOuterDimsPerm);

  SmallVector<ReassociationIndices> newReassocIndices = reassocIndices;
  // First apply the permutation on the reassociations of the outer dims.
  // For example given the permutation [1, 0], the reassociations [[0, 1], [2]]
  // -> [[0], [1, 2]]
  int64_t nextPos =
      applyPermutationAndReindexReassoc(newReassocIndices, outerDimsPerm);
  // Then add direct mapping for the inner tile dims.
  for (size_t i = 0; i < innerDimsPos.size(); ++i) {
    newReassocIndices.push_back({nextPos});
    nextPos += 1;
  }
  Value packedVal = packOp.getResult();
  auto newCollapseOp = tensor::CollapseShapeOp::create(
      rewriter, collapseOp.getLoc(), packOp.getResult().getType(),
      newPackOp.getResult(), newReassocIndices);
  rewriter.replaceOp(packOp, newCollapseOp);

  return success();
}

/// Project dimsPos to their collapsed positions in the reassocIndices.
///
/// For example, given dimsPos [0, 1, 2, 4], and matching reassocIndices
/// [[0], [1, 2], [3], [4]], it returns [0, 1, 1, 3]. Because for pos 0,
/// the reassoc dim [0] is 0. For pos 1 and 2, the reassoc dim in pos
/// [1, 2] is 1. And for pos 4, the reassoc dim [4] is 3.
static SmallVector<int64_t>
projectDimsPosIntoReassocPos(ArrayRef<int64_t> dimsPos,
                             ArrayRef<ReassociationIndices> reassocIndices) {
  SmallVector<int64_t> projectedPos;

  // Map each dimension to the position of corresponding reassociation index.
  for (auto pos : dimsPos) {
    for (auto [idx, indices] : llvm::enumerate(reassocIndices)) {
      // If the dimension is present in the current indices group, the group
      // position within the reassociation map is the desired projected
      // dimension position.
      if (llvm::is_contained(indices, pos)) {
        projectedPos.push_back(idx);
        break;
      }
    }
  }
  assert(projectedPos.size() == dimsPos.size() && "Invalid dim pos projection");

  return projectedPos;
}

/// Bubble up pack op through expand shape op.
///
/// For example:
///
/// %expand = tensor.expand_shape %in [[0], [1, 2]]
///     : tensor<?x64xf32> into tensor<?x4x16xf32>
/// %pack = linalg.pack %expand outer_dims_perm = [0, 1]
///     inner_dims_pos = [2] inner_tiles = [8] into %empty
///     : tensor<?x4x16xf32> -> tensor<?x4x2x8xf32>
///
/// can be transformed into:
///
/// %pack = linalg.pack %in outer_dims_perm = [1, 2]
///     inner_dims_pos = [1] inner_tiles = [8] into %empty
///     : tensor<?x64xf32> -> tensor<?x8x8xf32>
/// %expand = tensor.expand_shape %pack [[0], [1, 2], [3]]
///     : tensor<?x8x8xf32> into tensor<?x4x2x8xf32>
static LogicalResult
bubbleUpPackOpThroughExpandShape(tensor::ExpandShapeOp expandOp,
                                 linalg::PackOp packOp,
                                 PatternRewriter &rewriter) {
  // Outer dimensions permutation is not supported currently.
  // TODO: Handle outer_dims_perm variants.
  ArrayRef<int64_t> outerDimsPerm = packOp.getOuterDimsPerm();
  if (!outerDimsPerm.empty() && !isIdentityPermutation(outerDimsPerm)) {
    return rewriter.notifyMatchFailure(packOp,
                                       "non-identity outer dims perm NYI");
  }

  // Validate dimensions' relations between shape expansion and packing.
  SmallVector<ReassociationIndices, 4> reassoc =
      expandOp.getReassociationIndices();
  ArrayRef<int64_t> packInnerDims = packOp.getInnerDimsPos();
  // Different than upstream begin
  // the difference is small and it might not be even needed since
  // we updated the upstream llvm version used
  llvm::SetVector<int64_t> packDimsPos(packInnerDims.begin(),
                                       packInnerDims.end());
  // Different than upstream end

  for (auto [idx, indices] : llvm::enumerate(reassoc)) {
    // For each expand_shape reassociation, figure out which dimensions get
    // packed if any.
    // Different than upstream begin
    // the difference is small and it might not be even needed since
    // we updated the upstream llvm version used
    llvm::SetVector<int64_t> expandDimPos(indices.begin(), indices.end());
    // Different than upstream end
    llvm::SetVector<int64_t> packedDims =
        llvm::set_intersection(packDimsPos, expandDimPos);

    // The expanded dimension is not packed so, it does not affect moving pack
    // before shape expansion - simply continue.
    if (packedDims.empty())
      continue;
    // Shape expansion cannot be propagated when multiple expanded dimension are
    // packed - in this case operation reordering would affect final element
    // positions and/or shapes can no longer be projected.
    if (packedDims.size() != 1)
      return rewriter.notifyMatchFailure(
          packOp, "only one of the expanded dimensions can be packed");
    // Only the inner-most expanded dimension should be packed. Otherwise,
    // elements order will be affected after operation reordering.
    if (packedDims.front() != indices.back())
      return rewriter.notifyMatchFailure(
          packOp, "can only pack the inner-most expanded dimension");
  }

  // Project pack.inner_dims_pos to positions before shape expansion.
  SmallVector<int64_t> projectedInnerDimsPos =
      projectDimsPosIntoReassocPos(packInnerDims, reassoc);

  // Project the shape expansion to new packed shape.
  // The pack.outer_dims_perm is restricted to identity so, the permutation can
  // be omitted for simplicity.
  // TODO: Account for outer dimensions permutation.
  //
  // If reassociation is not possible, then reordering cannot happen.
  // This can be caused by pack padding affecting previously expanded
  // dimensions or packing extending dimensions.
  RankedTensorType newPackType = linalg::PackOp::inferPackedTensorType(
      expandOp.getSrcType(), packOp.getStaticInnerTiles(),
      projectedInnerDimsPos, /*outerDimsPerm=*/SmallVector<int64_t>{});
  auto reassocExpand =
      getReassociationIndicesForReshape(newPackType, packOp.getDestType());
  if (!reassocExpand)
    return rewriter.notifyMatchFailure(
        packOp, "could not reassociate dims after bubbling up");

  Value destTensor = linalg::PackOp::createDestinationTensor(
      rewriter, packOp.getLoc(), expandOp.getSrc(), packOp.getMixedTiles(),
      projectedInnerDimsPos, /*outerDimsPerm=*/SmallVector<int64_t>{});
  PackOp packedVal = linalg::PackOp::create(
      rewriter, packOp.getLoc(), expandOp.getSrc(), destTensor,
      projectedInnerDimsPos, packOp.getMixedTiles(), packOp.getPaddingValue(),
      /*outerDimsPerm=*/SmallVector<int64_t>{});

  Value newExpandOp = tensor::ExpandShapeOp::create(
      rewriter, packOp.getLoc(), packOp.getDestType(), packedVal.getResult(),
      *reassocExpand);
  rewriter.replaceOp(packOp, newExpandOp);

  return success();
}

class BubbleUpPackOpThroughReshapeOp final
    : public OpRewritePattern<linalg::PackOp> {
public:
  BubbleUpPackOpThroughReshapeOp(MLIRContext *context, ControlPropagationFn fun)
      : OpRewritePattern<linalg::PackOp>(context), controlFn(std::move(fun)) {}

  LogicalResult matchAndRewrite(linalg::PackOp packOp,
                                PatternRewriter &rewriter) const override {
    Operation *srcOp = packOp.getSource().getDefiningOp();
    // Currently only support when the pack op is the only user.
    if (!srcOp || !(srcOp->getNumResults() == 1) ||
        !srcOp->getResult(0).hasOneUse()) {
      return failure();
    }
    // Currently only support static inner tile sizes.
    if (llvm::any_of(packOp.getStaticTiles(), ShapedType::isDynamic))
      return failure();

    // User controlled propagation function.
    if (!controlFn(&packOp.getSourceMutable()))
      return failure();

    return TypeSwitch<Operation *, LogicalResult>(srcOp)
        .Case([&](tensor::CollapseShapeOp op) {
          return bubbleUpPackOpThroughCollapseShape(op, packOp, rewriter);
        })
        .Case([&](tensor::ExpandShapeOp op) {
          return bubbleUpPackOpThroughExpandShape(op, packOp, rewriter);
        })
        .Default([](Operation *) { return failure(); });
  }

private:
  ControlPropagationFn controlFn;
};

/// Push down unpack op through expand shape op when the packed dims can be
/// projected to the dims after expanding. This is possible when the inner tile
/// sizes can divide the projected dims.
///
/// For example:
///
/// %unpack = linalg.unpack %in outer_dims_perm = [0, 1]
///     inner_dims_pos = [0, 1] inner_tiles = [8, 8] into %empty
///     : tensor<?x32x8x8xf32> -> tensor<?x256xf32>
/// %expanded = tensor.expand_shape %unpack [[0, 1], [2]]
///     : tensor<?x256xf32> into tensor<?x256x256xf32>
///
/// can be transformed into:
///
/// %expanded = tensor.expand_shape %ain [[0, 1], [2], [3], [4]]
///     : tensor<?x32x8x8xf32> into tensor<?x32x32x8x8xf32>
/// %unpack = linalg.unpack %expanded outer_dims_perm = [0, 1, 2]
///     inner_dims_pos = [1, 2] inner_tiles = [8, 8] into %empty
///     : tensor<?x32x32x8x8xf32> -> tensor<?x256x256xf32>
static LogicalResult pushDownUnPackOpThroughExpandShape(
    linalg::UnPackOp unPackOp, tensor::ExpandShapeOp expandOp,
    PatternRewriter &rewriter, ControlPropagationFn controlFn) {
  // User controlled propagation function.
  if (!controlFn(&expandOp.getSrcMutable()))
    return failure();

  SmallVector<int64_t> innerTileSizes = unPackOp.getStaticTiles();
  ArrayRef<int64_t> innerDimsPos = unPackOp.getInnerDimsPos();
  ArrayRef<int64_t> outerDimsPerm = unPackOp.getOuterDimsPerm();

  auto expandTy = dyn_cast<RankedTensorType>(expandOp.getType());
  if (!expandTy)
    return failure();
  ArrayRef<int64_t> dstShape = expandTy.getShape();
  SmallVector<ReassociationIndices> reassocIndices =
      expandOp.getReassociationIndices();
  // Project inner tile pos to the dim pos after expanding. For example, if dims
  // [z] is expanded into [x, y], unpacking on dim z can be projected to unpack
  // on dim y.
  //
  // Project to inner-most non-unit dims to increase the chance that they can be
  // divided by the inner tile sizes. This is correct because for [..., x, 1],
  // unpacking on dim 1 is equivalent to unpacking on dim x.
  SmallVector<int64_t> projectedInnerDimsPos =
      projectToInnerMostNonUnitDimsPos(innerDimsPos, reassocIndices, dstShape);

  if (!isDimsDivisibleByTileSizes(projectedInnerDimsPos, dstShape,
                                  innerTileSizes)) {
    return failure();
  }
  // Expand the outer dims permutation with the associated expanded dims for the
  // new permutation after pushing. This is because moving a source dim is
  // equivalent to moving the associated expanded dims together.
  SmallVector<int64_t> newOuterDimsPerm;
  for (auto outerPos : outerDimsPerm)
    llvm::append_range(newOuterDimsPerm, reassocIndices[outerPos]);

  SmallVector<ReassociationIndices> newReassocIndices = reassocIndices;
  // First apply the permutation on the reassociations of the outer dims.
  // For example given the permutation [1, 0], the reassociations [[0, 1], [2]]
  // -> [[0], [1, 2]]
  int64_t nextPos =
      applyPermutationAndReindexReassoc(newReassocIndices, outerDimsPerm);
  // Then add direct mapping for the inner tile dims.
  for (size_t i = 0; i < innerDimsPos.size(); ++i) {
    newReassocIndices.push_back({nextPos});
    nextPos += 1;
  }

  RankedTensorType newExpandType = linalg::PackOp::inferPackedTensorType(
      expandTy, innerTileSizes, projectedInnerDimsPos, newOuterDimsPerm);
  auto newExpandOp =
      tensor::ExpandShapeOp::create(rewriter, expandOp.getLoc(), newExpandType,
                                    unPackOp.getSource(), newReassocIndices);

  auto emptyOp = linalg::UnPackOp::createDestinationTensor(
      rewriter, unPackOp.getLoc(), newExpandOp, unPackOp.getMixedTiles(),
      projectedInnerDimsPos, newOuterDimsPerm);
  auto newUnPackOp = linalg::UnPackOp::create(
      rewriter, unPackOp.getLoc(), newExpandOp.getResult(), emptyOp,
      projectedInnerDimsPos, unPackOp.getMixedTiles(), newOuterDimsPerm);
  rewriter.replaceOp(expandOp, newUnPackOp);

  return success();
}

class PushDownUnPackOpThroughReshapeOp final
    : public OpRewritePattern<linalg::UnPackOp> {
public:
  PushDownUnPackOpThroughReshapeOp(MLIRContext *context,
                                   ControlPropagationFn fun)
      : OpRewritePattern<linalg::UnPackOp>(context), controlFn(std::move(fun)) {
  }

  LogicalResult matchAndRewrite(linalg::UnPackOp unPackOp,
                                PatternRewriter &rewriter) const override {
    Value result = unPackOp.getResult();
    // Currently only support unpack op with the single user.
    if (!result.hasOneUse()) {
      return failure();
    }
    // Currently only support static inner tile sizes.
    if (llvm::any_of(unPackOp.getStaticTiles(), ShapedType::isDynamic))
      return failure();

    Operation *consumerOp = *result.user_begin();
    return TypeSwitch<Operation *, LogicalResult>(consumerOp)
        .Case([&](tensor::ExpandShapeOp op) {
          return pushDownUnPackOpThroughExpandShape(unPackOp, op, rewriter,
                                                    controlFn);
        })
        .Default([](Operation *) { return failure(); });
  }

private:
  ControlPropagationFn controlFn;
};

struct UnpackedOperandInfo {
  OpOperand *operand;        // the operand to the generic op
  linalg::UnPackOp unPackOp; // the unpack op defining the operand
};

static FailureOr<SmallVector<UnpackedOperandInfo>>
getUnPackedOperands(GenericOp genericOp) {
  SmallVector<UnpackedOperandInfo> unPackedOperands;
  for (OpOperand &operand : genericOp->getOpOperands()) {
    if (auto unPackOp = operand.get().getDefiningOp<linalg::UnPackOp>()) {
      unPackedOperands.push_back({&operand, unPackOp});
    }
  }
  if (unPackedOperands.empty())
    return failure();
  return unPackedOperands;
}

/// Push down a linalg.unpack op through a generic op.
/// The new generic op works on packed domain; pack ops are created for input
/// and output operands. A linalg.unpack op is inserted right after the packed
/// generic. E.g.
///
/// #map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
///
/// %arg0 = tensor<12x2x56x56x32xf32> // packed arg.
///
/// %0 = tensor.empty() : tensor<12x56x56x64xf32>
/// %1 = linalg.unpack %arg0 outer_dims_perm = [0, 3, 1, 2]
///                          inner_dims_pos = [3] inner_tiles = [32] into %0
/// %2 = linalg.generic {indexing_maps = [#map],
///      iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
///      outs(%1 : tensor<12x56x56x64xf32>) {
///      ^bb0(%out : f32):
///         linalg.yield %out : f32
///      } -> tensor<12x56x56x64xf32>
///
/// will be converted to
///
/// #map = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
///
/// %0 = tensor.empty() : tensor<12x56x56x64xf32>
/// %1 = linalg.generic {indexing_maps = [#map],
///      iterator_types = ["parallel", "parallel", "parallel",
///                        "parallel", "parallel"]}
///      outs(%arg0 : tensor<12x2x56x56x32xf32>) {
///      ^bb0(%out : f32):
///         linalg.yield %out : f32
///      } -> tensor<12x2x56x56x32xf32>
/// %2 = linalg.unpack %1 outer_dims_perm = [0, 3, 1, 2]
///                       inner_dims_pos = [3] inner_tiles = [32] into %0
///
static FailureOr<std::tuple<GenericOp, SmallVector<Value>>>
pushDownUnPackOpThroughGenericOp(RewriterBase &rewriter, GenericOp genericOp,
                                 ControlPropagationFn controlFn) {
  if (hasGatherSemantics(genericOp))
    return failure();

  // Collect the unPacked operand, if present.
  auto unpackedOperands = getUnPackedOperands(genericOp);
  if (failed(unpackedOperands))
    return failure();

  SmallVector<UnpackedOperandInfo> unPackedOperands = *unpackedOperands;
  // Respect user control function.
  for (auto &info : unPackedOperands) {
    if (!controlFn(info.operand))
      return failure();
  }
  // Use the first unpack-fed operand as reference.
  UnpackedOperandInfo &ref = unPackedOperands.front();

  auto refOperandType =
      dyn_cast<RankedTensorType>(ref.operand->get().getType());
  auto refUnpackType =
      dyn_cast<RankedTensorType>(ref.unPackOp.getResult().getType());
  if (!refOperandType || !refUnpackType)
    return failure();

  auto packInfo =
      getPackingInfoFromOperand(ref.operand, genericOp, ref.unPackOp);
  if (failed(packInfo))
    return failure();

  // All other unpack-fed operands must be compatible with the reference one.
  //  - Same inner dims pos.
  //  - Same tile sizes
  //  - Same indexing map structure
  //  - Same outer dims perm

  ArrayRef<int64_t> refInnerDimsPos = ref.unPackOp.getInnerDimsPos();
  auto refTiles = ref.unPackOp.getMixedTiles();
  auto refOuterDimsPerm = ref.unPackOp.getOuterDimsPerm();
  AffineMap refIndexingMap = genericOp.getMatchingIndexingMap(ref.operand);

  for (auto &info : llvm::drop_begin(unPackedOperands)) {
    // types must match the reference one.
    auto operandType =
        dyn_cast<RankedTensorType>(info.operand->get().getType());
    if (!operandType || operandType != refUnpackType)
      return failure();

    if (info.unPackOp.getInnerDimsPos() != refInnerDimsPos)
      return failure();
    if (info.unPackOp.getMixedTiles() != refTiles)
      return failure();
    if (info.unPackOp.getOuterDimsPerm() != refOuterDimsPerm)
      return failure();

    AffineMap currMap = genericOp.getMatchingIndexingMap(info.operand);
    if (currMap != refIndexingMap)
      return failure();
  }

  // Holds the DPS output values
  SmallVector<Value> dests;
  // Holds the packed version of dests
  SmallVector<PackOp> destPacks;
  SmallVector<AffineMap> destIndexingMaps;
  int numDpsInits = genericOp.getNumDpsInits();
  for (int i = 0; i < numDpsInits; i++) {
    auto dpsInitOp = genericOp.getDpsInitOperand(i);
    // Rebuild the indexing map for the corresponding init operand.
    auto packedOutViewOrFailure = getOrCreatePackedViewOfOperand(
        rewriter, genericOp.getLoc(), *packInfo, genericOp, dpsInitOp);
    if (failed(packedOutViewOrFailure))
      return failure();
    auto [packedOutOperand, packedOutIndexingMap] = *packedOutViewOrFailure;
    auto destPack = packedOutOperand.getDefiningOp<linalg::PackOp>();

    // If the dps init operand of the generic is a tensor.empty, do not pack it
    // and forward the new tensor.empty as a destination.
    Value dest = packedOutOperand;
    if (auto initTensor = dpsInitOp->get().getDefiningOp<tensor::EmptyOp>()) {
      if (destPack)
        dest = destPack.getDest();
    }

    dests.push_back(dest);
    destPacks.push_back(destPack);
    destIndexingMaps.push_back(packedOutIndexingMap);
  }

  // Pack the genericOp.
  // pack(unpack) is foldable in this case. This is because in pushing down the
  // unpack, by default we will populate an additional pack op after the unpack.
  // This guarantees them to be foldable.
  auto newGenericOpOrFailure =
      packGenericOp(rewriter, genericOp, dests, destIndexingMaps, *packInfo,
                    /*isFoldableUnpackPack=*/true);
  if (failed(newGenericOpOrFailure))
    return failure();
  GenericOp newGenericOp = *newGenericOpOrFailure;

  // Holds the unpack Op of the generic op results
  SmallVector<Value> unPackOpResults;
  for (int i = 0; i < numDpsInits; i++) {
    Value newResult =
        newGenericOp.getTiedOpResult(newGenericOp.getDpsInitOperand(i));

    auto destPack = destPacks[i];
    // If the output is unaffected, no need to unpack.
    if (!destPack) {
      unPackOpResults.push_back(newResult);
      continue;
    }

    auto mixedTiles = destPack.getMixedTiles();
    auto innerDimsPos = destPack.getInnerDimsPos();
    auto outerDimsPerm = destPack.getOuterDimsPerm();

    // Insert an unPackOp right after the packed generic.
    Value unPackOpRes =
        linalg::UnPackOp::create(rewriter, genericOp.getLoc(), newResult,
                                 destPack.getSource(), innerDimsPos, mixedTiles,
                                 outerDimsPerm)
            .getResult();

    unPackOpResults.push_back(unPackOpRes);
  }
  return std::make_tuple(newGenericOp, unPackOpResults);
}

// Wrapper pattern that applies pushDownUnPackOpThroughGenericOp method.
struct PushDownUnPackOpThroughGenericOp : public OpRewritePattern<GenericOp> {
public:
  PushDownUnPackOpThroughGenericOp(MLIRContext *context,
                                   ControlPropagationFn fun)
      : OpRewritePattern<GenericOp>(context), controlFn(std::move(fun)) {}

  LogicalResult matchAndRewrite(GenericOp genericOp,
                                PatternRewriter &rewriter) const override {
    auto genericAndRepl =
        pushDownUnPackOpThroughGenericOp(rewriter, genericOp, controlFn);
    if (failed(genericAndRepl))
      return failure();
    rewriter.replaceOp(genericOp, std::get<1>(*genericAndRepl));
    return success();
  }

private:
  ControlPropagationFn controlFn;
};

/// Propagate a linalg.unpack operation through a tensor.pad. The idea is to
/// add as many zero padding dimensions in `high` and `low` based on the number
/// of point loops.
struct PushDownUnPackThroughPadOp : public OpRewritePattern<tensor::PadOp> {
  PushDownUnPackThroughPadOp(MLIRContext *context, ControlPropagationFn fun)
      : OpRewritePattern<tensor::PadOp>(context), controlFn(std::move(fun)) {}

  LogicalResult matchAndRewrite(tensor::PadOp padOp,
                                PatternRewriter &rewriter) const override {
    linalg::UnPackOp unpackOp =
        padOp.getSource().getDefiningOp<linalg::UnPackOp>();
    if (!unpackOp)
      return failure();

    if (!controlFn(&padOp.getSourceMutable()))
      return failure();

    Location loc = padOp.getLoc();
    // Bail out if one of the padded dimension is a tiled one.
    llvm::SmallBitVector paddedDims = padOp.getPaddedDims();
    ArrayRef<int64_t> innerDimsPos = unpackOp.getInnerDimsPos();
    llvm::SmallBitVector innerDims(paddedDims.size());
    for (int64_t dim : innerDimsPos)
      innerDims.flip(dim);
    if (paddedDims.anyCommon(innerDims))
      return failure();

    Value paddingVal = padOp.getConstantPaddingValue();
    if (!paddingVal)
      return failure();

    // If we have `outer_dims_perms` we need to adjust the padded dimensions.
    ArrayRef<int64_t> outerDimsPerm = unpackOp.getOuterDimsPerm();
    SmallVector<OpFoldResult> lowPad = padOp.getMixedLowPad();
    SmallVector<OpFoldResult> highPad = padOp.getMixedHighPad();
    if (!outerDimsPerm.empty()) {
      applyPermutationToVector<OpFoldResult>(lowPad, outerDimsPerm);
      applyPermutationToVector<OpFoldResult>(highPad, outerDimsPerm);
    }
    // Add zero padding for the point loops.
    size_t pointLoopsSize = innerDimsPos.size();
    lowPad.append(pointLoopsSize, rewriter.getIndexAttr(0));
    highPad.append(pointLoopsSize, rewriter.getIndexAttr(0));

    auto newPadOp = tensor::PadOp::create(rewriter, loc, /*result=*/Type(),
                                          unpackOp.getSource(), lowPad, highPad,
                                          paddingVal, padOp.getNofold());

    // Inject the linalg.unpack right after the packed padOp.
    Value outputUnPack =
        tensor::EmptyOp::create(rewriter, loc, padOp.getResultType().getShape(),
                                padOp.getResultType().getElementType());

    UnPackOp replacement = linalg::UnPackOp::create(
        rewriter, loc, newPadOp.getResult(), outputUnPack, innerDimsPos,
        unpackOp.getMixedTiles(), outerDimsPerm);
    rewriter.replaceOp(padOp, replacement);
    return success();
  }

private:
  ControlPropagationFn controlFn;
};

// Different than upstream begin
struct HexagonExtendPackPass
    : public ::impl::HexagonExtendPackBase<HexagonExtendPackPass> {
  explicit HexagonExtendPackPass(const HexagonExtendPackOptions &options)
      : HexagonExtendPackBase(options) {}
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<func::FuncDialect>();
  }

  void runOnOperation() override {
    auto funcOp = getOperation();
    auto context = &getContext();
    RewritePatternSet propagationPatterns(context);

    auto upperFrontierFn = [&](OpOperand *opOperand) {
      if (not upperFrontier)
        return false;

      auto linalgOp = opOperand->get().getDefiningOp<linalg::LinalgOp>();

      if (!linalgOp)
        return true;

      bool allIteratorsAreParallel = llvm::all_of(
          linalgOp.getIteratorTypesArray(), linalg::isParallelIterator);

      if ((not parallelsOnly) or allIteratorsAreParallel)
        return true;

      return false;
    };

    auto lowerFrontierFn = [&](OpOperand *opOperand) {
      if (not lowerFrontier)
        return false;

      auto operandUses = (opOperand->get()).getUses();
      for (auto &use : operandUses) {
        auto genericUse = dyn_cast<linalg::GenericOp>(use.getOwner());

        if (!genericUse)
          continue;
        bool allIteratorsAreParallel = llvm::all_of(
            genericUse.getIteratorTypesArray(), linalg::isParallelIterator);
        if (parallelsOnly and (not allIteratorsAreParallel))
          return false;
      }
      return true;
    };

    propagationPatterns
        .add<BubbleUpPackOpThroughGenericOpPattern, BubbleUpPackThroughPadOp,
             BubbleUpPackOpThroughReshapeOp>(context, upperFrontierFn);

    propagationPatterns
        .add<PushDownUnPackOpThroughReshapeOp, PushDownUnPackOpThroughGenericOp,
             PushDownUnPackThroughPadOp>(context, lowerFrontierFn);

    // Different than upstream begin
    propagationPatterns.add<PushDownCollapseShapeOpThroughGenericOp>(
        context, lowerFrontierFn);

    propagationPatterns.add<BubbleUpExpandShapeOpThroughGenericOp>(
        context, upperFrontierFn);
    // Different than upstream end

    auto config = GreedyRewriteConfig();
    if (failed(applyPatternsGreedily(funcOp, std::move(propagationPatterns),
                                     config))) {
      return signalPassFailure();
    }
  }
};
} // namespace

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
hexagon::createHexagonExtendPackPass(const HexagonExtendPackOptions &options) {
  return std::make_unique<HexagonExtendPackPass>(options);
}
// Different than upstream end
