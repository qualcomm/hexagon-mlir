//===--- AffinePipelineFusionPass.cpp - Affine Pipeline Fusion Pass ---=======//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// This pass implements pipeline fusion of affine loops.
// It fuses together source loops with a single output and any perfectly nested
// destination loop.
//
//===----------------------------------------------------------------------===//

/*
 * This pass performs loop fusion without duplicating work, by taking ideas from
 * polyhedral scheduling. The pass schedules producer iterations in the consumer
 * iteration space, subject to dependence constraints, while maximizing temporal
 * locality. The dependence constraints require that producer iterations must be
 * scheduled before the consumer iterations that use those results. By using the
 * dependence polyhedron formed between producer and consumer iterations, we can
 * formalize this constraint by requiring that the schedule is a lower bound (in
 * the consumer iteration space). Then maximum locality is obtained if the lower
 * bound is as tight as possible. Solving this in full generality requires full-
 * blown polyhedral scheduling (Farkas's lemma, etc.), but we if restrict bounds
 * to be constraints in the original polyhedron, this problem is more tractable.
 * Similar to regular scheduling, we also have the requirement that the schedule
 * for each consumer dimension be linearly independent; schedules with dependent
 * dimensions place a whole loop at a single iteration. In such cases, we simply
 * terminate the fusion at that depth. Overall, the fusion process is summarized
 * in the following steps:
 *
 * 0. Find suitable loops to fuse. For each consumer (destination) loop, we find
 * the loops that write to the memrefs read in that loop. Then we check that the
 * producer loop has no cyclic dependences, and precedes the destination.
 *
 * 1. Check fusion preconditions (checkFusionPreconditions).
 *
 * 2. Compute dependence polyhedron (computeSrcToDstRelation). This computes the
 * access relation (loop iteration --> memref dimensions) for each loop and then
 * composes dst . src^-1.
 *
 * 3. Compute optimal schedule (computeSchedule). For each consumer dimension we
 * find the set of lower bounds. Then we compute the schedule with maximal depth
 * using matroid intersection: the ground set is the set of possible bounds, the
 * first matroid comes from the linear independence constraint (linear matroid),
 * and the second matroid comes from the constraint that each consumer dimension
 * can have at most one bound (partition matroid). Also, we restrict tiled loops
 * to only fuse the inter-tile loops, and we tighten bounds using the tile size.
 * Finally, we compute the Hermite normal form of the schedule.
 *
 * 4. Perform pipeline fusion using the computed schedule (pipelineFuse). First,
 * we insert a copy of the producer loop restricted to the domain not covered by
 * the scheduled iterations in the consumer loop. Then we copy the producer body
 * into the consumer loop using the computed schedule, copying the loops for non
 * pivot dimensions. This is guarded by a branch to ensure we are in bounds.
 */

#include "mlir/Analysis/Presburger/IntegerRelation.h"
#include "mlir/Analysis/Presburger/PresburgerSpace.h"
#include "mlir/Analysis/Presburger/Simplex.h"
#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/LoopAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include <memory>
#include <optional>

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/bit.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"

#include "hexagon/Conversion/AffineToLLVM/Passes.h"
#include "hexagon/Conversion/AffineToLLVM/Utils/MatroidIntersection.h"

#define DEBUG_TYPE "affine-pipeline-fusion"

#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define DBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
using namespace hexagon;
using mlir::func::FuncOp;
using namespace mlir::affine;
using namespace mlir::presburger;

#define GEN_PASS_DEF_AFFINEPIPELINEFUSION
#include "hexagon/Conversion/AffineToLLVM/Passes.h.inc"

namespace {

struct AffinePipelineFusionPass
    : public ::impl::AffinePipelineFusionBase<AffinePipelineFusionPass> {
public:
  AffinePipelineFusionPass() : Base() {}

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<func::FuncDialect, affine::AffineDialect,
                    memref::MemRefDialect>();
  }

  void runOnBlock(Block *);
  void runOnOperation() override;
};

/**
 * Copies given integer slice to Fraction array.
 * @param slice Input integer slice.
 * @param[out] out Output array to be filled with values from slice.
 */
inline void copySliceToFracVec(ArrayRef<DynamicAPInt> slice,
                               MutableArrayRef<Fraction> out) {
  std::transform(slice.begin(), slice.end(), out.begin(),
                 [](const DynamicAPInt &x) { return Fraction(x); });
}

/**
 * Row reduces the given row using the span matrix, and adds as a new row.
 * Returns true if a new row was added (row is linearly independent).
 * @param span The matrix to use for row reduction.
 * @param row The row to be reduced.
 * @return true if row is linearly independent, false otherwise.
 */
bool rowReduceAndAdd(FracMatrix &span, MutableArrayRef<Fraction> row) {
  unsigned R = span.getNumRows(), C = row.size();
  if (R != 0)
    assert(span.getNumColumns() == C);
  // RREF matrix is full rank.
  if (R >= C)
    return false;
  unsigned pivot = 0;
  unsigned insertPos = R;
  unsigned insertPivot = C;
  // Check if zero first.
  bool isZero = true;
  for (const Fraction &x : row) {
    if (x != 0)
      isZero = false;
  }
  if (isZero)
    return false;
  // Reduce row, find insert position.
  for (unsigned i = 0; i < R && pivot < C; i++) {
    while (pivot < C && span.at(i, pivot) == 0) {
      if (insertPos == R && row[pivot] != 0) {
        insertPos = i;
        insertPivot = pivot;
      }
      pivot++;
    }
    if (pivot == C) {
      break;
    }
    Fraction scale = row[pivot];
    if (scale != 0) {
      if (insertPos == R)
        isZero = true;
      for (unsigned j = pivot; j < C; j++) {
        row[j] -= scale * span.at(i, j);
        if (row[j] != 0)
          isZero = false;
      }
      if (isZero)
        return false;
    }
    pivot++;
  }
  while (insertPos == R && pivot < C) {
    if (row[pivot] != 0) {
      insertPivot = pivot;
      break;
    }
    pivot++;
  }
  if (insertPivot == C) {
    return false;
  }
  // Reduce row.
  Fraction scale = row[insertPivot];
  assert(scale != 0);
  for (Fraction &x : row) {
    x /= scale;
  }
  if (R == 0) {
    span.resize(R + 1, C);
  } else {
    span.insertRow(insertPos);
  }
  span.setRow(insertPos, row);
  return true;
}

/**
 * Returns the index of a lower bound for var, of -1 if unconstrained below.
 * Selected coefficients from domainStart to domainEnd must be linearly
 * independent from rows of span, and the resulting vector is added to span.
 * @param domain The polyhedron from which to select a lower bound.
 * @param var The variable for which to select a lower bound.
 * @param domainStart The start of the variables that must be linearly
 * independent.
 * @param domainEnd The end of the linearly independent variables.
 * @param span Matrix of previously added bound coefficients.
 * @param[out] isEquality Set to true if the added bound is an equality, false
 * if inequality.
 */
int selectConstrainingLowerBound(FlatAffineValueConstraints &domain, int var,
                                 int domainStart, int domainEnd,
                                 FracMatrix &span, bool &isEquality) {
  LLVM_DEBUG(domain.dump());
  llvm::SmallVector<unsigned> lbIndices, ubIndices, eqIndices;
  llvm::SmallVector<Fraction> vec;
  vec.resize(domainEnd - domainStart);
  domain.getLowerAndUpperBoundIndices(var, &lbIndices, &ubIndices, &eqIndices);
  if (!eqIndices.empty()) {
    LLVM_DEBUG({
      llvm::dbgs() << "eq for var " << var << ": ";
      domain.dumpRow(domain.getEquality64(eqIndices[0]));
      llvm::dbgs() << "\n";
    });
    isEquality = true;
    copySliceToFracVec(domain.getEquality(eqIndices[0])
                           .slice(domainStart, domainEnd - domainStart),
                       vec);
    if (!rowReduceAndAdd(span, vec)) {
      return -1;
    }
    return eqIndices[0];
  }
  isEquality = false;
  if (lbIndices.empty())
    return -1;
  int bestNonZero = -1;
  // Two passes -- first pass only considers +-1 coefs.
  for (int pass = 0; pass < 2; pass++) {
    for (int i : lbIndices) {
      LLVM_DEBUG({
        llvm::dbgs() << "lb[" << i << "] for var " << var << ": ";
        domain.dumpRow(domain.getInequality64(i));
        llvm::dbgs() << "\n";
      });
      for (int j = domainStart; j < domainEnd; j++) {
        auto val = domain.getInequality(i)[j];
        if (pass == 0 && val != 1 && val != -1) {
          continue;
        }
        copySliceToFracVec(
            domain.getInequality(i).slice(domainStart, domainEnd - domainStart),
            vec);
        if (rowReduceAndAdd(span, vec)) {
          return i;
        }
      }
    }
  }
  return -1;
}

/**
 * Removes loops after the last tiled loop, if the outermost loop is tiled.
 * Also projects out the corresponding variables in the given relation.
 * @param loopIVs The loops to consider.
 * @param rel Optional relation to project out.
 * @param[out] outDepth Set to the number of tiled loops, or `loopIVs.size()`.
 * @param relVarKind The type of variables to project out.
 * @param[out] isTiled Set to true if the outermost loop is tiled.
 * @return failure() if fusion is judged to be impossible.
 */
LogicalResult restrictToTiledLoops(SmallVectorImpl<AffineForOp> &loopIVs,
                                   IntegerRelation *rel, unsigned *outDepth,
                                   bool *isTiled,
                                   VarKind relVarKind = VarKind::Domain) {
  if (loopIVs[0].getStep() == 1) {
    if (outDepth) {
      *outDepth = loopIVs.size();
    }
    if (isTiled) {
      *isTiled = false;
    }
    return success();
  }
  unsigned depth = 1;
  while (depth < loopIVs.size() && loopIVs[depth].getStep() != 1) {
    depth++;
  }
  loopIVs.resize(depth);
  if (rel) {
    rel->projectOut(rel->getVarKindOffset(relVarKind) + depth,
                    rel->getNumVarKind(relVarKind) - depth);
  }
  if (outDepth) {
    *outDepth = depth;
  }
  // Make sure we have constant lower bound; we require exact representations of
  // domains.
  for (AffineForOp &loop : loopIVs) {
    if (!loop.hasConstantLowerBound()) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Tiled loop " << loop
                 << " does not have constant lower bound; stopping\n");
      return failure();
    }
  }
  if (isTiled) {
    *isTiled = true;
  }
  return success();
}

/**
 * Computes the relation between src loop iterations and dependent dst loop
 * iterations.
 * @param opsA Source operations to analyze
 * @param opsB Destination operations to analyze
 * @param[inout] loopDepth The depth of loops to consider. May be reduced if
 * tiled.
 * @param numCommonLoops Number of common surrounding loops
 * @param[out] dstElim Vector where dstElim[i] will contain the relation with
 * dst IVs after i eliminated
 * @param[out] srcLoopIVs Will be populated with the source loop induction
 * variables
 * @param[out] dstLoopIVs Will be populated with the destination loop induction
 * variables
 * @param[out] srcUsed Will be set to the space of used src indices.
 * @return success if the relation could be computed, failure otherwise
 */
LogicalResult computeSrcToDstRelation(
    ArrayRef<Operation *> opsA, ArrayRef<Operation *> opsB, unsigned &loopDepth,
    unsigned numCommonLoops, SmallVectorImpl<IntegerRelation> &dstElim,
    SmallVectorImpl<AffineForOp> &srcLoopIVs,
    SmallVectorImpl<AffineForOp> &dstLoopIVs, IntegerPolyhedron &srcUsed) {
  // Only allow one dependent src operation
  Operation *srcOp = nullptr;
  PresburgerSpace space = PresburgerSpace::getRelationSpace();
  IntegerRelation srcRel(space), mergedDstRel(space);
  SmallVector<Operation *, 2> dependentOps;
  Value srcMemref;
  for (Operation *a : opsA) {
    MemRefAccess srcAccess(a);
    bool hasDep = false;
    for (Operation *b : opsB) {
      MemRefAccess dstAccess(b);
      if (srcAccess.memref != dstAccess.memref)
        continue;
      if (loopDepth > getNestingDepth(b)) {
        LLVM_DEBUG(llvm::dbgs() << "Invalid loop depth\n");
        return failure();
      }
      FlatAffineValueConstraints dependenceConstraints;
      // Check dependence between 'srcAccess' and 'dstAccess'.
      DependenceResult result = checkMemrefAccessDependence(
          srcAccess, dstAccess, /*loopDepth=*/numCommonLoops + 1,
          &dependenceConstraints, /*dependenceComponents=*/nullptr, false);
      if (result.value == DependenceResult::Failure) {
        LLVM_DEBUG(llvm::dbgs() << "Dependence check failed\n");
        return failure();
      }
      if (result.value == DependenceResult::NoDependence)
        continue;
      dependentOps.push_back(b);
      hasDep = true;
    }
    if (!hasDep)
      continue;
    if (!isa<AffineWriteOpInterface>(a)) {
      LLVM_DEBUG(llvm::dbgs() << "Dependent src op is not a write\n");
      return failure();
    }
    if (srcOp != nullptr) {
      LLVM_DEBUG(llvm::dbgs() << "Multiple dependent src ops\n");
      return failure();
    }
    if (failed(srcAccess.getAccessRelation(srcRel))) {
      return failure();
    }
    srcMemref = srcAccess.memref;
    srcOp = a;
  }
  if (srcOp == nullptr) {
    LLVM_DEBUG(llvm::dbgs() << "No dependent src op\n");
    return failure();
  }
  SmallVector<AffineForOp, 4> tmpLoopIVs;
  for (Operation *b : dependentOps) {
    MemRefAccess dstAccess(b);

    PresburgerSpace space = PresburgerSpace::getRelationSpace();
    IntegerRelation dstRel(space);
    if (failed(dstAccess.getAccessRelation(dstRel))) {
      return failure();
    }
    dstRel.projectOut(loopDepth, dstRel.getNumDomainVars() - loopDepth);
    dstRel.convertVarKind(VarKind::Domain, 0, loopDepth, VarKind::Symbol);

    if (mergedDstRel.getNumDimAndSymbolVars() == 0) {
      mergedDstRel = dstRel;
    } else {
      if (mergedDstRel.isObviouslyEqual(dstRel)) {
        LLVM_DEBUG(llvm::dbgs() << "Obviously equal, skipping\n");
        continue;
      }
      LLVM_DEBUG({
        llvm::dbgs() << "Merging rel:\n";
        dstRel.dump();
        llvm::dbgs() << "into rel:\n";
        mergedDstRel.dump();
      });
      if (failed(mergedDstRel.unionBoundingBox(dstRel))) {
        LLVM_DEBUG(llvm::dbgs()
                   << "Failed to merge bounding boxes for two accesses\n");
        return failure();
      }
    }
    // Get loop nest surrounding dst operation.
    tmpLoopIVs.clear();
    getAffineForIVs(*dstAccess.opInst, &tmpLoopIVs);
    if (dstLoopIVs.empty()) {
      dstLoopIVs = tmpLoopIVs;
    }
    // Checked previously
    assert(dstLoopIVs.size() >= loopDepth && tmpLoopIVs.size() >= loopDepth);
    // Just to make sure
    for (int i = 0; i < loopDepth; i++) {
      if (tmpLoopIVs[i] != dstLoopIVs[i]) {
        LLVM_DEBUG(llvm::dbgs()
                   << "Destination accesses not in same loop nest\n");
        return failure();
      }
    }
  }
  LLVM_DEBUG({
    llvm::dbgs() << "Merged dst rel:\n";
    mergedDstRel.dump();
  });
  if (mergedDstRel.getNumDimAndSymbolVars() == 0) {
    assert(!"No dst operation was fused");
    return failure();
  }
  // Restrict to tiled, if non-unit step
  // Get loop nest surrounding src operation.
  getAffineForIVs(*srcOp, &srcLoopIVs);
  unsigned srcDepth, dstTiledDepth;
  bool srcTiled, dstTiled;
  if (failed(restrictToTiledLoops(srcLoopIVs, &srcRel, &srcDepth, &srcTiled))) {
    return failure();
  }
  if (srcTiled) {
    LLVM_DEBUG({
      llvm::dbgs() << "Src loop is tiled; restricted to depth " << srcDepth
                   << "\n";
      llvm::dbgs() << "Projected src loop:\n";
      srcRel.dump();
    });
  }
  mergedDstRel.convertVarKind(VarKind::Symbol, 0, loopDepth, VarKind::Domain);
  if (failed(restrictToTiledLoops(dstLoopIVs, &mergedDstRel, &dstTiledDepth,
                                  &dstTiled))) {
    return failure();
  }
  if (dstTiled) {
    LLVM_DEBUG({
      llvm::dbgs() << "Dst loop is tiled; restricted to depth " << dstTiledDepth
                   << "/" << loopDepth << "\n";
      llvm::dbgs() << "Projected dst loop:\n";
      mergedDstRel.dump();
    });
  }
  if (dstTiledDepth < loopDepth) {
    loopDepth = dstTiledDepth;
  }

  unsigned numSrcDims = srcRel.getNumDomainVars();
  mergedDstRel.inverse();
  IntegerRelation dstRel = mergedDstRel;
  dstRel.mergeAndCompose(srcRel);
  // Add ordering constraints on common loops.
  unsigned numCols = dstRel.getNumCols();
  SmallVector<int64_t, 4> eq(numCols);
  unsigned numCommonLoopConstraints = numCommonLoops;
  for (unsigned i = 0; i < numCommonLoopConstraints; i++) {
    std::fill(eq.begin(), eq.end(), 0);
    eq[i] = -1;
    eq[i + numSrcDims] = 1;
    // All iterations equal, skipping inequality case
    dstRel.addEquality(eq);
  }

  LLVM_DEBUG({
    llvm::dbgs() << "Adjusted dependence polyhedron:\n";
    dstRel.dump();
  });

  dstRel.projectOut(dstRel.getVarKindOffset(VarKind::Local),
                    dstRel.getNumLocalVars());

  srcUsed = dstRel.getDomainSet();
  LLVM_DEBUG({
    llvm::dbgs() << "Src used (before projecting):\n";
    srcUsed.dump();
  });
  srcUsed.projectOut(srcUsed.getVarKindOffset(VarKind::Local),
                     srcUsed.getNumLocalVars());

  LLVM_DEBUG({
    llvm::dbgs() << "After projecting:\n";
    dstRel.dump();
  });

  // Find the minimum dst index for each src.
  dstRel.inverse();

  dstElim.reserve(loopDepth);
  for (int i = loopDepth - 1; i >= 0; i--) {
    LLVM_DEBUG({
      llvm::dbgs() << "dstElim " << i << " from end:\n";
      dstRel.dump();
    });
    dstElim.push_back(dstRel);
    LLVM_DEBUG(llvm::dbgs() << "Projecting out dst var " << i << "\n");
    dstRel.projectOut(i);
    dstRel.simplify();
  }
  if (dstRel.isEmpty()) {
    LLVM_DEBUG(llvm::dbgs() << "No pipelined section\n");
    return failure();
  }
  std::reverse(dstElim.begin(), dstElim.end());

  return success();
}

/**
 * Appends row to matroid matrices given constraint row.
 */
bool updateMatroid(IntMatrix &matroidMat, IntMatrix &matroidL,
                   SmallVectorImpl<DynamicAPInt> &matroidC,
                   SmallVectorImpl<unsigned> &groups,
                   ArrayRef<DynamicAPInt> row, unsigned i,
                   unsigned numSrcLoopIVs) {
  assert(matroidMat.getNumRows() == matroidL.getNumRows());
  assert(row.size() == i + 1 + numSrcLoopIVs + 1);
  LLVM_DEBUG({
    llvm::dbgs() << "choice for dim " << i << ":";
    for (auto &coef : row) {
      llvm::dbgs() << " " << coef;
    }
    llvm::dbgs() << "\n";
  });
  if (row[i] != 1 && row[i] != -1) {
    return false;
  }
  matroidMat.appendExtraRow(row.slice(i + 1, numSrcLoopIVs));
  unsigned j = matroidL.appendExtraRow();
  matroidL.fillRow(j, 0);
  auto it = row.begin();
  std::copy_n(it, i + 1, matroidL.getRow(j).begin());
  matroidC.push_back(-row.back());
  groups.push_back(i);
  // Constraint is of the form L [v] + M [r] + c ?= 0.
  // But we want L [v] = M [r] + c, so negate one of the two
  // so that L has 1s on the diagonal.
  if (row[i] > 0) {
    matroidMat.negateRow(j);
  } else {
    matroidL.negateRow(j);
    matroidC.back() *= -1;
  }
  return true;
}

/**
 * Computes schedule for pipeline fusion.
 * @param dstElim The result of computeSrcToDstRelation(), used to determine
 * lower bound to be used for schedule.
 * @param[inout] loopDepth The loop depth at which to fuse. May be reduced.
 * @param srcLoopIVs The source loop IVs.
 * @param dstLoopIVs The source loop IVs.
 * @param[inout] srcOperands The destination loop IVs, used as affine operands.
 * Schedule variables are appended, for backward substitution.
 * @param[out] Udst Affine expressions computed as U applied to the destination
 * IVs.
 * @param[out] freeVars Positions of free variables in H.
 * @param[out] pivots The positions (row, col) of the pivots in H.
 * @param[out] H The schedule in Hermite normal form.
 * @param[out] srcToDstInt The output schedule, as a matrix.
 */
LogicalResult
computeSchedule(SmallVectorImpl<IntegerRelation> &dstElim, unsigned &loopDepth,
                MutableArrayRef<AffineForOp> srcLoopIVs,
                MutableArrayRef<AffineForOp> dstLoopIVs,
                SmallVectorImpl<Value> &srcOperands,
                SmallVectorImpl<AffineExpr> &Udst,
                SmallVectorImpl<unsigned> &freeVars,
                SmallVectorImpl<std::pair<unsigned, unsigned>> &pivots,
                IntMatrix &H, IntMatrix &srcToDstInt) {
  MLIRContext *ctx = srcLoopIVs.front()->getContext();
  unsigned numSrcLoopIVs = srcLoopIVs.size();
  LLVM_DEBUG(llvm::dbgs() << "=== SCHEDULE ===\n");
  LLVM_DEBUG(llvm::dbgs() << "srcToDst: " << loopDepth << " x "
                          << numSrcLoopIVs + 1 << "\n");
  // Rows hold possible LBs.
  IntMatrix matroidMat(0, numSrcLoopIVs, loopDepth, numSrcLoopIVs);
  // Dim index for each row.
  SmallVector<unsigned> groups;
  // Constants c.
  SmallVector<DynamicAPInt> matroidC;
  // Lower-triangular matrix for solve.
  // We have L_[i] = M_[i] [r_i] + c_[i],
  // where [i] is the selected rows, and [v] and [r] are
  // src and dst variables.
  IntMatrix matroidL(0, loopDepth, loopDepth, loopDepth);
  for (int i = 0; i < loopDepth; i++) {
    // Pick a constraining equation for var i
    llvm::SmallVector<unsigned> lbIndices, ubIndices, eqIndices;
    IntegerRelation &cst = dstElim[i];
    cst.getLowerAndUpperBoundIndices(i, &lbIndices, &ubIndices, &eqIndices);
    LLVM_DEBUG(llvm::dbgs() << "adding choices for row " << i << "\n");
    if (!eqIndices.empty()) {
      for (unsigned j : eqIndices) {
        ArrayRef<DynamicAPInt> row = cst.getEquality(j);
        updateMatroid(matroidMat, matroidL, matroidC, groups, row, i,
                      numSrcLoopIVs);
      }
    } else {
      for (unsigned j : lbIndices) {
        ArrayRef<DynamicAPInt> row = cst.getInequality(j);
        updateMatroid(matroidMat, matroidL, matroidC, groups, row, i,
                      numSrcLoopIVs);
      }
    }
  }
  LLVM_DEBUG({
    llvm::dbgs() << "independence matrix:\n";
    matroidMat.dump();
    llvm::dbgs() << "L:\n";
    matroidL.dump();
    llvm::dbgs() << "c: [";
    for (auto &c : matroidC) {
      llvm::dbgs() << " " << c;
    }
    llvm::dbgs() << " ]\n";
    llvm::dbgs() << "groups: [";
    for (auto &c : groups) {
      llvm::dbgs() << " " << c;
    }
    llvm::dbgs() << " ]\n";
  });
  // Choose rows for each group under independence constraint.
  SmallVector<unsigned> choice =
      matroidIntersectionLP(IntMatrix(matroidMat), groups,
                            /*prefixOnly=*/true);
  loopDepth = choice.size();
  srcToDstInt.resize(0, numSrcLoopIVs + 1);
  srcToDstInt.reserveRows(loopDepth);
  for (unsigned i = 0; i < loopDepth; i++) {
    unsigned r = choice[i];
    ArrayRef<DynamicAPInt> Li = matroidL.getRow(r);
    unsigned j = srcToDstInt.appendExtraRow();
    MutableArrayRef<DynamicAPInt> out = srcToDstInt.getRow(j);
    ArrayRef<DynamicAPInt> row = matroidMat.getRow(r);
    std::copy(row.begin(), row.end(), out.begin());
    out.back() = matroidC[r];
    // Forward-eliminate L
    assert(Li[i] == 1 && "L should have 1s on the diagonal");
    for (unsigned j = 0; j < i; j++) {
      srcToDstInt.addToRow(j, i, -Li[j]);
    }
  }
  if (loopDepth == 0) {
    LLVM_DEBUG(llvm::dbgs() << "Cannot pipeline at any depth\n");
    return failure();
  }
  LLVM_DEBUG({
    llvm::dbgs() << "matrix:\n";
    srcToDstInt.dump();
  });

  if (srcLoopIVs[0].getStep() != 1) {
    LLVM_DEBUG(llvm::dbgs()
               << "Src is tiled; adjusting schedule to step size\n");
    // We must have that coef_j * stepsize_j = stepsize_i for all dst i, src j
    // Then align constant_i to stepsize_i (rounding UP)
    // It's safe to round up, as the previous is out of bounds, and we want to
    // maximize locality.
    for (unsigned j = 0; j < numSrcLoopIVs; j++) {
      if (srcLoopIVs[j].getStep().ugt(INT64_MAX)) {
        return failure();
      }
    }
    for (unsigned i = 0; i < loopDepth; i++) {
      if (dstLoopIVs[i].getStep().ugt(INT64_MAX)) {
        return failure();
      }
      int64_t dst_step = dstLoopIVs[i].getStepAsInt();
      // To ensure covered, sufficient to ensure at least one loop has the same
      // step size
      bool hasMin = false;
      for (unsigned j = 0; j < numSrcLoopIVs; j++) {
        if (srcToDstInt.at(i, j) < 0 || srcToDstInt.at(i, j) > INT64_MAX) {
          return failure();
        }
        int64_t coef = int64_t(srcToDstInt.at(i, j)),
                src_step = srcLoopIVs[j].getStepAsInt(), result;
        if (llvm::MulOverflow(coef, src_step, result)) {
          return failure();
        }
        if (result == dst_step) {
          hasMin = true;
        }
        if (result % dst_step != 0) {
          LLVM_DEBUG(llvm::dbgs() << "Tiled loop steps do not match: must have "
                                  << coef << " * " << src_step
                                  << " divisible by " << dst_step << "\n");
          return failure();
        }
      }
      if (!hasMin) {
        LLVM_DEBUG(llvm::dbgs() << "Tiled loop step " << dst_step << " [" << i
                                << "] is not covered by src loop steps\n");
        return failure();
      }
      auto div =
          ceilDiv(srcToDstInt.at(i, numSrcLoopIVs), DynamicAPInt(dst_step));
      srcToDstInt.at(i, numSrcLoopIVs) = div * dstLoopIVs[i].getStepAsInt();
    }
  }

  SmallVector<unsigned, 4> missing;
  std::pair<IntMatrix, IntMatrix> hermiteT =
      IntMatrix(srcToDstInt.transpose()).computeHermiteNormalForm();
  H = hermiteT.first.transpose();
  IntMatrix U(hermiteT.second.transpose());

  LLVM_DEBUG({
    llvm::dbgs() << "H:\n";
    H.dump();
    llvm::dbgs() << "U:\n";
    U.dump();
  });

  // Now we construct affine exprs to convert from dst loop -> src loop.
  // If any pivot entry is not 1, then we can't solve.

  // note: M: loopDepth (R) x numSrcLoopIVs + 1 (C)
  //       U: loopDepth x loopDepth
  //       H: same as M
  // Solving: M src = dst
  //          U M src = U dst
  //          H src = U dst

  unsigned col = 0;
  for (unsigned row = 0; row <= loopDepth; row++) {
    // Note: last col is fixed at 1; don't include in free vars.
    while (col < numSrcLoopIVs && (row >= loopDepth || H.at(row, col) == 0)) {
      freeVars.push_back(col);
      col++;
    }
    if (col >= numSrcLoopIVs) {
      break;
    }
    if (row < loopDepth && H.at(row, col) != 1) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Pivot value is not 1; cannot solve in integers.");
      return failure();
    }
    pivots.emplace_back(row, col);
    col++;
  }
  // Reverse for backward substitution
  std::reverse(pivots.begin(), pivots.end());
  LLVM_DEBUG({
    llvm::dbgs() << "free vars:";
    for (auto x : freeVars)
      llvm::dbgs() << " " << x;
    llvm::dbgs() << "\n";
  });

  SmallVector<int64_t, 4> coef;
  Udst.reserve(loopDepth);
  coef.resize(loopDepth + 1);
  for (unsigned i = 0; i < loopDepth; i++) {
    for (unsigned j = 0; j < loopDepth; j++) {
      coef[j] = (int64_t)U.at(i, j);
    }
    Udst.push_back(getAffineExprFromFlatForm(coef, loopDepth, 0, {}, ctx));
    srcOperands.push_back(dstLoopIVs[i].getBody()->getArgument(0));
  }
  return success();
}

/**
 * Performs pipeline fusion.
 * @param srcLoopIVs The source loop IVs.
 * @param dstLoopIVs The source loop IVs.
 * @param srcOperands The destination loop IVs (and substited src vars), used as
 * affine operands.
 * @param Udst Affine expressions computed as U applied to the destination IVs.
 * @param freeVars Positions of free variables in H.
 * @param pivots The positions (row, col) of the pivots in H.
 * @param H The schedule in Hermite normal form.
 * @param srcToDstInt The output schedule, as a matrix.
 * @param loopDepth The loop depth at which to fuse. May be reduced.
 * @param canRemoveSrcLoop If true, dst is the only relevant consumer of src,
 * and the unused parts can be removed.
 * @param srcUsed The region of src which is used by dst.
 */
LogicalResult pipelineFuse(MutableArrayRef<AffineForOp> srcLoopIVs,
                           MutableArrayRef<AffineForOp> dstLoopIVs,
                           SmallVectorImpl<Value> &srcOperands,
                           ArrayRef<AffineExpr> Udst,
                           ArrayRef<unsigned> freeVars,
                           ArrayRef<std::pair<unsigned, unsigned>> pivots,
                           IntMatrix H, IntMatrix srcToDstInt, int loopDepth,
                           bool canRemoveSrcLoop, IntegerPolyhedron &srcUsed) {
  unsigned numSrcLoopIVs = srcLoopIVs.size();
  assert(numSrcLoopIVs > 0);
  assert(loopDepth <= dstLoopIVs.size());
  assert(loopDepth >= 0);

  MLIRContext *ctx = srcLoopIVs.front()->getContext();
  Block *dstBlock = dstLoopIVs[loopDepth - 1].getBody();
  OpBuilder builder(dstBlock, dstBlock->begin());

  // Array of currently known vars
  SmallVector<AffineExpr, 4> srcVars;
  srcVars.resize(numSrcLoopIVs + 1);
  srcVars[numSrcLoopIVs] = getAffineConstantExpr(1, ctx);

  // Generate loops for free vars.
  for (unsigned varIdx : freeVars) {
    AffineForOp srcLoop = srcLoopIVs[varIdx];
    auto loop = builder.create<AffineForOp>(
        srcLoop.getLoc(), srcLoop.getLowerBoundOperands(),
        srcLoop.getLowerBoundMap(), srcLoop.getUpperBoundOperands(),
        srcLoop.getUpperBoundMap(), srcLoop.getStepAsInt());
    builder.setInsertionPoint(loop.getBody(), loop.getBody()->begin());
    srcVars[varIdx] = getAffineDimExpr((unsigned)srcOperands.size(), ctx);
    srcOperands.push_back(loop.getBody()->getArgument(0));
  }

  // Set pivot vars.
  for (auto pivot : pivots) {
    unsigned i = pivot.first;
    AffineExpr result = Udst[i];
    for (unsigned j = pivot.second + 1; j <= numSrcLoopIVs; j++) {
      result =
          result - getAffineConstantExpr((int64_t)H.at(i, j), ctx) * srcVars[j];
    }
    srcVars[pivot.second] = result;
  }

  // Generate loops for unused src domain.
  FlatAffineValueConstraints srcDomain, dstDomain;
  SmallVector<Operation *, 4> srcLoopOps, dstLoopOps;
  srcLoopOps.resize(srcLoopIVs.size());
  dstLoopOps.resize(loopDepth);
  for (unsigned i = 0; i < srcLoopIVs.size(); i++) {
    srcLoopOps[i] = srcLoopIVs[i];
  }
  for (unsigned i = 0; i < loopDepth; i++) {
    dstLoopOps[i] = dstLoopIVs[i];
  }
  assert(!failed(getIndexSet(srcLoopOps, &srcDomain)));
  assert(!failed(getIndexSet(dstLoopOps, &dstDomain)));
  IntegerRelation dstToSrcRel(
      PresburgerSpace::getRelationSpace(loopDepth, numSrcLoopIVs));
  SmallVector<int64_t, 4> relRow;
  relRow.resize(loopDepth + numSrcLoopIVs + 1);
  for (unsigned i = 0; i < loopDepth; i++) {
    // dst[i] = srcToDst[i] src
    std::fill(relRow.begin(), relRow.end(), 0);
    relRow[i] = -1;
    for (unsigned j = 0; j <= numSrcLoopIVs; j++) {
      relRow[loopDepth + j] = (int64_t)srcToDstInt.at(i, j);
    }
    dstToSrcRel.addEquality(relRow);
  }
  IntegerPolyhedron spanned = dstDomain;
  spanned.compose(dstToSrcRel);

  if (spanned.isEmpty()) {
    LLVM_DEBUG(llvm::dbgs() << "Pipelined section is empty\n");
    return failure();
  }

  if (canRemoveSrcLoop) {
    // The best we can do is replace srcDomain with the part that's used.
    LLVM_DEBUG({
      llvm::dbgs() << "intersecting src domain:\n";
      srcDomain.dump();
      llvm::dbgs() << "with used:\n";
      srcUsed.dump();
    });
    srcDomain.intersect(srcUsed);
    LLVM_DEBUG({
      llvm::dbgs() << "intersected with used:\n";
      srcUsed.dump();
    });
  }
  // Replace src loop with only the non-pipelined section.
  PresburgerSet toRemove(spanned);
  PresburgerSet unused = srcDomain.subtract(toRemove);
  LLVM_DEBUG({
    llvm::dbgs() << "unused space:\n";
    unused.dump();
  });
  OpBuilder unusedBuilder(dstLoopIVs.front());
  SmallVector<IntegerSet, 2> unusedLoops;
  for (const IntegerRelation &rel : unused.getAllDisjuncts()) {
    if (rel.isIntegerEmpty())
      continue;

    SmallVector<AffineExpr, 0> localExprs;
    unsigned numIneqs = rel.getNumInequalities();
    unsigned numLocals = rel.getNumLocalVars();
    llvm::SmallBitVector divCst(rel.getNumConstraints());
    std::vector<MaybeLocalRepr> repr(numLocals);
    DivisionRepr divs = rel.getLocalReprs(&repr);
    LLVM_DEBUG({
      llvm::dbgs() << "division repr:\n";
      divs.dump();
    });
    unsigned numDivCsts = 0;
    for (auto &var : repr) {
      if (!var) {
        LLVM_DEBUG(llvm::dbgs()
                   << "Failed to find division repr for local var\n");
        return failure();
      }
      if (var.kind == ReprKind::Inequality) {
        divCst[var.repr.inequalityPair.lowerBoundIdx] = true;
        divCst[var.repr.inequalityPair.upperBoundIdx] = true;
        numDivCsts += 2;
      } else {
        divCst[numIneqs + var.repr.equalityIdx] = true;
        numDivCsts += 1;
      }
    }
    // Create AffineExpr for each local var.
    localExprs.resize(numLocals);
    // Perform "topological sort"/dataflow on local vars.
    // Stores the next unknown column for each local var.
    unsigned totalVars = rel.getNumVars();
    unsigned offset = rel.getNumDimAndSymbolVars();
    SmallVector<unsigned> nextCol(numLocals);
    SmallVector<int64_t> intRow(totalVars + 1);
    while (true) {
      LLVM_DEBUG(llvm::dbgs() << "beginning iteration of topsort\n");
      bool allKnown = true;
      bool changed = false;
      for (unsigned i = 0; i < numLocals; i++) {
        if (localExprs[i])
          continue;
        unsigned &j = nextCol[i];
        LLVM_DEBUG(llvm::dbgs() << "row " << i << ": " << j << " -> ");
        auto row = divs.getDividend(i);
        while (j < numLocals) {
          if (row[offset + j] == 0 || localExprs[j]) {
            j++;
          } else {
            break;
          }
        }
        LLVM_DEBUG(llvm::dbgs() << j << "\n");
        if (j < numLocals) {
          allKnown = false;
          continue;
        }
        LLVM_DEBUG(llvm::dbgs() << "--> now known!\n");
        // Var i is now known; compute its local map
        changed = true;
        std::transform(row.begin(), row.end(), intRow.begin(),
                       [](const DynamicAPInt &x) { return int64_t(x); });
        localExprs[i] =
            getAffineExprFromFlatForm(intRow, offset, 0, localExprs, ctx)
                .floorDiv(uint64_t(int64_t(divs.getDenom(i))));
        LLVM_DEBUG(localExprs[i].dump());
      }
      if (allKnown) {
        break;
      }
      assert(changed &&
             "Could not iteratively find affine maps for each local var\n");
    }

    SmallVector<bool, 16> eqFlags;
    SmallVector<AffineExpr, 8> exprs;
    eqFlags.reserve(rel.getNumConstraints() - numDivCsts);
    exprs.reserve(rel.getNumConstraints() - numDivCsts);

    assert(rel.getNumSymbolVars() == 0);
    for (unsigned i = 0; i < rel.getNumInequalities(); i++) {
      if (divCst[i])
        continue;
      exprs.push_back(getAffineExprFromFlatForm(
          rel.getInequality64(i), rel.getNumDimVars(), 0, localExprs, ctx));
      eqFlags.push_back(false);
    }
    for (unsigned i = 0; i < rel.getNumEqualities(); i++) {
      if (divCst[i + numIneqs])
        continue;
      exprs.push_back(getAffineExprFromFlatForm(
          rel.getEquality64(i), rel.getNumDimVars(), 0, localExprs, ctx));
      eqFlags.push_back(true);
    }

    unusedLoops.push_back(
        IntegerSet::get(rel.getNumDimVars(), 0, exprs, eqFlags));
  }
  // Note: subtraction implementation only performs union
  // on disjoint parts, so this *should* be disjoint.
  SmallVector<Value, 4> ifOperands;
  ifOperands.resize(numSrcLoopIVs);
  for (IntegerSet &condition : unusedLoops) {
    AffineForOp cloned =
        cast<AffineForOp>(unusedBuilder.clone(*srcLoopIVs.front()));
    for (unsigned i = 0; i < numSrcLoopIVs - 1; i++) {
      ifOperands[i] = cloned.getBody()->getArgument(0);
      cloned = cast<AffineForOp>(cloned.getBody()->front());
    }
    ifOperands[numSrcLoopIVs - 1] = cloned.getBody()->getArgument(0);
    assert(condition.getNumDims() == numSrcLoopIVs);
    OpBuilder ifBuilder(cloned.getBody(), cloned.getBody()->begin());
    AffineIfOp branch = ifBuilder.create<AffineIfOp>(
        srcLoopIVs.front().getLoc(), condition, ifOperands, false);
    // Move body to branch
    auto oldYield = --cloned.getBody()->end();
    assert(oldYield != cloned.getBody()->end());
    assert(isa<AffineYieldOp>(*oldYield) && "Last op should be yield");
    branch.getThenBlock()->getOperations().splice(
        branch.getThenBlock()->begin(), cloned.getBody()->getOperations(),
        ++cloned.getBody()->begin(), oldYield);
  }

  IntegerRelation srcToDstRel = dstToSrcRel;
  srcToDstRel.inverse();
  FlatAffineValueConstraints dstPipeline = srcDomain;
  dstPipeline.compose(srcToDstRel);
  dstPipeline.projectOut(dstPipeline.getVarKindOffset(VarKind::Local),
                         dstPipeline.getNumLocalVars());

  IntegerSet pipelineSet = dstPipeline.getAsIntegerSet(ctx);
  assert(bool(pipelineSet) && "Pipelined section should not be empty");

  // Clone loop body.
  IRMapping mapper;
  SmallVector<Value, 4> branchOperands;
  branchOperands.reserve(numSrcLoopIVs);
  for (int i = 0; i < numSrcLoopIVs; i++) {
    AffineMap map = AffineMap::get(srcOperands.size(), 0, srcVars[i]);
    auto applyOp =
        builder.create<AffineApplyOp>(srcLoopIVs[i].getLoc(), map, srcOperands);
    mapper.map(srcLoopIVs[i].getBody()->getArgument(0), applyOp);
    branchOperands.push_back(applyOp);
  }
  auto pipelineBranch = builder.create<AffineIfOp>(
      srcLoopIVs.front().getLoc(), pipelineSet,
      ArrayRef(srcOperands).take_front(loopDepth), false);
  builder.setInsertionPointToStart(pipelineBranch.getThenBlock());
  for (Operation &op : *srcLoopIVs.back().getBody()) {
    if (isa<AffineYieldOp>(op)) {
      // Yield is auto-generated, so don't clone it again.
      continue;
    }
    builder.clone(op, mapper);
  }

  // Remove source loop.
  srcLoopIVs.front()->erase();

  return success();
}

bool isEscapingMemRef(Value memref, Block *block) {
  Operation *defOp = memref.getDefiningOp();
  // Check if 'memref' is a block argument.
  if (!defOp)
    return true;

  // Check if this is defined to be an alias of another memref.
  if (auto viewOp = dyn_cast<mlir::ViewLikeOpInterface>(defOp))
    if (isEscapingMemRef(viewOp.getViewSource(), block))
      return true;

  // Any op besides allocating ops wouldn't guarantee alias freedom
  if (!hasSingleEffect<mlir::MemoryEffects::Allocate>(defOp, memref))
    return true;

  // Check if 'memref' is used by a non-deferencing op (including unknown ones)
  // (e.g., call ops, alias creating ops, etc.).
  return llvm::any_of(memref.getUsers(), [&](Operation *user) {
    // Ignore users outside of `block`.
    Operation *ancestorOp = block->getParent()->findAncestorOpInRegion(*user);
    if (!ancestorOp)
      return true;
    if (ancestorOp->getBlock() != block)
      return false;
    return !isa<AffineMapAccessInterface>(*user);
  });
}

void checkFusionPreconditions(const MemRefDependenceGraph::Node &srcNode,
                              const MemRefDependenceGraph::Node &dstNode,
                              MemRefDependenceGraph &g, bool &canFuse,
                              bool &canRemoveSrcNode) {
  canRemoveSrcNode = true;
  Operation *srcOp = g.getNode(srcNode.id)->op;
  Operation *dstOp = g.getNode(dstNode.id)->op;
  Block *block = dstOp->getBlock();
  // Gather memrefs involved in dependence
  llvm::SmallDenseSet<Value, 2> storeMemRefs, dependentMemRefs;
  for (Operation *store : srcNode.stores) {
    storeMemRefs.insert(cast<AffineWriteOpInterface>(store).getMemRef());
  }
  for (Operation *load : dstNode.loads) {
    auto memref = cast<AffineReadOpInterface>(load).getMemRef();
    if (storeMemRefs.count(memref) > 0) {
      dependentMemRefs.insert(memref);
    }
  }
  // Check if there are any stores between src and dst
  for (auto &inEdge : g.inEdges.lookup(dstNode.id)) {
    if (inEdge.id == srcNode.id)
      continue;
    Operation *depOp = g.getNode(inEdge.id)->op;
    if (depOp->getBlock() != block) {
      // Either dep is before both src and dst or after both src and dst.
      // In both cases, it doesn't matter.
      continue;
    }
    if (srcOp->isBeforeInBlock(depOp)) {
      bool hasDep = false;
      for (Operation *depStore : g.getNode(inEdge.id)->stores) {
        auto memref = cast<AffineWriteOpInterface>(depStore).getMemRef();
        if (dependentMemRefs.count(memref) == 0) {
          hasDep = true;
          break;
        }
      }
      if (!hasDep)
        continue;
      LLVM_DEBUG(llvm::dbgs() << "Can't fuse: there is a dependent op ");
      depOp->dump();
      LLVM_DEBUG(llvm::dbgs() << " between src op ");
      srcOp->dump();
      LLVM_DEBUG(llvm::dbgs() << " and dst op ");
      dstOp->dump();
      LLVM_DEBUG(llvm::dbgs() << "\n");
      canFuse = false;
      return;
    }
  }
  canFuse = true;
  for (auto &outEdge : g.outEdges.lookup(srcNode.id)) {
    if (outEdge.id == dstNode.id)
      continue;
    canRemoveSrcNode = false;
    return;
  }
  // Check if src node writes to any escaping memrefs.
  for (Operation *op : srcNode.stores) {
    Value memref = cast<AffineWriteOpInterface>(op).getMemRef();
    if (isEscapingMemRef(memref, block)) {
      canRemoveSrcNode = false;
      return;
    }
  }
}

void AffinePipelineFusionPass::runOnBlock(Block *block) {
  using Node = MemRefDependenceGraph::Node;
  MemRefDependenceGraph g(*block);
  if (!g.init()) {
    LLVM_DEBUG(llvm::dbgs() << "MDG init failed\n");
    return;
  }
  for (auto &idAndNode : g.nodes) {
    LLVM_DEBUG(llvm::dbgs() << "evaluating node " << idAndNode.first << "\n");
    const Node &node = idAndNode.second;
    if (!isa<AffineForOp>(node.op)) {
      continue;
    }
    if (node.op->getNumResults() > 0) {
      continue;
    }
    if (g.inEdges.count(node.id) == 0) {
      continue;
    }
    // Gather memefs from loads in dst.
    DenseSet<Value> consumedMemRefs;
    DenseSet<Value> producedMemRefs;
    for (Operation *op : node.loads) {
      consumedMemRefs.insert(cast<AffineReadOpInterface>(op).getMemRef());
    }
    // Find suitable src loop.
    // For now, only fuse one loop at a time.
    unsigned srcId;
    bool found = false;
    for (const auto &srcEdge : g.inEdges.lookup(node.id)) {
      if (node.id == srcEdge.id)
        continue;
      const auto *srcNode = g.getNode(srcEdge.id);
      if (!isa<AffineForOp>(srcNode->op)) {
        continue;
      }
      bool valid = true;
      bool hasDependence = false;
      producedMemRefs.clear();
      for (const Operation *op : srcNode->stores) {
        auto storeOp = cast<AffineWriteOpInterface>(op);
        if (consumedMemRefs.count(storeOp.getMemRef()) > 0) {
          hasDependence = true;
          // We can only fuse src loops that write once.
          if (!producedMemRefs.insert(storeOp.getMemRef()).second) {
            valid = false;
            break;
          }
        }
      }
      if (!valid || !hasDependence) {
        continue;
      }
      srcId = srcEdge.id;
      found = true;
      break;
    }
    if (!found)
      continue;
    auto dstOp = cast<AffineForOp>(node.op);
    auto *srcNode = g.getNode(srcId);
    auto srcOp = cast<AffineForOp>(srcNode->op);
    LLVM_DEBUG(llvm::dbgs() << "Trying to fuse producer loop nest " << srcId
                            << " with consumer loop nest " << node.id << "\n");
    LLVM_DEBUG(llvm::dbgs() << "Producer loop nest:\n"
                            << *node.op << "\n"
                            << "Consumer loop nest:\n"
                            << *srcNode->op << "\n");
    auto &producerConsumerMemRefs = producedMemRefs;
    Operation *fusedLoopInsPoint =
        g.getFusedLoopNestInsertionPoint(srcId, node.id);
    if (fusedLoopInsPoint == nullptr)
      continue;
    // Note: it is not sufficient to check only producer/consumer memrefs,
    // as we may arbitrarily reorder the iterations in src.
    if (hasCyclicDependence(srcOp)) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Stopping as src loop has cyclic dependence\n");
      continue;
    }
    // Find the perfectly nested loops in dst.
    SmallVector<AffineForOp, 4> dstNestedLoops;
    getPerfectlyNestedLoops(dstNestedLoops, dstOp);
    // Check preconditions
    auto *block = srcOp->getBlock();
    if (block != dstOp->getBlock()) {
      LLVM_DEBUG(llvm::dbgs() << "Cannot fuse loops in different blocks\n");
      continue;
    }
    if (!srcOp->isBeforeInBlock(dstOp)) {
      LLVM_DEBUG(llvm::dbgs() << "Cannot fuse dst loop before src loop\n");
      continue;
    }
    bool canFuse = true;
    bool canRemoveSrcLoop = true;
    checkFusionPreconditions(*srcNode, node, g, canFuse, canRemoveSrcLoop);
    if (!canFuse) {
      LLVM_DEBUG(llvm::dbgs() << "Cannot fuse loops\n");
      continue;
    }
    // Obtain schedule by finding the min dst iteration that accesses the src
    // write location for a symbolic src iteration.
    bool hasIfOp = false;
    SmallVector<Operation *, 4> opsSrc;
    srcOp.walk([&](Operation *op) {
      if (isa<AffineWriteOpInterface>(op)) {
        opsSrc.push_back(op);
        LLVM_DEBUG(llvm::dbgs() << "src op:\n" << *op << "\n");
      } else if (isa<AffineIfOp>(op)) {
        hasIfOp = true;
      }
    });
    SmallVector<Operation *, 4> opsDst;
    dstOp.walk([&](Operation *op) {
      if (isa<AffineReadOpInterface, AffineWriteOpInterface>(op)) {
        opsDst.push_back(op);
        LLVM_DEBUG(llvm::dbgs() << "dst op:\n" << *op << "\n");
      } else if (isa<AffineIfOp>(op)) {
        hasIfOp = true;
      }
    });
    if (hasIfOp) {
      LLVM_DEBUG(llvm::dbgs() << "Stopping as there is an if op\n");
      continue;
    }
    unsigned numCommonLoops = getNumCommonSurroundingLoops(*srcOp, *dstOp);
    SmallVector<IntegerRelation, 2> dstElim;
    SmallVector<AffineForOp, 4> srcLoopIVs;
    SmallVector<AffineForOp, 4> dstLoopIVs;
    unsigned loopDepth = dstNestedLoops.size() + numCommonLoops;
    IntegerPolyhedron srcUsed(PresburgerSpace::getRelationSpace());
    // loopDepth = 2;
    if (failed(computeSrcToDstRelation(opsSrc, opsDst, loopDepth,
                                       numCommonLoops, dstElim, srcLoopIVs,
                                       dstLoopIVs, srcUsed))) {
      LLVM_DEBUG(llvm::dbgs() << "computeSrcToDstRelation failed\n");
      continue;
    }
    LLVM_DEBUG(llvm::dbgs() << "fusion loop depth: " << loopDepth << "\n");
    SmallVector<Value, 4> srcOperands;
    SmallVector<AffineExpr, 4> Udst;
    SmallVector<unsigned, 2> freeVars;
    SmallVector<std::pair<unsigned, unsigned>, 4> pivots;
    IntMatrix H(0, 0), srcToDstInt(0, 0);
    if (failed(computeSchedule(dstElim, loopDepth, srcLoopIVs, dstLoopIVs,
                               srcOperands, Udst, freeVars, pivots, H,
                               srcToDstInt))) {
      LLVM_DEBUG(llvm::dbgs() << "computeSchedule failed\n");
      continue;
    }
    if (failed(pipelineFuse(srcLoopIVs, dstLoopIVs, srcOperands, Udst, freeVars,
                            pivots, H, srcToDstInt, loopDepth, canRemoveSrcLoop,
                            srcUsed))) {
      LLVM_DEBUG(llvm::dbgs() << "pipelineFuse failed\n");
    }
  }
}

void AffinePipelineFusionPass::runOnOperation() {
  getOperation()->walk([&](Operation *op) {
    for (Region &region : op->getRegions()) {
      for (Block &block : region.getBlocks()) {
        auto affineFors = block.getOps<AffineForOp>();
        if (!affineFors.empty() && !llvm::hasSingleElement(affineFors))
          runOnBlock(&block);
      }
    }
  });
}

} // namespace

std::unique_ptr<OperationPass<FuncOp>>
hexagon::createAffinePipelineFusionPass() {
  return std::make_unique<AffinePipelineFusionPass>();
}
