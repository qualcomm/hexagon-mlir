//===- BroadcastedReductionToButterflyPass.cpp ---------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// Rewrites `vector.reduction` -> (scalar arith/math chain) ->
// `vector.broadcast` into an in-place butterfly reduction so every lane
// already holds the reduced value.
//===----------------------------------------------------------------------===//

#include "hexagon/Transforms/Transforms.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Types.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "broadcasted-reduction-to-butterfly"
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define DBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
using namespace hexagon;

#define GEN_PASS_DEF_BROADCASTEDREDUCTIONTOBUTTERFLY
#include "hexagon/Transforms/Passes.h.inc"

namespace {

constexpr int64_t kNativeVectorWidthInBytes = 128;

// Bail out on overly long chains to avoid pathological cases.
constexpr size_t kMaxChainLength = 10;

static FailureOr<int64_t> hvxLaneCount(Type elemTy) {
  if (!elemTy.isIntOrFloat())
    return failure();
  unsigned bitWidth = elemTy.getIntOrFloatBitWidth();
  if (bitWidth == 0 || bitWidth % 8 != 0)
    return failure();
  int64_t bytes = bitWidth / 8;
  if (kNativeVectorWidthInBytes % bytes != 0)
    return failure();
  return kNativeVectorWidthInBytes / bytes;
}

// Do elementwise arith operations instead of vector.reduction
static Value createElementwiseOp(IRRewriter &rewriter, Location loc,
                                 vector::CombiningKind kind, Value lhs,
                                 Value rhs, arith::FastMathFlags fmf) {
  using K = vector::CombiningKind;
  Type elemTy = getElementTypeOrSelf(lhs.getType());
  bool isFloat = isa<FloatType>(elemTy);
  bool isInt = isa<IntegerType>(elemTy);

  if (isFloat) {
    switch (kind) {
    case K::ADD:
      return arith::AddFOp::create(rewriter, loc, lhs, rhs, fmf);
    case K::MUL:
      return arith::MulFOp::create(rewriter, loc, lhs, rhs, fmf);
    case K::MAXNUMF:
      return arith::MaxNumFOp::create(rewriter, loc, lhs, rhs, fmf);
    case K::MAXIMUMF:
      return arith::MaximumFOp::create(rewriter, loc, lhs, rhs, fmf);
    case K::MINNUMF:
      return arith::MinNumFOp::create(rewriter, loc, lhs, rhs, fmf);
    case K::MINIMUMF:
      return arith::MinimumFOp::create(rewriter, loc, lhs, rhs, fmf);
    default:
      return nullptr;
    }
  }
  if (isInt) {
    switch (kind) {
    case K::ADD:
      return arith::AddIOp::create(rewriter, loc, lhs, rhs);
    case K::MUL:
      return arith::MulIOp::create(rewriter, loc, lhs, rhs);
    default:
      return nullptr;
    }
  }
  return nullptr;
}

// log2(N) shuffle+combine steps with XOR strides N/2, N/4, ..., 1; every lane
// of the result holds the global reduction. N must be a power of two.
static Value buildButterfly(IRRewriter &rewriter, Location loc, Value vec,
                            vector::CombiningKind kind,
                            arith::FastMathFlags fmf) {
  auto vecTy = dyn_cast<VectorType>(vec.getType());
  if (!vecTy || vecTy.getRank() != 1)
    return nullptr;
  int64_t n = vecTy.getNumElements();
  if (n <= 1 || (n & (n - 1)) != 0)
    return nullptr;

  Value cur = vec;
  for (int64_t stride = n / 2; stride >= 1; stride >>= 1) {
    SmallVector<int64_t> mask(n);
    for (int64_t i = 0; i < n; ++i)
      mask[i] = i ^ stride;
    Value shuffled = vector::ShuffleOp::create(rewriter, loc, cur, cur, mask);
    Value combined =
        createElementwiseOp(rewriter, loc, kind, cur, shuffled, fmf);
    if (!combined)
      return nullptr;
    cur = combined;
  }
  return cur;
}

struct ChainResult {
  vector::BroadcastOp broadcast;
  // Forward order: chain[0] consumes the reduction; the last entry feeds
  // `broadcast`. Empty when reduction feeds broadcast directly.
  SmallVector<Operation *> chain;
};

static FailureOr<ChainResult> traceChain(Value start, Type expectedT) {
  ChainResult result;
  Value cur = start;
  while (true) {
    if (!cur.hasOneUse())
      return failure();
    Operation *user = *cur.getUsers().begin();
    if (auto bcast = dyn_cast<vector::BroadcastOp>(user)) {
      result.broadcast = bcast;
      return result;
    }
    Dialect *dialect = user->getDialect();
    if (!dialect)
      return failure();
    StringRef ns = dialect->getNamespace();
    if (ns != "arith" && ns != "math")
      return failure();
    if (user->getNumResults() != 1)
      return failure();
    Value res = user->getResult(0);
    if (res.getType() != expectedT)
      return failure();
    result.chain.push_back(user);
    if (result.chain.size() > kMaxChainLength)
      return failure();
    cur = res;
  }
}

// Operands matching `scalarChain` swap to `vectorChain`; the rest splat to
// vecTy. Same opcode + attributes — arith and math ops are polymorphic over
// scalar/vector element types
static Value vectorizeScalarOp(IRRewriter &rewriter, Operation *scalarOp,
                               Value vectorChain, Value scalarChain,
                               VectorType vecTy) {
  Location loc = scalarOp->getLoc();
  SmallVector<Value> newOperands;
  newOperands.reserve(scalarOp->getNumOperands());
  for (Value op : scalarOp->getOperands()) {
    if (op == scalarChain) {
      newOperands.push_back(vectorChain);
    } else {
      Value bcast = vector::BroadcastOp::create(rewriter, loc, vecTy, op);
      newOperands.push_back(bcast);
    }
  }

  OperationState state(loc, scalarOp->getName());
  state.addOperands(newOperands);
  state.addAttributes(scalarOp->getAttrs());
  state.addTypes(TypeRange{Type(vecTy)});
  Operation *newOp = rewriter.create(state);
  return newOp->getResult(0);
}

static Value adjustVectorLanes(IRRewriter &rewriter, Location loc, Value vec,
                               int64_t targetLanes) {
  int64_t srcLanes = cast<VectorType>(vec.getType()).getNumElements();
  if (targetLanes == srcLanes)
    return vec;
  if (targetLanes < srcLanes)
    return vector::ExtractStridedSliceOp::create(
        rewriter, loc, vec,
        /*offsets=*/ArrayRef<int64_t>{0},
        /*sizes=*/ArrayRef<int64_t>{targetLanes},
        /*strides=*/ArrayRef<int64_t>{1});
  Value cur = vec;
  int64_t curLanes = srcLanes;
  while (curLanes < targetLanes) {
    SmallVector<int64_t> mask(2 * curLanes);
    for (int64_t i = 0; i < 2 * curLanes; ++i)
      mask[i] = i % curLanes;
    cur = vector::ShuffleOp::create(rewriter, loc, cur, cur, mask);
    curLanes *= 2;
  }
  return cur;
}

// Reduction kinds handled by createElementwiseOp; checked as a precondition
// so downstream IR construction can't fail on an unsupported combo.
static bool isSupportedReductionKind(vector::CombiningKind kind, Type elemTy) {
  using K = vector::CombiningKind;
  if (isa<FloatType>(elemTy)) {
    switch (kind) {
    case K::ADD:
    case K::MUL:
    case K::MAXNUMF:
    case K::MAXIMUMF:
    case K::MINNUMF:
    case K::MINIMUMF:
      return true;
    default:
      return false;
    }
  }
  if (isa<IntegerType>(elemTy))
    return kind == K::ADD || kind == K::MUL;
  return false;
}

// Skipped: pattern didn't match, no IR mutated.
// Rewrote: rewrite applied successfully.
// Fatal:   invariant violated mid-rewrite; IR may be partially mutated.
enum class RewriteOutcome { Skipped, Rewrote, Fatal };

struct BroadcastedReductionToButterflyPass
    : public ::impl::BroadcastedReductionToButterflyBase<
          BroadcastedReductionToButterflyPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, math::MathDialect,
                    vector::VectorDialect>();
  }

  RewriteOutcome tryRewrite(IRRewriter &rewriter, vector::ReductionOp redOp) {
    Type scalarT = redOp.getResult().getType();
    if (!isSupportedReductionKind(redOp.getKind(), scalarT))
      return RewriteOutcome::Skipped;

    auto chainOr = traceChain(redOp.getResult(), scalarT);
    if (failed(chainOr)) {
      DBG("no broadcast-terminated scalar chain from " << redOp);
      return RewriteOutcome::Skipped;
    }
    auto &chain = *chainOr;

    Value srcVec = redOp.getVector();
    auto srcVecTy = dyn_cast<VectorType>(srcVec.getType());
    if (!srcVecTy || srcVecTy.getRank() != 1)
      return RewriteOutcome::Skipped;
    int64_t redLanes = srcVecTy.getNumElements();
    if (redLanes <= 1 || (redLanes & (redLanes - 1)) != 0)
      return RewriteOutcome::Skipped;

    // Lane count need not match the source: a wider broadcast is satisfied by
    // shuffle-replication later, a narrower one by extract_strided_slice.
    auto bcastTy = dyn_cast<VectorType>(chain.broadcast.getType());
    if (!bcastTy || bcastTy.getRank() != 1)
      return RewriteOutcome::Skipped;
    if (bcastTy.getElementType() != srcVecTy.getElementType())
      return RewriteOutcome::Skipped;
    int64_t bcastLanes = bcastTy.getNumElements();
    if (bcastLanes < 1 || (bcastLanes & (bcastLanes - 1)) != 0)
      return RewriteOutcome::Skipped;

    auto hvxLanesOr = hvxLaneCount(srcVecTy.getElementType());
    if (failed(hvxLanesOr)) {
      DBG("Issues with the float or int bitwidth used");
      return RewriteOutcome::Skipped;
    }
    int64_t hvxLanes = *hvxLanesOr;
    auto chainVecTy = VectorType::get({hvxLanes}, srcVecTy.getElementType());

    Location loc = redOp.getLoc();
    arith::FastMathFlags fmf = redOp.getFastmath();
    rewriter.setInsertionPoint(redOp);

    // Butterfly stays at srcVecTy even when wider than hvxLanes;
    // to process all the elements
    // Split reduction can produce results wider than hvxLanes.
    // The chain narrows to hvxLanes.
    Value butterflyResult =
        buildButterfly(rewriter, loc, srcVec, redOp.getKind(), fmf);
    // case should not occur due to isSupportedReductionKind
    if (!butterflyResult)
      return RewriteOutcome::Fatal;

    // Every lane of butterflyResult holds the reduction value. Narrow when
    // redLanes > hvxLanes, replicate when redLanes < hvxLanes so the scalar
    // chain always vectorizes at full native width.
    Value chainStart =
        adjustVectorLanes(rewriter, loc, butterflyResult, hvxLanes);

    // Fold the reduction's accumulator (init) lane-wise to match
    // vector.reduction's combine(acc, reduce(srcVec)) semantics.
    if (Value acc = redOp.getAcc()) {
      Value accSplat =
          vector::BroadcastOp::create(rewriter, loc, chainVecTy, acc);
      chainStart = createElementwiseOp(rewriter, loc, redOp.getKind(),
                                       chainStart, accSplat, fmf);
      // case should not occur due to isSupportedReductionKind
      if (!chainStart)
        return RewriteOutcome::Fatal;
    }

    Value vecChain = chainStart;
    Value scalarChain = redOp.getResult();
    for (Operation *scalarOp : chain.chain) {
      vecChain = vectorizeScalarOp(rewriter, scalarOp, vecChain, scalarChain,
                                   chainVecTy);
      scalarChain = scalarOp->getResult(0);
    }

    // Every lane of vecChain holds the final value, so contraction is a
    // contiguous prefix and expansion is shuffle-replication (chained for >2x).
    Value finalVec = adjustVectorLanes(rewriter, loc, vecChain, bcastLanes);

    rewriter.replaceOp(chain.broadcast, finalVec);

    for (Operation *op : llvm::reverse(chain.chain))
      rewriter.eraseOp(op);
    rewriter.eraseOp(redOp);
    return RewriteOutcome::Rewrote;
  }

  void runOnOperation() override {
    SmallVector<vector::ReductionOp> candidates;
    getOperation().walk(
        [&](vector::ReductionOp redOp) { candidates.push_back(redOp); });
    IRRewriter rewriter(getOperation().getContext());
    for (vector::ReductionOp redOp : candidates) {
      if (!redOp->getBlock())
        continue;
      if (tryRewrite(rewriter, redOp) == RewriteOutcome::Fatal) {
        signalPassFailure();
        return;
      }
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
hexagon::createBroadcastedReductionToButterflyPass() {
  return std::make_unique<BroadcastedReductionToButterflyPass>();
}
