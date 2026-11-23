//===- CopyCanonicalizationCommon.h - shared copy-canon utilities ---------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

#ifndef HEXAGON_LIB_TRANSFORMS_COPYCANONICALIZATIONCOMMON_H
#define HEXAGON_LIB_TRANSFORMS_COPYCANONICALIZATIONCOMMON_H

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"

namespace mlir::hexagon::detail {

/// True if `user` writes through `use` and never reads or frees it.
/// Conservatively false when effects are unknown: the op does not implement
/// `MemoryEffectOpInterface`, or it does but reports no effects on this
/// operand (an empty list is ambiguous between "no effect" and "unspecified",
/// and we cannot prove write-only).
inline bool hasWriteOnlyEffect(OpOperand &use) {
  auto effectOp = dyn_cast<MemoryEffectOpInterface>(use.getOwner());
  if (!effectOp)
    return false;
  SmallVector<MemoryEffects::EffectInstance> effects;
  effectOp.getEffectsOnValue(use.get(), effects);
  if (effects.empty())
    return false;
  bool hasWrite = false;
  for (auto &e : effects) {
    if (isa<MemoryEffects::Read, MemoryEffects::Free>(e.getEffect()))
      return false;
    if (isa<MemoryEffects::Write>(e.getEffect()))
      hasWrite = true;
  }
  return hasWrite;
}

inline bool isWriteOnlyUseChain(Value v, Operation *consumer);

/// True if `use` reaches only write-only leaves: the owner is either a
/// single-result `ViewLikeOpInterface` whose result is itself write-only,
/// or a leaf op with a write-only effect on the operand.  `consumer` is
/// excluded from the check (its read of the alloc is the one the rewrite
/// removes).
inline bool isWriteOnlyUseRoot(OpOperand &use, Operation *consumer) {
  Operation *user = use.getOwner();
  if (auto view = dyn_cast<ViewLikeOpInterface>(user)) {
    if (user->getNumResults() != 1)
      return false;
    return isWriteOnlyUseChain(user->getResult(0), consumer);
  }
  return hasWriteOnlyEffect(use);
}

/// True if every use of `v` other than `consumer` is a write-only root.
inline bool isWriteOnlyUseChain(Value v, Operation *consumer) {
  for (OpOperand &use : v.getUses()) {
    if (use.getOwner() == consumer)
      continue;
    if (!isWriteOnlyUseRoot(use, consumer))
      return false;
  }
  return true;
}

/// `destType` after a `memref.memory_space_cast` aligning its memory space
/// with `allocType`'s.  Returns `destType` unchanged when spaces match,
/// null `MemRefType` when the cast is infeasible.
inline MemRefType alignMemorySpace(MemRefType destType, MemRefType allocType) {
  if (destType.getMemorySpace() == allocType.getMemorySpace())
    return destType;
  auto candidate =
      MemRefType::get(destType.getShape(), destType.getElementType(),
                      destType.getLayout(), allocType.getMemorySpace());
  if (!memref::MemorySpaceCastOp::areCastCompatible(destType, candidate))
    return MemRefType{};
  return candidate;
}

/// True if `dest` is reachable from `allocType` via the cast chain that
/// `buildDestReplacement` would build.  Use to validate feasibility before
/// mutating the IR.
inline bool canBuildDestReplacement(Value dest, MemRefType allocType) {
  auto destType = cast<MemRefType>(dest.getType());
  if (allocType == destType)
    return true;
  auto interType = alignMemorySpace(destType, allocType);
  if (!interType)
    return false;
  if (interType == allocType)
    return true;
  return memref::CastOp::areCastCompatible(interType, allocType);
}

/// Build a value of type `allocType` aliasing `dest`, chaining
/// `memref.memory_space_cast` then `memref.cast` as needed.  Returns a
/// null `Value` when the chain is infeasible — callers treat that as a
/// match failure.
inline Value buildDestReplacement(PatternRewriter &rewriter, Value dest,
                                  MemRefType allocType, Operation *castPoint) {
  auto destType = cast<MemRefType>(dest.getType());
  if (allocType == destType)
    return dest;

  auto interType = alignMemorySpace(destType, allocType);
  if (!interType)
    return Value{};

  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(castPoint);

  Value v = dest;
  if (interType != destType)
    v = memref::MemorySpaceCastOp::create(rewriter, castPoint->getLoc(),
                                          interType, v);

  if (v.getType() == allocType)
    return v;
  if (!memref::CastOp::areCastCompatible(v.getType(), allocType))
    return Value{};
  return memref::CastOp::create(rewriter, castPoint->getLoc(), allocType, v);
}

/// True if `a` runs strictly before `b` in program order.  Walks `a` up
/// via `getParentOp` until it shares a block with an ancestor of `b`.
/// False when `a` is a proper ancestor of `b` or the two are in unrelated
/// regions.
///
/// Mirrors the file-local `happensBefore` helpers in
/// `Bufferization/Transforms/OptimizeAllocationLiveness.cpp`,
/// `Bufferization/Transforms/OneShotAnalysis.cpp`, and
/// `Transform/Interfaces/TransformInterfaces.cpp` — there is no public
/// utility for this query.
inline bool happensBefore(Operation *a, Operation *b) {
  do {
    if (a->isProperAncestor(b))
      return false;
    if (Operation *bAncestor = a->getBlock()->findAncestorOpInBlock(*b))
      return a->isBeforeInBlock(bAncestor);
  } while ((a = a->getParentOp()));
  return false;
}

} // namespace mlir::hexagon::detail

#endif // HEXAGON_LIB_TRANSFORMS_COPYCANONICALIZATIONCOMMON_H
