//===- CopyCanonicalizationPass.cpp - erase redundant copies   ------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// Canonicalize memref.copy chains: drop intermediate buffers, route loop
// iter_args through copies, and rewrite copies sourced from zero-filled
// allocs into a fill on the destination.
//
//===----------------------------------------------------------------------===//

#include "hexagon/Transforms/CopyCanonicalizationAllocPatterns.h"
#include "hexagon/Transforms/CopyCanonicalizationCommon.h"
#include "hexagon/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
using namespace mlir;

#define DEBUG_TYPE "copy-canonicalization"

#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define DBG(X) LLVM_DEBUG(DBGS() << X << "\n")

#define GEN_PASS_DEF_COPYCANONICALIZATION
#include "hexagon/Transforms/Passes.h.inc"

namespace {

// Set of (callee FuncOp*, argument index) for which the callee is known to
// write into that memref argument (directly or via nested calls).
using WrittenArgsSet = llvm::DenseSet<std::pair<Operation *, unsigned>>;

// Walk alias-producing users of `v` (SubViewOp / CastOp / ExpandShapeOp /
// CollapseShapeOp) transitively, calling `fn(user, aliasedVal)` on every
// non-alias user.  Returns true as soon as `fn` returns true.
static bool forEachAliasUser(Value v,
                             llvm::function_ref<bool(Operation *, Value)> fn) {
  SmallVector<Value> wl{v};
  llvm::SmallDenseSet<Value> seen;
  while (!wl.empty()) {
    Value cur = wl.pop_back_val();
    if (!seen.insert(cur).second)
      continue;
    for (Operation *user : cur.getUsers()) {
      // even though there should not be any ExtractStridedMetadata at the
      // current location in pass pipeline we still add support for it since
      // it's low cost and makes pass less fragile
      if (isa<memref::SubViewOp, memref::CastOp, memref::ExpandShapeOp,
              memref::CollapseShapeOp, memref::ReshapeOp,
              memref::ReinterpretCastOp, memref::ExtractStridedMetadataOp>(
              user)) {
        wl.push_back(user->getResult(0));
        continue;
      }
      if (fn(user, cur))
        return true;
    }
  }
  return false;
}

// Build the module-wide WrittenArgsSet by fixed-point analysis.
// (func, argIdx) is in the set iff the callee writes to that argument:
//   - any op with MemoryEffects::Write on the arg or an alias of it, OR
//   - the arg (or an alias) is passed to a nested callee whose corresponding
//     parameter is already in the set (transitivity via the fixed-point loop).
// External functions and functions with multiple blocks are skipped; unknown
// callees are handled conservatively at use sites.
static WrittenArgsSet buildWrittenArgsSet(ModuleOp module) {
  WrittenArgsSet written;
  bool changed = true;
  while (changed) {
    changed = false;
    for (auto func : module.getOps<func::FuncOp>()) {
      if (func.isExternal() || !func.getBody().hasOneBlock())
        continue;
      for (BlockArgument arg : func.getBody().front().getArguments()) {
        if (!isa<MemRefType>(arg.getType()))
          continue;
        std::pair<Operation *, unsigned> key{func.getOperation(),
                                             arg.getArgNumber()};
        if (written.contains(key))
          continue;
        bool isWritten = forEachAliasUser(
            arg, [&](Operation *user, Value aliasedVal) -> bool {
              // Direct write via MemoryEffectOpInterface.
              if (mlir::hasEffect<MemoryEffects::Write>(user, aliasedVal))
                return true;
              // Transitive write: alias passed to a nested call whose callee
              // is already known to write that parameter.
              if (auto call = dyn_cast<func::CallOp>(user)) {
                auto callee =
                    module.lookupSymbol<func::FuncOp>(call.getCallee());
                if (!callee)
                  return true; // unknown callee: conservative
                for (OpOperand &use : user->getOpOperands()) {
                  if (use.get() == aliasedVal &&
                      written.contains(
                          {callee.getOperation(), use.getOperandNumber()}))
                    return true;
                }
              }
              return false;
            });
        if (isWritten) {
          written.insert(key);
          changed = true;
        }
      }
    }
  }
  return written;
}

/// Returns true if no op in (from, to) writes to 'val', including nested
/// regions.
static bool noWritesBetween(Value val, Operation *from, Operation *to) {
  assert(from->getBlock() == to->getBlock());
  bool inRange = false;
  for (Operation &op : from->getBlock()->getOperations()) {
    if (&op == from) {
      inRange = true;
      continue;
    }
    if (&op == to)
      break;
    if (!inRange)
      continue;
    if (mlir::hasEffect<MemoryEffects::Write>(&op, val))
      return false;
    for (Region &region : op.getRegions()) {
      bool writtenInRegion = false;
      region.walk([&](Operation *nested) {
        if (mlir::hasEffect<MemoryEffects::Write>(nested, val))
          writtenInRegion = true;
      });
      if (writtenInRegion)
        return false;
    }
  }
  return true;
}

/// Returns true if 'val' has any write users that appear after 'afterOp' in
/// program order. Handles writes in the same block and writes inside nested
/// regions (e.g., scf.for bodies) by walking up to the ancestor in
/// afterOp's block and checking ordering.
/// Tracks alias chains through subviews, casts, expand/collapse/reshape,
/// reinterpret_cast, extract_strided_metadata, and scf.for iter_args.
/// Function calls are resolved via the interprocedural WrittenArgsSet.
static bool hasWriteAfter(Value val, Operation *afterOp,
                          const WrittenArgsSet &writtenArgs) {
  ModuleOp module = afterOp->getParentOfType<ModuleOp>();
  SmallVector<Value> worklist = {val};
  llvm::SmallDenseSet<Value> visited;
  while (!worklist.empty()) {
    Value v = worklist.pop_back_val();
    if (!visited.insert(v).second)
      continue;
    for (Operation *user : v.getUsers()) {
      if (user == afterOp)
        continue;
      if (auto subview = dyn_cast<memref::SubViewOp>(user)) {
        worklist.push_back(subview.getResult());
        continue;
      }
      // Pure alias ops — the result is the same underlying buffer.
      if (isa<memref::CastOp, memref::ExpandShapeOp, memref::CollapseShapeOp,
              memref::ReshapeOp, memref::ReinterpretCastOp>(user)) {
        worklist.push_back(user->getResult(0));
        continue;
      }
      // extract_strided_metadata: result(0) is the base-buffer alias.
      if (isa<memref::ExtractStridedMetadataOp>(user)) {
        worklist.push_back(user->getResult(0));
        continue;
      }
      // When val is used as an init_arg of a scf.for that executes after
      // afterOp, propagate the write check into the corresponding region
      // iter-arg — it aliases val on the first iteration.
      if (auto forOp = dyn_cast<scf::ForOp>(user)) {
        if (!hexagon::detail::happensBefore(afterOp, forOp))
          continue;
        for (auto [initArg, iterArg] :
             llvm::zip(forOp.getInitArgs(), forOp.getRegionIterArgs())) {
          if (initArg == v)
            worklist.push_back(iterArg);
        }
        continue;
      }
      if (!hexagon::detail::happensBefore(afterOp, user))
        continue;
      // For function calls, use the interprocedural WrittenArgsSet:
      // func::CallOp does not implement MemoryEffectOpInterface, so hasEffect
      // returns false even when the callee accumulates into the argument.
      if (auto call = dyn_cast<func::CallOp>(user)) {
        if (!module)
          return true; // no module context: conservative
        auto callee = module.lookupSymbol<func::FuncOp>(call.getCallee());
        if (!callee)
          return true; // unknown callee: conservative
        for (OpOperand &use : user->getOpOperands()) {
          if (use.get() == v && writtenArgs.count({callee.getOperation(),
                                                   use.getOperandNumber()}))
            return true;
        }
        continue; // call resolved: do not fall through to hasEffect
      }
      if (mlir::hasEffect<MemoryEffects::Write>(user, v))
        return true;
    }
  }
  return false;
}

/// Pattern to eliminate a single intermediate buffer allocation in copy chains.
///
/// This pattern matches on a copy operation that writes to an
/// intermediate allocation, then replaces all subsequent reading copies
/// to directly use the original source, eliminating the intermediate buffer.
///
/// Example transformation:
///  ```
///   %src = ...
///   %intermediate = memref.alloc() : memref<NxT>
///   memref.copy %src, %intermediate          // Writing copy (matched here)
///   memref.copy %intermediate, %dst1         // Reading copy 1
///   memref.copy %intermediate, %dst2         // Reading copy 2
///  ```
/// Becomes:
///  ```
///   %src = ...
///   memref.copy %src, %dst1
///   memref.copy %src, %dst2
///  ```
/// The pattern handles both single and multiple reader cases.
struct IntermediateBufferEliminationPattern final
    : public OpRewritePattern<memref::CopyOp> {
  IntermediateBufferEliminationPattern(MLIRContext *ctx)
      : OpRewritePattern(ctx) {}

  LogicalResult matchAndRewrite(memref::CopyOp op,
                                PatternRewriter &rewriter) const override {
    Value source = op.getSource();
    Value target = op.getTarget();
    auto intermediateAlloc = target.getDefiningOp<memref::AllocOp>();
    if (!intermediateAlloc)
      return failure();

    // Validates that all users except this op (writing copy) are copies that
    // read from the intermediate buffer
    SmallVector<memref::CopyOp> readingCopies;
    for (Operation *user : intermediateAlloc->getUsers()) {
      if (user == op.getOperation())
        continue;

      auto copyOp = dyn_cast<memref::CopyOp>(user);
      if (!copyOp)
        return failure();

      if (copyOp.getSource() != intermediateAlloc)
        return failure();

      if (copyOp.getTarget() == intermediateAlloc)
        return failure();

      if (copyOp->getBlock() != op->getBlock() || !op->isBeforeInBlock(copyOp))
        return failure();

      readingCopies.push_back(copyOp);
    }

    if (readingCopies.empty())
      return failure();

    for (auto readingCopy : readingCopies) {
      DBG("Replacing reading copy: " << readingCopy);
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPoint(readingCopy);
      auto newCopy = rewriter.replaceOpWithNewOp<memref::CopyOp>(
          readingCopy, source, readingCopy.getTarget());
      DBG("With new copy: " << newCopy);
    }

    DBG("Removed writing copy: " << op);
    DBG("Removed intermediate buffer: " << intermediateAlloc);
    rewriter.eraseOp(op);
    rewriter.eraseOp(intermediateAlloc);
    return success();
  }
};

/// Pattern to eliminate a duplicate copy from the same source.
/// When two memref.copy ops in the same block copy the same source to two
/// fresh allocations of identical type, and the source is not modified between
/// the two copies, the first copy is redundant if its destination has no other
/// uses. All uses of the first destination are replaced with the second, and
/// the first copy and alloc are erased.
/// e.g.
///    memref.copy %src, %alloc_a   : memref<NxT>
///    memref.copy %src, %alloc_b   : memref<NxT>
///         ..use %alloc_a...
///         ..use %alloc_b..
/// rewritten as:
///    memref.copy %src, %alloc_b   : memref<NxT>
///        ...use %alloc_b...
///        ...use %alloc_b...
struct DuplicateCopyEliminationPattern final
    : public OpRewritePattern<memref::CopyOp> {
  DuplicateCopyEliminationPattern(MLIRContext *ctx,
                                  const WrittenArgsSet *writtenArgs)
      : OpRewritePattern(ctx), writtenArgs(writtenArgs) {}
  const WrittenArgsSet *writtenArgs;

  LogicalResult matchAndRewrite(memref::CopyOp op,
                                PatternRewriter &rewriter) const override {
    Value src = op.getSource();
    Value dst = op.getTarget();

    if (!dst.getDefiningOp<memref::AllocOp>())
      return failure();
    auto dstType = cast<MemRefType>(dst.getType());

    // Walk backwards to find nearest prior copy with matching src and dst.
    memref::CopyOp priorCopy;
    for (Operation *candidate = op->getPrevNode(); candidate != nullptr;
         candidate = candidate->getPrevNode()) {
      auto prevCopy = dyn_cast<memref::CopyOp>(candidate);
      if (!prevCopy || prevCopy.getSource() != src ||
          cast<MemRefType>(prevCopy.getTarget().getType()) != dstType)
        continue;
      if (!prevCopy.getTarget().getDefiningOp<memref::AllocOp>())
        continue;
      priorCopy = prevCopy;
      break;
    }

    if (!priorCopy)
      return failure();

    // src must be unmodified between the two copies, and priorDst must
    // have no use other than priorCopy so it is safe to erase it.
    if (!noWritesBetween(src, priorCopy.getOperation(), op.getOperation()))
      return failure();

    Value priorDst = priorCopy.getTarget();
    if (!priorDst.hasOneUse())
      return failure();
    // priorDst is only a pure snapshot when nothing writes into it after its
    // initialising copy.  Without this check, two independent accumulation
    // buffers initialised from the same source get merged into one.
    if (hasWriteAfter(priorDst, priorCopy.getOperation(), *writtenArgs))
      return failure();

    auto priorAlloc = priorDst.getDefiningOp<memref::AllocOp>();
    auto curAlloc = dst.getDefiningOp<memref::AllocOp>();
    uint64_t priorAlign = priorAlloc.getAlignment().value_or(1);
    uint64_t curAlign = curAlloc.getAlignment().value_or(1);
    if (curAlign < priorAlign)
      return failure();

    DBG("Replacing duplicate copy: " << op);
    DBG("Reusing target of current copy: " << op);
    rewriter.replaceAllUsesWith(priorDst, dst);
    rewriter.eraseOp(priorCopy);
    rewriter.eraseOp(priorDst.getDefiningOp());

    return success();
  }
};

/// Pattern to eliminate a redundant copy by forwarding the source directly.
///
/// When a memref.copy writes to a fresh alloc and the source is never written
/// after the copy, the destination is a redundant snapshot — all its uses can
/// read from the source directly.
///
/// Example:
///   memref.copy %src, %dst   : memref<NxT, 1> to memref<NxT, 1>
///   use_read_only(%dst)
/// Becomes:
///   use_read_only(%src)
///   // %dst alloc and copy are erased
struct CopyForwardingPattern final : public OpRewritePattern<memref::CopyOp> {
  CopyForwardingPattern(MLIRContext *ctx, const WrittenArgsSet *writtenArgs)
      : OpRewritePattern(ctx), writtenArgs(writtenArgs) {}
  const WrittenArgsSet *writtenArgs;

  LogicalResult matchAndRewrite(memref::CopyOp op,
                                PatternRewriter &rewriter) const override {
    Value src = op.getSource();
    Value dst = op.getTarget();

    // Types must match for direct substitution.
    if (src.getType() != dst.getType())
      return failure();

    // dst must be a fresh alloc.
    if (!dst.getDefiningOp<memref::AllocOp>())
      return failure();

    // dst must never be written after this copy.
    if (hasWriteAfter(dst, op.getOperation(), *writtenArgs))
      return failure();

    // src must not be written after this copy.
    if (hasWriteAfter(src, op.getOperation(), *writtenArgs))
      return failure();

    // src must not be used as the source of any other memref.copy whose target
    // is written to after that copy. If it is, IntermediateBufferElimination
    // may redirect those writes back into src, making src effectively written
    // after this copy even though hasWriteAfter does not detect it.
    // Note: we check hasWriteAfter on the other copy's target rather than
    // checking whether the other copy still exists, because the greedy driver
    // may eliminate the other copy first (via DuplicateCopyElimination),
    // leaving src with no copy users when this pattern fires.
    for (Operation *user : src.getUsers()) {
      if (user == op.getOperation())
        continue;
      auto otherCopy = dyn_cast<memref::CopyOp>(user);
      if (!otherCopy || otherCopy.getSource() != src)
        continue;
      if (hasWriteAfter(otherCopy.getTarget(), otherCopy.getOperation(),
                        *writtenArgs))
        return failure();
    }

    DBG("Forwarding copy source: replacing all uses of dst with src");
    rewriter.replaceAllUsesWith(dst, src);
    rewriter.eraseOp(op);
    Operation *allocOp = dst.getDefiningOp();
    if (allocOp->use_empty())
      rewriter.eraseOp(allocOp);

    return success();
  }
};

/// This pattern explicitly adds copies if the yielded value does not match the
/// init_arg. This simplifies subsequent buffer deallocation analyses to dynamic
/// memref aliasing via deallocation helper. In conjuction with other passes and
/// patterns, this enables removal of redundant alloc and copies.
///   Example transformation:
///   ```
///   scf.yield %A : memref<10xf32>
///   ```
///   Becomes:
///   ```
///   memref.copy %A, %loop_arg : memref<10xf32> to memref<10xf32>
///   scf.yield %loop_arg : memref<10xf32>
///   ```
struct AddCopyBeforeYieldPattern final : public OpRewritePattern<scf::YieldOp> {
  AddCopyBeforeYieldPattern(MLIRContext *ctx)
      : OpRewritePattern<scf::YieldOp>(ctx) {}

  LogicalResult matchAndRewrite(scf::YieldOp yieldOp,
                                PatternRewriter &rewriter) const override {
    auto forOp = dyn_cast<scf::ForOp>(yieldOp->getParentOp());
    if (!forOp)
      return failure();

    bool changed = false;
    SmallVector<Value> newYieldOperands =
        llvm::to_vector<4>(yieldOp.getOperands());

    for (const auto &operand : llvm::enumerate(yieldOp.getOperands())) {
      size_t index = operand.index();
      Value yieldedValue = operand.value();

      if (!mlir::isa<MemRefType>(yieldedValue.getType()))
        continue;

      BlockArgument iterArg = forOp.getRegionIterArgs()[index];
      if (yieldedValue == iterArg)
        continue;

      // If the yieldedValue is a memref and is NOT the iter_arg.
      // Then, we need to insert a copy to canonicalize the IR.
      rewriter.setInsertionPoint(yieldOp); // Insert copy before the yieldOp
      auto newCopy = memref::CopyOp::create(rewriter, yieldOp.getLoc(),
                                            yieldedValue, iterArg);
      DBG("Added memref.copy before scf.yield: " << newCopy);

      // The new operand for the yield will be the iter_arg itself
      newYieldOperands[index] = iterArg;
      changed = true;
    }

    if (!changed)
      return failure();

    rewriter.replaceOpWithNewOp<scf::YieldOp>(yieldOp, newYieldOperands);
    return success();
  }
};

// Walk every non-subview user of `v` and of any memref.subview chained off
// it, calling `visit(user, parent)`. Returns true iff visit returned true
// for every user.
static bool walkAllocUsers(Value v,
                           llvm::function_ref<bool(Operation *, Value)> visit) {
  for (Operation *user : v.getUsers()) {
    if (auto sv = dyn_cast<memref::SubViewOp>(user)) {
      if (!walkAllocUsers(sv.getResult(), visit))
        return false;
      continue;
    }
    if (!visit(user, v))
      return false;
  }
  return true;
}

// True if `v` is an arith.constant of zero (scalar or splat).
static bool isZeroConstantValue(Value v) {
  auto constOp = v.getDefiningOp<arith::ConstantOp>();
  if (!constOp)
    return false;
  Attribute attr = constOp.getValue();
  if (auto dense = dyn_cast<DenseElementsAttr>(attr)) {
    if (!dense.isSplat())
      return false;
    attr = dense.getSplatValue<Attribute>();
  }
  if (auto f = dyn_cast<FloatAttr>(attr))
    return f.getValue().isZero();
  if (auto i = dyn_cast<IntegerAttr>(attr))
    return i.getValue().isZero();
  return false;
}

// True if `op` writes a zero into `target` via linalg.fill of a zero scalar
// or vector.transfer_write of a zero splat.
static bool isZeroFillInto(Operation *op, Value target) {
  if (auto fill = dyn_cast<linalg::FillOp>(op))
    return fill.getOutputs()[0] == target &&
           isZeroConstantValue(fill.getInputs()[0]);
  if (auto tw = dyn_cast<vector::TransferWriteOp>(op))
    return tw.getBase() == target && isZeroConstantValue(tw.getVector());
  return false;
}

// True if every user of `allocOp` (through subview chains) is a zero-fill
// writer, a memref.copy reader, or a memref.dealloc.
static bool isZeroFilledAlloc(memref::AllocOp allocOp) {
  return walkAllocUsers(allocOp->getResult(0), [](Operation *user, Value v) {
    if (isa<memref::DeallocOp>(user))
      return true;
    if (auto copy = dyn_cast<memref::CopyOp>(user))
      return copy.getSource() == v;
    return isZeroFillInto(user, v);
  });
}

// Defends against a non-zero writer reaching the alloc through an aliasing
// path that the SSA user-walk in isZeroFilledAlloc cannot see (e.g. a
// captured subview, or the buffer passed to a call). Requires the alloc's
// first non-dealloc user in its own block to be a zero-fill of the buffer.
static bool firstUseIsZeroFill(memref::AllocOp allocOp) {
  Value alloc = allocOp->getResult(0);
  Block *allocBlock = allocOp->getBlock();
  Operation *firstUser = nullptr;
  for (Operation *user : alloc.getUsers()) {
    if (user->getBlock() != allocBlock)
      return false;
    if (isa<memref::DeallocOp>(user))
      continue;
    if (!firstUser || user->isBeforeInBlock(firstUser))
      firstUser = user;
  }
  return firstUser && isZeroFillInto(firstUser, alloc);
}

// True if any user of `v` (through subviews) is a memref.copy.
static bool hasCopyUserThroughSubviews(Value v) {
  return !walkAllocUsers(
      v, [](Operation *user, Value) { return !isa<memref::CopyOp>(user); });
}

// Collect every non-dealloc user of `v` and the subviews themselves, in
// post-order so leaves are erased before their parent subviews.
static void collectErasable(Value v, SmallVectorImpl<Operation *> &out) {
  for (Operation *user : v.getUsers()) {
    if (auto sv = dyn_cast<memref::SubViewOp>(user)) {
      collectErasable(sv.getResult(), out);
      out.push_back(sv);
    } else if (!isa<memref::DeallocOp>(user)) {
      out.push_back(user);
    }
  }
}

// Rewrite a memref.copy whose source is a zero-filled alloc into a fill on
// the destination, e.g.:
//
// Replace:
//   %zeros = memref.alloc()
//   linalg.fill ins(%zero) outs(%zeros)
//   memref.copy %zeros, %dst
// With:
//    linalg.fill ins(%zero) outs(%dst)
//
// If no other copy reads from the alloc, the alloc and its fills are
// erased — turning a malloc + memset + memcpy into a single memset.
struct CopyFromZeroFilledAllocPattern final
    : public OpRewritePattern<memref::CopyOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::CopyOp op,
                                PatternRewriter &rewriter) const override {
    Value source = op.getSource();
    auto allocOp = source.getDefiningOp<memref::AllocOp>();
    if (!allocOp || !isZeroFilledAlloc(allocOp) || !firstUseIsZeroFill(allocOp))
      return failure();

    Type elemType = cast<MemRefType>(source.getType()).getElementType();
    Value zeroVal = arith::ConstantOp::create(rewriter, op.getLoc(),
                                              rewriter.getZeroAttr(elemType));
    rewriter.replaceOpWithNewOp<linalg::FillOp>(op, ValueRange{zeroVal},
                                                ValueRange{op.getTarget()});

    if (!hasCopyUserThroughSubviews(allocOp->getResult(0))) {
      SmallVector<Operation *> toErase;
      collectErasable(allocOp->getResult(0), toErase);
      for (Operation *o : toErase)
        rewriter.eraseOp(o);
      rewriter.eraseOp(allocOp);
      DBG("Erased zero alloc and its fills: " << allocOp);
    }
    return success();
  }
};

struct CopyCanonicalizationPass
    : public ::impl::CopyCanonicalizationBase<CopyCanonicalizationPass> {

  void runOnOperation() override {
    // Build the interprocedural written-args map once for this function's
    // enclosing module so hasWriteAfter can resolve whether a callee writes to
    // a memref argument.  Falls back to an empty set when there is no enclosing
    // module (isolated unit tests).
    WrittenArgsSet writtenArgs;
    if (auto module = getOperation()->getParentOfType<ModuleOp>())
      writtenArgs = buildWrittenArgsSet(module);

    DominanceInfo &domInfo = getAnalysis<DominanceInfo>();
    RewritePatternSet patterns(&getContext());
    patterns.add<IntermediateBufferEliminationPattern>(patterns.getContext());
    patterns.add<CopyForwardingPattern>(patterns.getContext(), &writtenArgs);
    patterns.add<DuplicateCopyEliminationPattern>(patterns.getContext(),
                                                  &writtenArgs);
    patterns.add<AddCopyBeforeYieldPattern>(patterns.getContext());
    patterns.add<CopyFromZeroFilledAllocPattern>(patterns.getContext());
    mlir::hexagon::populateCopyCanonicalizationAllocPatterns(patterns, domInfo);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      return signalPassFailure();
  }
};

} // end anonymous namespace

std::unique_ptr<OperationPass<func::FuncOp>>
hexagon::createCopyCanonicalizationPass() {
  return std::make_unique<CopyCanonicalizationPass>();
}
