//===- SignInference.h - Tri-state sign inference -------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// Tri-state sign inference (Positive / Negative / Unknown) over MLIR values,
// with an optimistic-init fixpoint for loop-carried CFG block arguments.
//
// Shared between FastInversePass (div->rsqrt rewrite) and ExpandMathOpsPass
// (pow->exp+log rewrite when base is provably positive).
//
//===----------------------------------------------------------------------===//

#ifndef HEXAGON_COMMON_SIGNINFERENCE_H
#define HEXAGON_COMMON_SIGNINFERENCE_H

#include "hexagon/Transforms/Transforms.h" // kHexagonNonNegativeAttrName
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace hexagon {

// Tri-state sign inference for the fast-inverse reciprocal guard.
//
// The generic rewrite  p / q  -->  p * Quake_rsqrt(q*q)  is only sound when q
// is provably non-negative (rsqrt(q*q) == 1/|q| == 1/q iff q >= 0).  With a
// tri-state result we can also rewrite the provably-negative case:
//   p / q  (q < 0)  -->  -(p * Quake_rsqrt(q*q))
// because |q| == -q, so 1/q == -1/|q|.
//
// Also used in ExpandMathOpsPass to decide whether math.powf(base, exp) can
// safely be rewritten to math.exp(arith.mulf(exp, math.log(base))): the
// expansion is only valid when base > 0 (log(base) is NaN for base <= 0).
//
// Inference rules (structural, memoized def-use walk):
//   Positive:  absf, sqrt, exp, exp2, hexagon.nonneg marker
//              mulf(Pos, Pos) | mulf(Neg, Neg) | self-mulf(x, x)
//              addf(Pos, Pos) | maximumf/maxnumf(Pos, Pos)
//              negf(Negative)
//              select(_, Pos, Pos)
//              reduction<add|mul|maxf>(Positive vector)
//              broadcast(Positive)
//              shuffle(Positive, Positive) | extract_strided_slice(Positive)
//              | extract(Positive)
//   Negative:  negf(Positive)
//              mulf(Pos, Neg) | mulf(Neg, Pos)
//              addf(Neg, Neg) | maximumf/maxnumf(Neg, Neg)
//              select(_, Neg, Neg)
//              reduction<add|mul|maxf>(Negative vector)
//              broadcast(Negative)
//              shuffle(Negative, Negative) | extract_strided_slice(Negative)
//              | extract(Negative)
//   Unknown:   everything else
//
// Note: arith.divf with a constant denominator is folded to mulf(x, recip)
// before FastInverse runs, so constant denominators never appear as the
// top-level denom. A non-negative float constant encountered while recursing
// (e.g. the 0.0 init operand of a loop-carried accumulator) is treated as
// Positive; -0.0 and everything non-provable stay Unknown, bounding the
// analysis conservatively.
//
// Loop-carried accumulators (online-softmax / flash-attention `l_i`) are the
// motivating case. FastInverse runs AFTER SCFToControlFlow, so the accumulator
// is a `cf` block argument, not an scf.for iter-arg. Its sign depends on the
// init operand and on the loop-carried yielded value, which references the
// block argument itself -- a def-use cycle. A one-shot DFS reads back the
// tentative cycle-break seed and collapses to Unknown. To prove such an
// accumulator >= 0 we run an optimistic-init greatest-fixpoint over the CFG
// block arguments in the denominator's dependency slice:
//
//   1. Seed every block argument to Positive (optimistic top, ">= 0").
//   2. Repeatedly recompute each block argument as the meet of the values
//      forwarded to it by every CFG predecessor (via BranchOpInterface),
//      lowering estimates until stable.
//   3. A block argument that survives as Positive is proved >= 0, because at
//      the fixpoint every predecessor value is >= 0 under the same estimates
//      (a valid inductive proof). Monotone descent over a height-3 lattice
//      guarantees termination; op results form a DAG (no extra cycles).
//
// The optimistic seed can only prove Positive (a genuinely negative or
// unprovable loop-carried value meets down to Unknown), so it never claims a
// sign the value does not have.
enum class Sign { Positive, Negative, Unknown };

namespace {
// Sign inference driven by an optimistic-init fixpoint over CFG block
// arguments (see the block comment above). One instance per query; the public
// entry point is inferSign() below.
class SignInference {
public:
  Sign infer(Value root) {
    // Discover the cyclic leaves (CFG block arguments) reachable from `root`.
    llvm::DenseSet<Value> visited;
    collectBlockArgs(root, visited);

    // Pure DAG (no loop-carried block arg): a single memoized pass suffices.
    if (blockArgs.empty()) {
      llvm::DenseMap<Value, Sign> opCache;
      return eval(root, opCache);
    }

    // Optimistic greatest-fixpoint over the discovered block arguments.
    for (Value ba : blockArgs)
      argSign[ba] = Sign::Positive;

    bool changed = true;
    unsigned guard = 0;
    const unsigned maxIters = blockArgs.size() * 3 + 3;
    while (changed && guard++ < maxIters) {
      changed = false;
      for (Value ba : blockArgs) {
        Sign next = meet(argSign[ba], evalBlockArg(cast<BlockArgument>(ba)));
        if (next != argSign[ba]) {
          argSign[ba] = next;
          changed = true;
        }
      }
    }

    llvm::DenseMap<Value, Sign> opCache;
    return eval(root, opCache);
  }

private:
  // Block arguments in the query's dependency slice, seeded/refined by the
  // fixpoint. SmallVector keeps a stable iteration order; collectBlockArgs
  // pushes each value at most once (guarded by the shared visited set).
  llvm::SmallVector<Value, 4> blockArgs;
  llvm::DenseMap<Value, Sign> argSign;

  // Lattice meet toward Unknown: equal signs are preserved, disagreement is
  // Unknown. This is the only combinator the fixpoint needs.
  static Sign meet(Sign a, Sign b) { return (a == b) ? a : Sign::Unknown; }

  // Recompute one block argument as the meet of the values forwarded to it by
  // every CFG predecessor. Forwarded op-result operands are evaluated with a
  // fresh op cache that reads the current block-arg estimates as leaves.
  Sign evalBlockArg(BlockArgument ba) {
    Block *block = ba.getOwner();
    unsigned idx = ba.getArgNumber();
    bool any = false;
    Sign acc = Sign::Positive;
    for (Block *pred : block->getPredecessors()) {
      Operation *term = pred->getTerminator();
      // An edge we cannot reason about (non-branch terminator) makes the whole
      // join Unknown.
      auto branch = dyn_cast<BranchOpInterface>(term);
      if (!branch)
        return Sign::Unknown;
      for (unsigned s = 0, e = term->getNumSuccessors(); s < e; ++s) {
        if (term->getSuccessor(s) != block)
          continue;
        Value fwd = branch.getSuccessorOperands(s)[idx];
        Sign fs = Sign::Unknown;
        if (fwd) {
          llvm::DenseMap<Value, Sign> opCache;
          fs = eval(fwd, opCache);
        }
        acc = any ? meet(acc, fs) : fs;
        any = true;
      }
    }
    // A block argument with no CFG predecessors (a function/entry parameter)
    // has no provable sign.
    return any ? acc : Sign::Unknown;
  }

  // Gather the block arguments reachable from `v` through op operands and,
  // for each block argument, through the values its predecessors forward to
  // it. The shared visited set bounds the walk and breaks the loop backedge.
  void collectBlockArgs(Value v, llvm::DenseSet<Value> &visited) {
    if (!visited.insert(v).second)
      return;
    if (auto ba = dyn_cast<BlockArgument>(v)) {
      blockArgs.push_back(v);
      Block *block = ba.getOwner();
      unsigned idx = ba.getArgNumber();
      for (Block *pred : block->getPredecessors()) {
        Operation *term = pred->getTerminator();
        auto branch = dyn_cast<BranchOpInterface>(term);
        if (!branch)
          continue;
        for (unsigned s = 0, e = term->getNumSuccessors(); s < e; ++s) {
          if (term->getSuccessor(s) != block)
            continue;
          if (Value fwd = branch.getSuccessorOperands(s)[idx])
            collectBlockArgs(fwd, visited);
        }
      }
      return;
    }
    if (Operation *def = v.getDefiningOp())
      for (Value operand : def->getOperands())
        collectBlockArgs(operand, visited);
  }

  // Walk a memref value backward through subview / reinterpret_cast /
  // expand_shape / collapse_shape to find the root memref.alloca. Returns
  // nullptr if the root is not a local alloca (e.g. function argument or
  // dynamic allocation we cannot reason about).
  static Value traceMemrefToAlloca(Value memref) {
    constexpr unsigned kMaxSteps = 10;
    Value current = memref;
    for (unsigned i = 0; i < kMaxSteps; ++i) {
      Operation *def = current.getDefiningOp();
      if (!def)
        return nullptr; // block argument (function param) — can't reason
      if (isa<memref::AllocaOp>(def))
        return current;
      if (auto sv = dyn_cast<memref::SubViewOp>(def)) {
        current = sv.getSource();
        continue;
      }
      if (auto rc = dyn_cast<memref::ReinterpretCastOp>(def)) {
        current = rc.getSource();
        continue;
      }
      if (auto es = dyn_cast<memref::ExpandShapeOp>(def)) {
        current = es.getSrc();
        continue;
      }
      if (auto cs = dyn_cast<memref::CollapseShapeOp>(def)) {
        current = cs.getSrc();
        continue;
      }
      return nullptr; // unknown op in the chain
    }
    return nullptr;
  }

  // Given a root alloca, find all vector.transfer_write ops that store to it
  // (through subviews etc.) and return the meet of their stored values' signs.
  // If any writer cannot be proven non-negative, returns Unknown.
  Sign analyzeAllocaWriters(Value alloca, llvm::DenseMap<Value, Sign> &cache) {
    Sign acc = Sign::Positive;
    bool foundWriter = false;
    // Walk all transitive users of the alloca to find transfer_write ops.
    SmallVector<Value, 8> worklist;
    llvm::DenseSet<Value> seen;
    worklist.push_back(alloca);
    seen.insert(alloca);
    while (!worklist.empty()) {
      Value current = worklist.pop_back_val();
      for (Operation *user : current.getUsers()) {
        // A transfer_write stores a value — check its sign.
        if (auto write = dyn_cast<vector::TransferWriteOp>(user)) {
          foundWriter = true;
          Sign ws = eval(write.getVector(), cache);
          acc = meet(acc, ws);
          if (acc == Sign::Unknown)
            return Sign::Unknown; // early exit
          continue;
        }
        // A subview / reinterpret_cast / expand / collapse produces a new
        // memref aliasing the same alloca — follow its users too.
        if (isa<memref::SubViewOp, memref::ReinterpretCastOp,
                memref::ExpandShapeOp, memref::CollapseShapeOp>(user)) {
          Value result = user->getResult(0);
          if (seen.insert(result).second)
            worklist.push_back(result);
          continue;
        }
        // A transfer_read is fine — it only reads, doesn't affect stored sign.
        if (isa<vector::TransferReadOp>(user))
          continue;
        // memref.dealloc and extract_strided_metadata don't write data.
        if (isa<memref::DeallocOp, memref::ExtractStridedMetadataOp>(user))
          continue;
        // Any other user is conservatively treated as a potential writer.
        return Sign::Unknown;
      }
    }
    return foundWriter ? acc : Sign::Unknown;
  }

  // Memoized structural sign of an op-result value. Block-argument leaves read
  // the fixpoint estimate; op results form a DAG so a plain memo (no cycle
  // seed) is sufficient.
  Sign eval(Value value, llvm::DenseMap<Value, Sign> &cache) {
    if (isa<BlockArgument>(value)) {
      auto it = argSign.find(value);
      return it != argSign.end() ? it->second : Sign::Unknown;
    }

    auto it = cache.find(value);
    if (it != cache.end())
      return it->second;

    auto record = [&](Sign s) -> Sign {
      cache[value] = s;
      return s;
    };

    Operation *def = value.getDefiningOp();
    if (!def)
      return record(Sign::Unknown);

    // Explicit non-negativity marker placed by an earlier pass (e.g. the
    // fast-exp approximation advertises its result is >= 0 via hexagon.nonneg
    // so that sum(approx_exp) denominators keep the reciprocal optimization).
    if (def->hasAttr(kHexagonNonNegativeAttrName))
      return record(Sign::Positive);

    // ub.poison is undefined behavior to observe, so it can take any sign.
    // Treating it as Positive prevents it from poisoning shuffle evaluations
    // where poison is used as the unused V2 operand.
    if (isa<ub::PoisonOp>(def))
      return record(Sign::Positive);

    // A non-negative (and not -0.0) float constant is >= 0. This proves the
    // 0.0 init operand of a loop-carried accumulator; -0.0 stays Unknown so
    // the negative-zero divide is preserved.
    if (auto cst = dyn_cast<arith::ConstantOp>(def)) {
      if (auto fattr = dyn_cast<FloatAttr>(cst.getValue())) {
        const llvm::APFloat &v = fattr.getValue();
        if (!v.isNegative() && !v.isNaN())
          return record(Sign::Positive);
      } else if (auto dense = dyn_cast<DenseFPElementsAttr>(cst.getValue())) {
        bool allNonNeg = true;
        for (const llvm::APFloat &v : dense.getValues<llvm::APFloat>())
          if (v.isNegative() || v.isNaN()) {
            allNonNeg = false;
            break;
          }
        if (allNonNeg)
          return record(Sign::Positive);
      }
      return record(Sign::Unknown);
    }

    // |x| and sqrt(x) are always >= 0.
    if (isa<math::AbsFOp, math::SqrtOp>(def))
      return record(Sign::Positive);

    // exp(x) and exp2(x) are always > 0.
    if (isa<math::ExpOp, math::Exp2Op>(def))
      return record(Sign::Positive);

    // negf flips the sign of its operand.
    if (isa<arith::NegFOp>(def)) {
      Sign inner = eval(def->getOperand(0), cache);
      if (inner == Sign::Positive)
        return record(Sign::Negative);
      if (inner == Sign::Negative)
        return record(Sign::Positive);
      return record(Sign::Unknown);
    }

    // mulf: sign is the product of the operand signs.
    //   Pos * Pos = Pos,  Neg * Neg = Pos,  Pos * Neg = Neg,  x * x = Pos.
    if (auto mul = dyn_cast<arith::MulFOp>(def)) {
      if (mul.getLhs() == mul.getRhs())
        return record(Sign::Positive);
      Sign ls = eval(mul.getLhs(), cache);
      Sign rs = eval(mul.getRhs(), cache);
      if (ls == Sign::Unknown || rs == Sign::Unknown)
        return record(Sign::Unknown);
      return record((ls == rs) ? Sign::Positive : Sign::Negative);
    }

    // addf: both same sign => result has that sign; else Unknown.
    if (auto add = dyn_cast<arith::AddFOp>(def)) {
      Sign ls = eval(add.getLhs(), cache);
      Sign rs = eval(add.getRhs(), cache);
      if (ls != Sign::Unknown && ls == rs)
        return record(ls);
      return record(Sign::Unknown);
    }

    // maximumf / maxnumf: both same sign => result has that sign; else Unknown.
    if (auto mx = dyn_cast<arith::MaximumFOp>(def)) {
      Sign ls = eval(mx.getLhs(), cache);
      Sign rs = eval(mx.getRhs(), cache);
      if (ls != Sign::Unknown && ls == rs)
        return record(ls);
      return record(Sign::Unknown);
    }
    if (auto mx = dyn_cast<arith::MaxNumFOp>(def)) {
      Sign ls = eval(mx.getLhs(), cache);
      Sign rs = eval(mx.getRhs(), cache);
      if (ls != Sign::Unknown && ls == rs)
        return record(ls);
      return record(Sign::Unknown);
    }

    // Horizontal reductions: add / mul / max preserve the element sign when it
    // is uniform across the vector. This is the softmax case:
    //   vector.reduction <add> over exp(...) --> Positive.
    if (auto red = dyn_cast<vector::ReductionOp>(def)) {
      switch (red.getKind()) {
      case vector::CombiningKind::ADD:
      case vector::CombiningKind::MUL:
      case vector::CombiningKind::MAXIMUMF:
      case vector::CombiningKind::MAXNUMF:
        return record(eval(red.getVector(), cache));
      default:
        return record(Sign::Unknown);
      }
    }

    // A broadcast preserves the sign of its source.
    if (auto bcast = dyn_cast<vector::BroadcastOp>(def))
      return record(eval(bcast.getSource(), cache));

    // Lane-permuting / lane-selecting ops preserve the element sign when the
    // value(s) they draw from are uniformly signed. These appear when the
    // BroadcastedReductionToButterfly pass rewrites a softmax-style
    //   broadcast(reduction<add>(exp(...)))
    // denominator into a lane-wise butterfly built from vector.shuffle +
    // arith.addf + vector.extract_strided_slice. Without these passthrough
    // rules inferSign() would return Unknown for the post-butterfly
    // denominator and the reciprocal optimization would silently stop firing
    // for softmax.

    // vector.shuffle selects lanes from its two source vectors; if both
    // sources share a sign, every selected lane has that sign.
    if (auto shuffle = dyn_cast<vector::ShuffleOp>(def)) {
      Sign v1 = eval(shuffle.getV1(), cache);
      Sign v2 = eval(shuffle.getV2(), cache);
      if (v1 != Sign::Unknown && v1 == v2)
        return record(v1);
      return record(Sign::Unknown);
    }

    // vector.extract_strided_slice and vector.extract take a sub-slice /
    // single lane of their source, preserving the per-element sign.
    if (auto slice = dyn_cast<vector::ExtractStridedSliceOp>(def))
      return record(eval(slice.getSource(), cache));
    if (auto extract = dyn_cast<vector::ExtractOp>(def))
      return record(eval(extract.getSource(), cache));

    // arith.select: if both branches have the same sign, the result has that
    // sign. This handles the integer-select inf/underflow fixup emitted by
    // FastExpApproxPass, where both the true (posInf / zero) and false (nonneg
    // mulf) branches are non-negative.
    if (auto sel = dyn_cast<arith::SelectOp>(def)) {
      Sign ts = eval(sel.getTrueValue(), cache);
      Sign fs = eval(sel.getFalseValue(), cache);
      if (ts != Sign::Unknown && ts == fs)
        return record(ts);
      return record(Sign::Unknown);
    }

    // Float extension/truncation preserves sign (widening or narrowing a
    // positive value stays positive).
    if (isa<arith::ExtFOp, arith::TruncFOp>(def))
      return record(eval(def->getOperand(0), cache));

    // vector.transfer_read from a local alloca: if every transfer_write to
    // that alloca stores a non-negative value, the loaded value is also
    // non-negative. This bridges the sign gap introduced when
    // SplitReduceGeneric + bufferization spill exp() results to scratch.
    if (auto read = dyn_cast<vector::TransferReadOp>(def)) {
      Value base = traceMemrefToAlloca(read.getBase());
      if (base)
        return record(analyzeAllocaWriters(base, cache));
      return record(Sign::Unknown);
    }

    return record(Sign::Unknown);
  }
};
} // namespace

static Sign inferSign(Value value) { return SignInference().infer(value); }

} // namespace hexagon
} // namespace mlir

#endif // HEXAGON_COMMON_SIGNINFERENCE_H
