//===-- RewriteI64PtrArgsToI32.cpp ---------------------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
// [TTIR] Fix index tensor ABI mismatch: treat !tt.ptr<i64> args as
// !tt.ptr<i32> + extsi.
//
// The runtime allocates index-like tensors as i32 buffers (even when the
// logical/FX type is i64) to avoid duplicating kernel coverage for i64. Triton
// TTIR kernels may still declare these inputs as !tt.ptr<i64> and issue i64
// loads, causing incorrect reads / crashes against the i32 storage.
//
// This transform normalizes the pointer ABI:
// 1) Rewrite tt.func entry args !tt.ptr<i64> to !tt.ptr<i32>.
// 2) Propagate the rewrite through all derived pointer-like SSA values
//    originating from those args (without hard-coding an op chain).
// 3) Rewrite tt.load from the rewritten pointer closure to load i32, then
//    insert arith.extsi at each use-site to preserve i64 semantics.
//
// Runs early in TTIR, before lowering introduces extra corner cases.
//===----------------------------------------------------------------------===//

#include "python/triton_qcom_hexagon_backend_api.h"

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "rewrite-i64-ptr-args-to-i32"
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define DBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace hexagon_backend {
namespace {
using namespace mlir;
namespace triton = mlir::triton;

// Type helpers
bool isI64PtrType(Type ty) {
  auto ptrTy = dyn_cast<triton::PointerType>(ty);
  if (!ptrTy)
    return false;
  auto intTy = dyn_cast<IntegerType>(ptrTy.getPointeeType());
  return intTy && intTy.getWidth() == 64;
}

bool isPointerLike(Type ty) {
  if (isa<triton::PointerType>(ty))
    return true;
  if (auto tTy = dyn_cast<RankedTensorType>(ty))
    return isa<triton::PointerType>(tTy.getElementType());
  return false;
}

triton::PointerType getPointerElement(Type ty) {
  if (auto p = dyn_cast<triton::PointerType>(ty))
    return p;
  if (auto tTy = dyn_cast<RankedTensorType>(ty))
    return dyn_cast<triton::PointerType>(tTy.getElementType());
  return {};
}

Type withPointerPointeeAsI32(Type oldTy) {
  auto oldPtr = getPointerElement(oldTy);
  if (!oldPtr)
    return oldTy;

  auto oldPointeeInt = dyn_cast<IntegerType>(oldPtr.getPointeeType());
  if (!oldPointeeInt || oldPointeeInt.getWidth() != 64)
    return oldTy;

  MLIRContext *ctx = oldTy.getContext();
  auto i32 = IntegerType::get(ctx, 32);
  auto newPtr = triton::PointerType::get(i32, oldPtr.getAddressSpace());

  if (isa<triton::PointerType>(oldTy))
    return newPtr;

  auto tTy = cast<RankedTensorType>(oldTy);
  return RankedTensorType::get(tTy.getShape(), newPtr, tTy.getEncoding());
}

Type i64ToI32Like(Type oldTy) {
  auto i32 = IntegerType::get(oldTy.getContext(), 32);

  if (auto iTy = dyn_cast<IntegerType>(oldTy))
    return iTy.getWidth() == 64 ? i32 : oldTy;

  if (auto tTy = dyn_cast<RankedTensorType>(oldTy)) {
    auto elem = dyn_cast<IntegerType>(tTy.getElementType());
    if (elem && elem.getWidth() == 64)
      return RankedTensorType::get(tTy.getShape(), i32, tTy.getEncoding());
  }

  return oldTy;
}

// Collect derived pointer-like values (no op-kind assumptions): the transitive
// closure from `root` of pointer-like values (ptr or tensor of ptr) produced by
// ops that consume an already-tracked value.
void collectDerivedPointerLikeValues(Value root,
                                     llvm::DenseSet<Value> &outSet) {
  SmallVector<Value, 64> worklist;

  auto push = [&](Value v) {
    if (!v || !isPointerLike(v.getType()) || outSet.count(v))
      return;
    outSet.insert(v);
    worklist.push_back(v);
  };

  push(root);
  while (!worklist.empty()) {
    Value v = worklist.pop_back_val();
    for (Operation *user : v.getUsers()) {
      // Require at least one tracked operand (not necessarily `v`).
      bool usesTracked = false;
      for (Value operand : user->getOperands()) {
        if (outSet.count(operand)) {
          usesTracked = true;
          break;
        }
      }
      if (!usesTracked)
        continue;
      for (Value res : user->getResults()) {
        if (withPointerPointeeAsI32(res.getType()) != res.getType())
          push(res);
      }
    }
  }
}

// Per-use-site extsi insertion
void insertExtSIAtUse(IRRewriter &rewriter, OpOperand &use, Value newI32Value,
                      Type oldI64Type) {
  Operation *userOp = use.getOwner();
  // In valid SSA the definition always dominates all its uses, so inserting
  // immediately before the user is always safe.
  rewriter.setInsertionPoint(userOp);
  auto extOp = arith::ExtSIOp::create(rewriter, userOp->getLoc(), oldI64Type,
                                      newI32Value);
  use.set(extOp.getResult());
}

// Narrow a store's i64 value operand to match a pointer that was rewritten to
// ptr<i32>. Computation is kept in i64 (loads are extsi'd on the way in), so a
// store into a narrowed destination must trunci on the way out. This is the
// symmetric counterpart to insertExtSIAtUse.
void insertTruncIBeforeStore(IRRewriter &rewriter, OpOperand &valueUse,
                             Type newI32Type) {
  Operation *storeOp = valueUse.getOwner();
  Value oldValue = valueUse.get();
  rewriter.setInsertionPoint(storeOp);
  auto truncOp = arith::TruncIOp::create(rewriter, storeOp->getLoc(),
                                         newI32Type, oldValue);
  valueUse.set(truncOp.getResult());
  DBG("trunc(s) have been inserted before a tt.store at " << storeOp->getLoc());
}

} // namespace

bool rewriteI64PtrArgsToI32(mlir::ModuleOp &module_op) {
  bool changed = false;
  MLIRContext *ctx = module_op.getContext();
  IRRewriter rewriter(ctx);

  module_op.walk([&](triton::FuncOp func) {
    if (func.getBody().empty())
      return;

    Block &entry = func.getBody().front();
    FunctionType oldFnTy = func.getFunctionType();
    SmallVector<unsigned, 8> candidateIdx;
    SmallVector<Type> newInputs(oldFnTy.getInputs().begin(),
                                oldFnTy.getInputs().end());

    // 1) Find all !tt.ptr<i64> args.
    for (BlockArgument barg : entry.getArguments()) {
      if (!isI64PtrType(barg.getType()))
        continue;
      unsigned idx = barg.getArgNumber();
      candidateIdx.push_back(idx);
      newInputs[idx] = withPointerPointeeAsI32(barg.getType());
    }

    if (candidateIdx.empty())
      return;

    // 2) Update function signature
    auto newFnTy = FunctionType::get(ctx, newInputs, oldFnTy.getResults());
    func.setType(newFnTy);
    // 3) Update block arg types
    for (unsigned idx : candidateIdx)
      entry.getArgument(idx).setType(newInputs[idx]);
    changed = true;

    // Build the pointer-like closure across all rewritten args once. The union
    // is sufficient for every downstream step: retyping (4a) is idempotent,
    // the load filter (4b) only needs membership, and store reconciliation (5)
    // needs the full set regardless of arg order. Collecting per-arg as well
    // would just re-traverse the same SSA graph.
    llvm::DenseSet<Value> narrowedPtrs;
    for (unsigned idx : candidateIdx)
      collectDerivedPointerLikeValues(entry.getArgument(idx), narrowedPtrs);

    // 4a) Retype derived pointer-like SSA values to ptr<i32>.
    for (Value v : narrowedPtrs) {
      Type oldTy = v.getType();
      Type newTy = withPointerPointeeAsI32(oldTy);
      if (newTy == oldTy)
        continue;
      if (auto *defOp = v.getDefiningOp())
        rewriter.modifyOpInPlace(defOp, [&] { v.setType(newTy); });
      else
        v.setType(newTy);
    }

    // 4b) Rewrite tt.load whose ptr operand is in the closure: load i32, then
    // insert extsi at each use-site.
    func.walk([&](triton::LoadOp load) {
      Value ptr = load.getPtr();
      if (!narrowedPtrs.count(ptr))
        return;

      Value loadRes = load.getResult();
      Type oldLoadTy = loadRes.getType();
      Type newLoadTy = i64ToI32Like(oldLoadTy);
      if (newLoadTy == oldLoadTy) {
        DBG("Skipping tt.load rewrite because result type does not need "
            "i64->i32 narrowing at "
            << load.getLoc());
        return;
      }

      // An "other" operand must match the result type; truncate i64 to i32.
      if (auto other = load.getOther()) {
        Type oldOtherTy = other.getType();
        Type newOtherTy = i64ToI32Like(oldOtherTy);
        if (newOtherTy != oldOtherTy) {
          rewriter.setInsertionPoint(load);
          auto truncOp = arith::TruncIOp::create(rewriter, load.getLoc(),
                                                 newOtherTy, other);
          load.getOtherMutable().assign(truncOp.getResult());
        }
      }

      loadRes.setType(newLoadTy);
      // Snapshot uses before rewriting them, then extsi each use-site so all
      // downstream computation stays in i64 exactly as authored. Stores into
      // a narrowed (ptr<i32>) destination — including a direct load->store
      // copy — are reconciled uniformly in step 5 by trunci'ing the value
      // back to i32 to match the pointee.
      SmallVector<OpOperand *, 16> uses;
      uses.reserve(std::distance(loadRes.use_begin(), loadRes.use_end()));
      for (OpOperand &u : loadRes.getUses())
        uses.push_back(&u);
      for (OpOperand *usePtr : uses)
        insertExtSIAtUse(rewriter, *usePtr, loadRes, oldLoadTy);

      changed = true;
    });

    // 5) Reconcile stores whose destination pointer was narrowed to ptr<i32>.
    // The stored value is still i64 (computation is unchanged); narrow it with
    // trunci so the value type matches the i32 pointee. Covers both the direct
    // load->store copy and the case where i64 arithmetic sits in between.
    // Snapshot the stores first, then mutate, so the walk never observes IR
    // changed underneath it (mirrors the use-snapshot in step 4b).
    SmallVector<triton::StoreOp, 16> storesToNarrow;
    func.walk([&](triton::StoreOp store) {
      if (!narrowedPtrs.count(store.getPtr()))
        return;
      Value storeValue = store.getValueMutable().get();
      // tt.store's value operand is non-optional in ODS, so a verified op
      // always has it; assert the invariant rather than branch on it.
      assert(storeValue && "tt.store must have a value operand");
      // The type guard is load-bearing, not redundant: a narrowed pointer does
      // not imply an i64 value. The same destination may store a non-i64 value
      // (e.g. an f32, as in triton_poi_fused__to_copy_unsqueeze_1), for which
      // i64ToI32Like is a no-op and trunci would be invalid. Only queue stores
      // whose value actually needs i64->i32 narrowing.
      Type valueTy = storeValue.getType();
      if (i64ToI32Like(valueTy) != valueTy)
        storesToNarrow.push_back(store);
    });
    for (triton::StoreOp store : storesToNarrow) {
      OpOperand &valueUse = store.getValueMutable();
      Type newValueTy = i64ToI32Like(valueUse.get().getType());
      insertTruncIBeforeStore(rewriter, valueUse, newValueTy);
      changed = true;
    }
  });

  return changed;
}

} // namespace hexagon_backend
