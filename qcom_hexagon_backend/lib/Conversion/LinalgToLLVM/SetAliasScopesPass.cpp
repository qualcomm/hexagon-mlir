//===- SetAliasScopesPass.cpp - Emit alias-scope metadata on LLVM ops ----===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// For each llvm.load/llvm.store, walk backward through the lowering chain to
// find a root buffer, then stamp per-access alias metadata:
//   alias_scopes   = [scope(root)]
//   noalias_scopes = [scopes(all other roots)]
// A root is the terminal provenance value the walk lands on: either a
// function entry-block argument, or a fresh local allocation
// (memref.alloca / memref.alloc / hexagonmem.alloc).
//
// The walk follows only provenance-preserving hops:
//   llvm.getelementptr, llvm.extractvalue, single-operand
//   builtin.unrealized_conversion_cast, memref.reinterpret_cast,
//   memref.extract_strided_metadata.
//
// Anything else is left untraced (including multi-operand unrealized casts,
// non-entry block arguments, unknown ops, or chains beyond kMaxSteps). Untraced
// ops simply keep no alias metadata and therefore stay conservatively
// may-alias.
//
// If fewer than two distinct roots trace successfully in a function, the
// pass attaches no metadata to anything in that function — disjointness is
// only a meaningful claim once there are at least two buffers to compare.
//
// Existing alias_scopes/noalias_scopes already present on an op are merged
// with, not replaced by, the scopes this pass computes (existing entries
// first, new unique entries appended). Root/scope collection order follows
// the order ops are encountered while walking the function, so the emitted
// alias_scopes/noalias_scopes array ordering is deterministic.
//
// A local-allocation root's noalias_scopes list omits other local-allocation
// roots: LLVM's BasicAA already proves two fresh allocations disjoint once
// canonicalize (later in the pipeline) folds away their descriptor-struct
// wrapping, so stamping that claim here would be redundant. Argument-vs-
// argument and argument-vs-local claims are unaffected — BasicAA can't
// resolve those on its own.
//
// This pass is intended to be gated in the main linalg-to-llvm pipeline by the
// `enableSetAliasScopes` option. Enabling it assumes the compilation flow
// provides the required no-alias contract for distinct function arguments.
//
//===----------------------------------------------------------------------===//

#include <algorithm>

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/MapVector.h"
#include "llvm/Support/Debug.h"

#include "hexagon/Dialect/HexagonMem/IR/HexagonMemDialect.h"

#define GEN_PASS_DEF_SETALIASSCOPES
#include "hexagon/Transforms/Transforms.h"

#define DEBUG_TYPE "set-alias-scopes"

#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define DBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;

namespace {

//===----------------------------------------------------------------------===//
// Root tracing
//===----------------------------------------------------------------------===//

// Walk backward from a pointer value to its accepted root, or return null.
static Value traceToRoot(Value ptr) {
  Value current = ptr;
  // Stops the walk from looping forever, not a real chain-length limit.
  constexpr unsigned kMaxSteps = 20;

  for (unsigned steps = 0; steps < kMaxSteps; ++steps) {
    Operation *defOp = current.getDefiningOp();

    if (!defOp) {
      // Only function entry-block arguments are accepted roots.
      if (auto blockArg = dyn_cast<BlockArgument>(current)) {
        Block *block = blockArg.getOwner();
        if (block->isEntryBlock() && isa<func::FuncOp>(block->getParentOp()))
          return current;
      }
      return nullptr;
    }

    if (auto gep = dyn_cast<LLVM::GEPOp>(defOp)) {
      current = gep.getBase();
      continue;
    }

    if (auto ev = dyn_cast<LLVM::ExtractValueOp>(defOp)) {
      current = ev.getContainer();
      continue;
    }

    if (auto cast = dyn_cast<UnrealizedConversionCastOp>(defOp)) {
      // Single-operand cast: provenance is unambiguous.
      // Multi-operand cast: no stable provenance rule, so do not guess.
      if (cast.getNumOperands() == 1) {
        current = cast.getOperand(0);
        continue;
      }
    }

    if (auto rc = dyn_cast<memref::ReinterpretCastOp>(defOp)) {
      current = rc.getSource();
      continue;
    }

    if (auto esm = dyn_cast<memref::ExtractStridedMetadataOp>(defOp)) {
      current = esm.getSource();
      continue;
    }

    // Fresh local allocation is a valid root.
    if (isa<memref::AllocaOp, memref::AllocOp, hexagonmem::AllocOp>(defOp))
      return current;

    return nullptr;
  }
  return nullptr;
}

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

// Return the pointer operand for supported memory ops.
static Value getPointerOperand(Operation *op) {
  if (auto load = dyn_cast<LLVM::LoadOp>(op))
    return load.getAddr();
  if (auto store = dyn_cast<LLVM::StoreOp>(op))
    return store.getAddr();
  return nullptr;
}

// Merge scope arrays as: existing first, then new unique entries.
static ArrayAttr mergeScopeArrays(OpBuilder &builder, ArrayAttr existing,
                                  ArrayAttr incoming) {
  if (!existing)
    return incoming;
  if (!incoming)
    return existing;

  SmallVector<Attribute> merged(existing.begin(), existing.end());
  for (Attribute scope : incoming) {
    if (std::find(merged.begin(), merged.end(), scope) == merged.end())
      merged.push_back(scope);
  }
  return builder.getArrayAttr(merged);
}

//===----------------------------------------------------------------------===//
// Pass implementation
//===----------------------------------------------------------------------===//

struct SetAliasScopesPass
    : public hexagon::impl::SetAliasScopesBase<SetAliasScopesPass> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SetAliasScopesPass)

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();

    // Step 1: For each root, collect every load/store whose pointer traces
    // back to it.
    llvm::MapVector<Value, SmallVector<Operation *>> rootToOps;
    unsigned unknownCount = 0;

    funcOp.walk([&](Operation *op) {
      if (!isa<LLVM::LoadOp, LLVM::StoreOp>(op))
        return;
      Value ptr = getPointerOperand(op);
      Value root = traceToRoot(ptr);
      if (root) {
        rootToOps[root].push_back(op);
      } else {
        ++unknownCount;
        DBG("Untraced pointer operand for " << *op);
      }
    });

    if (unknownCount > 0)
      DBG(unknownCount
          << " memory op(s) could not be traced to a root buffer and were "
             "left without alias-scope metadata");

    if (rootToOps.size() < 2) {
      DBG(funcOp.getSymName()
          << ": found " << rootToOps.size() << " traceable root(s) ("
          << unknownCount << " untraced) — need at least 2, skipping");
      return;
    }

    // Step 2: One domain for this function, then one named scope per root
    // — "argN" for argument roots, "local" for allocations.
    OpBuilder builder(funcOp.getContext());
    auto domainAttr = LLVM::AliasScopeDomainAttr::get(
        funcOp.getContext(), DistinctAttr::create(builder.getUnitAttr()),
        builder.getStringAttr(funcOp.getSymName()));

    llvm::MapVector<Value, LLVM::AliasScopeAttr> rootToScope;
    for (auto &[root, ops] : rootToOps) {
      std::string name;
      if (auto blockArg = dyn_cast<BlockArgument>(root))
        name = "arg" + std::to_string(blockArg.getArgNumber());
      else
        name = "local";

      auto scopeAttr = LLVM::AliasScopeAttr::get(
          funcOp.getContext(), DistinctAttr::create(builder.getUnitAttr()),
          domainAttr, builder.getStringAttr(name));
      rootToScope[root] = scopeAttr;
      DBG("Created scope \"" << name << "\" for root " << root << " ("
                             << ops.size() << " op(s))");
    }

    // Step 3: Build per-root alias/noalias lists and merge onto each op.
    for (auto &[root, ops] : rootToOps) {
      LLVM::AliasScopeAttr myScope = rootToScope[root];
      bool rootIsLocal = !isa<BlockArgument>(root);

      // Disjointness is inferred by root identity: allocation roots are fresh;
      // distinct argument roots are treated as disjoint under the caller
      // contract assumed when enableSetAliasScopes is enabled.
      SmallVector<Attribute> noaliasScopes;
      for (auto &[otherRoot, otherScope] : rootToScope) {
        // Skip the root itself.
        if (otherRoot == root)
          continue;
        // Skip other local roots when the current root is local.
        if (rootIsLocal && !isa<BlockArgument>(otherRoot))
          continue;
        noaliasScopes.push_back(otherScope);
      }

      auto aliasScopesAttr = builder.getArrayAttr({myScope});
      auto noaliasScopesAttr = builder.getArrayAttr(noaliasScopes);

      for (Operation *op : ops) {
        // op is always a load/store; this reaches setAliasScopes through the
        // interface.
        if (auto iface = dyn_cast<LLVM::AliasAnalysisOpInterface>(op)) {
          ArrayAttr mergedAliasScopes = mergeScopeArrays(
              builder, iface.getAliasScopesOrNull(), aliasScopesAttr);
          ArrayAttr mergedNoAliasScopes = mergeScopeArrays(
              builder, iface.getNoAliasScopesOrNull(), noaliasScopesAttr);
          iface.setAliasScopes(mergedAliasScopes);
          iface.setNoAliasScopes(mergedNoAliasScopes);
        }
      }
    }

    DBG(funcOp.getSymName() << ": stamped " << rootToScope.size() << " scopes, "
                            << unknownCount << " untraced ops");
  }
};
} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
hexagon::createSetAliasScopesPass() {
  return std::make_unique<SetAliasScopesPass>();
}
