//===-- HmxWorkspaceResidentPass.cpp - per-launch VTCM workspace, resident -===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// Every HMX kernel pays the same per-launch VTCM prologue: one allocation and
// one deallocation per workspace buffer (the crouton arrays of `matmul-to-hmx`,
// the partition pass's conversion state, activation-staging ring slots,
// statuses and the crouton-row scratch). The runtime's allocator does a
// best-fit scan, a split/coalesce and the buffer-manager bookkeeping for each,
// and the cost is size-independent -- measured at ~6.3 us per alloc/free pair,
// so on a small shape (S3) it is more than half the launch.
//
// A workspace has no compile-time image: the kernel refills it on every launch.
// So all it needs is a stable buffer, which is exactly the residency mechanism
// the weight path already uses, minus the copy. This pass gives each per-launch
// VTCM workspace allocation a compile-time key and tags it
// `hmx.workspace_resident`; its lowering emits one call to
// `hexagon_runtime_workspace_resident(key, bytes)`, which allocates the buffer
// on the first launch, pins it against the per-launch deallocation and returns
// the same address forever after. The allocation and deallocation calls
// disappear, leaving only the call and the lookup.
//
// The key is `(hash(function symbol) << 32) | index` with the top bit set: it
// is stable across launches of one kernel, unique per (function, buffer), and
// cannot collide with the address keys the weight residency uses. It is not
// stable across recompiles, which is fine -- the runtime's residency map lives
// in one process running one compiled kernel.
//
// Correctness boundary: a resident buffer is shared by every launch in the
// process. Sequential launches are fine because each one overwrites the whole
// buffer before reading it (the buffers are all full-overwrite workspaces). A
// grid>1 launch, however, runs the same kernel on several threads in one
// process, and those instances would clobber each other's workspace. The pass
// therefore only runs when the caller opts in (`enableWorkspaceResident`), and
// that option promises single-instance execution. This is the same
// process-boundary contract the weight residency documents; a per-instance key
// would be the alternative and is deliberately not guessed here.
//
// Runs after `hmx-partition` (so the ring slots/statuses and the scratch exist)
// and before `convert-to-hexagonmem` (which carries the tag onto the
// `hexagonmem.alloc` the lowering reads).
//
//===----------------------------------------------------------------------===//

#include "hexagon/Common/Common.h"
#include "hexagon/Dialect/Hmx/IR/HmxDialect.h"
#include "hexagon/Dialect/Hmx/Transforms/Transforms.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include <cstdint>

#define DEBUG_TYPE "hmx-workspace-resident"

using namespace mlir;
using namespace mlir::hmx;

namespace mlir {
namespace hmx {
#define GEN_PASS_DEF_HMXWORKSPACERESIDENT
#include "hexagon/Dialect/Hmx/Transforms/Passes.h.inc"
} // namespace hmx
} // namespace mlir

namespace {

/// Per-buffer residency record on the allocation. `key` is the runtime map key,
/// `bytes` the buffer size; the lowering reads both to build the resident call.
constexpr const char *kResidentAttr = "hmx.workspace_resident";
constexpr const char *kResidentKeyAttr = "key";
constexpr const char *kResidentBytesAttr = "bytes";

/// Namespaces a workspace key away from the weight keys. The weight path keys
/// on a 32-bit process address, so setting the top bit makes a collision
/// impossible without needing a shared allocator for the two.
constexpr uint64_t kWorkspaceKeyTag = 1ull << 63;

struct HmxWorkspaceResidentPass
    : public mlir::hmx::impl::HmxWorkspaceResidentBase<
          HmxWorkspaceResidentPass> {
  using HmxWorkspaceResidentBase::HmxWorkspaceResidentBase;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<memref::MemRefDialect>();
  }

  void runOnOperation() override {
    func::FuncOp func = cast<func::FuncOp>(getOperation());

    // Only HMX kernels have the per-launch workspace this pass is about. A
    // function the HMX path did not touch keeps its allocations exactly as
    // they were.
    bool hasHmx = false;
    func.walk([&](Operation *op) {
      Dialect *dialect = op->getDialect();
      if (dialect && dialect->getNamespace() == HmxDialect::getDialectNamespace())
        hasHmx = true;
    });
    if (!hasHmx)
      return;

    SmallVector<memref::AllocOp> workspaces;
    func.walk([&](memref::AllocOp alloc) {
      // The weight path already pinned its buffers; leave them to it.
      if (alloc->hasAttr("hmx.weight_resident"))
        return;
      auto type = dyn_cast<MemRefType>(alloc.getType());
      if (!type || !type.hasStaticShape() ||
          type.getMemorySpaceAsInt() != hexagon::VTCM_ADDRESS_SPACE)
        return;
      Type element = type.getElementType();
      // Sub-byte elements (i1 masks) have no byte size this pass could declare.
      if (!element.isIntOrFloat() || element.getIntOrFloatBitWidth() % 8 != 0 ||
          type.getNumElements() == 0)
        return;
      workspaces.push_back(alloc);
    });
    if (workspaces.empty())
      return;

    // Key by the function symbol and the buffer's order within it, never by an
    // address: the key has to be the same on every launch of the same kernel.
    uint64_t hash = llvm::hash_value(func.getSymName());

    MLIRContext *context = func.getContext();
    auto i64 = IntegerType::get(context, 64);
    for (auto [index, alloc] : llvm::enumerate(workspaces)) {
      auto type = cast<MemRefType>(alloc.getType());
      int64_t bytes = type.getNumElements() * (type.getElementTypeBitWidth() / 8);
      uint64_t key = kWorkspaceKeyTag | (hash << 32) |
                     (static_cast<uint64_t>(index) & 0xFFFFFFFFull);
      alloc->setAttr(
          kResidentAttr,
          DictionaryAttr::get(
              context,
              {NamedAttribute(StringAttr::get(context, kResidentKeyAttr),
                              IntegerAttr::get(i64, key)),
               NamedAttribute(StringAttr::get(context, kResidentBytesAttr),
                              IntegerAttr::get(i64, bytes))}));

      // The pinned buffer is released by the process, not by the launch, so its
      // per-launch deallocation is dropped. Any other user of the buffer stays.
      SmallVector<memref::DeallocOp> deallocs;
      for (Operation *user : alloc->getUsers())
        if (auto dealloc = dyn_cast<memref::DeallocOp>(user))
          deallocs.push_back(dealloc);
      for (memref::DeallocOp dealloc : deallocs)
        dealloc.erase();

      LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE "] resident workspace #" << index
                              << " (" << bytes << " bytes, key 0x"
                              << llvm::Twine::utohexstr(key) << ")\n");
    }
  }
};

} // namespace

std::unique_ptr<InterfacePass<FunctionOpInterface>>
mlir::hmx::createHmxWorkspaceResidentPass() {
  return std::make_unique<HmxWorkspaceResidentPass>();
}
