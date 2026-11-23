//===- InjectHMXQueueArgPass.cpp - Inject HMX queue argument --------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// Injects a trailing !llvm.ptr queue argument into every func.func.  The
// argument carries the hmx.queue_arg function attribute so downstream
// passes (e.g. DecomposeHexKLMatmulToDaemonPass) can locate it without
// re-scanning the argument list.
//
// The wrapper allocates one HMXQueue per kernel invocation, initialises it,
// and forwards the pointer as the kernel function's last argument.  In the
// HMX-daemon model this is the channel through which workers (threads 0-3)
// submit matmul jobs to thread 4.
//
// Pass order convention: this pass runs AFTER InjectHMXThreadIdArgPass, so
// the resulting argument layout is `..., %tid: i32, %queue: !llvm.ptr` —
// queue is the new last arg, threadId is second-to-last.  Both passes are
// idempotent.
//
//===----------------------------------------------------------------------===//

#include "hexagon/Transforms/Transforms.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace mlir {
namespace hexagon {
#define GEN_PASS_DEF_INJECTHMXQUEUEARG
#include "hexagon/Transforms/Passes.h.inc"
} // namespace hexagon
} // namespace mlir

namespace {

struct InjectHMXQueueArgPass
    : public hexagon::impl::InjectHMXQueueArgBase<InjectHMXQueueArgPass> {
  using hexagon::impl::InjectHMXQueueArgBase<
      InjectHMXQueueArgPass>::InjectHMXQueueArgBase;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<LLVM::LLVMDialect>();
  }

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();

    // Idempotent: skip if already injected.
    if (funcOp->hasAttr("hmx.queue_arg"))
      return;

    auto ptrTy = LLVM::LLVMPointerType::get(funcOp.getContext());
    (void)funcOp.insertArgument(funcOp.getNumArguments(), ptrTy,
                                DictionaryAttr::get(funcOp.getContext()),
                                funcOp.getLoc());
    funcOp->setAttr("hmx.queue_arg", UnitAttr::get(funcOp.getContext()));
  }
};

} // namespace

namespace mlir {
namespace hexagon {

std::unique_ptr<Pass> createInjectHMXQueueArgPass() {
  return std::make_unique<InjectHMXQueueArgPass>();
}

} // namespace hexagon
} // namespace mlir
