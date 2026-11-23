//===- InjectHMXThreadIdArgPass.cpp - Inject HMX thread-ID argument -------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// Injects a trailing i32 thread-ID argument into every func.func.  The
// argument carries the hmx.thread_id_arg function attribute so that
// downstream passes (e.g. DecomposeHexKLMatmulToDaemonPass) can locate it
// without re-scanning the argument list.
//
// The wrapper that calls the kernel function passes entryConfig->threadID for
// each hardware thread, giving every thread its own real TID.
//
// The pass is idempotent: if hmx.thread_id_arg is already present on the
// function, it returns immediately without inserting a second argument.
//
//===----------------------------------------------------------------------===//

#include "hexagon/Transforms/Transforms.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace mlir {
namespace hexagon {
#define GEN_PASS_DEF_INJECTHMXTHREADIDARG
#include "hexagon/Transforms/Passes.h.inc"
} // namespace hexagon
} // namespace mlir

namespace {

struct InjectHMXThreadIdArgPass
    : public hexagon::impl::InjectHMXThreadIdArgBase<InjectHMXThreadIdArgPass> {
  using hexagon::impl::InjectHMXThreadIdArgBase<
      InjectHMXThreadIdArgPass>::InjectHMXThreadIdArgBase;

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();

    // Idempotent: skip if already injected.
    if (funcOp->hasAttr("hmx.thread_id_arg"))
      return;

    (void)funcOp.insertArgument(
        funcOp.getNumArguments(), IntegerType::get(funcOp.getContext(), 32),
        DictionaryAttr::get(funcOp.getContext()), funcOp.getLoc());
    funcOp->setAttr("hmx.thread_id_arg",
                    UnitAttr::get(funcOp.getContext()));
  }
};

} // namespace

namespace mlir {
namespace hexagon {

std::unique_ptr<Pass> createInjectHMXThreadIdArgPass() {
  return std::make_unique<InjectHMXThreadIdArgPass>();
}

} // namespace hexagon
} // namespace mlir
