//===- InjectHMXVtcmBaseArgPass.cpp - Inject HMX VTCM-base argument -------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// Injects a trailing memref<?xi8, 1> VTCM-base argument into every func.func.
// The argument carries the hexkl_hmx.vtcm_base_arg function attribute so
// downstream passes (e.g. DecomposeHexKLMatmulToDaemonPass) can locate it
// without re-scanning the argument list.
//
// In the HMX micro-daemon model the wrapper acquires one shared VTCM region
// and forwards its un-sliced base through this argument.  The decompose pass
// then derives each worker's slice as `threadId * vtcmPerInstanceBytes`, so the
// base passed here must be the global (un-sliced) VTCM base.  The element type
// and address space (i8, space 1) mirror the alloc type in
// DecomposeHexKLMatmulPass so the two paths agree on the VTCM memref type.
//
// Pass-order convention: this pass runs AFTER InjectHMXThreadIdArgPass and
// BEFORE InjectHMXQueueArgPass, so the resulting trailing args are
// `..., %tid: i32, %vtcm_base: memref<?xi8, 1>, %queue: !llvm.ptr`.  All three
// inject passes are idempotent.
//
//===----------------------------------------------------------------------===//

#include "hexagon/Transforms/Transforms.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace mlir {
namespace hexagon {
#define GEN_PASS_DEF_INJECTHMXVTCMBASEARG
#include "hexagon/Transforms/Passes.h.inc"
} // namespace hexagon
} // namespace mlir

namespace {

struct InjectHMXVtcmBaseArgPass
    : public hexagon::impl::InjectHMXVtcmBaseArgBase<InjectHMXVtcmBaseArgPass> {
  using hexagon::impl::InjectHMXVtcmBaseArgBase<
      InjectHMXVtcmBaseArgPass>::InjectHMXVtcmBaseArgBase;

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();

    // Idempotent: skip if already injected.
    if (funcOp->hasAttr("hexkl_hmx.vtcm_base_arg"))
      return;

    MLIRContext *ctx = funcOp.getContext();
    Builder builder(ctx);

    // memref<?xi8, 1>: dynamic-length i8 buffer in VTCM (address space 1),
    // matching the alloc type used by DecomposeHexKLMatmulPass.
    auto vtcmType = MemRefType::get({ShapedType::kDynamic}, builder.getI8Type(),
                                    MemRefLayoutAttrInterface{},
                                    builder.getI32IntegerAttr(1));
    (void)funcOp.insertArgument(funcOp.getNumArguments(), vtcmType,
                                DictionaryAttr::get(ctx), funcOp.getLoc());
    funcOp->setAttr("hexkl_hmx.vtcm_base_arg", UnitAttr::get(ctx));
  }
};

} // namespace

namespace mlir {
namespace hexagon {

std::unique_ptr<Pass> createInjectHMXVtcmBaseArgPass() {
  return std::make_unique<InjectHMXVtcmBaseArgPass>();
}

} // namespace hexagon
} // namespace mlir
