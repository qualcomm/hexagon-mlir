//===- InsertScratchArgPass.cpp - Inject VTCM scratch argument ------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// Injects a per-instance VTCM scratch buffer argument into the kernel function
// signature. The argument is typed as memref<Nxi8, 1> {hexagon.scratch} where
// N is the per-instance budget (bytes) provided via the `scratch` pass option.
//
// The arg is inserted before the trailing 6 Triton program-info args:
//   [num_programs_x, num_programs_y, num_programs_z, pid_x, pid_y, pid_z]
//
// The wrapper allocates scratch * prod(grid) bytes total and passes each
// instance a non-overlapping slice of size `scratch` bytes.
//
//===----------------------------------------------------------------------===//

#include "hexagon/Transforms/Transforms.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"

using namespace mlir;
using namespace hexagon;

#define GEN_PASS_DEF_INSERTSCRATCHARG
#include "hexagon/Transforms/Passes.h.inc"

namespace {

/// VTCM alignment requirement in bytes.
static constexpr int64_t kVTCMAlignment = 2048;

/// Returns true if the last 6 args of `func` are all i32.
static bool hasTrailingProgramInfoPack(func::FuncOp func) {
  if (func.getNumArguments() < 6)
    return false;
  unsigned start = func.getNumArguments() - 6;
  for (unsigned i = start; i < func.getNumArguments(); ++i) {
    auto ty = dyn_cast<IntegerType>(func.getArgument(i).getType());
    if (!ty || ty.getWidth() != 32)
      return false;
  }
  return true;
}

struct InsertScratchArgPass
    : ::impl::InsertScratchArgBase<InsertScratchArgPass> {
  using Base::Base;

  LogicalResult initialize(MLIRContext *ctx) override {
    if (scratch <= 0) {
      llvm::errs() << "InsertScratchArg: scratch must be > 0, got " << scratch
                   << "\n";
      return failure();
    }
    if (scratch % kVTCMAlignment != 0) {
      llvm::errs() << "InsertScratchArg: scratch must be aligned to "
                   << kVTCMAlignment << " bytes, got " << scratch << "\n";
      return failure();
    }
    return success();
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();

    // Idempotent: skip if hexagon.scratch arg already present.
    if (llvm::any_of(func.getArguments(), [&](BlockArgument arg) {
          return func.getArgAttr(arg.getArgNumber(), "hexagon.scratch");
        }))
      return;

    // Only inject into kernel functions with the Triton program-info ABI.
    if (!hasTrailingProgramInfoPack(func))
      return;

    MLIRContext *ctx = func.getContext();

    // Static-sized memref<Nxi8, 1> — N is the per-instance budget.
    // Static type enables compile-time validation in MemoryOffsetsPass.
    auto scratchType =
        MemRefType::get({scratch}, IntegerType::get(ctx, 8), {}, 1);
    auto scratchAttrs = DictionaryAttr::get(
        ctx, {NamedAttribute(StringAttr::get(ctx, "hexagon.scratch"),
                             UnitAttr::get(ctx))});

    unsigned insertIdx = func.getNumArguments() - 6;
    (void)func.insertArgument(insertIdx, scratchType, scratchAttrs,
                              func.getLoc());
  }
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
hexagon::createInsertScratchArgPass(const InsertScratchArgOptions &options) {
  return std::make_unique<InsertScratchArgPass>(options);
}
