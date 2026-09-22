//===- HvxToLLVMPass.cpp - Lower hvx ops to LLVM ---------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// hvx.* -> llvm.call on llvm.hexagon.* intrinsics.
//
// These intrinsics are selected by our own LLVM (unlike the HMX
// `llvm.hexagon.M8.*` builtins, which only the SDK's clang lowers), so a plain
// `llvm.func` declaration plus `llvm.call` is the whole lowering. The
// intrinsic's register image is `<32 x i32>` (one 128-byte vector), so the
// op's lane-typed vector is bitcast in and out; an LLVM bitcast between
// equal-width vectors is a register rename.
//
// Runs as the last dialect-to-LLVM conversion (after HmxToLLVM), so everything
// it emits is already in its final form: LLVM dialect ops inside llvm.func,
// with `llvm.constant` for the rotate amount (arith-to-llvm has long since
// run). The rewrite is a flat per-op replacement -- no conversion framework,
// because no type changes hands and the only pattern is total.
//
//===----------------------------------------------------------------------===//

#include "hexagon/Conversion/HvxToLLVM/HvxToLLVM.h"
#include "hexagon/Dialect/Hvx/IR/HvxDialect.h"

#include "mlir/Dialect/LLVMIR/FunctionCallUtils.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace mlir::hvx;

// At global scope, like HmxToLLVM: the generated pass base lands in ::impl.
#define GEN_PASS_DEF_HVXTOLLVM
#include "hexagon/Conversion/HvxToLLVM/Passes.h.inc"

namespace {

/// The intrinsic's register image: one 128-byte HVX vector spelled as
/// `<32 x i32>` (Hexagon_v32i32_v32i32i32_Intrinsic in IntrinsicsHexagonDep.td).
static constexpr int kHvxRegLanes = 32;

static VectorType hvxRegImageType(MLIRContext *ctx) {
  return VectorType::get({kHvxRegLanes}, IntegerType::get(ctx, 32));
}

/// `llvm.hexagon.V6.vror.128B(<32 x i32>, i32) -> <32 x i32>`, declared once
/// per module like any intrinsic (the Hexagon backend selects it into V6_vror
/// under UseHVX128B, HexagonDepMapAsm2Intrin.td).
static FailureOr<LLVM::LLVMFuncOp>
getOrCreateVrorIntrinsic(ModuleOp module, OpBuilder &builder) {
  auto vecTy = hvxRegImageType(module.getContext());
  auto i32Ty = IntegerType::get(module.getContext(), 32);
  return LLVM::lookupOrCreateFn(
      builder, module, "llvm.hexagon.V6.vror.128B",
      /*paramTypes=*/ArrayRef<Type>{vecTy, i32Ty},
      /*resultType=*/vecTy);
}

/// Bitcast between the op's lane-typed vector and the intrinsic's register
/// image. Both are 128 bytes, so the bitcast is a register rename; a no-op
/// when the lane type is already i32.
static Value toRegImage(OpBuilder &builder, Location loc, Value v) {
  Type imageTy = hvxRegImageType(v.getContext());
  if (v.getType() == imageTy)
    return v;
  return LLVM::BitcastOp::create(builder, loc, imageTy, v);
}

static void lowerVror(VrorOp op, LLVM::LLVMFuncOp intrinsic) {
  Location loc = op.getLoc();
  OpBuilder builder(op);
  Value image = toRegImage(builder, loc, op.getInput());
  Value amount = LLVM::ConstantOp::create(
      builder, loc, builder.getI32Type(),
      builder.getI32IntegerAttr(op.getBytes()));
  auto call = LLVM::CallOp::create(
      builder, loc, TypeRange{intrinsic.getResultTypes()},
      FlatSymbolRefAttr::get(intrinsic), ValueRange{image, amount});
  Value result = call.getResult();
  if (result.getType() != op.getResult().getType())
    result = LLVM::BitcastOp::create(builder, loc, op.getResult().getType(),
                                     result);
  op.replaceAllUsesWith(result);
  op.erase();
}

struct HvxToLLVMPass : public ::impl::HvxToLLVMBase<HvxToLLVMPass> {
public:
  explicit HvxToLLVMPass() = default;

  void runOnOperation() override {
    auto moduleOp = getOperation();
    SmallVector<VrorOp> ops;
    moduleOp.walk([&](VrorOp op) { ops.push_back(op); });
    if (ops.empty())
      return;

    OpBuilder builder(moduleOp.getContext());
    FailureOr<LLVM::LLVMFuncOp> intrinsic =
        getOrCreateVrorIntrinsic(moduleOp, builder);
    if (failed(intrinsic)) {
      signalPassFailure();
      return;
    }
    for (VrorOp op : ops)
      lowerVror(op, *intrinsic);
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>> mlir::hvx::createHvxToLLVMPass() {
  return std::make_unique<HvxToLLVMPass>();
}
