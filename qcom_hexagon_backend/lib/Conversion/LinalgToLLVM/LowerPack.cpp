//===-- LowerPack.cpp -  wrapper pass for lowerpack           -------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// This pass is needed as pack/unpack are not handled by bufferizer.
//
//===----------------------------------------------------------------------===//

#include "hexagon/Conversion/LinalgToLLVM/LinalgToLLVM.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "-lower-pack"

#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define DBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
using namespace mlir::linalg;

#define GEN_PASS_DEF_LOWERPACK
#include "hexagon/Conversion/LinalgToLLVM/Passes.h.inc"

namespace {

struct LowerPackPass : public ::impl::LowerPackBase<LowerPackPass> {
  void runOnOperation() override {
    auto funcOp = getOperation();
    funcOp.walk([&](linalg::PackOp op) {
      IRRewriter rewriter(op.getContext());
      FailureOr<LowerPackResult> res = lowerPack(rewriter, op);
      assert(!failed(res));
    });
    funcOp.walk([&](linalg::UnPackOp op) {
      IRRewriter rewriter(op.getContext());
      FailureOr<LowerUnPackOpResult> res = lowerUnPack(rewriter, op);
      assert(!failed(res));
    });
  }
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>> hexagon::createLowerPackPass() {
  return std::make_unique<LowerPackPass>();
}
