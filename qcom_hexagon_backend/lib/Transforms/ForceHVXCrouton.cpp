//===- ForceHVXCrouton.cpp - Enable standalone HVX croutonization ---------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// Croutoniation (HVX)
// Core idea:
//  For eligible tensors, we materialize:
//    tensorIn --pack --> croutonPacked --unpack--> tensorOut
//
// Assumptions:
//  - Operates on ranked-tensor values, which are FP16 and have a rank equals 4.
//  - Tensors must have a static shape.
//  - Consumer ops targeted are "parallel" linalg.generic ops.
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/Debug.h"

#include "hexagon/Conversion/LinalgToLLVM/Common.h"
#include "hexagon/Conversion/LinalgToLLVM/LinalgToLLVM.h"
#include "hexagon/Transforms/PackUnpackUtils.h"
#include "hexagon/Transforms/Transforms.h"

#define DEBUG_TYPE "force-hvx-crouton"

#define DBGS() (llvm::errs() << '[' << DEBUG_TYPE << "] ")
#define DBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
using namespace hexagon;

#define GEN_PASS_DEF_FORCEHVXCROUTON
#include "hexagon/Transforms/Passes.h.inc"

namespace {

struct PackUnpackResult {
  linalg::PackOp packOp;
  Value originalTensor;
  Value unpackedValue;
};

/// Helper function: Given an argument with NHWC fp16 type,
/// emit the crouton pack operation.
static PackUnpackResult
createCroutonPackUnpack(Value tensorIn, RewriterBase &rewriter, Location loc) {
  auto rtt = dyn_cast<RankedTensorType>(tensorIn.getType());
  if (!rtt || rtt.getRank() != 4 || !rtt.getElementType().isF16() ||
      !rtt.hasStaticShape())
    return {};
  auto toTensor = tensorIn.getDefiningOp<bufferization::ToTensorOp>();
  if (!toTensor)
    return {};

  rewriter.setInsertionPointAfter(toTensor);
  auto argType = rtt;
  auto originalShape = argType.getShape();
  // Something to think about: Should we insert an attribute
  // for the arguments we are handling, such as "force_crouton"?
  // Then we can impose finer-grain control by exit early by
  // if (!arg->hasAttr("force_crouton")) ...
  auto padValue =
      arith::ConstantOp::create(rewriter, loc, rewriter.getF16FloatAttr(0.0f));

  // Setup pack parameters.
  SmallVector<int64_t> innerDimsPosTemp = {1, 2, 3}; // H, W, C
  SmallVector<int64_t> innerTilesTempVals = {8, 4, 32};
  SmallVector<OpFoldResult> innerTilesTemp =
      mlir::getAsIndexOpFoldResult(rewriter.getContext(), innerTilesTempVals);

  SmallVector<int64_t> innerDimsPos = {5}; // C packed dim
  SmallVector<int64_t> innerTilesVals = {2};
  SmallVector<OpFoldResult> innerTiles =
      mlir::getAsIndexOpFoldResult(rewriter.getContext(), innerTilesVals);

  auto packOp = buildF16PackOps(rewriter, tensorIn, argType, loc, padValue,
                                innerDimsPosTemp, innerTilesTemp, innerDimsPos,
                                innerTiles);

  Value packed = packOp.getResult();
  // pack to unpack
  auto unpackedOp =
      unpackF16Output(rewriter, packed, loc, innerDimsPos, innerTiles,
                      innerDimsPosTemp, innerTilesTemp, argType, originalShape);

  return PackUnpackResult{/*packOp=*/packOp, /*originalTensor=*/tensorIn,
                          /*unpackedValue=*/unpackedOp.getResult()};
}

/// We visit all linalg.generic ops in the function.
/// For each generic op, we examine its DPS inputs (tensor operands).
/// And for each operand, we attempt to materialize a crouton layout
/// round-trip.
/// This pass does not rewrite block args but targets concrete
/// tensor consumers to the linalg.generic op.
/// We use an InsertionGuard and set the insertion point at the current
/// generic op, which enables the new pack/unpack ops to be inserted in a
/// controlled local position.
struct ForceHVXCroutonPass
    : public ::impl::ForceHVXCroutonBase<ForceHVXCroutonPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect>();
  }
  void runOnOperation() override {
    auto funcOp = getOperation();
    IRRewriter rewriter(funcOp.getContext());
    Location loc = funcOp.getLoc();

    funcOp.walk([&](linalg::GenericOp generic) {
      mlir::IRRewriter::InsertionGuard g(rewriter);
      rewriter.setInsertionPoint(generic);

      for (OpOperand *in : generic.getDpsInputOperands()) {
        Value tensorIn = in->get();
        auto res = createCroutonPackUnpack(tensorIn, rewriter, loc);
        if (!res.packOp)
          continue;

        // Replace just this operand
        in->set(res.unpackedValue);
      }
    });
  }
};

} // namespace

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
hexagon::createForceHVXCroutonPass() {
  return std::make_unique<ForceHVXCroutonPass>();
}
