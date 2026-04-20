//===- EliminateRedundantUnpackPackPass.cpp - Eliminate unpack/pack pairs ===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// This pass eliminates redundant unpack/pack pairs when:
// 1. An unpack operation's result is consumed by a pack operation
// 2. Both operations have the same inner_dims_pos and inner_tiles
// 3. Both operations have the same outer_dims_perm
// 4. The tensor type of unpack's source matches pack's destination type
//
// Pattern:
//   %unpacked = linalg.unpack %packed inner_dims_pos = [...] inner_tiles =
//   	[...]
//   %repacked = linalg.pack %unpacked [padding_value(%pad)]
//   	inner_dims_pos = [...] inner_tiles = [...]
//
// The tensor type check (unpack's source type == pack's dest type) is a
// NECESSARY but NOT SUFFICIENT condition for correctness. It ensures structural
// compatibility but does NOT guarantee that the padding values added by pack
// match the values discarded by unpack.
//
// IMPORTANT LIMITATION:
// This pass currently does NOT verify that the padding value used by pack
// matches the values being discarded by unpack. This verification requires
// data flow analysis to prove that the padded values discarded by unpack
// are the same as the pack's padding value.
//
// Users of this pass must ensure this property holds in their use case, or
// accept that the optimization may be unsound in general. Future improvements
// will add data flow analysis to automatically verify this property.
//
// After replacing pack with unpack's source, the pass checks if the unpack
// operation becomes dead (has no users) and eliminates it immediately.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace {

using namespace mlir;

// Helper to check if two arrays of OpFoldResult are equal
static bool areOpFoldResultsEqual(ArrayRef<OpFoldResult> a,
                                  ArrayRef<OpFoldResult> b) {
  if (a.size() != b.size())
    return false;

  for (size_t i = 0; i < a.size(); ++i) {
    // Both are attributes
    if (isa<Attribute>(a[i]) && isa<Attribute>(b[i])) {
      if (cast<Attribute>(a[i]) != cast<Attribute>(b[i]))
        return false;
    }
    // Both are values
    else if (isa<Value>(a[i]) && isa<Value>(b[i])) {
      if (cast<Value>(a[i]) != cast<Value>(b[i]))
        return false;
    }
    // One is attribute, other is value - not equal
    else {
      return false;
    }
  }

  return true;
}

// Pattern to eliminate redundant unpack/pack pairs
struct EliminateRedundantUnpackPack : public OpRewritePattern<linalg::PackOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::PackOp packOp,
                                PatternRewriter &rewriter) const override {
    // Check if the input to pack is an unpack operation
    auto unpackOp = packOp.getSource().getDefiningOp<linalg::UnPackOp>();
    if (!unpackOp)
      return failure();

    // Check if inner_dims_pos match
    if (packOp.getInnerDimsPos() != unpackOp.getInnerDimsPos())
      return failure();

    // Check if inner_tiles match
    if (!areOpFoldResultsEqual(packOp.getMixedTiles(),
                               unpackOp.getMixedTiles()))
      return failure();

    // Check if outer_dims_perm match (if present)
    auto packOuterPerm = packOp.getOuterDimsPerm();
    auto unpackOuterPerm = unpackOp.getOuterDimsPerm();
    if (packOuterPerm != unpackOuterPerm)
      return failure();

    // CRITICAL CHECK: Verify that unpack's source type matches pack's dest type
    // This is a NECESSARY but NOT SUFFICIENT condition for correctness.
    // It ensures structural compatibility, but does NOT guarantee that the
    // padding values added by pack match the values discarded by unpack.
    // That verification requires data flow analysis (future work).
    auto unpackSourceType = unpackOp.getSource().getType();
    auto packDestType = packOp.getDest().getType();
    if (unpackSourceType != packDestType)
      return failure();

    // At this point, we have a matching unpack/pack pair with compatible types
    // Replace the pack operation with the source of the unpack
    rewriter.replaceOp(packOp, unpackOp.getSource());

    // Check if the unpack operation is now dead (has no users)
    // If so, eliminate it immediately
    if (unpackOp->use_empty()) {
      rewriter.eraseOp(unpackOp);
    }

    return success();
  }
};

// Pass definition
struct EliminateRedundantUnpackPackPass
    : public PassWrapper<EliminateRedundantUnpackPackPass, OperationPass<>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(EliminateRedundantUnpackPackPass)

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<EliminateRedundantUnpackPack>(&getContext());

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      return signalPassFailure();
    }
  }

  StringRef getArgument() const final {
    return "eliminate-redundant-unpack-pack";
  }

  StringRef getDescription() const final {
    return "Eliminate redundant unpack/pack pairs with zero padding";
  }
};

} // namespace

namespace mlir {
namespace hexagon {

std::unique_ptr<Pass> createEliminateRedundantUnpackPackPass() {
  return std::make_unique<EliminateRedundantUnpackPackPass>();
}

} // namespace hexagon
} // namespace mlir
