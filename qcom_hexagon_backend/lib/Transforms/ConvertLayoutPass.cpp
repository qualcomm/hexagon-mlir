//===-- ConvertLayoutPass.cpp - convert hexagonmem layout to generic ------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// This pass will rewrite:
//    #map = affine_map<(d0, d1) -> (d0 floordiv 4, d1 floordiv 4, d0 mod 4, d1
//    mod 4)> %modified_arg1 = memref.alloc() : memref<64x62xi8, #map>
//    hexagonmem.convert_layout
//                       ins(%arg1 : memref<64x62xi8>),
//                       outs(%modified_arg1 : memref<64x62xi8, #map>)
//                       {pad_value = 2:i8}
// as:
//    #map = affine_map<(d0, d1) -> (d0 floordiv 4, d1 floordiv 4, d0 mod 4, d1
//    mod 4)> #map_src = affine_map<(d0, d1) -> (d0, d1)> %c2_i8 =
//    arith.constant 2 : i8 %alloc = memref.alloc() : memref<64x62xi8, #map>
//    %reinterpret_cast = memref.reinterpret_cast %alloc to offset: [0], sizes:
//    [4096], strides: [1] :
//                                        memref<64x62xi8, #map> to
//                                        memref<4096xi8>
//    linalg.fill ins(%c2_i8 : i8) outs(%reinterpret_cast : memref<4096xi8>)
//    linalg.generic
//              {indexing_maps = [#map_src, #map1], iterator_types =
//              ["parallel", "parallel"]} ins(%arg1 : memref<64x62xi8>)
//              outs(%alloc : memref<64x62xi8, #map>) {
//             ^bb0(%in: i8, %out: i8):
//                      linalg.yield %in : i8
//     }
// For non-padded layouts, we directly move to linalg.generic.
//
//===----------------------------------------------------------------------===//

#include "hexagon/Dialect/HexagonMem/IR/HexagonMemDialect.h"
#include "hexagon/Transforms/Transforms.h"
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

#define DEBUG_TYPE "-convert-layout"

#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define DBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
using namespace mlir::linalg;
using namespace mlir::affine;
using namespace hexagon;

#define GEN_PASS_DEF_CONVERTLAYOUT
#include "hexagon/Transforms/Passes.h.inc"

namespace {
struct ConvertLayout final
    : public OpRewritePattern<hexagonmem::ConvertLayoutOp> {
  ConvertLayout(MLIRContext *ctx) : OpRewritePattern(ctx) {}

  LogicalResult matchAndRewrite(hexagonmem::ConvertLayoutOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value A = op.getSource();
    Value B = op.getTarget();

    // Define the affine maps
    auto srcType = cast<MemRefType>(A.getType());
    auto tgtType = cast<MemRefType>(B.getType());
    auto map_src = srcType.getLayout().getAffineMap();

    // Calculate sizes
    int srcSize =
        std::accumulate(srcType.getShape().begin(), srcType.getShape().end(), 1,
                        std::multiplies<int>());
    int tgtSize =
        std::accumulate(tgtType.getShape().begin(), tgtType.getShape().end(), 1,
                        std::multiplies<int>());
    AffineMapAttr layout;
    auto affineMap = op.getLayout();
    if (!affineMap) {
      llvm::errs() << "Error: Layout is null or invalid.\n";
    }

    if (srcSize != tgtSize) {
      auto attr = op.getPadValue();
      TypedAttr padValue;
      if (attr.has_value()) {
        padValue = cast<TypedAttr>(attr.value());
      } else {
        if (tgtType.isInteger(8)) {
          padValue = rewriter.getIntegerAttr(rewriter.getIntegerType(8), 0);
        } else if (tgtType.isInteger(16)) {
          padValue = rewriter.getIntegerAttr(rewriter.getIntegerType(16), 0);
        } else if (tgtType.isInteger(32)) {
          padValue = rewriter.getIntegerAttr(rewriter.getIntegerType(32), 0);
        } else if (tgtType.isF16()) {
          padValue = rewriter.getF16FloatAttr(0.0);
        } else if (tgtType.isF32()) {
          padValue = rewriter.getF32FloatAttr(0.0);
        }
      }
      auto padded = rewriter.create<arith::ConstantOp>(loc, padValue);
      rewriter.create<linalg::FillOp>(loc, ValueRange{padded}, ValueRange{B});
    }

    // Create the linalg.generic operation
    rewriter.replaceOpWithNewOp<linalg::GenericOp>(
        op, TypeRange(), ValueRange{A}, ValueRange{B},
        ArrayRef<AffineMap>({map_src, affineMap}),
        SmallVector<utils::IteratorType>(srcType.getRank(),
                                         utils::IteratorType::parallel),
        [=](OpBuilder &b, Location loc, ValueRange args) {
          b.create<linalg::YieldOp>(loc, args[0]);
        });

    return success();
  }
};

struct ConvertLayoutPass : public ::impl::ConvertLayoutBase<ConvertLayoutPass> {
  void runOnOperation() override {
    auto funcOp = getOperation();
    RewritePatternSet patterns(funcOp.getContext());
    patterns.add<ConvertLayout>(patterns.getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
hexagon::createConvertLayoutPass() {
  return std::make_unique<ConvertLayoutPass>();
}
