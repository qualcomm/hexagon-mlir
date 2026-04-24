//===-- DecomposeHexKLMatmulPass.cpp - Decompose hexkl.matmul to micro ops ===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// Patterns to decompose hexkl::MatmulOp into hexkl micro HMX operations.
// This pass implements the tiling strategy from hexkl_matmul_f16f16_f32.
//
//===----------------------------------------------------------------------===//

#include "hexagon/Dialect/HexKL/IR/HexKLDialect.h"
#include "hexagon/Dialect/HexagonMem/IR/HexagonMemDialect.h"
#include "hexagon/Transforms/Transforms.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "decompose-hexkl-matmul"

#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define DBG(X) LLVM_DEBUG(DBGS() << X << "\\n")

using namespace mlir;
using namespace hexagon;

#define GEN_PASS_DEF_DECOMPOSEHEXKLMATMUL
#include "hexagon/Transforms/Passes.h.inc"

namespace {

struct DecomposeHexKLMatmul final : public OpRewritePattern<hexkl::MatmulOp> {
  DecomposeHexKLMatmul(MLIRContext *ctx) : OpRewritePattern(ctx) {}

  LogicalResult matchAndRewrite(hexkl::MatmulOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value lhs = op.getLhs();
    Value rhs = op.getRhs();
    Value result = op.getOuts();

    auto lhsType = cast<MemRefType>(lhs.getType());
    auto rhsType = cast<MemRefType>(rhs.getType());
    auto resultType = cast<MemRefType>(result.getType());

    // Validate rank (must be 2D)
    ArrayRef<int64_t> lhsShape = lhsType.getShape();
    ArrayRef<int64_t> rhsShape = rhsType.getShape();
    ArrayRef<int64_t> resultShape = resultType.getShape();

    if (lhsShape.size() != 2 || rhsShape.size() != 2 ||
        resultShape.size() != 2) {
      return rewriter.notifyMatchFailure(op, "only 2D matmul supported");
    }

    // Validate static shape compatibility if available
    if (lhsType.hasStaticShape() && rhsType.hasStaticShape() &&
        resultType.hasStaticShape()) {
      int64_t M = lhsShape[0];
      int64_t K_lhs = lhsShape[1];
      int64_t K_rhs = rhsShape[0];
      int64_t N = rhsShape[1];
      int64_t M_out = resultShape[0];
      int64_t N_out = resultShape[1];

      // Validate dimensions match: lhs(M×K) × rhs(K×N) = result(M×N)
      if (K_lhs != K_rhs) {
        return rewriter.notifyMatchFailure(
            op, "inner dimensions mismatch: lhs K != rhs K");
      }
      if (M != M_out || N != N_out) {
        return rewriter.notifyMatchFailure(op, "output dimensions mismatch");
      }
    }

    // Create constants
    auto i32Ty = rewriter.getI32Type();

    Value idx0 = arith::ConstantIndexOp::create(rewriter, loc, 0);
    Value idx1 = arith::ConstantIndexOp::create(rewriter, loc, 1);
    Value idx32 = arith::ConstantIndexOp::create(rewriter, loc, 32);
    Value idx31 = arith::ConstantIndexOp::create(rewriter, loc, 31);

    Value i32_0 = arith::ConstantIntOp::create(rewriter, loc, i32Ty, 0);
    Value i32_1 = arith::ConstantIntOp::create(rewriter, loc, i32Ty, 1);
    Value i32_32 = arith::ConstantIntOp::create(rewriter, loc, i32Ty, 32);
    Value i32_4096 = arith::ConstantIntOp::create(rewriter, loc, i32Ty, 4096);
    Value idx4096 = arith::ConstantIndexOp::create(rewriter, loc, 4096);

    // Get dimensions dynamically
    Value dimM = memref::DimOp::create(rewriter, loc, lhs, idx0);
    Value dimK = memref::DimOp::create(rewriter, loc, lhs, idx1);
    Value dimN = memref::DimOp::create(rewriter, loc, rhs, idx1);

    Value M = arith::IndexCastOp::create(rewriter, loc, i32Ty, dimM);
    Value K = arith::IndexCastOp::create(rewriter, loc, i32Ty, dimK);
    Value N = arith::IndexCastOp::create(rewriter, loc, i32Ty, dimN);

    // Calculate numKTiles = (k + 31) / 32
    Value kPlus31 = arith::AddIOp::create(rewriter, loc, dimK, idx31);
    Value kTiles = arith::DivUIOp::create(rewriter, loc, kPlus31, idx32);
    Value kTilesI32 = arith::IndexCastOp::create(rewriter, loc, i32Ty, kTiles);

    // Calculate VTCM size: (numKTiles*2 + 2 + 3) * 4096
    // Layout: [act_tiles | scratch_tiles | flat_out | acc_read | extra]
    Value twoKTiles =
        arith::MulIOp::create(rewriter, loc, kTiles,
                              arith::ConstantIndexOp::create(rewriter, loc, 2));
    Value dataTiles =
        arith::AddIOp::create(rewriter, loc, twoKTiles,
                              arith::ConstantIndexOp::create(rewriter, loc, 2));
    Value vtcmTiles =
        arith::AddIOp::create(rewriter, loc, dataTiles,
                              arith::ConstantIndexOp::create(rewriter, loc, 3));
    Value vtcmBytes = arith::MulIOp::create(rewriter, loc, vtcmTiles, idx4096);

    // Manual VTCM alloc; tag for bufferization and dealloc after outer loop.
    auto vtcmType =
        MemRefType::get({ShapedType::kDynamic}, rewriter.getI8Type(),
                        MemRefLayoutAttrInterface{},
                        IntegerAttr::get(rewriter.getI32Type(), 1));
    auto vtcmAlloc =
        hexagonmem::AllocOp::create(rewriter, loc, vtcmType, vtcmBytes);
    vtcmAlloc->setAttr("bufferization.manual_deallocation",
                       rewriter.getUnitAttr());
    Value vtcm = vtcmAlloc.getResult();

    // Setup HMX accumulator
    hexkl::MicroHMXSetupAccReadF16Op::create(rewriter, loc, vtcm);

    // Hoist loop-invariant offset calculations
    // weightOffset = numKTiles * 4096 (weight buffer starts after activation
    // tiles)
    Value wOff = arith::MulIOp::create(rewriter, loc, kTilesI32, i32_4096);

    // flatOffset = numKTiles * 4096 (flat output buffer location)
    Value flatOff = wOff;

    // accReadOffset = (numKTiles + 1) * 4096 (accumulator readback location)
    Value kTilesPlus1 = arith::AddIOp::create(rewriter, loc, kTilesI32, i32_1);
    Value accOff = arith::MulIOp::create(rewriter, loc, kTilesPlus1, i32_4096);

    // Outer loop: iterate over rows (M dimension) in 32-row tiles
    auto outerFor = scf::ForOp::create(
        rewriter, loc, idx0, dimM, idx32, ValueRange{},
        [&](OpBuilder &b, Location loc, Value row, ValueRange) {
          Value rowI32 = arith::IndexCastOp::create(b, loc, i32Ty, row);
          Value rowTile = arith::DivUIOp::create(b, loc, rowI32, i32_32);

          // Load and layout activation tiles for this row
          scf::ForOp::create(
              b, loc, idx0, kTiles, idx1, ValueRange{},
              [&](OpBuilder &bb, Location loc, Value ktIdx, ValueRange) {
                Value kt = arith::IndexCastOp::create(bb, loc, i32Ty, ktIdx);

                // scratchOffset = (numKTiles + kTile) * 4096
                Value scrIdx = arith::AddIOp::create(bb, loc, kTilesI32, kt);
                Value scrOff = arith::MulIOp::create(bb, loc, scrIdx, i32_4096);

                // Copy activation tile to scratch
                hexkl::MicroHMXCopySubmatrixToF16Op::create(
                    bb, loc, vtcm, scrOff, lhs, rowTile, kt, M, K);

                // Layout activation from scratch to activation buffer
                Value actOff = arith::MulIOp::create(bb, loc, kt, i32_4096);
                hexkl::MicroHMXRmToAhF16Op::create(bb, loc, vtcm, actOff,
                                                   scrOff);

                scf::YieldOp::create(bb, loc);
              });

          // Inner loop: iterate over columns (N dimension) in 32-col tiles
          scf::ForOp::create(
              b, loc, idx0, dimN, idx32, ValueRange{},
              [&](OpBuilder &bb, Location loc, Value col, ValueRange) {
                Value colI32 = arith::IndexCastOp::create(bb, loc, i32Ty, col);
                Value colTile = arith::DivUIOp::create(bb, loc, colI32, i32_32);

                // Clear accumulator
                hexkl::MicroHMXAccClearF16Op::create(bb, loc);

                // Innermost loop: iterate over K tiles for accumulation
                scf::ForOp::create(
                    bb, loc, idx0, kTiles, idx1, ValueRange{},
                    [&](OpBuilder &bbb, Location loc, Value ktIdx, ValueRange) {
                      Value kt =
                          arith::IndexCastOp::create(bbb, loc, i32Ty, ktIdx);

                      // Load weight tile
                      hexkl::MicroHMXRmToWhF16Op::create(bbb, loc, vtcm, wOff,
                                                         rhs, kt, colTile, N);

                      // Perform matrix multiplication (accumulates)
                      Value actOff2 =
                          arith::MulIOp::create(bbb, loc, kt, i32_4096);
                      hexkl::MicroHMXMmF16Op::create(bbb, loc, vtcm, actOff2,
                                                     wOff);

                      scf::YieldOp::create(bbb, loc);
                    });

                // Read accumulator
                hexkl::MicroHMXAccReadF16Op::create(bb, loc, vtcm, accOff);

                // Convert from activation layout to flat layout
                hexkl::MicroHMXAhToRmF16Op::create(bb, loc, vtcm, flatOff,
                                                   accOff);

                // Copy result to output (f16 -> f32)
                hexkl::MicroHMXCopyF16ToF32SubmatrixOp::create(
                    bb, loc, vtcm, flatOff, result, rowTile, colTile, M, N);

                scf::YieldOp::create(bb, loc);
              });

          scf::YieldOp::create(b, loc);
        });

    // Explicitly deallocate VTCM buffer to avoid relying on ConvertToHexagonmem
    // rewriting of generic memref.dealloc for dynamic VTCM types.
    rewriter.setInsertionPointAfter(outerFor);
    hexagonmem::DeallocOp::create(rewriter, loc, vtcm);

    rewriter.eraseOp(op);
    return success();
  }
};

void populateDecomposeHexKLMatmulPatterns(RewritePatternSet &patterns) {
  patterns.add<DecomposeHexKLMatmul>(patterns.getContext());
}

struct DecomposeHexKLMatmulPass
    : public ::impl::DecomposeHexKLMatmulBase<DecomposeHexKLMatmulPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<hexkl::HexKLDialect, hexagonmem::HexagonMemDialect,
                arith::ArithDialect, scf::SCFDialect, memref::MemRefDialect>();
  }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populateDecomposeHexKLMatmulPatterns(patterns);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
hexagon::createDecomposeHexKLMatmulPass() {
  return std::make_unique<DecomposeHexKLMatmulPass>();
}
