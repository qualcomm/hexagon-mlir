//===- DecomposeHexKLMatmulToDaemonPass.cpp - hexkl.matmul -> daemon ops --===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// Patterns to decompose hexkl::MatmulOp for the dedicated-HMX-thread (daemon)
// model.  Unlike DecomposeHexKLMatmulPass, which emits the inline accumulator
// loop (setup_acc_read / acc_clear / mm / acc_read) that each worker runs on
// its own HMX engine, this pass keeps only the worker-side layout ops and
// replaces the accumulator work with hexkl.micro_hmx_dispatch (one K=32 job)
// plus a per-output-tile hexkl.micro_hmx_wait_idle.  A single daemon thread
// owns HMX and drains the dispatched jobs.
//
//===----------------------------------------------------------------------===//

#include "hexagon/Dialect/HexKL/IR/HexKLDialect.h"
#include "hexagon/Transforms/Transforms.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
// Runtime queue header: pure standard-C, reachable via the backend's
// include_directories(bin). Gives this pass the one true
// HMX_MICRO_QUEUE_LANE_DEPTH so the kRing invariant below is not a stale
// mirror.
#include "runtime/HMXMicroDaemon/HMXMicroDaemon.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "decompose-hexkl-matmul-to-daemon"

using namespace mlir;
using namespace hexagon;

#define GEN_PASS_DEF_DECOMPOSEHEXKLMATMULTODAEMON
#include "hexagon/Transforms/Passes.h.inc"

namespace {

// HMX block geometry and alignment, mirrored from the HexKL addon header
// hexkl_micro.h. That header is the device addon's (built against the Hexagon
// toolchain by bin/runtime), so it is NOT on this host pass's include path --
// hence these copies. Names match the addon macros so they stay greppable if
// the addon changes.
//
// Int8 blocks are 64x32x32, not f16 blocks rescaled by element size: activation
// tiles are 2048 B and i32 accumulator tiles are 8192 B.
static constexpr int64_t kHmxActivationAlignment = 2048; // ..._ACTIVATION_...
static constexpr int64_t kHmxWeightsAlignment = 128;     // ..._WEIGHTS_...
static constexpr int64_t kF16BlockRows = 32;             // ..._F16_BLOCK_N_ROW
static constexpr int64_t kF16BlockCols = 32;             // ..._F16_BLOCK_N_COL
static constexpr int64_t kF16BlockInner = 32;  // ..._F16_BLOCK_N_INNER
static constexpr int64_t kInt8BlockRows = 64;  // ..._INT8_BLOCK_N_ROW
static constexpr int64_t kInt8BlockCols = 32;  // ..._INT8_BLOCK_N_COL
static constexpr int64_t kInt8BlockInner = 32; // ..._INT8_BLOCK_N_INNER

// Per-worker VTCM layout:
//   [ act[0..kRing) | wgt[0..kRing) | flat[0..kNumFlat) | acc[0..kNumAcc) |
//     scr[0..kNumScr) ]
// kRing is power-of-two and exceeds queue depth, so dispatch backpressure
// prevents WAR reuse of a ring slot.
static constexpr int64_t kRing = 8;    // act/wgt ring depth (power of two)
static constexpr int64_t kNumFlat = 1; // flat (RM) result buffers
static constexpr int64_t kNumAcc = 1;  // acc-read (crouton) result buffers
static constexpr int64_t kNumScr = 1;  // flat->crouton scratch buffers
static constexpr int64_t kRingMask = kRing - 1;
static_assert((kRing & kRingMask) == 0, "kRing must be a power of two");
// Enforce the WAR invariant documented above: dispatch backpressure bounds
// outstanding jobs to the lane depth, and a ring slot is reused every kRing
// K-blocks, so its prior tile is drained before reuse only when kRing exceeds
// the lane depth. Was comment-only; assert it against the real queue constant.
static_assert(kRing >= HMX_MICRO_QUEUE_LANE_DEPTH + 1,
              "kRing must exceed HMX_MICRO_QUEUE_LANE_DEPTH so dispatch "
              "backpressure drains a ring slot's prior tile before reuse");

// Per-instance VTCM layout and tile shape. u8i8 uses distinct activation,
// weight, and accumulator sizes; fused load/store leave flat and scratch empty.
struct TileLayout {
  int64_t actTileBytes;  // activation ring slot stride
  int64_t wgtTileBytes;  // weight ring slot stride
  int64_t flatTileBytes; // flat (RM) result staging; 0 if unused
  int64_t accTileBytes;  // acc-read destination
  int64_t scrTileBytes;  // flat->AH activation staging; 0 if unused
  int64_t rowBlockRows;  // M rows per output tile (block N_ROW)
  int64_t colBlockCols;  // N cols per output tile (block N_COL)
  int64_t innerBlock;    // K reduced per dispatched job (block N_INNER)
  int64_t wgtBase;
  int64_t flatOff;
  int64_t accOff;
  int64_t scrOff;
  int64_t perInstBytes;
};

static constexpr TileLayout
makeTileLayout(int64_t actTileBytes, int64_t wgtTileBytes,
               int64_t flatTileBytes, int64_t accTileBytes,
               int64_t scrTileBytes, int64_t rowBlockRows, int64_t colBlockCols,
               int64_t innerBlock) {
  TileLayout l{};
  l.actTileBytes = actTileBytes;
  l.wgtTileBytes = wgtTileBytes;
  l.flatTileBytes = flatTileBytes;
  l.accTileBytes = accTileBytes;
  l.scrTileBytes = scrTileBytes;
  l.rowBlockRows = rowBlockRows;
  l.colBlockCols = colBlockCols;
  l.innerBlock = innerBlock;
  l.wgtBase = kRing * actTileBytes;
  l.flatOff = l.wgtBase + kRing * wgtTileBytes;
  l.accOff = l.flatOff + kNumFlat * flatTileBytes;
  l.scrOff = l.accOff + kNumAcc * accTileBytes;
  l.perInstBytes = l.scrOff + kNumScr * scrTileBytes;
  return l;
}

// F16 uses 32x32, 2048-byte tiles and staged activation/result conversion.
static constexpr int64_t kF16ActTileBytes = kF16BlockRows * kF16BlockInner * 2;
static constexpr int64_t kF16WgtTileBytes = kF16BlockInner * kF16BlockCols * 2;
static constexpr int64_t kF16ResTileBytes = kF16BlockRows * kF16BlockCols * 2;
static constexpr TileLayout kF16Layout = makeTileLayout(
    kF16ActTileBytes, kF16WgtTileBytes, kF16ResTileBytes, kF16ResTileBytes,
    kF16ActTileBytes, kF16BlockRows, kF16BlockCols, kF16BlockInner);

// u8i8 uses 2048/1024/8192-byte activation/weight/accumulator tiles. Its fused
// load and store operations need no flat or scratch staging.
static constexpr int64_t kU8I8ActTileBytes =
    kInt8BlockRows * kInt8BlockInner * 1;
static constexpr int64_t kU8I8WgtTileBytes =
    kInt8BlockInner * kInt8BlockCols * 1;
static constexpr int64_t kU8I8AccTileBytes =
    kInt8BlockRows * kInt8BlockCols * 4;
static constexpr TileLayout kU8I8Layout =
    makeTileLayout(kU8I8ActTileBytes, kU8I8WgtTileBytes, /*flatTileBytes=*/0,
                   kU8I8AccTileBytes, /*scrTileBytes=*/0, kInt8BlockRows,
                   kInt8BlockCols, kInt8BlockInner);
static_assert(kU8I8Layout.perInstBytes == 32768,
              "u8i8 layout exactly consumes its 32 KiB per-instance budget");

// Addon alignment preconditions: activation slot and thread strides must align,
// as must the weight base and stride.
static_assert(kF16Layout.actTileBytes % kHmxActivationAlignment == 0 &&
                  kU8I8Layout.actTileBytes % kHmxActivationAlignment == 0,
              "activation ring stride must be HEXKL_HMX_ACTIVATION_ALIGNMENT "
              "aligned");
static_assert(kF16Layout.perInstBytes % kHmxActivationAlignment == 0 &&
                  kU8I8Layout.perInstBytes % kHmxActivationAlignment == 0,
              "per-instance stride must keep each thread's activation offsets "
              "HEXKL_HMX_ACTIVATION_ALIGNMENT aligned");
static_assert(kF16Layout.wgtBase % kHmxWeightsAlignment == 0 &&
                  kU8I8Layout.wgtBase % kHmxWeightsAlignment == 0 &&
                  kF16Layout.wgtTileBytes % kHmxWeightsAlignment == 0 &&
                  kU8I8Layout.wgtTileBytes % kHmxWeightsAlignment == 0,
              "weight base and ring stride must be "
              "HEXKL_HMX_WEIGHTS_ALIGNMENT aligned");

struct DecomposeHexKLMatmulToDaemon final
    : public OpRewritePattern<hexkl::MatmulOp> {
  DecomposeHexKLMatmulToDaemon(MLIRContext *ctx) : OpRewritePattern(ctx) {}

  LogicalResult matchAndRewrite(hexkl::MatmulOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value lhs = op.getLhs();
    Value rhs = op.getRhs();
    Value result = op.getOuts();

    auto lhsType = cast<MemRefType>(lhs.getType());
    auto rhsType = cast<MemRefType>(rhs.getType());
    auto resultType = cast<MemRefType>(result.getType());

    // hexkl.matmul's verifier only allows (f16,f16,{f16,f32}) or
    // (i8,i8,i32); tell the two paths apart once here.
    bool isF16 = lhsType.getElementType().isF16();
    bool isU8I8 = lhsType.getElementType().isInteger(8);
    if (!isF16 && !isU8I8) {
      return rewriter.notifyMatchFailure(
          op, "expected f16 or i8 (u8i8) element type");
    }

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

      if (K_lhs != K_rhs) {
        return rewriter.notifyMatchFailure(
            op, "inner dimensions mismatch: lhs K != rhs K");
      }
      if (M != M_out || N != N_out) {
        return rewriter.notifyMatchFailure(op, "output dimensions mismatch");
      }
    }

    // Locate the injected daemon args on the enclosing func.  Order (tail):
    //   [..., thread_id (n-3), vtcm_base (n-2), queue (n-1)]
    auto funcOp = op->getParentOfType<func::FuncOp>();
    if (!funcOp || !funcOp->hasAttr("hmx.thread_id_arg") ||
        !funcOp->hasAttr("hexkl_hmx.vtcm_base_arg") ||
        !funcOp->hasAttr("hmx.queue_arg")) {
      op.emitError("hexkl.matmul (daemon) requires hmx.thread_id_arg, "
                   "hexkl_hmx.vtcm_base_arg and hmx.queue_arg on the "
                   "enclosing func; run InjectHMXThreadIdArgPass, "
                   "InjectHMXVtcmBaseArgPass and InjectHMXQueueArgPass first");
      return failure();
    }
    unsigned numArgs = funcOp.getNumArguments();
    Value queue = funcOp.getArgument(numArgs - 1);
    Value vtcm = funcOp.getArgument(numArgs - 2);
    Value threadId = funcOp.getArgument(numArgs - 3);

    // One function cannot mix job kinds because its daemon VTCM slice has one
    // layout. The daemon itself supports per-job accumulator modes.
    StringRef jobKind = isF16 ? "f16" : "u8i8";
    if (auto existing =
            funcOp->getAttrOfType<StringAttr>("hexkl_hmx.daemon_job_kind")) {
      if (existing.getValue() != jobKind) {
        op.emitError("hexkl.matmul (daemon) of kind '")
            << jobKind << "' conflicts with this func's existing "
            << "hexkl_hmx.daemon_job_kind = '" << existing.getValue()
            << "': a single daemon queue cannot serve mixed f16/u8i8 jobs";
        return failure();
      }
    } else {
      funcOp->setAttr("hexkl_hmx.daemon_job_kind",
                      rewriter.getStringAttr(jobKind));
    }

    // Constants
    auto i32Ty = rewriter.getI32Type();

    // Select the element-type-specific tile geometry and VTCM layout.
    const TileLayout &layout = isF16 ? kF16Layout : kU8I8Layout;

    Value idx0 = arith::ConstantIndexOp::create(rewriter, loc, 0);
    Value idx1 = arith::ConstantIndexOp::create(rewriter, loc, 1);
    Value idxRowStep =
        arith::ConstantIndexOp::create(rewriter, loc, layout.rowBlockRows);
    Value idxColStep =
        arith::ConstantIndexOp::create(rewriter, loc, layout.colBlockCols);
    Value idxInner =
        arith::ConstantIndexOp::create(rewriter, loc, layout.innerBlock);
    Value idxInnerMinus1 =
        arith::ConstantIndexOp::create(rewriter, loc, layout.innerBlock - 1);

    Value i32_rowBlock =
        arith::ConstantIntOp::create(rewriter, loc, i32Ty, layout.rowBlockRows);
    Value i32_colBlock =
        arith::ConstantIntOp::create(rewriter, loc, i32Ty, layout.colBlockCols);
    // Activation and weight rings have independent strides.
    Value i32_actTileBytes =
        arith::ConstantIntOp::create(rewriter, loc, i32Ty, layout.actTileBytes);
    Value i32_wgtTileBytes =
        arith::ConstantIntOp::create(rewriter, loc, i32Ty, layout.wgtTileBytes);
    // Ring mask and weight-ring base offset.
    Value i32_ringMask =
        arith::ConstantIntOp::create(rewriter, loc, i32Ty, kRingMask);
    Value i32_wgtBase =
        arith::ConstantIntOp::create(rewriter, loc, i32Ty, layout.wgtBase);

    // Dimensions
    Value dimM = memref::DimOp::create(rewriter, loc, lhs, idx0);
    Value dimK = memref::DimOp::create(rewriter, loc, lhs, idx1);
    Value dimN = memref::DimOp::create(rewriter, loc, rhs, idx1);

    Value M = arith::IndexCastOp::create(rewriter, loc, i32Ty, dimM);
    Value K = arith::IndexCastOp::create(rewriter, loc, i32Ty, dimK);
    Value N = arith::IndexCastOp::create(rewriter, loc, i32Ty, dimN);

    // numKTiles = ceil(k / innerBlock)
    Value kPlusInner =
        arith::AddIOp::create(rewriter, loc, dimK, idxInnerMinus1);
    Value kTiles = arith::DivUIOp::create(rewriter, loc, kPlusInner, idxInner);
    Value kTilesMinus1 = arith::SubIOp::create(rewriter, loc, kTiles, idx1);

    // Loop-invariant result-buffer offsets.
    Value accLocal =
        arith::ConstantIntOp::create(rewriter, loc, i32Ty, layout.accOff);
    Value perInstBytesI32 =
        arith::ConstantIntOp::create(rewriter, loc, i32Ty, layout.perInstBytes);

    // Convert a per-instance offset to a global VTCM offset.
    Value tidBase =
        arith::MulIOp::create(rewriter, loc, threadId, perInstBytesI32);
    auto g = [&](OpBuilder &b, Location loc, Value localOff) -> Value {
      return arith::AddIOp::create(b, loc, tidBase, localOff);
    };

    // The staged f16 path alone uses flat and scratch buffers.
    Value accOffG = g(rewriter, loc, accLocal);
    Value flatOffG, scrOffG;
    if (isF16) {
      Value flatLocal =
          arith::ConstantIntOp::create(rewriter, loc, i32Ty, layout.flatOff);
      Value scrLocal =
          arith::ConstantIntOp::create(rewriter, loc, i32Ty, layout.scrOff);
      flatOffG = g(rewriter, loc, flatLocal);
      scrOffG = g(rewriter, loc, scrLocal);
    }

    // K is innermost to bound live ring tiles independently of K.
    scf::ForOp::create(
        rewriter, loc, idx0, dimM, idxRowStep, ValueRange{},
        [&](OpBuilder &b, Location loc, Value row, ValueRange) {
          Value rowI32 = arith::IndexCastOp::create(b, loc, i32Ty, row);
          Value rowTile = arith::DivUIOp::create(b, loc, rowI32, i32_rowBlock);

          scf::ForOp::create(
              b, loc, idx0, dimN, idxColStep, ValueRange{},
              [&](OpBuilder &bb, Location loc, Value col, ValueRange) {
                Value colI32 = arith::IndexCastOp::create(bb, loc, i32Ty, col);
                Value colTile =
                    arith::DivUIOp::create(bb, loc, colI32, i32_colBlock);

                // Dispatch each K block through a backpressure-protected slot.
                scf::ForOp::create(
                    bb, loc, idx0, kTiles, idx1, ValueRange{},
                    [&](OpBuilder &bbb, Location loc, Value ktIdx, ValueRange) {
                      Value kt =
                          arith::IndexCastOp::create(bbb, loc, i32Ty, ktIdx);

                      // Ring slot p = kt & kRingMask; activation and weight use
                      // independent strides.
                      Value p =
                          arith::AndIOp::create(bbb, loc, kt, i32_ringMask);
                      Value actLocal =
                          arith::MulIOp::create(bbb, loc, p, i32_actTileBytes);
                      Value actOffG = g(bbb, loc, actLocal);
                      Value wgtSlot =
                          arith::MulIOp::create(bbb, loc, p, i32_wgtTileBytes);
                      Value wgtLocal =
                          arith::AddIOp::create(bbb, loc, wgtSlot, i32_wgtBase);
                      Value wgtOffG = g(bbb, loc, wgtLocal);

                      // F16 stages activation; u8i8 has fused copy and layout.
                      if (isF16) {
                        hexkl::MicroHMXCopySubmatrixToF16Op::create(
                            bbb, loc, vtcm, scrOffG, lhs, rowTile, kt, M, K);
                        hexkl::MicroHMXRmToAhF16Op::create(bbb, loc, vtcm,
                                                           actOffG, scrOffG);
                        hexkl::MicroHMXRmToWhF16Op::create(
                            bbb, loc, vtcm, wgtOffG, rhs, kt, colTile, N);
                      } else {
                        hexkl::MicroHMXCopySubmatrixTo8bActivationOp::create(
                            bbb, loc, vtcm, actOffG, lhs, rowTile, kt, M, K);
                        hexkl::MicroHMXRmToWhI8Op::create(
                            bbb, loc, vtcm, wgtOffG, rhs, kt, colTile, N);
                      }

                      // clear_acc on first K-block, write_out on last.
                      Value clearAcc = arith::CmpIOp::create(
                          bbb, loc, arith::CmpIPredicate::eq, ktIdx, idx0);
                      Value writeOut = arith::CmpIOp::create(
                          bbb, loc, arith::CmpIPredicate::eq, ktIdx,
                          kTilesMinus1);

                      hexkl::MicroHMXDispatchOp::create(
                          bbb, loc, queue, threadId, actOffG, wgtOffG, accOffG,
                          clearAcc, writeOut, jobKind);
                      scf::YieldOp::create(bbb, loc);
                    });

                // Wait until the daemon drains this lane and publishes
                // acc_read.
                hexkl::MicroHMXWaitIdleOp::create(bb, loc, queue, threadId);

                // F16 needs AH-to-RM conversion; u8i8 stores directly.
                if (isF16) {
                  hexkl::MicroHMXAhToRmF16Op::create(bb, loc, vtcm, flatOffG,
                                                     accOffG);
                  hexkl::MicroHMXCopyF16ToF32SubmatrixOp::create(
                      bb, loc, vtcm, flatOffG, result, rowTile, colTile, M, N);
                } else {
                  hexkl::MicroHMXCopy32bToSubmatrixOp::create(
                      bb, loc, vtcm, accOffG, result, rowTile, colTile, M, N);
                }
                scf::YieldOp::create(bb, loc);
              });

          scf::YieldOp::create(b, loc);
        });

    rewriter.eraseOp(op);
    return success();
  }
};

void populateDecomposeHexKLMatmulToDaemonPatterns(RewritePatternSet &patterns) {
  patterns.add<DecomposeHexKLMatmulToDaemon>(patterns.getContext());
}

struct DecomposeHexKLMatmulToDaemonPass
    : public ::impl::DecomposeHexKLMatmulToDaemonBase<
          DecomposeHexKLMatmulToDaemonPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<hexkl::HexKLDialect, arith::ArithDialect, scf::SCFDialect,
                    memref::MemRefDialect>();
  }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populateDecomposeHexKLMatmulToDaemonPatterns(patterns);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
hexagon::createDecomposeHexKLMatmulToDaemonPass() {
  return std::make_unique<DecomposeHexKLMatmulToDaemonPass>();
}
