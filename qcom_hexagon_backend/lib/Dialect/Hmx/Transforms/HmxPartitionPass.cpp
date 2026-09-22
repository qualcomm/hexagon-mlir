//===-- HmxPartitionPass.cpp - hmx.matmul to the tile loop ----------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// The tile level of the HMX path: an `hmx.matmul` on crouton arrays becomes
// bias init -> (m, n) tiles -> K loop of mmas -> read-out.
//
// VTCM ownership (docs/hmx/hmx-system-design.md §10.12): the crouton arrays
// stay ordinary memory-space-1 buffers, and this pass deliberately does not carve
// a region out for them. The VTCM the kernel runs in is the one the runtime
// already owns, and the existing space-1 machinery (convert-to-hexagonmem -> the
// runtime's VTCM pool) is what places them there, together with every other VTCM
// buffer of the kernel. Acquiring a region of our own instead -- which is what
// this pass used to do -- takes megabytes away from that pool, the pool then
// fails its own "at least 1 MB available" check, and the DSP process dies.
//
// Staged tile loop: the activation bridge is a loop of HVX packs from a
// row-major DDR source, and on a single thread an HVX pack cannot overlap the
// (synchronous) HMX engine -- only the DMA engine can (measured: probe (B-A)/H =
// 0.998, exp/hmx/async_probe). So when the activation behind an `hmx.matmul` is
// that bridge loop, this pass stages one 32 x K activation tile at a time into a
// static ring of VTCM slots with `hmx.stage`, and re-hosts the pack to read the
// awaited slot (VTCM -> crouton) instead of DDR.
//
// The tile loop is not hand-pipelined. The pass emits the serial source form --
// one iteration issues tile m's transfer with `hmx.stage`, awaits it and
// computes it -- and at depth 2 hands that loop to the SCF software pipeliner
// (`scf::pipelineForLoop`) with a two-stage schedule: issue in stage 0, await
// and compute in stage 1. The pipeliner generates the prologue, the steady
// kernel and the peeled epilogue, and versions the cross-stage values -- the
// token and the slot select -- as its own iter_args, so the ring rotation is
// the pipeliner's value versioning rather than hand-written rotation. What
// makes the two static slots alternate is a parity select over the tile index,
// a stage-0 op the pipeliner re-evaluates with the shifted induction variable
// in the kernel's issue part. Depth 1 is the same source loop left unpipelined:
// issue and await stay adjacent, one slot suffices, and there is nothing to
// overlap. Either way the weight stays whole and resident, and the accumulator
// chain is untouched: one clear -> K mmas -> fused read per (m, n) tile, never
// two overlapping chains.
//
// Staging is profitable only when K is deep enough to hide the DMA engine's
// fixed per-transfer cost, so `auto` declines a shape with `Kt <
// kStageMinKTiles` (32) and keeps the plain tile loop. An explicit
// `pipeline-depth=1/2` bypasses the floor -- those are the A/B arms.
//
// The pack destination is one crouton-row scratch, not the whole activation
// array: the pack and its mmas are in the same iteration, so exactly one crouton
// row is live at a time and a single statically-allocated
// `memref<1 x Kt x 16 x 32 x 2 x f16, 1>` suffices. The pack's result *is* that
// buffer, so the mmas consume the packed croutons by address -- no extra copy
// between the two -- and there is no reason to keep a second in-flight row. The
// activation array itself is dead in this path and is retired whole (bridge
// loop, pack writers, deallocation and allocation), so only the row-major
// staging ring and the scratch occupy VTCM on top of the weight and the
// accumulator. A staged loop therefore never carries a full second copy of the
// activation.
//
// The last tiles are the pipeliner's peeled epilogue, which only awaits and
// computes -- it stages nothing. That is what keeps every `hmx.stage` row in
// range without a guard or a token sentinel: the kernel stops one iteration
// short per pipeline stage, so no issue ever runs past tile Mt-1.
// `hmx.stage`/`hmx.await` therefore lower to one unconditional DMA runtime call
// each -- a sentinel is not an option because the runtime's real tokens start at
// 0.
//
// The ring is static -- `depth` slots and `depth` status words, allocated once
// outside the loop and released once after the epilogue -- so the loop body
// allocates nothing. When the pipeline declines for any reason, the plain tile
// loop is emitted and a remark names the reason.
//===----------------------------------------------------------------------===//

#include "hexagon/Common/Common.h"
#include "hexagon/Dialect/Hmx/IR/HmxDialect.h"
#include "hexagon/Dialect/Hmx/Transforms/HmxTarget.h"
#include "hexagon/Dialect/Hmx/Transforms/Transforms.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <optional>

#define DEBUG_TYPE "hmx-partition"

using namespace mlir;
using namespace mlir::hmx;

namespace mlir {
namespace hmx {
#define GEN_PASS_DEF_HMXPARTITION
#include "hexagon/Dialect/Hmx/Transforms/Passes.h.inc"
} // namespace hmx
} // namespace mlir

namespace {

/// The conversion state block is 256 B, and the bias registers address it with
/// the low eight bits, so it has to be 256-byte aligned.
constexpr int64_t kConvStateBytes = 256;
constexpr int64_t kConvStateAlignment = 256;

/// `pipeline-depth` selecting the unstaged serial tile loop: no staging rewrite,
/// the activation bridge and its array kept as ordinary memory. It is the A/B
/// arm that reproduces the pre-pipeline codegen; the staged emitter never sees
/// this value.
constexpr int64_t kSerialPipelineDepth = 3;

/// The K extent (in croutons) at or above which `auto` (`pipeline-depth=0`) is
/// willing to stage. It is a profitability floor, not a footprint one: the
/// staged path can pay for the ring at any K, but the overlap only wins when
/// the transfer is large next to the DMA engine's fixed per-transfer cost
/// (issue + wait + scratch addressing). One staged tile is `32 * K * 2` bytes
/// with `K = 32 * Kt`, i.e. `2 KiB * Kt`: at the Kt=32 floor that is 64 KiB
/// (2 KiB per source row) and it shrinks linearly with Kt. Measured on device
/// in one build (warm A/B, `auto` staging vs the `pipeline-depth=3` serial arm,
/// on the S1/S2/S3 anchor shapes; drivers under exp/hmx):
///   * S2 (Kt=64): 297 us -> 153 us = 1.94x  (overlap wins)
///   * S1 (Kt=2):  125.5 us -> 126.5 us      (flat, nothing to win)
///   * S3 (Kt=4):  60.5 us -> 67 us = -10%   (overlap loses)
/// so the cutoff sits at Kt=32. Only `auto` is gated: `pipeline-depth=1/2` are
/// the A/B arms and stage whatever the shape is, and `3` never reaches the
/// staged emitter.
constexpr int64_t kStageMinKTiles = 32;

/// The tile sizes of a matmul, read off the crouton arrays' leading dimensions.
struct TileShape {
  int64_t m, n, k;
};

std::optional<TileShape> getTileShape(MatmulOp op) {
  auto lhs = dyn_cast<MemRefType>(op.getLhs().getType());
  auto rhs = dyn_cast<MemRefType>(op.getRhs().getType());
  auto out = dyn_cast<MemRefType>(op.getOuts().getType());
  if (!lhs || !rhs || !out)
    return std::nullopt;
  if (lhs.getRank() != 5 || rhs.getRank() != 5 || out.getRank() != 5)
    return std::nullopt;
  return TileShape{lhs.getDimSize(0), rhs.getDimSize(0), lhs.getDimSize(1)};
}

/// Bytes already committed to VTCM in this function. After bufferization the
/// static space-1 crouton allocations (`hmx.alloc_crouton`, the tensor-level
/// image of this walk) are `memref.alloc`s, so this is the tile-level image of
/// `MatmulToHmxPass::vtcmBytesCommitted`: the budget the attribution checked is
/// the same budget the pipeline checks here.
static int64_t vtcmBytesCommitted(func::FuncOp func) {
  int64_t bytes = 0;
  func.walk([&](memref::AllocOp alloc) {
    auto type = dyn_cast<MemRefType>(alloc.getType());
    if (!type || !type.hasStaticShape() ||
        type.getMemorySpaceAsInt() != hexagon::VTCM_ADDRESS_SPACE)
      return;
    Type elem = type.getElementType();
    if (!elem.isIntOrFloat())
      return;
    bytes += type.getNumElements() * (elem.getIntOrFloatBitWidth() / 8);
  });
  // Resident constant weights are `hexagonmem.alloc`s (or already lowered),
  // never `memref.alloc`s, so the resident declaration is the only place their
  // footprint is visible here. Reading the same module attribute the
  // attribution checked keeps one budget for both levels.
  if (auto module = func->getParentOfType<ModuleOp>())
    if (auto resident =
            module->getAttrOfType<IntegerAttr>("hmx.weight_resident_bytes"))
      bytes += resident.getInt();
  return bytes;
}

/// The activation bridge behind an `hmx.matmul`: the pack(s) that fill the
/// activation crouton array from one row-major source. That source is the DDR
/// side the tile loop stages; without it (a chained read-out -- already a
/// crouton in VTCM -- or a plain allocation) there is nothing to stage and the
/// plain tile loop is the right form.
///
/// Two shapes reach here. After bufferization the bridge carries the array as a
/// loop iter_arg (the `matmul-to-hmx` tensor form); canonicalization folds that
/// DPS write into a straight in-place pack loop on the allocation. Both are the
/// same bridge and both are staged.
///
/// This is a finder, not a validator: the bridge is emitted by `matmul-to-hmx`,
/// so the one structural fact the tile loop needs is that every producer of the
/// array shares one source and one enclosing loop. The old body-shape whitelist
/// is gone with the hand-written pipeline.
struct ActivationBridge {
  Value src;                    // row-major source (the DDR side to stage)
  Value buffer;                 // the crouton buffer the array is built on
  scf::ForOp loop;              // enclosing pack loop; null when unlooped
  SmallVector<PackActOp> packs; // every pack that writes the array
};

/// Every `hmx.pack_act` that writes `array` as its destination.
static SmallVector<PackActOp> packWriters(Value array) {
  SmallVector<PackActOp> packs;
  for (OpOperand &u : array.getUses())
    if (auto p = dyn_cast<PackActOp>(u.getOwner()))
      if (p.getDst() == array)
        packs.push_back(p);
  return packs;
}

std::optional<ActivationBridge> findActivationBridge(Value act) {
  SmallVector<PackActOp> packs;
  Value buffer = act;
  scf::ForOp loop;
  if (act.getDefiningOp<memref::AllocOp>()) {
    // Canonicalized form: the packs write the allocation in place.
    packs = packWriters(act);
    if (!packs.empty())
      loop = packs.front()->getParentOfType<scf::ForOp>();
  } else if (auto carried = act.getDefiningOp<scf::ForOp>()) {
    // Carried form: the array is the pack loop's iter_arg, so the buffer is the
    // iter_arg's initial value (the allocation) and the producers write the
    // region iter_arg.
    loop = carried;
    if (loop.getNumResults() != 1 || loop.getResult(0) != act ||
        loop.getInitArgs().empty())
      return std::nullopt;
    buffer = loop.getInitArgs()[0];
    for (Operation &inner : loop.getBody()->without_terminator())
      if (auto p = dyn_cast<PackActOp>(inner))
        if (p.getDst() == loop.getRegionIterArg(0))
          packs.push_back(p);
  } else {
    return std::nullopt;
  }
  if (packs.empty())
    return std::nullopt;

  // The bridge is one source packed by every writer; an unrolled body just has
  // more writers of the same source. Writers of different sources into one array
  // are not the bridge shape.
  Value src = packs.front().getSrc();
  for (PackActOp p : packs)
    if (p.getSrc() != src || p->getParentOfType<scf::ForOp>() != loop)
      return std::nullopt;
  return ActivationBridge{src, buffer, loop, packs};
}

/// One `hmx.pack_act`: the 32x32 block at (`row`, `col`) of `src` into crouton
/// (`row`, `col`) of `dst`. The memref form returns its destination (interface
/// plan §9.3), so the written buffer is an SSA value; the tile loop does not
/// need to read it back (the pack and its mma are in the same iteration), but
/// the result is what makes the dependency visible to an external scheduler.
static Value emitPackAct(IRRewriter &rewriter, Location loc, Value dst,
                         Value src, Value row, Value col,
                         IntegerAttr count = {}) {
  return PackActOp::create(rewriter, loc, TypeRange{dst.getType()}, dst, src,
                           row, col, count)
      ->getResult(0);
}

/// One `hmx.stage`: start the DMA of activation tile `row` from the row-major
/// DDR `src` into VTCM slot `dst`, recording completion in `status`. Returns the
/// token that `emitAwait` waits on -- an i32 register, the ring's only handle.
/// The op issues its transfer unconditionally; keeping every `row` in range is
/// the tile loop's job, done by construction (see the pass header).
static Value emitStage(IRRewriter &rewriter, Location loc, Value src, Value row,
                       Value dst, Value status) {
  return StageOp::create(rewriter, loc, TypeRange{rewriter.getI32Type()},
                         ValueRange{src, row, dst, status})
      ->getResult(0);
}

/// Wait for `token`'s transfer and hand back the now-ready slot. The returned
/// memref is `dst` itself (the op returns what it waited on), which is what the
/// pack consumes: the wait -> pack edge is an SSA value edge, not an effect.
static Value emitAwait(IRRewriter &rewriter, Location loc, Value token,
                       Value dst) {
  return AwaitOp::create(rewriter, loc, TypeRange{dst.getType()},
                         ValueRange{token, dst})
      ->getResult(0);
}

/// Emit the K traversal of one tile as `hmx.mma`s: one mma per K crouton.
///
/// `act`/`wt` are the whole crouton arrays and `m`/`n` the tile indices.
/// `zero`/`step`/`cKt` are the loop bounds the caller already materialised.
static void emitMmaKLoop(IRRewriter &rewriter, Location loc, Value act, Value wt,
                         Value m, Value n, Value zero, Value step, Value cKt) {
  auto kLoop = scf::ForOp::create(rewriter, loc, zero, cKt, step, ValueRange{});
  rewriter.setInsertionPointToStart(kLoop.getBody());
  MmaOp::create(rewriter, loc, act, wt, m, n, kLoop.getInductionVar(),
                rewriter.getI32IntegerAttr(1));
  rewriter.setInsertionPointAfter(kLoop);
}

/// The serial tile loop: bias clear, the (m, n) tiles, one K loop of mmas and a
/// fused read-out per tile. This is the form whenever the pipeline does not
/// apply, including the `pipeline-depth=3` arm.
///
/// It touches only the matmul's own operands: `act`, `wt` and the output are
/// consumed where they are, so a bridge that fills `act` (and the array itself)
/// survives untouched. That is what makes the serial arm work for a bridge shape
/// -- the pack loop stays and the mma reads the whole array, with nothing
/// staged. There is no structural assumption here beyond the tile shapes the
/// caller already resolved.
static void emitSerialTileLoop(IRRewriter &rewriter, Location opLoc, Value bias,
                               MatmulOp op, TileShape shape) {
  Value act = op.getLhs();
  Value wt = op.getRhs();
  Value ar = op.getOuts();

  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(op);
  auto zero = arith::ConstantIndexOp::create(rewriter, opLoc, 0);
  auto step = arith::ConstantIndexOp::create(rewriter, opLoc, 1);
  auto mM = arith::ConstantIndexOp::create(rewriter, opLoc, shape.m);
  auto nN = arith::ConstantIndexOp::create(rewriter, opLoc, shape.n);
  auto kK = arith::ConstantIndexOp::create(rewriter, opLoc, shape.k);

  auto mLoop = scf::ForOp::create(rewriter, opLoc, zero, mM, step, ValueRange{});
  rewriter.setInsertionPointToStart(mLoop.getBody());
  Value m = mLoop.getInductionVar();

  auto nLoop = scf::ForOp::create(rewriter, opLoc, zero, nN, step, ValueRange{});
  rewriter.setInsertionPointToStart(nLoop.getBody());
  AccClearOp::create(rewriter, opLoc);
  emitMmaKLoop(rewriter, opLoc, act, wt, m, nLoop.getInductionVar(), zero, step,
               kK);
  AccReadOp::create(rewriter, opLoc, bias, ar, m, nLoop.getInductionVar(),
                    rewriter.getI32IntegerAttr(0));
}

/// The compute half of one tile: pack the awaited slot's 32 x K rows into the
/// one crouton-row scratch, then one (acc_clear, K mmas, acc_read) per N tile.
/// This is the compute part of the source loop's body; the pipeliner clones it
/// into the kernel and the peeled epilogue, both of which consume the awaited
/// slot the same way. The caller has already emitted the await and placed the
/// insertion point where the tile's work belongs.
///
/// The scratch is always crouton row 0: the slot holds exactly the 32 source
/// rows of tile `m`, and the pack and its mma are in the same iteration, so the
/// crouton row index the engine reads is 0 regardless of which output tile this
/// iteration computes. The output tile row `m` is still what `acc_read` writes.
/// Rewriting the whole scratch every iteration costs nothing extra -- the pack
/// has to write every crouton of the tile anyway, and no other iteration's data
/// is live in it -- and it is what lets one buffer (not the whole array) hold
/// the activation.
static void emitTileCompute(IRRewriter &rewriter, Location loc, Value bias,
                            Value scratch, Value wt, Value ar, Value m,
                            Value stagedSlot, Value c0, Value c1, Value cKt,
                            Value cNt) {
  // The pack's destination is the scratch itself: its crouton grid is one row
  // tall, so the destination crouton is (0, k) while the source block is
  // (row 0, column k) of the staged slot. The scratch's contiguous axis is K,
  // so one ranged pack covers the whole K run -- the same shape the row-major
  // bridge uses (`packCroutonsWithLeaves`); `cKt` is not needed as a loop bound.
  auto scratchType = cast<MemRefType>(scratch.getType());
  emitPackAct(rewriter, loc, scratch, stagedSlot, c0, c0,
              rewriter.getI64IntegerAttr(scratchType.getDimSize(1)));

  auto nLoop = scf::ForOp::create(rewriter, loc, c0, cNt, c1, ValueRange{});
  rewriter.setInsertionPointToStart(nLoop.getBody());
  AccClearOp::create(rewriter, loc);
  emitMmaKLoop(rewriter, loc, scratch, wt, c0, nLoop.getInductionVar(), c0, c1,
               cKt);
  AccReadOp::create(rewriter, loc, bias, ar, m, nLoop.getInductionVar(),
                    rewriter.getI32IntegerAttr(0));
  rewriter.setInsertionPointAfter(nLoop);
}

/// The staged tile loop. Returns false (and leaves the IR alone) when the
/// activation is not a bridge with a row-major source, when `auto` sees a K
/// extent below the staging floor (`kStageMinKTiles`), when the shape does not
/// fit the staging geometry, or when even the serial ring does not fit the VTCM
/// budget; the caller then emits the plain tile loop. Every decline names its
/// reason in a remark -- the pipeline is an optimisation, never a precondition,
/// and a silent fallback is an unobservable one.
///
/// The loop is the interface contract of
/// `docs/hmx/hmx-scheduling-interface-plan.md` §9.4 in its pipelined form: the
/// pass emits the serial source loop (issue -> await -> compute per tile) and,
/// at depth 2, hands it to `scf::pipelineForLoop` with the schedule "issue in
/// stage 0, await and compute in stage 1". The pipeliner generates the
/// prologue, the steady kernel and the peeled epilogue and versions the
/// cross-stage (token, slot) pair as iter_args; the depth-2 overlap -- tile
/// m+1's transfer in flight during tile m's compute -- is the pipeliner's
/// kernel `issue(m+1) await(m) compute(m)`, whose DMA/await/compute order is
/// identical to the hand-written ring it replaced. `depth` is 2 when the double
/// ring fits and `Mt > depth`, and 1 otherwise: the same source loop left
/// unpipelined (issue and await adjacent, one slot, no overlap).
///
/// `requestedDepth` is the pass's `pipeline-depth` knob: 0 lets the budget and
/// the tile count pick the deepest ring that fits, 1 forces the serial ring, and
/// 2 asks for the double ring (narrowed to the deepest ring that fits, with a
/// remark, when the budget cannot pay for it). The unstaged arm (3) never
/// reaches this emitter -- the caller handles it before asking for staging.
/// Whatever the knob says, the pass never exceeds the budget: an impossible
/// request becomes a remark plus the best depth that fits, never a silent
/// over-commit.
static bool emitStageLoop(IRRewriter &rewriter, Location opLoc, Value bias,
                          MatmulOp op, func::FuncOp func, int64_t vtcmBudget,
                          int64_t requestedDepth) {
  Value act = op.getLhs();
  Value wt = op.getRhs();
  Value ar = op.getOuts();

  auto bridge = findActivationBridge(act);
  if (!bridge) {
    op.emitRemark() << "HMX pipeline not applied: the activation is not a "
                       "row-major staging bridge (nothing to re-host onto a "
                       "VTCM slot), so the tile loop reads the crouton array "
                       "where it is";
    return false;
  }

  // The array is refilled by the tile loop, so the only other readers allowed
  // are the bridge's own packs, this matmul, and its deallocation. A second
  // reader would observe the array after the tile loop rewrote it.
  for (OpOperand &u : act.getUses()) {
    Operation *owner = u.getOwner();
    if (owner == op.getOperation() ||
        (isa<PackActOp>(owner) && cast<PackActOp>(owner).getDst() == act) ||
        isa<memref::DeallocOp, scf::YieldOp>(owner))
      continue;
    op.emitRemark() << "HMX pipeline not applied: the activation array has a "
                       "reader the staging rewrite does not own";
    return false;
  }

  auto actType = dyn_cast<MemRefType>(bridge->buffer.getType());
  auto wtType = dyn_cast<MemRefType>(wt.getType());
  auto srcType = dyn_cast<MemRefType>(bridge->src.getType());
  if (!actType || !wtType || !srcType || srcType.getRank() != 2 ||
      !srcType.hasStaticShape() ||
      !(srcType.getElementType().isF16() || srcType.getElementType().isF32())) {
    op.emitRemark() << "HMX pipeline not applied: the activation staging "
                       "geometry does not describe rank-5 f16 crouton arrays "
                       "with a static row-major f16/f32 source";
    return false;
  }

  int64_t Mt = actType.getDimSize(0), Kt = actType.getDimSize(1);
  int64_t Nt = wtType.getDimSize(0);
  int64_t K = srcType.getDimSize(1);
  if (Mt < 1 || Kt < 1) {
    op.emitRemark() << "HMX pipeline not applied: the activation staging "
                       "geometry has an empty crouton grid";
    return false;
  }
  if (srcType.getDimSize(0) != Mt * crouton::kTileEdge ||
      K != Kt * crouton::kTileEdge || wtType.getDimSize(1) != Kt) {
    op.emitRemark() << "HMX pipeline not applied: the source, activation and "
                       "weight grids disagree on the staging tile shape";
    return false;
  }

  // `auto` declines a shallow-K shape: below `kStageMinKTiles` the transfer is
  // too small for the fixed DMA cost to hide behind the tile's compute, so the
  // overlap does not pay (the threshold's mechanism and the device numbers are
  // on the constant). An explicit `pipeline-depth=1/2` skips the floor -- those
  // are the A/B arms -- and `3` never reaches this emitter.
  if (requestedDepth <= 0 && Kt < kStageMinKTiles) {
    op.emitRemark() << "HMX pipeline not applied: Kt " << Kt
                    << " is below the staging floor of " << kStageMinKTiles
                    << " -- one transfer is too small to hide the DMA engine's "
                       "fixed cost behind the tile's compute";
    return false;
  }

  // The staged loop allocates, on top of the arrays the attribution already
  // counted: one crouton-row scratch (the pack destination, reused by every
  // tile) and `depth` staging slots (one 32 x K f16 tile each) plus `depth`
  // status words. The whole activation array is retired by this path, so its
  // bytes come back to the room -- not counting them would turn a ring that fits
  // into a spurious fallback.
  int64_t actBytes = actType.getNumElements() * 2;
  int64_t scratchBytes = Kt * crouton::kCroutonBytes;
  int64_t srcElemBytes = srcType.getElementTypeBitWidth() / 8;
  int64_t slotBytes = crouton::kTileEdge * K * srcElemBytes;
  int64_t statusBytes = 4;
  int64_t ringBytes = slotBytes + statusBytes;
  int64_t room = vtcmBudget - vtcmBytesCommitted(func) + actBytes;

  // The deepest ring the budget can pay for, `scratch + depth * ring`: 2, or 1
  // when only the serial ring fits. 0 means not even that fits, so the plain
  // tile loop runs. The K floor above is about profitability, not footprint:
  // what a shape pays for here is the ring, not a second copy of the whole
  // activation.
  auto fits = [&](int64_t d) { return scratchBytes + d * ringBytes <= room; };
  int64_t budgetDepth = fits(2) ? 2 : (fits(1) ? 1 : 0);
  if (budgetDepth == 0) {
    op.emitRemark() << "HMX pipeline not applied: activation staging needs "
                    << (scratchBytes + ringBytes)
                    << " bytes of VTCM (one crouton scratch plus the serial "
                       "ring), only "
                    << room << " are free";
    return false;
  }

  // The knob selects a depth, then the budget caps it. `requestedDepth` <= 0 is
  // auto; anything above 2 is clamped to the deepest ring that exists.
  int64_t depth = requestedDepth <= 0 ? budgetDepth
                                      : std::min<int64_t>(requestedDepth, 2);
  if (depth > budgetDepth) {
    op.emitRemark() << "HMX pipeline depth " << depth
                    << " requested, but only " << room
                    << " bytes of VTCM are free after the crouton scratch: "
                    << "a depth-" << budgetDepth << " ring needs "
                    << (scratchBytes + budgetDepth * ringBytes)
                    << " bytes; using depth " << budgetDepth;
    depth = budgetDepth;
  }
  // A ring at least as deep as the tile count has no steady state worth a
  // second slot -- the extra transfer would overlap nothing -- so geometry caps
  // it too. This is a shape property, not a budget one, so it is only a debug
  // note. (The pipeliner's prologue could not stage past the end anyway: it
  // issues exactly one tile per pipeline stage, and the kernel stops that many
  // iterations short.)
  if (Mt <= depth) {
    LLVM_DEBUG(llvm::dbgs()
               << "hmx-partition: ring depth capped by tile count Mt=" << Mt
               << " (wanted " << depth << ")\n");
    depth = 1;
  }
  // Auto narrowing from the double ring to the serial one is worth a remark too:
  // the pipeline still applies, but the overlap it was chosen for is gone.
  if (requestedDepth <= 0 && budgetDepth == 1 && Mt > 1)
    op.emitRemark() << "HMX pipeline not applied at depth 2: double-buffered "
                       "activation staging needs "
                    << (scratchBytes + 2 * ringBytes)
                    << " bytes of VTCM, only " << room
                    << " are free; using the serial ring";
  LLVM_DEBUG(llvm::dbgs()
             << "hmx-partition: activation staging depth " << depth
             << " (requested " << requestedDepth << ", budget depth "
             << budgetDepth << ", room " << room << " B, scratch "
             << scratchBytes << " B, ring/depth " << ringBytes << " B)\n");

  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(op);
  Location loc = opLoc;

  auto slotType = MemRefType::get({crouton::kTileEdge, K},
                                  srcType.getElementType(), AffineMap{},
                                  hexagon::VTCM_ADDRESS_SPACE);
  auto statusType = MemRefType::get({1}, rewriter.getI32Type());
  // One crouton row: the pack destination, re-filled (not re-allocated) every
  // tile. It replaces the whole activation array, so the staged path pays one
  // row of croutons instead of `Mt` of them. The pack's DPS result is this same
  // buffer, so the mma consumes the packed croutons by address -- no copy
  // between pack and mma, and no second row in flight.
  auto scratchType =
      MemRefType::get({1, Kt, crouton::kCroutonPair, crouton::kCroutonCol,
                       crouton::kCroutonHalf},
                      rewriter.getF16Type(), AffineMap{},
                      hexagon::VTCM_ADDRESS_SPACE);

  // The static ring: `depth` (slot, status) pairs, one per in-flight tile, plus
  // the one scratch. All allocated once here and released once after the
  // epilogue; the loop body allocates nothing.
  Value scratch =
      memref::AllocOp::create(rewriter, loc, scratchType, ValueRange{});
  SmallVector<Value> slots, statuses;
  for (int64_t i = 0; i < depth; ++i) {
    slots.push_back(memref::AllocOp::create(rewriter, loc, slotType, ValueRange{},
                                            rewriter.getI64IntegerAttr(128)));
    statuses.push_back(memref::AllocOp::create(
        rewriter, loc, statusType, ValueRange{}, rewriter.getI64IntegerAttr(4)));
  }

  auto c0 = arith::ConstantIndexOp::create(rewriter, loc, 0);
  auto c1 = arith::ConstantIndexOp::create(rewriter, loc, 1);
  auto cTileEdge = arith::ConstantIndexOp::create(rewriter, loc, crouton::kTileEdge);
  auto cKt = arith::ConstantIndexOp::create(rewriter, loc, Kt);
  auto cNt = arith::ConstantIndexOp::create(rewriter, loc, Nt);
  auto cMt = arith::ConstantIndexOp::create(rewriter, loc, Mt);

  // The serial source loop: iteration m issues tile m's transfer, awaits it and
  // computes it. Row is an element row offset, so tile m begins at source row
  // m * 32. On its own this is the depth-1 staged loop -- issue and await
  // adjacent, nothing in flight during the compute -- and at depth 1 it is
  // emitted as-is. At depth 2 it is the schedule input of the SCF pipeliner,
  // which splits it into "issue" (stage 0) and "await + compute" (stage 1) and
  // generates the prologue, the steady kernel and the peeled epilogue itself.
  //
  // The slot of tile m is slot(m mod 2) at depth 2, picked by a parity select
  // over the tile index. The select is a stage-0 op: the pipeliner re-evaluates
  // it with the shifted induction variable in the kernel's issue part (tile
  // m+1's transfer lands in slot((m+1) mod 2), never in the slot tile m is
  // being read from) and versions its result across stages -- the alternation
  // of the static slots is carried by the pipeliner's iter_args, not by
  // hand-written rotation. The status select needs no versioning: only the
  // stage op reads it, in the same stage.
  auto mLoop = scf::ForOp::create(rewriter, loc, c0, cMt, c1, ValueRange{});
  Block *mBody = mLoop.getBody();
  rewriter.setInsertionPointToStart(mBody);
  Value m = mLoop.getInductionVar();

  Value slotSel = slots[0];
  Value statusSel = statuses[0];
  if (depth == 2) {
    auto parity = arith::AndIOp::create(rewriter, loc, m, c1);
    auto odd =
        arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::ne, parity,
                              c0);
    slotSel = arith::SelectOp::create(rewriter, loc, odd, slots[1], slots[0])
                  .getResult();
    statusSel =
        arith::SelectOp::create(rewriter, loc, odd, statuses[1], statuses[0])
            .getResult();
  }

  // Issue -> await: the pack reads the slot the await handed back, so the
  // transfer -> compute dependency is an SSA edge, not an effect -- and the
  // token is `hmx.stage`'s result, which is exactly what the pipeliner versions
  // across the two stages.
  Value row = arith::MulIOp::create(rewriter, loc, m, cTileEdge);
  Value token = emitStage(rewriter, loc, bridge->src, row, slotSel, statusSel);
  Value ready = emitAwait(rewriter, loc, token, slotSel);
  emitTileCompute(rewriter, loc, bias, scratch, wt, ar, m, ready, c0, c1, cKt,
                  cNt);

  // The pipeliner owns the schedule from here: everything through the
  // `hmx.stage` is stage 0, the await and the compute are stage 1. With the
  // epilogue peeled it emits the prologue (the issue of tile 0), the kernel
  // `issue(m+1) await(m) compute(m)` over tiles 0 .. Mt-2, and an epilogue that
  // only awaits and computes tile Mt-1 -- the same DMA/await/compute order the
  // hand-written ring produced, with the (token, slot) rotation done by value
  // versioning. A static-bound loop is never predicated, so a failure here can
  // only happen before the pipeliner touches the IR: the source loop above is
  // intact and serially correct (issue -> await -> compute), so it stands as
  // the serial staged loop.
  if (depth == 2) {
    Operation *stageOp = token.getDefiningOp();
    scf::PipeliningOption options;
    options.peelEpilogue = true;
    options.getScheduleFn =
        [stageOp](scf::ForOp forOp,
                  std::vector<std::pair<Operation *, unsigned>> &schedule) {
          bool issue = true;
          for (Operation &bodyOp : forOp.getBody()->without_terminator()) {
            schedule.emplace_back(&bodyOp, issue ? 0u : 1u);
            if (&bodyOp == stageOp)
              issue = false;
          }
        };
    // The pipeliner emits the prologue and the kernel at the rewriter's
    // current insertion point, so it must be before the loop, not inside the
    // body this emitter just filled.
    rewriter.setInsertionPoint(mLoop);
    bool modifiedIR = false;
    if (failed(scf::pipelineForLoop(rewriter, mLoop, options, &modifiedIR))) {
      assert(!modifiedIR && "pipelining failed after rewriting the IR");
      // The source loop is intact: keep it as the serial staged loop, with
      // everything after it following the loop as usual.
      rewriter.setInsertionPointAfter(mLoop);
      op.emitRemark() << "HMX pipeline not applied at depth 2: the SCF loop "
                         "pipeliner declined the schedule; the staged loop "
                         "runs serially";
    }
  } else {
    // Depth 1 is the source loop as emitted: move the insertion point back out
    // of the body the compute left it in, so what follows follows the loop.
    rewriter.setInsertionPointAfter(mLoop);
  }

  // The ring and the scratch live exactly as long as the tile loop. The
  // insertion point is after the pipelined epilogue (or after the serial loop
  // at depth 1), so neither is freed before its last reader.
  for (Value slot : slots)
    memref::DeallocOp::create(rewriter, loc, slot);
  for (Value status : statuses)
    memref::DeallocOp::create(rewriter, loc, status);
  memref::DeallocOp::create(rewriter, loc, scratch);

  // Retire the bridge, the matmul and the activation array. The array is dead in
  // this path: the tile loop writes the scratch instead, so neither the bridge
  // loop that filled it nor the array itself survives. Collect the array's
  // deallocations before its bridge loop is erased -- in the carried form the
  // loop is what defines the array -- then drop the bridge and the matmul, then
  // the deallocations, and finally the allocation underneath. The earlier reader
  // check guarantees nothing else observes the array.
  SmallVector<memref::DeallocOp> actDeallocs;
  for (Operation *user : act.getUsers())
    if (auto d = dyn_cast<memref::DeallocOp>(user))
      actDeallocs.push_back(d);

  rewriter.eraseOp(op);
  if (bridge->loop) {
    rewriter.eraseOp(bridge->loop);
  } else {
    for (PackActOp p : bridge->packs)
      rewriter.eraseOp(p);
  }
  for (memref::DeallocOp d : actDeallocs)
    rewriter.eraseOp(d);

  // The allocation is `act` itself in the canonicalized form and the bridge
  // loop's init arg in the carried one; either way it is unreferenced now. Drop
  // its deallocation (if the array value was not the thing deallocated) and then
  // the allocation, so the activation stops occupying VTCM.
  if (auto alloc = bridge->buffer.getDefiningOp<memref::AllocOp>()) {
    SmallVector<memref::DeallocOp> bufferDeallocs;
    for (Operation *user : bridge->buffer.getUsers())
      if (auto d = dyn_cast<memref::DeallocOp>(user))
        bufferDeallocs.push_back(d);
    for (memref::DeallocOp d : bufferDeallocs)
      rewriter.eraseOp(d);
    if (alloc->use_empty())
      rewriter.eraseOp(alloc);
  }
  return true;
}

struct HmxPartitionPass
    : public mlir::hmx::impl::HmxPartitionBase<HmxPartitionPass> {
  using HmxPartitionBase::HmxPartitionBase;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<HmxDialect, arith::ArithDialect, memref::MemRefDialect,
                    scf::SCFDialect>();
  }

  void runOnOperation() override {
    func::FuncOp func = cast<func::FuncOp>(getOperation());

    SmallVector<MatmulOp> matmuls;
    func.walk([&](MatmulOp op) { matmuls.push_back(op); });
    if (matmuls.empty())
      return;

    // The engine's budget, with the one field a caller may narrow: 0 means the
    // device default (see HmxTarget).
    const int64_t vtcmBudget =
        this->vtcmBudgetBytes > 0 ? this->vtcmBudgetBytes
                                  : HmxTarget::defaultVtcmBudget;

    Location loc = func.getLoc();
    IRRewriter rewriter(func.getContext());
    rewriter.setInsertionPointToStart(&func.getBody().front());

    // The identity conversion state is kernel-level setup, so it is created once
    // for the whole function. It is an ordinary space-1 block: the VTCM machinery
    // below places it next to the crouton arrays.
    auto biasType = MemRefType::get({kConvStateBytes}, rewriter.getI8Type(),
                                    AffineMap{}, hexagon::VTCM_ADDRESS_SPACE);
    auto bias = memref::AllocOp::create(
        rewriter, loc, biasType, ValueRange{},
        rewriter.getI64IntegerAttr(kConvStateAlignment));
    BiasInitOp::create(rewriter, loc, bias);

    for (MatmulOp op : matmuls) {
      auto shape = getTileShape(op);
      if (!shape) {
        op.emitError("hmx.matmul must operate on 5D crouton arrays here");
        return signalPassFailure();
      }
      // Single-op grid self-consistency: each dot keeps its own grid -- grids
      // are never unified across dots. A mismatch is a loud failure, never a
      // silently wrong loop nest. (The op verifier checks the same contract on
      // construction; this is the lowering half, guarding whatever reaches
      // this pass.)
      auto lhs = cast<MemRefType>(op.getLhs().getType());
      auto rhs = cast<MemRefType>(op.getRhs().getType());
      auto out = cast<MemRefType>(op.getOuts().getType());
      if (lhs.getDimSize(1) != rhs.getDimSize(1) ||
          out.getDimSize(0) != lhs.getDimSize(0) ||
          out.getDimSize(1) != rhs.getDimSize(0)) {
        op.emitError("hmx.matmul tile grids disagree: lhs [")
            << lhs.getDimSize(0) << ", " << lhs.getDimSize(1) << "] rhs ["
            << rhs.getDimSize(0) << ", " << rhs.getDimSize(1) << "] out ["
            << out.getDimSize(0) << ", " << out.getDimSize(1) << "]";
        return signalPassFailure();
      }

      // The matmul itself is replaced by the loop that walks its crouton arrays.
      Location opLoc = op.getLoc();
      // `pipeline-depth=3` is the unstaged arm: skip the staging rewrite
      // outright and emit the plain tile loop, which leaves the activation
      // bridge and its array in place. It is chosen here rather than by asking
      // the staged emitter to decline, so a bridge shape is preserved -- letting
      // the staged loop run and then undoing it would already have retired
      // `act`.
      if (this->pipelineDepth != kSerialPipelineDepth &&
          emitStageLoop(rewriter, opLoc, bias, op, func, vtcmBudget,
                        this->pipelineDepth))
        continue;
      emitSerialTileLoop(rewriter, opLoc, bias, op, *shape);
      rewriter.eraseOp(op);
    }

    // The conversion-state block is kernel-level setup: it is filled once at
    // entry and read by every tile's read-out, so it is released on the way out,
    // after the last tile loop that reads it -- every exit gets its own
    // deallocation, and the entry block defines the buffer so it dominates all
    // of them. The tile loops were inserted before their `hmx.matmul`, which the
    // erase above removed, so no reader follows a return. Without this
    // deallocation the 256 B state leaks on every launch -- it is the one VTCM
    // allocation this pass makes with no partner -- and the runtime's pool never
    // gets it back.
    SmallVector<func::ReturnOp> returns;
    func.walk([&](func::ReturnOp ret) { returns.push_back(ret); });
    for (func::ReturnOp ret : returns) {
      rewriter.setInsertionPoint(ret);
      memref::DeallocOp::create(rewriter, loc, bias);
    }
  }
};

} // namespace

std::unique_ptr<InterfacePass<FunctionOpInterface>>
mlir::hmx::createHmxPartitionPass(
    const mlir::hmx::HmxPartitionOptions &options) {
  return std::make_unique<HmxPartitionPass>(options);
}
