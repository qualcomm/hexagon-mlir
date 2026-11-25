//===--- VTCMMaskWidening.cpp - widen i1 masks bound for VTCM --------====//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
// This pass must run immediately before VTCMTilingPass, with no other pass
// running in between. A tensor<...xi1> mask that VTCMTilingPass stages into
// a VTCM/DDR scratchpad copy can end up read through two incompatible
// in-memory conventions by different consumers: a vectorized access
// (`vector.transfer_write`/`read` of `vector<Nxi1>`), which
// ConvertVectorToLLVM lowers to a bit-packed store/load (ceil(N/8) bytes
// total), versus a scalarized access (the linalg.generic+scf.if pattern used
// for data-dependent/gather-indexed masked loads), which lowers to one full
// byte per element. When the same buffer is written through one convention
// and read through the other, most lanes read garbage.
//
// VTCMTilingPass decides which operands get staged into VTCM/DDR via its own
// per-op call to getVTCMTilingOptions (the `prefetch` output). This pass
// replicates that exact same query -- on the identical IR, since nothing
// runs between this pass and VTCMTilingPass -- to find the same operands,
// and for any of them with an i1 element type, retypes it (and its
// producer/consumers) to i8 in place before VTCMTilingPass ever runs. By the
// time VTCMTilingPass's own selection query fires, there is no i1-typed
// operand left for it to stage, so it never needs any i1-specific handling
// of its own.
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

#include "llvm/Support/Debug.h"

#include "hexagon/Conversion/LinalgToLLVM/LinalgToLLVM.h"
#include "hexagon/Conversion/LinalgToLLVM/VTCMTilingOptions.h"
#include "hexagon/Transforms/OptionsParsing.h"

#define DEBUG_TYPE "vtcm-mask-widening"
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define DBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
using namespace mlir::linalg;
using namespace hexagon;

#define GEN_PASS_DEF_VTCMMASKWIDENING
#include "hexagon/Conversion/LinalgToLLVM/Passes.h.inc"

namespace {

struct VTCMMaskWideningPass
    : public ::impl::VTCMMaskWideningBase<VTCMMaskWideningPass> {
  explicit VTCMMaskWideningPass(const VTCMMaskWideningOptions &options)
      : Base(options) {}
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect>();
  }
  void runOnOperation() override;
};

/// Returns true if the outs operand at the given DPS init index is write-only.
bool isOutsWriteOnly(linalg::GenericOp op, unsigned outsIdx) {
  return op.getRegionOutputArgs()[outsIdx].use_empty();
}

/// If `type` has an `i1` element type, returns the equivalent tensor with an
/// `i8` element type; otherwise returns `type` unchanged.
RankedTensorType getByteBackedType(RankedTensorType type) {
  if (!type.getElementType().isInteger(1))
    return type;
  return type.clone(IntegerType::get(type.getContext(), 8));
}

/// Elementwise-casts `input` to `resultType` via a linalg.generic.
///
/// A bare `arith.extui`/`arith.trunci` on a tensor operand has no
/// bufferization pattern registered with OneShotBufferize and fails to
/// bufferize ("op was not bufferized"); wrapping the cast in a
/// linalg.generic makes it bufferize through linalg's normal
/// DestinationStyleOpInterface support.
Value castElementwiseTensor(IRRewriter &rewriter, Value input,
                            RankedTensorType resultType, Location loc) {
  auto rank = resultType.getRank();
  Value init = tensor::EmptyOp::create(rewriter, loc, resultType.getShape(),
                                       resultType.getElementType());
  AffineMap identityMap = rewriter.getMultiDimIdentityMap(rank);
  SmallVector<utils::IteratorType> iteratorTypes(rank,
                                                 utils::IteratorType::parallel);
  unsigned dstBits = resultType.getElementTypeBitWidth();
  unsigned srcBits =
      ::llvm::cast<RankedTensorType>(input.getType()).getElementTypeBitWidth();
  auto genericOp = GenericOp::create(
      rewriter, loc, TypeRange{resultType}, ValueRange{input}, ValueRange{init},
      ArrayRef<AffineMap>{identityMap, identityMap},
      ArrayRef<utils::IteratorType>(iteratorTypes),
      [&](OpBuilder &b, Location nestedLoc, ValueRange args) {
        Value casted =
            dstBits > srcBits
                ? arith::ExtUIOp::create(b, nestedLoc,
                                         resultType.getElementType(), args[0])
                      .getResult()
                : arith::TruncIOp::create(b, nestedLoc,
                                          resultType.getElementType(), args[0])
                      .getResult();
        linalg::YieldOp::create(b, nestedLoc, casted);
      });
  return genericOp.getResult(0);
}

/// Widens an i1-element tensor value to i8 (identity for any other type).
Value widenToByte(IRRewriter &rewriter, Value rankedTensor, Location loc) {
  auto type = ::llvm::cast<RankedTensorType>(rankedTensor.getType());
  auto byteType = getByteBackedType(type);
  if (byteType == type)
    return rankedTensor;
  return castElementwiseTensor(rewriter, rankedTensor, byteType, loc);
}

/// For each `outs` operand whose element type differs from the current init
/// block argument's type, patch `op`'s region in place so it stays
/// well-formed under the new type: update the block argument's type and
/// insert an `arith.extui` converting the yielded value to the new element
/// type.
///
/// Only reachable for write-only outs, so the init block argument has no
/// uses and simply retyping it is safe.
void widenGenericOutsInPlace(IRRewriter &rewriter, GenericOp op,
                             ArrayRef<Value> newOuts) {
  auto yieldOp = cast<linalg::YieldOp>(op.getBody()->getTerminator());
  for (auto [i, newOut] : llvm::enumerate(newOuts)) {
    auto newElemType =
        ::llvm::cast<RankedTensorType>(newOut.getType()).getElementType();
    BlockArgument arg = op.getRegionOutputArgs()[i];
    if (arg.getType() == newElemType)
      continue;
    assert(arg.use_empty() &&
           "widening a read outs argument would also require narrowing it "
           "on every use inside the region");
    arg.setType(newElemType);
    rewriter.setInsertionPoint(yieldOp);
    Value widened = arith::ExtUIOp::create(rewriter, yieldOp.getLoc(),
                                           newElemType, yieldOp.getOperand(i));
    yieldOp.setOperand(i, widened);
  }
}

/// For each `ins` operand whose element type differs from the current input
/// block argument's type, patch `op`'s region in place: retype the block
/// argument to the new (wider) element type and insert a cast back to the
/// original element type immediately after it, redirecting the block
/// argument's original uses to that cast's result.
///
/// This keeps the narrowing cast a transient scalar value consumed
/// immediately inside the body -- never its own materialized tensor -- so it
/// can't independently diverge between a vectorized write and a scalarized
/// read the way a standalone narrowed tensor would (see getByteBackedType).
void widenGenericInsInPlace(IRRewriter &rewriter, GenericOp op,
                            ArrayRef<Value> newIns) {
  for (auto [i, newIn] : llvm::enumerate(newIns)) {
    auto newElemType =
        ::llvm::cast<RankedTensorType>(newIn.getType()).getElementType();
    BlockArgument arg = op.getRegionInputArgs()[i];
    Type oldElemType = arg.getType();
    if (oldElemType == newElemType)
      continue;
    arg.setType(newElemType);
    rewriter.setInsertionPointToStart(op.getBody());
    bool widening = newElemType.getIntOrFloatBitWidth() >
                    oldElemType.getIntOrFloatBitWidth();
    Value restored =
        widening
            ? arith::TruncIOp::create(rewriter, op.getLoc(), oldElemType, arg)
                  .getResult()
            : arith::ExtUIOp::create(rewriter, op.getLoc(), oldElemType, arg)
                  .getResult();
    rewriter.replaceAllUsesExcept(arg, restored, restored.getDefiningOp());
  }
}

/// A tensor.extract's result type is fixed to its source tensor's element
/// type at the point it is created. If a later RAUW widens that source
/// tensor's element type (e.g. i1 -> i8, see widenGenericInsInPlace) -- which
/// happens here when the extract sits in a loop body that
/// ConvertLinalgToLoopsPass already scalarized out of a linalg.generic before
/// this pass ever widened that generic's operand -- the extract's declared
/// result type goes stale relative to its (now-widened) operand, and the
/// verifier rejects it. Patch it the same way widenGenericInsInPlace patches
/// a stale block argument: retype the extract's result to the widened
/// element type, then insert a cast back to the original element type
/// immediately after, redirecting the extract's original uses to that cast's
/// result.
void widenTensorExtractInPlace(IRRewriter &rewriter, tensor::ExtractOp op) {
  Type newElemType =
      ::llvm::cast<RankedTensorType>(op.getTensor().getType()).getElementType();
  Type oldElemType = op.getType();
  if (oldElemType == newElemType)
    return;
  Value result = op.getResult();
  result.setType(newElemType);
  rewriter.setInsertionPointAfter(op);
  bool widening =
      newElemType.getIntOrFloatBitWidth() > oldElemType.getIntOrFloatBitWidth();
  Value restored =
      widening
          ? arith::TruncIOp::create(rewriter, op.getLoc(), oldElemType, result)
                .getResult()
          : arith::ExtUIOp::create(rewriter, op.getLoc(), oldElemType, result)
                .getResult();
  rewriter.replaceAllUsesExcept(result, restored, restored.getDefiningOp());
}

/// A `tensor.expand_shape`/`tensor.collapse_shape` op's own declared result
/// element type is fixed at the point it is created and is never itself an
/// `ins`/`outs` operand of a `linalg.generic` or the tensor of a
/// `tensor.extract` -- so neither of the widening helpers above ever touches
/// it. If its source operand's element type is later widened (e.g. i1 -> i8,
/// because the op that produced the source is a `linalg.generic` whose
/// result this same pass retyped via `widenGenericOutsInPlace`), the
/// reshape's declared result element type goes stale relative to its (now
/// widened) source, and the verifier rejects it: reshaping changes shape,
/// never element type, so source and result element types must always
/// match. Unlike a `linalg.generic`'s region block argument or a
/// `tensor.extract`'s declared result type, a reshape op has no region body
/// and no separate "restore to the original type" consumer to patch --
/// simply retyping its result in place (same shape, new element type) is
/// sufficient; any `linalg.generic` that later consumes this (now correctly
/// widened) result is reconciled by the ordinary `widenGenericInsInPlace`
/// check at the top of the main walk below, exactly as if that generic had
/// consumed an already-i8 value all along.
///
/// Must be visited in the same walk, in program order, as the
/// `linalg.generic` handling below -- not a separate post-pass sweep like
/// `widenTensorExtractInPlace` -- so that a reshape op's result is corrected
/// before any `linalg.generic` consuming it is reached and reconciled.
void widenTensorReshapeInPlace(Operation *op) {
  Value src = op->getOperand(0);
  Value result = op->getResult(0);
  auto srcType = cast<RankedTensorType>(src.getType());
  auto resultType = cast<RankedTensorType>(result.getType());
  if (srcType.getElementType() == resultType.getElementType())
    return;
  result.setType(resultType.clone(srcType.getElementType()));
}

/// For every operand of `op` whose tensor element type is i1, retype it to
/// i8 in place: an ins operand gets a widening cast substituted in its slot,
/// a write-only outs operand gets a fresh i8-typed tensor.empty substituted
/// in its slot (mirroring what VTCMTilingPass's own
/// allocOnMemorySpace/copyToMemorySpace would have done, minus the VTCM/DDR
/// copy itself -- that stays VTCMTilingPass's job).
/// `op`'s region and, for outs, its own declared result type are then
/// reconciled so every existing use observes the new type without any RAUW.
///
/// This applies to *every* operand of a VTCM-tiling-selected `op`, not just
/// ones with `prefetch[idx] == true`: VTCMTilingPass's own
/// replaceTiledGenericWithVTCMSlices later stages every `prefetch[idx] ==
/// false` operand into VTCM too, one tile at a time, via the same
/// copyVTCM/allocOnVTCM helpers -- so every operand of a selected op ends up
/// VTCM-resident one way or another, and every i1 one of them needs the same
/// widening. The caller only calls this once it has confirmed `op` was
/// selected at all (via a successful getVTCMTilingOptions query); this
/// function itself does not consult the per-operand prefetch flags.
void widenSelectedOperandsInPlace(IRRewriter &rewriter, GenericOp op) {
  bool changedOuts = false;
  for (OpOperand &opOperand : op->getOpOperands()) {
    unsigned idx = opOperand.getOperandNumber();
    auto type = dyn_cast<RankedTensorType>(opOperand.get().getType());
    if (!type || getByteBackedType(type) == type)
      continue;

    // Any new op materializing the widened value (a casting linalg.generic
    // for ins, or nothing at all for a fresh tensor.empty outs) must be
    // inserted where its own operand -- opOperand.get(), already dominating
    // `op` since it is one of `op`'s current operands -- still dominates it.
    // Immediately before `op` itself is always such a point.
    rewriter.setInsertionPoint(op);

    if (op.isDpsInit(&opOperand)) {
      if (!isOutsWriteOnly(op, idx - op.getNumDpsInputs()))
        continue;
      Value widened =
          tensor::EmptyOp::create(rewriter, op.getLoc(), type.getShape(),
                                  IntegerType::get(op.getContext(), 8));
      opOperand.set(widened);
      changedOuts = true;
    } else {
      opOperand.set(widenToByte(rewriter, opOperand.get(), op.getLoc()));
    }
  }

  SmallVector<Value> currentIns, currentOuts;
  for (OpOperand &opOperand : op->getOpOperands()) {
    Value v = opOperand.get();
    op.isDpsInit(&opOperand) ? currentOuts.push_back(v)
                             : currentIns.push_back(v);
  }
  widenGenericInsInPlace(rewriter, op, currentIns);
  widenGenericOutsInPlace(rewriter, op, currentOuts);

  if (changedOuts)
    for (auto [i, newOut] : llvm::enumerate(currentOuts))
      op->getResult(i).setType(newOut.getType());
}

void VTCMMaskWideningPass::runOnOperation() {
  auto userProvidedTileSizes = parseTileSizes(tileSizes);
  auto funcOp = getOperation();

  // tensor.expand_shape/collapse_shape are handled inline in this same walk
  // (not a separate post-pass sweep) so that a reshape's result is widened
  // in program order, before any linalg.generic consuming it is reached
  // below -- see widenTensorReshapeInPlace's comment.
  funcOp.walk([&](Operation *rawOp) {
    if (isa<tensor::ExpandShapeOp, tensor::CollapseShapeOp>(rawOp)) {
      widenTensorReshapeInPlace(rawOp);
      return WalkResult::advance();
    }

    auto op = dyn_cast<linalg::GenericOp>(rawOp);
    if (!op)
      return WalkResult::advance();

    // This pass only widens tensor-level i1 operands ahead of VTCMTilingPass;
    // buffer-semantics generics (already bufferized, e.g. hand-written
    // memref-level kernels) never reach VTCMTilingPass's own selection query
    // (see getVTCMTilingOptions's identical hasPureTensorSemantics guard) and
    // have no tensor-typed operands for widenGenericInsInPlace/
    // widenGenericOutsInPlace's unchecked cast<RankedTensorType> to act on.
    if (!op.hasPureTensorSemantics())
      return WalkResult::advance();

    IRRewriter rewriter(op.getContext());

    // An earlier iteration of this walk may have already widened one of
    // this op's operands (e.g. i1 -> i8) via the setType/setOperand
    // mutation in widenSelectedOperandsInPlace, when rewriting whichever op
    // produced it. Reconcile against this op's *current* operands
    // unconditionally, before making any new widening decision, so every
    // consumer of a widened value stays well-formed -- not just ones this
    // pass chooses to widen further below.
    {
      SmallVector<Value> currentIns, currentOuts;
      for (OpOperand &opOperand : op->getOpOperands()) {
        Value v = opOperand.get();
        op.isDpsInit(&opOperand) ? currentOuts.push_back(v)
                                 : currentIns.push_back(v);
      }
      widenGenericInsInPlace(rewriter, op, currentIns);
      widenGenericOutsInPlace(rewriter, op, currentOuts);
    }

    // Skip generics already enclosed by a tiled scf.for annotated with
    // `tiled_generic` (e.g. produced by PromoteTiledSCFLoopToVTCMPass) --
    // mirrors VTCMTilingPass's own skip exactly, so both passes agree on which
    // ops are even considered.
    for (Operation *parent = op->getParentOp(); parent;
         parent = parent->getParentOp()) {
      if (isa<scf::ForOp>(parent) && parent->hasAttr("tiled_generic"))
        return WalkResult::advance();
    }

    // Same query, same arguments, as the one VTCMTilingPass is about to run
    // moments later on this identical IR -- guaranteed to select the same
    // operands.
    SmallVector<bool> prefetch(op.getNumOperands(), false);
    FailureOr<linalg::LinalgTilingOptions> vtcmTilingOptions =
        getVTCMTilingOptions(op, userProvidedTileSizes, prefetch, vtcmBudget);
    if (failed(vtcmTilingOptions))
      return WalkResult::advance();

    widenSelectedOperandsInPlace(rewriter, op);
    return WalkResult::advance();
  });

  // The walk above only reconciles linalg.generic consumers of a widened
  // value. A widened value can also flow into a bare tensor.extract left
  // over from an earlier ConvertLinalgToLoopsPass scalarization. Sweep
  // those separately, once all of the above widening has already happened.
  funcOp.walk([&](tensor::ExtractOp op) {
    IRRewriter rewriter(op.getContext());
    widenTensorExtractInPlace(rewriter, op);
  });
}
} // namespace

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
hexagon::createVTCMMaskWideningPass(const VTCMMaskWideningOptions &options) {
  return std::make_unique<VTCMMaskWideningPass>(options);
}
