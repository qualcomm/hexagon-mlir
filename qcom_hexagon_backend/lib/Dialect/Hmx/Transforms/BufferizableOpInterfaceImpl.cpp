//===- BufferizableOpInterfaceImpl.cpp - Impl. of BufferizableOpInterface -=//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// `hmx.matmul` is a destination-style op, so one-shot bufferization can handle it
// once it knows which operands are the inits. Without this the op would simply
// not bufferize and the tile loop would never see a memref form.
//
// The buffer form of the pack/unpack ops is created with its destination as a
// result (`TypeRange{*dstBuffer}`), not with an empty `TypeRange`. The written
// buffer has to be an SSA value so a consumer or an `scf.yield` depends on the
// producer through a value, which is what lets an external scheduler see the
// dependency (docs/hmx/hmx-scheduling-interface-plan.md section 9.3). The result
// is the same buffer as the `dst` operand, so the lowering emits no copy.
//
// `hmx.stage`/`hmx.await` take and produce memrefs only, so they have no tensor
// form to bufferize and no model is needed here.
//
// `hmx.alloc_crouton` is not destination-style: it models upstream's
// `bufferization.alloc_tensor` (its method set is that op's), because it *is*
// the crouton-array allocation -- the op whose `getBufferType` applies the
// `#hmx.crouton` -> `#hmx.crouton_memref_layout` mapping the stock
// `alloc_tensor` never could (hmx-interface-gaps.md section 2.2).
//
// The shape follows HexKL's implementation of the same interface.
//===----------------------------------------------------------------------===//

#include "hexagon/Common/Common.h"
#include "hexagon/Dialect/Hmx/Transforms/BufferizableOpInterfaceImpl.h"
#include "hexagon/Dialect/Hmx/IR/HmxDialect.h"

#include "mlir/Dialect/Bufferization/IR/DstBufferizableOpInterfaceImpl.h"

using namespace mlir;
using namespace mlir::hmx;
using namespace mlir::bufferization;

namespace {

struct MatmulOpInterface
    : public DstBufferizableOpInterfaceExternalModel<MatmulOpInterface,
                                                     MatmulOp> {
  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options,
                          BufferizationState &state) const {
    auto dstOp = cast<DestinationStyleOpInterface>(op);
    if (dstOp.hasPureBufferSemantics())
      return success();
    if (!dstOp.hasPureTensorSemantics())
      return op->emitError() << "op does not have pure tensor semantics";

    auto matmulOp = cast<MatmulOp>(op);
    FailureOr<Value> lhsBuffer =
        getBuffer(rewriter, matmulOp.getLhs(), options, state);
    if (failed(lhsBuffer))
      return failure();
    FailureOr<Value> rhsBuffer =
        getBuffer(rewriter, matmulOp.getRhs(), options, state);
    if (failed(rhsBuffer))
      return failure();
    FailureOr<Value> outBuffer =
        getBuffer(rewriter, matmulOp.getOuts(), options, state);
    if (failed(outBuffer))
      return failure();

    MatmulOp::create(rewriter, matmulOp.getLoc(), /*result=*/TypeRange(),
                     *lhsBuffer, *rhsBuffer, *outBuffer);
    replaceOpWithBufferizedValues(rewriter, op, *outBuffer);
    return success();
  }
};

struct PackActOpInterface
    : public DstBufferizableOpInterfaceExternalModel<PackActOpInterface,
                                                     PackActOp> {
  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options,
                          BufferizationState &state) const {
    auto dstOp = cast<DestinationStyleOpInterface>(op);
    if (dstOp.hasPureBufferSemantics())
      return success();
    if (!dstOp.hasPureTensorSemantics())
      return op->emitError() << "op does not have pure tensor semantics";

    auto packOp = cast<PackActOp>(op);
    FailureOr<Value> dstBuffer = getBuffer(rewriter, packOp.getDst(), options, state);
    if (failed(dstBuffer))
      return failure();
    FailureOr<Value> srcBuffer = getBuffer(rewriter, packOp.getSrc(), options, state);
    if (failed(srcBuffer))
      return failure();

    PackActOp::create(rewriter, packOp.getLoc(), /*result=*/TypeRange{*dstBuffer},
                      *dstBuffer, *srcBuffer, packOp.getRow(), packOp.getCol(),
                      packOp.getCountAttr());
    replaceOpWithBufferizedValues(rewriter, op, *dstBuffer);
    return success();
  }
};

struct PackWeightOpInterface
    : public DstBufferizableOpInterfaceExternalModel<PackWeightOpInterface,
                                                     PackWeightOp> {
  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options,
                          BufferizationState &state) const {
    auto dstOp = cast<DestinationStyleOpInterface>(op);
    if (dstOp.hasPureBufferSemantics())
      return success();
    if (!dstOp.hasPureTensorSemantics())
      return op->emitError() << "op does not have pure tensor semantics";

    auto packOp = cast<PackWeightOp>(op);
    FailureOr<Value> dstBuffer = getBuffer(rewriter, packOp.getDst(), options, state);
    if (failed(dstBuffer))
      return failure();
    FailureOr<Value> srcBuffer = getBuffer(rewriter, packOp.getSrc(), options, state);
    if (failed(srcBuffer))
      return failure();

    PackWeightOp::create(rewriter, packOp.getLoc(), /*result=*/TypeRange{*dstBuffer},
                         *dstBuffer, *srcBuffer, packOp.getKTile(),
                         packOp.getNTile(), packOp.getCountAttr());
    replaceOpWithBufferizedValues(rewriter, op, *dstBuffer);
    return success();
  }
};

struct UnpackAccOpInterface
    : public DstBufferizableOpInterfaceExternalModel<UnpackAccOpInterface,
                                                     UnpackAccOp> {
  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options,
                          BufferizationState &state) const {
    auto dstOp = cast<DestinationStyleOpInterface>(op);
    if (dstOp.hasPureBufferSemantics())
      return success();
    if (!dstOp.hasPureTensorSemantics())
      return op->emitError() << "op does not have pure tensor semantics";

    auto unpackOp = cast<UnpackAccOp>(op);
    FailureOr<Value> srcBuffer = getBuffer(rewriter, unpackOp.getSrc(), options, state);
    if (failed(srcBuffer))
      return failure();
    FailureOr<Value> dstBuffer = getBuffer(rewriter, unpackOp.getDst(), options, state);
    if (failed(dstBuffer))
      return failure();

    UnpackAccOp::create(rewriter, unpackOp.getLoc(), /*result=*/TypeRange{*dstBuffer},
                        *srcBuffer, *dstBuffer, unpackOp.getRow(),
                        unpackOp.getCol(), unpackOp.getCountAttr());
    replaceOpWithBufferizedValues(rewriter, op, *dstBuffer);
    return success();
  }
};

struct UnpackAccF32OpInterface
    : public DstBufferizableOpInterfaceExternalModel<UnpackAccF32OpInterface,
                                                     UnpackAccF32Op> {
  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options,
                          BufferizationState &state) const {
    auto dstOp = cast<DestinationStyleOpInterface>(op);
    if (dstOp.hasPureBufferSemantics())
      return success();
    if (!dstOp.hasPureTensorSemantics())
      return op->emitError() << "op does not have pure tensor semantics";

    auto unpackOp = cast<UnpackAccF32Op>(op);
    FailureOr<Value> srcBuffer = getBuffer(rewriter, unpackOp.getSrc(), options, state);
    if (failed(srcBuffer))
      return failure();
    FailureOr<Value> dstBuffer = getBuffer(rewriter, unpackOp.getDst(), options, state);
    if (failed(dstBuffer))
      return failure();

    Value residual = unpackOp.getResidual();
    Value residualBuffer;
    if (residual) {
      FailureOr<Value> buf = getBuffer(rewriter, residual, options, state);
      if (failed(buf))
        return failure();
      residualBuffer = *buf;
    }

    UnpackAccF32Op::create(rewriter, unpackOp.getLoc(), /*result=*/TypeRange{*dstBuffer},
                           *srcBuffer, *dstBuffer, unpackOp.getRow(),
                           unpackOp.getCol(), residualBuffer,
                           unpackOp.getCountAttr());
    replaceOpWithBufferizedValues(rewriter, op, *dstBuffer);
    return success();
  }
};

/// The crouton-array allocation, modelled method-for-method on upstream's
/// `bufferization.alloc_tensor` (BufferizationOps.cpp) -- the same "new
/// allocation that aliases nothing" contract, with the buffer type decided
/// here instead of by stock inference:
///
///   * result carries `#hmx.crouton` -> a `memref<...,
///     #hmx.crouton_memref_layout<logical>, 1>`: the encoding survives
///     bufferization as the identity-map layout the memref side can carry;
///   * encoding erased (`drop-encodings`, the default) -> the same
///     static-identity, space-1 memref type `alloc_tensor` produces, so the
///     default pipeline's bufferized IR is byte-identical.
struct AllocCroutonOpInterface
    : public BufferizableOpInterface::ExternalModel<AllocCroutonOpInterface,
                                                    AllocCroutonOp> {
  bool bufferizesToAllocation(Operation *op, Value value) const { return true; }

  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const AnalysisState &state) const {
    // The op has no operands (unreachable with a tensor operand).
    return false;
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                               const AnalysisState &state) const {
    // The op has no operands (unreachable with a tensor operand).
    return false;
  }

  // This is a new allocation. It does not alias with any other buffer.
  AliasingValueList getAliasingValues(Operation *op, OpOperand &opOperand,
                                      const AnalysisState &state) const {
    return {};
  }

  // The array is freshly allocated and its contents undefined; nothing is
  // written by the allocation itself (same as upstream `alloc_tensor` without
  // a `copy`).
  bool resultBufferizesToMemoryWrite(Operation *op, OpResult opResult,
                                     const AnalysisState &state) const {
    return false;
  }

  bool isWritable(Operation *op, Value value,
                  const AnalysisState &state) const {
    return true;
  }

  FailureOr<BufferLikeType>
  getBufferType(Operation *op, Value value, const BufferizationOptions &options,
                const BufferizationState &state,
                SmallVector<Value> &invocationStack) const {
    assert(value == cast<AllocCroutonOp>(op).getResult() &&
           "invalid value");
    auto tensorType =
        cast<RankedTensorType>(cast<AllocCroutonOp>(op).getResult().getType());

    // The allocation is always in VTCM: that is what the op means, the role
    // `alloc_tensor`'s optional `memory_space` attribute plays upstream.
    Attribute memorySpace = IntegerAttr::get(
        IntegerType::get(op->getContext(), 64), hexagon::VTCM_ADDRESS_SPACE);

    // The layout of the physical rank-5 array when the tensor carries the
    // encoding; null (plain identity) when `drop-encodings` erased it.
    Attribute layout = croutonMemRefLayoutOf(tensorType);
    if (layout)
      return cast<BufferLikeType>(bufferization::getMemRefType(
          tensorType, options, cast<MemRefLayoutAttrInterface>(layout),
          memorySpace));
    return cast<BufferLikeType>(
        getMemRefTypeWithStaticIdentityLayout(tensorType, memorySpace));
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options,
                          BufferizationState &state) const {
    auto allocOp = cast<AllocCroutonOp>(op);
    OpBuilder::InsertionGuard g(rewriter);
    Location loc = allocOp.getLoc();

    // Nothing to do for dead allocations.
    if (allocOp->getUses().empty()) {
      rewriter.eraseOp(op);
      return success();
    }

    FailureOr<BufferLikeType> allocType =
        bufferization::getBufferType(allocOp.getResult(), options, state);
    if (failed(allocType))
      return failure();
    // The result is fully static (the op verifier pins it), so there are no
    // dynamic extents to fill in.
    SmallVector<Value> dynamicDims;
    FailureOr<Value> alloc =
        options.createAlloc(rewriter, loc, cast<MemRefType>(*allocType),
                            dynamicDims);
    if (failed(alloc))
      return failure();

    replaceOpWithBufferizedValues(rewriter, op, *alloc);
    return success();
  }
};

} // namespace

void mlir::hmx::registerBufferizableOpInterfaceExternalModels(
    DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, HmxDialect *dialect) {
    MatmulOp::attachInterface<MatmulOpInterface>(*ctx);
    AllocCroutonOp::attachInterface<AllocCroutonOpInterface>(*ctx);
    PackActOp::attachInterface<PackActOpInterface>(*ctx);
    PackWeightOp::attachInterface<PackWeightOpInterface>(*ctx);
    UnpackAccOp::attachInterface<UnpackAccOpInterface>(*ctx);
    UnpackAccF32Op::attachInterface<UnpackAccF32OpInterface>(*ctx);
  });
}
