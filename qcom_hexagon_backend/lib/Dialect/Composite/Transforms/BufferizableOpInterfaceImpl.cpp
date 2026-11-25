//===- BufferizableOpInterfaceImpl.cpp - Bufferization for Composite ops --===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

#include "hexagon/Dialect/Composite/Transforms/BufferizableOpInterfaceImpl.h"
#include "hexagon/Dialect/Composite/IR/CompositeDialect.h"
#include "hexagon/Dialect/Composite/IR/CompositeOps.h"
#include "mlir/Dialect/Bufferization/IR/DstBufferizableOpInterfaceImpl.h"

using namespace mlir;
using namespace mlir::composite;
using namespace mlir::bufferization;

namespace {

struct SoftmaxOpInterface
    : public DstBufferizableOpInterfaceExternalModel<SoftmaxOpInterface,
                                                     composite::SoftmaxOp> {
  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options,
                          BufferizationState &state) const {
    auto softmaxOp = cast<composite::SoftmaxOp>(op);

    auto dstOp = cast<DestinationStyleOpInterface>(op);
    if (dstOp.hasPureBufferSemantics())
      return success();

    if (!dstOp.hasPureTensorSemantics())
      return op->emitError() << "op does not have pure tensor semantics";

    FailureOr<Value> inputBuffer =
        getBuffer(rewriter, softmaxOp.getInput(), options, state);
    if (failed(inputBuffer))
      return failure();

    FailureOr<Value> initBuffer =
        getBuffer(rewriter, softmaxOp.getInit(), options, state);
    if (failed(initBuffer))
      return failure();

    composite::SoftmaxOp::create(rewriter, softmaxOp.getLoc(),
                                 /*result=*/TypeRange(),
                                 softmaxOp.getAxisAttr(), *inputBuffer,
                                 *initBuffer);
    replaceOpWithBufferizedValues(rewriter, op, *initBuffer);
    return success();
  }
};

struct RmsNormOpInterface
    : public DstBufferizableOpInterfaceExternalModel<RmsNormOpInterface,
                                                     composite::RmsNormOp> {
  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options,
                          BufferizationState &state) const {
    auto rmsNormOp = cast<composite::RmsNormOp>(op);

    auto dstOp = cast<DestinationStyleOpInterface>(op);
    if (dstOp.hasPureBufferSemantics())
      return success();

    if (!dstOp.hasPureTensorSemantics())
      return op->emitError() << "op does not have pure tensor semantics";

    FailureOr<Value> inputBuffer =
        getBuffer(rewriter, rmsNormOp.getInput(), options, state);
    if (failed(inputBuffer))
      return failure();

    FailureOr<Value> weightsBuffer =
        getBuffer(rewriter, rmsNormOp.getWeights(), options, state);
    if (failed(weightsBuffer))
      return failure();

    FailureOr<Value> initBuffer =
        getBuffer(rewriter, rmsNormOp.getInit(), options, state);
    if (failed(initBuffer))
      return failure();

    composite::RmsNormOp::create(
        rewriter, rmsNormOp.getLoc(),
        /*result=*/TypeRange(), rmsNormOp.getAxisAttr(),
        rmsNormOp.getEpsilonAttr(), rmsNormOp.getNormSizeAttr(), *inputBuffer,
        *weightsBuffer, *initBuffer);
    replaceOpWithBufferizedValues(rewriter, op, *initBuffer);
    return success();
  }
};

struct LayerNormOpInterface
    : public DstBufferizableOpInterfaceExternalModel<LayerNormOpInterface,
                                                     composite::LayerNormOp> {
  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options,
                          BufferizationState &state) const {
    auto layerNormOp = cast<composite::LayerNormOp>(op);

    auto dstOp = cast<DestinationStyleOpInterface>(op);
    if (dstOp.hasPureBufferSemantics())
      return success();

    // The op may have mixed tensor/memref semantics when optional mean_out/
    // rstd_out memref side outputs are present alongside tensor operands.
    // Skip the pure-tensor check and bufferize the tensor operands only.

    FailureOr<Value> inputBuffer =
        getBuffer(rewriter, layerNormOp.getInput(), options, state);
    if (failed(inputBuffer))
      return failure();

    FailureOr<Value> weightsBuffer =
        getBuffer(rewriter, layerNormOp.getWeights(), options, state);
    if (failed(weightsBuffer))
      return failure();

    FailureOr<Value> biasBuffer =
        getBuffer(rewriter, layerNormOp.getBias(), options, state);
    if (failed(biasBuffer))
      return failure();

    FailureOr<Value> initBuffer =
        getBuffer(rewriter, layerNormOp.getInit(), options, state);
    if (failed(initBuffer))
      return failure();

    // mean_out and rstd_out are already memrefs, pass through unchanged.
    composite::LayerNormOp::create(
        rewriter, layerNormOp.getLoc(),
        /*result=*/TypeRange(), layerNormOp.getAxisAttr(),
        layerNormOp.getEpsilonAttr(), layerNormOp.getNormSizeAttr(),
        *inputBuffer, *weightsBuffer, *biasBuffer, *initBuffer,
        layerNormOp.getMeanOut(), layerNormOp.getRstdOut());
    replaceOpWithBufferizedValues(rewriter, op, *initBuffer);
    return success();
  }
};

} // namespace

void mlir::composite::registerBufferizableOpInterfaceExternalModels(
    DialectRegistry &registry) {
  registry.addExtension(
      +[](MLIRContext *ctx, composite::CompositeDialect *dialect) {
        SoftmaxOp::attachInterface<SoftmaxOpInterface>(*ctx);
        RmsNormOp::attachInterface<RmsNormOpInterface>(*ctx);
        LayerNormOp::attachInterface<LayerNormOpInterface>(*ctx);
      });
}
