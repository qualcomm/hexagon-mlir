//===- BufferizableOpInterfaceImpl.cpp - Impl. of BufferizableOpInterface -===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

#include "hexagon/Dialect/HexKL/Transforms/BufferizableOpInterfaceImpl.h"
#include "hexagon/Dialect/HexKL/IR/HexKLDialect.h"
#include "mlir/Dialect/Bufferization/IR/DstBufferizableOpInterfaceImpl.h"

using namespace mlir;
using namespace hexkl;
using namespace mlir::bufferization;

namespace {

struct MatmulOpInterface
    : public DstBufferizableOpInterfaceExternalModel<MatmulOpInterface,
                                                     hexkl::MatmulOp> {
  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options,
                          BufferizationState &state) const {

    auto destinationStyleOp = cast<DestinationStyleOpInterface>(op);
    // Nothing to do. This op is already bufferized.
    if (destinationStyleOp.hasPureBufferSemantics())
      return success();

    // Ensure op has only tensors.
    if (!destinationStyleOp.hasPureTensorSemantics())
      return op->emitError() << "op does not have pure tensor semantics";

    auto matmulOp = cast<hexkl::MatmulOp>(op);
    FailureOr<Value> inputBuffer =
        getBuffer(rewriter, matmulOp.getLhs(), options, state);
    if (failed(inputBuffer))
      return failure();
    FailureOr<Value> inputBuffer2 =
        getBuffer(rewriter, matmulOp.getRhs(), options, state);
    if (failed(inputBuffer2))
      return failure();
    FailureOr<Value> outputBuffer =
        getBuffer(rewriter, matmulOp.getOuts(), options, state);
    if (failed(outputBuffer))
      return failure();
    hexkl::MatmulOp::create(rewriter, matmulOp.getLoc(),
                            /*result=*/TypeRange(), *inputBuffer, *inputBuffer2,
                            *outputBuffer);
    replaceOpWithBufferizedValues(rewriter, op, *outputBuffer);
    return success();
  }
};
} // namespace

void mlir::hexkl::registerBufferizableOpInterfaceExternalModels(
    DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, hexkl::HexKLDialect *dialect) {
    MatmulOp::attachInterface<MatmulOpInterface>(*ctx);
  });
}
