//===-- TTXOps.cpp - TTX dialect operations -------------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// This file implements the TTX dialect operations.
//===----------------------------------------------------------------------===//

#include "hexagon/Dialect/TTX/IR/TTXDialect.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

using namespace mlir;
using namespace mlir::ttx;

/// Dialect creation, the instance will be owned by the context. This is the
/// point of registration of custom operations for the dialect.
void TTXDialect::registerOperations() {
  addOperations<
#define GET_OP_LIST
#include "hexagon/Dialect/TTX/IR/TTXOps.cpp.inc"
      >();
}

LogicalResult CummulativeSumOp::verify() { return success(); }

void CummulativeSumOp::print(OpAsmPrinter &p) {
  p.printOptionalAttrDict(this->getOperation()->getAttrs(),
                          /*elidedAttrs=*/{"operand_segment_sizes"});

  if (!getInputs().empty())
    p << " ins(" << getInputs() << " : " << getInputs().getTypes() << ")";

  if (!getOutputs().empty())
    p << " outs(" << getOutputs() << " : " << getOutputs().getTypes() << ")";

  if (!getResultTypes().empty())
    p.printOptionalArrowTypeList(getResultTypes());
}

ParseResult CummulativeSumOp::parse(OpAsmParser &parser,
                                    OperationState &result) {
  SmallVector<Type> inputTypes;
  SmallVector<Type> outputTypes;
  SMLoc inputsOperandsLoc, outputsOperandsLoc;
  SmallVector<OpAsmParser::UnresolvedOperand, 4> inputsOperands,
      outputsOperands;
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();

  if (succeeded(parser.parseOptionalKeyword("ins"))) {
    if (parser.parseLParen())
      return failure();

    inputsOperandsLoc = parser.getCurrentLocation();
    if (parser.parseOperandList(inputsOperands) ||
        parser.parseColonTypeList(inputTypes) || parser.parseRParen())
      return failure();
  }

  if (succeeded(parser.parseOptionalKeyword("outs"))) {
    outputsOperandsLoc = parser.getCurrentLocation();
    if (parser.parseLParen() || parser.parseOperandList(outputsOperands) ||
        parser.parseColonTypeList(outputTypes) || parser.parseRParen())
      return failure();
  }

  if (parser.resolveOperands(inputsOperands, inputTypes, inputsOperandsLoc,
                             result.operands) ||
      parser.resolveOperands(outputsOperands, outputTypes, outputsOperandsLoc,
                             result.operands))
    return failure();

  result.addAttribute("operand_segment_sizes",
                      parser.getBuilder().getDenseI32ArrayAttr(
                          {static_cast<int32_t>(inputsOperands.size()),
                           static_cast<int32_t>(outputsOperands.size())}));

  SmallVector<Type, 1> resultTypes;
  if (parser.parseOptionalArrowTypeList(resultTypes))
    return failure();
  result.addTypes(resultTypes);

  return success();
}

//===----------------------------------------------------------------------===//
// ODS-Generated Declarations
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "hexagon/Dialect/TTX/IR/TTXOps.cpp.inc"
