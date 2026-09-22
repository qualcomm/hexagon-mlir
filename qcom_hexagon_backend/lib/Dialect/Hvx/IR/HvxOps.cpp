//===- HvxOps.cpp - HVX Ops -----------------------------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

#include "hexagon/Dialect/Hvx/IR/HvxDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"

using namespace mlir;
using namespace mlir::hvx;

//===----------------------------------------------------------------------===//
// ODS-Generated Declarations
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "hexagon/Dialect/Hvx/IR/HvxOps.cpp.inc"

//===----------------------------------------------------------------------===//
// Verifiers
//===----------------------------------------------------------------------===//

LogicalResult VrorOp::verify() {
  auto vecTy = cast<VectorType>(getInput().getType());
  // The instruction is defined on exactly one implemented vector register;
  // the kernels this dialect serves are 128-byte (HVX length 128B, v79).
  int64_t byteWidth = vecTy.getElementTypeBitWidth() / 8 *
                      vecTy.getShape().front();
  if (byteWidth != 128)
    return emitOpError() << "operand must be one 128-byte vector, got "
                         << byteWidth << " bytes";
  int64_t bytes = getBytes();
  // The instruction rotates by Rt mod VWIDTH; a full-width amount is a no-op,
  // so spell it out rather than silently accepting dead work. (The ODS
  // accessor is signless, so a negative amount is not representable.)
  if (bytes >= 128)
    return emitOpError() << "rotate amount must be in [0, 128) bytes, got "
                         << bytes;
  return success();
}
