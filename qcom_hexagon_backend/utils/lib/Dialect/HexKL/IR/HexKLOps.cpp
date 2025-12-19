//===- HexKLOps.cpp - HexKL Ops  ------------------------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

#include "hexagon/Common/Common.h"

#include "hexagon/Dialect/HexKL/IR/HexKLDialect.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Support/LLVM.h"

using namespace mlir;
using namespace mlir::hexkl;

/// Dialect creation, the instance will be owned by the context. This is the
/// point of registration of custom operations for the dialect.
void HexKLDialect::registerOperations() {
  addOperations<
#define GET_OP_LIST
#include "hexagon/Dialect/HexKL/IR/HexKLOps.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// ODS-Generated Declarations
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "hexagon/Dialect/HexKL/IR/HexKLOps.cpp.inc"

// Requires 1D memref<i8> in VTCM (memory space 1).
static LogicalResult verifyHmxBlockMemref(Operation *op, Value v) {
  auto memrefType = dyn_cast<MemRefType>(v.getType());
  if (!memrefType)
    return op->emitOpError("expected hmx_block to be a memref");

  if (!memrefType.hasRank() || memrefType.getRank() != 1)
    return op->emitOpError("hmx_block must be a rank-1 memref");

  auto elem = dyn_cast<IntegerType>(memrefType.getElementType());
  if (!elem || elem.getWidth() != 8)
    return op->emitOpError("hmx_block element type must be i8");

  if (memrefType.getMemorySpaceAsInt() != hexagon::VTCM_ADDRESS_SPACE)
    return op->emitOpError(
        "hmx_block must be in VTCM address space (memory space 1)");

  return success();
}

/// Verify that if a value is a constant integer, it must be non-negative.
static LogicalResult verifyNonNegativeIfConst(Operation *op, Value v,
                                              StringRef name) {
  if (auto c = mlir::getConstantIntValue(v))
    if (*c < 0)
      return op->emitOpError() << name << " must be non-negative";
  return success();
}

/// Helper function to verify non-negative operands.
/// Returns failure if verification fails.
template <typename... Args>
static LogicalResult verifyNonNegative(Operation *op, Value v, StringRef name,
                                       Args... args) {
  if (failed(verifyNonNegativeIfConst(op, v, name)))
    return failure();
  if constexpr (sizeof...(args) > 0) {
    return verifyNonNegative(op, args...);
  }
  return success();
}

LogicalResult MicroHMXSetupAccReadF16Op::verify() {
  return verifyHmxBlockMemref(getOperation(), getHmxBlock());
}

LogicalResult MicroHMXCopySubmatrixToF16Op::verify() {
  Operation *op = getOperation();
  if (failed(verifyHmxBlockMemref(op, getHmxBlock())))
    return failure();

  return verifyNonNegative(op, getOutOffset(), "out_offset", getTileRow(),
                           "tile_row", getTileCol(), "tile_col", getInputRows(),
                           "input_rows", getInputCols(), "input_cols");
}

LogicalResult MicroHMXRmToAhF16Op::verify() {
  Operation *op = getOperation();
  if (failed(verifyHmxBlockMemref(op, getHmxBlock())))
    return failure();

  return verifyNonNegative(op, getActivationOutOffset(),
                           "activation_out_offset", getFlatInOffset(),
                           "flat_in_offset");
}

LogicalResult MicroHMXRmToWhF16Op::verify() {
  Operation *op = getOperation();
  if (failed(verifyHmxBlockMemref(op, getHmxBlock())))
    return failure();

  return verifyNonNegative(op, getWeightOffset(), "weight_offset", getTileRow(),
                           "tile_row", getTileCol(), "tile_col", getWtCols(),
                           "wt_cols");
}

LogicalResult MicroHMXMmF16Op::verify() {
  Operation *op = getOperation();
  if (failed(verifyHmxBlockMemref(op, getHmxBlock())))
    return failure();

  return verifyNonNegative(op, getActivationOffset(), "activation_offset",
                           getWeightOffset(), "weight_offset");
}

LogicalResult MicroHMXAccReadF16Op::verify() {
  Operation *op = getOperation();
  if (failed(verifyHmxBlockMemref(op, getHmxBlock())))
    return failure();

  return verifyNonNegative(op, getOutOffset(), "out_offset");
}

LogicalResult MicroHMXAhToRmF16Op::verify() {
  Operation *op = getOperation();
  if (failed(verifyHmxBlockMemref(op, getHmxBlock())))
    return failure();

  return verifyNonNegative(op, getFlatOutOffset(), "flat_out_offset",
                           getActivationInOffset(), "activation_in_offset");
}

LogicalResult MicroHMXCopyF16ToF32SubmatrixOp::verify() {
  Operation *op = getOperation();
  if (failed(verifyHmxBlockMemref(op, getHmxBlock())))
    return failure();

  return verifyNonNegative(op, getInOffset(), "in_offset", getTileRow(),
                           "tile_row", getTileCol(), "tile_col",
                           getOutputRows(), "output_rows", getOutputCols(),
                           "output_cols");
}
