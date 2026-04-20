//===-- HexagonMemOps.cpp - HexagonMem dialect ops ------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// This file implements the HexagonMem dialect operations.
//===----------------------------------------------------------------------===//

#include "hexagon/Common/Common.h"
#include "hexagon/Dialect/HexagonMem/IR/HexagonMemDialect.h"
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
using namespace mlir::hexagonmem;
using namespace mlir::crouton;

/// Dialect creation, the instance will be owned by the context. This is the
/// point of registration of custom operations for the dialect.
void HexagonMemDialect::registerOperations() {
  addOperations<
#define GET_OP_LIST
#include "hexagon/Dialect/HexagonMem/IR/HexagonMemOps.cpp.inc"
      >();
}

LogicalResult AllocOp::verify() {
  auto type = getBuffer().getType();
  if (auto croutonType = mlir::dyn_cast<CroutonType>(type)) {
    if (getAlignment() != 2048)
      return emitOpError(
          "Crouton allocations are expected to have an alignment of 2048");
    if (static_cast<int64_t>(getDynamicSizes().size()) != 0)
      return emitOpError(
          "Dynamic shapes are not supported for crouton allocations");

  } else if (auto memRefType = mlir::dyn_cast<MemRefType>(type)) {
    if (static_cast<int64_t>(getDynamicSizes().size()) !=
        memRefType.getNumDynamicDims())
      return emitOpError("dimension operand count does not equal memref "
                         "dynamic dimension count");
  }
  return success();
}

LogicalResult verifyTypeCompatibility(Operation *op, MemRefType memrefType,
                                      CroutonType croutonType) {
  if (!memrefType.hasRank())
    return op->emitOpError("Unsupported: unranked memrefs");

  if (!memrefType.hasStaticShape())
    return op->emitOpError("Unsupported: dynamic shaped memrefs");

  auto memrefElementType = memrefType.getElementType();
  auto croutonElementType = croutonType.getElementType();

  if (memrefElementType != croutonElementType) {
    return op->emitOpError("Element types don't match");
  }

  auto elementSize = memrefElementType.getIntOrFloatBitWidth() / 8;

  auto memrefShape = memrefType.getShape();
  auto croutonTableShape = croutonType.getShape();

  if (memrefShape.back() * elementSize != 2048) {
    return op->emitOpError("The product of the last dimension of memrefType is "
                           "expected to be equal to 2048 bytes");
  }

  if (memrefType.getRank() - 1 != croutonType.getRank()) {
    return op->emitOpError()
           << "Memref rank " << memrefType.getRank() - 1
           << " does not match crouton table rank " << croutonType.getRank()
           << " for the first n-1 dimensions";
  }

  for (size_t i = 0; i < croutonTableShape.size(); ++i) {
    if (memrefShape[i] != croutonTableShape[i]) {
      return op->emitOpError()
             << "Memref shape does not match crouton shape for dimension " << i;
    }
  }

  if (hexagon::isInVTCMAddressSpace(memrefType) !=
      croutonType.getVtcm().getValue()) {
    return op->emitOpError(
        "Memref and crouton type address spaces don't match");
  }

  return success();
}

LogicalResult MemrefToCroutonOp::verify() {
  auto memrefType = llvm::cast<MemRefType>(getSource().getType());
  auto croutonType = llvm::cast<CroutonType>(getResult().getType());

  return verifyTypeCompatibility(getOperation(), memrefType, croutonType);
}

LogicalResult CroutonToMemrefOp::verify() {
  auto croutonType = llvm::cast<CroutonType>(getSource().getType());
  auto memrefType = llvm::cast<MemRefType>(getResult().getType());

  return verifyTypeCompatibility(getOperation(), memrefType, croutonType);
}

//===----------------------------------------------------------------------===//
// ODS-Generated Declarations
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "hexagon/Dialect/HexagonMem/IR/HexagonMemOps.cpp.inc"
