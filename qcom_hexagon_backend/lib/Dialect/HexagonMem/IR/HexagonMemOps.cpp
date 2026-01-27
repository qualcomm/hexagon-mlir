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
  if (auto memRefType = mlir::dyn_cast<MemRefType>(type)) {
    if (static_cast<int64_t>(getDynamicSizes().size()) !=
        memRefType.getNumDynamicDims())
      return emitOpError("dimension operand count does not equal memref "
                         "dynamic dimension count");
  }
  return success();
}

//===----------------------------------------------------------------------===//
// ODS-Generated Declarations
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "hexagon/Dialect/HexagonMem/IR/HexagonMemOps.cpp.inc"
