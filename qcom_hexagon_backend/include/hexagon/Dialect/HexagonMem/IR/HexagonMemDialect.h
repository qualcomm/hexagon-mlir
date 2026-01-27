//===- HexagonMemDialect.h ------------------------------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

#ifndef HEXAGON_DIALECT_HEXAGONMEM_IR_HEXAGONMEM_DIALECT_H
#define HEXAGON_DIALECT_HEXAGONMEM_IR_HEXAGONMEM_DIALECT_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/TypeSupport.h"
#include "mlir/IR/Types.h"
#include "mlir/Interfaces/CopyOpInterface.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"

//===----------------------------------------------------------------------===//
// HexagonMem Dialect
//===----------------------------------------------------------------------===//
#include "hexagon/Dialect/HexagonMem/IR/HexagonMemDialect.h.inc"

//===----------------------------------------------------------------------===//
// HexagonMem Enums
//===----------------------------------------------------------------------===//
#include "hexagon/Dialect/HexagonMem/IR/HexagonMemEnums.h.inc"

//===----------------------------------------------------------------------===//
// HexagonMem Attributes
//===----------------------------------------------------------------------===//
#define GET_ATTRDEF_CLASSES
#include "hexagon/Dialect/HexagonMem/IR/HexagonMemAttrs.h.inc"

//===----------------------------------------------------------------------===//
// HexagonMem Types
//===----------------------------------------------------------------------===//
#define GET_TYPEDEF_CLASSES
#include "hexagon/Dialect/HexagonMem/IR/HexagonMemTypes.h.inc"

//===----------------------------------------------------------------------===//
// HexagonMem Ops
//===----------------------------------------------------------------------===//
#define GET_OP_CLASSES
#include "hexagon/Dialect/HexagonMem/IR/HexagonMemOps.h.inc"

#endif // HEXAGON_DIALECT_HEXAGONMEM_IR_HEXAGONMEM_DIALECT_H
