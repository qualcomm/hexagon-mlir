//===- HexagonTPtrDialect.h -----------------------------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

#ifndef HEXAGON_DIALECT_TPTR_IR_TPTR_DIALECT_H_
#define HEXAGON_DIALECT_TPTR_IR_TPTR_DIALECT_H_

#include "mlir/Dialect/Ptr/IR/PtrDialect.h"
#include "mlir/Dialect/Ptr/IR/PtrTypes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/TypeSupport.h"
#include "mlir/IR/Types.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"

//===----------------------------------------------------------------------===//
// TPtr Dialect
//===----------------------------------------------------------------------===//
#include "hexagon/Dialect/HexagonTPtr/IR/HexagonTPtrDialect.h.inc"

// TPtr Ops
//===----------------------------------------------------------------------===//
#define GET_OP_CLASSES
#include "hexagon/Dialect/HexagonTPtr/IR/HexagonTPtrOps.h.inc"

// TPtr Types
//===----------------------------------------------------------------------===//
#define GET_TYPEDEF_CLASSES
#include "hexagon/Dialect/HexagonTPtr/IR/HexagonTPtrTypes.h.inc"

// TPtr Attributes
//===----------------------------------------------------------------------===//
#define GET_ATTRDEF_CLASSES
#include "hexagon/Dialect/HexagonTPtr/IR/HexagonTPtrAttributes.h.inc"

#endif // HEXAGON_DIALECT_TPTR_IR_TPTR_DIALECT_H_
