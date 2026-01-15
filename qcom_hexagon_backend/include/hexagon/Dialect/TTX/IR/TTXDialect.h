//===- TTXDialect.h -------------------------------------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

#ifndef HEXAGON_DIALECT_TTX_IR_TTX_DIALECT_H
#define HEXAGON_DIALECT_TTX_IR_TTX_DIALECT_H

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
// TTX Dialect
//===----------------------------------------------------------------------===//
#include "hexagon/Dialect/TTX/IR/TTXDialect.h.inc"

//===----------------------------------------------------------------------===//
// TTX Enums
//===----------------------------------------------------------------------===//
#include "hexagon/Dialect/TTX/IR/TTXEnums.h.inc"

//===----------------------------------------------------------------------===//
// TTX Attributes
//===----------------------------------------------------------------------===//
#define GET_ATTRDEF_CLASSES
#include "hexagon/Dialect/TTX/IR/TTXAttrs.h.inc"

//===----------------------------------------------------------------------===//
// TTX Types
//===----------------------------------------------------------------------===//
#define GET_TYPEDEF_CLASSES
#include "hexagon/Dialect/TTX/IR/TTXTypes.h.inc"

//===----------------------------------------------------------------------===//
// TTX Ops
//===----------------------------------------------------------------------===//
#define GET_OP_CLASSES
#include "hexagon/Dialect/TTX/IR/TTXOps.h.inc"

#endif // HEXAGON_DIALECT_TTX_IR_TTX_DIALECT_H
