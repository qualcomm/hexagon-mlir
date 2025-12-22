//===- HexKLDialect.h - HexKL Dialect  ------------------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

#ifndef HEXAGON_DIALECT_HEXKL_IR_HEXKL_DIALECT_H
#define HEXAGON_DIALECT_HEXKL_IR_HEXKL_DIALECT_H

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
// HexKL Dialect
//===----------------------------------------------------------------------===//
#include "hexagon/Dialect/HexKL/IR/HexKLDialect.h.inc"

//===----------------------------------------------------------------------===//
// HexKL Ops
//===----------------------------------------------------------------------===//
#define GET_OP_CLASSES
#include "hexagon/Dialect/HexKL/IR/HexKLOps.h.inc"

#endif // HEXAGON_DIALECT_HEXKL_IR_HEXKL_DIALECT_H
