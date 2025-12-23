//===- CroutonDialect.h ---------------------------------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

#ifndef HEXAGON_DIALECT_CROUTON_IR_CROUTON_DIALECT_H
#define HEXAGON_DIALECT_CROUTON_IR_CROUTON_DIALECT_H

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
// Crouton Dialect
//===----------------------------------------------------------------------===//
#include "hexagon/Dialect/Crouton/IR/CroutonDialect.h.inc"

//===----------------------------------------------------------------------===//
// Crouton Types
//===----------------------------------------------------------------------===//
#define GET_TYPEDEF_CLASSES
#include "hexagon/Dialect/Crouton/IR/CroutonTypes.h.inc"

#endif
