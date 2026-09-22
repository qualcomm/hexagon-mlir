//===- HvxDialect.h - HVX Dialect ------------------------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

#ifndef HEXAGON_DIALECT_HVX_IR_HVX_DIALECT_H
#define HEXAGON_DIALECT_HVX_IR_HVX_DIALECT_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

//===----------------------------------------------------------------------===//
// HVX Dialect
//===----------------------------------------------------------------------===//
#include "hexagon/Dialect/Hvx/IR/HvxDialect.h.inc"

//===----------------------------------------------------------------------===//
// HVX Ops
//===----------------------------------------------------------------------===//
#define GET_OP_CLASSES
#include "hexagon/Dialect/Hvx/IR/HvxOps.h.inc"

#endif // HEXAGON_DIALECT_HVX_IR_HVX_DIALECT_H
