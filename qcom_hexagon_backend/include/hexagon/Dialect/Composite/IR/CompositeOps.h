//===- CompositeOps.h - Composite op declarations -------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

#ifndef HEXAGON_DIALECT_COMPOSITE_IR_COMPOSITE_OPS_H
#define HEXAGON_DIALECT_COMPOSITE_IR_COMPOSITE_OPS_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/TilingInterface.h"

#include "hexagon/Dialect/Composite/IR/CompositeDialect.h"

#define GET_OP_CLASSES
#include "hexagon/Dialect/Composite/IR/CompositeOps.h.inc"

#endif // HEXAGON_DIALECT_COMPOSITE_IR_COMPOSITE_OPS_H
