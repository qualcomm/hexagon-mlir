//===- TmTensorDialect.h - TmTensor dialect header file -------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

#ifndef HEXAGON_DIALECT_TMTENSOR_IR_TMTENSOR_DIALECT_H
#define HEXAGON_DIALECT_TMTENSOR_IR_TMTENSOR_DIALECT_H

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
// TmTensor Dialect
//===----------------------------------------------------------------------===//
#include "hexagon/Dialect/TmTensor/IR/TmTensorDialect.h.inc"

//===----------------------------------------------------------------------===//
// TmTensor Enums
//===----------------------------------------------------------------------===//
#include "hexagon/Dialect/TmTensor/IR/TmTensorEnums.h.inc"

//===----------------------------------------------------------------------===//
// TmTensor Attributes
//===----------------------------------------------------------------------===//
#define GET_ATTRDEF_CLASSES
#include "hexagon/Dialect/TmTensor/IR/TmTensorAttrs.h.inc"

//===----------------------------------------------------------------------===//
// TmTensor Types
//===----------------------------------------------------------------------===//
#define GET_TYPEDEF_CLASSES
#include "hexagon/Dialect/TmTensor/IR/TmTensorTypes.h.inc"

//===----------------------------------------------------------------------===//
// TmTensor Ops
//===----------------------------------------------------------------------===//
#define GET_OP_CLASSES
#include "hexagon/Dialect/TmTensor/IR/TmTensorOps.h.inc"

#endif // HEXAGON_DIALECT_TMTENSOR_IR_TMTENSOR_DIALECT_H
