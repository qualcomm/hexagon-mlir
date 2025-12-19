//===- HexagonTPtrDialect.td ----------------------------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

#ifndef HEXAGON_DIALECT_TPTR_IR_TPTR_DIALECT_TD
#define HEXAGON_DIALECT_TPTR_IR_TPTR_DIALECT_TD

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Dialect/Ptr/IR/PtrDialect.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinTypeInterfaces.td"

def HexagonTPtr_Dialect : Dialect {
  let name = "tptr";

  let cppNamespace = "::mlir::tptr";

  let summary = "Temporary Pointer Dialect";

  let description = [{
    Hexagon Typed Pointer Dialect.
  }];

  let dependentDialects = [
    "mlir::linalg::LinalgDialect",
    "mlir::arith::ArithDialect",
    "mlir::ptr::PtrDialect"
  ];

  let extraClassDeclaration = [{
    void registerTypes();
    void registerAttributes();
    void registerOperations();
  }];

  let useDefaultAttributePrinterParser = 1;
  let usePropertiesForAttributes = 1;
}


#endif // HEXAGON_DIALECT_TPTR_IR_TPTR_DIALECT_TD
