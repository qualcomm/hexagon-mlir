//===- TTXDialect.td ------------------------------------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

#ifndef HEXAGON_DIALECT_TTX_IR_TTX_DIALECT_TD
#define HEXAGON_DIALECT_TTX_IR_TTX_DIALECT_TD

include "mlir/IR/OpBase.td"

def TTX_Dialect : Dialect {
  let name = "ttx";

  let cppNamespace = "::mlir::ttx";

  let summary = "TTX operations";

  let description = [{
    TTX Dialect.
  }];

  let dependentDialects = [
    "mlir::linalg::LinalgDialect",
  ];

  let extraClassDeclaration = [{
    void registerTypes();
    void registerAttributes();
    void registerOperations();
  }];

  let usePropertiesForAttributes = 1;
}

#endif // HEXAGON_DIALECT_TTX_IR_TTX_DIALECT_TD
