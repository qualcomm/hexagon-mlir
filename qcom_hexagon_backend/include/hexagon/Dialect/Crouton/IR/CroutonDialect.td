//===- CroutonDialect.td --------------------------------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

#ifndef HEXAGON_DIALECT_CROUTON_IR_CROUTON_DIALECT_TD
#define HEXAGON_DIALECT_CROUTON_IR_CROUTON_DIALECT_TD

include "mlir/IR/OpBase.td"

def Crouton_Dialect : Dialect {
  let name = "crouton";

  let cppNamespace = "::mlir::crouton";

  let summary = "Crouton operations";

  let description = [{
    Crouton Dialect.
  }];

  let dependentDialects = [
    "mlir::linalg::LinalgDialect"
  ];

  let extraClassDeclaration = [{
    void registerTypes();
    void registerAttributes();
    void registerOperations();
  }];

  let useDefaultTypePrinterParser = 1;
  let usePropertiesForAttributes = 1;
}

#endif // HEXAGON_DIALECT_CROUTON_IR_CROUTON_DIALECT_TD
