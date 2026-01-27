//===- HexKLDialect.td - HexKL Dialect ---------------------*- tablegen -*-===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
// Describes the hexkl dialect.
//===----------------------------------------------------------------------===//

#ifndef HEXAGON_DIALECT_HEXKL_IR_HEXKL_DIALECT_TD
#define HEXAGON_DIALECT_HEXKL_IR_HEXKL_DIALECT_TD

include "mlir/IR/OpBase.td"

def HexKL_Dialect : Dialect {
  let name = "hexkl";

  let cppNamespace = "::mlir::hexkl";

  let summary = "HexKL operations";

  let description = [{
    The `hexkl` dialect is intended to abtract implementation of
    ops such as matmul using HexKL micro and macro APIs.
  }];

  let dependentDialects = [
    "mlir::linalg::LinalgDialect",
  ];

  let extraClassDeclaration = [{
    void registerOperations();
  }];

  let usePropertiesForAttributes = 1;
}

#endif // HEXAGON_DIALECT_HEXKL_IR_HEXKL_DIALECT_TD
