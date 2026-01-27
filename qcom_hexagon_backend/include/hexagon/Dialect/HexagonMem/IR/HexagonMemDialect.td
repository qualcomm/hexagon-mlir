//===- HexagonMemDialect.td -----------------------------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

#ifndef HEXAGON_DIALECT_HEXAGONMEM_IR_HEXAGONMEM_DIALECT_TD
#define HEXAGON_DIALECT_HEXAGONMEM_IR_HEXAGONMEM_DIALECT_TD

include "mlir/IR/OpBase.td"

def HexagonMem_Dialect : Dialect {
  let name = "hexagonmem";

  let cppNamespace = "::mlir::hexagonmem";

  let summary = "HexagonMem operations";

  let description =
      [{The `hexagonmem` dialect is intended to hold ops and types related to
            Hexagon VTCM or DDR memory manipulation including allocation,
        de-allocation and copy ops.}];

  let extraClassDeclaration = [{
    void registerTypes();
    void registerAttributes();
    void registerOperations();
  }];

  let usePropertiesForAttributes = 1;
}

#endif // HEXAGON_DIALECT_HEXAGONMEM_IR_HEXAGONMEM_DIALECT_TD
