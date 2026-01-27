//===- HexagonTPtrTypes.td ------------------------------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

#ifndef HEXAGON_DIALECT_TPTR_IR_TPTR_TYPES_TD
#define HEXAGON_DIALECT_TPTR_IR_TPTR_TYPES_TD

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/BuiltinTypeInterfaces.td"
include "hexagon/Dialect/HexagonTPtr/IR/HexagonTPtrDialect.td"

class HexagonTPtrTypeDef<string name, string _mnemonic, list<Trait> traits = []>
    : TypeDef<HexagonTPtr_Dialect, name, traits> {
    // Used by printer/parser
    let mnemonic = _mnemonic;
}

#endif // HEXAGON_DIALECT_TPTR_IR_TPTR_TYPES_TD
