//===- HexagonTPtrAttributes.td -------------------------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

#ifndef HEXAGON_T_PTR_ATTRIBUTES_TD
#define HEXAGON_T_PTR_ATTRIBUTES_TD

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Dialect/Ptr/IR/PtrDialect.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinTypeInterfaces.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/Dialect/Ptr/IR/MemorySpaceInterfaces.td"
include "hexagon/Dialect/HexagonTPtr/IR/HexagonTPtrDialect.td"

// Base Class for all HexagonTPtr attributes.

class HexagonTPtr_Attr<string name, string _mnemonic,
                     list<Trait> traits = []>
    : AttrDef<HexagonTPtr_Dialect, name, traits> {
  let mnemonic = _mnemonic;
}

def HexagonDefaultMemorySpaceAttr
    : HexagonTPtr_Attr<"HexagonDefaultMemorySpace", "default_memory_space",
                     [DeclareAttrInterfaceMethods<MemorySpaceAttrInterface>]> {
}

#endif // HEXAGON_T_PTR_ATTRIBUTES_TD
