//===- HexagonTPtrOps.td --------------------------------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

#ifndef HEXAGON_DIALECT_TPTR_IR_TPTR_OPS_TD
#define HEXAGON_DIALECT_TPTR_IR_TPTR_OPS_TD

include "mlir/Interfaces/DestinationStyleOpInterface.td"

include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Dialect/Ptr/IR/PtrDialect.td"
include "hexagon/Dialect/HexagonTPtr/IR/HexagonTPtrDialect.td"
include "hexagon/Dialect/HexagonTPtr/IR/HexagonTPtrTypes.td"

// Op Base
class HexagonTPtr_Op<string mnemonic, list<Trait> traits = []> :
    Op<HexagonTPtr_Dialect, mnemonic, traits> {
}

def HexagonTPtr_HexagonTypeOffsetOp : HexagonTPtr_Op<"type_offset", [ConstantLike, Pure]> {
  let summary = "tptr type_offset op";
  let description = [{
    The type_offset op returns the target specific
      `sizeof(input-type)`
    Example:
    ```mlir
       %1 = tptr.type_offset f16  : i64
    ```
     The return value will be '2'
  }];

  let arguments = (ins TypeAttr:$baseType);
  let results = (outs AnySignlessIntegerOrIndex:$result);
  let assemblyFormat = [{
     attr-dict $baseType custom<IntType>(type($result))
  }];
  let hasFolder = 1;
}

def HexagonTPtr_HexagonFromMemrefOp : HexagonTPtr_Op<"from_memref", [Pure]> {
  let arguments = (ins AnyMemRef:$input);
  let results = (outs Ptr_PtrType:$result);
  let assemblyFormat = "$input attr-dict `:` type($input) `to` type($result)";
}

def HexagonTPtr_HexagonToMemrefOp : HexagonTPtr_Op<"to_memref", [Pure]> {
  let arguments = (ins Ptr_PtrType:$arg);
  let results = (outs AnyStaticShapeMemRef:$res);
  let assemblyFormat = "$arg attr-dict `:` type($arg) `to` type($res)";
}

def HexagonTPtr_HexagonPtrAddOp
     : HexagonTPtr_Op<"ptradd",
                      [Pure, AllTypesMatch<["base", "result"]>]> {
  let summary = "Pointer-index add operation";
  let description = [{
    The `ptradd` operation adds an `address` and an integer
    or index to produce a new address.

    Example:
    ```mlir
    %addr = ptr.ptradd %addr : !ptr.ptr<3 : i32>, %c10 : i32
    ```
  }];

  let arguments = (ins Ptr_PtrType:$base, AnySignlessIntegerOrIndex:$offset);
  let results = (outs Ptr_PtrType:$result);
  let assemblyFormat = [{
    $base $offset attr-dict
      `:` type($base) `,` type($offset) `to` type($result)
  }];
}


#endif // HEXAGON_DIALECT_TPTR_IR_TPTR_OPS_TD
