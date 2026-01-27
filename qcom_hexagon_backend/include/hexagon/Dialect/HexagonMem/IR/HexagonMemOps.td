//===- HexagonMemOps.td ---------------------------------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

#ifndef HEXAGON_DIALECT_HEXAGONMEM_IR_HEXAGONMEM_OPS_TD
#define HEXAGON_DIALECT_HEXAGONMEM_IR_HEXAGONMEM_OPS_TD

include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/DestinationStyleOpInterface.td"
include "mlir/IR/CommonTypeConstraints.td"
include "mlir/Interfaces/CopyOpInterface.td"
include "hexagon/Dialect/HexagonMem/IR/HexagonMemDialect.td"
include "hexagon/Dialect/HexagonMem/IR/HexagonMemTypes.td"

// Base class for HexagonMem dialect ops.
class HexagonMem_Op<string mnemonic, list<Trait> traits = []> :
    Op<HexagonMem_Dialect, mnemonic, traits> {
}

// Type constraint for memref types
def AnyMemRefType : AnyTypeOf<[AnyMemRef]>;

def ExpDataTypes : AnyAttrOf<[I8Attr, I16Attr, I32Attr, I64Attr, F32Attr]>;
//===----------------------------------------------------------------------===//
// AllocOp
//===----------------------------------------------------------------------===//
def HexagonMem_AllocOp : HexagonMem_Op<"alloc"> {
  let summary = "HexagonMem buffer allocation operation";

  let description = [{
    Allocates memory on VTCM or DDR memory for Hexagon. This op allocates
    a `memref` type in "vtcm" or "ddr"(default).

    A memref type call allocates a single contiguous chunk of memory on VTCM or DDR. The size of the
    allocation is determined by the shape and dtype of the output type specified. This returns a
    memref type value and is similar to the value returned by memref.alloc, but with a specific
    address space specified.

    Example:

    ```mlir
    %0 = hexagonmem.alloc() : memref<64xi8, 1>
    %1 = hexagonmem.alloc() : memref<64xi8>
    %1 = hexagonmem.alloc(%x) : memref<?xi8>
    ```
  }];

  let arguments = (ins Variadic<Index>:$dynamicSizes,
                       ConfinedAttr<DefaultValuedOptionalAttr<I64Attr, "128">,
                                [IntMinValue<0>]>:$alignment);

  let results = (outs Res<AnyMemRefType, "",
                          [MemAlloc<DefaultResource, 0, FullEffect>]>:$buffer);

  let assemblyFormat = "`(`$dynamicSizes`)` attr-dict `:` qualified(type($buffer))";

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// CopyOp
//===----------------------------------------------------------------------===//

def HexagonMem_CopyOp : HexagonMem_Op<"copy", [CopyOpInterface, SameOperandsElementType,
    SameOperandsShape]> {

  let description = [{
    Copies the data from the source to the destination memref value.

    Example:

    ```mlir
    hexagonmem.copy %arg0, %arg1 : memref<64xi8> to memref<64xi8, 1>
    ```

    Source and destination are expected to have the same value type, element type and shape.
    Otherwise, the result is undefined. They may have different address space (DDR/VTCM)
  }];

  let arguments = (ins Arg<AnyMemRefType, "the memref to copy from",
                           [MemReadAt<0, FullEffect>]>:$source,
                       Arg<AnyMemRefType, "the memref to copy to",
                           [MemWriteAt<0, FullEffect>]>:$target);

  let assemblyFormat = [{
    $source `,` $target attr-dict `:` qualified(type($source)) `to` qualified(type($target))
  }];

  // TODO
  // let hasCanonicalizer = 1;
  // let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// DeallocOp
//===----------------------------------------------------------------------===//

def HexagonMem_DeallocOp : HexagonMem_Op<"dealloc"> {
  let summary = "HexagonMem buffer deallocation operation";

  let description = [{
    Deallocates a memref type referenced by the input argument.

    The deallocation can happen from VTCM or DDR
    depending on whether the vtcm attr is set on the input type.

    Example:

    ```mlir
    hexagonmem.dealloc %0 : memref<64xi8, 1>
    hexagonmem.dealloc %0 : memref<64xi8>
    ```
  }];

  let arguments = (ins Arg<AnyMemRefType, "",
                           [MemFreeAt<0, FullEffect>]>:$buffer);

  let results = (outs);

  let assemblyFormat = "$buffer attr-dict `:` qualified(type($buffer))";

  // TODO
  //let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// ConvertLayoutOp
//===----------------------------------------------------------------------===//

def HexagonMem_ConvertLayoutOp : HexagonMem_Op<"convert_layout"> {
  let summary = "Convert memref into a new memory layout";

  let description = [{
    This operation takes a memref and transforms it into a new memref with a specified layout.
    It is designed to lower to an implementation that copies data from one layout to another.

    Both the input and output memrefs are pre-allocated, following the destination passing style.

    Example:

    ```mlir
    #map = affine_map<(d0) -> (d0 floordiv 4, d0 mod 4)>
    hexagonmem.convert_layout ins(%arg0 : memref<62xi8>), outs(%arg1 : memref<16x4xi8>) {pad_value = 0, layout = #map}
    ```
  }];

  let arguments = (ins Arg<AnyNon0RankedMemRef, "memref for layout change">:$source,
                       Arg<AnyNon0RankedMemRef, "memref with new layout">:$target,
                       OptionalAttr<ExpDataTypes>:$pad_value,
                       AffineMapAttr:$layout);

  let assemblyFormat = [{
    `ins` `(` $source `:` type($source)`)` `,`
      `outs` `(` $target `:` type($target)`)` attr-dict
  }];

  // TODO
  //let hasVerifier = 1;
}

#endif // HEXAGON_DIALECT_HEXAGONMEM_IR_HEXAGONMEM_OPS_TD
