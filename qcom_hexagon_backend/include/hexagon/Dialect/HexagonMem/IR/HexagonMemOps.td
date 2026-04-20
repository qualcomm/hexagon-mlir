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
include "hexagon/Dialect/Crouton/IR/CroutonDialect.td"
include "hexagon/Dialect/Crouton/IR/CroutonTypes.td"
include "hexagon/Dialect/HexagonMem/IR/HexagonMemDialect.td"
include "hexagon/Dialect/HexagonMem/IR/HexagonMemTypes.td"

// Base class for crouton dialect ops.
class HexagonMem_Op<string mnemonic, list<Trait> traits = []> :
    Op<HexagonMem_Dialect, mnemonic, traits> {
}

def AnyMemRefOrCroutonType : AnyTypeOf<[AnyMemRef, Crouton_CroutonType]>;

def ExpDataTypes : AnyAttrOf<[I8Attr, I16Attr, I32Attr, I64Attr, F32Attr]>;
//===----------------------------------------------------------------------===//
// AllocOp
//===----------------------------------------------------------------------===//
def HexagonMem_AllocOp : HexagonMem_Op<"alloc"> {
  let summary = "HexagonMem buffer allocation operation";

  let description = [{
    Allocates memory on VTCM or DDR memory for Hexagon. Based on the output type, this op can
    allocate either a `memref` type or a `crouton` type and in "vtcm" or "ddr"(default).

    A memref type call allocates a single contiguous chunk of memory on VTCM or DDR. The size of the
    allocation is determined by the shape and dtype of the output type specified. This returns a
    memref type value and is similar to the value returned by memref.alloc, but with a specific
    address space specified.

    A crouton type allocation call allocates one or more 2KB sized memory on VTCM/DDR. The number of
    2KB chunks are determined by shape operand to the op. The Op returns a crouton type value which
    can be likened to a MemRef descriptor return by memref.alloc.

    Example:

    ```mlir
    %0 = hexagonmem.alloc() : memref<64xi8, 1>
    %1 = hexagonmem.alloc() : memref<64xi8>
    %1 = hexagonmem.alloc(%x) : memref<?xi8>
    %2 = hexagonmem.alloc() : !crouton.crouton<64xi8, vtcm>
    %3 = hexagonmem.alloc() : !crouton.crouton<64xi8>
    ```
  }];

  let arguments = (ins Variadic<Index>:$dynamicSizes,
                       ConfinedAttr<DefaultValuedOptionalAttr<I64Attr, "128">,
                                [IntMinValue<0>]>:$alignment);

  let results = (outs Res<AnyMemRefOrCroutonType, "",
                          [MemAlloc<DefaultResource, 0, FullEffect>]>:$buffer);

  let assemblyFormat = "`(`$dynamicSizes`)` attr-dict `:` qualified(type($buffer))";

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// CopyOp
//===----------------------------------------------------------------------===//

def HexagonMem_CopyOp : HexagonMem_Op<"copy", [SameOperandsElementType,
    SameOperandsShape]> {

  let description = [{
    Copies the data from the source to the destination memref/crouton value.

    Example:

    ```mlir
    hexagonmem.copy %arg0, %arg1 : memref<64xi8> to memref<64xi8, 1>
    hexagonmem.copy %arg2, %arg3 : !crouton.crouton<64xi8> to !crouton.crouton<64xi8, vtcm>
    ```

    Source and destination are expected to have the same value type, element type and shape.
    Otherwise, the result is undefined. They may have different address space (DDR/VTCM)
  }];

  let arguments = (ins Arg<AnyMemRefOrCroutonType, "the memref/crouton to copy from",
                           [MemReadAt<0, FullEffect>]>:$source,
                       Arg<AnyMemRefOrCroutonType, "the memref/crouton to copy to",
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
    Deallocates either a memref or crouton type referenced by the input
    argument. In case of crouton type, it deallocates all 2KB chunks referenced
    by the crouton-table.

    The deallocation can happen from VTCM or DDR
    depending on whether the vtcm attr is set on the input type.

    Example:

    ```mlir
    hexagonmem.dealloc %0 : memref<64xi8, 1>
    hexagonmem.dealloc %0 : memref<64xi8>
    hexagonmem.dealloc %0 : !crouton.crouton<64xi8, vtcm>
    hexagonmem.dealloc %0 : !crouton.crouton<64xi8>
    ```
  }];

  let arguments = (ins Arg<AnyMemRefOrCroutonType, "",
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

//===----------------------------------------------------------------------===//
// MemrefToCroutonOp
//===----------------------------------------------------------------------===//

def HexagonMem_MemrefToCroutonOp : HexagonMem_Op<"memref_to_crouton"> {

  let summary = "Convert from memref to crouton type";

  let description = [{
    Converts memref to crouton type. Data source and target should be in same
    adddress space.

    This op just reinterprets crouton-type from the contiguous memref type.
    The crouton table will be built with entries that are pointers to the 2048
    sized chunks in this contiguous memref buffer.

    When converting to crouton, it is assumed that the last dimension in the shape
    of memref type is going to represent the size of a single crouton and the
    remaining dimensions represent the shape of the crouton pointer table.

    Example:

    ```mlir
    %x = hexagonmem.memref_to_crouton (%y : memref<2x4x2x1024xf16, 1>) : !crouton.crouton<2x4x2xf16, vtcm>
    %x = hexagonmem.memref_to_crouton (%y : memref<2x4x2x2048xi8, 1>) : !crouton.crouton<2x4x2xi8, vtcm>
    ```
  }];

  let arguments = (ins Arg<AnyNon0RankedMemRef, "The source memref",
                           [MemReadAt<0, FullEffect>]>:$source);

  let results = (outs Res<Crouton_CroutonType, "The result crouton",
                           []>:$result);

  let assemblyFormat = [{
    `(` $source `:` type($source) `)` attr-dict `:` qualified(type($result))
  }];

  let hasVerifier = 1;
}

def HexagonMem_CroutonToMemrefOp : HexagonMem_Op<"crouton_to_memref"> {

  let summary = "Convert from crouton to memref type";

  let description = [{
    Converts crouton to memref type. Data source and target should be in same
    adddress space.

    This op just returns the pointer the base pointer of the crouton type if
    the crouton pointers are contiguous.

    If the croutons in the crouton table are not contiguous in memory, this op
    would result in a copy of data.

    Example:

    ```mlir
    %y = hexagonmem.crouton_to_memref (%x : !crouton.crouton<2x4x2xf16, vtcm>) : memref<2x4x2x1024xf16, vtcm>
    %y = hexagonmem.memref_to_crouton (%x : crouton<2x4x2xi8, vtcm>) : memref<2x4x2x2048xi8, 1>
    ```
  }];

  let arguments = (ins Arg<Crouton_CroutonType, "The source crouton",
                           []>:$source);

  let results = (outs Res<AnyNon0RankedMemRef, "The result memref",
                           [MemReadAt<0, FullEffect>]>:$result);

  let assemblyFormat = [{
    `(` $source `:` type($source) `)` attr-dict `:` qualified(type($result))
  }];

  let hasVerifier = 1;
}

#endif // HEXAGON_DIALECT_HEXAGONMEM_IR_HEXAGONMEM_OPS_TD
