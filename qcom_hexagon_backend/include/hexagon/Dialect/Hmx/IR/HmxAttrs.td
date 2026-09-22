//===- HmxAttrs.td - HMX Dialect attributes ----------------*- tablegen -*-===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
// The crouton layout as a first-class tensor encoding.
//
// The layout used to be a shape convention: "a rank-5 type whose trailing dims
// are 16x32x2 is a crouton array". That made the layout invisible to every pass
// except the ones that knew the convention, so layout propagation had to be
// re-derived ad hoc. `#hmx.crouton` makes it explicit and checkable instead.
//
// It is attached as the *encoding* of the `RankedTensorType` (`tensor<...,
// #hmx.crouton<...>>`), not as a `MemRefType` layout: the crouton layout is a
// rank-raising permutation, which `MemRefLayoutAttrInterface` cannot represent
// (it is specified for simple strided buffers only), and the whole propagation
// problem lives pre-bufferization, where values are tensors. See
// docs/hmx/layout-representation-survey.md sections 1.1/1.2/4.1.
//
// After bufferization the permutation is over -- the physical array *is* the
// rank-5 shape -- so the memref side can carry the layout as the identity-map
// `#hmx.crouton_memref_layout` (`CroutonMemRefLayoutAttr` below), which the
// interface's strided-buffer contract does allow.
//
// There is deliberately no `kind` (AH/WH/AR) in the encoding: the three roles
// are byte-for-byte the same permutation (verified on device), so role is a
// property of *use* -- the operand position of `hmx.matmul` -- and putting it in
// the type would break the "AR result == AH operand" type equality that makes a
// matmul chain zero-copy.
//===----------------------------------------------------------------------===//

#ifndef HEXAGON_DIALECT_HMX_IR_HMX_ATTRS_TD
#define HEXAGON_DIALECT_HMX_IR_HMX_ATTRS_TD

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/IR/TensorEncoding.td"
include "hexagon/Dialect/Hmx/IR/HmxDialect.td"

// Base class for HMX dialect attributes.
class Hmx_Attr<string name, string _mnemonic, list<Trait> traits = []>
    : AttrDef<Hmx_Dialect, name, traits> {
  let mnemonic = _mnemonic;
}

def Hmx_CroutonLayoutAttr
    : Hmx_Attr<"CroutonLayout", "crouton",
               [DeclareAttrInterfaceMethods<VerifiableTensorEncoding>]> {
  let summary = "The HMX engine's crouton layout, as a tensor encoding";

  let description = [{
    The layout of the rank-5 crouton array the HMX engine reads and writes:

    ```mlir
    %c = #hmx.crouton<logical = [M, N]>
    tensor<M/32, N/32, 16, 32, 2 x f16, %c>
    ```

    The trailing three dims are one crouton: `16` pairs `x 32` columns `x 2`
    halves of a pair. The leading two dims are the crouton grid. `logical` is the
    row-major `[M, N]` f16 matrix the array holds, and it is what the verifier
    cross-checks the grid against (`logical == grid * 32`). The tile edge (32) is
    a fixed property of the engine's crouton and is not a parameter.

    The same encoding is used for an activation, a weight and an accumulator
    read-out: the three are the same permutation, so the role is a property of
    the operation that uses the array, not of the type (see the dialect's layout
    contract and docs/hmx/layout-representation-survey.md section 4.2).

    `VerifiableTensorEncoding` means `RankedTensorType::verify` checks the
    encoding automatically whenever the type is formed, so an illegal crouton
    type cannot exist in the IR; no per-op verifier has to repeat it.
  }];

  let parameters = (ins
    ArrayRefParameter<"int64_t">:$logical
  );

  let hasCustomAssemblyFormat = 1;
  let genVerifyDecl = 1;

  let extraClassDeclaration = [{
    /// The edge of one crouton: a 32x32 fp16 tile. The numbers live in
    /// HmxCroutonLayout.h so the passes and the host contract share them.
    static constexpr int64_t kTile = mlir::hmx::crouton::kTileEdge;

    /// The tail of one crouton in the physical rank-5 type: 16 pairs x 32
    /// columns x 2 halves of a pair.
    static constexpr int64_t kCroutonPair = mlir::hmx::crouton::kCroutonPair;
    static constexpr int64_t kCroutonCol = mlir::hmx::crouton::kCroutonCol;
    static constexpr int64_t kCroutonHalf = mlir::hmx::crouton::kCroutonHalf;

    /// The logical (row-major) shape this crouton array holds: [M, N].
    int64_t getLogicalM() const { return getLogical()[0]; }
    int64_t getLogicalN() const { return getLogical()[1]; }
  }];
}

def Hmx_CroutonMemRefLayoutAttr
    : Hmx_Attr<"CroutonMemRefLayout", "crouton_memref_layout",
               [DeclareAttrInterfaceMethods<MemRefLayoutAttrInterface,
                                            ["verifyLayout"]>]> {
  let summary = "The crouton layout on a memref, as an identity-map layout";

  let description = [{
    The memref-side carrier of the crouton layout, produced by bufferization
    for a tensor that carries `#hmx.crouton`:

    ```mlir
    %l = #hmx.crouton_memref_layout<logical = [M, N]>
    memref<M/32, N/32, 16, 32, 2 x f16, %l, 1>
    ```

    The rank-raising permutation is done once the array is materialised, so on
    the memref side the layout is plain contiguous storage -- the physical
    rank-5 shape *is* the layout. This attribute therefore implements
    `MemRefLayoutAttrInterface` with the **identity map** over that rank-5
    shape: it is exactly the strided-buffer layout the interface is specified
    for, `isIdentity()` holds (the default judges it from `getAffineMap()`),
    and `MemRefType::getStridesAndOffset` returns the same static strides a
    layout-less memref of the same shape has. It carries `logical` (like
    `#hmx.crouton`) so post-bufferization code can still cross-check the array
    against the row-major matrix it holds, and `verifyLayout` enforces that
    agreement whenever the type is formed.

    Default bufferization loses the layout entirely: the stock one-shot
    bufferize pass cannot be given a custom type converter, and
    `bufferization.alloc_tensor` infers its own static-identity buffer type.
    The dialect therefore has its own allocation op, `hmx.alloc_crouton`
    (emitted by `matmul-to-hmx`), whose BufferizableOpInterface applies this
    mapping -- `mlir::hmx::croutonMemRefLayoutOf` -- to its encoded result, so
    the layout survives bufferization on the memref side (the seam of
    docs/hmx/hmx-interface-gaps.md section 2.2).
  }];

  let parameters = (ins
    ArrayRefParameter<"int64_t">:$logical
  );

  let hasCustomAssemblyFormat = 1;
  let genVerifyDecl = 1;

  let extraClassDeclaration = [{
    /// The edge of one crouton: a 32x32 fp16 tile.
    static constexpr int64_t kTile = mlir::hmx::crouton::kTileEdge;

    /// The tail of one crouton in the physical rank-5 type.
    static constexpr int64_t kCroutonPair = mlir::hmx::crouton::kCroutonPair;
    static constexpr int64_t kCroutonCol = mlir::hmx::crouton::kCroutonCol;
    static constexpr int64_t kCroutonHalf = mlir::hmx::crouton::kCroutonHalf;

    /// The logical (row-major) shape this crouton array holds: [M, N].
    int64_t getLogicalM() const { return getLogical()[0]; }
    int64_t getLogicalN() const { return getLogical()[1]; }
  }];
}

#endif // HEXAGON_DIALECT_HMX_IR_HMX_ATTRS_TD
