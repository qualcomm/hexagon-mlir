//===- crouton-materialize.mlir - the layout materialisation boundary ------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
// `hmx.pack_act`/`hmx.pack_weight`/`hmx.unpack_acc` are the only conversions
// between the row-major and the crouton layout, so their direction is part of
// their contract: pack is row-major -> crouton, unpack is crouton -> row-major.
//
// The check is gated on the `#hmx.crouton` encoding being present. It is inert
// for the memref form (bufferization drops the encoding; the rank-5 shape is all
// that is left) and for the current unannotated tensor form, but the moment an
// encoding appears it must be on the crouton side and its `logical` must agree
// with the row-major side.
//
// RUN: linalg-hexagon-opt %s -verify-diagnostics -split-input-file
//===----------------------------------------------------------------------===//

//--- !!! pack: the row-major source must not carry the crouton encoding
func.func @pack_act_encoded_src(%src: tensor<1x1x16x32x2xf16, #hmx.crouton<logical = [32, 32]>>,
                                %dst: tensor<1x1x16x32x2xf16, #hmx.crouton<logical = [32, 32]>>,
                                %i: index, %j: index)
    -> tensor<1x1x16x32x2xf16, #hmx.crouton<logical = [32, 32]>> {
  // expected-error @+1 {{row-major side}}
  %0 = hmx.pack_act ins(%src, %i, %j : tensor<1x1x16x32x2xf16, #hmx.crouton<logical = [32, 32]>>)
                     outs(%dst : tensor<1x1x16x32x2xf16, #hmx.crouton<logical = [32, 32]>>)
      -> tensor<1x1x16x32x2xf16, #hmx.crouton<logical = [32, 32]>>
  return %0 : tensor<1x1x16x32x2xf16, #hmx.crouton<logical = [32, 32]>>
}

// -----

//--- !!! pack: the crouton side's logical shape must match the source. A weight
//--- stores Wᵀ, so its logical shape is the *transpose* of the row-major source.
func.func @pack_weight_logical_mismatch(%src: tensor<32x32xf16>,
                                        %dst: tensor<2x2x16x32x2xf16, #hmx.crouton<logical = [64, 64]>>,
                                        %i: index, %j: index)
    -> tensor<2x2x16x32x2xf16, #hmx.crouton<logical = [64, 64]>> {
  // expected-error @+1 {{must match the transpose of src shape}}
  %0 = hmx.pack_weight ins(%src, %i, %j : tensor<32x32xf16>)
                       outs(%dst : tensor<2x2x16x32x2xf16, #hmx.crouton<logical = [64, 64]>>)
      -> tensor<2x2x16x32x2xf16, #hmx.crouton<logical = [64, 64]>>
  return %0 : tensor<2x2x16x32x2xf16, #hmx.crouton<logical = [64, 64]>>
}

// -----

//--- !!! unpack: the row-major destination must not carry the crouton encoding
func.func @unpack_acc_encoded_dst(%src: tensor<1x1x16x32x2xf16>,
                                  %dst: tensor<1x1x16x32x2xf16, #hmx.crouton<logical = [32, 32]>>,
                                  %i: index, %j: index)
    -> tensor<1x1x16x32x2xf16, #hmx.crouton<logical = [32, 32]>> {
  // expected-error @+1 {{row-major side}}
  %0 = hmx.unpack_acc ins(%src, %i, %j : tensor<1x1x16x32x2xf16>)
                      outs(%dst : tensor<1x1x16x32x2xf16, #hmx.crouton<logical = [32, 32]>>)
      -> tensor<1x1x16x32x2xf16, #hmx.crouton<logical = [32, 32]>>
  return %0 : tensor<1x1x16x32x2xf16, #hmx.crouton<logical = [32, 32]>>
}

// -----

//--- !!! unpack: the crouton side's logical shape must match the result
func.func @unpack_acc_logical_mismatch(%src: tensor<2x2x16x32x2xf16, #hmx.crouton<logical = [64, 64]>>,
                                       %dst: tensor<32x32xf16>, %i: index, %j: index)
    -> tensor<32x32xf16> {
  // expected-error @+1 {{must match dst shape}}
  %0 = hmx.unpack_acc ins(%src, %i, %j : tensor<2x2x16x32x2xf16, #hmx.crouton<logical = [64, 64]>>)
                      outs(%dst : tensor<32x32xf16>) -> tensor<32x32xf16>
  return %0 : tensor<32x32xf16>
}

// -----

//--- accepted: correctly annotated conversions, and the unannotated form the
//--- pipeline still produces before the encoding migration lands.
func.func @materialize_ok(%src: tensor<32x32xf16>,
                          %cdst: tensor<1x1x16x32x2xf16, #hmx.crouton<logical = [32, 32]>>,
                          %csrc: tensor<1x1x16x32x2xf16, #hmx.crouton<logical = [32, 32]>>,
                          %rdst: tensor<32x32xf16>,
                          %plain: tensor<1x1x16x32x2xf16>,
                          %i: index, %j: index) {
  %0 = hmx.pack_act ins(%src, %i, %j : tensor<32x32xf16>)
                    outs(%cdst : tensor<1x1x16x32x2xf16, #hmx.crouton<logical = [32, 32]>>)
      -> tensor<1x1x16x32x2xf16, #hmx.crouton<logical = [32, 32]>>
  %1 = hmx.unpack_acc ins(%csrc, %i, %j : tensor<1x1x16x32x2xf16, #hmx.crouton<logical = [32, 32]>>)
                      outs(%rdst : tensor<32x32xf16>) -> tensor<32x32xf16>
  %2 = hmx.pack_weight ins(%src, %i, %j : tensor<32x32xf16>)
                       outs(%plain : tensor<1x1x16x32x2xf16>) -> tensor<1x1x16x32x2xf16>
  return
}
