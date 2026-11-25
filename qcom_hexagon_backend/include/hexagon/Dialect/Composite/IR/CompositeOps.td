//===- CompositeOps.td - Composite op definitions ----------*- tablegen -*-===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// Operation definitions for the Composite dialect.
//
//===----------------------------------------------------------------------===//

#ifndef HEXAGON_DIALECT_COMPOSITE_IR_COMPOSITE_OPS_TD
#define HEXAGON_DIALECT_COMPOSITE_IR_COMPOSITE_OPS_TD

include "hexagon/Dialect/Composite/IR/CompositeDialect.td"
include "mlir/Interfaces/DestinationStyleOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/TilingInterface.td"
include "mlir/IR/CommonTypeConstraints.td"

//===----------------------------------------------------------------------===//
// composite.softmax
//===----------------------------------------------------------------------===//

def Composite_SoftmaxOp
  : Composite_Op<"softmax",
      [DestinationStyleOpInterface,
       DeclareOpInterfaceMethods<TilingInterface,
         ["getLoopIteratorTypes", "getIterationDomain",
          "getTiledImplementation", "getResultTilePosition",
          "generateResultTileValue"]>,
       MemoryEffects<[MemRead, MemWrite]>]> {

  let summary = "Softmax along a given axis";
  let description = [{
    Computes softmax along dimension `axis` of `input` and writes the result
    into `init`:

      max_i   = max_j input[..., j, ...]   (reduction over axis)
      exp_i   = exp(input[..., i, ...] - max_i)
      sum_i   = sum_j exp_j
      out[i]  = exp_i / sum_i

    The `axis` attribute specifies which dimension to reduce over. Currently
    only the innermost dimension (axis == rank - 1) is supported by the
    lowering pass.

    Supports both tensor semantics (returns a tensor result) and memref
    semantics (no result, operates in-place on the `init` buffer). The
    `DestinationStyleOpInterface` enables automatic bufferization between
    the two forms.

    Example (tensor form):
    ```mlir
    %out = composite.softmax axis = 1 %input, %init
        : tensor<256x512xf32>, tensor<256x512xf32> -> tensor<256x512xf32>
    ```

    Example (memref form):
    ```mlir
    composite.softmax axis = 1 %input, %init
        : memref<256x512xf32>, memref<256x512xf32>
    ```
  }];

  let arguments = (ins
    I64Attr:$axis,
    AnyTypeOf<[AnyRankedTensor, AnyMemRef]>:$input,
    AnyTypeOf<[AnyRankedTensor, AnyMemRef]>:$init
  );

  let results = (outs Variadic<AnyRankedTensor>:$result);

  let assemblyFormat = [{
    `axis` `=` $axis $input `,` $init `:` type($input) `,` type($init)
    attr-dict
    (`->` type($result)^)?
  }];

  let extraClassDeclaration = [{
    MutableOperandRange getDpsInitsMutable() { return getInitMutable(); }
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// composite.rms_norm
//===----------------------------------------------------------------------===//

def Composite_RmsNormOp : Composite_Op<"rms_norm", [
  DestinationStyleOpInterface,
  DeclareOpInterfaceMethods<TilingInterface,
                            [
                              "getLoopIteratorTypes", "getIterationDomain",
                              "getTiledImplementation", "getResultTilePosition",
                              "generateResultTileValue"
                            ]>,
  MemoryEffects<[MemRead, MemWrite]>
]> {

  let summary = "RMS normalization with learned weights";
  let description = [{
    Computes RMS normalization along dimension `axis` of `input`,
    applying learned `weights`:

      mean_sq = sum(input^2) / norm_size
      out     = (input * rsqrt(mean_sq + epsilon)) * weights

    The `axis` attribute specifies the reduction dimension.  The `epsilon`
    attribute is the numerical stability constant added before the square
    root.  The `norm_size` attribute is the actual element count used as
    the divisor in the mean computation (may differ from the axis dimension
    size when the tensor is padded).

    Supports both tensor and memref semantics via
    `DestinationStyleOpInterface`.

    Example (tensor form):
    ```mlir
    %out = composite.rms_norm axis = 0 eps = 1.0e-5 norm_size = 1.024000e+03
        %input, %weights, %init
        : tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>
        -> tensor<1024xf32>
    ```
  }];

  let arguments = (ins
    I64Attr:$axis,
    F32Attr:$epsilon,
    F32Attr:$norm_size,
    AnyTypeOf<[AnyRankedTensor, AnyMemRef]>:$input,
    AnyTypeOf<[AnyRankedTensor, AnyMemRef]>:$weights,
    AnyTypeOf<[AnyRankedTensor, AnyMemRef]>:$init
  );

  let results = (outs Variadic<AnyRankedTensor> : $result);

  let assemblyFormat = [{
    `axis` `=` $axis `eps` `=` $epsilon `norm_size` `=` $norm_size
    $input `,` $weights `,` $init
    `:` type($input) `,` type($weights) `,` type($init)
    attr-dict
    (`->` type($result)^)?
  }];

  let extraClassDeclaration = [{
    MutableOperandRange getDpsInitsMutable() { return getInitMutable(); }
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// composite.layer_norm
//===----------------------------------------------------------------------===//

def Composite_LayerNormOp : Composite_Op<"layer_norm", [
  AttrSizedOperandSegments,
  DestinationStyleOpInterface,
  DeclareOpInterfaceMethods<TilingInterface,
                            [
                              "getLoopIteratorTypes", "getIterationDomain",
                              "getTiledImplementation", "getResultTilePosition",
                              "generateResultTileValue"
                            ]>,
  MemoryEffects<[MemRead, MemWrite]>
]> {

  let summary = "Layer normalization with learned affine transform";
  let description = [{
    Computes layer normalization along dimension `axis` of `input`,
    applying learned `weights` and `bias`:

      mean = sum(input) / norm_size
      var  = sum((input - mean)^2) / norm_size
      rstd = 1 / sqrt(var + epsilon)
      out  = (input - mean) * rstd * weights + bias

    The `axis` attribute specifies the reduction dimension.  The `epsilon`
    attribute is the numerical stability constant added before the square
    root.  The `norm_size` attribute is the actual element count used as
    the divisor (may differ from the axis dimension size when the tensor
    is padded).

    The optional `mean_out` and `rstd_out` memref operands receive the
    per-row mean and inverse standard deviation scalars (useful for the
    backward pass).  When absent the scalars are not stored.

    Supports both tensor and memref semantics via
    `DestinationStyleOpInterface`.

    Example (tensor form):
    ```mlir
    %out = composite.layer_norm axis = 1 eps = 1.0e-5
        norm_size = 5.120000e+02
        %input, %weights, %bias, %init
        : tensor<4x512xf32>, tensor<4x512xf32>, tensor<4x512xf32>,
          tensor<4x512xf32>
        -> tensor<4x512xf32>
    ```

    Example (memref form with side outputs):
    ```mlir
    composite.layer_norm axis = 1 eps = 1.0e-5
        norm_size = 5.120000e+02
        %input, %weights, %bias, %init,
        %mean_out, %rstd_out
        : memref<4x512xf32>, memref<4x512xf32>, memref<4x512xf32>,
          memref<4x512xf32>, memref<4xf32>, memref<4xf32>
    ```
  }];

  let arguments = (ins
    I64Attr:$axis,
    F32Attr:$epsilon,
    F32Attr:$norm_size,
    AnyTypeOf<[AnyRankedTensor, AnyMemRef]>:$input,
    AnyTypeOf<[AnyRankedTensor, AnyMemRef]>:$weights,
    AnyTypeOf<[AnyRankedTensor, AnyMemRef]>:$bias,
    AnyTypeOf<[AnyRankedTensor, AnyMemRef]>:$init,
    Optional<AnyMemRef>:$mean_out,
    Optional<AnyMemRef>:$rstd_out
  );

  let results = (outs Variadic<AnyRankedTensor>:$result);

  let assemblyFormat = [{
    `axis` `=` $axis `eps` `=` $epsilon `norm_size` `=` $norm_size
    $input `,` $weights `,` $bias `,` $init
    (`,` $mean_out^ `,` $rstd_out)?
    `:` type($input) `,` type($weights) `,` type($bias) `,` type($init)
    (`,` type($mean_out)^ `,` type($rstd_out))?
    attr-dict
    (`->` type($result)^)?
  }];

  let extraClassDeclaration = [{
    MutableOperandRange getDpsInitsMutable() { return getInitMutable(); }
  }];

  let hasVerifier = 1;
}

#endif // HEXAGON_DIALECT_COMPOSITE_IR_COMPOSITE_OPS_TD
