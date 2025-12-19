//===- TmTensorOps.td -----------------------------------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

#ifndef HEXAGON_DIALECT_TMTENSOR_IR_TMTENSOR_OPS_TD
#define HEXAGON_DIALECT_TMTENSOR_IR_TMTENSOR_OPS_TD

include "mlir/Interfaces/DestinationStyleOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "hexagon/Dialect/TmTensor/IR/TmTensorDialect.td"
include "hexagon/Dialect/TmTensor/IR/TmTensorTypes.td"

// Base class for TmTensor dialect ops.
class TmTensor_Op<string mnemonic, list<Trait> traits> :
    Op<TmTensor_Dialect, mnemonic, traits> {

}

//===----------------------------------------------------------------------===//
// TmTensor Operations
//===----------------------------------------------------------------------===//

def TmTensor_AttentionOp : TmTensor_Op<"attention", []> {

  let summary = "Scaled Dot Product Attention operation";
  let description = [{
    Computes SDPA given Query, Key, Value, and Mask tensors.
    Equivalent to : softmax(QK^T / sqrt(d_k) + mask) * V
    Produces one output tensor with the attention result.
  }];

  let arguments = (ins
    AnyRankedTensor:$query,
    AnyRankedTensor:$key,
    AnyRankedTensor:$value,
    AnyRankedTensor:$mask,
    AnyRankedTensor:$out
  );
  let results = (outs AnyRankedTensor:$result);

  let assemblyFormat = [{
    `ins` `(`
       $query `,` $key `,`  $value `,` $mask
       `:` type($query) `,` type($key) `,` type($value) `,` type($mask)
        `)`
    `outs` `(` $out `:` type($out) `)`
    attr-dict `->` type($result)
  }];

  let hasVerifier = 1;
}

#endif // HEXAGON_DIALECT_TMTENSOR_IR_TMTENSOR_OPS_TD
