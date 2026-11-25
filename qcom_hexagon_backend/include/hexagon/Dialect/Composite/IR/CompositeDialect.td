//===- CompositeDialect.td - Composite dialect definition --*- tablegen -*-===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// Defines the Composite dialect, which contains fused high-level operations
// (e.g. softmax) that replace multi-op patterns for optimized lowering.
//
//===----------------------------------------------------------------------===//

#ifndef HEXAGON_DIALECT_COMPOSITE_IR_COMPOSITE_DIALECT_TD
#define HEXAGON_DIALECT_COMPOSITE_IR_COMPOSITE_DIALECT_TD

include "mlir/IR/OpBase.td"

def Composite_Dialect : Dialect {
  let name = "composite";

  let cppNamespace = "::mlir::composite";

  let summary = "Fused composite operations for optimized lowering";

  let description = [{
    The Composite dialect contains high-level operations that replace
    multi-op patterns recognized in the IR. Each composite op encapsulates a
    well-known computation (e.g. softmax) so that a dedicated lowering pass
    can emit an optimized implementation instead of relying on the generic
    linalg-to-loops pipeline.

    Composite ops support both tensor and memref forms via the
    DestinationStyleOpInterface, enabling automatic bufferization.
  }];
}

class Composite_Op<string mnemonic, list<Trait> traits = []> :
    Op<Composite_Dialect, mnemonic, traits> {
}

#endif // HEXAGON_DIALECT_COMPOSITE_IR_COMPOSITE_DIALECT_TD
