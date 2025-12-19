//===- TTXOps.td ----------------------------------------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

#ifndef HEXAGON_DIALECT_TTX_IR_TTX_OPS_TD
#define HEXAGON_DIALECT_TTX_IR_TTX_OPS_TD

include "mlir/Interfaces/DestinationStyleOpInterface.td"

include "mlir/Interfaces/SideEffectInterfaces.td"
include "hexagon/Dialect/TTX/IR/TTXDialect.td"
include "hexagon/Dialect/TTX/IR/TTXTypes.td"

// Base class for TTX dialect ops.
class TTX_Op<string mnemonic, list<Trait> traits> :
    Op<TTX_Dialect, mnemonic, traits> {

}

def TTX_ScanOp : TTX_Op<"scan", []> {
  let arguments = (ins
    AnyRankedTensor:$input,
    I32Attr:$axis,
    BoolAttr:$reverse
  );

  let results = (outs AnyRankedTensor:$output);
  let regions = (region AnyRegion:$body);

  let extraClassDeclaration = [{
    int64_t getRank() {
      return cast<ShapedType>(getInput().getType()).getRank();
    }
  }];
}

def TTX_ScanReturnOp : TTX_Op<"scan.return", [Pure, MemRefsNormalizable,
                                              Terminator]> {
  let summary = "ttx.scan.return operation";
  let description = [{
    scan.return is equivalent to a yield of a single scalar.
  }];

  let arguments = (ins Variadic<AnyType>:$operands);

  let builders = [OpBuilder<(ins), [{
    build($_builder, $_state, {});
  }]>];

  let assemblyFormat = "attr-dict ($operands^ `:` type($operands))?";
  let hasVerifier = 0;

  let extraClassDeclaration = [{
    Value getYieldValue() { return getOperands()[0]; }
  }];
}

def TTX_CummulativeSumOp : TTX_Op<"cumsum", [AttrSizedOperandSegments]> {
  let arguments = (ins
    Variadic<AnyRankedTensor>:$inputs,
    Variadic<AnyRankedTensor>:$outputs,
    UI32Attr:$axis
  );

  let results = (outs Variadic<AnyRankedTensor>:$result_tensors);

  let hasVerifier = 1;
  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = [{
    // Implemented as part of DestinationStyleOpInterface
    MutableOperandRange getDpsInitsMutable() { return getOutputsMutable(); }

    int64_t getRank() {
      return cast<ShapedType>(getInput().getType()).getRank();
    }

    Value getInput() {
      return getInputs()[0];
    }

    Value getOutput() {
      return getOutputs()[0];
    }

    static StringRef getAxisAttrStrName() { return "axis"; }
  }];
}
#endif // HEXAGON_DIALECT_TTX_IR_TTX_OPS_TD
