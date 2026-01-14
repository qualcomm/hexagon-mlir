//===- Passes.td - HexKL to LLVM passes --------------------*- tablegen -*-===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
// Converts hexkl to llvm ops.
//===----------------------------------------------------------------------===//

#ifndef HEXAGON_CONVERSION_HEXKLTOLLVM_PASSES_TD
#define HEXAGON_CONVERSION_HEXKLTOLLVM_PASSES_TD

include "mlir/Pass/PassBase.td"

def HexKLToLLVM : Pass<"hexkl-to-llvm", "mlir::ModuleOp"> {
  let summary = "Convert HexKL dialect to LLVM";
  let description = [{
    Lowers HexKL ops to LLVM function calls corresponding to one or multiple HexKL runtime APIs.
    For example, `hexkl.matmul ins(%A, %B : memref<32x64xf16>, memref<64x128xf16>)
                                        outs(%out : memref<32x128xf32>)`
    can be lowered to:

    ```mlir
        %3 = llvm.extractvalue %0[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
        %4 = llvm.extractvalue %2[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
        %5 = llvm.extractvalue %1[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
        %6 = llvm.extractvalue %2[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
        %7 = llvm.extractvalue %2[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
        %8 = llvm.extractvalue %0[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
        llvm.call @hexkl_matmul_f16f16_f32(%6, %8, %7, %3, %4, %5) : (i64, i64, i64, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
        ...
    ```
    }];
  let constructor = "mlir::hexkl::createHexKLToLLVMPass()";
}

#endif // HEXAGON_CONVERSION_HEXKLTOLLVM_PASSES_TD
