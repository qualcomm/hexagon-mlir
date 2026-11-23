//===- Passes.td - Shard to LLVM passes --------------------*- tablegen -*-===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// Converts shard dialect ops used by NSP materialization into standard
// arith/tensor ops so the existing linalg-to-llvm pipeline can finish lowering.
//
//===----------------------------------------------------------------------===//

#ifndef HEXAGON_CONVERSION_SHARDTOLLVM_PASSES_TD
#define HEXAGON_CONVERSION_SHARDTOLLVM_PASSES_TD

include "mlir/Pass/PassBase.td"

def ShardToLLVM : Pass<"shard-to-llvm", "mlir::ModuleOp"> {
  let summary = "Lower shard dialect elementwise ops (e.g., vadd materialized) "
                "to arith/tensor";
  let constructor = "mlir::hexagon::createShardToLLVMPass()";

  let dependentDialects = ["mlir::arith::ArithDialect",
                           "mlir::tensor::TensorDialect",
                           "mlir::func::FuncDialect",
                           "mlir::shard::ShardDialect"];
}

#endif // HEXAGON_CONVERSION_SHARDTOLLVM_PASSES_TD
