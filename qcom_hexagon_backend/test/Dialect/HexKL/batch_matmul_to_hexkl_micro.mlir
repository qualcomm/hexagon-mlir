//===- batch_matmul_to_hexkl_micro.mlir -----------------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

// RUN: linalg-hexagon-opt %s --pass-pipeline='builtin.module(func.func(matmul-to-hexkl))' | FileCheck %s

// CHECK-LABEL: func.func @batch_matmul_fun
module {
  func.func @batch_matmul_fun(%arg0: tensor<4x256x128xf16>, %arg1: tensor<4x128x16xf16>, %arg2: tensor<4x256x16xf16>) -> tensor<4x256x16xf16> {
    // CHECK-NOT: linalg.batch_matmul
    %out = linalg.batch_matmul ins(%arg0, %arg1 : tensor<4x256x128xf16>, tensor<4x128x16xf16>) outs(%arg2 : tensor<4x256x16xf16>) -> tensor<4x256x16xf16> 
    
    // CHECK: %[[C0:.*]] = arith.constant 0 : index
    // CHECK: %[[C1:.*]] = arith.constant 1 : index
    // CHECK: %[[C4:.*]] = arith.constant 4 : index

    // CHECK: %[[RES:.*]] = scf.for %arg3 = %[[C0]] to %[[C4]] step %[[C1]]  iter_args(%arg4 = %arg2) -> (tensor<4x256x16xf16>)
    
      // CHECK-DAG: %[[SLA:.*]] = tensor.extract_slice %arg0[%arg3, 0, 0] [1, 256, 128] [1, 1, 1] : tensor<4x256x128xf16> to tensor<256x128xf16>
      // CHECK-DAG: %[[SLB:.*]] = tensor.extract_slice %arg1[%arg3, 0, 0] [1, 128, 16] [1, 1, 1] : tensor<4x128x16xf16> to tensor<128x16xf16>
      // CHECK-DAG: %[[SLC:.*]] = tensor.empty() : tensor<256x16xf16>
      // CHECK: %[[SLO:.*]] = hexkl.matmul ins(%[[SLA]], %[[SLB]] : tensor<256x128xf16>, tensor<128x16xf16>) outs(%[[SLC]] : tensor<256x16xf16>) -> tensor<256x16xf16>
      // CHECK: %[[NEXT:.*]] = tensor.insert_slice %[[SLO]] into %arg4[%arg3, 0, 0] [1, 256, 16] [1, 1, 1] : tensor<256x16xf16> into tensor<4x256x16xf16>
      // CHECK: scf.yield %[[NEXT]] : tensor<4x256x16xf16>
    
    return %out : tensor<4x256x16xf16>
    // CHECK: return %[[RES]] : tensor<4x256x16xf16>
  }
}
