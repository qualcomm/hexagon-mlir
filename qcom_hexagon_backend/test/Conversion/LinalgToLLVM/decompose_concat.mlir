//===- decompose_concat.mlir ----------------------------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(func.func(triton-decompose-tensor-concat))' | FileCheck %s -check-prefixes=CHECK

func.func @decompose_1d_concat(%arg0 : tensor<5xf32>,
                            %arg1 : tensor<6xf32>) -> tensor<11xf32> {
  %0 = tensor.concat dim(0) %arg0, %arg1
             : (tensor<5xf32>, tensor<6xf32>) -> tensor<11xf32>
  return %0 : tensor<11xf32>
}

//CHECK: func @decompose_1d_concat
//       CHECK:    tensor.empty() : tensor<11xf32>
//       CHECK:    tensor.insert_slice %{{.*}}[0] [5] [1] : tensor<5xf32> into tensor<11xf32>
//       CHECK:    %[[CONCAT:.+]] = tensor.insert_slice %{{.*}}[5] [6] [1] : tensor<6xf32> into tensor<11xf32>
//       CHECK:    return %[[CONCAT]] : tensor<11xf32>
