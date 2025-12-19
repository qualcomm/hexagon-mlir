// RUN: linalg-hexagon-opt %s  -pass-pipeline='builtin.module(func.func(lower-ttx))' \
// RUN:     | FileCheck %s -check-prefixes=CHECK

module {
  func.func @cumsum_2D(%arg0: memref<16x32xf32>, %arg1 : memref<16x32xf32>) { 
    %x = bufferization.to_tensor %arg0 restrict writable : memref<16x32xf32> to tensor<16x32xf32>
    %y = bufferization.to_tensor %arg1 restrict writable : memref<16x32xf32> to tensor<16x32xf32>

    %sum = ttx.cumsum
             {axis = 1 : ui32, operandSegmentSizes = array<i32: 1, 1>}
             ins(%x : tensor<16x32xf32>) outs(%y : tensor<16x32xf32>) -> tensor<16x32xf32>
    bufferization.materialize_in_destination %sum in writable %arg1 : (tensor<16x32xf32>, memref<16x32xf32>) -> ()
    return
  }
}

// CHECK-LABEL: @cumsum_2D
// CHECK-DAG: %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG: %[[C32:.+]] = arith.constant 32 : index
//
// CHECK: %[[X:.+]] = bufferization.to_tensor
// CHECK: %[[Init:.+]] = tensor.extract_slice %[[X]][0, 0] [16, 1] [1, 1] : tensor<16x32xf32> to tensor<16x1xf32>
//
// CHECK: %[[Out:.+]]:2 = scf.for %[[I:.+]] = %[[C1]] to %[[C32]] step %[[C1]]
// CHECK-SAME:              iter_args(%[[P:.+]] = %[[Init]], %[[PARTIAL_SUM:.+]] = %[[X]]) -> (tensor<16x1xf32>, tensor<16x32xf32>) { 
// CHECK:                   %[[VAL:.+]] =   tensor.extract_slice %[[PARTIAL_SUM]][0, %[[I]]] [16, 1] [1, 1]
// CHECK-SAME:                                 : tensor<16x32xf32> to tensor<16x1xf32>
// CHECK:                   %[[NEW_P:.+]] = arith.addf %[[P]], %[[VAL]] : tensor<16x1xf32>
// CHECK:                   %[[NEW_SUM:.+]] = tensor.insert_slice %[[NEW_P]] into %[[PARTIAL_SUM]][0, %[[I]]] [16, 1] [1, 1]
// CHECK-SAME:                                 : tensor<16x1xf32> into tensor<16x32xf32>
// CHECK:                   scf.yield %[[NEW_P]], %[[NEW_SUM]] : tensor<16x1xf32>, tensor<16x32xf32>
// CHECK:  }
// CHECK: bufferization.materialize_in_destination %[[Out]]#1
