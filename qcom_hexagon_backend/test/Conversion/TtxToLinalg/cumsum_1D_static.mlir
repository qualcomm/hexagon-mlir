// RUN: linalg-hexagon-opt %s  -pass-pipeline='builtin.module(func.func(lower-ttx))' \
// RUN:     | FileCheck %s -check-prefixes=CHECK

module {
  func.func @cumsum_1D_static(%arg0: memref<*xf32> {tt.divisibility = 16 : i32},
                              %arg1: memref<*xf32> {tt.divisibility = 16 : i32},
                              %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32) {

    %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [0], sizes: [32], strides: [1] : memref<*xf32> to memref<32xf32, strided<[1]>>
    %alloc = memref.alloc() : memref<32xf32>
    memref.copy %reinterpret_cast, %alloc : memref<32xf32, strided<[1]>> to memref<32xf32>
    %0 = bufferization.to_tensor %alloc restrict writable : memref<32xf32> to tensor<32xf32>
    %1 = tensor.empty() : tensor<32xf32>

    %2 = ttx.cumsum {axis = 0 : ui32, operandSegmentSizes = array<i32: 1, 1>} ins(%0 : tensor<32xf32>) outs(%1 : tensor<32xf32>) -> tensor<32xf32>

    %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [0], sizes: [32], strides: [1] : memref<*xf32> to memref<32xf32, strided<[1]>>
    bufferization.materialize_in_destination %2 in writable %reinterpret_cast_0 : (tensor<32xf32>, memref<32xf32, strided<[1]>>) -> ()
    return
  }
}
// CHECK-LABEL: @cumsum_1D_static
// CHECK-DAG: %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG: %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG: %[[C32:.+]] = arith.constant 32 : index
//
// CHECK: %[[X:.+]] = bufferization.to_tensor
// CHECK: %[[Init:.+]] = tensor.extract %[[X]][%[[C0]]] : tensor<32xf32> 
// CHECK: %[[Out:.+]]:2 = scf.for %[[I:.+]] = %[[C1]] to %[[C32]] step %[[C1]]
// CHECK-SAME:             iter_args(%[[P:.+]] = %[[Init]], %[[PARTIAL_SUM:.+]] = %[[X]]) -> (f32, tensor<32xf32>) {
// CHECK:                  %[[VAL:.+]] =   tensor.extract %[[PARTIAL_SUM]][%[[I]]] : tensor<32xf32>
// CHECK:                  %[[NEW_P:.+]] = arith.addf %[[P]], %[[VAL]] : f32
// CHECK:                  %[[NEW_SUM:.+]] = tensor.insert %[[NEW_P]] into %[[PARTIAL_SUM]][%[[I]]] : tensor<32xf32>
// CHECK:                  scf.yield %[[NEW_P]], %[[NEW_SUM]] : f32, tensor<32xf32>
// CHECK:  }
// CHECK: bufferization.materialize_in_destination %[[Out]]#1 
