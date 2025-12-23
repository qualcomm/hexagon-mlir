// RUN: linalg-hexagon-opt %s  -pass-pipeline='builtin.module(func.func(lower-ttx))' \
// RUN:     | FileCheck %s -check-prefixes=CHECK

module {
  func.func @associative_scan_1D(%x : tensor<256xi32>) -> tensor<256xi32> {
    %c-65536_i32 = arith.constant 65536 : i32
    %result = "ttx.scan"(%x) <{axis = 0 : i32, reverse = false}> ({
      ^bb0(%arg23: i32, %arg24: i32):
        %446 = arith.andi %arg23, %c-65536_i32 : i32
        %447 = arith.andi %arg24, %c-65536_i32 : i32
        %448 = arith.cmpi eq, %446, %447 : i32
        %449 = arith.addi %arg23, %arg24 : i32
        %450 = arith.subi %449, %446 : i32
        %451 = arith.select %448, %450, %arg24 : i32
        ttx.scan.return %451 :  i32
      }) : (tensor<256xi32>) -> tensor<256xi32>
    return %result : tensor<256xi32>
  }
}

// CHECK-LABEL: @associative_scan_1D
// CHECK-SAME: (%[[X:.+]]:  tensor<256xi32>) -> tensor<256xi32> { 
// 
// CHECK-DAG: %[[C65536:.+]] = arith.constant 65536 : i32
// CHECK-DAG: %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG: %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG: %[[C256:.+]] = arith.constant 256 : index
//
// CHECK: %[[INIT:.+]] = tensor.extract %[[X]][%[[C0]]] : tensor<256xi32>
// CHECK: %[[Out:.+]]:2 = scf.for %[[I:.+]] = %[[C1]] to %[[C256]] step %[[C1]] iter_args(%[[ACC:.+]] = %[[INIT]], %[[PARTIAL:.+]] = %[[X]]) -> (i32, tensor<256xi32>) {
// CHECK:                   %[[CURR:.+]] = tensor.extract %[[PARTIAL]][%[[I]]] : tensor<256xi32> 
// CHECK:                   %[[A:.+]] = arith.andi %[[ACC]], %[[C65536]] : i32
// CHECK:                   %[[B:.+]] = arith.andi %[[CURR]], %[[C65536]] : i32
// CHECK:                   %[[C:.+]] = arith.cmpi eq, %[[A]], %[[B]] : i32
// CHECK:                   %[[D:.+]] = arith.addi %[[ACC]], %[[CURR]] : i32
// CHECK:                   %[[E:.+]] = arith.subi %[[D]], %[[A]] : i32
// CHECK:                   %[[S:.+]] = arith.select %[[C]], %[[E]], %[[CURR]] : i32
// CHECK:                   %[[T:.+]] = tensor.insert %[[S]] into  %[[PARTIAL]][%[[I]]] : tensor<256xi32>
// CHECK:                   scf.yield %[[S]], %[[T]] :  i32, tensor<256xi32>
// CHECK:  }
// CHECK:  return %[[Out]]#1 : tensor<256xi32> 
