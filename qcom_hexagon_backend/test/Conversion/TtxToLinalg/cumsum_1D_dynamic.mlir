// RUN: linalg-hexagon-opt %s  -pass-pipeline='builtin.module(func.func(lower-ttx))' \
// RUN:     | FileCheck %s -check-prefixes=CHECK
//
module {
  func.func @cumsum_1D_dynamic(%X: tensor<?xf32>, %Y: tensor<?xf32>)  -> tensor<?xf32> {

    %res = ttx.cumsum {axis = 0 : ui32, operandSegmentSizes = array<i32: 1, 1>} ins(%X : tensor<?xf32>) outs(%Y : tensor<?xf32>) -> tensor<?xf32>
    return %res :  tensor<?xf32>
  }
}

// CHECK-LABEL: @cumsum_1D_dynamic
// CHECK-SAME: (%[[X:.+]]: tensor<?xf32>, %{{.*}}: tensor<?xf32>)  -> tensor<?xf32>
// CHECK-DAG: %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG: %[[C1:.+]] = arith.constant 1 : index
//
// CHECK:  %[[DIM0:.+]] = tensor.dim %{{.*}}, %[[C0]] : tensor<?xf32>
// CHECK: %[[Init:.+]] = tensor.extract %{{.*}}[%[[C0]]] : tensor<?xf32>
//
// CHECK: %[[Out:.+]]:2 = scf.for %[[I:.+]] = %[[C1]] to %[[DIM0]] step %[[C1]] iter_args(%[[P:.+]] = %[[Init]], %[[PARTIAL_SUM:.+]] = %[[X]]) -> (f32, tensor<?xf32>) {
// CHECK:                  %[[VAL:.+]] =   tensor.extract %[[PARTIAL_SUM]][%[[I]]] : tensor<?xf32>
// CHECK:                  %[[NEW_P:.+]] = arith.addf %[[P]], %[[VAL]] : f32
// CHECK:                  %[[NEW_SUM:.+]] = tensor.insert %[[NEW_P]] into %[[PARTIAL_SUM]][%[[I]]] : tensor<?xf32>
// CHECK:                  scf.yield %[[NEW_P]], %[[NEW_SUM]] : f32, tensor<?xf32>
// CHECK:  }
// CHECK:  return %[[Out]]#1 : tensor<?xf32> 
