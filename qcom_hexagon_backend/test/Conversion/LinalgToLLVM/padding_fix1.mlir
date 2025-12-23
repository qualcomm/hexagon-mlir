// RUN: linalg-hexagon-opt %s \
// RUN:   -pass-pipeline='builtin.module(linalg-generalize-named-ops,hexagon-tiling{split-tiling-range=true},eliminate-empty-tensors,canonicalize,cse,one-shot-bufferize)' \
// RUN:  | FileCheck %s -check-prefixes=CHECK,SPLIT
//
// RUN: linalg-hexagon-opt %s \
// RUN:   -pass-pipeline='builtin.module(linalg-generalize-named-ops,hexagon-tiling{split-tiling-range=false},eliminate-empty-tensors,canonicalize,cse,one-shot-bufferize)' \
// RUN:  | FileCheck %s -check-prefixes=CHECK,PAD

// This test check the fix for error: 
//         'bufferization.materialize_in_destination' : cannot avoid RaW conflict'
// which occurs when padding with `BufferizationMaterializeInDestination` option.
func.func @gpt2_extract(%x : tensor<65xf32>, %y : tensor<65xf32>, %z : tensor<65xf32>, %w : tensor<65xf32>)
                        -> (tensor<65xf32>, tensor<65xf32>) {
   %e = tensor.empty() : tensor<65xf32>
   %add =  linalg.add ins(%x, %y : tensor<65xf32>, tensor<65xf32>) outs(%e : tensor<65xf32>) -> tensor<65xf32>
   %add2 =  linalg.add ins(%z, %w : tensor<65xf32>, tensor<65xf32>) outs(%e : tensor<65xf32>) -> tensor<65xf32>
   return %add, %add2 : tensor<65xf32>, tensor<65xf32>
}

// CHECK-LABEL: @gpt2_extract
// CHECK-DAG: %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG: %[[C32:.+]] = arith.constant 32 : index
//
// SPLIT-DAG: %[[C64:.+]] = arith.constant 64 : index
// SPLIT: %subview = memref.subview %{{.*}}[0] [64] [1] : memref<65xf32, {{.*}}> to memref<64xf32, {{.*}}>
// SPLIT: %{{.*}} = scf.for %arg4 = %[[C0]] to %[[C64]] step %[[C32]] iter_args
// SPLIT: linalg.generic
// SPLIT: %{{.*}} = memref.subview %4[64] [1] [1] : memref<65xf32, {{.*}}> to memref<1xf32, {{.*}}>
// SPLIT: linalg.generic

// PAD-DAG:   %[[CST_0:.+]] = arith.constant 0.000000e+00 : f32
// PAD-DAG: %[[C96:.+]] = arith.constant 96 : index
// PAD: %[[PADDED:.+]] = memref.alloc() {alignment = 64 : i64} : memref<96xf32>
// PAD: linalg.map outs(%[[PADDED]] : memref<96xf32>)
// PAD: linalg.yield %[[CST_0]] : f32
// PAD: %4 = scf.for %{{.*}} = %[[C0]] to %[[C96]] step %[[C32]] iter_args(%{{.*}} = %{{.*}}) -> (memref<96xf32>)
// PAD: linalg.generic

