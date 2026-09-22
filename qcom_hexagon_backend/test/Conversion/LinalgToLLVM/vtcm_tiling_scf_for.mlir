// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(func.func(vtcm-tiling),canonicalize,one-shot-bufferize{bufferize-function-boundaries allow-return-allocs-from-loops})' | FileCheck %s

func.func @add_kernel() -> tensor<32xf32> {
  %cst = arith.constant 1.000000e+00 : f32
  %cst_0 = arith.constant 0.000000e+00 : f32
  %c2_i32 = arith.constant 2 : i32
  %c1_i32 = arith.constant 1 : i32
  %c0_i32 = arith.constant 0 : i32
  %0 = tensor.empty() : tensor<32xf32>
  %fill = linalg.fill ins(%cst : f32) outs(%0 : tensor<32xf32>) -> tensor<32xf32>
  %result = scf.for %I = %c0_i32 to %c2_i32 step %c1_i32
                          iter_args(%cF = %fill) -> (tensor<32xf32>)  : i32 {
    %genericRes = linalg.generic
             {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
             iterator_types = ["parallel"]} ins(%cF : tensor<32xf32>) outs(%0: tensor<32xf32>) {
    ^bb0(%in_2: f32, %out_3: f32):
      %5 = arith.addf %in_2, %cst : f32
      linalg.yield %5 : f32
    } -> tensor<32xf32>
    scf.yield %genericRes : tensor<32xf32>
  }
  return %result : tensor<32xf32>
}

// The generic's result is yielded by the enclosing scf.for (loop-carried), so
// VTCMTiling deliberately skips staging it: staging would leave the loop's
// init_arg in DDR and the yielded value in VTCM, which one-shot-bufferize
// rejects. It runs on DDR buffers directly instead.
// CHECK: %[[ALLOC:.+]] = memref.alloc() {alignment = 64 : i64} : memref<32xf32>
// CHECK: %[[ALLOC0:.+]] = memref.alloc() {alignment = 64 : i64} : memref<32xf32>
// CHECK: linalg.fill ins({{.*}} : f32) outs(%[[ALLOC0]] : memref<32xf32>)
// CHECK: %[[FOR:.+]] = scf.for {{.*}} iter_args(%arg1 = %[[ALLOC0]]) -> (memref<32xf32>)  : i32 {
// CHECK:       linalg.generic {{.*}} ins(%arg1 : memref<32xf32>) outs(%[[ALLOC]] : memref<32xf32>) {
// CHECK:       %[[ALLOC_RES:.+]] = memref.alloc() {alignment = 64 : i64} : memref<32xf32>
// CHECK-NEXT:  memref.copy %[[ALLOC]], %[[ALLOC_RES]] : memref<32xf32> to memref<32xf32>
// CHECK-NEXT:  scf.yield %[[ALLOC_RES]] : memref<32xf32>
