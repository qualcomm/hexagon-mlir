// RUN: linalg-hexagon-opt --split-reduce-generic -cse -canonicalize %s | FileCheck %s

#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> ()>

// Max/min reductions must NOT be split: the split materialises the accumulator in
// memory and the reduction then degrades to element-by-element scalar compares
// (measured 78 -> 941 machine instructions for a 128 element f16 max on v79).
// CHECK-LABEL: func.func @test_maxnumf_reduction
// CHECK-SAME: (%[[ARG0:.*]]: tensor<2048xf32>) -> tensor<f32>
// CHECK-DAG: %[[CST:.*]] = arith.constant 0xFF800000 : f32
// CHECK-DAG: %[[ALLOC:.*]] = bufferization.alloc_tensor() : tensor<f32>
// CHECK: %[[INSERTED:.*]] = tensor.insert %[[CST]] into %[[ALLOC]][] : tensor<f32>
// CHECK-NOT: tensor.expand_shape
// CHECK: linalg.generic {indexing_maps = [#{{.*}}, #{{.*}}], iterator_types = ["reduction"]} ins(%[[ARG0]] : tensor<2048xf32>) outs(%[[INSERTED]] : tensor<f32>)
// CHECK: ^bb0(%[[IN:.*]]: f32, %[[OUT:.*]]: f32):
// CHECK: %[[MAX:.*]] = arith.maxnumf %[[IN]], %[[OUT]] : f32
// CHECK: linalg.yield %[[MAX]] : f32
// CHECK: return
func.func @test_maxnumf_reduction(%arg0: tensor<2048xf32>) -> tensor<f32> {
  %cst = arith.constant 0xFF800000 : f32
  %3 = bufferization.alloc_tensor() : tensor<f32>
  %inserted = tensor.insert %cst into %3[] : tensor<f32>

  %4 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg0 : tensor<2048xf32>) outs(%inserted : tensor<f32>) {
  ^bb0(%in: f32, %out: f32):
    %12 = arith.maxnumf %in, %out : f32
    linalg.yield %12 : f32
  } -> tensor<f32>

  return %4 : tensor<f32>
}

// Additive reductions still get split: they vectorise afterwards and do benefit
// from it (measured on the rms_norm row reduction: 1239 -> 686 instructions).
// CHECK-LABEL: func.func @test_addf_reduction
// CHECK-SAME: (%[[ARG0:.*]]: tensor<2048xf32>) -> tensor<f32>
// CHECK-DAG: %[[CST:.*]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG: %[[ALLOC:.*]] = bufferization.alloc_tensor() : tensor<f32>
// CHECK: %[[INSERTED:.*]] = tensor.insert %[[CST]] into %[[ALLOC]][] : tensor<f32>
// CHECK: %[[EXPANDED:.*]] = tensor.expand_shape %[[ARG0]] {{\[}}[0, 1]{{\]}} output_shape [64, 32] : tensor<2048xf32> into tensor<64x32xf32>
// CHECK: %[[EMPTY:.*]] = tensor.empty() : tensor<32xf32>
// CHECK: %[[FILLED:.*]] = linalg.fill ins(%[[CST]] : f32) outs(%[[EMPTY]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK: %[[PARTIAL:.*]] = linalg.generic {indexing_maps = [#{{.*}}, #{{.*}}], iterator_types = ["reduction", "parallel"]} ins(%[[EXPANDED]] : tensor<64x32xf32>) outs(%[[FILLED]] : tensor<32xf32>)
// CHECK: ^bb0(%[[IN:.*]]: f32, %[[OUT:.*]]: f32):
// CHECK: %[[ADD1:.*]] = arith.addf %[[IN]], %[[OUT]] : f32
// CHECK: linalg.yield %[[ADD1]] : f32
// CHECK: } -> tensor<32xf32>
// CHECK: %[[FINAL:.*]] = linalg.generic {indexing_maps = [#{{.*}}, #{{.*}}], iterator_types = ["reduction"]} ins(%[[PARTIAL]] : tensor<32xf32>) outs(%[[INSERTED]] : tensor<f32>)
// CHECK: ^bb0(%[[IN:.*]]: f32, %[[OUT:.*]]: f32):
// CHECK: %[[ADD2:.*]] = arith.addf %[[IN]], %[[OUT]] : f32
// CHECK: linalg.yield %[[ADD2]] : f32
// CHECK: } -> tensor<f32>
// CHECK: return %[[FINAL]] : tensor<f32>
func.func @test_addf_reduction(%arg0: tensor<2048xf32>) -> tensor<f32> {
  %cst = arith.constant 0.000000e+00 : f32
  %3 = bufferization.alloc_tensor() : tensor<f32>
  %inserted = tensor.insert %cst into %3[] : tensor<f32>

  %4 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg0 : tensor<2048xf32>) outs(%inserted : tensor<f32>) {
  ^bb0(%in: f32, %out: f32):
    %12 = arith.addf %in, %out : f32
    linalg.yield %12 : f32
  } -> tensor<f32>

  return %4 : tensor<f32>
}

