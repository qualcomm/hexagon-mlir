// RUN: linalg-hexagon-opt -split-input-file %s -linalg-to-llvm="split-tiling-range=false enable-split-reduction=false" | FileCheck %s

#map = affine_map<(d0,d1) -> (d0,d1)>
func.func @vecadd_40by40(
  %arg0: tensor<40x40xf32>, %arg1: tensor<40x40xf32>, %arg2: tensor<40x40xf32>) -> tensor<40x40xf32> {
  %0 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0, %arg1 : tensor<40x40xf32>, tensor<40x40xf32>) outs(%arg2 : tensor<40x40xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %1 = arith.addf %in, %in_0 : f32
      linalg.yield %1 : f32
    } -> tensor<40x40xf32>
    return %0 : tensor<40x40xf32>
}
// CHECK-LABEL: @vecadd_40by40
// CHECK: %{{[0-9]+}} = llvm.fadd {{.*}}, {{.*}}  {fastmathFlags = #llvm.fastmath<fast>} : vector<32xf32>

// -----

func.func @matmul(
  %arg0: tensor<?x1000xf32>, %arg1: tensor<?x1000xf32>, %arg2: tensor<?x1000xf32>)
    -> tensor<?x1000xf32> {
  %0 = linalg.matmul  ins(%arg0, %arg1: tensor<?x1000xf32>, tensor<?x1000xf32>)
                     outs(%arg2: tensor<?x1000xf32>)
    -> tensor<?x1000xf32>
  return %0 : tensor<?x1000xf32>
}

// CHECK-LABEL: @matmul
// CHECK: %{{[0-9]+}} = llvm.fmul %{{[0-9]+}}, %{{[0-9]+}}  {fastmathFlags = #llvm.fastmath<fast>} : vector<32xf32>
