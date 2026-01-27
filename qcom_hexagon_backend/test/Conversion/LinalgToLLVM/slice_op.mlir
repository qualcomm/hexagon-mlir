// RUN: linalg-hexagon-opt %s --hexagon-slicing="slicing-factor=4" | FileCheck %s -check-prefixes=CHECK


#map = affine_map<(d0) -> (d0)>
func.func @slice_vec_add(
  %arg0: tensor<1000xf32>, %arg1: tensor<1000xf32>, %arg2: tensor<1000xf32>) -> tensor<1000xf32> {
  %0 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%arg0, %arg1 : tensor<1000xf32>, tensor<1000xf32>) outs(%arg2 : tensor<1000xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %1 = arith.addf %in, %in_0 : f32
      linalg.yield %1 : f32
    } -> tensor<1000xf32>
    return %0 : tensor<1000xf32>
}

// CHECK-LABEL: slice_vec_add
// CHECK: %{{.*}} = tensor.extract_slice %{{.*}}[0] [256] [1] : tensor<1000xf32> to tensor<256xf32>
// CHECK: %[[SLICE1:.+]] = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%{{.*}}, {{.*}} : tensor<256xf32>, tensor<256xf32>) outs({{.*}} : tensor<256xf32>)
// CHECK: %{{.*}} = arith.addf %{{.*}}, %{{.*}} : f32
// CHECK: %{{.*}} = tensor.extract_slice %{{.*}}[256] [256] [1] : tensor<1000xf32> to tensor<256xf32>
// CHECK: %[[SLICE2:.+]] = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%{{.*}}, {{.*}} : tensor<256xf32>, tensor<256xf32>) outs({{.*}} : tensor<256xf32>)
// CHECK: %{{.*}} = arith.addf %{{.*}}, %{{.*}} : f32
// CHECK: %{{.*}} = tensor.extract_slice %{{.*}}[512] [256] [1] : tensor<1000xf32> to tensor<256xf32>
// CHECK: %[[SLICE3:.+]] = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%{{.*}}, {{.*}} : tensor<256xf32>, tensor<256xf32>) outs({{.*}} : tensor<256xf32>)
// CHECK: %{{.*}} = arith.addf %{{.*}}, %{{.*}} : f32
// CHECK: %{{.*}} = tensor.extract_slice %{{.*}}[768] [232] [1] : tensor<1000xf32> to tensor<232xf32>
// CHECK: %[[SLICE4:.+]] = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%{{.*}}, {{.*}} : tensor<232xf32>, tensor<232xf32>) outs({{.*}} : tensor<232xf32>)
// CHECK: %{{.*}} = arith.addf %{{.*}}, %{{.*}} : f32
// CHECK: %[[CONCAT:.+]] = tensor.concat dim(0) %[[SLICE1]], %[[SLICE2]], %[[SLICE3]], %[[SLICE4]] : (tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<232xf32>) -> tensor<1000xf32>
// CHECK: return %[[CONCAT]] : tensor<1000xf32>

