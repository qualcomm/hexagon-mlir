// RUN: linalg-hexagon-opt %s -split-input-file --pass-pipeline="builtin.module(erase-unused-linalg-operands)" | FileCheck %s

// CHECK-LABEL: func @extra_ins(
// CHECK-SAME: %[[X:.*]]: tensor<?xf32>, %[[Y:.*]]: tensor<?xf32>, %[[Z:.*]]: tensor<?xf32>,
// CHECK: %[[GENERIC_OP:.*]] = linalg.generic
// CHECK-SAME: ins(%[[X]], %[[Y]] : tensor<?xf32>, tensor<?xf32>)
#map = affine_map<(d0) -> (d0)>
func.func @extra_ins(%X: tensor<?xf32>, %Y: tensor<?xf32>, 
                                  %Z: tensor<?xf32>, %Out: tensor<?xf32>)
                        -> (tensor<?xf32>) {
  %res = linalg.generic 
              {indexing_maps = [#map, #map, #map, #map], 
               iterator_types=["parallel"]} 
               ins(%X, %Y, %Z : tensor<?xf32>, tensor<?xf32>, tensor<?xf32>) 
               outs (%Out: tensor<?xf32>) {
          ^bb0(%x: f32, %y: f32, %z: f32, %out: f32):
              %r = arith.addf %x, %y : f32
              linalg.yield %r : f32
          } -> tensor<?xf32>
  return %res : tensor<?xf32>
}
