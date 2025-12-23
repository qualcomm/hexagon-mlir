// RUN: linalg-hexagon-opt %s | FileCheck %s

// CHECK: #[[$MAP_0:.+]] = affine_map<(d0) -> (d0 floordiv 4, d0 mod 4)>
// CHECK: #[[$MAP_1:.+]] = affine_map<(d0, d1) -> (d0 * 4 + d1)>

// CHECK-LABEL:   func.func @add(
// CHECK-SAME:                   %[[INP0:.*]]: memref<62xi8>,
// CHECK-SAME:                   %[[INP1:.*]]: memref<63xi8>,
// CHECK-SAME:                   %[[OUT:.*]]: memref<64xi8>) {
// CHECK:           %[[VAL_0:.*]] = hexagonmem.alloc() : memref<16x4xi8>
// CHECK:           %[[VAL_1:.*]] = hexagonmem.alloc() : memref<16x4xi8>
// CHECK:           %[[VAL_2:.*]] = hexagonmem.alloc() : memref<16x4xi8>
// CHECK:           hexagonmem.convert_layout ins(%[[INP0]] : memref<62xi8>), outs(%[[VAL_0]] : memref<16x4xi8>) {layout = #[[$MAP_0]], pad_value = 0 : i8}
// CHECK:           hexagonmem.convert_layout ins(%[[INP1]] : memref<63xi8>), outs(%[[VAL_1]] : memref<16x4xi8>) {layout = #[[$MAP_0]]}
// CHECK:           hexagonmem.convert_layout ins(%[[OUT]] : memref<64xi8>), outs(%[[VAL_2]] : memref<16x4xi8>) {layout = #[[$MAP_0]], pad_value = 0 : i8}
// CHECK:           linalg.add ins(%[[VAL_0]], %[[VAL_1]] : memref<16x4xi8>, memref<16x4xi8>) outs(%[[VAL_2]] : memref<16x4xi8>)
// CHECK:           hexagonmem.convert_layout ins(%[[VAL_2]] : memref<16x4xi8>), outs(%[[OUT]] : memref<64xi8>) {layout = #[[$MAP_1]]}
// CHECK:           return
// CHECK:         }

#map = affine_map<(d0) -> (d0 floordiv 4, d0 mod 4)>
#map1 = affine_map<(d0, d1) -> ((d0 * 4) + d1)>
func.func @add(%arg0: memref<62xi8>, %arg1: memref<63xi8>, %arg2: memref<64xi8>) -> () {
  // Convert arguments to expected layout
  %modified_arg0 = hexagonmem.alloc() : memref<16x4xi8>
  %modified_arg1 = hexagonmem.alloc() : memref<16x4xi8>
  %modified_arg2 = hexagonmem.alloc() : memref<16x4xi8>
  hexagonmem.convert_layout
                      ins(%arg0 : memref<62xi8>),
                      outs(%modified_arg0 : memref<16x4xi8>) {pad_value = 0:i8, layout = #map}
  hexagonmem.convert_layout
                      ins(%arg1 : memref<63xi8>),
                      outs(%modified_arg1 : memref<16x4xi8>) {layout = #map}
  hexagonmem.convert_layout
                      ins(%arg2 : memref<64xi8>),
                      outs(%modified_arg2 : memref<16x4xi8>) {pad_value = 0:i8, layout = #map}
  // Perform op on modified layout
  linalg.add ins(%modified_arg0, %modified_arg1 : memref<16x4xi8>, memref<16x4xi8>)
             outs(%modified_arg2: memref<16x4xi8>)
 
  // Convert output back to original layout in result
  hexagonmem.convert_layout
                      ins(%modified_arg2 : memref<16x4xi8>),
                      outs(%arg2 : memref<64xi8>) {layout=#map1}
  return
}
