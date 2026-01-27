// RUN: linalg-hexagon-opt %s | FileCheck %s

// CHECK: #[[$MAP_0:.+]] = affine_map<(d0, d1) -> (d0 floordiv 4, d1 floordiv 4, d0 mod 4, d1 mod 4)>
// CHECK: #[[$MAP_1:.+]] = affine_map<(d0, d1, d2, d3) -> (d0 * 4 + d2, d1 * 4 + d3)>
// CHECK-LABEL:   func.func @Division2D(
// CHECK-SAME:      %[[INP:.*]]: memref<64x64xi8>) -> memref<64x64xi8> {
// CHECK:           %[[VAL:.*]] = memref.alloc() : memref<16x16x4x4xi8>
// CHECK:           hexagonmem.convert_layout ins(%[[INP]] : memref<64x64xi8>), outs(%[[VAL]] : memref<16x16x4x4xi8>) {layout = #[[$MAP_0]]}
// CHECK:           %[[RES:.*]] = memref.alloc() : memref<64x64xi8>
// CHECK:           hexagonmem.convert_layout ins(%[[VAL]] : memref<16x16x4x4xi8>), outs(%[[RES]] : memref<64x64xi8>) {layout = #[[$MAP_1]]}
// CHECK:           return %[[RES]] : memref<64x64xi8>
// CHECK:         }

#map = affine_map<(d0, d1) -> (d0 floordiv 4, d1 floordiv 4, d0 mod 4, d1 mod 4)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0*4+d2, d1*4+d3)>
func.func @Division2D(%arg0: memref<64x64xi8>) -> memref<64x64xi8>{
  %modified_arg0 = memref.alloc() : memref<16x16x4x4xi8>
  hexagonmem.convert_layout
                      ins(%arg0 : memref<64x64xi8>), 
                      outs(%modified_arg0 : memref<16x16x4x4xi8>) {layout = #map}
  %modified_arg1 = memref.alloc() : memref<64x64xi8>
  hexagonmem.convert_layout
                      ins(%modified_arg0 : memref<16x16x4x4xi8>),
                      outs(%modified_arg1 : memref<64x64xi8>) {layout = #map1}
  return %modified_arg1 : memref<64x64xi8>
}
