// RUN: linalg-hexagon-opt %s | FileCheck %s

// CHECK-LABEL:   func.func @memref_to_crouton(
// CHECK-SAME:                                 %[[VAL_0:.*]]: memref<2x4x2x1024xf16, 1>) -> !crouton.crouton<2x4x2xf16, vtcm> {
// CHECK:           %[[VAL_1:.*]] = hexagonmem.memref_to_crouton(%[[VAL_0]] : memref<2x4x2x1024xf16, 1>) : !crouton.crouton<2x4x2xf16, vtcm>
// CHECK:           return %[[VAL_1]] : !crouton.crouton<2x4x2xf16, vtcm>
// CHECK:         }
func.func @memref_to_crouton(%arg0: memref<2x4x2x1024xf16, 1>) -> !crouton.crouton<2x4x2xf16, vtcm> {
  %out = hexagonmem.memref_to_crouton (%arg0 : memref<2x4x2x1024xf16, 1>) : !crouton.crouton<2x4x2xf16, vtcm>
  return %out : !crouton.crouton<2x4x2xf16, vtcm>
}
