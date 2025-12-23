// RUN: linalg-hexagon-opt %s | FileCheck %s

// CHECK:   func.func @crouton_to_memref(%[[VAL0:.+]]: !crouton.crouton<2x4x2xf16, vtcm>) -> memref<2x4x2x1024xf16, 1> {
// CHECK:     %[[VAL1:.+]] = hexagonmem.crouton_to_memref(%[[VAL0]] : <2x4x2xf16, vtcm>) : memref<2x4x2x1024xf16, 1>
// CHECK:     return %[[VAL1:.+]] : memref<2x4x2x1024xf16, 1>
// CHECK:   }

func.func @crouton_to_memref(%arg0: !crouton.crouton<2x4x2xf16, vtcm>) -> memref<2x4x2x1024xf16, 1> {
  %out = hexagonmem.crouton_to_memref (%arg0 : !crouton.crouton<2x4x2xf16, vtcm>) : memref<2x4x2x1024xf16, 1>
  return %out : memref<2x4x2x1024xf16, 1>
}

