// RUN: linalg-hexagon-opt %s | linalg-hexagon-opt | FileCheck %s

// CHECK-LABEL: func.func @add
// CHECK: [[ALLOC0:%.*]] = hexagonmem.alloc() {alignment = 2048 : i64} : !crouton.crouton<64xi8, vtcm>
// CHECK: [[ALLOC1:%.*]] = hexagonmem.alloc() {alignment = 2048 : i64} : !crouton.crouton<64xi8, vtcm>
// CHECK: [[ALLOC2:%.*]] = hexagonmem.alloc() {alignment = 2048 : i64} : !crouton.crouton<64xi8, vtcm>
// CHECK: hexagonmem.copy [[ARG0:%.*]], [[ALLOC0]] : !crouton.crouton<64xi8> to !crouton.crouton<64xi8, vtcm>
// CHECK: hexagonmem.copy [[ARG1:%.*]], [[ALLOC1]] : !crouton.crouton<64xi8> to !crouton.crouton<64xi8, vtcm>
// CHECK: hexagonmem.copy [[ALLOC2]], [[ARG2:%.*]] : !crouton.crouton<64xi8, vtcm> to !crouton.crouton<64xi8>
// CHECK: hexagonmem.dealloc [[ALLOC0]] : !crouton.crouton<64xi8, vtcm>
// CHECK: hexagonmem.dealloc [[ALLOC1]] : !crouton.crouton<64xi8, vtcm>
// CHECK: hexagonmem.dealloc [[ALLOC2]] : !crouton.crouton<64xi8, vtcm>

// Just test the 3 major types of ops introduced
// (hexagonmem.alloc, hexagonmem.copy, hexagonmem.dealloc) with crouton types
func.func @add(%arg0: !crouton.crouton<64xi8>, %arg1: !crouton.crouton<64xi8>, %arg2: !crouton.crouton<64xi8>) {
  %0 = hexagonmem.alloc() {alignment = 2048 : i64} : !crouton.crouton<64xi8, vtcm>
  %1 = hexagonmem.alloc() {alignment = 2048 : i64} : !crouton.crouton<64xi8, vtcm>
  %2 = hexagonmem.alloc() {alignment = 2048 : i64} : !crouton.crouton<64xi8, vtcm>
  hexagonmem.copy %arg0, %0 : !crouton.crouton<64xi8> to !crouton.crouton<64xi8, vtcm>
  hexagonmem.copy %arg1, %1 : !crouton.crouton<64xi8> to !crouton.crouton<64xi8, vtcm>
  // This is not yet supported
  // linalg.add ins(%0, %1 : !crouton.crouton<64xi8, vtcm>, !crouton.crouton<64xi8, vtcm>) outs(%2 : !crouton.crouton<64xi8, vtcm>)
  hexagonmem.copy %2, %arg2 : !crouton.crouton<64xi8, vtcm> to !crouton.crouton<64xi8>
  hexagonmem.dealloc %0 : !crouton.crouton<64xi8, vtcm>
  hexagonmem.dealloc %1 : !crouton.crouton<64xi8, vtcm>
  hexagonmem.dealloc %2 : !crouton.crouton<64xi8, vtcm>
  return
}
