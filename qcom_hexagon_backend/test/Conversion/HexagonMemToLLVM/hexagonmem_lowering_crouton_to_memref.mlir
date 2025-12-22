// RUN: linalg-hexagon-opt %s -hexagonmem-to-llvm | FileCheck %s

// CHECK: func.func @crouton_to_memref(%[[ARG0:.+]]: !crouton.crouton<2x4x2xf16, vtcm>)
// CHECK: %[[CROUTON_TO_PTR:.+]] = builtin.unrealized_conversion_cast %[[ARG0]] : !crouton.crouton<2x4x2xf16, vtcm> to !llvm.ptr
// CHECK: %[[MEMREF_PTR:.+]] = llvm.call @hexagon_rutime_get_contiguous_memref(%[[CROUTON_TO_PTR]]) : (!llvm.ptr) -> !llvm.ptr
// CHECK: {{.*}} = llvm.insertvalue %[[MEMREF_PTR]], {{.*}}[0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK: {{.*}} = llvm.insertvalue %[[MEMREF_PTR]], {{.*}}[1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>

func.func @crouton_to_memref(%arg0: !crouton.crouton<2x4x2xf16, vtcm>) -> memref<2x4x2x1024xf16, 1> {
  %out = hexagonmem.crouton_to_memref (%arg0 : !crouton.crouton<2x4x2xf16, vtcm>) : memref<2x4x2x1024xf16, 1>
  return %out : memref<2x4x2x1024xf16, 1>
}

