// RUN: linalg-hexagon-opt %s -hexagonmem-to-llvm | FileCheck %s

// CHECK:  %[[BUFFER_PTR_FROM_MEMREF:.+]] = llvm.extractvalue
// CHECK:  %[[NUM_BYTES:.+]] = llvm.mlir.constant(32768 : i32) : i32
// CHECK:  %{{.*}} = llvm.call @hexagon_runtime_build_crouton_dsp(%[[BUFFER_PTR_FROM_MEMREF]], %[[NUM_BYTES]]) : (!llvm.ptr, i32) -> !llvm.ptr
// CHECK:   %{{.*}} = builtin.unrealized_conversion_cast %{{.*}} : !llvm.ptr to !crouton.crouton<2x4x2xf16, vtcm>

func.func @memref_to_crouton(%arg0: memref<2x4x2x1024xf16, 1>) -> !crouton.crouton<2x4x2xf16, vtcm> {
  %out = hexagonmem.memref_to_crouton (%arg0 : memref<2x4x2x1024xf16, 1>) : !crouton.crouton<2x4x2xf16, vtcm>
  return %out : !crouton.crouton<2x4x2xf16, vtcm>
}

