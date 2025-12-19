// RUN: linalg-hexagon-opt %s -convert-zero-size-memref | FileCheck %s

func.func @convert_zero_size_memref() {
  %alloc = memref.alloc() : memref<0xi1>
  memref.dealloc %alloc : memref<0xi1>
  return
}

// CHECK-LABEL: @convert_zero_size_memref
// CHECK:      %[[ALLOC:.+]] = memref.alloc() : memref<1xi1>
// CHECK:      memref.dealloc %[[ALLOC]] : memref<1xi1>
