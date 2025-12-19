// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(linalg-generalize-named-ops,\
// RUN:   func.func(vtcm-tiling),canonicalize,\
// RUN:   one-shot-bufferize{bufferize-function-boundaries allow-return-allocs-from-loops},\
// RUN:   func.func(buffer-loop-hoisting),canonicalize,buffer-deallocation-pipeline)' | FileCheck %s

module {
  func.func @crouton_constraints_i8(%in1: memref<4x3x2x1024x8x8x32xi8>,%in2 : memref<4x3x2x1024x8x8x32xi8>,%result: memref<4x3x2x1024x8x8x32xi8>) {
    %crouton_input1 = bufferization.to_tensor %in1 restrict : memref<4x3x2x1024x8x8x32xi8> to tensor<4x3x2x1024x8x8x32xi8>
    %crouton_input2 = bufferization.to_tensor %in2 restrict : memref<4x3x2x1024x8x8x32xi8> to tensor<4x3x2x1024x8x8x32xi8>
    %crouton_out = tensor.empty() : tensor<4x3x2x1024x8x8x32xi8>
    %add = linalg.add ins(%crouton_input1, %crouton_input2 : tensor<4x3x2x1024x8x8x32xi8>, tensor<4x3x2x1024x8x8x32xi8>) outs(%crouton_out: tensor<4x3x2x1024x8x8x32xi8>) -> tensor<4x3x2x1024x8x8x32xi8>
    bufferization.materialize_in_destination %add in writable %result : (tensor<4x3x2x1024x8x8x32xi8>, memref<4x3x2x1024x8x8x32xi8>) -> ()
    return
  }
}

// CHECK-LABEL:   func.func @crouton_constraints_i8
// CHECK-SAME:    %[[X:.+]]: memref<4x3x2x1024x8x8x32xi8>, %[[Y:.+]]: memref<4x3x2x1024x8x8x32xi8>, %[[Z:.+]]: memref<4x3x2x1024x8x8x32xi8>

// CHECK:           %[[ALLOC_DDR:.*]] = memref.alloc() {alignment = 64 : i64} : memref<4x3x2x1024x8x8x32xi8>
// CHECK:           %[[ALLOC_INPUT1:.*]] = memref.alloc() {alignment = 64 : i64} : memref<4x1x1x64x8x8x32xi8, 1>
// CHECK:           %[[ALLOC_INPUT2:.*]] = memref.alloc() {alignment = 64 : i64} : memref<4x1x1x64x8x8x32xi8, 1>
// CHECK:           %[[ALLOC_OUTPUT:.*]] = memref.alloc() {alignment = 64 : i64} : memref<4x1x1x64x8x8x32xi8, 1>

// CHECK:           scf.for %[[I:.*]] = %c0 to %c3 step %c1 {
// CHECK-NEXT:        scf.for %[[J:.*]] = %c0 to %c2 step %c1 {
// CHECK-NEXT:          scf.for %[[K:.*]] = %c0 to %c1024 step %c64 {

// CHECK:                 %[[INPUT1_SUBVIEW:.*]] = memref.subview %[[X]][0, %[[I]], %[[J]], %[[K]], 0, 0, 0] [4, 1, 1, 64, 8, 8, 32] [1, 1, 1, 1, 1, 1, 1]
// CHECK-SAME:            : memref<4x3x2x1024x8x8x32xi8> to memref<4x1x1x64x8x8x32xi8, strided<[12582912, 4194304, 2097152, 2048, 256, 32, 1], offset: ?>>
// CHECK:                 memref.copy %[[INPUT1_SUBVIEW]], %[[ALLOC_INPUT1]] : memref<4x1x1x64x8x8x32xi8, strided<[12582912, 4194304, 2097152, 2048, 256, 32, 1], offset: ?>> to memref<4x1x1x64x8x8x32xi8, 1>
// CHECK:                 %[[INPUT2_SUBVIEW:.*]] = memref.subview %[[Y]][0, %[[I]], %[[J]], %[[K]], 0, 0, 0] [4, 1, 1, 64, 8, 8, 32] [1, 1, 1, 1, 1, 1, 1]
// CHECK-SAME:            : memref<4x3x2x1024x8x8x32xi8> to memref<4x1x1x64x8x8x32xi8, strided<[12582912, 4194304, 2097152, 2048, 256, 32, 1], offset: ?>>
// CHECK:                 memref.copy %[[INPUT2_SUBVIEW]], %[[ALLOC_INPUT2]] : memref<4x1x1x64x8x8x32xi8, strided<[12582912, 4194304, 2097152, 2048, 256, 32, 1], offset: ?>> to memref<4x1x1x64x8x8x32xi8, 1>
// CHECK:                 %[[OUTPUT_SUBVIEW:.*]] = memref.subview %[[ALLOC_DDR]][0, %[[I]], %[[J]], %[[K]], 0, 0, 0] [4, 1, 1, 64, 8, 8, 32] [1, 1, 1, 1, 1, 1, 1] 
// CHECK-SAME:            : memref<4x3x2x1024x8x8x32xi8> to memref<4x1x1x64x8x8x32xi8, strided<[12582912, 4194304, 2097152, 2048, 256, 32, 1], offset: ?>>
// CHECK:                 memref.copy %[[OUTPUT_SUBVIEW]], %[[ALLOC_OUTPUT]] : memref<4x1x1x64x8x8x32xi8, strided<[12582912, 4194304, 2097152, 2048, 256, 32, 1], offset: ?>> to memref<4x1x1x64x8x8x32xi8, 1>
// CHECK:                 linalg.generic {{.*}} ins(%[[ALLOC_INPUT1]], %[[ALLOC_INPUT2]] : memref<4x1x1x64x8x8x32xi8, 1>, memref<4x1x1x64x8x8x32xi8, 1>) outs(%[[ALLOC_OUTPUT]] : memref<4x1x1x64x8x8x32xi8, 1>) {
// CHECK:                 memref.copy %[[ALLOC_OUTPUT]], %[[OUTPUT_SUBVIEW]] : memref<4x1x1x64x8x8x32xi8, 1> to memref<4x1x1x64x8x8x32xi8, strided<[12582912, 4194304, 2097152, 2048, 256, 32, 1], offset: ?>>

// CHECK:           memref.copy %[[ALLOC_DDR]], %[[Z]] : memref<4x3x2x1024x8x8x32xi8> to memref<4x3x2x1024x8x8x32xi8>
// CHECK:           memref.dealloc %[[ALLOC_DDR]] : memref<4x3x2x1024x8x8x32xi8>
// CHECK:           memref.dealloc %[[ALLOC_INPUT1]] : memref<4x1x1x64x8x8x32xi8, 1>
// CHECK:           memref.dealloc %[[ALLOC_INPUT2]] : memref<4x1x1x64x8x8x32xi8, 1>
// CHECK:           memref.dealloc %[[ALLOC_OUTPUT]] : memref<4x1x1x64x8x8x32xi8, 1>
