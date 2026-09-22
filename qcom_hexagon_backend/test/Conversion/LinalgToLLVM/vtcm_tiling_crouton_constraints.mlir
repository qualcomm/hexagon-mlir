// RUN: linalg-hexagon-opt %s -split-input-file -pass-pipeline='builtin.module(linalg-generalize-named-ops,\
// RUN:   func.func(vtcm-tiling),canonicalize,\
// RUN:   one-shot-bufferize{bufferize-function-boundaries allow-return-allocs-from-loops},\
// RUN:   func.func(buffer-loop-hoisting),canonicalize,buffer-deallocation-pipeline)' | FileCheck %s
//
// Two cases on the same rank-7 int8 crouton-shaped iteration space:
//   1. a pure parallel + identity generic  => the streaming skip: no VTCM
//      staging, the op runs straight on DDR;
//   2. a non-streaming generic (one reduction loop) => VTCM staging, whose
//      tile sizes must respect the crouton lower bounds on the last three
//      loops (8 x 8 x 32).

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

// `linalg.add` generalizes to an all-parallel identity_map generic, so this is
// the streaming case: VTCMTiling deliberately skips staging and the add runs
// straight on the DDR memrefs (no space-1 buffers, no tile loops).
// CHECK-LABEL:   func.func @crouton_constraints_i8(
// CHECK-SAME:    %[[X:.+]]: memref<4x3x2x1024x8x8x32xi8>, %[[Y:.+]]: memref<4x3x2x1024x8x8x32xi8>, %[[Z:.+]]: memref<4x3x2x1024x8x8x32xi8>
// CHECK:           %[[ALLOC:.*]] = memref.alloc() {alignment = 64 : i64} : memref<4x3x2x1024x8x8x32xi8>
// CHECK:           linalg.generic {{.*}} ins(%[[X]], %[[Y]] : memref<4x3x2x1024x8x8x32xi8>, memref<4x3x2x1024x8x8x32xi8>) outs(%[[ALLOC]] : memref<4x3x2x1024x8x8x32xi8>) {
// CHECK:           memref.copy %[[ALLOC]], %[[Z]] : memref<4x3x2x1024x8x8x32xi8> to memref<4x3x2x1024x8x8x32xi8>
// CHECK:           memref.dealloc %[[ALLOC]] : memref<4x3x2x1024x8x8x32xi8>

// -----

#map_id = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3, d4, d5, d6)>
#map_red = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d4, d5, d6)>
module {
  // The d3 loop is a reduction, so this generic is not streaming and VTCM
  // staging fires. The crouton lower bounds leave the last three loops at
  // 8 x 8 x 32 and only split the d3 = 1024 dimension (step 64).
  func.func @crouton_constraints_i8_reduction(%in1: memref<4x3x2x1024x8x8x32xi8>, %in2: memref<4x3x2x1024x8x8x32xi8>, %result: memref<4x3x2x8x8x32xi8>) {
    %crouton_input1 = bufferization.to_tensor %in1 restrict : memref<4x3x2x1024x8x8x32xi8> to tensor<4x3x2x1024x8x8x32xi8>
    %crouton_input2 = bufferization.to_tensor %in2 restrict : memref<4x3x2x1024x8x8x32xi8> to tensor<4x3x2x1024x8x8x32xi8>
    %crouton_out = tensor.empty() : tensor<4x3x2x8x8x32xi8>
    %sum = linalg.generic {
        indexing_maps = [#map_id, #map_id, #map_red],
        iterator_types = ["parallel", "parallel", "parallel", "reduction", "parallel", "parallel", "parallel"]}
        ins(%crouton_input1, %crouton_input2 : tensor<4x3x2x1024x8x8x32xi8>, tensor<4x3x2x1024x8x8x32xi8>)
        outs(%crouton_out : tensor<4x3x2x8x8x32xi8>) {
    ^bb0(%in: i8, %in_2: i8, %out: i8):
      %4 = arith.addi %in, %in_2 : i8
      %5 = arith.addi %out, %4 : i8
      linalg.yield %5 : i8
    } -> tensor<4x3x2x8x8x32xi8>
    bufferization.materialize_in_destination %sum in writable %result : (tensor<4x3x2x8x8x32xi8>, memref<4x3x2x8x8x32xi8>) -> ()
    return
  }
}

// Non-streaming => staged through VTCM: both inputs and the accumulator get
// space-1 allocs (", 1>") copied in, the generic runs on those, and the result
// is copied back out. The 8 x 8 x 32 crouton tail stays untiled.
// CHECK-LABEL:   func.func @crouton_constraints_i8_reduction(
// CHECK-SAME:    %[[X:.+]]: memref<4x3x2x1024x8x8x32xi8>, %[[Y:.+]]: memref<4x3x2x1024x8x8x32xi8>, %[[Z:.+]]: memref<4x3x2x8x8x32xi8>

// CHECK:           %[[ALLOC_DDR:.*]] = memref.alloc() {alignment = 64 : i64} : memref<4x3x2x8x8x32xi8>
// CHECK:           %[[ALLOC_IN1:.*]] = memref.alloc() {alignment = 64 : i64} : memref<4x1x1x64x8x8x32xi8, 1>
// CHECK:           %[[ALLOC_IN2:.*]] = memref.alloc() {alignment = 64 : i64} : memref<4x1x1x64x8x8x32xi8, 1>
// CHECK:           %[[ALLOC_OUT:.*]] = memref.alloc() {alignment = 64 : i64} : memref<4x1x1x8x8x32xi8, 1>

// CHECK:           scf.for %[[I:.*]] = %c0 to %c3 step %c1 {
// CHECK-NEXT:        scf.for %[[J:.*]] = %c0 to %c2 step %c1 {
// CHECK-NEXT:          scf.for %[[K:.*]] = %c0 to %c1024 step %c64 {

// CHECK:                 %[[IN1_SUBVIEW:.*]] = memref.subview %[[X]][0, %[[I]], %[[J]], %[[K]], 0, 0, 0] [4, 1, 1, 64, 8, 8, 32] [1, 1, 1, 1, 1, 1, 1]
// CHECK-SAME:            : memref<4x3x2x1024x8x8x32xi8> to memref<4x1x1x64x8x8x32xi8, strided<[12582912, 4194304, 2097152, 2048, 256, 32, 1], offset: ?>>
// CHECK:                 memref.copy %[[IN1_SUBVIEW]], %[[ALLOC_IN1]] : memref<4x1x1x64x8x8x32xi8, strided<[12582912, 4194304, 2097152, 2048, 256, 32, 1], offset: ?>> to memref<4x1x1x64x8x8x32xi8, 1>
// CHECK:                 %[[IN2_SUBVIEW:.*]] = memref.subview %[[Y]][0, %[[I]], %[[J]], %[[K]], 0, 0, 0] [4, 1, 1, 64, 8, 8, 32] [1, 1, 1, 1, 1, 1, 1]
// CHECK-SAME:            : memref<4x3x2x1024x8x8x32xi8> to memref<4x1x1x64x8x8x32xi8, strided<[12582912, 4194304, 2097152, 2048, 256, 32, 1], offset: ?>>
// CHECK:                 memref.copy %[[IN2_SUBVIEW]], %[[ALLOC_IN2]] : memref<4x1x1x64x8x8x32xi8, strided<[12582912, 4194304, 2097152, 2048, 256, 32, 1], offset: ?>> to memref<4x1x1x64x8x8x32xi8, 1>
// CHECK:                 %[[OUT_SUBVIEW:.*]] = memref.subview %[[ALLOC_DDR]][0, %[[I]], %[[J]], 0, 0, 0] [4, 1, 1, 8, 8, 32] [1, 1, 1, 1, 1, 1]
// CHECK-SAME:            : memref<4x3x2x8x8x32xi8> to memref<4x1x1x8x8x32xi8, strided<[12288, 4096, 2048, 256, 32, 1], offset: ?>>
// CHECK:                 memref.copy %[[OUT_SUBVIEW]], %[[ALLOC_OUT]] : memref<4x1x1x8x8x32xi8, strided<[12288, 4096, 2048, 256, 32, 1], offset: ?>> to memref<4x1x1x8x8x32xi8, 1>
// CHECK:                 linalg.generic {{.*}} ins(%[[ALLOC_IN1]], %[[ALLOC_IN2]] : memref<4x1x1x64x8x8x32xi8, 1>, memref<4x1x1x64x8x8x32xi8, 1>) outs(%[[ALLOC_OUT]] : memref<4x1x1x8x8x32xi8, 1>) {
// CHECK:                 memref.copy %[[ALLOC_OUT]], %[[OUT_SUBVIEW]] : memref<4x1x1x8x8x32xi8, 1> to memref<4x1x1x8x8x32xi8, strided<[12288, 4096, 2048, 256, 32, 1], offset: ?>>

// CHECK:           memref.copy %[[ALLOC_DDR]], %[[Z]] : memref<4x3x2x8x8x32xi8> to memref<4x3x2x8x8x32xi8>
// CHECK:           memref.dealloc %[[ALLOC_DDR]] : memref<4x3x2x8x8x32xi8>
// CHECK:           memref.dealloc %[[ALLOC_IN1]] : memref<4x1x1x64x8x8x32xi8, 1>
// CHECK:           memref.dealloc %[[ALLOC_IN2]] : memref<4x1x1x64x8x8x32xi8, 1>
// CHECK:           memref.dealloc %[[ALLOC_OUT]] : memref<4x1x1x8x8x32xi8, 1>
