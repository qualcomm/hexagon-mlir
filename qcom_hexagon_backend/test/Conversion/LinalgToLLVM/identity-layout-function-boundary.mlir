//===- identity-layout-function-boundary.mlir -----------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

// RUN: linalg-hexagon-opt %s -split-input-file -linalg-to-llvm \
// RUN:   -mlir-print-ir-after=one-shot-bufferize -o /dev/null 2>&1 \
// RUN: | FileCheck %s
// RUN: linalg-hexagon-opt %s -split-input-file -linalg-to-llvm \
// RUN: | FileCheck %s --check-prefix=LOWER

// The pipeline configures One-Shot Bufferize with
// functionBoundaryTypeConversion = IdentityLayoutMap, so every ranked tensor
// at a func.func boundary becomes a memref with no layout map. Without that
// option the default InferLayoutMap gives arguments a fully dynamic layout
// (`memref<..., strided<[?, ...], offset: ?>>`) and infers result layouts from
// the body, so each signature check below fails if the option is dropped.
//
// The second RUN line checks that the identity-layout boundaries still lower
// all the way to the LLVM dialect.

// -----
// Static ranked tensors on a public entry function and on the private helper
// it calls. Both boundaries and the call site become identity-layout memrefs;
// the scalar operand is left alone.

#map = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @entry(%arg0: tensor<2x3xf32>, %scale: f32) -> tensor<2x3xf32> {
    %0 = call @helper(%arg0) : (tensor<2x3xf32>) -> tensor<2x3xf32>
    %1 = tensor.empty() : tensor<2x3xf32>
    %2 = linalg.generic {indexing_maps = [#map, #map],
                         iterator_types = ["parallel", "parallel"]}
         ins(%0 : tensor<2x3xf32>) outs(%1 : tensor<2x3xf32>) {
    ^bb0(%in: f32, %out: f32):
      %m = arith.mulf %in, %scale : f32
      linalg.yield %m : f32
    } -> tensor<2x3xf32>
    return %2 : tensor<2x3xf32>
  }
  func.func private @helper(%v: tensor<2x3xf32>) -> tensor<2x3xf32> {
    return %v : tensor<2x3xf32>
  }
}

// CHECK-LABEL: func.func @entry(
// CHECK-SAME:    %[[ARG0:.*]]: memref<2x3xf32>, %{{.*}}: f32) -> memref<2x3xf32>
// CHECK:         call @helper(%[[ARG0]]) : (memref<2x3xf32>) -> memref<2x3xf32>
// CHECK-LABEL: func.func private @helper(
// CHECK-SAME:    %{{.*}}: memref<2x3xf32>) -> memref<2x3xf32>

// LOWER: llvm.func @entry(
// LOWER: llvm.func @helper(

// -----
// Dynamic ranked tensors. The identity contract applies to the layout, not to
// the extents: the dynamic dimensions stay dynamic and no layout map appears.

#map = affine_map<(d0, d1) -> (d0, d1)>
func.func @dynamic_shape(%a: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %d0 = tensor.dim %a, %c0 : tensor<?x?xf32>
  %d1 = tensor.dim %a, %c1 : tensor<?x?xf32>
  %e = tensor.empty(%d0, %d1) : tensor<?x?xf32>
  %r = linalg.generic {indexing_maps = [#map, #map],
                       iterator_types = ["parallel", "parallel"]}
       ins(%a : tensor<?x?xf32>) outs(%e : tensor<?x?xf32>) {
  ^bb0(%in: f32, %o: f32):
    %m = arith.mulf %in, %in : f32
    linalg.yield %m : f32
  } -> tensor<?x?xf32>
  return %r : tensor<?x?xf32>
}

// CHECK-LABEL: func.func @dynamic_shape(
// CHECK-SAME:    %{{.*}}: memref<?x?xf32>) -> memref<?x?xf32>

// LOWER: llvm.func @dynamic_shape(

// -----
// Unranked tensors are unaffected: an unranked memref cannot carry a layout
// map, so IdentityLayoutMap and the default both produce `memref<*xf32>`.

func.func @unranked_unchanged(%a: tensor<*xf32>) -> tensor<*xf32> {
  return %a : tensor<*xf32>
}

// CHECK-LABEL: func.func @unranked_unchanged(
// CHECK-SAME:    %{{.*}}: memref<*xf32>) -> memref<*xf32>

// LOWER: llvm.func @unranked_unchanged(

// -----
// Destination-passing style: the output tensor is an argument as well, so both
// the input and the output boundary must be identity-layout memrefs.

#map = affine_map<(d0, d1) -> (d0, d1)>
func.func @destination_passing(%a: tensor<8x16xf32>,
                               %out: tensor<8x16xf32>) -> tensor<8x16xf32> {
  %r = linalg.generic {indexing_maps = [#map, #map],
                       iterator_types = ["parallel", "parallel"]}
       ins(%a : tensor<8x16xf32>) outs(%out : tensor<8x16xf32>) {
  ^bb0(%in: f32, %o: f32):
    %m = arith.addf %in, %o : f32
    linalg.yield %m : f32
  } -> tensor<8x16xf32>
  return %r : tensor<8x16xf32>
}

// CHECK-LABEL: func.func @destination_passing(
// CHECK-SAME:    %{{.*}}: memref<8x16xf32>, %{{.*}}: memref<8x16xf32>) -> memref<8x16xf32>

// LOWER: llvm.func @destination_passing(

// -----
// A returned slice is the case where the identity contract costs a copy: the
// subview has a non-identity layout that cannot be cast away, so the result is
// materialized into a fresh identity-layout buffer. Under the default
// InferLayoutMap the strided subview would be returned directly.

func.func @slice_result(%a: tensor<8x16xf32>) -> tensor<4x8xf32> {
  %s = tensor.extract_slice %a[2, 4] [4, 8] [1, 1]
     : tensor<8x16xf32> to tensor<4x8xf32>
  return %s : tensor<4x8xf32>
}

// CHECK-LABEL: func.func @slice_result(
// CHECK-SAME:    %[[ARG0:.*]]: memref<8x16xf32>) -> memref<4x8xf32>
// CHECK:         %[[SV:.*]] = memref.subview %[[ARG0]][2, 4] [4, 8] [1, 1] : memref<8x16xf32> to memref<4x8xf32, strided<[16, 1], offset: 36>>
// CHECK:         %[[ALLOC:.*]] = memref.alloc() {{.*}} : memref<4x8xf32>
// CHECK:         memref.copy %[[SV]], %[[ALLOC]]
// CHECK:         return %[[ALLOC]] : memref<4x8xf32>

// LOWER: llvm.func @slice_result(

// -----
// A higher-rank shape with unit-extent dimensions, where adjacent row-major
// strides repeat. The boundary is identity-layout; strided views created inside
// the body are unaffected, because the contract is a signature-level promise
// only.

#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
func.func @unit_dims_4d(%a: tensor<2x1x3x512xf32>,
                        %out: tensor<2x1x3x512xf32>) -> tensor<2x1x3x512xf32> {
  %r = linalg.generic {indexing_maps = [#map, #map],
                       iterator_types = ["parallel", "parallel", "parallel",
                                         "parallel"]}
       ins(%a : tensor<2x1x3x512xf32>) outs(%out : tensor<2x1x3x512xf32>) {
  ^bb0(%in: f32, %o: f32):
    %m = arith.mulf %in, %in : f32
    linalg.yield %m : f32
  } -> tensor<2x1x3x512xf32>
  return %r : tensor<2x1x3x512xf32>
}

// CHECK-LABEL: func.func @unit_dims_4d(
// CHECK-SAME:    %{{.*}}: memref<2x1x3x512xf32>, %{{.*}}: memref<2x1x3x512xf32>) -> memref<2x1x3x512xf32>
// CHECK:         memref.subview {{.*}} to memref<32xf32, strided<[1], offset: ?>>

// LOWER: llvm.func @unit_dims_4d(

// -----
// Functions that already take memrefs are not rewritten: an argument that was
// written with an explicit strided layout keeps it, because the option only
// controls how tensor boundaries are converted.

#map = affine_map<(d0) -> (d0)>
func.func @preexisting_memref_args(%arg0: memref<64xf32, strided<[?], offset: ?>>,
                                   %arg1: memref<64xf32>) {
  %t = bufferization.to_tensor %arg0 restrict
     : memref<64xf32, strided<[?], offset: ?>> to tensor<64xf32>
  %e = tensor.empty() : tensor<64xf32>
  %r = linalg.generic {indexing_maps = [#map, #map],
                       iterator_types = ["parallel"]}
       ins(%t : tensor<64xf32>) outs(%e : tensor<64xf32>) {
  ^bb0(%in: f32, %o: f32):
    %m = arith.addf %in, %in : f32
    linalg.yield %m : f32
  } -> tensor<64xf32>
  bufferization.materialize_in_destination %r in writable %arg1
    : (tensor<64xf32>, memref<64xf32>) -> ()
  return
}

// CHECK-LABEL: func.func @preexisting_memref_args(
// CHECK-SAME:    %{{.*}}: memref<64xf32, strided<[?], offset: ?>>, %{{.*}}: memref<64xf32>)

// LOWER: llvm.func @preexisting_memref_args(

// -----
// Multiple tensor results: every result of the boundary is converted, not just
// the first one.

#map = affine_map<(d0) -> (d0)>
func.func @multiple_results(%a: tensor<32xf32>)
    -> (tensor<32xf32>, tensor<32xf32>) {
  %e = tensor.empty() : tensor<32xf32>
  %r = linalg.generic {indexing_maps = [#map, #map],
                       iterator_types = ["parallel"]}
       ins(%a : tensor<32xf32>) outs(%e : tensor<32xf32>) {
  ^bb0(%in: f32, %o: f32):
    %m = arith.mulf %in, %in : f32
    linalg.yield %m : f32
  } -> tensor<32xf32>
  return %r, %a : tensor<32xf32>, tensor<32xf32>
}

// CHECK-LABEL: func.func @multiple_results(
// CHECK-SAME:    %{{.*}}: memref<32xf32>) -> (memref<32xf32>, memref<32xf32>)

// LOWER: llvm.func @multiple_results(
