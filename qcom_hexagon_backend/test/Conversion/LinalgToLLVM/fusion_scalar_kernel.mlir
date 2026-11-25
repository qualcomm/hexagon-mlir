//===- fusion_scalar_kernel.mlir ------------------------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(func.func(hexagon-fusion))' | FileCheck %s
//
// Test that HexagonFusion correctly handles a scalar (0-D) kernel where
// LinalgFoldUnitExtentDims has collapsed all loop dimensions, leaving a
// linalg.generic with 0-D tensor DPS operands (from linalg.fill) and
// empty iterator_types. The fusion pass constant-folds the fills into the
// body and must then inline the resulting no-operand 0-D linalg.generic
// directly into the parent block rather than leaving a degenerate op that
// no downstream lowering pass can handle.
//
// This pattern arises from triton-shared scalar kernels such as:
//   tl.store(out_ptr0, tl.load(in_ptr0).to(tl.float32))
// where xnumel=1 and LinalgFoldUnitExtentDims collapses the single-element
// parallel loop away entirely.

#map = affine_map<() -> ()>

module {
  func.func @triton_poi_fused__to_copy_unsqueeze_1(
      %arg0: memref<*xi32>,
      %arg1: memref<*xf32> {tt.divisibility = 16 : i32},
      %arg2: i32, %arg3: i32, %arg4: i32,
      %arg5: i32, %arg6: i32, %arg7: i32) {
    %c0_i32 = arith.constant 0 : i32
    %reinterpret_cast = memref.reinterpret_cast %arg0
        to offset: [0], sizes: [1], strides: [1]
        : memref<*xi32> to memref<1xi32, strided<[1]>>
    %0 = affine.load %reinterpret_cast[0] : memref<1xi32, strided<[1]>>
    %1 = arith.extsi %0 : i32 to i64
    %2 = arith.sitofp %1 : i64 to f32
    %cast = memref.cast %arg1 : memref<*xf32> to memref<?xf32>
    %3 = tensor.empty() : tensor<i32>
    %4 = linalg.fill ins(%c0_i32 : i32) outs(%3 : tensor<i32>) -> tensor<i32>
    %5 = tensor.empty() : tensor<f32>
    %6 = linalg.fill ins(%2 : f32) outs(%5 : tensor<f32>) -> tensor<f32>
    linalg.generic {indexing_maps = [#map, #map], iterator_types = []}
        ins(%4, %6 : tensor<i32>, tensor<f32>) {
    ^bb0(%in: i32, %in_0: f32):
      %7 = arith.index_cast %in : i32 to index
      memref.store %in_0, %cast[%7] : memref<?xf32>
      linalg.yield
    }
    return
  }

  // Test that InlineZeroDimGenericPattern correctly remaps intra-body operand
  // references when the 0-D generic body is a dependency chain (more than one
  // op before the terminator).  Without an IRMapping, the second body op
  // (%sum) would still reference the original region's %neg after eraseOp
  // deleted it, leaving a dangling operand.  With the IRMapping the cloned
  // %neg is threaded through so %sum binds to it instead.
  func.func @zero_dim_generic_body_chain(%arg0: f32, %arg1: f32,
                                          %arg2: memref<f32>) {
    %empty = tensor.empty() : tensor<f32>
    %fill = linalg.fill ins(%arg0 : f32) outs(%empty : tensor<f32>) -> tensor<f32>
    linalg.generic {indexing_maps = [#map], iterator_types = []}
        ins(%fill : tensor<f32>) {
    ^bb0(%in: f32):
      %neg = arith.negf %in : f32
      %sum = arith.addf %neg, %arg1 : f32
      memref.store %sum, %arg2[] : memref<f32>
      linalg.yield
    }
    return
  }
}

// After fusion, the 0-D linalg.generic with no operands must be inlined.
// The memref.store should appear directly in the function body with the
// constant index 0 (from index_cast of the constant-folded i32 zero fill)
// and the f32 value from sitofp.

// CHECK-LABEL: func.func @triton_poi_fused__to_copy_unsqueeze_1
// CHECK:         %[[IDX:.*]] = arith.constant 0 : index
// CHECK:         %[[VAL:.*]] = arith.sitofp {{.*}} : i64 to f32
// CHECK:         memref.store %[[VAL]], {{.*}}[%[[IDX]]] : memref<?xf32>
// CHECK-NOT:     linalg.generic
// CHECK-NOT:     linalg.fill
// CHECK-NOT:     tensor.empty

// After constant-folding %fill into the body, %in becomes %arg0; after
// EraseUnusedOperandsAndResults the generic has no DPS operands. The cleanup
// round then applies InlineZeroDimGenericPattern, which must produce:
//   %neg_cloned = arith.negf %arg0
//   %sum_cloned = arith.addf %neg_cloned, %arg1   <- uses cloned result
//   memref.store %sum_cloned, %arg2[]

// CHECK-LABEL: func.func @zero_dim_generic_body_chain
// CHECK:         %[[NEG:.*]] = arith.negf %arg0 : f32
// CHECK:         %[[SUM:.*]] = arith.addf %[[NEG]], %arg1 : f32
// CHECK:         memref.store %[[SUM]], %arg2[] : memref<f32>
// CHECK-NOT:     linalg.generic
// CHECK-NOT:     linalg.fill
// CHECK-NOT:     tensor.empty
