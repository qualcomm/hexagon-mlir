//===- vector-row-reduce.mlir - 2-D row reduce to butterfly -----------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
// RUN: linalg-hexagon-opt %s -split-input-file -vector-row-reduce | FileCheck %s
//===----------------------------------------------------------------------===//

// FA shape (post split-reduction): [256, 32] f32 row max, one HVX vector per
// row. Butterfly = 5 rotate+max steps (64, 32, 16, 8, 4 bytes); after them
// every lane holds the row max, so the scalar extract is exact.
//
// Semantics note: the emitted `arith.maxnumf` lowers to the HVX IEEE max
// (`vmax`), which returns NaN when either operand is NaN -- unlike strict
// maxnumf, which returns the non-NaN operand. Bit-identical for non-NaN
// inputs, which is the contract the kernels rely on (they mask to -inf
// upstream of the reduce).
//
// CHECK-LABEL: func.func @rowmax_f32_one_vector
func.func @rowmax_f32_one_vector(%src: memref<256x32xf32>, %dst: memref<256xf32>) {
  // CHECK: scf.for {{%.*}} = %c0 to %c256 step %c1 {
  // CHECK: %[[row:.*]] = memref.subview {{%.*}}[{{%.*}}, 0] [1, 32] [1, 1] : memref<256x32xf32> to memref<32xf32, strided<[1], offset: ?>>
  // CHECK: %[[v0:.*]] = vector.transfer_read %[[row]][%c0], {{%.*}} {in_bounds = [true]} : memref<32xf32, strided<[1], offset: ?>>, vector<32xf32>
  // Rotate right by 64 B: lane k now holds old lane (k+16) mod 32.
  // CHECK: %[[r64:.*]] = hvx.vror %[[v0]], 64 : vector<32xf32>
  // CHECK: %[[m64:.*]] = arith.maxnumf %[[r64]], %[[v0]] : vector<32xf32>
  // CHECK: %[[r32:.*]] = hvx.vror %[[m64]], 32 : vector<32xf32>
  // CHECK: %[[m32:.*]] = arith.maxnumf %[[r32]], %[[m64]] : vector<32xf32>
  // CHECK: %[[r16:.*]] = hvx.vror %[[m32]], 16 : vector<32xf32>
  // CHECK: %[[m16:.*]] = arith.maxnumf %[[r16]], %[[m32]] : vector<32xf32>
  // CHECK: %[[r8:.*]] = hvx.vror %[[m16]], 8 : vector<32xf32>
  // CHECK: %[[m8:.*]] = arith.maxnumf %[[r8]], %[[m16]] : vector<32xf32>
  // CHECK: %[[r4:.*]] = hvx.vror %[[m8]], 4 : vector<32xf32>
  // CHECK: %[[m4:.*]] = arith.maxnumf %[[r4]], %[[m8]] : vector<32xf32>
  // The reduce's own init (the outs element) is folded in, then lane 0 is
  // stored: the scalar contract of the reduce is preserved.
  // CHECK: %[[init:.*]] = memref.load {{%.*}}[{{%.*}}] : memref<256xf32>
  // CHECK: %[[initv:.*]] = vector.broadcast %[[init]] : f32 to vector<32xf32>
  // CHECK: %[[acc:.*]] = arith.maxnumf %[[m4]], %[[initv]] : vector<32xf32>
  // CHECK: %[[sc:.*]] = vector.extract %[[acc]][0] : f32 from vector<32xf32>
  // CHECK: memref.store %[[sc]], {{%.*}}[{{%.*}}] : memref<256xf32>
  // CHECK-NOT: linalg.reduce
  linalg.reduce ins(%src : memref<256x32xf32>) outs(%dst : memref<256xf32>) dimensions = [1]
    (%in: f32, %init: f32) {
    %0 = arith.maxnumf %in, %init : f32
    linalg.yield %0 : f32
  }
  return
}

// -----

// Two HVX vectors per row (a [256, 64] row, e.g. the unsplit FA reduce): the
// chunks are folded elementwise first, then the butterfly runs on one vector.
// CHECK-LABEL: func.func @rowmax_f32_two_vectors
func.func @rowmax_f32_two_vectors(%src: memref<256x64xf32>, %dst: memref<256xf32>) {
  // CHECK: %[[row:.*]] = memref.subview {{%.*}}[{{%.*}}, 0] [1, 64] [1, 1] : memref<256x64xf32> to memref<64xf32, strided<[1], offset: ?>>
  // CHECK: %[[c0:.*]] = vector.transfer_read %[[row]][%c0], {{%.*}} {in_bounds = [true]} : memref<64xf32, strided<[1], offset: ?>>, vector<32xf32>
  // CHECK: %[[c1:.*]] = vector.transfer_read %[[row]][%c32], {{%.*}} {in_bounds = [true]} : memref<64xf32, strided<[1], offset: ?>>, vector<32xf32>
  // CHECK: %[[m:.*]] = arith.maxnumf %[[c0]], %[[c1]] : vector<32xf32>
  // CHECK: %[[r64:.*]] = hvx.vror %[[m]], 64 : vector<32xf32>
  // CHECK: %[[m64:.*]] = arith.maxnumf %[[r64]], %[[m]] : vector<32xf32>
  // CHECK: hvx.vror %[[m64]], 32 : vector<32xf32>
  linalg.reduce ins(%src : memref<256x64xf32>) outs(%dst : memref<256xf32>) dimensions = [1]
    (%in: f32, %init: f32) {
    %0 = arith.maxnumf %in, %init : f32
    linalg.yield %0 : f32
  }
  return
}

// -----

// f16: 64 lanes of 2 bytes, so the butterfly takes 6 steps down to 2 bytes.
// CHECK-LABEL: func.func @rowmax_f16
func.func @rowmax_f16(%src: memref<256x64xf16>, %dst: memref<256xf16>) {
  // CHECK: vector.transfer_read {{%.*}} : memref<64xf16, strided<[1], offset: ?>>, vector<64xf16>
  // CHECK: %[[r64:.*]] = hvx.vror {{%.*}}, 64 : vector<64xf16>
  // CHECK: arith.maxnumf %[[r64]]
  // CHECK: hvx.vror {{%.*}}, 32 : vector<64xf16>
  // CHECK: hvx.vror {{%.*}}, 16 : vector<64xf16>
  // CHECK: hvx.vror {{%.*}}, 8 : vector<64xf16>
  // CHECK: hvx.vror {{%.*}}, 4 : vector<64xf16>
  // CHECK: hvx.vror {{%.*}}, 2 : vector<64xf16>
  linalg.reduce ins(%src : memref<256x64xf16>) outs(%dst : memref<256xf16>) dimensions = [1]
    (%in: f16, %init: f16) {
    %0 = arith.maxnumf %in, %init : f16
    linalg.yield %0 : f16
  }
  return
}

// -----

// Row sum: same skeleton with addf, keeping the body's fastmath flags.
// CHECK-LABEL: func.func @rowsum_f32
func.func @rowsum_f32(%src: memref<256x32xf32>, %dst: memref<256xf32>) {
  // CHECK: %[[r64:.*]] = hvx.vror {{%.*}}, 64 : vector<32xf32>
  // CHECK: arith.addf %[[r64]], {{%.*}} fastmath<fast> : vector<32xf32>
  linalg.reduce ins(%src : memref<256x32xf32>) outs(%dst : memref<256xf32>) dimensions = [1]
    (%in: f32, %init: f32) {
    %0 = arith.addf %in, %init fastmath<fast> : f32
    linalg.yield %0 : f32
  }
  return
}

// -----

// Negative: dynamic row count is not matchable -- the scalar path keeps it.
// CHECK-LABEL: func.func @dynamic_rows
func.func @dynamic_rows(%src: memref<?x32xf32>, %dst: memref<?xf32>) {
  // CHECK: linalg.reduce
  linalg.reduce ins(%src : memref<?x32xf32>) outs(%dst : memref<?xf32>) dimensions = [1]
    (%in: f32, %init: f32) {
    %0 = arith.maxnumf %in, %init : f32
    linalg.yield %0 : f32
  }
  return
}

// -----

// Negative: a row that is not a whole number of HVX vectors (32 f16 = 64 B).
// CHECK-LABEL: func.func @partial_vector_row
func.func @partial_vector_row(%src: memref<256x32xf16>, %dst: memref<256xf16>) {
  // CHECK: linalg.reduce
  linalg.reduce ins(%src : memref<256x32xf16>) outs(%dst : memref<256xf16>) dimensions = [1]
    (%in: f16, %init: f16) {
    %0 = arith.maxnumf %in, %init : f16
    linalg.yield %0 : f16
  }
  return
}

// -----

// Negative: reduction along dim 0 (column reduce) -- rows of one scalar, not a
// vector walk.
// CHECK-LABEL: func.func @reduce_dim0
func.func @reduce_dim0(%src: memref<32x256xf32>, %dst: memref<256xf32>) {
  // CHECK: linalg.reduce
  linalg.reduce ins(%src : memref<32x256xf32>) outs(%dst : memref<256xf32>) dimensions = [0]
    (%in: f32, %init: f32) {
    %0 = arith.maxnumf %in, %init : f32
    linalg.yield %0 : f32
  }
  return
}

// -----

// Negative: a body that is not a single fold of the two block args -- here an
// extra square -- keeps the scalar path.
// CHECK-LABEL: func.func @non_fold_body
func.func @non_fold_body(%src: memref<256x32xf32>, %dst: memref<256xf32>) {
  // CHECK: linalg.reduce
  linalg.reduce ins(%src : memref<256x32xf32>) outs(%dst : memref<256xf32>) dimensions = [1]
    (%in: f32, %init: f32) {
    %0 = arith.mulf %in, %in : f32
    %1 = arith.maxnumf %0, %init : f32
    linalg.yield %1 : f32
  }
  return
}
