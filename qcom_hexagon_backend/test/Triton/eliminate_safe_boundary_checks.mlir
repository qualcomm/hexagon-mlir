//===- eliminate_safe_boundary_checks.mlir --------------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

// RUN: %PYTHON %triton_pass_opt hexagon-eliminate-safe-boundary-checks,canonicalize,cse %s | FileCheck %s
// RUN: %PYTHON %triton_pass_opt hexagon-eliminate-safe-boundary-checks %s | FileCheck %s --check-prefix=STATS

// Tests for EliminateSafeBoundaryChecks.
//
// The pass removes per-dimension mask terms on tt.load/tt.store where the
// tensor size is a compile-time constant divisible by the block size. It
// matches the mask pattern emitted by Triton's RewriteTensorDescriptorToPointer
// (the lowering of tl.make_tensor_descriptor loads/stores), which runs just
// before this pass:
//
//   coord_d = expand_dims(offset_d + make_range(0, blockSize_d)) // reduced rank
//   valid_d = tt.broadcast((coord_d >= 0) & (coord_d < tensorSize_d))
//   mask    = andi(valid_0, valid_1)
//   tt.load %ptr, %mask
//
// Note the per-dim cmps live at reduced rank (tensor<16x1>, tensor<1x64>)
// *below* a tt.broadcast, so the pass looks through the broadcast to reach
// them. "Safe" when tensorSize_d is a compile-time constant and
// tensorSize_d % blockSize_d == 0.

// Test 1 (both_dims_safe): tensorSize0=64 blockSize0=16, tensorSize1=128
// blockSize1=64 -> all safe, drop the mask on both the load and the store.
// CHECK-LABEL: @both_dims_safe
// CHECK:       %[[LOAD:.*]] = tt.load %{{.*}} : tensor<16x64x!tt.ptr<f16>>
// CHECK-NOT:   tt.load %{{.*}}, %{{.*}} : tensor<16x64
// CHECK:       tt.store %{{.*}}, %[[LOAD]] : tensor<16x64x!tt.ptr<f16>>
// CHECK-NOT:   tt.store %{{.*}}, %{{.*}}, %{{.*}} : tensor<16x64

// Test 2 (one_dim_unsafe): tensorSize0=70 blockSize0=16 (70 % 16 != 0, unsafe),
// tensorSize1=128 blockSize1=64 (safe). The safe dim's leaves sit at reduced
// rank below a broadcast, so rebuilding the mask from the remaining (unsafe)
// reduced-rank leaf would be ill-typed (tensor<16x1xi1> vs tensor<16x64xi1>);
// the pass conservatively leaves the mask untouched.
// CHECK-LABEL: @one_dim_unsafe
// CHECK:       tt.load %{{.*}}, %{{.*}} : tensor<16x64x!tt.ptr<f16>>

// Test 3 (runtime_shape): tensor sizes are runtime args -> not constant ->
// pass must not fire.
// CHECK-LABEL: @runtime_shape_no_elim
// CHECK:       tt.load %{{.*}}, %{{.*}} : tensor<16x64x!tt.ptr<f16>>

// STATS: [EliminateSafeBoundaryChecks] masked_loads=3 fully_eliminated=1

module {

tt.func public @both_dims_safe(%base: !tt.ptr<f16>, %off0: i64, %off1: i64) {
  // Dim 0: blockSize=16, tensorSize=64 (64 % 16 == 0 -> safe). Coord at rank 16x1.
  %range0 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
  %ext0 = arith.extsi %range0 : tensor<16xi32> to tensor<16xi64>
  %soff0 = tt.splat %off0 : i64 -> tensor<16xi64>
  %coord0_1d = arith.addi %soff0, %ext0 : tensor<16xi64>
  %coord0_2d = tt.expand_dims %coord0_1d {axis = 1 : i32} : tensor<16xi64> -> tensor<16x1xi64>
  // Dim 1: blockSize=64, tensorSize=128 (128 % 64 == 0 -> safe). Coord at rank 1x64.
  %range1 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
  %ext1 = arith.extsi %range1 : tensor<64xi32> to tensor<64xi64>
  %soff1 = tt.splat %off1 : i64 -> tensor<64xi64>
  %coord1_1d = arith.addi %soff1, %ext1 : tensor<64xi64>
  %coord1_2d = tt.expand_dims %coord1_1d {axis = 0 : i32} : tensor<64xi64> -> tensor<1x64xi64>
  // Pointer (16x64) built from broadcast coords; not analyzed by the pass.
  %base_splat = tt.splat %base : !tt.ptr<f16> -> tensor<16x64x!tt.ptr<f16>>
  %coord0_bc = tt.broadcast %coord0_2d : tensor<16x1xi64> -> tensor<16x64xi64>
  %stride0 = arith.constant dense<128> : tensor<16x64xi64>
  %scaled0 = arith.muli %coord0_bc, %stride0 : tensor<16x64xi64>
  %p0 = tt.addptr %base_splat, %scaled0 : tensor<16x64x!tt.ptr<f16>>, tensor<16x64xi64>
  %coord1_bc = tt.broadcast %coord1_2d : tensor<1x64xi64> -> tensor<16x64xi64>
  %p1 = tt.addptr %p0, %coord1_bc : tensor<16x64x!tt.ptr<f16>>, tensor<16x64xi64>
  // Per-dim validity computed at reduced rank, then broadcast, then anded.
  %zero0 = arith.constant dense<0> : tensor<16x1xi64>
  %s0 = arith.constant dense<64> : tensor<16x1xi64>
  %ge0 = arith.cmpi sge, %coord0_2d, %zero0 : tensor<16x1xi64>
  %lt0 = arith.cmpi slt, %coord0_2d, %s0 : tensor<16x1xi64>
  %valid0 = arith.andi %ge0, %lt0 : tensor<16x1xi1>
  %valid0_bc = tt.broadcast %valid0 : tensor<16x1xi1> -> tensor<16x64xi1>
  %zero1 = arith.constant dense<0> : tensor<1x64xi64>
  %s1 = arith.constant dense<128> : tensor<1x64xi64>
  %ge1 = arith.cmpi sge, %coord1_2d, %zero1 : tensor<1x64xi64>
  %lt1 = arith.cmpi slt, %coord1_2d, %s1 : tensor<1x64xi64>
  %valid1 = arith.andi %ge1, %lt1 : tensor<1x64xi1>
  %valid1_bc = tt.broadcast %valid1 : tensor<1x64xi1> -> tensor<16x64xi1>
  %mask = arith.andi %valid0_bc, %valid1_bc : tensor<16x64xi1>
  %result = tt.load %p1, %mask : tensor<16x64x!tt.ptr<f16>>
  %out = tt.splat %base : !tt.ptr<f16> -> tensor<16x64x!tt.ptr<f16>>
  tt.store %out, %result, %mask : tensor<16x64x!tt.ptr<f16>>
  tt.return
}

tt.func public @one_dim_unsafe(%base: !tt.ptr<f16>, %off0: i64, %off1: i64) -> tensor<16x64xf16> {
  // Dim 0: blockSize=16, tensorSize=70 (70 % 16 != 0 -> UNSAFE).
  %range0 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
  %ext0 = arith.extsi %range0 : tensor<16xi32> to tensor<16xi64>
  %soff0 = tt.splat %off0 : i64 -> tensor<16xi64>
  %coord0_1d = arith.addi %soff0, %ext0 : tensor<16xi64>
  %coord0_2d = tt.expand_dims %coord0_1d {axis = 1 : i32} : tensor<16xi64> -> tensor<16x1xi64>
  // Dim 1: blockSize=64, tensorSize=128 (128 % 64 == 0 -> safe).
  %range1 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
  %ext1 = arith.extsi %range1 : tensor<64xi32> to tensor<64xi64>
  %soff1 = tt.splat %off1 : i64 -> tensor<64xi64>
  %coord1_1d = arith.addi %soff1, %ext1 : tensor<64xi64>
  %coord1_2d = tt.expand_dims %coord1_1d {axis = 0 : i32} : tensor<64xi64> -> tensor<1x64xi64>
  %base_splat = tt.splat %base : !tt.ptr<f16> -> tensor<16x64x!tt.ptr<f16>>
  %coord0_bc = tt.broadcast %coord0_2d : tensor<16x1xi64> -> tensor<16x64xi64>
  %stride0 = arith.constant dense<128> : tensor<16x64xi64>
  %scaled0 = arith.muli %coord0_bc, %stride0 : tensor<16x64xi64>
  %p0 = tt.addptr %base_splat, %scaled0 : tensor<16x64x!tt.ptr<f16>>, tensor<16x64xi64>
  %coord1_bc = tt.broadcast %coord1_2d : tensor<1x64xi64> -> tensor<16x64xi64>
  %p1 = tt.addptr %p0, %coord1_bc : tensor<16x64x!tt.ptr<f16>>, tensor<16x64xi64>
  %zero0 = arith.constant dense<0> : tensor<16x1xi64>
  %s0 = arith.constant dense<70> : tensor<16x1xi64>
  %ge0 = arith.cmpi sge, %coord0_2d, %zero0 : tensor<16x1xi64>
  %lt0 = arith.cmpi slt, %coord0_2d, %s0 : tensor<16x1xi64>
  %valid0 = arith.andi %ge0, %lt0 : tensor<16x1xi1>
  %valid0_bc = tt.broadcast %valid0 : tensor<16x1xi1> -> tensor<16x64xi1>
  %zero1 = arith.constant dense<0> : tensor<1x64xi64>
  %s1 = arith.constant dense<128> : tensor<1x64xi64>
  %ge1 = arith.cmpi sge, %coord1_2d, %zero1 : tensor<1x64xi64>
  %lt1 = arith.cmpi slt, %coord1_2d, %s1 : tensor<1x64xi64>
  %valid1 = arith.andi %ge1, %lt1 : tensor<1x64xi1>
  %valid1_bc = tt.broadcast %valid1 : tensor<1x64xi1> -> tensor<16x64xi1>
  %mask = arith.andi %valid0_bc, %valid1_bc : tensor<16x64xi1>
  %result = tt.load %p1, %mask : tensor<16x64x!tt.ptr<f16>>
  tt.return %result : tensor<16x64xf16>
}

tt.func public @runtime_shape_no_elim(%base: !tt.ptr<f16>, %off0: i64, %off1: i64,
                                       %S0: i32, %S1: i32) -> tensor<16x64xf16> {
  %range0 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
  %ext0 = arith.extsi %range0 : tensor<16xi32> to tensor<16xi64>
  %soff0 = tt.splat %off0 : i64 -> tensor<16xi64>
  %coord0_1d = arith.addi %soff0, %ext0 : tensor<16xi64>
  %coord0_2d = tt.expand_dims %coord0_1d {axis = 1 : i32} : tensor<16xi64> -> tensor<16x1xi64>
  %range1 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
  %ext1 = arith.extsi %range1 : tensor<64xi32> to tensor<64xi64>
  %soff1 = tt.splat %off1 : i64 -> tensor<64xi64>
  %coord1_1d = arith.addi %soff1, %ext1 : tensor<64xi64>
  %coord1_2d = tt.expand_dims %coord1_1d {axis = 0 : i32} : tensor<64xi64> -> tensor<1x64xi64>
  %base_splat = tt.splat %base : !tt.ptr<f16> -> tensor<16x64x!tt.ptr<f16>>
  %coord0_bc = tt.broadcast %coord0_2d : tensor<16x1xi64> -> tensor<16x64xi64>
  %stride0 = arith.constant dense<128> : tensor<16x64xi64>
  %scaled0 = arith.muli %coord0_bc, %stride0 : tensor<16x64xi64>
  %p0 = tt.addptr %base_splat, %scaled0 : tensor<16x64x!tt.ptr<f16>>, tensor<16x64xi64>
  %coord1_bc = tt.broadcast %coord1_2d : tensor<1x64xi64> -> tensor<16x64xi64>
  %p1 = tt.addptr %p0, %coord1_bc : tensor<16x64x!tt.ptr<f16>>, tensor<16x64xi64>
  // Runtime tensor sizes -> not constant -> NOT safe.
  %zero0 = arith.constant dense<0> : tensor<16x1xi64>
  %s0_i64 = arith.extsi %S0 : i32 to i64
  %s0_splat = tt.splat %s0_i64 : i64 -> tensor<16x1xi64>
  %ge0 = arith.cmpi sge, %coord0_2d, %zero0 : tensor<16x1xi64>
  %lt0 = arith.cmpi slt, %coord0_2d, %s0_splat : tensor<16x1xi64>
  %valid0 = arith.andi %ge0, %lt0 : tensor<16x1xi1>
  %valid0_bc = tt.broadcast %valid0 : tensor<16x1xi1> -> tensor<16x64xi1>
  %zero1 = arith.constant dense<0> : tensor<1x64xi64>
  %s1_i64 = arith.extsi %S1 : i32 to i64
  %s1_splat = tt.splat %s1_i64 : i64 -> tensor<1x64xi64>
  %ge1 = arith.cmpi sge, %coord1_2d, %zero1 : tensor<1x64xi64>
  %lt1 = arith.cmpi slt, %coord1_2d, %s1_splat : tensor<1x64xi64>
  %valid1 = arith.andi %ge1, %lt1 : tensor<1x64xi1>
  %valid1_bc = tt.broadcast %valid1 : tensor<1x64xi1> -> tensor<16x64xi1>
  %mask = arith.andi %valid0_bc, %valid1_bc : tensor<16x64xi1>
  %result = tt.load %p1, %mask : tensor<16x64x!tt.ptr<f16>>
  tt.return %result : tensor<16x64xf16>
}

} // module
