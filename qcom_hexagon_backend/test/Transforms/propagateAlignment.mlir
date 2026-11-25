//===- propagateAlignment.mlir --------------------------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

// RUN: linalg-hexagon-opt %s -split-input-file -pass-pipeline='builtin.module(llvm-alignment-propagation{set-memref-alloc-alignment=false})' | FileCheck %s

// -----
// Test 1: Aligned base with static zero offset - should maintain alignment
func.func @aligned_base_zero_offset(%arg0: memref<1024xf32, 1> {tt.divisibility = 128 : i32}) {
  %base_buffer, %offset, %sizes, %strides = memref.extract_strided_metadata %arg0 : memref<1024xf32, 1> -> memref<f32, 1>, index, index, index
  %reinterpret_cast = memref.reinterpret_cast %base_buffer to offset: [0], sizes: [32], strides: [1] : memref<f32, 1> to memref<32xf32, strided<[1]>, 1>
  %cast = builtin.unrealized_conversion_cast %reinterpret_cast : memref<32xf32, strided<[1]>, 1> to !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
  %ptr = llvm.extractvalue %cast[1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
  %c0 = arith.constant 0 : index
  %c0_i64 = builtin.unrealized_conversion_cast %c0 : index to i64
  %gep = llvm.getelementptr %ptr[%c0_i64] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
  %val = llvm.load %gep : !llvm.ptr<1> -> f32
  return
}

// CHECK-LABEL: @aligned_base_zero_offset
// CHECK: memref.reinterpret_cast {{.*}} {alignment = 128 : i64}
// CHECK: llvm.load {{.*}} {alignment = 128 : i64}

// -----
// Test 2: Aligned base with static aligned offset (32 elements = 128 bytes for f32)
func.func @aligned_base_aligned_offset(%arg0: memref<1024xf32, 1> {tt.divisibility = 128 : i32}) {
  %base_buffer, %offset, %sizes, %strides = memref.extract_strided_metadata %arg0 : memref<1024xf32, 1> -> memref<f32, 1>, index, index, index
  %reinterpret_cast = memref.reinterpret_cast %base_buffer to offset: [32], sizes: [32], strides: [1] : memref<f32, 1> to memref<32xf32, strided<[1], offset: 32>, 1>
  %cast = builtin.unrealized_conversion_cast %reinterpret_cast : memref<32xf32, strided<[1], offset: 32>, 1> to !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
  %ptr = llvm.extractvalue %cast[1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
  %c0 = arith.constant 0 : index
  %c0_i64 = builtin.unrealized_conversion_cast %c0 : index to i64
  %gep = llvm.getelementptr %ptr[%c0_i64] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
  %val = llvm.load %gep : !llvm.ptr<1> -> f32
  return
}

// CHECK-LABEL: @aligned_base_aligned_offset
// CHECK: memref.reinterpret_cast {{.*}} {alignment = 128 : i64}
// CHECK: llvm.load {{.*}} {alignment = 128 : i64}

// -----
// Test 3: Aligned base with static unaligned offset (16 elements = 64 bytes for f32)
// GCD(128, 64) = 64
func.func @aligned_base_unaligned_offset(%arg0: memref<1024xf32, 1> {tt.divisibility = 128 : i32}) {
  %base_buffer, %offset, %sizes, %strides = memref.extract_strided_metadata %arg0 : memref<1024xf32, 1> -> memref<f32, 1>, index, index, index
  %reinterpret_cast = memref.reinterpret_cast %base_buffer to offset: [16], sizes: [32], strides: [1] : memref<f32, 1> to memref<32xf32, strided<[1], offset: 16>, 1>
  %cast = builtin.unrealized_conversion_cast %reinterpret_cast : memref<32xf32, strided<[1], offset: 16>, 1> to !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
  %ptr = llvm.extractvalue %cast[1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
  %c0 = arith.constant 0 : index
  %c0_i64 = builtin.unrealized_conversion_cast %c0 : index to i64
  %gep = llvm.getelementptr %ptr[%c0_i64] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
  %val = llvm.load %gep : !llvm.ptr<1> -> f32
  return
}

// CHECK-LABEL: @aligned_base_unaligned_offset
// CHECK: memref.reinterpret_cast {{.*}} {alignment = 64 : i64}
// CHECK: llvm.load {{.*}} {alignment = 64 : i64}

// -----
// Test 4: Aligned base with known block arg (init=0, step=32 elements = 128 bytes)
func.func @aligned_base_block_arg_aligned(%arg0: memref<1024xf32, 1> {tt.divisibility = 128 : i32}) {
  %c0 = arith.constant 0 : index
  %c32 = arith.constant 32 : index
  %c1024 = arith.constant 1024 : index
  cf.br ^bb1(%c0 : index)
^bb1(%idx: index):
  %cond = arith.cmpi slt, %idx, %c1024 : index
  cf.cond_br %cond, ^bb2, ^bb3
^bb2:
  %base_buffer, %offset, %sizes, %strides = memref.extract_strided_metadata %arg0 : memref<1024xf32, 1> -> memref<f32, 1>, index, index, index
  %reinterpret_cast = memref.reinterpret_cast %base_buffer to offset: [%idx], sizes: [32], strides: [1] : memref<f32, 1> to memref<32xf32, strided<[1], offset: ?>, 1>
  %cast = builtin.unrealized_conversion_cast %reinterpret_cast : memref<32xf32, strided<[1], offset: ?>, 1> to !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
  %ptr = llvm.extractvalue %cast[1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
  %c0_2 = arith.constant 0 : index
  %c0_i64 = builtin.unrealized_conversion_cast %c0_2 : index to i64
  %gep = llvm.getelementptr %ptr[%c0_i64] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
  %val = llvm.load %gep : !llvm.ptr<1> -> f32
  %next_idx = arith.addi %idx, %c32 : index
  cf.br ^bb1(%next_idx : index)
^bb3:
  return
}

// CHECK-LABEL: @aligned_base_block_arg_aligned
// CHECK: memref.reinterpret_cast {{.*}} {alignment = 128 : i64}
// CHECK: llvm.load {{.*}} {alignment = 128 : i64}

// -----
// Test 5: Aligned base with known block arg (init=0, step=16 elements = 64 bytes)
// GCD(128, 64) = 64
func.func @aligned_base_block_arg_unaligned(%arg0: memref<1024xf32, 1> {tt.divisibility = 128 : i32}) {
  %c0 = arith.constant 0 : index
  %c16 = arith.constant 16 : index
  %c1024 = arith.constant 1024 : index
  cf.br ^bb1(%c0 : index)
^bb1(%idx: index):
  %cond = arith.cmpi slt, %idx, %c1024 : index
  cf.cond_br %cond, ^bb2, ^bb3
^bb2:
  %base_buffer, %offset, %sizes, %strides = memref.extract_strided_metadata %arg0 : memref<1024xf32, 1> -> memref<f32, 1>, index, index, index
  %reinterpret_cast = memref.reinterpret_cast %base_buffer to offset: [%idx], sizes: [16], strides: [1] : memref<f32, 1> to memref<16xf32, strided<[1], offset: ?>, 1>
  %cast = builtin.unrealized_conversion_cast %reinterpret_cast : memref<16xf32, strided<[1], offset: ?>, 1> to !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
  %ptr = llvm.extractvalue %cast[1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
  %c0_2 = arith.constant 0 : index
  %c0_i64 = builtin.unrealized_conversion_cast %c0_2 : index to i64
  %gep = llvm.getelementptr %ptr[%c0_i64] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
  %val = llvm.load %gep : !llvm.ptr<1> -> f32
  %next_idx = arith.addi %idx, %c16 : index
  cf.br ^bb1(%next_idx : index)
^bb3:
  return
}

// CHECK-LABEL: @aligned_base_block_arg_unaligned
// CHECK: memref.reinterpret_cast {{.*}} {alignment = 64 : i64}
// CHECK: llvm.load {{.*}} {alignment = 64 : i64}

// -----
// Test 6: Aligned base with arith.addi - both operands aligned
// arg1 has divisibility=128 (32 elements), loop var has step=32 (128 bytes)
// GCD(128, 128) = 128
func.func @aligned_base_addi_both_aligned(%arg0: memref<1024xf32, 1> {tt.divisibility = 128 : i32}, %arg1: index {tt.divisibility = 32 : i32}) {
  %c0 = arith.constant 0 : index
  %c32 = arith.constant 32 : index
  %c1024 = arith.constant 1024 : index
  cf.br ^bb1(%c0 : index)
^bb1(%idx: index):
  %cond = arith.cmpi slt, %idx, %c1024 : index
  cf.cond_br %cond, ^bb2, ^bb3
^bb2:
  %base_buffer, %offset, %sizes, %strides = memref.extract_strided_metadata %arg0 : memref<1024xf32, 1> -> memref<f32, 1>, index, index, index
  %combined_offset = arith.addi %arg1, %idx : index
  %reinterpret_cast = memref.reinterpret_cast %base_buffer to offset: [%combined_offset], sizes: [32], strides: [1] : memref<f32, 1> to memref<32xf32, strided<[1], offset: ?>, 1>
  %cast = builtin.unrealized_conversion_cast %reinterpret_cast : memref<32xf32, strided<[1], offset: ?>, 1> to !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
  %ptr = llvm.extractvalue %cast[1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
  %c0_2 = arith.constant 0 : index
  %c0_i64 = builtin.unrealized_conversion_cast %c0_2 : index to i64
  %gep = llvm.getelementptr %ptr[%c0_i64] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
  %val = llvm.load %gep : !llvm.ptr<1> -> f32
  %next_idx = arith.addi %idx, %c32 : index
  cf.br ^bb1(%next_idx : index)
^bb3:
  return
}

// CHECK-LABEL: @aligned_base_addi_both_aligned
// CHECK: memref.reinterpret_cast {{.*}} {alignment = 128 : i64}
// CHECK: llvm.load {{.*}} {alignment = 128 : i64}

// -----
// Test 7: Aligned base with arith.addi - arg aligned (32 elem), loop unaligned (16 elem)
// GCD(128, 64) = 64 bytes
func.func @aligned_base_addi_mixed(%arg0: memref<1024xf32, 1> {tt.divisibility = 128 : i32}, %arg1: index {tt.divisibility = 32 : i32}) {
  %c0 = arith.constant 0 : index
  %c16 = arith.constant 16 : index
  %c1024 = arith.constant 1024 : index
  cf.br ^bb1(%c0 : index)
^bb1(%idx: index):
  %cond = arith.cmpi slt, %idx, %c1024 : index
  cf.cond_br %cond, ^bb2, ^bb3
^bb2:
  %base_buffer, %offset, %sizes, %strides = memref.extract_strided_metadata %arg0 : memref<1024xf32, 1> -> memref<f32, 1>, index, index, index
  %combined_offset = arith.addi %arg1, %idx : index
  %reinterpret_cast = memref.reinterpret_cast %base_buffer to offset: [%combined_offset], sizes: [16], strides: [1] : memref<f32, 1> to memref<16xf32, strided<[1], offset: ?>, 1>
  %cast = builtin.unrealized_conversion_cast %reinterpret_cast : memref<16xf32, strided<[1], offset: ?>, 1> to !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
  %ptr = llvm.extractvalue %cast[1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
  %c0_2 = arith.constant 0 : index
  %c0_i64 = builtin.unrealized_conversion_cast %c0_2 : index to i64
  %gep = llvm.getelementptr %ptr[%c0_i64] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
  %val = llvm.load %gep : !llvm.ptr<1> -> f32
  %next_idx = arith.addi %idx, %c16 : index
  cf.br ^bb1(%next_idx : index)
^bb3:
  return
}

// CHECK-LABEL: @aligned_base_addi_mixed
// CHECK: memref.reinterpret_cast {{.*}} {alignment = 64 : i64}
// CHECK: llvm.load {{.*}} {alignment = 64 : i64}

// -----
// Test 8: Aligned base with arith.addi - only loop var has alignment
// arg1 has no divisibility, loop var has step=32 (divisibility=32)
// Result has unknown alignment (unknown + aligned = unknown)
// This test verifies that we don't incorrectly propagate alignment
func.func @aligned_base_addi_only_loop(%arg0: memref<1024xf32, 1> {tt.divisibility = 128 : i32}, %arg1: index) {
  %c0 = arith.constant 0 : index
  %c32 = arith.constant 32 : index
  %c1024 = arith.constant 1024 : index
  cf.br ^bb1(%c0 : index)
^bb1(%idx: index):
  %cond = arith.cmpi slt, %idx, %c1024 : index
  cf.cond_br %cond, ^bb2, ^bb3
^bb2:
  %base_buffer, %offset, %sizes, %strides = memref.extract_strided_metadata %arg0 : memref<1024xf32, 1> -> memref<f32, 1>, index, index, index
  %combined_offset = arith.addi %arg1, %idx : index
  %reinterpret_cast = memref.reinterpret_cast %base_buffer to offset: [%combined_offset], sizes: [32], strides: [1] : memref<f32, 1> to memref<32xf32, strided<[1], offset: ?>, 1>
  %cast = builtin.unrealized_conversion_cast %reinterpret_cast : memref<32xf32, strided<[1], offset: ?>, 1> to !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
  %ptr = llvm.extractvalue %cast[1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
  %c0_2 = arith.constant 0 : index
  %c0_i64 = builtin.unrealized_conversion_cast %c0_2 : index to i64
  %gep = llvm.getelementptr %ptr[%c0_i64] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
  %val = llvm.load %gep : !llvm.ptr<1> -> f32
  %next_idx = arith.addi %idx, %c32 : index
  cf.br ^bb1(%next_idx : index)
^bb3:
  return
}

// CHECK-LABEL: @aligned_base_addi_only_loop
// CHECK-NOT: memref.reinterpret_cast {{.*}} {alignment
// CHECK: llvm.load {{.*}} {alignment = 4 : i64}

// -----
// Test 9: Unaligned base (no tt.divisibility) - should not propagate
// Note: The pass only processes functions with tt.divisibility attributes,
// so this function is not modified at all.
func.func @unaligned_base(%arg0: memref<1024xf32, 1>) {
  %base_buffer, %offset, %sizes, %strides = memref.extract_strided_metadata %arg0 : memref<1024xf32, 1> -> memref<f32, 1>, index, index, index
  %reinterpret_cast = memref.reinterpret_cast %base_buffer to offset: [0], sizes: [32], strides: [1] : memref<f32, 1> to memref<32xf32, strided<[1]>, 1>
  %cast = builtin.unrealized_conversion_cast %reinterpret_cast : memref<32xf32, strided<[1]>, 1> to !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
  %ptr = llvm.extractvalue %cast[1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
  %c0 = arith.constant 0 : index
  %c0_i64 = builtin.unrealized_conversion_cast %c0 : index to i64
  %gep = llvm.getelementptr %ptr[%c0_i64] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
  %val = llvm.load %gep : !llvm.ptr<1> -> f32
  return
}

// CHECK-LABEL: @unaligned_base
// CHECK-NOT: memref.reinterpret_cast {{.*}} {alignment
// CHECK-NOT: llvm.load {{.*}} {alignment

// -----
// Test 10: Small alignment with large offset - GCD(16, 64) = 16
func.func @small_alignment_large_offset(%arg0: memref<1024xf32, 1> {tt.divisibility = 16 : i32}) {
  %base_buffer, %offset, %sizes, %strides = memref.extract_strided_metadata %arg0 : memref<1024xf32, 1> -> memref<f32, 1>, index, index, index
  %reinterpret_cast = memref.reinterpret_cast %base_buffer to offset: [16], sizes: [32], strides: [1] : memref<f32, 1> to memref<32xf32, strided<[1], offset: 16>, 1>
  %cast = builtin.unrealized_conversion_cast %reinterpret_cast : memref<32xf32, strided<[1], offset: 16>, 1> to !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
  %ptr = llvm.extractvalue %cast[1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
  %c0 = arith.constant 0 : index
  %c0_i64 = builtin.unrealized_conversion_cast %c0 : index to i64
  %gep = llvm.getelementptr %ptr[%c0_i64] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
  %val = llvm.load %gep : !llvm.ptr<1> -> f32
  return
}

// CHECK-LABEL: @small_alignment_large_offset
// CHECK: memref.reinterpret_cast {{.*}} {alignment = 16 : i64}
// CHECK: llvm.load {{.*}} {alignment = 16 : i64}

// -----
// Test A: hexagonmem.alloc with implicit 128-byte alignment
// Even though function arg has tt.divisibility=16, the alloc's implicit 128-byte
// alignment takes precedence for operations on the allocated buffer
func.func @hexagonmem_alloc_implicit_alignment(%arg0: index {tt.divisibility = 16 : i32}) {
  %alloc = hexagonmem.alloc() : memref<1024xf32, 1>
  %base_buffer, %offset, %sizes, %strides = memref.extract_strided_metadata %alloc : memref<1024xf32, 1> -> memref<f32, 1>, index, index, index
  %reinterpret_cast = memref.reinterpret_cast %base_buffer to offset: [%arg0], sizes: [32], strides: [1] : memref<f32, 1> to memref<32xf32, strided<[1], offset: ?>, 1>
  %cast = builtin.unrealized_conversion_cast %reinterpret_cast : memref<32xf32, strided<[1], offset: ?>, 1> to !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
  %ptr = llvm.extractvalue %cast[1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
  %c0 = arith.constant 0 : index
  %c0_i64 = builtin.unrealized_conversion_cast %c0 : index to i64
  %gep = llvm.getelementptr %ptr[%c0_i64] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
  %val = llvm.load %gep : !llvm.ptr<1> -> f32
  llvm.store %val, %gep : f32, !llvm.ptr<1>
  return
}

// CHECK-LABEL: @hexagonmem_alloc_implicit_alignment
// CHECK: memref.reinterpret_cast {{.*}} {alignment = 64 : i64}
// CHECK: llvm.load {{.*}} {alignment = 64 : i64}
// CHECK: llvm.store {{.*}} {alignment = 64 : i64}

// -----
// Test B: memref.alloc with explicit alignment=64
// The explicit alignment attribute on memref.alloc is used for propagation
func.func @memref_alloc_explicit_alignment(%arg0: index {tt.divisibility = 32 : i32}) {
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<1024xf32, 1>
  %base_buffer, %offset, %sizes, %strides = memref.extract_strided_metadata %alloc : memref<1024xf32, 1> -> memref<f32, 1>, index, index, index
  %reinterpret_cast = memref.reinterpret_cast %base_buffer to offset: [%arg0], sizes: [32], strides: [1] : memref<f32, 1> to memref<32xf32, strided<[1], offset: ?>, 1>
  %cast = builtin.unrealized_conversion_cast %reinterpret_cast : memref<32xf32, strided<[1], offset: ?>, 1> to !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
  %ptr = llvm.extractvalue %cast[1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
  %c0 = arith.constant 0 : index
  %c0_i64 = builtin.unrealized_conversion_cast %c0 : index to i64
  %gep = llvm.getelementptr %ptr[%c0_i64] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
  %val = llvm.load %gep : !llvm.ptr<1> -> f32
  llvm.store %val, %gep : f32, !llvm.ptr<1>
  return
}

// CHECK-LABEL: @memref_alloc_explicit_alignment
// CHECK: memref.reinterpret_cast {{.*}} {alignment = 64 : i64}
// CHECK: llvm.load {{.*}} {alignment = 64 : i64}
// CHECK: llvm.store {{.*}} {alignment = 64 : i64}

// -----
// Test C: memref.alloc without explicit alignment
// Without explicit alignment attribute, memref.alloc doesn't provide alignment info
// Only natural alignment (4 bytes for f32) is used
func.func @memref_alloc_no_alignment(%arg0: index {tt.divisibility = 32 : i32}) {
  %alloc = memref.alloc() : memref<1024xf32, 1>
  %base_buffer, %offset, %sizes, %strides = memref.extract_strided_metadata %alloc : memref<1024xf32, 1> -> memref<f32, 1>, index, index, index
  %reinterpret_cast = memref.reinterpret_cast %base_buffer to offset: [%arg0], sizes: [32], strides: [1] : memref<f32, 1> to memref<32xf32, strided<[1], offset: ?>, 1>
  %cast = builtin.unrealized_conversion_cast %reinterpret_cast : memref<32xf32, strided<[1], offset: ?>, 1> to !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
  %ptr = llvm.extractvalue %cast[1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
  %c0 = arith.constant 0 : index
  %c0_i64 = builtin.unrealized_conversion_cast %c0 : index to i64
  %gep = llvm.getelementptr %ptr[%c0_i64] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
  %val = llvm.load %gep : !llvm.ptr<1> -> f32
  llvm.store %val, %gep : f32, !llvm.ptr<1>
  return
}

// CHECK-LABEL: @memref_alloc_no_alignment
// CHECK-NOT: memref.reinterpret_cast {{.*}} {alignment
// CHECK: llvm.load {{.*}} {alignment = 4 : i64}
// CHECK: llvm.store {{.*}} {alignment = 4 : i64}

// -----
// Test D: Index addition with both operands aligned
// arg0 has divisibility=128 (32 elements), arg1 has divisibility=64 (16 elements)
// GCD(128, 64) = 64 bytes
func.func @index_add_both_aligned(%arg0: index {tt.divisibility = 32 : i32}, %arg1: index {tt.divisibility = 16 : i32}, %arg2: memref<1024xf32, 1> {tt.divisibility = 128 : i32}) {
  %sum = arith.addi %arg0, %arg1 : index
  %base_buffer, %offset, %sizes, %strides = memref.extract_strided_metadata %arg2 : memref<1024xf32, 1> -> memref<f32, 1>, index, index, index
  %reinterpret_cast = memref.reinterpret_cast %base_buffer to offset: [%sum], sizes: [32], strides: [1] : memref<f32, 1> to memref<32xf32, strided<[1], offset: ?>, 1>
  %cast = builtin.unrealized_conversion_cast %reinterpret_cast : memref<32xf32, strided<[1], offset: ?>, 1> to !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
  %ptr = llvm.extractvalue %cast[1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
  %c0 = arith.constant 0 : index
  %c0_i64 = builtin.unrealized_conversion_cast %c0 : index to i64
  %gep = llvm.getelementptr %ptr[%c0_i64] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
  %val = llvm.load %gep : !llvm.ptr<1> -> f32
  return
}

// CHECK-LABEL: @index_add_both_aligned
// CHECK: memref.reinterpret_cast {{.*}} {alignment = 64 : i64}
// CHECK: llvm.load {{.*}} {alignment = 64 : i64}

// -----
// Test E: Index addition with one operand aligned
// arg0 has divisibility=32 elements (128 bytes), arg1 has no divisibility
// Result has unknown alignment (aligned + unknown = unknown)
// This test verifies that we don't incorrectly propagate alignment
func.func @index_add_one_aligned(%arg0: index {tt.divisibility = 32 : i32}, %arg1: index, %arg2: memref<1024xf32, 1> {tt.divisibility = 128 : i32}) {
  %sum = arith.addi %arg0, %arg1 : index
  %base_buffer, %offset, %sizes, %strides = memref.extract_strided_metadata %arg2 : memref<1024xf32, 1> -> memref<f32, 1>, index, index, index
  %reinterpret_cast = memref.reinterpret_cast %base_buffer to offset: [%sum], sizes: [32], strides: [1] : memref<f32, 1> to memref<32xf32, strided<[1], offset: ?>, 1>
  %cast = builtin.unrealized_conversion_cast %reinterpret_cast : memref<32xf32, strided<[1], offset: ?>, 1> to !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
  %ptr = llvm.extractvalue %cast[1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
  %c0 = arith.constant 0 : index
  %c0_i64 = builtin.unrealized_conversion_cast %c0 : index to i64
  %gep = llvm.getelementptr %ptr[%c0_i64] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
  %val = llvm.load %gep : !llvm.ptr<1> -> f32
  return
}

// CHECK-LABEL: @index_add_one_aligned
// CHECK-NOT: memref.reinterpret_cast {{.*}} {alignment
// CHECK: llvm.load {{.*}} {alignment = 4 : i64}

// -----
// Test F: Simple function call with aligned index
// Caller passes an aligned index to callee, which uses it for memory access
func.func @callee_simple(%idx: index {tt.divisibility = 32 : i32}, %buf: memref<1024xf32, 1> {tt.divisibility = 128 : i32}) {
  %base_buffer, %offset, %sizes, %strides = memref.extract_strided_metadata %buf : memref<1024xf32, 1> -> memref<f32, 1>, index, index, index
  %reinterpret_cast = memref.reinterpret_cast %base_buffer to offset: [%idx], sizes: [32], strides: [1] : memref<f32, 1> to memref<32xf32, strided<[1], offset: ?>, 1>
  %cast = builtin.unrealized_conversion_cast %reinterpret_cast : memref<32xf32, strided<[1], offset: ?>, 1> to !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
  %ptr = llvm.extractvalue %cast[1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
  %c0 = arith.constant 0 : index
  %c0_i64 = builtin.unrealized_conversion_cast %c0 : index to i64
  %gep = llvm.getelementptr %ptr[%c0_i64] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
  %val = llvm.load %gep : !llvm.ptr<1> -> f32
  return
}

// CHECK-LABEL: @callee_simple
// CHECK: memref.reinterpret_cast {{.*}} {alignment = 128 : i64}
// CHECK: llvm.load {{.*}} {alignment = 128 : i64}

// -----
// Test G: Function call with index used in arithmetic
// Callee receives aligned index and adds it to another aligned value
func.func @callee_with_arithmetic(%idx: index {tt.divisibility = 16 : i32}, %buf: memref<1024xf32, 1> {tt.divisibility = 128 : i32}) {
  %c16 = arith.constant 16 : index
  %new_idx = arith.addi %idx, %c16 : index
  %base_buffer, %offset, %sizes, %strides = memref.extract_strided_metadata %buf : memref<1024xf32, 1> -> memref<f32, 1>, index, index, index
  %reinterpret_cast = memref.reinterpret_cast %base_buffer to offset: [%new_idx], sizes: [16], strides: [1] : memref<f32, 1> to memref<16xf32, strided<[1], offset: ?>, 1>
  %cast = builtin.unrealized_conversion_cast %reinterpret_cast : memref<16xf32, strided<[1], offset: ?>, 1> to !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
  %ptr = llvm.extractvalue %cast[1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
  %c0 = arith.constant 0 : index
  %c0_i64 = builtin.unrealized_conversion_cast %c0 : index to i64
  %gep = llvm.getelementptr %ptr[%c0_i64] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
  %val = llvm.load %gep : !llvm.ptr<1> -> f32
  return
}

// CHECK-LABEL: @callee_with_arithmetic
// CHECK: memref.reinterpret_cast {{.*}} {alignment = 64 : i64}
// CHECK: llvm.load {{.*}} {alignment = 64 : i64}

// -----
// Test H: Cross-function alignment propagation (Caller → Callee)
// This test validates cross-function alignment propagation needed for the
// multithreading optimization, which adds calls to functions representing
// work to be done. The callee starts with NO tt.divisibility on its args;
// Phase 2 of the pass must infer them from the caller's call site
// (`%c64` for the index, and `%buf`'s tt.divisibility=128 for the memref).
func.func @callee_mt_work(%idx: index, %buf: memref<1024xf32, 1>) {
  %base_buffer, %offset, %sizes, %strides = memref.extract_strided_metadata %buf : memref<1024xf32, 1> -> memref<f32, 1>, index, index, index
  %reinterpret_cast = memref.reinterpret_cast %base_buffer to offset: [%idx], sizes: [32], strides: [1] : memref<f32, 1> to memref<32xf32, strided<[1], offset: ?>, 1>
  %cast = builtin.unrealized_conversion_cast %reinterpret_cast : memref<32xf32, strided<[1], offset: ?>, 1> to !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
  %ptr = llvm.extractvalue %cast[1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
  %c0 = arith.constant 0 : index
  %c0_i64 = builtin.unrealized_conversion_cast %c0 : index to i64
  %gep = llvm.getelementptr %ptr[%c0_i64] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
  %val = llvm.load %gep : !llvm.ptr<1> -> f32
  return
}

func.func @caller_mt(%buf: memref<1024xf32, 1> {tt.divisibility = 128 : i32}) {
  %c64 = arith.constant 64 : index
  func.call @callee_mt_work(%c64, %buf) : (index, memref<1024xf32, 1>) -> ()
  return
}

// Phase 2 must rewrite the callee's signature with inferred tt.divisibility
// before propagating, then load alignment must reflect the inferred bounds.
// CHECK-LABEL: func.func @callee_mt_work(
// CHECK-SAME: tt.divisibility = 64
// CHECK-SAME: tt.divisibility = 128
// CHECK: memref.reinterpret_cast {{.*}} {alignment = 128 : i64}
// CHECK: llvm.load {{.*}} {alignment = 128 : i64}

// -----
// Test I: Function call where passed index increments another aligned index
// Callee receives aligned index and uses it to increment a loop variable
func.func @callee_increment_aligned(%base_idx: index {tt.divisibility = 32 : i32}, %buf: memref<1024xf32, 1> {tt.divisibility = 128 : i32}) {
  %c0 = arith.constant 0 : index
  %c1024 = arith.constant 1024 : index
  cf.br ^bb1(%c0 : index)
^bb1(%loop_idx: index):
  %cond = arith.cmpi slt, %loop_idx, %c1024 : index
  cf.cond_br %cond, ^bb2, ^bb3
^bb2:
  %combined_idx = arith.addi %loop_idx, %base_idx : index
  %base_buffer, %offset, %sizes, %strides = memref.extract_strided_metadata %buf : memref<1024xf32, 1> -> memref<f32, 1>, index, index, index
  %reinterpret_cast = memref.reinterpret_cast %base_buffer to offset: [%combined_idx], sizes: [32], strides: [1] : memref<f32, 1> to memref<32xf32, strided<[1], offset: ?>, 1>
  %cast = builtin.unrealized_conversion_cast %reinterpret_cast : memref<32xf32, strided<[1], offset: ?>, 1> to !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
  %ptr = llvm.extractvalue %cast[1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
  %c0_2 = arith.constant 0 : index
  %c0_i64 = builtin.unrealized_conversion_cast %c0_2 : index to i64
  %gep = llvm.getelementptr %ptr[%c0_i64] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
  %val = llvm.load %gep : !llvm.ptr<1> -> f32
  %c32 = arith.constant 32 : index
  %next_idx = arith.addi %loop_idx, %c32 : index
  cf.br ^bb1(%next_idx : index)
^bb3:
  return
}

// CHECK-LABEL: @callee_increment_aligned
// CHECK: memref.reinterpret_cast {{.*}} {alignment = 128 : i64}
// CHECK: llvm.load {{.*}} {alignment = 128 : i64}

// -----
// Test J: Loop header reached by multiple forward edges with different
// constant initial values.
//
// ^bb1 has three predecessors:
//   ^entry_a (cf.br, forward edge): initial = 48 elements
//   ^entry_b (cf.br, forward edge): initial = 0 elements
//   ^bb2     (cf.br, back edge):    step    = 32 elements
//
// Conservative divisibility of %idx:
//   GCD(GCD(48, 0), 32) = GCD(48, 32) = 16 elements = 64 bytes for f32.
// Combined with base alignment 128: GCD(128, 64) = 64.
//
// Before the fix in analyzeLoopBlockArguments, only the first forward
// edge's init was recorded (whichever the predecessor walk visits first),
// so the pass over-estimated alignment as 128 bytes. With the GCD-merge
// fix, alignment is correctly reported as 64.
func.func @two_forward_edges_different_init(
    %arg0: memref<1024xf32, 1> {tt.divisibility = 128 : i32},
    %cond: i1) {
  %c0   = arith.constant 0    : index
  %c48  = arith.constant 48   : index
  %c32  = arith.constant 32   : index
  %c1024 = arith.constant 1024 : index
  cf.cond_br %cond, ^entry_a, ^entry_b
^entry_a:
  cf.br ^bb1(%c48 : index)
^entry_b:
  cf.br ^bb1(%c0 : index)
^bb1(%idx: index):
  %cond2 = arith.cmpi slt, %idx, %c1024 : index
  cf.cond_br %cond2, ^bb2, ^bb3
^bb2:
  %base_buffer, %offset, %sizes, %strides = memref.extract_strided_metadata %arg0 : memref<1024xf32, 1> -> memref<f32, 1>, index, index, index
  %reinterpret_cast = memref.reinterpret_cast %base_buffer to offset: [%idx], sizes: [32], strides: [1] : memref<f32, 1> to memref<32xf32, strided<[1], offset: ?>, 1>
  %cast = builtin.unrealized_conversion_cast %reinterpret_cast : memref<32xf32, strided<[1], offset: ?>, 1> to !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
  %ptr = llvm.extractvalue %cast[1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
  %c0_2 = arith.constant 0 : index
  %c0_i64 = builtin.unrealized_conversion_cast %c0_2 : index to i64
  %gep = llvm.getelementptr %ptr[%c0_i64] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
  %val = llvm.load %gep : !llvm.ptr<1> -> f32
  %next_idx = arith.addi %idx, %c32 : index
  cf.br ^bb1(%next_idx : index)
^bb3:
  return
}

// CHECK-LABEL: @two_forward_edges_different_init
// CHECK: memref.reinterpret_cast {{.*}} {alignment = 64 : i64}
// CHECK: llvm.load {{.*}} {alignment = 64 : i64}

// -----
// Test U: Two kernels with different divisibility call the same callee.
// K1 (tt.div=128) and K2 (tt.div=64) both pass their memref to @multi_caller_gcd.
// Expected: callee arg gets tt.divisibility = GCD(128, 64) = 64.
func.func @kernel_multi_a(%buf: memref<1024xf32, 1> {tt.divisibility = 128 : i32}) {
  func.call @multi_caller_gcd(%buf) : (memref<1024xf32, 1>) -> ()
  return
}

func.func @kernel_multi_b(%buf: memref<1024xf32, 1> {tt.divisibility = 64 : i32}) {
  func.call @multi_caller_gcd(%buf) : (memref<1024xf32, 1>) -> ()
  return
}

func.func @multi_caller_gcd(%buf: memref<1024xf32, 1>) {
  %base_buffer, %offset, %sizes, %strides = memref.extract_strided_metadata %buf : memref<1024xf32, 1> -> memref<f32, 1>, index, index, index
  %rc = memref.reinterpret_cast %base_buffer to offset: [0], sizes: [32], strides: [1] : memref<f32, 1> to memref<32xf32, strided<[1]>, 1>
  %cast = builtin.unrealized_conversion_cast %rc : memref<32xf32, strided<[1]>, 1> to !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
  %ptr = llvm.extractvalue %cast[1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
  %c0 = arith.constant 0 : index
  %c0_i64 = builtin.unrealized_conversion_cast %c0 : index to i64
  %gep = llvm.getelementptr %ptr[%c0_i64] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
  %val = llvm.load %gep : !llvm.ptr<1> -> f32
  return
}

// CHECK-LABEL: func.func @multi_caller_gcd(
// CHECK-SAME: tt.divisibility = 64
// CHECK: memref.reinterpret_cast {{.*}} {alignment = 64 : i64}
// CHECK: llvm.load {{.*}} {alignment = 64 : i64}

// -----
// Test V: One kernel caller and one non-kernel caller — callee gets no guarantee.
// K (tt.div=128) calls @one_unknown_caller. N (no tt.div) also calls it.
// Since N is never processed, its contribution is unknown → callee arg gets
// no tt.divisibility attribute.
func.func @kernel_known(%buf: memref<1024xf32, 1> {tt.divisibility = 128 : i32}) {
  func.call @one_unknown_caller(%buf) : (memref<1024xf32, 1>) -> ()
  return
}

func.func @non_kernel(%buf: memref<1024xf32, 1>) {
  func.call @one_unknown_caller(%buf) : (memref<1024xf32, 1>) -> ()
  return
}

func.func @one_unknown_caller(%buf: memref<1024xf32, 1>) {
  %base_buffer, %offset, %sizes, %strides = memref.extract_strided_metadata %buf : memref<1024xf32, 1> -> memref<f32, 1>, index, index, index
  %rc = memref.reinterpret_cast %base_buffer to offset: [0], sizes: [32], strides: [1] : memref<f32, 1> to memref<32xf32, strided<[1]>, 1>
  %cast = builtin.unrealized_conversion_cast %rc : memref<32xf32, strided<[1]>, 1> to !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
  %ptr = llvm.extractvalue %cast[1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
  %c0 = arith.constant 0 : index
  %c0_i64 = builtin.unrealized_conversion_cast %c0 : index to i64
  %gep = llvm.getelementptr %ptr[%c0_i64] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
  %val = llvm.load %gep : !llvm.ptr<1> -> f32
  return
}

// CHECK-LABEL: func.func @one_unknown_caller(
// CHECK-NOT: tt.divisibility
// CHECK: %base_buffer

// -----
// Test W: 3-level call chain A → B → C.
// A (tt.div=128) calls B; B calls C.
// B gets tt.div=128, which it propagates when calling C → C also gets tt.div=128.
func.func @chain3_a(%buf: memref<1024xf32, 1> {tt.divisibility = 128 : i32}) {
  func.call @chain3_b(%buf) : (memref<1024xf32, 1>) -> ()
  return
}

func.func @chain3_b(%buf: memref<1024xf32, 1>) {
  func.call @chain3_c(%buf) : (memref<1024xf32, 1>) -> ()
  return
}

func.func @chain3_c(%buf: memref<1024xf32, 1>) {
  %base_buffer, %offset, %sizes, %strides = memref.extract_strided_metadata %buf : memref<1024xf32, 1> -> memref<f32, 1>, index, index, index
  %rc = memref.reinterpret_cast %base_buffer to offset: [0], sizes: [32], strides: [1] : memref<f32, 1> to memref<32xf32, strided<[1]>, 1>
  %cast = builtin.unrealized_conversion_cast %rc : memref<32xf32, strided<[1]>, 1> to !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
  %ptr = llvm.extractvalue %cast[1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
  %c0 = arith.constant 0 : index
  %c0_i64 = builtin.unrealized_conversion_cast %c0 : index to i64
  %gep = llvm.getelementptr %ptr[%c0_i64] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
  %val = llvm.load %gep : !llvm.ptr<1> -> f32
  return
}

// CHECK-LABEL: func.func @chain3_b(
// CHECK-SAME: tt.divisibility = 128
// CHECK-LABEL: func.func @chain3_c(
// CHECK-SAME: tt.divisibility = 128
// CHECK: llvm.load {{.*}} {alignment = 128 : i64}

// -----
// Test X: 4-level call chain A → B → C → D.
// Alignment 128 propagates all the way from A's kernel arg to D's load.
func.func @chain4_a(%buf: memref<1024xf32, 1> {tt.divisibility = 128 : i32}) {
  func.call @chain4_b(%buf) : (memref<1024xf32, 1>) -> ()
  return
}

func.func @chain4_b(%buf: memref<1024xf32, 1>) {
  func.call @chain4_c(%buf) : (memref<1024xf32, 1>) -> ()
  return
}

func.func @chain4_c(%buf: memref<1024xf32, 1>) {
  func.call @chain4_d(%buf) : (memref<1024xf32, 1>) -> ()
  return
}

func.func @chain4_d(%buf: memref<1024xf32, 1>) {
  %base_buffer, %offset, %sizes, %strides = memref.extract_strided_metadata %buf : memref<1024xf32, 1> -> memref<f32, 1>, index, index, index
  %rc = memref.reinterpret_cast %base_buffer to offset: [0], sizes: [32], strides: [1] : memref<f32, 1> to memref<32xf32, strided<[1]>, 1>
  %cast = builtin.unrealized_conversion_cast %rc : memref<32xf32, strided<[1]>, 1> to !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
  %ptr = llvm.extractvalue %cast[1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
  %c0 = arith.constant 0 : index
  %c0_i64 = builtin.unrealized_conversion_cast %c0 : index to i64
  %gep = llvm.getelementptr %ptr[%c0_i64] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
  %val = llvm.load %gep : !llvm.ptr<1> -> f32
  return
}

// CHECK-LABEL: func.func @chain4_c(
// CHECK-SAME: tt.divisibility = 128
// CHECK-LABEL: func.func @chain4_d(
// CHECK-SAME: tt.divisibility = 128
// CHECK: llvm.load {{.*}} {alignment = 128 : i64}

// -----
// Test Y: Diamond — two kernels feed the same two intermediate functions,
// which then both call the same terminal callee.
// K1 (tt.div=128) → branch_b → leaf_d  (branch_b contributes div=128)
// K2 (tt.div=64)  → branch_c → leaf_d  (branch_c contributes div=64)
// leaf_d arg = GCD(128, 64) = 64.
func.func @diamond_k1(%buf: memref<1024xf32, 1> {tt.divisibility = 128 : i32}) {
  func.call @diamond_b(%buf) : (memref<1024xf32, 1>) -> ()
  return
}

func.func @diamond_k2(%buf: memref<1024xf32, 1> {tt.divisibility = 64 : i32}) {
  func.call @diamond_c(%buf) : (memref<1024xf32, 1>) -> ()
  return
}

func.func @diamond_b(%buf: memref<1024xf32, 1>) {
  func.call @diamond_leaf(%buf) : (memref<1024xf32, 1>) -> ()
  return
}

func.func @diamond_c(%buf: memref<1024xf32, 1>) {
  func.call @diamond_leaf(%buf) : (memref<1024xf32, 1>) -> ()
  return
}

func.func @diamond_leaf(%buf: memref<1024xf32, 1>) {
  %base_buffer, %offset, %sizes, %strides = memref.extract_strided_metadata %buf : memref<1024xf32, 1> -> memref<f32, 1>, index, index, index
  %rc = memref.reinterpret_cast %base_buffer to offset: [0], sizes: [32], strides: [1] : memref<f32, 1> to memref<32xf32, strided<[1]>, 1>
  %cast = builtin.unrealized_conversion_cast %rc : memref<32xf32, strided<[1]>, 1> to !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
  %ptr = llvm.extractvalue %cast[1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
  %c0 = arith.constant 0 : index
  %c0_i64 = builtin.unrealized_conversion_cast %c0 : index to i64
  %gep = llvm.getelementptr %ptr[%c0_i64] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
  %val = llvm.load %gep : !llvm.ptr<1> -> f32
  return
}

// CHECK-LABEL: func.func @diamond_leaf(
// CHECK-SAME: tt.divisibility = 64
// CHECK: memref.reinterpret_cast {{.*}} {alignment = 64 : i64}
// CHECK: llvm.load {{.*}} {alignment = 64 : i64}

// -----
// Test K: stride=2 f32 — |2-1|*4 = 4.
func.func @strided_s2(%arg0: memref<1024xf32, 1> {tt.divisibility = 128 : i32}) {
  %base_buffer, %offset, %sizes, %strides = memref.extract_strided_metadata %arg0 : memref<1024xf32, 1> -> memref<f32, 1>, index, index, index
  %reinterpret_cast = memref.reinterpret_cast %base_buffer to offset: [0], sizes: [8], strides: [2] : memref<f32, 1> to memref<8xf32, strided<[2]>, 1>
  %cast = builtin.unrealized_conversion_cast %reinterpret_cast : memref<8xf32, strided<[2]>, 1> to !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
  %ptr = llvm.extractvalue %cast[1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
  %c0 = arith.constant 0 : index
  %c0_i64 = builtin.unrealized_conversion_cast %c0 : index to i64
  %gep = llvm.getelementptr %ptr[%c0_i64] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
  %val = llvm.load %gep : !llvm.ptr<1> -> f32
  return
}

// CHECK-LABEL: @strided_s2
// CHECK: memref.reinterpret_cast {{.*}} {alignment = 4 : i64}
// CHECK: llvm.load {{.*}} {alignment = 128 : i64}

// -----
// Test L: stride=3 f32 — classic adversarial; |3-1|*4 = 8.
func.func @strided_s3(%arg0: memref<1024xf32, 1> {tt.divisibility = 128 : i32}) {
  %base_buffer, %offset, %sizes, %strides = memref.extract_strided_metadata %arg0 : memref<1024xf32, 1> -> memref<f32, 1>, index, index, index
  %reinterpret_cast = memref.reinterpret_cast %base_buffer to offset: [0], sizes: [8], strides: [3] : memref<f32, 1> to memref<8xf32, strided<[3]>, 1>
  %cast = builtin.unrealized_conversion_cast %reinterpret_cast : memref<8xf32, strided<[3]>, 1> to !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
  %ptr = llvm.extractvalue %cast[1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
  %c0 = arith.constant 0 : index
  %c0_i64 = builtin.unrealized_conversion_cast %c0 : index to i64
  %gep = llvm.getelementptr %ptr[%c0_i64] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
  %val = llvm.load %gep : !llvm.ptr<1> -> f32
  return
}

// CHECK-LABEL: @strided_s3
// CHECK: memref.reinterpret_cast {{.*}} {alignment = 8 : i64}
// CHECK: llvm.load {{.*}} {alignment = 128 : i64}

// -----
// Test M: stride=5 f32 — |5-1|*4 = 16.
func.func @strided_s5(%arg0: memref<1024xf32, 1> {tt.divisibility = 128 : i32}) {
  %base_buffer, %offset, %sizes, %strides = memref.extract_strided_metadata %arg0 : memref<1024xf32, 1> -> memref<f32, 1>, index, index, index
  %reinterpret_cast = memref.reinterpret_cast %base_buffer to offset: [0], sizes: [8], strides: [5] : memref<f32, 1> to memref<8xf32, strided<[5]>, 1>
  %cast = builtin.unrealized_conversion_cast %reinterpret_cast : memref<8xf32, strided<[5]>, 1> to !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
  %ptr = llvm.extractvalue %cast[1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
  %c0 = arith.constant 0 : index
  %c0_i64 = builtin.unrealized_conversion_cast %c0 : index to i64
  %gep = llvm.getelementptr %ptr[%c0_i64] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
  %val = llvm.load %gep : !llvm.ptr<1> -> f32
  return
}

// CHECK-LABEL: @strided_s5
// CHECK: memref.reinterpret_cast {{.*}} {alignment = 16 : i64}
// CHECK: llvm.load {{.*}} {alignment = 128 : i64}

// -----
// Test N: stride=9 f32 — |9-1|*4 = 32.
func.func @strided_s9(%arg0: memref<1024xf32, 1> {tt.divisibility = 128 : i32}) {
  %base_buffer, %offset, %sizes, %strides = memref.extract_strided_metadata %arg0 : memref<1024xf32, 1> -> memref<f32, 1>, index, index, index
  %reinterpret_cast = memref.reinterpret_cast %base_buffer to offset: [0], sizes: [8], strides: [9] : memref<f32, 1> to memref<8xf32, strided<[9]>, 1>
  %cast = builtin.unrealized_conversion_cast %reinterpret_cast : memref<8xf32, strided<[9]>, 1> to !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
  %ptr = llvm.extractvalue %cast[1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
  %c0 = arith.constant 0 : index
  %c0_i64 = builtin.unrealized_conversion_cast %c0 : index to i64
  %gep = llvm.getelementptr %ptr[%c0_i64] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
  %val = llvm.load %gep : !llvm.ptr<1> -> f32
  return
}

// CHECK-LABEL: @strided_s9
// CHECK: memref.reinterpret_cast {{.*}} {alignment = 32 : i64}
// CHECK: llvm.load {{.*}} {alignment = 128 : i64}

// -----
// Test O: stride=17 f32 — |17-1|*4 = 64.
func.func @strided_s17(%arg0: memref<1024xf32, 1> {tt.divisibility = 128 : i32}) {
  %base_buffer, %offset, %sizes, %strides = memref.extract_strided_metadata %arg0 : memref<1024xf32, 1> -> memref<f32, 1>, index, index, index
  %reinterpret_cast = memref.reinterpret_cast %base_buffer to offset: [0], sizes: [8], strides: [17] : memref<f32, 1> to memref<8xf32, strided<[17]>, 1>
  %cast = builtin.unrealized_conversion_cast %reinterpret_cast : memref<8xf32, strided<[17]>, 1> to !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
  %ptr = llvm.extractvalue %cast[1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
  %c0 = arith.constant 0 : index
  %c0_i64 = builtin.unrealized_conversion_cast %c0 : index to i64
  %gep = llvm.getelementptr %ptr[%c0_i64] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
  %val = llvm.load %gep : !llvm.ptr<1> -> f32
  return
}

// CHECK-LABEL: @strided_s17
// CHECK: memref.reinterpret_cast {{.*}} {alignment = 64 : i64}
// CHECK: llvm.load {{.*}} {alignment = 128 : i64}

// -----
// Test P: stride=33 f32 — preservation; |33-1|*4 = 128 = base, no tightening.
func.func @strided_s33(%arg0: memref<1024xf32, 1> {tt.divisibility = 128 : i32}) {
  %base_buffer, %offset, %sizes, %strides = memref.extract_strided_metadata %arg0 : memref<1024xf32, 1> -> memref<f32, 1>, index, index, index
  %reinterpret_cast = memref.reinterpret_cast %base_buffer to offset: [0], sizes: [8], strides: [33] : memref<f32, 1> to memref<8xf32, strided<[33]>, 1>
  %cast = builtin.unrealized_conversion_cast %reinterpret_cast : memref<8xf32, strided<[33]>, 1> to !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
  %ptr = llvm.extractvalue %cast[1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
  %c0 = arith.constant 0 : index
  %c0_i64 = builtin.unrealized_conversion_cast %c0 : index to i64
  %gep = llvm.getelementptr %ptr[%c0_i64] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
  %val = llvm.load %gep : !llvm.ptr<1> -> f32
  return
}

// CHECK-LABEL: @strided_s33
// CHECK: memref.reinterpret_cast {{.*}} {alignment = 128 : i64}
// CHECK: llvm.load {{.*}} {alignment = 128 : i64}

// -----
// Test Q: stride=0 broadcast — skipped, no tightening.
func.func @strided_s0_broadcast(%arg0: memref<1024xf32, 1> {tt.divisibility = 128 : i32}) {
  %base_buffer, %offset, %sizes, %strides = memref.extract_strided_metadata %arg0 : memref<1024xf32, 1> -> memref<f32, 1>, index, index, index
  %reinterpret_cast = memref.reinterpret_cast %base_buffer to offset: [0], sizes: [8], strides: [0] : memref<f32, 1> to memref<8xf32, strided<[0]>, 1>
  %cast = builtin.unrealized_conversion_cast %reinterpret_cast : memref<8xf32, strided<[0]>, 1> to !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
  %ptr = llvm.extractvalue %cast[1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
  %c0 = arith.constant 0 : index
  %c0_i64 = builtin.unrealized_conversion_cast %c0 : index to i64
  %gep = llvm.getelementptr %ptr[%c0_i64] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
  %val = llvm.load %gep : !llvm.ptr<1> -> f32
  return
}

// CHECK-LABEL: @strided_s0_broadcast
// CHECK: memref.reinterpret_cast {{.*}} {alignment = 128 : i64}
// CHECK: llvm.load {{.*}} {alignment = 128 : i64}

// -----
// Test R: offset=16 + stride=3 f32 — GCD(128, 64, 8) = 8; load via element-0 = 64.
func.func @strided_offset16_s3(%arg0: memref<1024xf32, 1> {tt.divisibility = 128 : i32}) {
  %base_buffer, %offset, %sizes, %strides = memref.extract_strided_metadata %arg0 : memref<1024xf32, 1> -> memref<f32, 1>, index, index, index
  %reinterpret_cast = memref.reinterpret_cast %base_buffer to offset: [16], sizes: [8], strides: [3] : memref<f32, 1> to memref<8xf32, strided<[3], offset: 16>, 1>
  %cast = builtin.unrealized_conversion_cast %reinterpret_cast : memref<8xf32, strided<[3], offset: 16>, 1> to !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
  %ptr = llvm.extractvalue %cast[1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
  %c0 = arith.constant 0 : index
  %c0_i64 = builtin.unrealized_conversion_cast %c0 : index to i64
  %gep = llvm.getelementptr %ptr[%c0_i64] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
  %val = llvm.load %gep : !llvm.ptr<1> -> f32
  return
}

// CHECK-LABEL: @strided_offset16_s3
// CHECK: memref.reinterpret_cast {{.*}} {alignment = 8 : i64}
// CHECK: llvm.load {{.*}} {alignment = 64 : i64}

// -----
// Test S: 2D strides=[3,1] f32 — outer fold; GCD(128, 8) = 8.
func.func @strided_2d_3_1(%arg0: memref<1024xf32, 1> {tt.divisibility = 128 : i32}) {
  %base_buffer, %offset, %sizes, %strides = memref.extract_strided_metadata %arg0 : memref<1024xf32, 1> -> memref<f32, 1>, index, index, index
  %reinterpret_cast = memref.reinterpret_cast %base_buffer to offset: [0], sizes: [4, 8], strides: [3, 1] : memref<f32, 1> to memref<4x8xf32, strided<[3, 1]>, 1>
  %cast = builtin.unrealized_conversion_cast %reinterpret_cast : memref<4x8xf32, strided<[3, 1]>, 1> to !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
  %ptr = llvm.extractvalue %cast[1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
  %c0 = arith.constant 0 : index
  %c0_i64 = builtin.unrealized_conversion_cast %c0 : index to i64
  %gep = llvm.getelementptr %ptr[%c0_i64] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
  %val = llvm.load %gep : !llvm.ptr<1> -> f32
  return
}

// CHECK-LABEL: @strided_2d_3_1
// CHECK: memref.reinterpret_cast {{.*}} {alignment = 8 : i64}
// CHECK: llvm.load {{.*}} {alignment = 128 : i64}

// -----
// Test T: 2D strides=[5,5] f32 — both ranks fold; GCD(128, 16, 16) = 16.
func.func @strided_2d_5_5(%arg0: memref<1024xf32, 1> {tt.divisibility = 128 : i32}) {
  %base_buffer, %offset, %sizes, %strides = memref.extract_strided_metadata %arg0 : memref<1024xf32, 1> -> memref<f32, 1>, index, index, index
  %reinterpret_cast = memref.reinterpret_cast %base_buffer to offset: [0], sizes: [4, 4], strides: [5, 5] : memref<f32, 1> to memref<4x4xf32, strided<[5, 5]>, 1>
  %cast = builtin.unrealized_conversion_cast %reinterpret_cast : memref<4x4xf32, strided<[5, 5]>, 1> to !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
  %ptr = llvm.extractvalue %cast[1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
  %c0 = arith.constant 0 : index
  %c0_i64 = builtin.unrealized_conversion_cast %c0 : index to i64
  %gep = llvm.getelementptr %ptr[%c0_i64] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
  %val = llvm.load %gep : !llvm.ptr<1> -> f32
  return
}

// CHECK-LABEL: @strided_2d_5_5
// CHECK: memref.reinterpret_cast {{.*}} {alignment = 16 : i64}
// CHECK: llvm.load {{.*}} {alignment = 128 : i64}

// -----
// Test with all optimizations on (multithreading, tiling, hexagonmem, DMA, double buffering)
// This test ensures that alignment propagation works correctly through:
// - async functions and multithreading
// - hexagonmem.alloc operations
// - DMA operations
// - block arguments and function calls

// CHECK-LABEL: @add_kernel
// CHECK: memref.reinterpret_cast {{.*}} {alignment = 128 : i64}
// CHECK: memref.reinterpret_cast {{.*}} {alignment = 128 : i64}
// CHECK: memref.reinterpret_cast {{.*}} {alignment = 128 : i64}
// CHECK: memref.reinterpret_cast {{.*}} {alignment = 128 : i64}
// CHECK: memref.reinterpret_cast {{.*}} {alignment = 128 : i64}
// CHECK: memref.reinterpret_cast {{.*}} {alignment = 128 : i64}
// CHECK: llvm.load {{.*}} {alignment = 128 : i64}
// CHECK: llvm.load {{.*}} {alignment = 128 : i64}
// CHECK: llvm.store {{.*}} {alignment = 128 : i64}

module attributes {llvm.target_triple = "hexagon"} {
  func.func @add_kernel(%arg0: memref<*xf32> {tt.divisibility = 128 : i32}, %arg1: memref<*xf32> {tt.divisibility = 128 : i32}, %arg2: memref<*xf32> {tt.divisibility = 128 : i32}, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32) {
    %true = arith.constant true
    %c4 = arith.constant 4 : index
    %c131072 = arith.constant 131072 : index
    %c32768 = arith.constant 32768 : index
    %c0 = arith.constant 0 : index
    %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [0], sizes: [131072], strides: [1] : memref<*xf32> to memref<131072xf32, strided<[1]>>
    %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [0], sizes: [131072], strides: [1] : memref<*xf32> to memref<131072xf32, strided<[1]>>
    %0 = hexagonmem.alloc() : memref<131072xf32, 1>
    %alloc = memref.alloc() : memref<1xi32>
    memref.dma_start %reinterpret_cast_0[%c0], %0[%c0], %c131072, %alloc[%c0] : memref<131072xf32, strided<[1]>>, memref<131072xf32, 1>, memref<1xi32>
    memref.dma_wait %alloc[%c0], %c131072 : memref<1xi32>
    memref.dealloc %alloc : memref<1xi32>
    %alloc_1 = memref.alloc() {alignment = 128 : i64} : memref<131072xf32>
    %1 = hexagonmem.alloc() : memref<131072xf32, 1>
    %alloc_2 = memref.alloc() : memref<1xi32>
    memref.dma_start %alloc_1[%c0], %1[%c0], %c131072, %alloc_2[%c0] : memref<131072xf32>, memref<131072xf32, 1>, memref<1xi32>
    memref.dma_wait %alloc_2[%c0], %c131072 : memref<1xi32>
    memref.dealloc %alloc_2 : memref<1xi32>
    %reinterpret_cast_3 = memref.reinterpret_cast %arg1 to offset: [0], sizes: [131072], strides: [1] : memref<*xf32> to memref<131072xf32, strided<[1]>>
    %2 = hexagonmem.alloc() : memref<131072xf32, 1>
    %alloc_4 = memref.alloc() : memref<1xi32>
    memref.dma_start %reinterpret_cast_3[%c0], %2[%c0], %c131072, %alloc_4[%c0] : memref<131072xf32, strided<[1]>>, memref<131072xf32, 1>, memref<1xi32>
    memref.dma_wait %alloc_4[%c0], %c131072 : memref<1xi32>
    memref.dealloc %alloc_4 : memref<1xi32>
    %3 = async.runtime.create_group %c4 : !async.group
    cf.br ^bb1(%c0 : index)
  ^bb1(%4: index):  // 2 preds: ^bb0, ^bb2
    %5 = arith.cmpi slt, %4, %c131072 : index
    cf.cond_br %5, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    %6 = call @async_execute_fn(%0, %4, %2, %1) : (memref<131072xf32, 1>, index, memref<131072xf32, 1>, memref<131072xf32, 1>) -> !async.token
    %7 = async.runtime.add_to_group %6, %3 : !async.token
    %8 = arith.addi %4, %c32768 : index
    cf.br ^bb1(%8 : index)
  ^bb3:  // pred: ^bb1
    async.runtime.await %3 : !async.group
    %9 = async.runtime.is_error %3 : !async.group
    %10 = arith.xori %9, %true : i1
    cf.assert %10, "Awaited async operand is in error state"
    %alloc_5 = memref.alloc() : memref<1xi32>
    memref.dma_start %1[%c0], %reinterpret_cast[%c0], %c131072, %alloc_5[%c0] : memref<131072xf32, 1>, memref<131072xf32, strided<[1]>>, memref<1xi32>
    memref.dma_wait %alloc_5[%c0], %c131072 : memref<1xi32>
    memref.dealloc %alloc_5 : memref<1xi32>
    hexagonmem.dealloc %0 : memref<131072xf32, 1>
    memref.dealloc %alloc_1 : memref<131072xf32>
    hexagonmem.dealloc %1 : memref<131072xf32, 1>
    hexagonmem.dealloc %2 : memref<131072xf32, 1>
    return
  }
  func.func private @async_execute_fn(%arg0: memref<131072xf32, 1>, %arg1: index, %arg2: memref<131072xf32, 1>, %arg3: memref<131072xf32, 1>) -> !async.token attributes {passthrough = ["presplitcoroutine"]} {
    %c32 = arith.constant 32 : index
    %c32768 = arith.constant 32768 : index
    %c0 = arith.constant 0 : index
    %0 = builtin.unrealized_conversion_cast %c0 : index to i64
    %1 = async.runtime.create : !async.token
    %2 = async.coro.id
    %3 = async.coro.begin %2
    %4 = async.coro.save %3
    async.runtime.resume %3
    async.coro.suspend %4, ^bb6, ^bb1, ^bb5
  ^bb1:  // pred: ^bb0
    cf.br ^bb2(%c0 : index)
  ^bb2(%5: index):  // 2 preds: ^bb1, ^bb3
    %6 = arith.cmpi slt, %5, %c32768 : index
    cf.cond_br %6, ^bb3, ^bb4
  ^bb3:  // pred: ^bb2
    %base_buffer, %offset, %sizes, %strides = memref.extract_strided_metadata %arg0 : memref<131072xf32, 1> -> memref<f32, 1>, index, index, index
    %7 = arith.addi %arg1, %5 : index
    %reinterpret_cast = memref.reinterpret_cast %base_buffer to offset: [%7], sizes: [32], strides: [1] : memref<f32, 1> to memref<32xf32, strided<[1], offset: ?>, 1>
    %8 = builtin.unrealized_conversion_cast %reinterpret_cast : memref<32xf32, strided<[1], offset: ?>, 1> to !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
    %base_buffer_0, %offset_1, %sizes_2, %strides_3 = memref.extract_strided_metadata %arg2 : memref<131072xf32, 1> -> memref<f32, 1>, index, index, index
    %9 = arith.addi %arg1, %5 : index
    %reinterpret_cast_4 = memref.reinterpret_cast %base_buffer_0 to offset: [%9], sizes: [32], strides: [1] : memref<f32, 1> to memref<32xf32, strided<[1], offset: ?>, 1>
    %10 = builtin.unrealized_conversion_cast %reinterpret_cast_4 : memref<32xf32, strided<[1], offset: ?>, 1> to !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
    %base_buffer_5, %offset_6, %sizes_7, %strides_8 = memref.extract_strided_metadata %arg3 : memref<131072xf32, 1> -> memref<f32, 1>, index, index, index
    %11 = arith.addi %arg1, %5 : index
    %reinterpret_cast_9 = memref.reinterpret_cast %base_buffer_5 to offset: [%11], sizes: [32], strides: [1] : memref<f32, 1> to memref<32xf32, strided<[1], offset: ?>, 1>
    %12 = builtin.unrealized_conversion_cast %reinterpret_cast_9 : memref<32xf32, strided<[1], offset: ?>, 1> to !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
    %13 = llvm.extractvalue %8[1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)> 
    %14 = llvm.extractvalue %8[2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)> 
    %15 = llvm.getelementptr %13[%14] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
    %16 = llvm.getelementptr %15[%0] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
    %17 = llvm.load %16 {alignment = 4 : i64} : !llvm.ptr<1> -> vector<32xf32>
    %18 = llvm.extractvalue %10[1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)> 
    %19 = llvm.extractvalue %10[2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)> 
    %20 = llvm.getelementptr %18[%19] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
    %21 = llvm.getelementptr %20[%0] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
    %22 = llvm.load %21 {alignment = 4 : i64} : !llvm.ptr<1> -> vector<32xf32>
    %23 = arith.addf %17, %22 fastmath<fast> : vector<32xf32>
    %24 = llvm.extractvalue %12[1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)> 
    %25 = llvm.extractvalue %12[2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)> 
    %26 = llvm.getelementptr %24[%25] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
    %27 = llvm.getelementptr %26[%0] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
    llvm.store %23, %27 {alignment = 4 : i64} : vector<32xf32>, !llvm.ptr<1>
    %28 = arith.addi %5, %c32 : index
    cf.br ^bb2(%28 : index)
  ^bb4:  // pred: ^bb2
    async.runtime.set_available %1 : !async.token
    cf.br ^bb5
  ^bb5:  // 2 preds: ^bb0, ^bb4
    async.coro.free %2, %3
    cf.br ^bb6
  ^bb6:  // 2 preds: ^bb0, ^bb5
    async.coro.end %3
    return %1 : !async.token
  }
}

// -----
// Test with all optimizations other than vectorization off:
//        enableMultiThreading=False,
//        enableVTCMTiling=False,
//        enableConvertToHexagonmem=False,
//        enableHexagonmemCopyToDMA=False,
//        enableDoubleBuffering=False,
// This test ensures that the optimiation works correctly directly on memrefs


// CHECK-LABEL: @add_kernel
// CHECK: memref.reinterpret_cast {{.*}} {alignment = 128 : i64}
// CHECK: memref.reinterpret_cast {{.*}} {alignment = 128 : i64}
// CHECK: memref.reinterpret_cast {{.*}} {alignment = 128 : i64}
// CHECK: memref.reinterpret_cast {{.*}} {alignment = 128 : i64}
// CHECK: memref.reinterpret_cast {{.*}} {alignment = 128 : i64}
// CHECK: memref.reinterpret_cast {{.*}} {alignment = 128 : i64}
// CHECK: llvm.load {{.*}} {alignment = 128 : i64}
// CHECK: llvm.load {{.*}} {alignment = 128 : i64}
// CHECK: llvm.store {{.*}} {alignment = 128 : i64}

module attributes {llvm.target_triple = "hexagon"} {
  func.func @add_kernel(%arg0: memref<*xf32> {tt.divisibility = 128 : i32}, %arg1: memref<*xf32> {tt.divisibility = 128 : i32}, %arg2: memref<*xf32> {tt.divisibility = 128 : i32}, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32) {
    %c131072 = arith.constant 131072 : index
    %c0 = arith.constant 0 : index
    %0 = builtin.unrealized_conversion_cast %c0 : index to i64
    %c32 = arith.constant 32 : index
    %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [0], sizes: [131072], strides: [1] : memref<*xf32> to memref<131072xf32, strided<[1]>>
    %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [0], sizes: [131072], strides: [1] : memref<*xf32> to memref<131072xf32, strided<[1]>>
    %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [0], sizes: [131072], strides: [1] : memref<*xf32> to memref<131072xf32, strided<[1]>>
    cf.br ^bb1(%c0 : index)
  ^bb1(%1: index):  // 2 preds: ^bb0, ^bb2
    %2 = arith.cmpi slt, %1, %c131072 : index
    cf.cond_br %2, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    %base_buffer, %offset, %sizes, %strides = memref.extract_strided_metadata %reinterpret_cast_0 : memref<131072xf32, strided<[1]>> -> memref<f32>, index, index, index
    %reinterpret_cast_2 = memref.reinterpret_cast %base_buffer to offset: [%1], sizes: [32], strides: [1] : memref<f32> to memref<32xf32, strided<[1], offset: ?>>
    %3 = builtin.unrealized_conversion_cast %reinterpret_cast_2 : memref<32xf32, strided<[1], offset: ?>> to !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %base_buffer_3, %offset_4, %sizes_5, %strides_6 = memref.extract_strided_metadata %reinterpret_cast_1 : memref<131072xf32, strided<[1]>> -> memref<f32>, index, index, index
    %reinterpret_cast_7 = memref.reinterpret_cast %base_buffer_3 to offset: [%1], sizes: [32], strides: [1] : memref<f32> to memref<32xf32, strided<[1], offset: ?>>
    %4 = builtin.unrealized_conversion_cast %reinterpret_cast_7 : memref<32xf32, strided<[1], offset: ?>> to !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %base_buffer_8, %offset_9, %sizes_10, %strides_11 = memref.extract_strided_metadata %reinterpret_cast : memref<131072xf32, strided<[1]>> -> memref<f32>, index, index, index
    %reinterpret_cast_12 = memref.reinterpret_cast %base_buffer_8 to offset: [%1], sizes: [32], strides: [1] : memref<f32> to memref<32xf32, strided<[1], offset: ?>>
    %5 = builtin.unrealized_conversion_cast %reinterpret_cast_12 : memref<32xf32, strided<[1], offset: ?>> to !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %6 = llvm.extractvalue %3[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %7 = llvm.extractvalue %3[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %8 = llvm.getelementptr %6[%7] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %9 = llvm.getelementptr %8[%0] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %10 = llvm.load %9 {alignment = 4 : i64} : !llvm.ptr -> vector<32xf32>
    %11 = llvm.extractvalue %4[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %12 = llvm.extractvalue %4[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %13 = llvm.getelementptr %11[%12] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %14 = llvm.getelementptr %13[%0] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %15 = llvm.load %14 {alignment = 4 : i64} : !llvm.ptr -> vector<32xf32>
    %16 = arith.addf %10, %15 fastmath<fast> : vector<32xf32>
    %17 = llvm.extractvalue %5[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %18 = llvm.extractvalue %5[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %19 = llvm.getelementptr %17[%18] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %20 = llvm.getelementptr %19[%0] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %16, %20 {alignment = 4 : i64} : vector<32xf32>, !llvm.ptr
    %21 = arith.addi %1, %c32 : index
    cf.br ^bb1(%21 : index)
  ^bb3:  // pred: ^bb1
    return
  }
}


// -----
// Test with tiling on:
//        enableMultiThreading=False,
//        enableVTCMTiling=True,
//        enableConvertToHexagonmem=False,
//        enableHexagonmemCopyToDMA=False,
//        enableDoubleBuffering=False,
// This test ensures that memref allocs are supported and
// intentionally allocated in an aligned way

// CHECK-LABEL: @add_kernel
// CHECK: memref.reinterpret_cast {{.*}} {alignment = 128 : i64}
// CHECK: memref.reinterpret_cast {{.*}} {alignment = 128 : i64}
// CHECK: memref.reinterpret_cast {{.*}} {alignment = 128 : i64}
// CHECK: memref.reinterpret_cast {{.*}} {alignment = 128 : i64}
// CHECK: memref.reinterpret_cast {{.*}} {alignment = 128 : i64}
// CHECK: memref.reinterpret_cast {{.*}} {alignment = 128 : i64}
// CHECK: llvm.load {{.*}} {alignment = 128 : i64}
// CHECK: llvm.load {{.*}} {alignment = 128 : i64}
// CHECK: llvm.store {{.*}} {alignment = 128 : i64}

module attributes {llvm.target_triple = "hexagon"} {
  func.func @add_kernel(%arg0: memref<*xf32> {tt.divisibility = 128 : i32}, %arg1: memref<*xf32> {tt.divisibility = 128 : i32}, %arg2: memref<*xf32> {tt.divisibility = 128 : i32}, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32) {
    %c131072 = arith.constant 131072 : index
    %c0 = arith.constant 0 : index
    %0 = builtin.unrealized_conversion_cast %c0 : index to i64
    %c32 = arith.constant 32 : index
    %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [0], sizes: [131072], strides: [1] : memref<*xf32> to memref<131072xf32, strided<[1]>>
    %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [0], sizes: [131072], strides: [1] : memref<*xf32> to memref<131072xf32, strided<[1]>>
    %alloc = memref.alloc() {alignment = 128 : i64} : memref<131072xf32, 1>
    memref.copy %reinterpret_cast_0, %alloc : memref<131072xf32, strided<[1]>> to memref<131072xf32, 1>
    %alloc_1 = memref.alloc() {alignment = 128 : i64} : memref<131072xf32>
    %alloc_2 = memref.alloc() {alignment = 128 : i64} : memref<131072xf32, 1>
    memref.copy %alloc_1, %alloc_2 : memref<131072xf32> to memref<131072xf32, 1>
    %reinterpret_cast_3 = memref.reinterpret_cast %arg1 to offset: [0], sizes: [131072], strides: [1] : memref<*xf32> to memref<131072xf32, strided<[1]>>
    %alloc_4 = memref.alloc() {alignment = 128 : i64} : memref<131072xf32, 1>
    memref.copy %reinterpret_cast_3, %alloc_4 : memref<131072xf32, strided<[1]>> to memref<131072xf32, 1>
    cf.br ^bb1(%c0 : index)
  ^bb1(%1: index):  // 2 preds: ^bb0, ^bb2
    %2 = arith.cmpi slt, %1, %c131072 : index
    cf.cond_br %2, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    %reinterpret_cast_5 = memref.reinterpret_cast %alloc to offset: [%1], sizes: [32], strides: [1] : memref<131072xf32, 1> to memref<32xf32, strided<[1], offset: ?>, 1>
    %3 = builtin.unrealized_conversion_cast %reinterpret_cast_5 : memref<32xf32, strided<[1], offset: ?>, 1> to !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
    %reinterpret_cast_6 = memref.reinterpret_cast %alloc_4 to offset: [%1], sizes: [32], strides: [1] : memref<131072xf32, 1> to memref<32xf32, strided<[1], offset: ?>, 1>
    %4 = builtin.unrealized_conversion_cast %reinterpret_cast_6 : memref<32xf32, strided<[1], offset: ?>, 1> to !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
    %reinterpret_cast_7 = memref.reinterpret_cast %alloc_2 to offset: [%1], sizes: [32], strides: [1] : memref<131072xf32, 1> to memref<32xf32, strided<[1], offset: ?>, 1>
    %5 = builtin.unrealized_conversion_cast %reinterpret_cast_7 : memref<32xf32, strided<[1], offset: ?>, 1> to !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
    %6 = llvm.extractvalue %3[1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)> 
    %7 = llvm.extractvalue %3[2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)> 
    %8 = llvm.getelementptr %6[%7] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
    %9 = llvm.getelementptr %8[%0] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
    %10 = llvm.load %9 {alignment = 4 : i64} : !llvm.ptr<1> -> vector<32xf32>
    %11 = llvm.extractvalue %4[1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)> 
    %12 = llvm.extractvalue %4[2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)> 
    %13 = llvm.getelementptr %11[%12] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
    %14 = llvm.getelementptr %13[%0] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
    %15 = llvm.load %14 {alignment = 4 : i64} : !llvm.ptr<1> -> vector<32xf32>
    %16 = arith.addf %10, %15 fastmath<fast> : vector<32xf32>
    %17 = llvm.extractvalue %5[1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)> 
    %18 = llvm.extractvalue %5[2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)> 
    %19 = llvm.getelementptr %17[%18] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
    %20 = llvm.getelementptr %19[%0] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
    llvm.store %16, %20 {alignment = 4 : i64} : vector<32xf32>, !llvm.ptr<1>
    %21 = arith.addi %1, %c32 : index
    cf.br ^bb1(%21 : index)
  ^bb3:  // pred: ^bb1
    memref.copy %alloc_2, %reinterpret_cast : memref<131072xf32, 1> to memref<131072xf32, strided<[1]>>
    memref.dealloc %alloc : memref<131072xf32, 1>
    memref.dealloc %alloc_1 : memref<131072xf32>
    memref.dealloc %alloc_2 : memref<131072xf32, 1>
    memref.dealloc %alloc_4 : memref<131072xf32, 1>
    return
  }
}

// -----
// Test with the following optimizations on:
//        enableMultiThreading=False,
//        enableVTCMTiling=True,
//        enableConvertToHexagonmem=True,
//        enableHexagonmemCopyToDMA=False,
//        enableDoubleBuffering=False,
// This test ensures that hexagonmem allocs are supported

// CHECK-LABEL: @add_kernel
// CHECK: memref.reinterpret_cast {{.*}} {alignment = 128 : i64}
// CHECK: memref.reinterpret_cast {{.*}} {alignment = 128 : i64}
// CHECK: memref.reinterpret_cast {{.*}} {alignment = 128 : i64}
// CHECK: memref.reinterpret_cast {{.*}} {alignment = 128 : i64}
// CHECK: memref.reinterpret_cast {{.*}} {alignment = 128 : i64}
// CHECK: memref.reinterpret_cast {{.*}} {alignment = 128 : i64}
// CHECK: llvm.load {{.*}} {alignment = 128 : i64}
// CHECK: llvm.load {{.*}} {alignment = 128 : i64}
// CHECK: llvm.store {{.*}} {alignment = 128 : i64}

module attributes {llvm.target_triple = "hexagon"} {
  func.func @add_kernel(%arg0: memref<*xf32> {tt.divisibility = 128 : i32}, %arg1: memref<*xf32> {tt.divisibility = 128 : i32}, %arg2: memref<*xf32> {tt.divisibility = 128 : i32}, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32) {
    %c131072 = arith.constant 131072 : index
    %c0 = arith.constant 0 : index
    %0 = builtin.unrealized_conversion_cast %c0 : index to i64
    %c32 = arith.constant 32 : index
    %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [0], sizes: [131072], strides: [1] : memref<*xf32> to memref<131072xf32, strided<[1]>>
    %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [0], sizes: [131072], strides: [1] : memref<*xf32> to memref<131072xf32, strided<[1]>>
    %1 = hexagonmem.alloc() : memref<131072xf32, 1>
    hexagonmem.copy %reinterpret_cast_0, %1 : memref<131072xf32, strided<[1]>> to memref<131072xf32, 1>
    %alloc = memref.alloc() {alignment = 128 : i64} : memref<131072xf32>
    %2 = hexagonmem.alloc() : memref<131072xf32, 1>
    hexagonmem.copy %alloc, %2 : memref<131072xf32> to memref<131072xf32, 1>
    %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [0], sizes: [131072], strides: [1] : memref<*xf32> to memref<131072xf32, strided<[1]>>
    %3 = hexagonmem.alloc() : memref<131072xf32, 1>
    hexagonmem.copy %reinterpret_cast_1, %3 : memref<131072xf32, strided<[1]>> to memref<131072xf32, 1>
    cf.br ^bb1(%c0 : index)
  ^bb1(%4: index):  // 2 preds: ^bb0, ^bb2
    %5 = arith.cmpi slt, %4, %c131072 : index
    cf.cond_br %5, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    %base_buffer, %offset, %sizes, %strides = memref.extract_strided_metadata %1 : memref<131072xf32, 1> -> memref<f32, 1>, index, index, index
    %reinterpret_cast_2 = memref.reinterpret_cast %base_buffer to offset: [%4], sizes: [32], strides: [1] : memref<f32, 1> to memref<32xf32, strided<[1], offset: ?>, 1>
    %6 = builtin.unrealized_conversion_cast %reinterpret_cast_2 : memref<32xf32, strided<[1], offset: ?>, 1> to !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
    %base_buffer_3, %offset_4, %sizes_5, %strides_6 = memref.extract_strided_metadata %3 : memref<131072xf32, 1> -> memref<f32, 1>, index, index, index
    %reinterpret_cast_7 = memref.reinterpret_cast %base_buffer_3 to offset: [%4], sizes: [32], strides: [1] : memref<f32, 1> to memref<32xf32, strided<[1], offset: ?>, 1>
    %7 = builtin.unrealized_conversion_cast %reinterpret_cast_7 : memref<32xf32, strided<[1], offset: ?>, 1> to !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
    %base_buffer_8, %offset_9, %sizes_10, %strides_11 = memref.extract_strided_metadata %2 : memref<131072xf32, 1> -> memref<f32, 1>, index, index, index
    %reinterpret_cast_12 = memref.reinterpret_cast %base_buffer_8 to offset: [%4], sizes: [32], strides: [1] : memref<f32, 1> to memref<32xf32, strided<[1], offset: ?>, 1>
    %8 = builtin.unrealized_conversion_cast %reinterpret_cast_12 : memref<32xf32, strided<[1], offset: ?>, 1> to !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
    %9 = llvm.extractvalue %6[1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)> 
    %10 = llvm.extractvalue %6[2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)> 
    %11 = llvm.getelementptr %9[%10] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
    %12 = llvm.getelementptr %11[%0] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
    %13 = llvm.load %12 {alignment = 4 : i64} : !llvm.ptr<1> -> vector<32xf32>
    %14 = llvm.extractvalue %7[1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)> 
    %15 = llvm.extractvalue %7[2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)> 
    %16 = llvm.getelementptr %14[%15] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
    %17 = llvm.getelementptr %16[%0] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
    %18 = llvm.load %17 {alignment = 4 : i64} : !llvm.ptr<1> -> vector<32xf32>
    %19 = arith.addf %13, %18 fastmath<fast> : vector<32xf32>
    %20 = llvm.extractvalue %8[1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)> 
    %21 = llvm.extractvalue %8[2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)> 
    %22 = llvm.getelementptr %20[%21] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
    %23 = llvm.getelementptr %22[%0] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
    llvm.store %19, %23 {alignment = 4 : i64} : vector<32xf32>, !llvm.ptr<1>
    %24 = arith.addi %4, %c32 : index
    cf.br ^bb1(%24 : index)
  ^bb3:  // pred: ^bb1
    hexagonmem.copy %2, %reinterpret_cast : memref<131072xf32, 1> to memref<131072xf32, strided<[1]>>
    hexagonmem.dealloc %1 : memref<131072xf32, 1>
    memref.dealloc %alloc : memref<131072xf32>
    hexagonmem.dealloc %2 : memref<131072xf32, 1>
    hexagonmem.dealloc %3 : memref<131072xf32, 1>
    return
  }
}

// -----
// Test with the following optimizations on:
//        enableMultiThreading=False,
//        enableVTCMTiling=True,
//        enableConvertToHexagonmem=True,
//        enableHexagonmemCopyToDMA=True,
//        enableDoubleBuffering=False,
// DMA introduces slightly more complex indexing so this test
// ensures we propagate alignment even in that case

// CHECK-LABEL: @add_kernel
// CHECK: memref.reinterpret_cast {{.*}} {alignment = 128 : i64}
// CHECK: memref.reinterpret_cast {{.*}} {alignment = 128 : i64}
// CHECK: memref.reinterpret_cast {{.*}} {alignment = 128 : i64}
// CHECK: memref.reinterpret_cast {{.*}} {alignment = 128 : i64}
// CHECK: memref.reinterpret_cast {{.*}} {alignment = 128 : i64}
// CHECK: memref.reinterpret_cast {{.*}} {alignment = 128 : i64}
// CHECK: llvm.load {{.*}} {alignment = 128 : i64}
// CHECK: llvm.load {{.*}} {alignment = 128 : i64}
// CHECK: llvm.store {{.*}} {alignment = 128 : i64}

module attributes {llvm.target_triple = "hexagon"} {
  func.func @add_kernel(%arg0: memref<*xf32> {tt.divisibility = 128 : i32}, %arg1: memref<*xf32> {tt.divisibility = 128 : i32}, %arg2: memref<*xf32> {tt.divisibility = 128 : i32}, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32) {
    %c131072 = arith.constant 131072 : index
    %c0 = arith.constant 0 : index
    %0 = builtin.unrealized_conversion_cast %c0 : index to i64
    %c32 = arith.constant 32 : index
    %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [0], sizes: [131072], strides: [1] : memref<*xf32> to memref<131072xf32, strided<[1]>>
    %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [0], sizes: [131072], strides: [1] : memref<*xf32> to memref<131072xf32, strided<[1]>>
    %1 = hexagonmem.alloc() : memref<131072xf32, 1>
    %alloc = memref.alloc() : memref<1xi32>
    memref.dma_start %reinterpret_cast_0[%c0], %1[%c0], %c131072, %alloc[%c0] : memref<131072xf32, strided<[1]>>, memref<131072xf32, 1>, memref<1xi32>
    memref.dma_wait %alloc[%c0], %c131072 : memref<1xi32>
    memref.dealloc %alloc : memref<1xi32>
    %alloc_1 = memref.alloc() {alignment = 128 : i64} : memref<131072xf32>
    %2 = hexagonmem.alloc() : memref<131072xf32, 1>
    %alloc_2 = memref.alloc() : memref<1xi32>
    memref.dma_start %alloc_1[%c0], %2[%c0], %c131072, %alloc_2[%c0] : memref<131072xf32>, memref<131072xf32, 1>, memref<1xi32>
    memref.dma_wait %alloc_2[%c0], %c131072 : memref<1xi32>
    memref.dealloc %alloc_2 : memref<1xi32>
    %reinterpret_cast_3 = memref.reinterpret_cast %arg1 to offset: [0], sizes: [131072], strides: [1] : memref<*xf32> to memref<131072xf32, strided<[1]>>
    %3 = hexagonmem.alloc() : memref<131072xf32, 1>
    %alloc_4 = memref.alloc() : memref<1xi32>
    memref.dma_start %reinterpret_cast_3[%c0], %3[%c0], %c131072, %alloc_4[%c0] : memref<131072xf32, strided<[1]>>, memref<131072xf32, 1>, memref<1xi32>
    memref.dma_wait %alloc_4[%c0], %c131072 : memref<1xi32>
    memref.dealloc %alloc_4 : memref<1xi32>
    cf.br ^bb1(%c0 : index)
  ^bb1(%4: index):  // 2 preds: ^bb0, ^bb2
    %5 = arith.cmpi slt, %4, %c131072 : index
    cf.cond_br %5, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    %base_buffer, %offset, %sizes, %strides = memref.extract_strided_metadata %1 : memref<131072xf32, 1> -> memref<f32, 1>, index, index, index
    %reinterpret_cast_5 = memref.reinterpret_cast %base_buffer to offset: [%4], sizes: [32], strides: [1] : memref<f32, 1> to memref<32xf32, strided<[1], offset: ?>, 1>
    %6 = builtin.unrealized_conversion_cast %reinterpret_cast_5 : memref<32xf32, strided<[1], offset: ?>, 1> to !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
    %base_buffer_6, %offset_7, %sizes_8, %strides_9 = memref.extract_strided_metadata %3 : memref<131072xf32, 1> -> memref<f32, 1>, index, index, index
    %reinterpret_cast_10 = memref.reinterpret_cast %base_buffer_6 to offset: [%4], sizes: [32], strides: [1] : memref<f32, 1> to memref<32xf32, strided<[1], offset: ?>, 1>
    %7 = builtin.unrealized_conversion_cast %reinterpret_cast_10 : memref<32xf32, strided<[1], offset: ?>, 1> to !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
    %base_buffer_11, %offset_12, %sizes_13, %strides_14 = memref.extract_strided_metadata %2 : memref<131072xf32, 1> -> memref<f32, 1>, index, index, index
    %reinterpret_cast_15 = memref.reinterpret_cast %base_buffer_11 to offset: [%4], sizes: [32], strides: [1] : memref<f32, 1> to memref<32xf32, strided<[1], offset: ?>, 1>
    %8 = builtin.unrealized_conversion_cast %reinterpret_cast_15 : memref<32xf32, strided<[1], offset: ?>, 1> to !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
    %9 = llvm.extractvalue %6[1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)> 
    %10 = llvm.extractvalue %6[2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)> 
    %11 = llvm.getelementptr %9[%10] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
    %12 = llvm.getelementptr %11[%0] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
    %13 = llvm.load %12 {alignment = 4 : i64} : !llvm.ptr<1> -> vector<32xf32>
    %14 = llvm.extractvalue %7[1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)> 
    %15 = llvm.extractvalue %7[2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)> 
    %16 = llvm.getelementptr %14[%15] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
    %17 = llvm.getelementptr %16[%0] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
    %18 = llvm.load %17 {alignment = 4 : i64} : !llvm.ptr<1> -> vector<32xf32>
    %19 = arith.addf %13, %18 fastmath<fast> : vector<32xf32>
    %20 = llvm.extractvalue %8[1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)> 
    %21 = llvm.extractvalue %8[2] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)> 
    %22 = llvm.getelementptr %20[%21] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
    %23 = llvm.getelementptr %22[%0] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
    llvm.store %19, %23 {alignment = 4 : i64} : vector<32xf32>, !llvm.ptr<1>
    %24 = arith.addi %4, %c32 : index
    cf.br ^bb1(%24 : index)
  ^bb3:  // pred: ^bb1
    %alloc_16 = memref.alloc() : memref<1xi32>
    memref.dma_start %2[%c0], %reinterpret_cast[%c0], %c131072, %alloc_16[%c0] : memref<131072xf32, 1>, memref<131072xf32, strided<[1]>>, memref<1xi32>
    memref.dma_wait %alloc_16[%c0], %c131072 : memref<1xi32>
    memref.dealloc %alloc_16 : memref<1xi32>
    hexagonmem.dealloc %1 : memref<131072xf32, 1>
    memref.dealloc %alloc_1 : memref<131072xf32>
    hexagonmem.dealloc %2 : memref<131072xf32, 1>
    hexagonmem.dealloc %3 : memref<131072xf32, 1>
    return
  }
}

// -----
// Test E1: hexagonmem.alloc with explicit alignment=256 (above the 128 default).
// Verifies the explicit attribute overrides the implicit default — the
// reinterpret_cast and load/store must reflect 256, not 128.
// arg0 divisibility=64 → offsetBytes = 64*4 = 256 → GCD(256, 256) = 256.
func.func @hexagonmem_alloc_explicit_256(%arg0: index {tt.divisibility = 64 : i32}) {
  %alloc = hexagonmem.alloc() {alignment = 256 : i64} : memref<1024xf32, 1>
  %base_buffer, %offset, %sizes, %strides = memref.extract_strided_metadata %alloc : memref<1024xf32, 1> -> memref<f32, 1>, index, index, index
  %reinterpret_cast = memref.reinterpret_cast %base_buffer to offset: [%arg0], sizes: [32], strides: [1] : memref<f32, 1> to memref<32xf32, strided<[1], offset: ?>, 1>
  %cast = builtin.unrealized_conversion_cast %reinterpret_cast : memref<32xf32, strided<[1], offset: ?>, 1> to !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
  %ptr = llvm.extractvalue %cast[1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
  %c0 = arith.constant 0 : index
  %c0_i64 = builtin.unrealized_conversion_cast %c0 : index to i64
  %gep = llvm.getelementptr %ptr[%c0_i64] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
  %val = llvm.load %gep : !llvm.ptr<1> -> f32
  llvm.store %val, %gep : f32, !llvm.ptr<1>
  return
}

// CHECK-LABEL: @hexagonmem_alloc_explicit_256
// CHECK: memref.reinterpret_cast {{.*}} {alignment = 256 : i64}
// CHECK: llvm.load {{.*}} {alignment = 256 : i64}
// CHECK: llvm.store {{.*}} {alignment = 256 : i64}

// -----
// Test E2: hexagonmem.alloc with explicit alignment=64 (below the 128 default).
// Verifies the pass does not silently clamp up to 128.
// arg0 divisibility=32 → offsetBytes = 32*4 = 128 → GCD(64, 128) = 64.
func.func @hexagonmem_alloc_explicit_64(%arg0: index {tt.divisibility = 32 : i32}) {
  %alloc = hexagonmem.alloc() {alignment = 64 : i64} : memref<1024xf32, 1>
  %base_buffer, %offset, %sizes, %strides = memref.extract_strided_metadata %alloc : memref<1024xf32, 1> -> memref<f32, 1>, index, index, index
  %reinterpret_cast = memref.reinterpret_cast %base_buffer to offset: [%arg0], sizes: [32], strides: [1] : memref<f32, 1> to memref<32xf32, strided<[1], offset: ?>, 1>
  %cast = builtin.unrealized_conversion_cast %reinterpret_cast : memref<32xf32, strided<[1], offset: ?>, 1> to !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
  %ptr = llvm.extractvalue %cast[1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
  %c0 = arith.constant 0 : index
  %c0_i64 = builtin.unrealized_conversion_cast %c0 : index to i64
  %gep = llvm.getelementptr %ptr[%c0_i64] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
  %val = llvm.load %gep : !llvm.ptr<1> -> f32
  llvm.store %val, %gep : f32, !llvm.ptr<1>
  return
}

// CHECK-LABEL: @hexagonmem_alloc_explicit_64
// CHECK: memref.reinterpret_cast {{.*}} {alignment = 64 : i64}
// CHECK: llvm.load {{.*}} {alignment = 64 : i64}
// CHECK: llvm.store {{.*}} {alignment = 64 : i64}

// -----
// Tests E3–E6: functions with explicit hexagonmem.alloc alignment but no
// naturally-aligned kernel arguments. A dummy `%dummy: index` argument marked
// `tt.divisibility = 128` is required to identify the function as a top-level
// kernel — `tt.divisibility` on at least one argument is the convention that
// tells the pass a function is an entry point worth processing.

// Test E3: hexagonmem.alloc with explicit alignment=1 (completely unaligned).
// GCD(1, anything) = 1 — the unaligned seed poisons all downstream alignments
// to 1, distinct from "no alignment" which would fall back to element size (4).
func.func @hexagonmem_alloc_explicit_1(%dummy: index {tt.divisibility = 128 : i32}) {
  %alloc = hexagonmem.alloc() {alignment = 1 : i64} : memref<1024xf32, 1>
  %base_buffer, %offset, %sizes, %strides = memref.extract_strided_metadata %alloc : memref<1024xf32, 1> -> memref<f32, 1>, index, index, index
  %reinterpret_cast = memref.reinterpret_cast %base_buffer to offset: [0], sizes: [32], strides: [1] : memref<f32, 1> to memref<32xf32, strided<[1], offset: 0>, 1>
  %cast = builtin.unrealized_conversion_cast %reinterpret_cast : memref<32xf32, strided<[1], offset: 0>, 1> to !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
  %ptr = llvm.extractvalue %cast[1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
  %c0 = arith.constant 0 : index
  %c0_i64 = builtin.unrealized_conversion_cast %c0 : index to i64
  %gep = llvm.getelementptr %ptr[%c0_i64] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
  %val = llvm.load %gep : !llvm.ptr<1> -> f32
  llvm.store %val, %gep : f32, !llvm.ptr<1>
  return
}

// CHECK-LABEL: @hexagonmem_alloc_explicit_1
// CHECK: memref.reinterpret_cast {{.*}} {alignment = 1 : i64}
// CHECK: llvm.load {{.*}} {alignment = 1 : i64}
// CHECK: llvm.store {{.*}} {alignment = 1 : i64}

// -----
// Test E4: two hexagonmem.alloc ops with distinct explicit alignments in the same function.
// Verifies per-value divisibility tracking rather than a function-wide constant.
// First alloc has 128, second has 256 — both use static zero offset to isolate.
func.func @hexagonmem_alloc_two_different_explicit_alignments(%dummy: index {tt.divisibility = 128 : i32}) {
  %alloc_128 = hexagonmem.alloc() {alignment = 128 : i64} : memref<512xf32, 1>
  %base_128, %off_128, %sz_128, %st_128 = memref.extract_strided_metadata %alloc_128 : memref<512xf32, 1> -> memref<f32, 1>, index, index, index
  %rc_128 = memref.reinterpret_cast %base_128 to offset: [0], sizes: [32], strides: [1] : memref<f32, 1> to memref<32xf32, strided<[1], offset: 0>, 1>
  %cast_128 = builtin.unrealized_conversion_cast %rc_128 : memref<32xf32, strided<[1], offset: 0>, 1> to !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
  %ptr_128 = llvm.extractvalue %cast_128[1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
  %c0 = arith.constant 0 : index
  %c0_i64 = builtin.unrealized_conversion_cast %c0 : index to i64
  %gep_128 = llvm.getelementptr %ptr_128[%c0_i64] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
  %val_128 = llvm.load %gep_128 : !llvm.ptr<1> -> f32

  %alloc_256 = hexagonmem.alloc() {alignment = 256 : i64} : memref<512xf32, 1>
  %base_256, %off_256, %sz_256, %st_256 = memref.extract_strided_metadata %alloc_256 : memref<512xf32, 1> -> memref<f32, 1>, index, index, index
  %rc_256 = memref.reinterpret_cast %base_256 to offset: [0], sizes: [32], strides: [1] : memref<f32, 1> to memref<32xf32, strided<[1], offset: 0>, 1>
  %cast_256 = builtin.unrealized_conversion_cast %rc_256 : memref<32xf32, strided<[1], offset: 0>, 1> to !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
  %ptr_256 = llvm.extractvalue %cast_256[1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
  %gep_256 = llvm.getelementptr %ptr_256[%c0_i64] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
  %val_256 = llvm.load %gep_256 : !llvm.ptr<1> -> f32
  return
}

// CHECK-LABEL: @hexagonmem_alloc_two_different_explicit_alignments
// CHECK: memref.reinterpret_cast {{.*}} {alignment = 128 : i64}
// CHECK: llvm.load {{.*}} {alignment = 128 : i64}
// CHECK: memref.reinterpret_cast {{.*}} {alignment = 256 : i64}
// CHECK: llvm.load {{.*}} {alignment = 256 : i64}

// -----
// Test E5a: explicit 256-byte alloc + static aligned offset (64 elements = 256 bytes for f32).
// GCD(256, 256) = 256 — alignment survives the offset.
func.func @hexagonmem_alloc_256_aligned_offset(%dummy: index {tt.divisibility = 128 : i32}) {
  %alloc = hexagonmem.alloc() {alignment = 256 : i64} : memref<2048xf32, 1>
  %base_buffer, %offset, %sizes, %strides = memref.extract_strided_metadata %alloc : memref<2048xf32, 1> -> memref<f32, 1>, index, index, index
  %reinterpret_cast = memref.reinterpret_cast %base_buffer to offset: [64], sizes: [64], strides: [1] : memref<f32, 1> to memref<64xf32, strided<[1], offset: 64>, 1>
  %cast = builtin.unrealized_conversion_cast %reinterpret_cast : memref<64xf32, strided<[1], offset: 64>, 1> to !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
  %ptr = llvm.extractvalue %cast[1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
  %c0 = arith.constant 0 : index
  %c0_i64 = builtin.unrealized_conversion_cast %c0 : index to i64
  %gep = llvm.getelementptr %ptr[%c0_i64] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
  %val = llvm.load %gep : !llvm.ptr<1> -> f32
  llvm.store %val, %gep : f32, !llvm.ptr<1>
  return
}

// CHECK-LABEL: @hexagonmem_alloc_256_aligned_offset
// CHECK: memref.reinterpret_cast {{.*}} {alignment = 256 : i64}
// CHECK: llvm.load {{.*}} {alignment = 256 : i64}
// CHECK: llvm.store {{.*}} {alignment = 256 : i64}

// -----
// Test E5b: explicit 256-byte alloc + static unaligned offset (32 elements = 128 bytes for f32).
// GCD(256, 128) = 128 — the offset reduces the alignment.
func.func @hexagonmem_alloc_256_unaligned_offset(%dummy: index {tt.divisibility = 128 : i32}) {
  %alloc = hexagonmem.alloc() {alignment = 256 : i64} : memref<2048xf32, 1>
  %base_buffer, %offset, %sizes, %strides = memref.extract_strided_metadata %alloc : memref<2048xf32, 1> -> memref<f32, 1>, index, index, index
  %reinterpret_cast = memref.reinterpret_cast %base_buffer to offset: [32], sizes: [64], strides: [1] : memref<f32, 1> to memref<64xf32, strided<[1], offset: 32>, 1>
  %cast = builtin.unrealized_conversion_cast %reinterpret_cast : memref<64xf32, strided<[1], offset: 32>, 1> to !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
  %ptr = llvm.extractvalue %cast[1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
  %c0 = arith.constant 0 : index
  %c0_i64 = builtin.unrealized_conversion_cast %c0 : index to i64
  %gep = llvm.getelementptr %ptr[%c0_i64] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
  %val = llvm.load %gep : !llvm.ptr<1> -> f32
  llvm.store %val, %gep : f32, !llvm.ptr<1>
  return
}

// CHECK-LABEL: @hexagonmem_alloc_256_unaligned_offset
// CHECK: memref.reinterpret_cast {{.*}} {alignment = 128 : i64}
// CHECK: llvm.load {{.*}} {alignment = 128 : i64}
// CHECK: llvm.store {{.*}} {alignment = 128 : i64}

// -----
// Test E6: dynamic-size hexagonmem.alloc with explicit alignment=256.
// The dynamic shape does not affect alignment seeding — the explicit attribute
// must still propagate to the downstream reinterpret_cast and load.
func.func @hexagonmem_alloc_dynamic_explicit_alignment(%n: index {tt.divisibility = 128 : i32}) {
  %alloc = hexagonmem.alloc(%n) {alignment = 256 : i64} : memref<?xf32, 1>
  %base_buffer, %offset, %sizes, %strides = memref.extract_strided_metadata %alloc : memref<?xf32, 1> -> memref<f32, 1>, index, index, index
  %reinterpret_cast = memref.reinterpret_cast %base_buffer to offset: [0], sizes: [256], strides: [1] : memref<f32, 1> to memref<256xf32, strided<[1], offset: 0>, 1>
  %cast = builtin.unrealized_conversion_cast %reinterpret_cast : memref<256xf32, strided<[1], offset: 0>, 1> to !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
  %ptr = llvm.extractvalue %cast[1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
  %c0 = arith.constant 0 : index
  %c0_i64 = builtin.unrealized_conversion_cast %c0 : index to i64
  %gep = llvm.getelementptr %ptr[%c0_i64] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
  %val = llvm.load %gep : !llvm.ptr<1> -> f32
  llvm.store %val, %gep : f32, !llvm.ptr<1>
  return
}

// CHECK-LABEL: @hexagonmem_alloc_dynamic_explicit_alignment
// CHECK: memref.reinterpret_cast {{.*}} {alignment = 256 : i64}
// CHECK: llvm.load {{.*}} {alignment = 256 : i64}
// CHECK: llvm.store {{.*}} {alignment = 256 : i64}
// -----
// Regression (QTOOL: sub-HVX alignment bug): integer (i32) loop induction
// variable driving a per-row sub-HVX access.
//
// This mirrors the vectorized rmsnorm/layernorm C=32 kernels. After the
// arith->LLVM lowering the row counter is a plain `i32` block argument (not
// `index`). Each row is `%iv * 32` f16 elements = 64 bytes apart, so on a
// 128-byte-aligned base the correct alignment is GCD(128, 64) = 64.
//
// The bug: `initializeBlockArgSeeds` only seeded `index`-typed block args, so
// the `i32` IV was untracked; its `muli %iv, 32` offset got divisibility 0
// ("unknown"), and GCD(128, 0) == 128 wrongly PRESERVED the full 128-byte
// alignment. The backend then emitted an aligned `vmem` off a 64-byte-aligned
// address, corrupting odd rows. The fix seeds integer IVs too, so the offset
// resolves to 64 bytes and the load/store correctly become `align 64`.
func.func @i32_iv_per_row_subhvx(%arg0: memref<*xf16> {tt.divisibility = 128 : i32}) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %c32_i32 = arith.constant 32 : i32
  cf.br ^bb1(%c0_i32 : i32)
^bb1(%iv: i32):  // i32 induction variable (0..32 step 1)
  %cond = arith.cmpi slt, %iv, %c32_i32 : i32
  cf.cond_br %cond, ^bb2, ^bb3
^bb2:
  // Row offset in elements = iv * 32  (= iv * 64 bytes for f16).
  %off_i32 = arith.muli %iv, %c32_i32 : i32
  %off = arith.index_cast %off_i32 : i32 to index
  %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%off], sizes: [32], strides: [1] : memref<*xf16> to memref<32xf16, strided<[1], offset: ?>>
  %cast = builtin.unrealized_conversion_cast %reinterpret_cast : memref<32xf16, strided<[1], offset: ?>> to !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
  %ptr = llvm.extractvalue %cast[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
  %c0 = arith.constant 0 : index
  %c0_i64 = builtin.unrealized_conversion_cast %c0 : index to i64
  %gep = llvm.getelementptr %ptr[%c0_i64] : (!llvm.ptr, i64) -> !llvm.ptr, f16
  %val = llvm.load %gep : !llvm.ptr -> vector<32xf16>
  llvm.store %val, %gep : vector<32xf16>, !llvm.ptr
  %next = arith.addi %iv, %c1_i32 : i32
  cf.br ^bb1(%next : i32)
^bb3:
  return
}

// CHECK-LABEL: @i32_iv_per_row_subhvx
// CHECK: memref.reinterpret_cast {{.*}} {alignment = 64 : i64}
// CHECK: llvm.load {{.*}} {alignment = 64 : i64}
// CHECK: llvm.store {{.*}} {alignment = 64 : i64}
