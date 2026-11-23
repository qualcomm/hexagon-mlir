//===- test_memory_offsets_errors.mlir ------------------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

// RUN: linalg-hexagon-opt %s -memory-offsets -split-input-file -verify-diagnostics

// Case 1: Invalid scratch type — warns and falls back to internal alloc.
// expected-warning @+1 {{Invalid hexagon.scratch arg type}}
func.func @test_dynamic_scratch_falls_back(
    %A: memref<32xf32>,
    %vtcm: memref<?xi8, 1> {hexagon.scratch}) {
  %0 = memref.alloc() : memref<32xf32, 1>
  memref.copy %A, %0 : memref<32xf32> to memref<32xf32, 1>
  memref.dealloc %0 : memref<32xf32, 1>
  return
}

// -----

// Case 2: Zero-sized scratch — warns and falls back.
// expected-warning @+1 {{Invalid hexagon.scratch arg type}}
func.func @test_zero_sized_scratch_falls_back(
    %A: memref<32xf32>,
    %vtcm: memref<0xi8, 1> {hexagon.scratch}) {
  %0 = memref.alloc() : memref<32xf32, 1>
  memref.copy %A, %0 : memref<32xf32> to memref<32xf32, 1>
  memref.dealloc %0 : memref<32xf32, 1>
  return
}

// -----

// Case 3: Peak pressure exceeds scratch budget — errors with per-alloc notes.
// expected-error @+1 {{VTCM peak 256 > budget 128}}
func.func @test_scratch_overflow(
    %A: memref<32xf32>,
    %vtcm: memref<128xi8, 1> {hexagon.scratch}) {
  // expected-note @+1 {{(128 B)}}
  %0 = memref.alloc() : memref<32xf32, 1>
  // expected-note @+1 {{(128 B)}}
  %1 = memref.alloc() : memref<32xf32, 1>
  memref.copy %A, %0 : memref<32xf32> to memref<32xf32, 1>
  memref.dealloc %0 : memref<32xf32, 1>
  memref.dealloc %1 : memref<32xf32, 1>
  return
}

// -----

// Case 4: Fallback budget exceeded — same error shape as external mode.
// expected-error @+1 {{VTCM peak 2097152 > budget 1048576}}
func.func @test_fallback_budget_exceeded(%A: memref<32xf32>) {
  // expected-note @+1 {{(1048576 B)}}
  %0 = memref.alloc() : memref<262144xf32, 1>
  // expected-note @+1 {{(1048576 B)}}
  %1 = memref.alloc() : memref<262144xf32, 1>
  memref.dealloc %0 : memref<262144xf32, 1>
  memref.dealloc %1 : memref<262144xf32, 1>
  return
}

// -----

// Case 5: VTCM alloc inside scf.forall — concurrent iterations would
// share offsets; pass errors out.
func.func @test_alloc_in_forall(%vtcm: memref<2048xi8, 1> {hexagon.scratch}) {
  scf.forall (%i) in (8) {
    // expected-error @+1 {{VTCM alloc in scf.forall not supported}}
    %0 = memref.alloc() : memref<128xf32, 1>
    memref.dealloc %0 : memref<128xf32, 1>
  }
  return
}
