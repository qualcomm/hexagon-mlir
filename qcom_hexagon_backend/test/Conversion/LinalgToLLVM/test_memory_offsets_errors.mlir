// RUN: linalg-hexagon-opt %s -memory-offsets -split-input-file -verify-diagnostics

// ===================================================================
// Case 1: Dynamic scratch arg — memref<?xi8, 1> is rejected because
// the pass requires a static size to validate at compile time.
// ===================================================================
// expected-error @+1 {{hexagon.scratch argument must have a static size}}
func.func @test_dynamic_scratch_rejected(
    %A: memref<32xf32>,
    %vtcm: memref<?xi8, 1> {hexagon.scratch}) {
  %0 = memref.alloc() : memref<32xf32, 1>
  memref.copy %A, %0 : memref<32xf32> to memref<32xf32, 1>
  memref.dealloc %0 : memref<32xf32, 1>
  return
}

// -----

// ===================================================================
// Case 2: Oversized allocs — kernel allocations exceed the scratch
// buffer size.
// ===================================================================
// expected-error @+1 {{Required VTCM size}}
func.func @test_scratch_overflow(
    %A: memref<32xf32>,
    %vtcm: memref<128xi8, 1> {hexagon.scratch}) {
  // Two 32xf32 allocs = 2 * 128 bytes = 256 bytes, aligned to 2048 each = 4096 bytes total.
  // Scratch is only 128 bytes — must fail.
  %0 = memref.alloc() : memref<32xf32, 1>
  %1 = memref.alloc() : memref<32xf32, 1>
  memref.copy %A, %0 : memref<32xf32> to memref<32xf32, 1>
  memref.dealloc %0 : memref<32xf32, 1>
  memref.dealloc %1 : memref<32xf32, 1>
  return
}

// -----

// ===================================================================
// Case 3: Zero-sized scratch arg — memref<0xi8, 1> must be rejected
// even though it has a static shape.
// ===================================================================
// expected-error @+1 {{hexagon.scratch argument must have a positive size}}
func.func @test_zero_sized_scratch(
    %A: memref<32xf32>,
    %vtcm: memref<0xi8, 1> {hexagon.scratch}) {
  %0 = memref.alloc() : memref<32xf32, 1>
  memref.copy %A, %0 : memref<32xf32> to memref<32xf32, 1>
  memref.dealloc %0 : memref<32xf32, 1>
  return
}

// -----

// ===================================================================
// Case 4: Fallback budget exceeded — no scratch arg, but allocs
// exceed the configured bufferSize budget.
// ===================================================================
// expected-error @+1 {{Required VTCM size}}
func.func @test_fallback_budget_exceeded(%A: memref<32xf32>) {
  // A single memref<262144xf32, 1> = 1MB exactly; two of them exceed the budget.
  %0 = memref.alloc() : memref<262144xf32, 1>
  %1 = memref.alloc() : memref<262144xf32, 1>
  memref.dealloc %0 : memref<262144xf32, 1>
  memref.dealloc %1 : memref<262144xf32, 1>
  return
}
