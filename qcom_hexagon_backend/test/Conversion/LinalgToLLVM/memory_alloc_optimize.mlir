// RUN: linalg-hexagon-opt %s -copy-canonicalization -canonicalize | FileCheck %s

// Test copy canonicalization with chained memory copies
// Memory spaces: 0 = DDR, 1 = VTCM

// -----

// Test 1: Same memory space chain (DDR -> DDR -> DDR)
func.func @same_space_chain() -> memref<1024xi8> {
  %src = memref.alloc() {alignment = 64 : i64} : memref<1024xi8>
  %intermediate = memref.alloc() {alignment = 64 : i64} : memref<1024xi8>
  %dst = memref.alloc() {alignment = 64 : i64} : memref<1024xi8>

  memref.copy %src, %intermediate : memref<1024xi8> to memref<1024xi8>
  memref.copy %intermediate, %dst : memref<1024xi8> to memref<1024xi8>
  return %dst : memref<1024xi8>
}

// CHECK-LABEL: @same_space_chain
// CHECK: %[[SRC:.*]] = memref.alloc()
// CHECK: %[[DST:.*]] = memref.alloc()
// CHECK: memref.copy %[[SRC]], %[[DST]]
// CHECK: return %[[DST]]

// -----

// Test 2: Different memory space chain (DDR -> VTCM -> DDR)
func.func @different_space_chain() -> memref<512xf16> {
  %src = memref.alloc() {alignment = 64 : i64} : memref<512xf16>
  %intermediate = memref.alloc() {alignment = 64 : i64} : memref<512xf16, 1>
  %dst = memref.alloc() {alignment = 64 : i64} : memref<512xf16>

  memref.copy %src, %intermediate : memref<512xf16> to memref<512xf16, 1>
  memref.copy %intermediate, %dst : memref<512xf16, 1> to memref<512xf16>
  return %dst : memref<512xf16>
}

// CHECK-LABEL: @different_space_chain
// CHECK: %[[SRC:.*]] = memref.alloc() {{.*}} : memref<512xf16>
// CHECK-NOT: memref.alloc() {{.*}} : memref<512xf16, 1>
// CHECK: %[[DST:.*]] = memref.alloc() {{.*}} : memref<512xf16>
// CHECK: memref.copy %[[SRC]], %[[DST]]
// CHECK: return %[[DST]]

// -----

// Test 3: Multiple readers from intermediate buffer
func.func @multiple_readers() -> (memref<256xf32>, memref<256xf32>) {
  %src = memref.alloc() {alignment = 64 : i64} : memref<256xf32>
  %intermediate = memref.alloc() {alignment = 64 : i64} : memref<256xf32, 1>
  %dst1 = memref.alloc() {alignment = 64 : i64} : memref<256xf32>
  %dst2 = memref.alloc() {alignment = 64 : i64} : memref<256xf32>

  memref.copy %src, %intermediate : memref<256xf32> to memref<256xf32, 1>
  memref.copy %intermediate, %dst1 : memref<256xf32, 1> to memref<256xf32>
  memref.copy %intermediate, %dst2 : memref<256xf32, 1> to memref<256xf32>
  return %dst1, %dst2 : memref<256xf32>, memref<256xf32>
}

// CHECK-LABEL: @multiple_readers
// CHECK: %[[SRC:.*]] = memref.alloc() {{.*}} : memref<256xf32>
// CHECK-NOT: memref.alloc() {{.*}} : memref<256xf32, 1>
// CHECK: memref.copy %[[SRC]], %{{.*}}
// CHECK: memref.copy %[[SRC]], %{{.*}}

// -----

// Test 4: Non-copy use prevents optimization (negative test)
func.func @non_copy_use() -> memref<256xf32> {
  %src = memref.alloc() {alignment = 64 : i64} : memref<256xf32>
  %intermediate = memref.alloc() {alignment = 64 : i64} : memref<256xf32, 1>
  %dst = memref.alloc() {alignment = 64 : i64} : memref<256xf32>
  %cst = arith.constant 0.0 : f32

  memref.copy %src, %intermediate : memref<256xf32> to memref<256xf32, 1>
  linalg.fill ins(%cst : f32) outs(%intermediate : memref<256xf32, 1>)
  memref.copy %intermediate, %dst : memref<256xf32, 1> to memref<256xf32>
  return %dst : memref<256xf32>
}

// CHECK-LABEL: @non_copy_use
// CHECK: %[[SRC:.*]] = memref.alloc() {{.*}} : memref<256xf32>
// CHECK: %[[INTERMEDIATE:.*]] = memref.alloc() {{.*}} : memref<256xf32, 1>
// CHECK: %[[DST:.*]] = memref.alloc() {{.*}} : memref<256xf32>
// CHECK: memref.copy %[[SRC]], %[[INTERMEDIATE]]
// CHECK: linalg.fill ins({{.*}}) outs(%[[INTERMEDIATE]]
// CHECK: memref.copy %[[INTERMEDIATE]], %[[DST]]
// CHECK: return %[[DST]]

// -----

// Test 5: AddCopyBeforeYieldPattern with intermediate VTCM buffer and copy-back
func.func @add_copy_before_yield_complex_test(%arg0: memref<10xf32>) -> memref<10xf32> {
  %c0 = arith.constant 0 : index
  %c10 = arith.constant 10 : index
  %c1 = arith.constant 1 : index
  %cst_0 = arith.constant 0.0 : f32
  %cst_1 = arith.constant 1.0 : f32

  // Initialize the input buffer to something
  linalg.fill ins(%cst_0 : f32) outs(%arg0 : memref<10xf32>)

  // Loop-carried variable is %arg0
  %res_buffer = scf.for %i = %c0 to %c10 step %c1 iter_args(%loop_arg = %arg0) -> (memref<10xf32>) {
    // 1. Copy from %arg0 (DDR) to a VTCM buffer
    %vtcm_buffer = memref.alloc() {alignment = 64 : i64} : memref<10xf32, 1>
    memref.copy %loop_arg, %vtcm_buffer : memref<10xf32> to memref<10xf32, 1>

    // 2. Fill the VTCM buffer
    linalg.fill ins(%cst_1 : f32) outs(%vtcm_buffer : memref<10xf32, 1>)

    // 3. Copy VTCM buffer back to an intermediate DDR buffer
    %intermediate_ddr_buffer = memref.alloc() {alignment = 64 : i64} : memref<10xf32>
    memref.copy %vtcm_buffer, %intermediate_ddr_buffer : memref<10xf32, 1> to memref<10xf32>
    
    // 4. Yield the intermediate_ddr_buffer. This should trigger the AddCopyBeforeYieldPattern.
    scf.yield %intermediate_ddr_buffer : memref<10xf32>
  }

  return %res_buffer : memref<10xf32>
}

// CHECK-LABEL: @add_copy_before_yield_complex_test
// CHECK: %[[C0:.*]] = arith.constant 0 : index
// CHECK: %[[C10:.*]] = arith.constant 10 : index
// CHECK: %[[C1:.*]] = arith.constant 1 : index
// CHECK: %[[CST_0_VAL:.*]] = arith.constant 0.000000e+00 : f32
// CHECK: %[[CST_1_VAL:.*]] = arith.constant 1.000000e+00 : f32
// CHECK: linalg.fill ins(%[[CST_0_VAL]] : f32) outs(%arg0 : memref<10xf32>)
// CHECK: scf.for %[[I:.*]] = %[[C0]] to %[[C10]] step %[[C1]] {
// CHECK:   %[[VTCM_BUFFER:.*]] = memref.alloc() {alignment = 64 : i64} : memref<10xf32, 1>
// CHECK:   memref.copy %arg0, %[[VTCM_BUFFER]] : memref<10xf32> to memref<10xf32, 1>
// CHECK:   linalg.fill ins(%[[CST_1_VAL]] : f32) outs(%[[VTCM_BUFFER]] : memref<10xf32, 1>)
// CHECK:   memref.copy %[[VTCM_BUFFER]], %arg0 : memref<10xf32, 1> to memref<10xf32>
// CHECK: }
// CHECK: return %arg0 : memref<10xf32>
