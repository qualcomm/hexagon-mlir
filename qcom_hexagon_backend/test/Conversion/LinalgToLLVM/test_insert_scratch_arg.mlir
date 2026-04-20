// RUN: linalg-hexagon-opt -split-input-file %s \
// RUN:   -pass-pipeline='builtin.module(func.func(insert-scratch-arg{scratch=4096}))' \
// RUN: | FileCheck %s --check-prefix=INSERT
//
// RUN: linalg-hexagon-opt -split-input-file %s \
// RUN:   -pass-pipeline='builtin.module(func.func(insert-scratch-arg{scratch=4096},memory-offsets))' \
// RUN: | FileCheck %s --check-prefix=PIPELINE
//
// RUN: not linalg-hexagon-opt %s \
// RUN:   -pass-pipeline='builtin.module(func.func(insert-scratch-arg{scratch=100}))' \
// RUN:   2>&1 | FileCheck %s --check-prefix=ERR-ALIGN
//
// PIPELINE checks only @kernel_pipeline; other splits are not checked under
// this prefix.
//
// ERR-ALIGN: InsertScratchArg: scratch must be aligned to 2048 bytes, got 100

// INSERT-LABEL: func.func @kernel_no_scratch(
// INSERT-SAME:  {{.*}}: memref<32xf32>,
// INSERT-SAME:  {{.*}}: memref<4096xi8, 1> {hexagon.scratch},
// INSERT-SAME:  {{.*}}: i32, {{.*}}: i32, {{.*}}: i32, {{.*}}: i32, {{.*}}: i32, {{.*}}: i32)
// INSERT: memref.alloc() : memref<32xf32, 1>
// INSERT: memref.copy
// INSERT: memref.dealloc
func.func @kernel_no_scratch(
    %A: memref<32xf32>,
    %num_programs_x: i32, %num_programs_y: i32, %num_programs_z: i32,
    %pid_x: i32, %pid_y: i32, %pid_z: i32) {
  %0 = memref.alloc() : memref<32xf32, 1>
  memref.copy %A, %0 : memref<32xf32> to memref<32xf32, 1>
  memref.dealloc %0 : memref<32xf32, 1>
  return
}

// -----

// Idempotent: existing scratch argument is preserved and no second arg
// inserted. The existing scratch size (1024) differs from the pass option
// (4096) — idempotency wins, original size is preserved.
// INSERT-LABEL: func.func @kernel_already_has_scratch(
// INSERT-SAME:  {{.*}}: memref<32xf32>,
// INSERT-SAME:  {{.*}}: memref<1024xi8, 1> {hexagon.scratch},
// INSERT-SAME:  {{.*}}: i32, {{.*}}: i32, {{.*}}: i32, {{.*}}: i32, {{.*}}: i32, {{.*}}: i32)
func.func @kernel_already_has_scratch(
    %A: memref<32xf32>,
    %scratch: memref<1024xi8, 1> {hexagon.scratch},
    %num_programs_x: i32, %num_programs_y: i32, %num_programs_z: i32,
    %pid_x: i32, %pid_y: i32, %pid_z: i32) {
  return
}

// -----

// Non-kernel ABI (no trailing i32 pack): should be unchanged.
// INSERT-LABEL: func.func @non_kernel(
// INSERT-SAME:  %[[A:.*]]: memref<32xf32>)
func.func @non_kernel(%A: memref<32xf32>) {
  return
}

// -----

// Only 5 trailing i32s — not a valid Triton program-info ABI (needs 6),
// so the pass should leave this function unchanged.
// INSERT-LABEL: func.func @kernel_five_i32s(
// INSERT-SAME:  {{.*}}: memref<32xf32>,
// INSERT-SAME:  {{.*}}: i32, {{.*}}: i32, {{.*}}: i32, {{.*}}: i32, {{.*}}: i32)
func.func @kernel_five_i32s(
    %A: memref<32xf32>,
    %a: i32, %b: i32, %c: i32, %d: i32, %e: i32) {
  return
}

// -----

// Multiple memref args before the trailing i32 pack: scratch is inserted
// right before the first i32 in the trailing 6-i32 group.
// INSERT-LABEL: func.func @kernel_multi_memref(
// INSERT-SAME:  {{.*}}: memref<32xf32>,
// INSERT-SAME:  {{.*}}: memref<64xf32>,
// INSERT-SAME:  {{.*}}: memref<16xi32>,
// INSERT-SAME:  {{.*}}: memref<4096xi8, 1> {hexagon.scratch},
// INSERT-SAME:  {{.*}}: i32, {{.*}}: i32, {{.*}}: i32, {{.*}}: i32, {{.*}}: i32, {{.*}}: i32)
func.func @kernel_multi_memref(
    %A: memref<32xf32>,
    %B: memref<64xf32>,
    %C: memref<16xi32>,
    %num_programs_x: i32, %num_programs_y: i32, %num_programs_z: i32,
    %pid_x: i32, %pid_y: i32, %pid_z: i32) {
  return
}

// -----

// PIPELINE-LABEL: func.func @kernel_pipeline(
// PIPELINE-SAME:  {{.*}}: memref<32xf32>,
// PIPELINE-SAME:  {{.*}}: memref<4096xi8, 1> {hexagon.scratch},
// PIPELINE-SAME:  {{.*}}: i32, {{.*}}: i32, {{.*}}: i32, {{.*}}: i32, {{.*}}: i32, {{.*}}: i32)
// PIPELINE: %[[C0:.*]] = arith.constant 0 : index
// PIPELINE: %[[VIEW:.*]] = memref.view {{.*}}[%[[C0]]][] : memref<4096xi8, 1> to memref<32xf32, 1>
// PIPELINE-NOT: memref.alloc
// PIPELINE: return
func.func @kernel_pipeline(
    %A: memref<32xf32>,
    %num_programs_x: i32, %num_programs_y: i32, %num_programs_z: i32,
    %pid_x: i32, %pid_y: i32, %pid_z: i32) {
  %0 = memref.alloc() : memref<32xf32, 1>
  memref.copy %A, %0 : memref<32xf32> to memref<32xf32, 1>
  memref.dealloc %0 : memref<32xf32, 1>
  return
}
