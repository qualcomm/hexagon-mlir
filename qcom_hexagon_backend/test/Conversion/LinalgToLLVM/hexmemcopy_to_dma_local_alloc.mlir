//===- hexmemcopy_to_dma_local_alloc.mlir ---------------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(func.func(hexmem-cpy-to-dma))' | FileCheck %s

// A DDR<->VTCM memref.copy whose DDR endpoint is a function argument is a
// coherent buffer and is still lowered to a DMA transfer.
// CHECK-LABEL: func.func @arg_ddr_to_vtcm
// CHECK: memref.dma_start
// CHECK: memref.dma_wait
// CHECK-NOT: memref.copy
func.func @arg_ddr_to_vtcm(%arg0: memref<256xf32>, %arg1: memref<256xf32, 1>) {
  memref.copy %arg0, %arg1 : memref<256xf32> to memref<256xf32, 1>
  return
}

// A DDR<->VTCM copy whose DDR endpoint is a function-local memref.alloc is NOT
// cache-coherent with the UDMA engine (the alloc is written on the DSP), so it
// must stay a memref.copy instead of becoming a DMA. This is the masked-load
// staging buffer shape produced by reductions / rms_norm.
// CHECK-LABEL: func.func @local_alloc_ddr_to_vtcm
// CHECK-NOT: memref.dma_start
// CHECK-NOT: memref.dma_wait
// CHECK: memref.copy
func.func @local_alloc_ddr_to_vtcm(%vtcm: memref<256xf32, 1>) {
  %alloc = memref.alloc() : memref<256xf32>
  memref.copy %alloc, %vtcm : memref<256xf32> to memref<256xf32, 1>
  return
}

// Same as above for the VTCM->DDR(local alloc) direction (reduction/result
// copy-out staging).
// CHECK-LABEL: func.func @local_alloc_vtcm_to_ddr
// CHECK-NOT: memref.dma_start
// CHECK-NOT: memref.dma_wait
// CHECK: memref.copy
func.func @local_alloc_vtcm_to_ddr(%vtcm: memref<256xf32, 1>) {
  %alloc = memref.alloc() : memref<256xf32>
  memref.copy %vtcm, %alloc : memref<256xf32, 1> to memref<256xf32>
  return
}

// Rank-0 (scalar) local alloc, the reduction accumulator/result temporary.
// CHECK-LABEL: func.func @local_alloc_scalar
// CHECK-NOT: memref.dma_start
// CHECK-NOT: memref.dma_wait
// CHECK: memref.copy
func.func @local_alloc_scalar(%vtcm: memref<f32, 1>) {
  %alloc = memref.alloc() : memref<f32>
  memref.copy %alloc, %vtcm : memref<f32> to memref<f32, 1>
  return
}

// The local-allocation check looks through view-like ops (subview), so a copy
// from a subview of a local alloc is also kept as memref.copy.
// CHECK-LABEL: func.func @local_alloc_through_subview
// CHECK-NOT: memref.dma_start
// CHECK-NOT: memref.dma_wait
// CHECK: memref.copy
func.func @local_alloc_through_subview(%vtcm: memref<129xf32, 1>) {
  %alloc = memref.alloc() : memref<256xf32>
  %sv = memref.subview %alloc[0] [129] [1] : memref<256xf32> to memref<129xf32, strided<[1]>>
  memref.copy %sv, %vtcm : memref<129xf32, strided<[1]>> to memref<129xf32, 1>
  return
}

// A memref.alloca (stack) DDR endpoint is also treated as local.
// CHECK-LABEL: func.func @local_alloca_ddr_to_vtcm
// CHECK-NOT: memref.dma_start
// CHECK-NOT: memref.dma_wait
// CHECK: memref.copy
func.func @local_alloca_ddr_to_vtcm(%vtcm: memref<256xf32, 1>) {
  %alloca = memref.alloca() : memref<256xf32>
  memref.copy %alloca, %vtcm : memref<256xf32> to memref<256xf32, 1>
  return
}
