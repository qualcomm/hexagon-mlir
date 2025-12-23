// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(func.func(hexmem-cpy-to-dma,cse))' | FileCheck %s --check-prefixes=CHECK

// CHECK-LABEL: func.func @hexmem_copy_firstTile_3d
// CHECK: %[[TILE:.*]] = hexagonmem.alloc() {alignment = 2048 : i64, bufferization.manual_deallocation} : memref<2x32x32xi8, 1>
// CHECK: %[[SUBVIEW_SRC:.*]] = memref.subview %arg0[0, 0, 0] [2, 32, 32] [1, 1, 1]
// CHECK: %[[SUBVIEW_DST:.*]] = memref.subview %arg1[0, 0, 0] [2, 32, 32] [1, 1, 1]
// CHECK: %[[DMA_ALLOC1:.*]] = memref.alloc() : memref<1xi32>
// CHECK: %[[C0:.*]] = arith.constant 0 : index
// CHECK: %[[C2:.*]] = arith.constant 2 : index
// CHECK: %[[C1:.*]] = arith.constant 1 : index
// CHECK: scf.for %[[IV1:.*]] = %[[C0]] to %[[C2]] step %[[C1]] {
// CHECK: %[[C1024:.*]] = arith.constant 1024 : index
// CHECK: %[[SUB1:.*]] = memref.subview %[[SUBVIEW_SRC]][%[[IV1]], 0, 0] [1, 32, 32] [1, 1, 1] : memref<2x32x32xi8, strided<[4096, 64, 1]>> to memref<1x32x32xi8, strided<[4096, 64, 1], offset: ?>>
// CHECK: %[[C64:.*]] = arith.constant 64 : index
// CHECK: %[[C32:.*]] = arith.constant 32 : index
// CHECK: memref.dma_start %[[SUB1]][%[[C0]], %[[C0]], %[[C0]]], %[[TILE]][%[[IV1]], %[[C0]], %[[C0]]], %[[C1024]], %[[DMA_ALLOC1]][%[[C0]]], %[[C64]], %[[C32]] : memref<1x32x32xi8, strided<[4096, 64, 1], offset: ?>>, memref<2x32x32xi8, 1>, memref<1xi32>
// CHECK: memref.dma_wait %[[DMA_ALLOC1]][%[[C0]]], %[[C1024]] : memref<1xi32>
// CHECK: }
// CHECK: memref.dealloc %[[DMA_ALLOC1]] : memref<1xi32>
// CHECK: %[[DMA_ALLOC2:.*]] = memref.alloc() : memref<1xi32>
// CHECK: scf.for %[[IV2:.*]] = %[[C0]] to %[[C2]] step %[[C1]] {
// CHECK: %[[C1024_2:.*]] = arith.constant 1024 : index
// CHECK: %[[SUB2:.*]] = memref.subview %[[SUBVIEW_DST]][%[[IV2]], 0, 0] [1, 32, 32] [1, 1, 1] : memref<2x32x32xi8, strided<[4096, 64, 1]>> to memref<1x32x32xi8, strided<[4096, 64, 1], offset: ?>> 
// CHECK: %[[C64:.*]] = arith.constant 64 : index
// CHECK: %[[C32:.*]] = arith.constant 32 : index
// CHECK: memref.dma_start %[[TILE]][%[[IV2]], %[[C0]], %[[C0]]], %[[SUB2]][%[[C0]], %[[C0]], %[[C0]]], %[[C1024_2]], %[[DMA_ALLOC2]][%[[C0]]], %[[C64]], %[[C32]] : memref<2x32x32xi8, 1>, memref<1x32x32xi8, strided<[4096, 64, 1], offset: ?>>, memref<1xi32>
// CHECK: memref.dma_wait %[[DMA_ALLOC2]][%[[C0]]], %[[C1024_2]] : memref<1xi32>
// CHECK: }
// CHECK: memref.dealloc %[[DMA_ALLOC2]] : memref<1xi32>
// CHECK: hexagonmem.dealloc %[[TILE]] : memref<2x32x32xi8, 1>
// CHECK: return

func.func @hexmem_copy_firstTile_3d(
        %arg0 : memref<2x64x64xi8, strided<[4096, 64, 1], offset: 0>>,
        %arg1 : memref<2x64x64xi8, strided<[4096, 64, 1], offset: 0>>) {
    // Allocate a vtcm buffers for tile of <2x32x32xi8>
    %tile = hexagonmem.alloc() {bufferization.manual_deallocation,
        alignment = 2048 : i64} : memref<2x32x32xi8, 1>

    %src0 = memref.subview %arg0[0, 0, 0][2, 32, 32][1, 1, 1]
        : memref<2x64x64xi8, strided<[4096, 64, 1], offset: 0>>
        to memref<2x32x32xi8, strided<[4096, 64, 1], offset: 0>>
    %dest0 = memref.subview %arg1[0, 0, 0][2, 32, 32][1, 1, 1]
        : memref<2x64x64xi8, strided<[4096, 64, 1], offset: 0>>
        to memref<2x32x32xi8, strided<[4096, 64, 1], offset: 0>>
    hexagonmem.copy %src0, %tile
        : memref<2x32x32xi8, strided<[4096,64, 1], offset: 0>>
        to memref<2x32x32xi8, 1>
    hexagonmem.copy %tile, %dest0
        : memref<2x32x32xi8, 1>
        to memref<2x32x32xi8, strided<[4096, 64, 1], offset: 0>>
    hexagonmem.dealloc %tile : memref<2x32x32xi8, 1>
    return
}

// CHECK-LABEL: func.func @hexmem_copy_middleTile_3d
// CHECK: %[[TILE:.*]] = hexagonmem.alloc() {alignment = 2048 : i64, bufferization.manual_deallocation} : memref<2x32x32xi8, 1>
// CHECK: %[[SUBVIEW_SRC:.*]] = memref.subview %arg0[4, 64, 64] [2, 32, 32] [1, 1, 1]
// CHECK: %[[SUBVIEW_DST:.*]] = memref.subview %arg1[4, 64, 64] [2, 32, 32] [1, 1, 1]
// CHECK: %[[DMA_ALLOC1:.*]] = memref.alloc() : memref<1xi32>
// CHECK: %[[C0:.*]] = arith.constant 0 : index
// CHECK: %[[C2:.*]] = arith.constant 2 : index
// CHECK: %[[C1:.*]] = arith.constant 1 : index
// CHECK: scf.for %[[IV1:.*]] = %[[C0]] to %[[C2]] step %[[C1]] {
// CHECK: %[[C1024:.*]] = arith.constant 1024 : index
// CHECK: %[[SUB1:.*]] = memref.subview %[[SUBVIEW_SRC]][%[[IV1]], 0, 0] [1, 32, 32] [1, 1, 1] : memref<2x32x32xi8, strided<[16384, 128, 1], offset: 73792>> to memref<1x32x32xi8, strided<[16384, 128, 1], offset: ?>>
// CHECK: %[[C128:.*]] = arith.constant 128 : index
// CHECK: %[[C32:.*]] = arith.constant 32 : index
// CHECK: memref.dma_start %[[SUB1]][%[[C0]], %[[C0]], %[[C0]]], %[[TILE]][%[[IV1]], %[[C0]], %[[C0]]], %[[C1024]], %[[DMA_ALLOC1]][%[[C0]]], %[[C128]], %[[C32]] :  memref<1x32x32xi8, strided<[16384, 128, 1], offset: ?>>, memref<2x32x32xi8, 1>, memref<1xi32>
// CHECK: memref.dma_wait %[[DMA_ALLOC1]][%[[C0]]], %[[C1024]] : memref<1xi32>
// CHECK: }
// CHECK: memref.dealloc %[[DMA_ALLOC1]] : memref<1xi32>
// CHECK: %[[DMA_ALLOC2:.*]] = memref.alloc() : memref<1xi32>
// CHECK: scf.for %[[IV2:.*]] = %[[C0]] to %[[C2]] step %[[C1]] {
// CHECK: %[[C1024_2:.*]] = arith.constant 1024 : index
// CHECK: %[[SUB2:.*]] = memref.subview %[[SUBVIEW_DST]][%[[IV2]], 0, 0] [1, 32, 32] [1, 1, 1] : memref<2x32x32xi8, strided<[16384, 128, 1], offset: 73792>> to memref<1x32x32xi8, strided<[16384, 128, 1], offset: ?>> 
// CHECK: %[[C128:.*]] = arith.constant 128 : index
// CHECK: %[[C32:.*]] = arith.constant 32 : index
// CHECK: memref.dma_start %[[TILE]][%[[IV2]], %[[C0]], %[[C0]]], %[[SUB2]][%[[C0]], %[[C0]], %[[C0]]], %[[C1024_2]], %[[DMA_ALLOC2]][%[[C0]]], %[[C128]], %[[C32]] : memref<2x32x32xi8, 1>, memref<1x32x32xi8, strided<[16384, 128, 1], offset: ?>>, memref<1xi32>
// CHECK: memref.dma_wait %[[DMA_ALLOC2]][%[[C0]]], %[[C1024_2]] : memref<1xi32>
// CHECK: }
// CHECK: memref.dealloc %[[DMA_ALLOC2]] : memref<1xi32>
// CHECK: hexagonmem.dealloc %[[TILE]] : memref<2x32x32xi8, 1>
// CHECK: return

func.func @hexmem_copy_middleTile_3d(
        %arg0 : memref<10x128x128xi8, strided<[16384, 128, 1], offset: 0>>,
        %arg1 : memref<10x128x128xi8, strided<[16384, 128, 1], offset: 0>>) {
    // Allocate a vtcm buffers for tile of <2x32x32xi8>
    %tile = hexagonmem.alloc() {bufferization.manual_deallocation,
        alignment = 2048 : i64} : memref<2x32x32xi8, 1>
    
    %src0 = memref.subview %arg0[4, 64, 64][2, 32, 32][1, 1, 1]
        : memref<10x128x128xi8, strided<[16384, 128, 1], offset: 0>>
        to memref<2x32x32xi8, strided<[16384, 128, 1], offset: 73792>>
    %dest0 = memref.subview %arg1[4, 64, 64][2, 32, 32][1, 1, 1]
        : memref<10x128x128xi8, strided<[16384, 128, 1], offset: 0>>
        to memref<2x32x32xi8, strided<[16384, 128, 1], offset: 73792>>
    hexagonmem.copy %src0, %tile
        : memref<2x32x32xi8, strided<[16384, 128, 1], offset: 73792>>
        to memref<2x32x32xi8, 1>
    hexagonmem.copy %tile, %dest0
        : memref<2x32x32xi8, 1>
        to memref<2x32x32xi8, strided<[16384, 128, 1], offset: 73792>>

    hexagonmem.dealloc %tile : memref<2x32x32xi8, 1>
    return
}

// CHECK-LABEL: func.func @hexmem_copy_middleTile_4d
// CHECK: %[[TILE:.*]] = hexagonmem.alloc() {alignment = 2048 : i64, bufferization.manual_deallocation} : memref<4x2x32x32xi8, 1>
// CHECK: %[[SUBVIEW_SRC:.*]] = memref.subview %arg0[8, 4, 64, 64] [4, 2, 32, 32] [1, 1, 1, 1]
// CHECK: %[[SUBVIEW_DST:.*]] = memref.subview %arg1[8, 4, 64, 64] [4, 2, 32, 32] [1, 1, 1, 1]
// CHECK: %[[DMA_ALLOC1:.*]] = memref.alloc() : memref<1xi32>
// CHECK: %[[C0:.*]] = arith.constant 0 : index
// CHECK: %[[C4:.*]] = arith.constant 4 : index
// CHECK: %[[C1:.*]] = arith.constant 1 : index
// CHECK: scf.for %[[IV1:.*]] = %[[C0]] to %[[C4]] step %[[C1]] {
// CHECK: %[[C2:.*]] = arith.constant 2 : index
// CHECK: scf.for %[[IV2:.*]] = %[[C0]] to %[[C2]] step %[[C1]] {
// CHECK: %[[C1024:.*]] = arith.constant 1024 : index
// CHECK: %[[SUB1:.*]] = memref.subview %[[SUBVIEW_SRC]][%[[IV1]], %[[IV2]], 0, 0] [1, 1, 32, 32] [1, 1, 1, 1]
// CHECK: %[[C128:.*]] = arith.constant 128 : index
// CHECK: %[[C32:.*]] = arith.constant 32 : index
// CHECK: memref.dma_start %[[SUB1]][%[[C0]], %[[C0]], %[[C0]], %[[C0]]], %[[TILE]][%[[IV1]], %[[IV2]], %[[C0]], %[[C0]]], %[[C1024]], %[[DMA_ALLOC1]][%[[C0]]], %[[C128]], %[[C32]]
// CHECK: memref.dma_wait %[[DMA_ALLOC1]][%[[C0]]], %[[C1024]]
// CHECK: }
// CHECK: }
// CHECK: memref.dealloc %[[DMA_ALLOC1]] : memref<1xi32>
// CHECK: %[[DMA_ALLOC2:.*]] = memref.alloc() : memref<1xi32>
// CHECK: scf.for %[[IV3:.*]] = %[[C0]] to %[[C4]] step %[[C1]] {
// CHECK: %[[C2_2:.*]] = arith.constant 2 : index
// CHECK: scf.for %[[IV4:.*]] = %[[C0]] to %[[C2_2]] step %[[C1]] {
// CHECK: %[[C1024_2:.*]] = arith.constant 1024 : index
// CHECK: %[[SUB2:.*]] = memref.subview %[[SUBVIEW_DST]][%[[IV3]], %[[IV4]], 0, 0] [1, 1, 32, 32] [1, 1, 1, 1]
// CHECK: %[[C128:.*]] = arith.constant 128 : index
// CHECK: %[[C32:.*]] = arith.constant 32 : index
// CHECK: memref.dma_start %[[TILE]][%[[IV3]], %[[IV4]], %[[C0]], %[[C0]]], %[[SUB2]][%[[C0]], %[[C0]], %[[C0]], %[[C0]]], %[[C1024_2]], %[[DMA_ALLOC2]][%[[C0]]], %[[C128]], %[[C32]]
// CHECK: memref.dma_wait %[[DMA_ALLOC2]][%[[C0]]], %[[C1024_2]]
// CHECK: }
// CHECK: }
// CHECK: memref.dealloc %[[DMA_ALLOC2]] : memref<1xi32>
// CHECK: hexagonmem.dealloc %[[TILE]] : memref<4x2x32x32xi8, 1>
// CHECK: return

func.func @hexmem_copy_middleTile_4d(
        %arg0 : memref<20x10x128x128xi8, strided<[163840, 16384, 128, 1], offset: 0>>,
        %arg1 : memref<20x10x128x128xi8, strided<[163840, 16384, 128, 1], offset: 0>>) {
    // Allocate a vtcm buffers for tile of <2x32x32xi8>
    %tile = hexagonmem.alloc() {bufferization.manual_deallocation,
        alignment = 2048 : i64} : memref<4x2x32x32xi8, 1>
    
    %src0 = memref.subview %arg0[8, 4, 64, 64][4, 2, 32, 32][1, 1, 1, 1]
        : memref<20x10x128x128xi8, strided<[163840, 16384, 128, 1], offset: 0>>
        to memref<4x2x32x32xi8, strided<[163840, 16384, 128, 1], offset: 1384512>>
    %dest0 = memref.subview %arg1[8, 4, 64, 64][4, 2, 32, 32][1, 1, 1, 1]
        : memref<20x10x128x128xi8, strided<[163840, 16384, 128, 1], offset: 0>>
        to memref<4x2x32x32xi8, strided<[163840, 16384, 128, 1], offset: 1384512>>
    hexagonmem.copy %src0, %tile
        : memref<4x2x32x32xi8, strided<[163840, 16384, 128, 1], offset: 1384512>>
        to memref<4x2x32x32xi8, 1>
    hexagonmem.copy %tile, %dest0
        : memref<4x2x32x32xi8, 1>
        to memref<4x2x32x32xi8, strided<[163840, 16384, 128, 1], offset: 1384512>>

    hexagonmem.dealloc %tile : memref<4x2x32x32xi8, 1>
    return
}

// CHECK-LABEL: func.func @hexmem_copy_same_memory
// CHECK: memref.dma_start
// CHECK: memref.dma_wait
// CHECK: memref.dma_start
// CHECK: memref.dma_wait
// CHECK-NOT: memref.dma_start
// CHECK-NOT: memref.dma_wait
// CHECK: hexagonmem.copy
// CHECK: hexagonmem.copy
func.func @hexmem_copy_same_memory(
        %arg0 : memref<20x10x128x128xi8, strided<[163840, 16384, 128, 1], offset: 0>>,
        %arg1 : memref<20x10x128x128xi8, strided<[163840, 16384, 128, 1], offset: 0>>) {
    // Allocate two vtcm buffers for tile of <2x32x32xi8>
    %tile1 = hexagonmem.alloc() {bufferization.manual_deallocation,
        alignment = 2048 : i64} : memref<4x2x32x32xi8, 1>
    %tile2 = hexagonmem.alloc() {bufferization.manual_deallocation,
        alignment = 2048 : i64} : memref<4x2x32x32xi8, 1>
    
    %src0 = memref.subview %arg0[8, 4, 64, 64][4, 2, 32, 32][1, 1, 1, 1]
        : memref<20x10x128x128xi8, strided<[163840, 16384, 128, 1], offset: 0>>
        to memref<4x2x32x32xi8, strided<[163840, 16384, 128, 1], offset: 1384512>>
    %dest0 = memref.subview %arg1[8, 4, 64, 64][4, 2, 32, 32][1, 1, 1, 1]
        : memref<20x10x128x128xi8, strided<[163840, 16384, 128, 1], offset: 0>>
        to memref<4x2x32x32xi8, strided<[163840, 16384, 128, 1], offset: 1384512>>
    hexagonmem.copy %src0, %tile1
        : memref<4x2x32x32xi8, strided<[163840, 16384, 128, 1], offset: 1384512>>
        to memref<4x2x32x32xi8, 1>
    hexagonmem.copy %dest0, %tile2
        : memref<4x2x32x32xi8, strided<[163840, 16384, 128, 1], offset: 1384512>>
        to memref<4x2x32x32xi8, 1>

    // DDR to DDR copy
    hexagonmem.copy %src0, %dest0
        : memref<4x2x32x32xi8, strided<[163840, 16384, 128, 1], offset: 1384512>>
        to memref<4x2x32x32xi8, strided<[163840, 16384, 128, 1], offset: 1384512>>

    // VTCM to VTCM copy
    hexagonmem.copy %tile1, %tile2
        : memref<4x2x32x32xi8, 1> to memref<4x2x32x32xi8, 1>

    hexagonmem.dealloc %tile1 : memref<4x2x32x32xi8, 1>
    hexagonmem.dealloc %tile2 : memref<4x2x32x32xi8, 1>
    return
}
