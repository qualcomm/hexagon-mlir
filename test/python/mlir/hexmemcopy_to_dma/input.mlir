module {
// test demonstrating 3D first tile hexagonmem copy of arg0 to arg1 through
// VTCM buffer allocated locally
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

// test demonstrating 3D first tile DMA copy of arg0 to arg1 through
// VTCM buffer allocated locally
// Single tile of arg0 of the size same as arg1 is copied to the arg1
func.func @dma_copy_firstTile_3d(
        %arg0: memref<2x64x64xi8, strided<[4096, 64, 1]>>,
        %arg1: memref<2x64x64xi8, strided<[4096, 64, 1]>>) {
    // Allocate a vtcm buffers for tile of <2x32x32xi8>
    %0 = hexagonmem.alloc() {bufferization.manual_deallocation,
        alignment = 2048 : i64} : memref<2x32x32xi8, 1>
        
    %src0 = memref.subview %arg0[0, 0, 0][2, 32, 32][1, 1, 1]
        : memref<2x64x64xi8, strided<[4096, 64, 1], offset: 0>>
        to memref<2x32x32xi8, strided<[4096, 64, 1], offset: 0>>
    %dest0 = memref.subview %arg1[0, 0, 0][2, 32, 32][1, 1, 1]
        : memref<2x64x64xi8, strided<[4096, 64, 1], offset: 0>>
        to memref<2x32x32xi8, strided<[4096, 64, 1], offset: 0>>
    
    %alloc = memref.alloc() : memref<1xi32>
    %c0 = arith.constant 0 : index
    %c64 = arith.constant 64 : index
    %c32 = arith.constant 32 : index
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    scf.for %arg3 = %c0 to %c2 step %c1 {
      %c1024 = arith.constant 1024 : index
      %subview = memref.subview %src0[%arg3, 0, 0] [1, 32, 32] [1, 1, 1]
        : memref<2x32x32xi8, strided<[4096, 64, 1], offset: 0>> to memref<1x32x32xi8,
        strided<[4096, 64, 1], offset: ?>>
      memref.dma_start %subview[%c0, %c0, %c0], %0[%arg3, %c0, %c0],
        %c1024, %alloc[%c0], %c64, %c32
        : memref<1x32x32xi8, strided<[4096, 64, 1], offset: ?>>,
        memref<2x32x32xi8, 1>, memref<1xi32>
      memref.dma_wait %alloc[%c0], %c1024 : memref<1xi32>
    }
    memref.dealloc %alloc : memref<1xi32>
    %alloc_0 = memref.alloc() : memref<1xi32>
    scf.for %arg3 = %c0 to %c2 step %c1 {
      %c1024 = arith.constant 1024 : index
      %subview = memref.subview %dest0[%arg3, 0, 0] [1, 32, 32] [1, 1, 1]
        : memref<2x32x32xi8, strided<[4096, 64, 1], offset: 0>> to
        memref<1x32x32xi8, strided<[4096, 64, 1], offset: ?>>
      memref.dma_start %0[%arg3, %c0, %c0], %subview[%c0, %c0, %c0],
        %c1024, %alloc_0[%c0], %c64, %c32
        : memref<2x32x32xi8, 1>, memref<1x32x32xi8,
        strided<[4096, 64, 1], offset: ?>>, memref<1xi32>
      memref.dma_wait %alloc_0[%c0], %c1024 : memref<1xi32>
    }
    memref.dealloc %alloc_0 : memref<1xi32>
    hexagonmem.dealloc %0 : memref<2x32x32xi8, 1>
    return
  }

// test demonstrating 3D middle tile hexagonmem copy of arg0 to arg1 through
// VTCM buffer allocated locally
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

// test demonstrating 3D middle tile DMA copy of arg0 to arg1 through
// VTCM buffer allocated locally
// Single tile of arg0 of the size same as arg1 is copied to the arg1
func.func @dma_copy_middleTile_3d(
        %arg0 : memref<10x128x128xi8, strided<[16384, 128, 1], offset: 0>>,
        %arg1 : memref<10x128x128xi8, strided<[16384, 128, 1], offset: 0>>) {
    %0 = hexagonmem.alloc() {bufferization.manual_deallocation,
        alignment = 2048 : i64} : memref<2x32x32xi8, 1>
    %src0 = memref.subview %arg0[4, 64, 64][2, 32, 32][1, 1, 1]
        : memref<10x128x128xi8, strided<[16384, 128, 1], offset: 0>>
        to memref<2x32x32xi8, strided<[16384, 128, 1], offset: 73792>>
    %dest0 = memref.subview %arg1[4, 64, 64][2, 32, 32][1, 1, 1]
        : memref<10x128x128xi8, strided<[16384, 128, 1], offset: 0>>
        to memref<2x32x32xi8, strided<[16384, 128, 1], offset: 73792>>
    %alloc = memref.alloc() : memref<1xi32>
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c6 = arith.constant 6 : index
    %c128 = arith.constant 128 : index
    %c32 = arith.constant 32 : index
    %c10 = arith.constant 10 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    scf.for %arg2 = %c0 to %c2 step %c1 {
      %c1024 = arith.constant 1024 : index
      %subview = memref.subview %src0[%arg2, 0, 0] [1, 32, 32] [1, 1, 1]
        : memref<2x32x32xi8, strided<[16384, 128, 1], offset: 73792>> to
        memref<1x32x32xi8, strided<[16384, 128, 1], offset: ?>>
      memref.dma_start %subview[%c0, %c0, %c0], %0[%arg2, %c0, %c0], %c1024, %alloc[%c0], %c128, %c32
        : memref<1x32x32xi8, strided<[16384, 128, 1], offset: ?>>,
      memref<2x32x32xi8, 1>, memref<1xi32>
      memref.dma_wait %alloc[%c0], %c1024 : memref<1xi32>
    }
    memref.dealloc %alloc : memref<1xi32>
    %alloc_0 = memref.alloc() : memref<1xi32>
    scf.for %arg2 = %c0 to %c2 step %c1 {
      %c1024 = arith.constant 1024 : index
      %subview = memref.subview %dest0[%arg2, 0, 0] [1, 32, 32] [1, 1, 1]
        : memref<2x32x32xi8, strided<[16384, 128, 1], offset: 73792>> to
        memref<1x32x32xi8, strided<[16384, 128, 1], offset: ?>>
      memref.dma_start %0[%arg2, %c0, %c0], %subview[%c0, %c0, %c0], %c1024, %alloc_0[%c0], %c128, %c32
        : memref<2x32x32xi8, 1>, memref<1x32x32xi8, strided<[16384, 128, 1], offset: ?>>,
        memref<1xi32>
      memref.dma_wait %alloc_0[%c0], %c1024 : memref<1xi32>
    }
    memref.dealloc %alloc_0 : memref<1xi32>
    hexagonmem.dealloc %0 : memref<2x32x32xi8, 1>
    return
  }

// test demonstrating 4D middle tile hexagonmem copy of arg0 to arg1 through
// VTCM buffer allocated locally
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

// test demonstrating 4D middle tile DMA copy of arg0 to arg1 through
// VTCM buffer allocated locally
// Single tile of arg0 of the size same as arg1 is copied to the arg1
func.func @dma_copy_middleTile_4d(
        %arg0 : memref<20x10x128x128xi8, strided<[163840, 16384, 128, 1], offset: 0>>,
        %arg1 : memref<20x10x128x128xi8, strided<[163840, 16384, 128, 1], offset: 0>>) {
    %0 = hexagonmem.alloc() {bufferization.manual_deallocation,
        alignment = 2048 : i64} : memref<4x2x32x32xi8, 1>
    %src0 = memref.subview %arg0[8, 4, 64, 64][4, 2, 32, 32][1, 1, 1, 1]
        : memref<20x10x128x128xi8, strided<[163840, 16384, 128, 1], offset: 0>>
        to memref<4x2x32x32xi8, strided<[163840, 16384, 128, 1], offset: 1384512>>
    %dest0 = memref.subview %arg1[8, 4, 64, 64][4, 2, 32, 32][1, 1, 1, 1]
        : memref<20x10x128x128xi8, strided<[163840, 16384, 128, 1], offset: 0>>
        to memref<4x2x32x32xi8, strided<[163840, 16384, 128, 1], offset: 1384512>>
    %alloc = memref.alloc() : memref<1xi32>
    %c0 = arith.constant 0 : index
    %c128 = arith.constant 128 : index
    %c32 = arith.constant 32 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index 
    scf.for %arg2 = %c0 to %c4 step %c1 {
      %c2 = arith.constant 2 : index
      scf.for %arg3 = %c0 to %c2 step %c1 {
        %c1024 = arith.constant 1024 : index
        %subview = memref.subview %src0[%arg2, %arg3, 0, 0] [1, 1, 32, 32] [1, 1, 1, 1]
          : memref<4x2x32x32xi8, strided<[163840, 16384, 128, 1], offset: 1384512>> to
          memref<1x1x32x32xi8, strided<[163840, 16384, 128, 1], offset: ?>>
        memref.dma_start %subview[%c0, %c0, %c0, %c0], %0[%arg2, %arg3, %c0, %c0], %c1024, %alloc[%c0], %c128, %c32
          : memref<1x1x32x32xi8, strided<[163840, 16384, 128, 1], offset: ?>>,
        memref<4x2x32x32xi8, 1>, memref<1xi32>
        memref.dma_wait %alloc[%c0], %c1024 : memref<1xi32>
      }
    }
    memref.dealloc %alloc : memref<1xi32>
    %alloc_0 = memref.alloc() : memref<1xi32>
    scf.for %arg2 = %c0 to %c4 step %c1 {
      %c2 = arith.constant 2 : index
      scf.for %arg3 = %c0 to %c2 step %c1 {
        %c1024 = arith.constant 1024 : index
        %subview = memref.subview %dest0[%arg2, %arg3, 0, 0] [1, 1, 32, 32] [1, 1, 1, 1]
          : memref<4x2x32x32xi8, strided<[163840, 16384, 128, 1], offset: 1384512>> to
          memref<1x1x32x32xi8, strided<[163840, 16384, 128, 1], offset: ?>>
        memref.dma_start %0[%arg2, %arg3, %c0, %c0], %subview[%c0, %c0, %c0, %c0], %c1024, %alloc_0[%c0], %c128, %c32
          : memref<4x2x32x32xi8, 1>, memref<1x1x32x32xi8, strided<[163840, 16384, 128, 1], offset: ?>>,
          memref<1xi32>
        memref.dma_wait %alloc_0[%c0], %c1024 : memref<1xi32>
      }
    }
    memref.dealloc %alloc_0 : memref<1xi32>
    hexagonmem.dealloc %0 : memref<4x2x32x32xi8, 1>
    return
  }
}
