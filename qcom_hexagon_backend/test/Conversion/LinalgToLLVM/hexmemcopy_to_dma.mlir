// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(func.func(hexmem-cpy-to-dma))' | FileCheck %s --check-prefixes=CHECK1
// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(func.func(hexmem-cpy-to-dma))' | linalg-hexagon-opt -convert-to-llvm | FileCheck %s --check-prefixes=CHECK2
// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(func.func(hexmem-cpy-to-dma))' | linalg-hexagon-opt -convert-to-llvm | FileCheck %s --check-prefixes=CHECK3

// CHECK1-LABEL: func.func @hexagon_memcpy
// CHECK1: %[[ALLOC:.*]] = hexagonmem.alloc() : memref<256x256xf32, 1>
// CHECK1: %[[DMA_ALLOC:.*]] = memref.alloc() : memref<1xi32>
// CHECK1: %[[C65536:.*]] = arith.constant 65536 : index
// CHECK1: memref.dma_start %arg0[%{{.*}}, %{{.*}}], %[[ALLOC]][%{{.*}}, %{{.*}}], %[[C65536]], %[[DMA_ALLOC]][%{{.*}}] : memref<256x256xf32, strided<[256, 1]>>, memref<256x256xf32, 1>, memref<1xi32>
// CHECK1: memref.dma_wait %[[DMA_ALLOC]][%{{.*}}], %[[C65536]] : memref<1xi32>
// CHECK1: %[[DMA_ALLOC2:.*]] = memref.alloc() : memref<1xi32>
// CHECK1: %[[C65536_2:.*]] = arith.constant 65536 : index
// CHECK1: memref.dma_start %[[ALLOC]][%{{.*}}, %{{.*}}], %arg1[%{{.*}}, %{{.*}}], %[[C65536_2]], %[[DMA_ALLOC2]][%{{.*}}] : memref<256x256xf32, 1>, memref<256x256xf32, strided<[256, 1]>>, memref<1xi32>
// CHECK1: memref.dma_wait %[[DMA_ALLOC2]][%{{.*}}], %[[C65536_2]] : memref<1xi32>
// CHECK1: return

func.func @hexagon_memcpy(%arg0: memref<256x256xf32, strided<[256, 1]>>, %arg1: memref<256x256xf32, strided<[256, 1]>>) {
  %0 = hexagonmem.alloc() : memref<256x256xf32, 1>
  hexagonmem.copy %arg0, %0 : memref<256x256xf32, strided<[256, 1]>> to memref<256x256xf32, 1>
  hexagonmem.copy %0, %arg1 : memref<256x256xf32, 1> to memref<256x256xf32, strided<[256, 1]>>
  return
}

// Tests to ensure that non-contiguous copies to/from same memspace are not replaced with DMA calls and instead gets lowered to `memrefCopy`;
// CHECK2-LABEL: llvm.func @subview
// CHECK2-NOT: memref.dma_start
// CHECK2-NOT: memref.dma_wait
// CHECK2: memrefCopy
// CHECK2: memrefCopy

func.func @subview(%0 : memref<2x64xf32, strided<[64, 1], offset: 0>>, %result : memref<2x32xf32, strided<[32, 1], offset: 0>>, %result1 : memref<2x32xf32, strided<[32, 1], offset: 0>>) {
  %1 = memref.subview %0[0, 0][2, 32][1, 1] : memref<2x64xf32, strided<[64, 1], offset: 0>> to memref<2x32xf32, strided<[64, 1], offset: 0>>
  %2 = memref.subview %0[0, 32][2, 32][1, 1] : memref<2x64xf32, strided<[64, 1], offset: 0>> to memref<2x32xf32, strided<[64, 1], offset: 32>>
  hexagonmem.copy %1, %result : memref<2x32xf32, strided<[64, 1], offset: 0>> to memref<2x32xf32, strided<[32, 1], offset: 0>>
  // both same memspace - no memref.dma*
  hexagonmem.copy %2, %result1 : memref<2x32xf32, strided<[64, 1], offset: 32>> to memref<2x32xf32, strided<[32, 1], offset: 0>>
  return
}

// Tests to ensure that 2D strided copies are replaced with DMA calls, if memspaces are different, and strided is slower memspace
// CHECK3-LABEL: llvm.func @subview_dma
// CHECK3: memref.dma_start
// CHECK3: memref.dma_wait
// CHECK3-NOT: memref.dma_start
// CHECK3-NOT: memref.dma_wait
// CHECK3: memrefCopy
// CHECK3: memrefCopy

func.func @subview_dma(%arg0 : memref<2x64xf32, strided<[64, 1], offset: 0>>,
    %arg1 : memref<2x64xf32, strided<[64, 1], offset: 0>, 1>,
    %result0 : memref<2x32xf32, strided<[32, 1], offset: 0>, 1>,
    %result1 : memref<2x32xf32, strided<[64, 1], offset: 0>, 1>,
    %result2 : memref<2x32xf32, strided<[32, 1], offset: 0>>,
    %ind: index) {
  %0 = memref.subview %arg0[%ind, %ind][2, 32][1, 1] : memref<2x64xf32, strided<[64, 1], offset: 0>> to memref<2x32xf32, strided<[64, 1], offset: ?>>
  %1 = memref.subview %arg0[0, 32][2, 32][1, 1] : memref<2x64xf32, strided<[64, 1], offset: 0>> to memref<2x32xf32, strided<[64, 1], offset: 32>>
  %2 = memref.subview %arg1[0, 0][2, 32][1, 1] : memref<2x64xf32, strided<[64, 1], offset: 0>, 1> to memref<2x32xf32, strided<[64, 1], offset: 0>, 1>
  // strided copy from strided slow mem to non-strided fast mem: will be replaced by memref.dma*
  hexagonmem.copy %0, %result0 : memref<2x32xf32, strided<[64, 1], offset: ?>> to memref<2x32xf32, strided<[32, 1], offset: 0>, 1>
  // both source and destination are strided - no dma copy
  hexagonmem.copy %1, %result1 : memref<2x32xf32, strided<[64, 1], offset: 32>> to memref<2x32xf32, strided<[64, 1], offset: 0>, 1>
  // strided copy: from fast strided to slow non-strided - no dma copy
  hexagonmem.copy %2, %result2 : memref<2x32xf32, strided<[64, 1], offset: 0>, 1> to memref<2x32xf32, strided<[32, 1], offset: 0>>
  return
}
