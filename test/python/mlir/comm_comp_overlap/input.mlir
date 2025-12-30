func.func @exp_log_kernel(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32}) {
  %cst = arith.constant 0.000000e+00 : f32
  %c32 = arith.constant 32 : index
  %c1048576 = arith.constant 1048576 : index
  %c0 = arith.constant 0 : index
  %c262144 = arith.constant 262144 : index
  %output_ddr = memref.reinterpret_cast %arg1 to offset: [0], sizes: [1048576], strides: [1] : memref<*xf32> to memref<1048576xf32, strided<[1]>>
  %input_ddr = memref.reinterpret_cast %arg0 to offset: [0], sizes: [1048576], strides: [1] : memref<*xf32> to memref<1048576xf32, strided<[1]>>
  %output_vtcm = hexagonmem.alloc() : memref<262144xf32, 1>
  %input_vtcm = hexagonmem.alloc() : memref<262144xf32, 1>
  scf.for %arg8 = %c0 to %c1048576 step %c262144 {
    %input_ddr_subview = memref.subview %input_ddr[%arg8] [262144] [1] : memref<1048576xf32, strided<[1]>> to memref<262144xf32, strided<[1], offset: ?>>
    %input_dma_tag = memref.alloc() : memref<1xi32>
    memref.dma_start %input_ddr_subview[%c0], %input_vtcm[%c0], %c262144, %input_dma_tag[%c0] : memref<262144xf32, strided<[1], offset: ?>>, memref<262144xf32, 1>, memref<1xi32>
    memref.dma_wait %input_dma_tag[%c0], %c262144 : memref<1xi32>
    memref.dealloc %input_dma_tag : memref<1xi32>
    %output_ddr_subview = memref.subview %output_ddr[%arg8] [262144] [1] : memref<1048576xf32, strided<[1]>> to memref<262144xf32, strided<[1], offset: ?>>
    %output_dma_tag = memref.alloc() : memref<1xi32>
    memref.dma_start %output_ddr_subview[%c0], %output_vtcm[%c0], %c262144, %output_dma_tag[%c0] : memref<262144xf32, strided<[1], offset: ?>>, memref<262144xf32, 1>, memref<1xi32>
    memref.dma_wait %output_dma_tag[%c0], %c262144 : memref<1xi32>
    memref.dealloc %output_dma_tag : memref<1xi32>
    scf.for %arg9 = %c0 to %c262144 step %c32 {
      %input_vtcm_vec_size = memref.subview %input_vtcm[%arg9] [32] [1] : memref<262144xf32, 1> to memref<32xf32, strided<[1], offset: ?>, 1>
      %output_vtcm_vec_size = memref.subview %output_vtcm[%arg9] [32] [1] : memref<262144xf32, 1> to memref<32xf32, strided<[1], offset: ?>, 1>
      %2 = vector.transfer_read %input_vtcm_vec_size[%c0], %cst {in_bounds = [true]} : memref<32xf32, strided<[1], offset: ?>, 1>, vector<32xf32>
      %3 = math.exp %2 fastmath<fast> : vector<32xf32>
      %4 = math.log %3 fastmath<fast> : vector<32xf32>
      vector.transfer_write %4, %output_vtcm_vec_size[%c0] {in_bounds = [true]} : vector<32xf32>, memref<32xf32, strided<[1], offset: ?>, 1>
    }
    %output_dma_tag2 = memref.alloc() : memref<1xi32>
    memref.dma_start %output_vtcm[%c0], %output_ddr_subview[%c0], %c262144, %output_dma_tag2[%c0] : memref<262144xf32, 1>, memref<262144xf32, strided<[1], offset: ?>>, memref<1xi32>
    memref.dma_wait %output_dma_tag2[%c0], %c262144 : memref<1xi32>
    memref.dealloc %output_dma_tag2 : memref<1xi32>
  }
  hexagonmem.dealloc %input_vtcm : memref<262144xf32, 1>
  hexagonmem.dealloc %output_vtcm : memref<262144xf32, 1>
  return
}
