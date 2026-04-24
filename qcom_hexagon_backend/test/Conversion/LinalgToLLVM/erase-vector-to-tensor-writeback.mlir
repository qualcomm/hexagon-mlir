// RUN: linalg-hexagon-opt %s -erase-vector-to-tensor-writeback -cse -canonicalize | FileCheck %s

// CHECK-LABEL: func.func @softmax_kernel_extract(
// CHECK-SAME: %[[ARG0:.*]]: memref<*xf32> {{.*}}, %[[ARG1:.*]]: memref<*xf32> {{.*}}, %[[ARG2:.*]]: i32, %[[ARG3:.*]]: i32, %[[ARG4:.*]]: i32, %[[ARG5:.*]]: i32, %[[ARG6:.*]]: i32, %[[ARG7:.*]]: i32)
// CHECK-DAG: %[[C1024:.*]] = arith.constant 1024 : index
// CHECK-DAG: %[[C32:.*]] = arith.constant 32 : index
// CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG: %[[CST:.*]] = arith.constant dense<0xFF800000> : vector<32xf32>
// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[C1024_I32:.*]] = arith.constant 1024 : i32
// CHECK-DAG: %[[C16384_I32:.*]] = arith.constant 16384 : i32
// CHECK-DAG: %[[CST_0:.*]] = arith.constant 0xFF800000 : f32
// CHECK-DAG: %[[CST_1:.*]] = arith.constant 0.000000e+00 : f32
// CHECK: %[[V0:.*]] = arith.muli %[[ARG5]], %[[C1024_I32]] : i32
// CHECK: %[[V1:.*]] = arith.muli %[[ARG2]], %[[C1024_I32]] : i32
// CHECK: %[[EMPTY:.*]] = tensor.empty() : tensor<1024xf32>
// CHECK-NOT: tensor.empty() : tensor<32xf32>
// CHECK-NOT: vector.transfer_write {{.*}} : vector<32xf32>, tensor<32xf32>
// CHECK: scf.for %[[ARG8:.*]] = %[[V0]] to %[[C16384_I32]] step %[[V1]] : i32 {
// CHECK: %[[V3:.*]] = arith.index_cast %[[ARG8]] : i32 to index
// CHECK: %[[REINTERPRET_CAST:.*]] = memref.reinterpret_cast %[[ARG1]] to offset: [%[[V3]]], sizes: [1024], strides: [1]
// CHECK: %[[REINTERPRET_CAST_2:.*]] = memref.reinterpret_cast %[[ARG0]] to offset: [%[[V3]]], sizes: [1024], strides: [1]
// CHECK: %[[V4:.*]] = bufferization.to_tensor %[[REINTERPRET_CAST_2]] restrict
// CHECK: %{{.*}} = bufferization.alloc_tensor() : tensor<f32>
// CHECK: %[[EXPANDED:.*]] = tensor.expand_shape %[[V4]] {{\[}}[0, 1]{{\]}} output_shape [32, 32]
// CHECK: %[[V6:.*]] = scf.for %[[ARG9:.*]] = %[[C0]] to %[[C32]] step %[[C1]] iter_args(%[[ARG10:.*]] = %[[CST]]) -> (vector<32xf32>) {
// CHECK: %[[EXTRACTED_SLICE:.*]] = tensor.extract_slice %[[EXPANDED]][%[[ARG9]], 0] [1, 32] [1, 1]
// CHECK: %{{.*}} = vector.transfer_read %[[EXTRACTED_SLICE]][%[[C0]]], %[[CST_1]] {in_bounds = [true]}
// CHECK-NOT: vector.transfer_read {{.*}}[%[[C0]]], %[[CST_1]] {in_bounds = [true]} : tensor<32xf32>, vector<32xf32>
// CHECK: %{{.*}} = arith.maxnumf %{{.*}}, %[[ARG10]] fastmath<fast> : vector<32xf32>
// CHECK-NOT: vector.transfer_write
// CHECK: scf.yield %{{.*}} : vector<32xf32>
// CHECK: }
// CHECK-NOT: vector.transfer_read %[[V6]]
// CHECK: %[[V7:.*]] = vector.reduction <maxnumf>, %[[V6]], %[[CST_0]] fastmath<fast> : vector<32xf32> into f32
// CHECK: %{{.*}} = scf.for %[[ARG9:.*]] = %[[C0]] to %[[C1024]] step %[[C32]] iter_args(%[[ARG10:.*]] = %[[EMPTY]]) -> (tensor<1024xf32>) {
// CHECK: %[[EXTRACTED_SLICE:.*]] = tensor.extract_slice %[[V4]][%[[ARG9]]] [32] [1]
// CHECK: %[[EXTRACTED_SLICE_3:.*]] = tensor.extract_slice %[[ARG10]][%[[ARG9]]] [32] [1]
// CHECK: %{{.*}} = vector.transfer_read %[[EXTRACTED_SLICE]][%[[C0]]], %[[CST_1]] {in_bounds = [true]}
// CHECK: %{{.*}} = vector.broadcast %[[V7]] : f32 to vector<32xf32>
// CHECK: %{{.*}} = arith.subf %{{.*}}, %{{.*}} fastmath<fast> : vector<32xf32>
// CHECK: %{{.*}} = math.exp %{{.*}} fastmath<fast> : vector<32xf32>
// CHECK: %{{.*}} = vector.transfer_write %{{.*}}, %[[EXTRACTED_SLICE_3]][%[[C0]]] {in_bounds = [true]}
// CHECK: %[[INSERTED_SLICE:.*]] = tensor.insert_slice %{{.*}} into %[[ARG10]][%[[ARG9]]] [32] [1]
// CHECK: scf.yield %[[INSERTED_SLICE]] : tensor<1024xf32>
// CHECK: }
// CHECK: bufferization.materialize_in_destination %{{.*}} in writable %[[REINTERPRET_CAST]]
// CHECK: }
// CHECK: return

module {
  func.func @softmax_kernel_extract(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, %arg1: memref<*xf32> {tt.divisibility = 16 : i32},
                                    %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32) {
    %c1024 = arith.constant 1024 : index
    %c32 = arith.constant 32 : index
    %c1 = arith.constant 1 : index
    %cst = arith.constant dense<0xFF800000> : vector<32xf32>
    %c0 = arith.constant 0 : index
    %c1024_i32 = arith.constant 1024 : i32
    %c16384_i32 = arith.constant 16384 : i32
    %cst_0 = arith.constant 0xFF800000 : f32
    %cst_1 = arith.constant 0.000000e+00 : f32
    %0 = arith.muli %arg5, %c1024_i32 : i32
    %1 = arith.muli %arg2, %c1024_i32 : i32
    %2 = tensor.empty() : tensor<1024xf32>
    %3 = tensor.empty() : tensor<32xf32>
    %4 = vector.transfer_write %cst, %3[%c0] {in_bounds = [true]} : vector<32xf32>, tensor<32xf32>
    scf.for %arg8 = %0 to %c16384_i32 step %1  : i32 {
      %5 = arith.index_cast %arg8 : i32 to index
      %reinterpret_cast = memref.reinterpret_cast %arg1 to offset: [%5], sizes: [1024], strides: [1] : memref<*xf32> to memref<1024xf32, strided<[1], offset: ?>>
      %reinterpret_cast_2 = memref.reinterpret_cast %arg0 to offset: [%5], sizes: [1024], strides: [1] : memref<*xf32> to memref<1024xf32, strided<[1], offset: ?>>
      %6 = bufferization.to_tensor %reinterpret_cast_2 restrict : memref<1024xf32, strided<[1], offset: ?>> to tensor<1024xf32>
      %7 = bufferization.alloc_tensor() : tensor<f32>
      %expanded = tensor.expand_shape %6 [[0, 1]] output_shape [32, 32] : tensor<1024xf32> into tensor<32x32xf32>
      %8 = scf.for %arg9 = %c0 to %c32 step %c1 iter_args(%arg10 = %4) -> (tensor<32xf32>) {
        %extracted_slice = tensor.extract_slice %expanded[%arg9, 0] [1, 32] [1, 1] : tensor<32x32xf32> to tensor<32xf32>
        %12 = vector.transfer_read %extracted_slice[%c0], %cst_1 {in_bounds = [true]} : tensor<32xf32>, vector<32xf32>
        %13 = vector.transfer_read %arg10[%c0], %cst_1 {in_bounds = [true]} : tensor<32xf32>, vector<32xf32>
        %14 = arith.maxnumf %12, %13 fastmath<fast> : vector<32xf32>
        %15 = vector.transfer_write %14, %3[%c0] {in_bounds = [true]} : vector<32xf32>, tensor<32xf32>
        scf.yield %15 : tensor<32xf32>
      }
      %9 = vector.transfer_read %8[%c0], %cst_1 {in_bounds = [true]} : tensor<32xf32>, vector<32xf32>
      %10 = vector.reduction <maxnumf>, %9, %cst_0 fastmath<fast> : vector<32xf32> into f32
      %11 = scf.for %arg9 = %c0 to %c1024 step %c32 iter_args(%arg10 = %2) -> (tensor<1024xf32>) {
        %extracted_slice = tensor.extract_slice %6[%arg9] [32] [1] : tensor<1024xf32> to tensor<32xf32>
        %extracted_slice_3 = tensor.extract_slice %arg10[%arg9] [32] [1] : tensor<1024xf32> to tensor<32xf32>
        %12 = vector.transfer_read %extracted_slice[%c0], %cst_1 {in_bounds = [true]} : tensor<32xf32>, vector<32xf32>
        %13 = vector.broadcast %10 : f32 to vector<32xf32>
        %14 = arith.subf %12, %13 fastmath<fast> : vector<32xf32>
        %15 = math.exp %14 fastmath<fast> : vector<32xf32>
        %16 = vector.transfer_write %15, %extracted_slice_3[%c0] {in_bounds = [true]} : vector<32xf32>, tensor<32xf32>
        %inserted_slice = tensor.insert_slice %16 into %arg10[%arg9] [32] [1] : tensor<32xf32> into tensor<1024xf32>
        scf.yield %inserted_slice : tensor<1024xf32>
      }
      bufferization.materialize_in_destination %11 in writable %reinterpret_cast : (tensor<1024xf32>, memref<1024xf32, strided<[1], offset: ?>>) -> ()
    }
    return
  }
}
