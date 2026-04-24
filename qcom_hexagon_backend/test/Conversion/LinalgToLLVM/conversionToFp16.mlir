// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(func.func(conversion-to-fp16,canonicalize,cse,optimize-extf-truncf-op,canonicalize,cse))' | FileCheck %s

// CHECK-LABEL:   func.func @test_generic_extf_truncf
func.func @test_generic_extf_truncf(%arg0: tensor<64xf16>, %arg1: tensor<64xf16>, %arg2: tensor<64xf16>) -> tensor<64xf16> {

    // CHECK-NEXT:           %[[ZERO:.*]] = arith.constant 0.000000e+00 : f16
    // CHECK-NEXT:           %[[EMPTY_TENSOR:.*]] = tensor.empty() : tensor<64xf16>
    // CHECK-NEXT:           %[[FILLED_TENSOR:.*]] = linalg.fill ins(%[[ZERO]] : f16) outs(%[[EMPTY_TENSOR]] : tensor<64xf16>) -> tensor<64xf16>
    // CHECK-NEXT:           %[[RESULT:.*]] = linalg.generic {{.*}} ins(%[[INPUT_A:.*]], %[[INPUT_B:.*]] : tensor<64xf16>, tensor<64xf16>) outs(%[[FILLED_TENSOR]] : tensor<64xf16>) {
    // CHECK-NEXT:           ^bb0(%[[IN_A:.*]]: f16, %[[IN_B:.*]]: f16, %[[OUT:.*]]: f16):
    // CHECK-NEXT:             %[[SUM:.*]] = arith.addf %[[IN_A]], %[[IN_B]] : f16
    // CHECK-NEXT:             linalg.yield %[[SUM]] : f16
    // CHECK-NEXT:           } -> tensor<64xf16>
    // CHECK-NEXT:           return %[[RESULT]] : tensor<64xf16>

    %cst = arith.constant 0.000000e+00 : f32

    %1 = tensor.empty() : tensor<64xf32>
    %2 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%arg0 : tensor<64xf16>) outs(%1 : tensor<64xf32>) {
    ^bb0(%in: f16, %out: f32):
        %100 = arith.extf %in : f16 to f32
        linalg.yield %100 : f32
    } -> tensor<64xf32>

    %3 = tensor.empty() : tensor<64xf32>
    %4 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%arg1: tensor<64xf16>) outs(%3 : tensor<64xf32>) {
    ^bb0(%in: f16, %out: f32):
        %100 = arith.extf %in : f16 to f32
        linalg.yield %100 : f32
    } -> tensor<64xf32>

    %5 = tensor.empty() : tensor<64xf32>
    %fill = linalg.fill ins(%cst : f32) outs(%5 : tensor<64xf32>) -> tensor<64xf32>
    %6 = linalg.generic {
            indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
            iterator_types = ["parallel"]
        } ins(%2, %4 : tensor<64xf32>, tensor<64xf32>) outs(%fill: tensor<64xf32>) {
    ^bb0(%in: f32, %in_3: f32, %out: f32):
        %100 = arith.addf %in, %in_3 : f32
        linalg.yield %100 : f32
    } -> tensor<64xf32>

    %8 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%6 : tensor<64xf32>) outs(%arg2 : tensor<64xf16>) {
    ^bb0(%in: f32, %out: f16):
        %100 = arith.truncf %in : f32 to f16
        linalg.yield %100 : f16
    } -> tensor<64xf16>
    return %8: tensor<64xf16>
}

// The test converts the operations to work on fp16 data, 
// it moves the transpose to operate on fp16 data,
// instead of extending the data and then applying transpose. 
// CHECK-LABEL:   func.func @test_transpose_matmul
func.func @test_transpose_matmul(%arg0: tensor<64x256xf16>, %arg1: tensor<64x64xf16>, %arg2: tensor<256x64xf16>) -> tensor<256x64xf16> {
    
    // CHECK-NEXT:       %[[ZERO:.*]] = arith.constant 0.000000e+00 : f32
    // CHECK-NEXT:       %[[EMPTY_F32_64x64:.*]] = tensor.empty() : tensor<64x64xf32>
    // CHECK-NEXT:       %[[RHS_EXTENDED:.*]] = linalg.generic {{.*}} ins(%[[ARG1:.*]] : tensor<64x64xf16>) outs(%[[EMPTY_F32_64x64]] : tensor<64x64xf32>) {
    // CHECK-NEXT:       ^bb0(%[[IN_F16:.*]]: f16, %[[OUT_F32:.*]]: f32):
    // CHECK-NEXT:         %[[EXTENDED:.*]] = arith.extf %[[IN_F16]] : f16 to f32
    // CHECK-NEXT:         linalg.yield %[[EXTENDED]] : f32
    // CHECK-NEXT:       } -> tensor<64x64xf32>
    // CHECK-NEXT:       %[[EMPTY_F32_256x64:.*]] = tensor.empty() : tensor<256x64xf32>
    // CHECK-NEXT:       %[[EMPTY_F16_256x64:.*]] = tensor.empty() : tensor<256x64xf16>
    // CHECK-NEXT:       %[[LHS_TRANSPOSED:.*]] = linalg.transpose ins(%[[ARG0:.*]] : tensor<64x256xf16>) outs(%[[EMPTY_F16_256x64]] : tensor<256x64xf16>) permutation = [1, 0]
    // CHECK-NEXT:       %[[LHS_EXTENDED:.*]] = linalg.generic {{.*}} ins(%[[LHS_TRANSPOSED]] : tensor<256x64xf16>) outs(%[[EMPTY_F32_256x64]] : tensor<256x64xf32>) {
    // CHECK-NEXT:       ^bb0(%[[IN_F16:.*]]: f16, %[[OUT_F32:.*]]: f32):
    // CHECK-NEXT:         %[[EXTENDED:.*]] = arith.extf %[[IN_F16]] : f16 to f32
    // CHECK-NEXT:         linalg.yield %[[EXTENDED]] : f32
    // CHECK-NEXT:       } -> tensor<256x64xf32>
    // CHECK-NEXT:       %[[MATMUL_INIT:.*]] = linalg.fill ins(%[[ZERO]] : f32) outs(%[[EMPTY_F32_256x64]] : tensor<256x64xf32>) -> tensor<256x64xf32>
    // CHECK-NEXT:       %[[MATMUL_RESULT:.*]] = linalg.matmul ins(%[[LHS_EXTENDED]], %[[RHS_EXTENDED]] : tensor<256x64xf32>, tensor<64x64xf32>) outs(%[[MATMUL_INIT]] : tensor<256x64xf32>) -> tensor<256x64xf32>
    // CHECK-NEXT:       %[[RESULT_TRUNCATED:.*]] = linalg.generic {{.*}} ins(%[[MATMUL_RESULT]] : tensor<256x64xf32>) outs(%[[ARG2:.*]] : tensor<256x64xf16>) {
    // CHECK-NEXT:       ^bb0(%[[IN_F32:.*]]: f32, %[[OUT_F16:.*]]: f16):
    // CHECK-NEXT:         %[[TRUNCATED:.*]] = arith.truncf %[[IN_F32]] : f32 to f16
    // CHECK-NEXT:         linalg.yield %[[TRUNCATED]] : f16
    // CHECK-NEXT:       } -> tensor<256x64xf16>
    // CHECK-NEXT:       return %[[RESULT_TRUNCATED]] : tensor<256x64xf16>

    %cst = arith.constant 0.000000e+00 : f32
    %1 = tensor.empty() : tensor<64x256xf32>
    %2 = linalg.generic {
            indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
            iterator_types = ["parallel", "parallel"]
        } ins(%arg0 : tensor<64x256xf16>) outs(%1 : tensor<64x256xf32>) {
    ^bb0(%in: f16, %out: f32):
        %100 = arith.extf %in : f16 to f32
        linalg.yield %100 : f32
    } -> tensor<64x256xf32>

    %3 = tensor.empty() : tensor<64x64xf32>
    %4 = linalg.generic {
            indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
            iterator_types = ["parallel", "parallel"]
        } ins(%arg1: tensor<64x64xf16>) outs(%3 : tensor<64x64xf32>) {
    ^bb0(%in: f16, %out: f32):
        %100 = arith.extf %in : f16 to f32
        linalg.yield %100 : f32
    } -> tensor<64x64xf32>

    %5 = tensor.empty() : tensor<256x64xf32>
    %transposed = linalg.transpose ins(%2 : tensor<64x256xf32>) outs(%5 : tensor<256x64xf32>) permutation = [1, 0]
    %fill = linalg.fill ins(%cst : f32) outs(%5 : tensor<256x64xf32>) -> tensor<256x64xf32>

    %6 = linalg.matmul ins(%transposed, %4: tensor<256x64xf32>, tensor<64x64xf32>) outs(%fill : tensor<256x64xf32>) -> tensor<256x64xf32>

    %7 = linalg.generic {
            indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
            iterator_types = ["parallel", "parallel"]
        } ins(%6 : tensor<256x64xf32>) outs(%arg2 : tensor<256x64xf16>) {
    ^bb0(%in: f32, %out: f16):
        %100 = arith.truncf %in : f32 to f16
        linalg.yield %100 : f16
    } -> tensor<256x64xf16>
    return %7: tensor<256x64xf16>
}
