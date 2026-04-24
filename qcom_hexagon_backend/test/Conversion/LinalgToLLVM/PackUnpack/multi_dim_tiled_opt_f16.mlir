// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(func.func(preprocess-tiled-conv2d, cse, canonicalize))' | FileCheck %s
// CHECK-LABEL: module {
// CHECK:   func.func @multi_dim_tiling(
// CHECK-SAME:     %arg0: memref<1x239x27x123xf16>,
// CHECK-SAME:     %arg1: memref<255x1x1x123xf16>,
// CHECK-SAME:     %arg2: memref<1x239x27x255xf16>) {

// ---- Constants and initial tensors ----

// CHECK:     %[[C8:.*]] = arith.constant 8 : index
// CHECK:     %[[RESHAPE_FILTER_SHAPE:.*]] = arith.constant dense<[8, 32, 1, 1, 4, 16, 2]> : tensor<7xindex>
// CHECK:     %[[CST0:.*]] = arith.constant 0.000000e+00 : f16
// CHECK:     %[[C0:.*]] = arith.constant 0 : index
// CHECK:     %[[C32:.*]] = arith.constant 32 : index
// CHECK:     %[[C64:.*]] = arith.constant 64 : index
// CHECK:     %[[C224:.*]] = arith.constant 224 : index
// CHECK:     %[[C192:.*]] = arith.constant 192 : index

// CHECK:     %[[T_IN:.*]] = bufferization.to_tensor %arg0 restrict writable : memref<1x239x27x123xf16> to tensor<1x239x27x123xf16>
// CHECK:     %[[W_IN:.*]] = bufferization.to_tensor %arg1 restrict writable : memref<255x1x1x123xf16> to tensor<255x1x1x123xf16>
// CHECK:     %[[OUT_TENSOR:.*]] = tensor.empty() : tensor<1x239x27x255xf16>


// ---- Init of full packed output ----

// CHECK:     %[[PACKED_OUT_EMPTY:.*]] = tensor.empty() : tensor<1x30x7x8x8x2x32x2xf16>
// CHECK:     %[[PACKED_OUT_ZEROED:.*]] = linalg.fill
// CHECK-SAME:       ins(%[[CST0]] : f16)
// CHECK-SAME:       outs(%[[PACKED_OUT_EMPTY]] : tensor<1x30x7x8x8x2x32x2xf16>)
// CHECK-SAME:       -> tensor<1x30x7x8x8x2x32x2xf16>

// ---- Loop nest 0 ----

// CHECK:     %[[RES0:.*]] = scf.for %[[I0:.*]] = %[[C0]] to %[[C192]] step %[[C64]] iter_args(%[[ACC0:.*]] = %[[PACKED_OUT_ZEROED]])
// CHECK-SAME:         -> (tensor<1x30x7x8x8x2x32x2xf16>) {
// CHECK:       %[[RES0_0:.*]] = scf.for %[[J0:.*]] = %[[C0]] to %[[C224]] step %[[C32]] iter_args(%[[ACC1:.*]] = %[[ACC0]])
// CHECK-SAME:           -> (tensor<1x30x7x8x8x2x32x2xf16>) {
// CHECK:         %[[H_OFF:.*]] = arith.floordivsi %[[J0]], %[[C8]] : index
// CHECK:         %[[IN_SLICE0:.*]] = tensor.extract_slice %[[PACKED_IN:.*]][0, %[[H_OFF]], 0, 0, 0, 0, 0, 0]
// CHECK-SAME:             [1, 4, 7, 4, 8, 2, 32, 2] [1, 1, 1, 1, 1, 1, 1, 1]
// CHECK-SAME:             : tensor<1x30x7x4x8x2x32x2xf16> to tensor<1x4x7x4x8x2x32x2xf16>

// CHECK:         %[[C_OFF:.*]] = arith.floordivsi %[[I0]], %[[C32]] : index
// CHECK:         %[[W_SLICE0:.*]] = tensor.extract_slice %[[PACKED_W:.*]][%[[C_OFF]], 0, 0, 0, 0, 0, 0]
// CHECK-SAME:             [2, 4, 1, 1, 16, 32, 2] [1, 1, 1, 1, 1, 1, 1]
// CHECK-SAME:             : tensor<8x4x1x1x16x32x2xf16> to tensor<2x4x1x1x16x32x2xf16>

// CHECK:         %[[UNPACKED_IN_SLICE0:.*]] = builtin.unrealized_conversion_cast %[[IN_SLICE0]]
// CHECK-SAME:             : tensor<1x4x7x4x8x2x32x2xf16> to tensor<1x32x27x123xf16>
// CHECK:         %[[UNPACKED_W_SLICE0:.*]] = builtin.unrealized_conversion_cast %[[W_SLICE0]]
// CHECK-SAME:             : tensor<2x4x1x1x16x32x2xf16> to tensor<64x1x1x123xf16>

// CHECK:         %[[OUT_SLICE0_EMPTY:.*]] = tensor.empty() : tensor<1x32x27x64xf16>
// CHECK:         %[[OUT_SLICE0_ZEROED:.*]] = linalg.fill
// CHECK-SAME:             ins(%[[CST0]] : f16) outs(%[[OUT_SLICE0_EMPTY]] : tensor<1x32x27x64xf16>)
// CHECK-SAME:             -> tensor<1x32x27x64xf16>

// CHECK:         %[[OUT_SLICE0:.*]] = linalg.conv_2d_nhwc_fhwc
// CHECK-SAME:             ins(%[[UNPACKED_IN_SLICE0]], %[[UNPACKED_W_SLICE0]]
// CHECK-SAME:                 : tensor<1x32x27x123xf16>, tensor<64x1x1x123xf16>)
// CHECK-SAME:             outs(%[[OUT_SLICE0_ZEROED]] : tensor<1x32x27x64xf16>)
// CHECK-SAME:             -> tensor<1x32x27x64xf16>

// CHECK:         %[[OUT_SLICE0_PACKED:.*]] = builtin.unrealized_conversion_cast %[[OUT_SLICE0]]
// CHECK-SAME:             : tensor<1x32x27x64xf16> to tensor<1x4x7x2x8x2x32x2xf16>

// CHECK:         %[[OUT_PARTIAL:.*]] = tensor.insert_slice %[[OUT_SLICE0_PACKED]] into %[[ACC1]][0, %[[H_OFF]], 0, %[[C_OFF]], 0, 0, 0, 0]
// CHECK-SAME:             [1, 4, 7, 2, 8, 2, 32, 2] [1, 1, 1, 1, 1, 1, 1, 1]
// CHECK-SAME:             : tensor<1x4x7x2x8x2x32x2xf16> into tensor<1x30x7x8x8x2x32x2xf16>

// CHECK:         scf.yield %[[OUT_PARTIAL]] : tensor<1x30x7x8x8x2x32x2xf16>
// CHECK:       }
// CHECK:       scf.yield %[[RES0_0]] : tensor<1x30x7x8x8x2x32x2xf16>
// CHECK:     }

// ---- Second outer loop (handles height tail 15) ----

// CHECK:     %[[RES1:.*]] = scf.for %[[I1:.*]] = %[[C0]] to %[[C192]] step %[[C64]] iter_args(%[[ACC2:.*]] = %[[RES0]])
// CHECK-SAME:         -> (tensor<1x30x7x8x8x2x32x2xf16>) {
// CHECK:       %[[IN_SLICE1:.*]] = tensor.extract_slice %[[PACKED_IN]][0, 28, 0, 0, 0, 0, 0, 0]
// CHECK-SAME:           [1, 2, 7, 4, 8, 2, 32, 2] [1, 1, 1, 1, 1, 1, 1, 1]
// CHECK-SAME:           : tensor<1x30x7x4x8x2x32x2xf16> to tensor<1x2x7x4x8x2x32x2xf16>

// CHECK:       %[[C_OFF:.*]] = arith.floordivsi %[[I1]], %[[C32]] : index
// CHECK:       %[[W_SLICE1:.*]] = tensor.extract_slice %[[PACKED_W]][%[[C_OFF]], 0, 0, 0, 0, 0, 0]
// CHECK-SAME:           [2, 4, 1, 1, 16, 32, 2] [1, 1, 1, 1, 1, 1, 1]
// CHECK-SAME:           : tensor<8x4x1x1x16x32x2xf16> to tensor<2x4x1x1x16x32x2xf16>

// CHECK:       %[[UNPACKED_IN_SLICE1:.*]] = builtin.unrealized_conversion_cast %[[IN_SLICE1]]
// CHECK-SAME:           : tensor<1x2x7x4x8x2x32x2xf16> to tensor<1x15x27x123xf16>
// CHECK:       %[[UNPACKED_W_SLICE1:.*]] = builtin.unrealized_conversion_cast %[[W_SLICE1]]
// CHECK-SAME:           : tensor<2x4x1x1x16x32x2xf16> to tensor<64x1x1x123xf16>

// CHECK:       %[[OUT_SLICE1_EMPTY:.*]] = tensor.empty() : tensor<1x15x27x64xf16>
// CHECK:       %[[OUT_SLICE1_ZEROED:.*]] = linalg.fill
// CHECK-SAME:           ins(%[[CST0]] : f16) outs(%[[OUT_SLICE1_EMPTY]] : tensor<1x15x27x64xf16>)
// CHECK-SAME:           -> tensor<1x15x27x64xf16>

// CHECK:       %[[OUT_SLICE1:.*]] = linalg.conv_2d_nhwc_fhwc
// CHECK-SAME:           ins(%[[UNPACKED_IN_SLICE1]], %[[UNPACKED_W_SLICE1]]
// CHECK-SAME:               : tensor<1x15x27x123xf16>, tensor<64x1x1x123xf16>)
// CHECK-SAME:           outs(%[[OUT_SLICE1_ZEROED]] : tensor<1x15x27x64xf16>)
// CHECK-SAME:           -> tensor<1x15x27x64xf16>

// CHECK:       %[[OUT_SLICE1_PACKED:.*]] = builtin.unrealized_conversion_cast %[[OUT_SLICE1]]
// CHECK-SAME:           : tensor<1x15x27x64xf16> to tensor<1x2x7x2x8x2x32x2xf16>

// CHECK:       %[[PARTIAL_OUT:.*]] = tensor.insert_slice %[[OUT_SLICE1_PACKED]] into %[[ACC2]][0, 28, 0, %[[C_OFF]], 0, 0, 0, 0]
// CHECK-SAME:           [1, 2, 7, 2, 8, 2, 32, 2] [1, 1, 1, 1, 1, 1, 1, 1]
// CHECK-SAME:           : tensor<1x2x7x2x8x2x32x2xf16> into tensor<1x30x7x8x8x2x32x2xf16>

// CHECK:       scf.yield %[[PARTIAL_OUT]] : tensor<1x30x7x8x8x2x32x2xf16>
// CHECK:     }

// ---- Third loop (handles channel tail with 63) ----

// CHECK:     %[[RES2:.*]] = scf.for %[[J2:.*]] = %[[C0]] to %[[C224]] step %[[C32]] iter_args(%[[ACC3:.*]] = %[[RES1]])
// CHECK-SAME:         -> (tensor<1x30x7x8x8x2x32x2xf16>) {
// CHECK:       %[[H_OFF:.*]] = arith.floordivsi %[[J2]], %[[C8]] : index
// CHECK:       %[[IN_SLICE2:.*]] = tensor.extract_slice %[[PACKED_IN]][0, %[[H_OFF]], 0, 0, 0, 0, 0, 0]
// CHECK-SAME:           [1, 4, 7, 4, 8, 2, 32, 2] [1, 1, 1, 1, 1, 1, 1, 1]
// CHECK-SAME:           : tensor<1x30x7x4x8x2x32x2xf16> to tensor<1x4x7x4x8x2x32x2xf16>

// CHECK:       %[[W_SLICE2:.*]] = tensor.extract_slice %[[PACKED_W]][6, 0, 0, 0, 0, 0, 0]
// CHECK-SAME:           [2, 4, 1, 1, 16, 32, 2] [1, 1, 1, 1, 1, 1, 1]
// CHECK-SAME:           : tensor<8x4x1x1x16x32x2xf16> to tensor<2x4x1x1x16x32x2xf16>

// CHECK:       %[[PACKED_IN_SLICE2:.*]] = builtin.unrealized_conversion_cast %[[IN_SLICE2]]
// CHECK-SAME:           : tensor<1x4x7x4x8x2x32x2xf16> to tensor<1x32x27x123xf16>
// CHECK:       %[[PACKED_W_SLICE2:.*]] = builtin.unrealized_conversion_cast %[[W_SLICE2]]
// CHECK-SAME:           : tensor<2x4x1x1x16x32x2xf16> to tensor<63x1x1x123xf16>

// CHECK:       %[[OUT_SLICE2_EMPTY:.*]] = tensor.empty() : tensor<1x32x27x63xf16>
// CHECK:       %[[OUT_SLICE2_ZEROED:.*]] = linalg.fill
// CHECK-SAME:           ins(%[[CST0]] : f16) outs(%[[OUT_SLICE2_EMPTY]] : tensor<1x32x27x63xf16>)
// CHECK-SAME:           -> tensor<1x32x27x63xf16>

// CHECK:       %[[OUT_SLICE2:.*]] = linalg.conv_2d_nhwc_fhwc
// CHECK-SAME:           ins(%[[PACKED_IN_SLICE2]], %[[PACKED_W_SLICE2]]
// CHECK-SAME:               : tensor<1x32x27x123xf16>, tensor<63x1x1x123xf16>)
// CHECK-SAME:           outs(%[[OUT_SLICE2_ZEROED]] : tensor<1x32x27x63xf16>)
// CHECK-SAME:           -> tensor<1x32x27x63xf16>

// CHECK:       %[[OUT_SLICE2_PACKED:.*]] = builtin.unrealized_conversion_cast %[[OUT_SLICE2]]
// CHECK-SAME:           : tensor<1x32x27x63xf16> to tensor<1x4x7x2x8x2x32x2xf16>

// CHECK:       %[[OUT_PARTIAL:.*]] = tensor.insert_slice %[[OUT_SLICE2_PACKED]] into %[[ACC3]][0, %[[H_OFF]], 0, 6, 0, 0, 0, 0]
// CHECK-SAME:           [1, 4, 7, 2, 8, 2, 32, 2] [1, 1, 1, 1, 1, 1, 1, 1]
// CHECK-SAME:           : tensor<1x4x7x2x8x2x32x2xf16> into tensor<1x30x7x8x8x2x32x2xf16>

// CHECK:       scf.yield %[[OUT_PARTIAL]] : tensor<1x30x7x8x8x2x32x2xf16>
// CHECK:     }

// ---- Final scalar tail computation (both height and channel tail) ----

// CHECK:     %[[IN_SLICE3:.*]] = tensor.extract_slice %[[PACKED_IN]][0, 28, 0, 0, 0, 0, 0, 0]
// CHECK-SAME:         [1, 2, 7, 4, 8, 2, 32, 2] [1, 1, 1, 1, 1, 1, 1, 1]
// CHECK-SAME:         : tensor<1x30x7x4x8x2x32x2xf16> to tensor<1x2x7x4x8x2x32x2xf16>

// CHECK:     %[[W_SLICE3:.*]] = tensor.extract_slice %[[PACKED_W]][6, 0, 0, 0, 0, 0, 0]
// CHECK-SAME:         [2, 4, 1, 1, 16, 32, 2] [1, 1, 1, 1, 1, 1, 1]
// CHECK-SAME:         : tensor<8x4x1x1x16x32x2xf16> to tensor<2x4x1x1x16x32x2xf16>

// CHECK:     %[[PACKED_IN_SLICE3:.*]] = builtin.unrealized_conversion_cast %[[IN_SLICE3]]
// CHECK-SAME:         : tensor<1x2x7x4x8x2x32x2xf16> to tensor<1x15x27x123xf16>
// CHECK:     %[[PACKED_W_SLICE3:.*]] = builtin.unrealized_conversion_cast %[[W_SLICE3]]
// CHECK-SAME:         : tensor<2x4x1x1x16x32x2xf16> to tensor<63x1x1x123xf16>

// CHECK:     %[[OUT_SLICE3_EMPTY:.*]] = tensor.empty() : tensor<1x15x27x63xf16>
// CHECK:     %[[OUT_SLICE3_ZEROED:.*]] = linalg.fill
// CHECK-SAME:         ins(%[[CST0]] : f16) outs(%[[OUT_SLICE3_EMPTY]] : tensor<1x15x27x63xf16>)
// CHECK-SAME:         -> tensor<1x15x27x63xf16>

// CHECK:     %[[OUT_SLICE3:.*]] = linalg.conv_2d_nhwc_fhwc
// CHECK-SAME:         ins(%[[PACKED_IN_SLICE3]], %[[PACKED_W_SLICE3]]
// CHECK-SAME:             : tensor<1x15x27x123xf16>, tensor<63x1x1x123xf16>)
// CHECK-SAME:         outs(%[[OUT_SLICE3_ZEROED]] : tensor<1x15x27x63xf16>)
// CHECK-SAME:         -> tensor<1x15x27x63xf16>

// CHECK:     %[[OUT_SLICE3_PACKED:.*]] = builtin.unrealized_conversion_cast %[[OUT_SLICE3]]
// CHECK-SAME:         : tensor<1x15x27x63xf16> to tensor<1x2x7x2x8x2x32x2xf16>

// CHECK:     %[[OUT_PACKED:.*]] = tensor.insert_slice %[[OUT_SLICE3_PACKED]] into %[[RES2]][0, 28, 0, 6, 0, 0, 0, 0]
// CHECK-SAME:         [1, 2, 7, 2, 8, 2, 32, 2] [1, 1, 1, 1, 1, 1, 1, 1]
// CHECK-SAME:         : tensor<1x2x7x2x8x2x32x2xf16> into tensor<1x30x7x8x8x2x32x2xf16>

// ---- Unpack back to original layout ----

// CHECK:     %[[UNPACK_DST0:.*]] = tensor.empty() : tensor<1x30x7x8x8x4x32xf16>
// CHECK:     %[[UNPACK0:.*]] = linalg.unpack %[[OUT_PACKED]]
// CHECK-SAME:         inner_dims_pos = [5]
// CHECK-SAME:         inner_tiles = [2]
// CHECK-SAME:         into %[[UNPACK_DST0]] : tensor<1x30x7x8x8x2x32x2xf16> -> tensor<1x30x7x8x8x4x32xf16>

// CHECK:     %[[UNPACK1:.*]] = linalg.unpack %[[UNPACK0]]
// CHECK-SAME:         inner_dims_pos = [1, 2, 3]
// CHECK-SAME:         inner_tiles = [8, 4, 32]
// CHECK-SAME:         into %[[OUT_TENSOR]] : tensor<1x30x7x8x8x4x32xf16> -> tensor<1x239x27x255xf16>

// CHECK:     bufferization.materialize_in_destination %[[UNPACK1]] in restrict writable %arg2
// CHECK-SAME:         : (tensor<1x239x27x255xf16>, memref<1x239x27x255xf16>) -> ()
// CHECK:     return
// CHECK:   }
// CHECK: }

module {
  func.func @multi_dim_tiling(%arg0: memref<1x239x27x123xf16>, %arg1: memref<255x1x1x123xf16>, %arg2: memref<1x239x27x255xf16>) {
    %cst = arith.constant dense<[8, 32, 1, 1, 4, 16, 2]> : tensor<7xindex>
    %cst_0 = arith.constant 0.000000e+00 : f16
    %c0 = arith.constant 0 : index
    %c32 = arith.constant 32 : index
    %c64 = arith.constant 64 : index
    %c224 = arith.constant 224 : index
    %c192 = arith.constant 192 : index
    %0 = bufferization.to_tensor %arg0 restrict writable : memref<1x239x27x123xf16> to tensor<1x239x27x123xf16>
    %1 = bufferization.to_tensor %arg1 restrict writable : memref<255x1x1x123xf16> to tensor<255x1x1x123xf16>
    %2 = tensor.empty() : tensor<1x239x27x255xf16>
    %3 = tensor.empty() : tensor<1x30x7x4x8x4x32xf16>
    %pack = linalg.pack %0 padding_value(%cst_0 : f16) inner_dims_pos = [1, 2, 3] inner_tiles = [8, 4, 32] into %3 : tensor<1x239x27x123xf16> -> tensor<1x30x7x4x8x4x32xf16>
    %4 = tensor.empty() : tensor<1x30x7x4x8x2x32x2xf16>
    %pack_1 = linalg.pack %pack inner_dims_pos = [5] inner_tiles = [2] into %4 : tensor<1x30x7x4x8x4x32xf16> -> tensor<1x30x7x4x8x2x32x2xf16>
    %5 = builtin.unrealized_conversion_cast %pack_1 : tensor<1x30x7x4x8x2x32x2xf16> to tensor<1x239x27x123xf16>
    %padded = tensor.pad %1 low[0, 0, 0, 0] high[1, 0, 0, 5] {
    ^bb0(%arg3: index, %arg4: index, %arg5: index, %arg6: index):
      tensor.yield %cst_0 : f16
    } : tensor<255x1x1x123xf16> to tensor<256x1x1x128xf16>
    %reshape = tensor.reshape %padded(%cst) : (tensor<256x1x1x128xf16>, tensor<7xindex>) -> tensor<8x32x1x1x4x16x2xf16>
    %6 = tensor.empty() : tensor<8x4x1x1x16x32x2xf16>
    %transposed = linalg.transpose ins(%reshape : tensor<8x32x1x1x4x16x2xf16>) outs(%6 : tensor<8x4x1x1x16x32x2xf16>) permutation = [0, 4, 2, 3, 5, 1, 6] 
    %7 = builtin.unrealized_conversion_cast %transposed : tensor<8x4x1x1x16x32x2xf16> to tensor<255x1x1x123xf16>
    %8 = scf.for %arg3 = %c0 to %c192 step %c64 iter_args(%arg4 = %2) -> (tensor<1x239x27x255xf16>) {
      %16 = scf.for %arg5 = %c0 to %c224 step %c32 iter_args(%arg6 = %arg4) -> (tensor<1x239x27x255xf16>) {
        %extracted_slice_4 = tensor.extract_slice %5[0, %arg5, 0, 0] [1, 32, 27, 123] [1, 1, 1, 1] : tensor<1x239x27x123xf16> to tensor<1x32x27x123xf16>
        %extracted_slice_5 = tensor.extract_slice %7[%arg3, 0, 0, 0] [64, 1, 1, 123] [1, 1, 1, 1] : tensor<255x1x1x123xf16> to tensor<64x1x1x123xf16>
        %17 = tensor.empty() : tensor<1x32x27x64xf16>
        %18 = linalg.fill ins(%cst_0 : f16) outs(%17 : tensor<1x32x27x64xf16>) -> tensor<1x32x27x64xf16>
        %19 = linalg.conv_2d_nhwc_fhwc {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%extracted_slice_4, %extracted_slice_5 : tensor<1x32x27x123xf16>, tensor<64x1x1x123xf16>) outs(%18 : tensor<1x32x27x64xf16>) -> tensor<1x32x27x64xf16>
        %inserted_slice_6 = tensor.insert_slice %19 into %arg6[0, %arg5, 0, %arg3] [1, 32, 27, 64] [1, 1, 1, 1] : tensor<1x32x27x64xf16> into tensor<1x239x27x255xf16>
        scf.yield %inserted_slice_6 : tensor<1x239x27x255xf16>
      }
      scf.yield %16 : tensor<1x239x27x255xf16>
    }
    %out0_packed = builtin.unrealized_conversion_cast %8 : tensor<1x239x27x255xf16> to tensor<1x30x7x8x8x2x32x2xf16>
    %out0_unpacked = builtin.unrealized_conversion_cast %out0_packed : tensor<1x30x7x8x8x2x32x2xf16> to tensor<1x239x27x255xf16>
    %9 = scf.for %arg3 = %c0 to %c192 step %c64 iter_args(%arg4 = %out0_unpacked) -> (tensor<1x239x27x255xf16>) {
      %extracted_slice_4 = tensor.extract_slice %5[0, 224, 0, 0] [1, 15, 27, 123] [1, 1, 1, 1] : tensor<1x239x27x123xf16> to tensor<1x15x27x123xf16>
      %extracted_slice_5 = tensor.extract_slice %7[%arg3, 0, 0, 0] [64, 1, 1, 123] [1, 1, 1, 1] : tensor<255x1x1x123xf16> to tensor<64x1x1x123xf16>
      %16 = tensor.empty() : tensor<1x15x27x64xf16>
      %17 = linalg.fill ins(%cst_0 : f16) outs(%16 : tensor<1x15x27x64xf16>) -> tensor<1x15x27x64xf16>
      %18 = linalg.conv_2d_nhwc_fhwc {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%extracted_slice_4, %extracted_slice_5 : tensor<1x15x27x123xf16>, tensor<64x1x1x123xf16>) outs(%17 : tensor<1x15x27x64xf16>) -> tensor<1x15x27x64xf16>
      %inserted_slice_6 = tensor.insert_slice %18 into %arg4[0, 224, 0, %arg3] [1, 15, 27, 64] [1, 1, 1, 1] : tensor<1x15x27x64xf16> into tensor<1x239x27x255xf16>
      scf.yield %inserted_slice_6 : tensor<1x239x27x255xf16>
    }
    %out1_packed = builtin.unrealized_conversion_cast %9 : tensor<1x239x27x255xf16> to tensor<1x30x7x8x8x2x32x2xf16>
    %out1_unpacked = builtin.unrealized_conversion_cast %out1_packed : tensor<1x30x7x8x8x2x32x2xf16> to tensor<1x239x27x255xf16>
    %10 = scf.for %arg3 = %c0 to %c224 step %c32 iter_args(%arg4 = %out1_unpacked) -> (tensor<1x239x27x255xf16>) {
      %extracted_slice_4 = tensor.extract_slice %5[0, %arg3, 0, 0] [1, 32, 27, 123] [1, 1, 1, 1] : tensor<1x239x27x123xf16> to tensor<1x32x27x123xf16>
      %extracted_slice_5 = tensor.extract_slice %7[192, 0, 0, 0] [63, 1, 1, 123] [1, 1, 1, 1] : tensor<255x1x1x123xf16> to tensor<63x1x1x123xf16>
      %16 = tensor.empty() : tensor<1x32x27x63xf16>
      %17 = linalg.fill ins(%cst_0 : f16) outs(%16 : tensor<1x32x27x63xf16>) -> tensor<1x32x27x63xf16>
      %18 = linalg.conv_2d_nhwc_fhwc {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%extracted_slice_4, %extracted_slice_5 : tensor<1x32x27x123xf16>, tensor<63x1x1x123xf16>) outs(%17 : tensor<1x32x27x63xf16>) -> tensor<1x32x27x63xf16>
      %inserted_slice_6 = tensor.insert_slice %18 into %arg4[0, %arg3, 0, 192] [1, 32, 27, 63] [1, 1, 1, 1] : tensor<1x32x27x63xf16> into tensor<1x239x27x255xf16>
      scf.yield %inserted_slice_6 : tensor<1x239x27x255xf16>
    }
    %out2_packed = builtin.unrealized_conversion_cast %10 : tensor<1x239x27x255xf16> to tensor<1x30x7x8x8x2x32x2xf16>
    %out2_unpacked = builtin.unrealized_conversion_cast %out2_packed : tensor<1x30x7x8x8x2x32x2xf16> to tensor<1x239x27x255xf16>
    %extracted_slice = tensor.extract_slice %5[0, 224, 0, 0] [1, 15, 27, 123] [1, 1, 1, 1] : tensor<1x239x27x123xf16> to tensor<1x15x27x123xf16>
    %extracted_slice_2 = tensor.extract_slice %7[192, 0, 0, 0] [63, 1, 1, 123] [1, 1, 1, 1] : tensor<255x1x1x123xf16> to tensor<63x1x1x123xf16>
    %11 = tensor.empty() : tensor<1x15x27x63xf16>
    %12 = linalg.fill ins(%cst_0 : f16) outs(%11 : tensor<1x15x27x63xf16>) -> tensor<1x15x27x63xf16>
    %13 = linalg.conv_2d_nhwc_fhwc {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%extracted_slice, %extracted_slice_2 : tensor<1x15x27x123xf16>, tensor<63x1x1x123xf16>) outs(%12 : tensor<1x15x27x63xf16>) -> tensor<1x15x27x63xf16>
    %inserted_slice = tensor.insert_slice %13 into %out2_unpacked[0, 224, 0, 192] [1, 15, 27, 63] [1, 1, 1, 1] : tensor<1x15x27x63xf16> into tensor<1x239x27x255xf16>
    %14 = builtin.unrealized_conversion_cast %inserted_slice : tensor<1x239x27x255xf16> to tensor<1x30x7x8x8x2x32x2xf16>
    %15 = tensor.empty() : tensor<1x30x7x8x8x4x32xf16>
    %unpack = linalg.unpack %14 inner_dims_pos = [5] inner_tiles = [2] into %15 : tensor<1x30x7x8x8x2x32x2xf16> -> tensor<1x30x7x8x8x4x32xf16>
    %unpack_3 = linalg.unpack %unpack inner_dims_pos = [1, 2, 3] inner_tiles = [8, 4, 32] into %2 : tensor<1x30x7x8x8x4x32xf16> -> tensor<1x239x27x255xf16>
    bufferization.materialize_in_destination %unpack_3 in restrict writable %arg2 : (tensor<1x239x27x255xf16>, memref<1x239x27x255xf16>) -> ()
    return
  }
}

