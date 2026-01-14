// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(func.func(preprocess-tiled-conv2d))' | FileCheck %s
// CHECK: func.func @conv2d_tile_h8
// CHECK: %[[C0:.+]] = arith.constant 0 : index
// CHECK: %[[C64:.+]] = arith.constant 64 : index
// CHECK: %[[C32:.+]] = arith.constant 32 : index
// CHECK: %[[OUTPUT_EMPTY:.+]] = tensor.empty() : tensor<1x2x8x2x8x2x32x2xf16>
// CHECK: %[[C0_FLOAT:.+]] = arith.constant 0.000000e+00 : f16
// CHECK: %[[OUTPUT_INITED:.+]] = linalg.fill ins(%[[C0_FLOAT]] : f16) outs(%[[OUTPUT_EMPTY]] : tensor<1x2x8x2x8x2x32x2xf16>) -> tensor<1x2x8x2x8x2x32x2xf16>
// CHECK: %[[NEW_LOOP_RES:.+]] = scf.for %[[IND_VAR:.+]] = %[[C0]] to %[[C64]] step %[[C32]] iter_args(%[[CARRIED:.+]] = %[[OUTPUT_INITED]]) -> (tensor<1x2x8x2x8x2x32x2xf16>) {
// CHECK: %[[C0_4:.+]] = arith.constant 0 : index
// CHECK: %[[C0_5:.+]] = arith.constant 0 : index
// CHECK: %[[C8_6:.+]] = arith.constant 8 : index
// CHECK: %[[SLICE_OFF_IMG_1:.+]] = arith.floordivsi %c0_5, %c8_6 : index
// CHECK: %[[C0_7:.+]] = arith.constant 0 : index
// CHECK: %[[C4_8:.+]] = arith.constant 4 : index
// CHECK: %[[SLICE_OFF_IMG_2:.+]] = arith.floordivsi %c0_7, %c4_8 : index
// CHECK: %[[C0_9:.+]] = arith.constant 0 : index
// CHECK: %[[C32_10:.+]] = arith.constant 32 : index
// CHECK: %[[SLICE_OFF_IMG_3:.+]] = arith.floordivsi %[[C0_9]], %[[C32_10]] : index
// CHECK: %[[C0_11:.+]] = arith.constant 0 : index
// CHECK: %[[C1_12:.+]] = arith.constant 1 : index
// CHECK: %[[SLICE_IMG:.+]] = tensor.extract_slice %[[_:.+]][%[[C0_4]], %[[SLICE_OFF_IMG_1]], %[[SLICE_OFF_IMG_2]], %[[SLICE_OFF_IMG_3]], %[[C0_11]], %[[C0_11]], %[[C0_11]], %[[C0_11]]] [1, 2, 8, 4, 8, 2, 32, 2] [1, 1, 1, 1, 1, 1, 1, 1] : tensor<1x2x8x4x8x2x32x2xf16> to tensor<1x2x8x4x8x2x32x2xf16>
// CHECK: %[[C32_13:.+]] = arith.constant 32 : index
// CHECK: %[[SLICE_OFF_FILTER_0:.+]] = arith.floordivsi %[[IND_VAR]], %c32_13 : index
// CHECK: %[[C0_14:.+]] = arith.constant 0 : index
// CHECK: %[[C32_15:.+]] = arith.constant 32 : index
// CHECK: %[[SLICE_OFF_FILTER_1:.+]] = arith.floordivsi %[[C0_14]], %[[C32_15]] : index
// CHECK: %[[C0_16:.+]] = arith.constant 0 : index
// CHECK: %[[C0_17:.+]] = arith.constant 0 : index
// CHECK: %[[C0_18:.+]] = arith.constant 0 : index
// CHECK: %[[C1_19:.+]] = arith.constant 1 : index
// CHECK: %[[FILTER_SLICE:.+]] = tensor.extract_slice %transposed[%[[SLICE_OFF_FILTER_0]], %[[SLICE_OFF_FILTER_1]], %[[C0_16]], %[[C0_17]], %[[C0_18]], %[[C0_18]], %[[C0_18]]] [1, 4, 1, 1, 16, 32, 2] [1, 1, 1, 1, 1, 1, 1] : tensor<2x4x1x1x16x32x2xf16> to tensor<1x4x1x1x16x32x2xf16>
// CHECK: %[[INV_PACK_IMAGE_SLICE:.+]] = builtin.unrealized_conversion_cast %[[SLICE_IMG:.+]] : tensor<1x2x8x4x8x2x32x2xf16> to tensor<1x16x32x128xf16>
// CHECK: %[[INV_PACK_FILTER_SLICE:.+]] = builtin.unrealized_conversion_cast %[[SLICE_FILTER:.+]] : tensor<1x4x1x1x16x32x2xf16> to tensor<32x1x1x128xf16>
// CHECK: %[[TENSOR_OUT_SLICE_EMPTY:.+]] = tensor.empty() : tensor<1x16x32x32xf16>
// CHECK: %[[ZERO_F16_AGAIN:.+]] = arith.constant 0.000000e+00 : f16
// CHECK: %[[TENSOR_OUT_SLICE_INITTED:.+]] = linalg.fill ins(%[[ZERO_F16_AGAIN]] : f16) outs(%[[TENSOR_OUT_SLICE_EMPTY]] : tensor<1x16x32x32xf16>) -> tensor<1x16x32x32xf16>
// CHECK: %[[CONV_OUT_SLICE:.+]] = linalg.conv_2d_nhwc_fhwc ins(%[[INV_PACK_IMAGE_SLICE]], %[[INV_PACK_FILTER_SLICE]] : tensor<1x16x32x128xf16>, tensor<32x1x1x128xf16>) outs(%[[TENSOR_OUT_SLICE_INITTED]] : tensor<1x16x32x32xf16>) -> tensor<1x16x32x32xf16>
// CHECK: %[[INV_UNPACK_SLICE_OUT:.+]] = builtin.unrealized_conversion_cast %[[CONV_OUT_SLICE]] : tensor<1x16x32x32xf16> to tensor<1x2x8x1x8x2x32x2xf16>
// CHECK: %[[C0_22:.+]] = arith.constant 0 : index
// CHECK: %[[C0_23:.+]] = arith.constant 0 : index
// CHECK: %[[C8_24:.+]] = arith.constant 8 : index
// CHECK: %[[OUT_SLICE_OFF_1:.+]] = arith.floordivsi %c0_23, %c8_24 : index
// CHECK: %[[C0_25:.+]] = arith.constant 0 : index
// CHECK: %[[C4_26:.+]] = arith.constant 4 : index
// CHECK: %[[OUT_SLICE_OFF_2:.+]] = arith.floordivsi %c0_25, %c4_26 : index
// CHECK: %[[C32_27:.+]] = arith.constant 32 : index
// CHECK: %[[OUT_SLICE_OFF_3:.+]] = arith.floordivsi %[[IND_VAR]], %c32_27 : index
// CHECK: %[[C0_28:.+]] = arith.constant 0 : index
// CHECK: %[[INSERTED_SLICE:.+]] = tensor.insert_slice %[[INV_UNPACK_SLICE_OUT]] into %[[CARRIED]][%[[C0_22]], %[[OUT_SLICE_OFF_1]], %[[OUT_SLICE_OFF_2]], %[[OUT_SLICE_OFF_3]], %[[C0_28]], %[[C0_28]], %[[C0_28]], %[[C0_28]]] [1, 2, 8, 1, 8, 2, 32, 2] [1, 1, 1, 1, 1, 1, 1, 1] : tensor<1x2x8x1x8x2x32x2xf16> into tensor<1x2x8x2x8x2x32x2xf16>
// CHECK: scf.yield %[[INSERTED_SLICE:.+]] : tensor<1x2x8x2x8x2x32x2xf16>
// CHECK: }
// CHECK: %[[TEMP_UNPACKED_OUT:.+]] = linalg.unpack %[[NEW_LOOP_RES]] inner_dims_pos = [5] inner_tiles = [2] into %[[_:.+]] : tensor<1x2x8x2x8x2x32x2xf16> -> tensor<1x2x8x2x8x4x32xf16>
// CHECK: %[[UNPACKED_OUT:.+]] = linalg.unpack %[[TEMP_UNPACKED_OUT]] inner_dims_pos = [1, 2, 3] inner_tiles = [8, 4, 32] into %[[_:.+]] : tensor<1x2x8x2x8x4x32xf16> -> tensor<1x16x32x64xf16>
// CHECK: return %[[UNPACKED_OUT]] : tensor<1x16x32x64xf16>
// CHECK: }
// CHECK: }

module {
  func.func @conv2d_tile_h8(
      %input: tensor<1x16x32x128xf16>,
      %filter: tensor<64x1x1x128xf16>
  ) -> tensor<1x16x32x64xf16> {
    %zero = arith.constant 0.0 : f16
    %c0   = arith.constant 0 : index
    %c8   = arith.constant 8 : index
    %c16  = arith.constant 16 : index
    %c64  = arith.constant 64 : index
    %c32  = arith.constant 32 : index
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %cst = arith.constant 0.000000e+00 : f16
 
    // Initialize full output
    %out_init = tensor.empty() : tensor<1x16x32x64xf16>

    %temp_packed_empty = tensor.empty() : tensor<1x2x8x4x8x4x32xf16>
    %temp_packed = linalg.pack %input padding_value(%cst : f16) inner_dims_pos = [1, 2, 3] inner_tiles = [8, 4, 32] into %temp_packed_empty : tensor<1x16x32x128xf16> -> tensor<1x2x8x4x8x4x32xf16>
    %0 = builtin.unrealized_conversion_cast %temp_packed : tensor<1x2x8x4x8x4x32xf16> to tensor<1x2x8x4x8x2x32x4xf16>
    %packed_empty = tensor.empty() : tensor<1x2x8x4x8x2x32x2xf16>
    %pack_0 = linalg.pack %temp_packed inner_dims_pos = [5] inner_tiles = [2] into %packed_empty : tensor<1x2x8x4x8x4x32xf16> -> tensor<1x2x8x4x8x2x32x2xf16>
    %3 = builtin.unrealized_conversion_cast %pack_0 : tensor<1x2x8x4x8x2x32x2xf16> to tensor<1x16x32x128xf16>

    %from_elements = tensor.from_elements %c2, %c32, %c1, %c1, %c4, %c16, %c2 : tensor<7xindex>
    %reshape = tensor.reshape %filter(%from_elements) : (tensor<64x1x1x128xf16>, tensor<7xindex>) -> tensor<2x32x1x1x4x16x2xf16>
    %4 = tensor.empty() : tensor<2x4x1x1x16x32x2xf16>
    %transposed = linalg.transpose ins(%reshape : tensor<2x32x1x1x4x16x2xf16>) outs(%4 : tensor<2x4x1x1x16x32x2xf16>) permutation = [0, 4, 2, 3, 5, 1, 6]
    %5 = builtin.unrealized_conversion_cast %transposed : tensor<2x4x1x1x16x32x2xf16> to tensor<64x1x1x128xf16>
    
    // Tile along channel (c) by 32
    %result = scf.for %c_start = %c0 to %c64 step %c32
              iter_args(%out_acc = %out_init)
              -> tensor<1x16x32x64xf16> {
      // Extract input and output tiles
      %in_tile = tensor.extract_slice %3
                  [0, 0, 0, 0] [1, 16, 32, 128] [1, 1, 1, 1]
                  : tensor<1x16x32x128xf16> to tensor<1x16x32x128xf16>

      // Extract input and output tiles

      %filter_tile = tensor.extract_slice %5
                  [%c_start, 0, 0, 0] [32, 1, 1, 128] [1, 1, 1, 1]
                  : tensor<64x1x1x128xf16> to tensor<32x1x1x128xf16>
 

       %out_tile_empty = tensor.empty(): tensor<1x16x32x32xf16>
       %out_tile = linalg.fill ins(%zero : f16)
                          outs(%out_tile_empty : tensor<1x16x32x32xf16>)
                -> tensor<1x16x32x32xf16>
 
      // Compute convolution on the tile
      %tile_result = linalg.conv_2d_nhwc_fhwc
                      {strides = dense<[1, 1]> : vector<2xi64>,
                       dilations = dense<[1, 1]> : vector<2xi64>}
                      ins(%in_tile, %filter_tile :
                          tensor<1x16x32x128xf16>, tensor<32x1x1x128xf16>)
                      outs(%out_tile : tensor<1x16x32x32xf16>)
                      -> tensor<1x16x32x32xf16>
 
      // Insert tile back into full output
      %out_next = tensor.insert_slice %tile_result into %out_acc
                   [0, 0, 0, %c_start] [1, 16, 32, 32] [1, 1, 1, 1]
                   : tensor<1x16x32x32xf16> into tensor<1x16x32x64xf16>
 
      scf.yield %out_next : tensor<1x16x32x64xf16>
    }

    %inverse_unpacked_out = builtin.unrealized_conversion_cast %result : tensor<1x16x32x64xf16> to tensor<1x2x8x2x8x2x32x2xf16>
    %unpacked_temp_out_empty = tensor.empty() : tensor<1x2x8x2x8x4x32xf16>
    %unpacked_temp_out = linalg.unpack %inverse_unpacked_out inner_dims_pos = [5] inner_tiles = [2] into %unpacked_temp_out_empty : tensor<1x2x8x2x8x2x32x2xf16> -> tensor<1x2x8x2x8x4x32xf16>
    %unpacked_empty = tensor.empty() : tensor<1x16x32x64xf16>
    %unpacked_out = linalg.unpack %unpacked_temp_out inner_dims_pos = [1, 2, 3] inner_tiles = [8, 4, 32] into %unpacked_empty : tensor<1x2x8x2x8x4x32xf16> -> tensor<1x16x32x64xf16>
    

 
    return %unpacked_out : tensor<1x16x32x64xf16>
  }
}
