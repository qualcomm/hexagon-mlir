// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(func.func(seed-layout-conversions))' | FileCheck %s

// CHECK: func.func @test1(%[[ARG0:.+]]: tensor<1x56x56x128xf32>) -> tensor<1x56x56x256xf32> {
// CHECK: %[[CST:.+]] = arith.constant dense_resource<tensor_1> : tensor<3x3x128x256xi8>
// CHECK: %[[CST0:.+]] = arith.constant dense_resource<tensor_2> : tensor<256xf32>
// CHECK: %[[QCAST0:.+]] = quant.qcast %[[CST0]] : tensor<256xf32> to tensor<256x!quant.uniform<i32:f32, 9.9089999999999994E-6>>
// CHECK: %[[SCAST0:.+]] = quant.scast %[[QCAST0]] : tensor<256x!quant.uniform<i32:f32, 9.9089999999999994E-6>> to tensor<256xi32>
// CHECK: %[[QCAST1:.+]] = quant.qcast %[[ARG0]] : tensor<1x56x56x128xf32> to tensor<1x56x56x128x!quant.uniform<i8:f32, 0.0051725930534303188>>
// CHECK: %[[SCAST1:.+]] = quant.scast %[[QCAST1]] : tensor<1x56x56x128x!quant.uniform<i8:f32, 0.0051725930534303188>> to tensor<1x56x56x128xi8>
// CHECK: %[[C0:.+]] = arith.constant 0 : i8
// CHECK: %[[PADDED:.+]] = tensor.pad %[[SCAST1]] low[0, 1, 1, 0] high[0, 1, 1, 0] {
// CHECK: ^bb0(%[[IDX1:.+]], %[[IDX2:.+]], %[[IDX3:.+]], %[[IDX4:.+]]):
// CHECK: tensor.yield %[[C0]] : i8
// CHECK: } : tensor<1x56x56x128xi8> to tensor<1x58x58x128xi8>
// CHECK: %[[C0_1:.+]] = arith.constant 0 : i8
// CHECK: %[[EMPTY:.+]] = tensor.empty() : tensor<1x56x56x256xi32>
// CHECK: %[[BROADCAST:.+]] = linalg.broadcast ins(%[[SCAST0]] : tensor<256xi32>) outs(%[[EMPTY]] : tensor<1x56x56x256xi32>) dimensions = [0, 1, 2]
// CHECK: %[[C0_2:.+]] = arith.constant 0 : i8
// CHECK: %[[PACKED:.+]] = linalg.pack %[[PADDED]] padding_value(%[[C0_2]] : i8) inner_dims_pos = [1, 2, 3] inner_tiles = [8, 8, 32] into [[PACKED_OUT:.*]] : tensor<1x58x58x128xi8> -> tensor<1x8x8x4x8x8x32xi8>
// CHECK: %[[CAST_PACKED:.+]] = builtin.unrealized_conversion_cast %[[PACKED]] : tensor<1x8x8x4x8x8x32xi8> to tensor<1x58x58x128xi8>
// CHECK: %[[CONV:.+]] = linalg.conv_2d_nhwc_hwcf_q ins(%[[CAST_PACKED]], %[[CST]], %[[C0]], %[[C0_1]] : tensor<1x58x58x128xi8>, tensor<3x3x128x256xi8>, i8, i8) outs([[CONV_OUT:.*]] : tensor<1x56x56x256xi32>) -> tensor<1x56x56x256xi32>
// CHECK: %[[SCAST2:.+]] = quant.scast %[[CONV]] : tensor<1x56x56x256xi32> to tensor<1x56x56x256x!quant.uniform<i32:f32, 9.9089999999999994E-6>>
// CHECK: %[[DCAST:.+]] = quant.dcast %[[SCAST2]] : tensor<1x56x56x256x!quant.uniform<i32:f32, 9.9089999999999994E-6>> to tensor<1x56x56x256xf32>
// CHECK: %[[QCAST2:.+]] = quant.qcast %[[DCAST]] : tensor<1x56x56x256xf32> to tensor<1x56x56x256x!quant.uniform<i8:f32, 0.018517576158046722:-135>>
// CHECK: %[[CAST_FINAL:.+]] = builtin.unrealized_conversion_cast %[[QCAST2]] : tensor<1x56x56x256x!quant.uniform<i8:f32, 0.018517576158046722:-135>> to tensor<1x7x7x8x8x8x32x!quant.uniform<i8:f32, 0.018517576158046722:-135>>
// CHECK: %[[UNPACK:.+]] = linalg.unpack %[[CAST_FINAL]] inner_dims_pos = [1, 2, 3] inner_tiles = [8, 8, 32] into [[UNPACKED:.*]] : tensor<1x7x7x8x8x8x32x!quant.uniform<i8:f32, 0.018517576158046722:-135>> -> tensor<1x56x56x256x!quant.uniform<i8:f32, 0.018517576158046722:-135>>
// CHECK: %[[FINAL_DCAST:.+]] = quant.dcast %[[UNPACK]] : tensor<1x56x56x256x!quant.uniform<i8:f32, 0.018517576158046722:-135>> to tensor<1x56x56x256xf32>
// CHECK: return %[[FINAL_DCAST]] : tensor<1x56x56x256xf32>

// Test case for stride=1 quantized int8 quantized int8 convolution
func.func @test1(%act: tensor<1x56x56x128xf32>) -> tensor<1x56x56x256xf32> {
  %wt = arith.constant dense_resource<tensor_1> : tensor<3x3x128x256xi8>
  %bias = arith.constant dense_resource<tensor_2> : tensor<256xf32>
  
  // Precomputed scale product
  // 0.0051725930534303188 * 0.0019158006180077791 = 9.90965696846455e-06
  %bias_qu = quant.qcast %bias : tensor<256xf32> to tensor<256x!quant.uniform<i32:f32, 0.000009909:0>>
  %bias_i32 = quant.scast %bias_qu : tensor<256x!quant.uniform<i32:f32, 0.000009909:0 >> to tensor<256xi32>
  %act_qu = quant.qcast %act : tensor<1x56x56x128xf32> to tensor<1x56x56x128x!quant.uniform<i8:f32, 0.0051725930534303188>>
  %act_i8 = quant.scast %act_qu : tensor<1x56x56x128x!quant.uniform<i8:f32, 0.0051725930534303188>> to tensor<1x56x56x128xi8>
  %act_zp = arith.constant 0 : i8
  %act_i8_padded = tensor.pad %act_i8 low[0, 1, 1, 0] high[0, 1, 1, 0] { // Conv2d specific padding
  ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
    tensor.yield %act_zp : i8
  } : tensor<1x56x56x128xi8> to tensor<1x58x58x128xi8>
  %wt_zp = arith.constant 0 : i8

  %init = tensor.empty() : tensor<1x56x56x256xi32> // Note the dtype for the output; it needs to be i32 as the bias is added separately before requantizing the results as per conv2d output quant params     
  %broadcasted_9 = linalg.broadcast ins(%bias_i32 : tensor<256xi32>) outs(%init : tensor<1x56x56x256xi32>) dimensions = [0, 1, 2] // Output is being initialized with bias values
  %conv = linalg.conv_2d_nhwc_hwcf_q  {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%act_i8_padded, %wt, %act_zp, %wt_zp : tensor<1x58x58x128xi8>, tensor<3x3x128x256xi8>, i8, i8) outs(%broadcasted_9: tensor<1x56x56x256xi32>) -> tensor<1x56x56x256xi32>
  %conv_output_q32 = quant.scast %conv : tensor<1x56x56x256xi32> to tensor<1x56x56x256x!quant.uniform<i32:f32, 0.000009909:0>> // zp = 0.0051725930534303188 * 0.0019158006180077791
  %dcasted = quant.dcast %conv_output_q32 : tensor<1x56x56x256x!quant.uniform<i32:f32, 0.000009909:0>> to tensor<1x56x56x256xf32>
  %qcasted = quant.qcast %dcasted : tensor<1x56x56x256xf32> to tensor<1x56x56x256x!quant.uniform<i8:f32, 0.018517576158046722:-135>>
    
  %out_f32 = quant.dcast %qcasted : tensor<1x56x56x256x!quant.uniform<i8:f32, 0.018517576158046722:-135>> to tensor<1x56x56x256xf32>
  return %out_f32 : tensor<1x56x56x256xf32>
}

// -----

// CHECK: func.func @test2(%[[ARG0:.+]]: tensor<1x56x56x128xf32>) -> tensor<1x28x28x256xf32> {
// CHECK: %[[CST:.+]] = arith.constant dense_resource<tensor_1> : tensor<3x3x128x256xi8>
// CHECK: %[[CST0:.+]] = arith.constant dense_resource<tensor_2> : tensor<256xf32>
// CHECK: %[[QCAST0:.+]] = quant.qcast %[[CST0]] : tensor<256xf32> to tensor<256x!quant.uniform<i32:f32, 9.9089999999999994E-6>>
// CHECK: %[[SCAST0:.+]] = quant.scast %[[QCAST0]] : tensor<256x!quant.uniform<i32:f32, 9.9089999999999994E-6>> to tensor<256xi32>
// CHECK: %[[QCAST1:.+]] = quant.qcast %[[ARG0]] : tensor<1x56x56x128xf32> to tensor<1x56x56x128x!quant.uniform<i8:f32, 0.0051725930534303188>>
// CHECK: %[[SCAST1:.+]] = quant.scast %[[QCAST1]] : tensor<1x56x56x128x!quant.uniform<i8:f32, 0.0051725930534303188>> to tensor<1x56x56x128xi8>
// CHECK: %[[C0:.+]] = arith.constant 0 : i8
// CHECK: %[[PADDED:.+]] = tensor.pad %[[SCAST1]] low[0, 0, 0, 0] high[0, 1, 1, 0] {
// CHECK: ^bb0(%[[IDX1:.+]], %[[IDX2:.+]], %[[IDX3:.+]], %[[IDX4:.+]]):
// CHECK: tensor.yield %[[C0]] : i8
// CHECK: } : tensor<1x56x56x128xi8> to tensor<1x57x57x128xi8>
// CHECK: %[[C0_1:.+]] = arith.constant 0 : i8
// CHECK: %[[EMPTY:.+]] = tensor.empty() : tensor<1x28x28x256xi32>
// CHECK: %[[BROADCAST:.+]] = linalg.broadcast ins(%[[SCAST0]] : tensor<256xi32>) outs(%[[EMPTY]] : tensor<1x28x28x256xi32>) dimensions = [0, 1, 2]
// CHECK: %[[CONV:.+]] = linalg.conv_2d_nhwc_hwcf_q {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>} ins(%[[PADDED]], %[[CST]], %[[C0]], %[[C0_1]] : tensor<1x57x57x128xi8>, tensor<3x3x128x256xi8>, i8, i8) outs(%[[BROADCAST]] : tensor<1x28x28x256xi32>) -> tensor<1x28x28x256xi32>
// CHECK: %[[SCAST2:.+]] = quant.scast %[[CONV]] : tensor<1x28x28x256xi32> to tensor<1x28x28x256x!quant.uniform<i32:f32, 9.9089999999999994E-6>>
// CHECK: %[[DCAST:.+]] = quant.dcast %[[SCAST2]] : tensor<1x28x28x256x!quant.uniform<i32:f32, 9.9089999999999994E-6>> to tensor<1x28x28x256xf32>
// CHECK: %[[QCAST2:.+]] = quant.qcast %[[DCAST]] : tensor<1x28x28x256xf32> to tensor<1x28x28x256x!quant.uniform<i8:f32, 0.018517576158046722:-135>>
// CHECK: %[[FINAL_DCAST:.+]] = quant.dcast %[[QCAST2]] : tensor<1x28x28x256x!quant.uniform<i8:f32, 0.018517576158046722:-135>> to tensor<1x28x28x256xf32>
// CHECK: return %[[FINAL_DCAST]] : tensor<1x28x28x256xf32>
// CHECK: }


// Test case for stride=2 quantized int8 convolution
func.func @test2(%act: tensor<1x56x56x128xf32>) -> tensor<1x28x28x256xf32> {
  %wt = arith.constant dense_resource<tensor_1> : tensor<3x3x128x256xi8>
  %bias = arith.constant dense_resource<tensor_2> : tensor<256xf32>
  
  // Precomputed scale product
  // 0.0051725930534303188 * 0.0019158006180077791 = 9.90965696846455e-06
  %bias_qu = quant.qcast %bias : tensor<256xf32> to tensor<256x!quant.uniform<i32:f32, 0.000009909:0>>
  %bias_i32 = quant.scast %bias_qu : tensor<256x!quant.uniform<i32:f32, 0.000009909:0 >> to tensor<256xi32>
  %act_qu = quant.qcast %act : tensor<1x56x56x128xf32> to tensor<1x56x56x128x!quant.uniform<i8:f32, 0.0051725930534303188>>
  %act_i8 = quant.scast %act_qu : tensor<1x56x56x128x!quant.uniform<i8:f32, 0.0051725930534303188>> to tensor<1x56x56x128xi8>
  %act_zp = arith.constant 0 : i8
  // For stride=2 and 3x3 kernel, SAME padding totals 1 on H and W; place on high side.
  %act_i8_padded = tensor.pad %act_i8 low[0, 0, 0, 0] high[0, 1, 1, 0] {
  ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
    tensor.yield %act_zp : i8
  } : tensor<1x56x56x128xi8> to tensor<1x57x57x128xi8>
  %wt_zp = arith.constant 0 : i8

  %init = tensor.empty() : tensor<1x28x28x256xi32>
  %broadcasted_9 = linalg.broadcast ins(%bias_i32 : tensor<256xi32>)
                                  outs(%init : tensor<1x28x28x256xi32>)
                                  dimensions = [0, 1, 2]

  %conv = linalg.conv_2d_nhwc_hwcf_q
            {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>}
            ins(%act_i8_padded, %wt, %act_zp, %wt_zp
                : tensor<1x57x57x128xi8>, tensor<3x3x128x256xi8>, i8, i8)
            outs(%broadcasted_9: tensor<1x28x28x256xi32>) -> tensor<1x28x28x256xi32>

  %conv_output_q32 = quant.scast %conv
      : tensor<1x28x28x256xi32>
      to tensor<1x28x28x256x!quant.uniform<i32:f32, 0.000009909:0>>

  %dcasted = quant.dcast %conv_output_q32
      : tensor<1x28x28x256x!quant.uniform<i32:f32, 0.000009909:0>>
        to tensor<1x28x28x256xf32>

  %qcasted = quant.qcast %dcasted
      : tensor<1x28x28x256xf32>
        to tensor<1x28x28x256x!quant.uniform<i8:f32, 0.018517576158046722:-135>>
    
  %out_f32 = quant.dcast %qcasted
      : tensor<1x28x28x256x!quant.uniform<i8:f32, 0.018517576158046722:-135>>
        to tensor<1x28x28x256xf32>

  return %out_f32 : tensor<1x28x28x256xf32>
}
