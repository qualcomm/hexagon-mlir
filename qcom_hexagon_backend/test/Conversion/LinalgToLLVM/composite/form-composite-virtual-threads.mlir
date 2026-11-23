//===- form-composite-virtual-threads.mlir --------------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(func.func(form-composite-virtual-threads{preferred-num-threads=4}), canonicalize)' | FileCheck %s

// Test that FormCompositeVirtualThreadsPass tiles composite ops into scf.forall
// for multi-threading along parallel (non-reduction) dimensions.

// CHECK-LABEL: func.func @rms_norm_mt
// CHECK-SAME:    (%[[INPUT:.*]]: tensor<128x256xf32>, %[[WEIGHTS:.*]]: tensor<128x256xf32>, %[[INIT:.*]]: tensor<128x256xf32>)
// CHECK:       scf.forall (%{{.*}}) = (0) to (128) step (32) shared_outs(%[[OUT:.*]] = %[[INIT]])
// CHECK:         tensor.extract_slice %[[INPUT]]{{.*}} [32, 256]
// CHECK:         tensor.extract_slice %[[WEIGHTS]]{{.*}} [32, 256]
// CHECK:         tensor.extract_slice %[[OUT]]{{.*}} [32, 256]
// CHECK:         composite.rms_norm axis = 1
// CHECK-SAME:      tensor<32x256xf32>
// CHECK:         scf.forall.in_parallel
// CHECK:           tensor.parallel_insert_slice
func.func @rms_norm_mt(%input: tensor<128x256xf32>,
                        %weights: tensor<128x256xf32>,
                        %init: tensor<128x256xf32>) -> tensor<128x256xf32> {
  %0 = composite.rms_norm axis = 1 eps = 1.0e-5 norm_size = 2.560000e+02
       %input, %weights, %init
       : tensor<128x256xf32>, tensor<128x256xf32>, tensor<128x256xf32>
       -> tensor<128x256xf32>
  return %0 : tensor<128x256xf32>
}

// CHECK-LABEL: func.func @softmax_mt
// CHECK-SAME:    (%[[INPUT:.*]]: tensor<128x256xf32>, %[[INIT:.*]]: tensor<128x256xf32>)
// CHECK:       scf.forall (%{{.*}}) = (0) to (128) step (32) shared_outs(%[[OUT:.*]] = %[[INIT]])
// CHECK:         tensor.extract_slice %[[INPUT]]{{.*}} [32, 256]
// CHECK:         tensor.extract_slice %[[OUT]]{{.*}} [32, 256]
// CHECK:         composite.softmax axis = 1
// CHECK-SAME:      tensor<32x256xf32>
// CHECK:         scf.forall.in_parallel
// CHECK:           tensor.parallel_insert_slice
func.func @softmax_mt(%input: tensor<128x256xf32>,
                       %init: tensor<128x256xf32>) -> tensor<128x256xf32> {
  %0 = composite.softmax axis = 1 %input, %init
       : tensor<128x256xf32>, tensor<128x256xf32>
       -> tensor<128x256xf32>
  return %0 : tensor<128x256xf32>
}

// Small inputs still get tiled — composite ops always have enough work
// per iteration point to justify threading. With 4 threads for 4 rows,
// each thread gets 1 row.
// CHECK-LABEL: func.func @rms_norm_small_mt
// CHECK:       scf.forall (%{{.*}}) in (4) shared_outs
// CHECK:         composite.rms_norm axis = 1
// CHECK-SAME:      tensor<1x256xf32>
func.func @rms_norm_small_mt(%input: tensor<4x256xf32>,
                              %weights: tensor<4x256xf32>,
                              %init: tensor<4x256xf32>) -> tensor<4x256xf32> {
  %0 = composite.rms_norm axis = 1 eps = 1.0e-5 norm_size = 2.560000e+02
       %input, %weights, %init
       : tensor<4x256xf32>, tensor<4x256xf32>, tensor<4x256xf32>
       -> tensor<4x256xf32>
  return %0 : tensor<4x256xf32>
}

// CHECK-LABEL: func.func @layer_norm_mt
// CHECK-SAME:    (%[[INPUT:.*]]: tensor<128x1024xf32>, %[[WEIGHTS:.*]]: tensor<128x1024xf32>, %[[BIAS:.*]]: tensor<128x1024xf32>, %[[INIT:.*]]: tensor<128x1024xf32>)
// CHECK:       scf.forall (%{{.*}}) = (0) to (128) step (32) shared_outs(%[[OUT:.*]] = %[[INIT]])
// CHECK:         tensor.extract_slice %[[INPUT]]{{.*}} [32, 1024]
// CHECK:         tensor.extract_slice %[[WEIGHTS]]{{.*}} [32, 1024]
// CHECK:         tensor.extract_slice %[[BIAS]]{{.*}} [32, 1024]
// CHECK:         tensor.extract_slice %[[OUT]]{{.*}} [32, 1024]
// CHECK:         composite.layer_norm axis = 1
// CHECK-SAME:      tensor<32x1024xf32>
// CHECK:         scf.forall.in_parallel
// CHECK:           tensor.parallel_insert_slice
func.func @layer_norm_mt(%input: tensor<128x1024xf32>,
                          %weights: tensor<128x1024xf32>,
                          %bias: tensor<128x1024xf32>,
                          %init: tensor<128x1024xf32>) -> tensor<128x1024xf32> {
  %0 = composite.layer_norm axis = 1 eps = 1.0e-5 norm_size = 1.024000e+03
       %input, %weights, %bias, %init
       : tensor<128x1024xf32>, tensor<128x1024xf32>, tensor<128x1024xf32>,
         tensor<128x1024xf32>
       -> tensor<128x1024xf32>
  return %0 : tensor<128x1024xf32>
}

// 1D tensor has only the reduction axis — no parallel dim to tile.
// CHECK-LABEL: func.func @rms_norm_1d_no_mt
// CHECK-NOT:   scf.forall
// CHECK:       composite.rms_norm
// CHECK-SAME:    tensor<256xf32>
func.func @rms_norm_1d_no_mt(%input: tensor<256xf32>,
                              %weights: tensor<256xf32>,
                              %init: tensor<256xf32>) -> tensor<256xf32> {
  %0 = composite.rms_norm axis = 0 eps = 1.0e-5 norm_size = 2.560000e+02
       %input, %weights, %init
       : tensor<256xf32>, tensor<256xf32>, tensor<256xf32>
       -> tensor<256xf32>
  return %0 : tensor<256xf32>
}
