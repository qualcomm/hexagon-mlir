//===- matmul-to-hmx-keep-encoding.mlir - the drop-encodings option --------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
// `matmul-to-hmx` erases every `#hmx.crouton` encoding when it is done
// (`drop-encodings`, default on): default bufferization would drop the
// encoding anyway and emit fully dynamic strides. With `drop-encodings=0` the
// encodings stay on the tensors, which is what revives the pack/unpack
// direction verifiers and what lets the pipeline's bufferization carry the
// layout over as `#hmx.crouton_memref_layout`.
//
// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(func.func(matmul-to-hmx{drop-encodings=0}))' -split-input-file | FileCheck %s
//
// The default keeps the IR byte-identical to before the option existed.
// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(func.func(matmul-to-hmx))' -split-input-file | FileCheck %s --check-prefix=DROP
//
// And the ON path now closes the loop: the crouton arrays are allocated by the
// dialect's own `hmx.alloc_crouton`, whose BufferizableOpInterface carries the
// `#hmx.crouton` encoding into the bufferized types as the identity-map
// `#hmx.crouton_memref_layout` (in VTCM, space 1) -- what the stock
// `bufferization.alloc_tensor` could never do.
// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(func.func(matmul-to-hmx{drop-encodings=0}),one-shot-bufferize{bufferize-function-boundaries})' -split-input-file | FileCheck %s --check-prefix=BUF
//===----------------------------------------------------------------------===//

// Same attribution as matmul-to-hmx.mlir's @aligned_f16; with the option on,
// every crouton tensor keeps its encoding (the pack bridge included).
// CHECK-LABEL: func.func @aligned_f16
// CHECK: hmx.pack_act {{.*}} {count = 4 : i64}
// CHECK-SAME: tensor<2x4x16x32x2xf16, #hmx.crouton<logical = [64, 128]>>
// CHECK: hmx.pack_weight {{.*}} {count = 4 : i64}
// CHECK-SAME: tensor<2x4x16x32x2xf16, #hmx.crouton<logical = [64, 128]>>
// CHECK: hmx.matmul ins(%{{.*}}, %{{.*}} : tensor<2x4x16x32x2xf16, #hmx.crouton<logical = [64, 128]>>, tensor<2x4x16x32x2xf16, #hmx.crouton<logical = [64, 128]>>) outs(%{{.*}} : tensor<2x2x16x32x2xf16, #hmx.crouton<logical = [64, 64]>>) -> tensor<2x2x16x32x2xf16, #hmx.crouton<logical = [64, 64]>>
// CHECK: hmx.unpack_acc {{.*}} {count = 16 : i64}
// BUF-LABEL: func.func @aligned_f16
// BUF-DAG: memref.alloc
// BUF-DAG: memref<2x4x16x32x2xf16, #hmx.crouton_memref_layout<logical = [64, 128]>, 1>
// BUF: memref<2x2x16x32x2xf16, #hmx.crouton_memref_layout<logical = [64, 64]>, 1>
// BUF-NOT: tensor<
// DROP-LABEL: func.func @aligned_f16
// DROP-NOT: #hmx
// DROP: hmx.matmul ins(%{{.*}}, %{{.*}} : tensor<2x4x16x32x2xf16>, tensor<2x4x16x32x2xf16>) outs(%{{.*}} : tensor<2x2x16x32x2xf16>) -> tensor<2x2x16x32x2xf16>
// DROP-NOT: #hmx
func.func @aligned_f16(%a: tensor<64x128xf16>, %b: tensor<128x64xf16>) -> tensor<64x64xf16> {
  %c = tensor.empty() : tensor<64x64xf16>
  %0 = linalg.matmul ins(%a, %b : tensor<64x128xf16>, tensor<128x64xf16>)
                     outs(%c : tensor<64x64xf16>) -> tensor<64x64xf16>
  return %0 : tensor<64x64xf16>
}
