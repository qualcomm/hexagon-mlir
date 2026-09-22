//===- matmul-to-hmx.mlir - engine attribution for the HMX engine ---------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
// Every "skipped" case below uses an empty initialiser and is legal on every
// other count, so each one isolates exactly the condition it is testing.
//
// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(func.func(matmul-to-hmx))' -split-input-file | FileCheck %s
//===----------------------------------------------------------------------===//

// A legal f16 matmul becomes hmx.matmul on croutons, bridged from and to
// row-major. 64x128 x 128x64 gives the crouton grid [2, 4] x [2, 4] -> [2, 2]
// (the weight is stored as Wᵀ, grid [Nt, Kt] = [2, 4]), each crouton being
// 16x32x2.
// The pack bridge walks the outer tile and covers the whole K run in one ranged
// op (`count = Kt = 4`); the unpack covers the whole tile row (`count = 16`).
// CHECK-LABEL: func.func @aligned_f16
// CHECK: hmx.pack_act {{.*}} {count = 4 : i64}
// CHECK: hmx.pack_weight {{.*}} {count = 4 : i64}
// CHECK: hmx.matmul ins(%{{.*}}, %{{.*}} : tensor<2x4x16x32x2xf16>, tensor<2x4x16x32x2xf16>) outs(%{{.*}} : tensor<2x2x16x32x2xf16>) -> tensor<2x2x16x32x2xf16>
// CHECK: hmx.unpack_acc {{.*}} {count = 16 : i64}
func.func @aligned_f16(%a: tensor<64x128xf16>, %b: tensor<128x64xf16>) -> tensor<64x64xf16> {
  %c = tensor.empty() : tensor<64x64xf16>
  %0 = linalg.matmul ins(%a, %b : tensor<64x128xf16>, tensor<128x64xf16>)
                     outs(%c : tensor<64x64xf16>) -> tensor<64x64xf16>
  return %0 : tensor<64x64xf16>
}

// -----

// The engine's read-out is fp16, so an f32 result used to be that fp16 image
// widened after the unpack -- which is exactly what llama.cpp's f32 HMX matmul
// does. The accumulation itself happens in the engine's 37-bit cells, so this
// is not an "fp16 accumulate". Now the tail is fused: one `hmx.unpack_acc_f32`
// per row-pair unpacks straight to f32, so neither the fp16 image nor the
// widening generic appears.
// CHECK-LABEL: func.func @f32_result
// CHECK: hmx.pack_act
// CHECK: hmx.pack_weight
// CHECK: hmx.matmul ins(%{{.*}}, %{{.*}} : tensor<2x4x16x32x2xf16>, tensor<2x4x16x32x2xf16>) outs(%{{.*}} : tensor<2x2x16x32x2xf16>) -> tensor<2x2x16x32x2xf16>
// CHECK: hmx.unpack_acc_f32
// CHECK-NOT: hmx.unpack_acc
// CHECK-NOT: arith.extf
func.func @f32_result(%a: tensor<64x128xf16>, %b: tensor<128x64xf16>) -> tensor<64x64xf32> {
  %c = tensor.empty() : tensor<64x64xf32>
  %0 = linalg.matmul ins(%a, %b : tensor<64x128xf16>, tensor<128x64xf16>)
                     outs(%c : tensor<64x64xf32>) -> tensor<64x64xf32>
  return %0 : tensor<64x64xf32>
}

// -----

// An incoming C from outside the function cannot be proven dense (it
// bufferizes to a strided view of unknown stride, while the fused leaf reads
// the residual through the static column count), so it keeps the old epilogue:
// the unpack plus the widening plus the descriptor-based add. This case pins
// that fallback.
// CHECK-LABEL: func.func @f32_accumulate
// CHECK: hmx.pack_act
// CHECK: hmx.pack_weight
// CHECK: hmx.matmul
// CHECK: hmx.unpack_acc
// CHECK-NOT: hmx.unpack_acc_f32
// CHECK: arith.extf
// CHECK: arith.addf
func.func @f32_accumulate(%a: tensor<64x128xf16>, %b: tensor<128x64xf16>,
                          %c: tensor<64x64xf32>) -> tensor<64x64xf32> {
  %0 = linalg.matmul ins(%a, %b : tensor<64x128xf16>, tensor<128x64xf16>)
                     outs(%c : tensor<64x64xf32>) -> tensor<64x64xf32>
  return %0 : tensor<64x64xf32>
}

// -----

// An incoming C produced densely inside the function (here a non-zero fill of
// a fresh empty) threads through the fused tail as the residual: one
// `hmx.unpack_acc_f32` replaces the unpack, the widening and the add.
// CHECK-LABEL: func.func @f32_dense_residual
// CHECK: hmx.pack_act
// CHECK: hmx.pack_weight
// CHECK: hmx.matmul
// CHECK: hmx.unpack_acc_f32 ins(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : tensor<2x2x16x32x2xf16>, tensor<64x64xf32>) outs(%{{.*}} : tensor<64x64xf32>)
// CHECK-NOT: hmx.unpack_acc
// CHECK-NOT: arith.extf
// CHECK-NOT: arith.addf
func.func @f32_dense_residual(%a: tensor<64x128xf16>, %b: tensor<128x64xf16>) -> tensor<64x64xf32> {
  %e = tensor.empty() : tensor<64x64xf32>
  %one = arith.constant 1.000000e+00 : f32
  %c = linalg.fill ins(%one : f32) outs(%e : tensor<64x64xf32>) -> tensor<64x64xf32>
  %0 = linalg.matmul ins(%a, %b : tensor<64x128xf16>, tensor<128x64xf16>)
                     outs(%c : tensor<64x64xf32>) -> tensor<64x64xf32>
  return %0 : tensor<64x64xf32>
}

// -----

// K off the 32-wide grid: left alone, with a remark (checked separately).
// CHECK-LABEL: func.func @k_not_aligned
// CHECK-NOT: hmx.matmul
// CHECK: linalg.matmul
func.func @k_not_aligned(%a: tensor<64x100xf16>, %b: tensor<100x64xf16>) -> tensor<64x64xf16> {
  %c = tensor.empty() : tensor<64x64xf16>
  %0 = linalg.matmul ins(%a, %b : tensor<64x100xf16>, tensor<100x64xf16>)
                     outs(%c : tensor<64x64xf16>) -> tensor<64x64xf16>
  return %0 : tensor<64x64xf16>
}

// -----

// N off the grid.
// CHECK-LABEL: func.func @n_not_aligned
// CHECK-NOT: hmx.matmul
// CHECK: linalg.matmul
func.func @n_not_aligned(%a: tensor<64x128xf16>, %b: tensor<128x48xf16>) -> tensor<64x48xf16> {
  %c = tensor.empty() : tensor<64x48xf16>
  %0 = linalg.matmul ins(%a, %b : tensor<64x128xf16>, tensor<128x48xf16>)
                     outs(%c : tensor<64x48xf16>) -> tensor<64x48xf16>
  return %0 : tensor<64x48xf16>
}

// -----

// Dynamic shapes are not the engine's business yet.
// CHECK-LABEL: func.func @dynamic
// CHECK-NOT: hmx.matmul
// CHECK: linalg.matmul
func.func @dynamic(%a: tensor<?x32xf16>, %b: tensor<32x64xf16>, %m: index) -> tensor<?x64xf16> {
  %c = tensor.empty(%m) : tensor<?x64xf16>
  %0 = linalg.matmul ins(%a, %b : tensor<?x32xf16>, tensor<32x64xf16>)
                     outs(%c : tensor<?x64xf16>) -> tensor<?x64xf16>
  return %0 : tensor<?x64xf16>
}

// -----

// An incoming f16 C also converts: the add happens in the initialiser's own
// element type, so nothing is silently dropped.
// CHECK-LABEL: func.func @non_empty_init
// CHECK: hmx.matmul
// CHECK: arith.addf
func.func @non_empty_init(%a: tensor<64x128xf16>, %b: tensor<128x64xf16>,
                          %c: tensor<64x64xf16>) -> tensor<64x64xf16> {
  %0 = linalg.matmul ins(%a, %b : tensor<64x128xf16>, tensor<128x64xf16>)
                     outs(%c : tensor<64x64xf16>) -> tensor<64x64xf16>
  return %0 : tensor<64x64xf16>
}

// -----

// A constant weight is prepacked at compile time (inference bakes W in), so no
// runtime packer runs for it: only the copy into VTCM is left.
// CHECK-LABEL: func.func @constant_weight
// CHECK-NOT: hmx.pack_weight
// CHECK: hmx.matmul
func.func @constant_weight(%a: tensor<64x64xf16>) -> tensor<64x64xf16> {
  %w = arith.constant dense<1.000000e+00> : tensor<64x64xf16>
  %empty = tensor.empty() : tensor<64x64xf16>
  %zero = arith.constant 0.000000e+00 : f16
  %c = linalg.fill ins(%zero : f16) outs(%empty : tensor<64x64xf16>) -> tensor<64x64xf16>
  %m = linalg.matmul ins(%a, %w : tensor<64x64xf16>, tensor<64x64xf16>)
                     outs(%c : tensor<64x64xf16>) -> tensor<64x64xf16>
  return %m : tensor<64x64xf16>
}

// -----

// A matmul chained off another matmul reuses the read-out array as its
// activation (AR and AH are the same permutation, verified bit-exactly on
// device): no unpack/pack pair appears between the two matmuls.
// CHECK-LABEL: func.func @chained
// CHECK: hmx.matmul
// CHECK-NOT: hmx.unpack_acc
// CHECK-NOT: hmx.pack_act
// CHECK: hmx.matmul
// CHECK: hmx.unpack_acc
func.func @chained(%a: tensor<64x64xf16>, %b: tensor<64x64xf16>,
                   %c: tensor<64x64xf16>) -> tensor<64x64xf16> {
  %e0 = tensor.empty() : tensor<64x64xf16>
  %z0 = arith.constant 0.000000e+00 : f16
  %c0 = linalg.fill ins(%z0 : f16) outs(%e0 : tensor<64x64xf16>) -> tensor<64x64xf16>
  %m0 = linalg.matmul ins(%a, %b : tensor<64x64xf16>, tensor<64x64xf16>)
                      outs(%c0 : tensor<64x64xf16>) -> tensor<64x64xf16>
  %e1 = tensor.empty() : tensor<64x64xf16>
  %z1 = arith.constant 0.000000e+00 : f16
  %c1 = linalg.fill ins(%z1 : f16) outs(%e1 : tensor<64x64xf16>) -> tensor<64x64xf16>
  %m1 = linalg.matmul ins(%m0, %c : tensor<64x64xf16>, tensor<64x64xf16>)
                      outs(%c1 : tensor<64x64xf16>) -> tensor<64x64xf16>
  return %m1 : tensor<64x64xf16>
}

// -----

// A conversion whose operand is loop-invariant is hoisted out of the loop: an
// attention inner loop packs its activation once instead of once per KV block.
// CHECK-LABEL: func.func @hoisted
// CHECK: hmx.pack_act
// CHECK: hmx.pack_weight
// CHECK-NOT: hmx.pack_act
// CHECK: scf.for
// CHECK: hmx.matmul
func.func @hoisted(%a: tensor<64x64xf16>, %b: tensor<64x64xf16>, %n: index)
    -> tensor<64x64xf16> {
  %empty = tensor.empty() : tensor<64x64xf16>
  %zero = arith.constant 0.000000e+00 : f16
  %init = linalg.fill ins(%zero : f16) outs(%empty : tensor<64x64xf16>) -> tensor<64x64xf16>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %r = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %init) -> (tensor<64x64xf16>) {
    %m = linalg.matmul ins(%a, %b : tensor<64x64xf16>, tensor<64x64xf16>)
                        outs(%acc : tensor<64x64xf16>) -> tensor<64x64xf16>
    scf.yield %m : tensor<64x64xf16>
  }
  return %r : tensor<64x64xf16>
}

// -----

// A pure elementwise map between two matmuls runs directly on the crouton array:
// it commutes with the layout permutation, so neither the unpack nor the pack
// appears. Reductions would keep a conversion (the co-scheduling boundary).
// CHECK-LABEL: func.func @map_between
// CHECK: hmx.matmul
// CHECK-NOT: hmx.unpack_acc
// CHECK-NOT: hmx.pack_act
// CHECK: linalg.generic
// CHECK-NOT: hmx.unpack_acc
// CHECK-NOT: hmx.pack_act
// CHECK: hmx.matmul
func.func @map_between(%a: tensor<64x64xf16>, %b: tensor<64x64xf16>,
                       %c: tensor<64x64xf16>) -> tensor<64x64xf16> {
  %e0 = tensor.empty() : tensor<64x64xf16>
  %z0 = arith.constant 0.000000e+00 : f16
  %i0 = linalg.fill ins(%z0 : f16) outs(%e0 : tensor<64x64xf16>) -> tensor<64x64xf16>
  %m0 = linalg.matmul ins(%a, %b : tensor<64x64xf16>, tensor<64x64xf16>)
                      outs(%i0 : tensor<64x64xf16>) -> tensor<64x64xf16>
  %e1 = tensor.empty() : tensor<64x64xf16>
  %map = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%m0 : tensor<64x64xf16>) outs(%e1 : tensor<64x64xf16>) {
  ^bb0(%in: f16, %out: f16):
    %half = arith.constant 5.000000e-01 : f16
    %mul = arith.mulf %in, %half : f16
    linalg.yield %mul : f16
  } -> tensor<64x64xf16>
  %e2 = tensor.empty() : tensor<64x64xf16>
  %z2 = arith.constant 0.000000e+00 : f16
  %i2 = linalg.fill ins(%z2 : f16) outs(%e2 : tensor<64x64xf16>) -> tensor<64x64xf16>
  %m1 = linalg.matmul ins(%map, %c : tensor<64x64xf16>, tensor<64x64xf16>)
                      outs(%i2 : tensor<64x64xf16>) -> tensor<64x64xf16>
  return %m1 : tensor<64x64xf16>
}

// -----

// An fp32 activation is the same attribution: the source keeps its own element
// type while the crouton is the engine's fp16, the pack quantising on the way in
// (see `hmx.pack_act`). Real attention has fp32 operands, and llama.cpp's
// f32-activation path has an fp32 activation against an fp16 weight.
// CHECK-LABEL: func.func @f32_activation
// CHECK: hmx.pack_act ins(%{{.*}}, %{{.*}}, %{{.*}} : tensor<64x128xf32>)
// CHECK: hmx.pack_weight ins(%{{.*}}, %{{.*}}, %{{.*}} : tensor<128x64xf16>)
// CHECK: hmx.matmul ins(%{{.*}}, %{{.*}} : tensor<2x4x16x32x2xf16>, tensor<2x4x16x32x2xf16>) outs(%{{.*}} : tensor<2x2x16x32x2xf16>) -> tensor<2x2x16x32x2xf16>
// CHECK: hmx.unpack_acc
func.func @f32_activation(%a: tensor<64x128xf32>, %b: tensor<128x64xf16>) -> tensor<64x64xf16> {
  %c = tensor.empty() : tensor<64x64xf16>
  %0 = linalg.matmul ins(%a, %b : tensor<64x128xf32>, tensor<128x64xf16>)
                     outs(%c : tensor<64x64xf16>) -> tensor<64x64xf16>
  return %0 : tensor<64x64xf16>
}

// -----

// Both operands fp32 with an fp32 result: the attention shape. `tl.dot` requires
// its two operands to agree, so this fp32 dot is what a real fp32 attention
// produces, and its fp32 read-out takes the fused tail.
// CHECK-LABEL: func.func @f32_operands
// CHECK: hmx.pack_act ins(%{{.*}}, %{{.*}}, %{{.*}} : tensor<64x64xf32>)
// CHECK: hmx.pack_weight ins(%{{.*}}, %{{.*}}, %{{.*}} : tensor<64x64xf32>)
// CHECK: hmx.matmul
// CHECK: hmx.unpack_acc_f32
func.func @f32_operands(%a: tensor<64x64xf32>, %b: tensor<64x64xf32>) -> tensor<64x64xf32> {
  %c = tensor.empty() : tensor<64x64xf32>
  %0 = linalg.matmul ins(%a, %b : tensor<64x64xf32>, tensor<64x64xf32>)
                     outs(%c : tensor<64x64xf32>) -> tensor<64x64xf32>
  return %0 : tensor<64x64xf32>
}

// -----

// A contraction too large to hold whole is walked in M blocks instead of being
// refused: M=N=4096, K=64 on the device budget (8 MiB) has a 32 MiB read-out,
// so it blocks M into 16 tiles = 512 rows (the largest divisor of M/32 = 128
// whose footprint fits). Each block's activation is [16, 2] (512 rows, Kt=2)
// and its read-out [16, 128]; the weight stays whole at [128, 2]. The block
// footprint is 512*64*2 + 64*4096*2 + 512*4096*2 = 4 784 128 bytes, under the
// 8 MiB budget.
// CHECK-LABEL: func.func @large_shape_blocks
// CHECK: scf.for {{.*}} step {{.*}} iter_args({{.*}}) -> (tensor<4096x4096xf16>)
// CHECK: tensor.extract_slice %arg0[%arg2, %c0] [512, 64] [1, 1] : tensor<4096x64xf16> to tensor<512x64xf16>
// CHECK: hmx.pack_act
// CHECK: hmx.matmul ins({{.*}} : tensor<16x2x16x32x2xf16>, tensor<128x2x16x32x2xf16>) outs({{.*}} : tensor<16x128x16x32x2xf16>) -> tensor<16x128x16x32x2xf16>
// CHECK: tensor.insert_slice {{.*}} into %arg3[%arg2, %c0] [512, 4096] [1, 1] : tensor<512x4096xf16> into tensor<4096x4096xf16>
func.func @large_shape_blocks(%a: tensor<4096x64xf16>, %b: tensor<64x4096xf16>) -> tensor<4096x4096xf16> {
  %c = tensor.empty() : tensor<4096x4096xf16>
  %0 = linalg.matmul ins(%a, %b : tensor<4096x64xf16>, tensor<64x4096xf16>)
                     outs(%c : tensor<4096x4096xf16>) -> tensor<4096x4096xf16>
  return %0 : tensor<4096x4096xf16>
}

// -----
