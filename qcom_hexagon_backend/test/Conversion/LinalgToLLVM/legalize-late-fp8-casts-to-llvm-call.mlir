//===- legalize-late-fp8-casts-to-llvm-call.mlir --------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

// RUN: linalg-hexagon-opt %s -legalize-late-fp8-casts-to-llvm-call -canonicalize | FileCheck %s

module {
  // CHECK-LABEL: llvm.func @kernel
  // CHECK: %[[U0:.*]] = llvm.mlir.undef : vector<1xf16>
  // CHECK: %[[F16:.*]] = llvm.fptrunc %{{.*}} : f32 to f16
  // CHECK: %[[SPLAT_IN:.*]] = llvm.bitcast %[[F16]] : f16 to vector<1xf16>
  // CHECK: %[[SPLAT:.*]] = llvm.shufflevector %[[SPLAT_IN]], %[[U0]] [{{.*}}] : vector<1xf16>
  // CHECK: %[[PACK_IN:.*]] = llvm.bitcast %[[SPLAT]] : vector<128xf16> to vector<64xi32>
  // CHECK-NOT: llvm.intr.vector.insert
  // CHECK: %[[PACKED:.*]] = llvm.call @convert_fp16_fp8_fnuz(%[[PACK_IN]]) : (vector<64xi32>) -> vector<32xi32>
  // CHECK: %[[I8VEC:.*]] = llvm.bitcast %[[PACKED]] : vector<32xi32> to vector<128xi8>
  // CHECK: %[[EXTRACTED:.*]] = llvm.intr.vector.extract %[[I8VEC]][0] : vector<1xi8> from vector<128xi8>
  // CHECK: %[[SCALAR:.*]] = llvm.bitcast %[[EXTRACTED]] : vector<1xi8> to i8
  // CHECK: llvm.return %[[SCALAR]] : i8
  // CHECK-NOT: arith.truncf
  // CHECK-NOT: builtin.unrealized_conversion_cast
  llvm.func @kernel(%arg0: f32) -> i8 {
    %0 = llvm.fptrunc %arg0 : f32 to f16
    %1 = arith.truncf %0 : f16 to f8E4M3FNUZ
    %2 = builtin.unrealized_conversion_cast %1 : f8E4M3FNUZ to i8
    llvm.return %2 : i8
  }

  // CHECK-LABEL: llvm.func @kernel_vec_direct
  // CHECK: llvm.fptrunc %{{.*}} : vector<128xf32> to vector<128xf16>
  // CHECK: llvm.bitcast %{{.*}} : vector<128xf16> to vector<64xi32>
  // CHECK: llvm.call @convert_fp16_fp8_fnuz(%{{.*}}) : (vector<64xi32>) -> vector<32xi32>
  // CHECK: llvm.bitcast %{{.*}} : vector<32xi32> to vector<128xi8>
  // CHECK-NOT: arith.truncf
  // CHECK-NOT: builtin.unrealized_conversion_cast
  llvm.func @kernel_vec_direct(%arg0: vector<128xf32>) -> vector<128xi8> {
    %0 = arith.truncf %arg0 fastmath<fast> : vector<128xf32> to vector<128xf8E4M3FNUZ>
    %1 = builtin.unrealized_conversion_cast %0 : vector<128xf8E4M3FNUZ> to vector<128xi8>
    llvm.return %1 : vector<128xi8>
  }

  // CHECK-LABEL: llvm.func @kernel_vec_direct_fn
  // CHECK: llvm.fptrunc %{{.*}} : vector<128xf32> to vector<128xf16>
  // CHECK: llvm.bitcast %{{.*}} : vector<128xf16> to vector<64xi32>
  // CHECK: llvm.call @convert_fp16_fp8_fn(%{{.*}}) : (vector<64xi32>) -> vector<32xi32>
  // CHECK: llvm.bitcast %{{.*}} : vector<32xi32> to vector<128xi8>
  // CHECK-NOT: arith.truncf
  // CHECK-NOT: builtin.unrealized_conversion_cast
  llvm.func @kernel_vec_direct_fn(%arg0: vector<128xf32>) -> vector<128xi8> {
    %0 = arith.truncf %arg0 fastmath<fast> : vector<128xf32> to vector<128xf8E4M3FN>
    %1 = builtin.unrealized_conversion_cast %0 : vector<128xf8E4M3FN> to vector<128xi8>
    llvm.return %1 : vector<128xi8>
  }

  // CHECK-LABEL: llvm.func @kernel_scalar_direct
  // CHECK: %[[U1:.*]] = llvm.mlir.undef : vector<1xf16>
  // CHECK: %[[F16_1:.*]] = llvm.fptrunc %{{.*}} : f32 to f16
  // CHECK: %[[SPLAT_IN_1:.*]] = llvm.bitcast %[[F16_1]] : f16 to vector<1xf16>
  // CHECK: %[[SPLAT_1:.*]] = llvm.shufflevector %[[SPLAT_IN_1]], %[[U1]] [{{.*}}] : vector<1xf16>
  // CHECK: %[[PACK_IN_1:.*]] = llvm.bitcast %[[SPLAT_1]] : vector<128xf16> to vector<64xi32>
  // CHECK: %[[PACKED_1:.*]] = llvm.call @convert_fp16_fp8_fnuz(%[[PACK_IN_1]]) : (vector<64xi32>) -> vector<32xi32>
  // CHECK: %[[I8VEC_1:.*]] = llvm.bitcast %[[PACKED_1]] : vector<32xi32> to vector<128xi8>
  // CHECK: %[[EXTRACTED_1:.*]] = llvm.intr.vector.extract %[[I8VEC_1]][0] : vector<1xi8> from vector<128xi8>
  // CHECK: llvm.bitcast %[[EXTRACTED_1]] : vector<1xi8> to i8
  // CHECK-NOT: arith.truncf
  // CHECK-NOT: builtin.unrealized_conversion_cast
  llvm.func @kernel_scalar_direct(%arg0: f32) -> i8 {
    %0 = arith.truncf %arg0 fastmath<fast> : f32 to f8E4M3FNUZ
    %1 = builtin.unrealized_conversion_cast %0 : f8E4M3FNUZ to i8
    llvm.return %1 : i8
  }

  // CHECK-LABEL: llvm.func @kernel_scalar_f16_direct
  // CHECK-NOT: llvm.fptrunc %{{.*}} : f32 to f16
  // CHECK: %[[U2:.*]] = llvm.mlir.undef : vector<1xf16>
  // CHECK: %[[SPLAT_IN_2:.*]] = llvm.bitcast %{{.*}} : f16 to vector<1xf16>
  // CHECK: %[[SPLAT_2:.*]] = llvm.shufflevector %[[SPLAT_IN_2]], %[[U2]] [{{.*}}] : vector<1xf16>
  // CHECK: %[[PACK_IN_2:.*]] = llvm.bitcast %[[SPLAT_2]] : vector<128xf16> to vector<64xi32>
  // CHECK: %[[PACKED_2:.*]] = llvm.call @convert_fp16_fp8_fnuz(%[[PACK_IN_2]]) : (vector<64xi32>) -> vector<32xi32>
  // CHECK: %[[I8VEC_2:.*]] = llvm.bitcast %[[PACKED_2]] : vector<32xi32> to vector<128xi8>
  // CHECK: %[[EXTRACTED_2:.*]] = llvm.intr.vector.extract %[[I8VEC_2]][0] : vector<1xi8> from vector<128xi8>
  // CHECK: llvm.bitcast %[[EXTRACTED_2]] : vector<1xi8> to i8
  // CHECK-NOT: arith.truncf
  // CHECK-NOT: builtin.unrealized_conversion_cast
  llvm.func @kernel_scalar_f16_direct(%arg0: f16) -> i8 {
    %0 = arith.truncf %arg0 fastmath<fast> : f16 to f8E4M3FNUZ
    %1 = builtin.unrealized_conversion_cast %0 : f8E4M3FNUZ to i8
    llvm.return %1 : i8
  }

  // CHECK-LABEL: llvm.func @kernel_extf_vec_fn
  // CHECK: llvm.bitcast %{{.*}} : vector<128xi8> to vector<32xi32>
  // CHECK: llvm.call @convert_fp8_fp16_fn(%{{.*}}) : (vector<32xi32>) -> vector<64xi32>
  // CHECK: llvm.bitcast %{{.*}} : vector<64xi32> to vector<128xf16>
  // CHECK-NOT: arith.extf
  // CHECK-NOT: builtin.unrealized_conversion_cast
  llvm.func @kernel_extf_vec_fn(%arg0: vector<128xi8>) -> vector<128xf16> {
    %0 = builtin.unrealized_conversion_cast %arg0 : vector<128xi8> to vector<128xf8E4M3FN>
    %1 = arith.extf %0 : vector<128xf8E4M3FN> to vector<128xf16>
    llvm.return %1 : vector<128xf16>
  }
}
