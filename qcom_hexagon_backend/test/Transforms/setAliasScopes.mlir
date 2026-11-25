//===- setAliasScopes.mlir ------------------------------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

// RUN: linalg-hexagon-opt %s -split-input-file -pass-pipeline='builtin.module(func.func(set-alias-scopes))' | FileCheck %s

// -----
// Test 1: Two arguments traced through the full reinterpret_cast ->
// unrealized_conversion_cast -> extractvalue -> getelementptr chain. Each
// resolves to its own root ("arg0", "arg1"), and all four loads/stores
// receive reciprocal alias_scopes/noalias_scopes.
// CHECK-DAG: #[[DOMAIN:.+]] = #llvm.alias_scope_domain<{{.*}}, description = "two_args">
// CHECK-DAG: #[[SCOPE0:.+]] = #llvm.alias_scope<{{.*}}, domain = #[[DOMAIN]], description = "arg0">
// CHECK-DAG: #[[SCOPE1:.+]] = #llvm.alias_scope<{{.*}}, domain = #[[DOMAIN]], description = "arg1">
// CHECK-LABEL: @two_args
func.func @two_args(%arg0: memref<64xf32, 1>, %arg1: memref<64xf32, 1>) {
  %base0, %offset0, %sizes0, %strides0 = memref.extract_strided_metadata %arg0 : memref<64xf32, 1> -> memref<f32, 1>, index, index, index
  %rc0 = memref.reinterpret_cast %base0 to offset: [0], sizes: [64], strides: [1] : memref<f32, 1> to memref<64xf32, strided<[1]>, 1>
  %cast0 = builtin.unrealized_conversion_cast %rc0 : memref<64xf32, strided<[1]>, 1> to !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
  %ptr0 = llvm.extractvalue %cast0[1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
  %c0 = arith.constant 0 : index
  %c0_i64 = builtin.unrealized_conversion_cast %c0 : index to i64
  %gep0 = llvm.getelementptr %ptr0[%c0_i64] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32

  %base1, %offset1, %sizes1, %strides1 = memref.extract_strided_metadata %arg1 : memref<64xf32, 1> -> memref<f32, 1>, index, index, index
  %rc1 = memref.reinterpret_cast %base1 to offset: [0], sizes: [64], strides: [1] : memref<f32, 1> to memref<64xf32, strided<[1]>, 1>
  %cast1 = builtin.unrealized_conversion_cast %rc1 : memref<64xf32, strided<[1]>, 1> to !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
  %ptr1 = llvm.extractvalue %cast1[1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
  %gep1 = llvm.getelementptr %ptr1[%c0_i64] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32

  %val0 = llvm.load %gep0 : !llvm.ptr<1> -> f32          // load from arg0
  llvm.store %val0, %gep1 : f32, !llvm.ptr<1>            // store to arg1
  %val1 = llvm.load %gep1 : !llvm.ptr<1> -> f32          // load from arg1
  llvm.store %val1, %gep0 : f32, !llvm.ptr<1>            // store to arg0
  return
}
// CHECK: %[[GEP0:.+]] = llvm.getelementptr %{{.*}}[%{{.*}}] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
// CHECK: %[[GEP1:.+]] = llvm.getelementptr %{{.*}}[%{{.*}}] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
// CHECK: %[[VAL0:.+]] = llvm.load %[[GEP0]] {alias_scopes = [#[[SCOPE0]]], noalias_scopes = [#[[SCOPE1]]]} : !llvm.ptr<1> -> f32
// CHECK: llvm.store %[[VAL0]], %[[GEP1]] {alias_scopes = [#[[SCOPE1]]], noalias_scopes = [#[[SCOPE0]]]} : f32, !llvm.ptr<1>
// CHECK: %[[VAL1:.+]] = llvm.load %[[GEP1]] {alias_scopes = [#[[SCOPE1]]], noalias_scopes = [#[[SCOPE0]]]} : !llvm.ptr<1> -> f32
// CHECK: llvm.store %[[VAL1]], %[[GEP0]] {alias_scopes = [#[[SCOPE0]]], noalias_scopes = [#[[SCOPE1]]]} : f32, !llvm.ptr<1>

// -----
// Test 2: A single buffer accessed by both the load and the store traces to
// one root, so rootToOps.size() < 2 and the pass bails out — no scope or
// alias attribute is attached to either op.
// CHECK-LABEL: @single_root_bailout
func.func @single_root_bailout(%arg0: memref<64xf32, 1>) {
  %base0, %offset0, %sizes0, %strides0 = memref.extract_strided_metadata %arg0 : memref<64xf32, 1> -> memref<f32, 1>, index, index, index
  %rc0 = memref.reinterpret_cast %base0 to offset: [0], sizes: [64], strides: [1] : memref<f32, 1> to memref<64xf32, strided<[1]>, 1>
  %cast0 = builtin.unrealized_conversion_cast %rc0 : memref<64xf32, strided<[1]>, 1> to !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
  %ptr0 = llvm.extractvalue %cast0[1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
  %c0 = arith.constant 0 : index
  %c0_i64 = builtin.unrealized_conversion_cast %c0 : index to i64
  %gep0 = llvm.getelementptr %ptr0[%c0_i64] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
  %val0 = llvm.load %gep0 : !llvm.ptr<1> -> f32          // load from arg0
  llvm.store %val0, %gep0 : f32, !llvm.ptr<1>            // store to arg0
  return
}
// CHECK-NOT: #llvm.alias_scope
// CHECK: %[[T2_GEP0:.+]] = llvm.getelementptr %{{.*}}[%{{.*}}] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
// CHECK: %[[T2_V0:.+]] = llvm.load %[[T2_GEP0]] : !llvm.ptr<1> -> f32
// CHECK: llvm.store %[[T2_V0]], %[[T2_GEP0]] : f32, !llvm.ptr<1>

// -----
// Test 3: A function argument alongside a memref.alloca() -- local alloc.
// CHECK-DAG: #[[DOMAIN:.+]] = #llvm.alias_scope_domain<{{.*}}, description = "arg_and_alloca">
// CHECK-DAG: #[[SCOPE0:.+]] = #llvm.alias_scope<{{.*}}, domain = #[[DOMAIN]], description = "arg0">
// CHECK-DAG: #[[SCOPE1:.+]] = #llvm.alias_scope<{{.*}}, domain = #[[DOMAIN]], description = "local">
// CHECK-LABEL: @arg_and_alloca
func.func @arg_and_alloca(%arg0: memref<64xf32, 1>) {
  %base0, %offset0, %sizes0, %strides0 = memref.extract_strided_metadata %arg0 : memref<64xf32, 1> -> memref<f32, 1>, index, index, index
  %rc0 = memref.reinterpret_cast %base0 to offset: [0], sizes: [64], strides: [1] : memref<f32, 1> to memref<64xf32, strided<[1]>, 1>
  %cast0 = builtin.unrealized_conversion_cast %rc0 : memref<64xf32, strided<[1]>, 1> to !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
  %ptr0 = llvm.extractvalue %cast0[1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
  %c0 = arith.constant 0 : index
  %c0_i64 = builtin.unrealized_conversion_cast %c0 : index to i64
  %gep0 = llvm.getelementptr %ptr0[%c0_i64] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32

  %local = memref.alloca() : memref<64xf32>
  %castl = builtin.unrealized_conversion_cast %local : memref<64xf32> to !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
  %ptrl = llvm.extractvalue %castl[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
  %c0_i64_2 = builtin.unrealized_conversion_cast %c0 : index to i64
  %gepl = llvm.getelementptr %ptrl[%c0_i64_2] : (!llvm.ptr, i64) -> !llvm.ptr, f32

  %val0 = llvm.load %gep0 : !llvm.ptr<1> -> f32          // load from arg0
  %vall = llvm.load %gepl : !llvm.ptr -> f32             // load from local (alloca)
  return
}
// CHECK: %[[T3_GEP0:.+]] = llvm.getelementptr %{{.*}}[%{{.*}}] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
// CHECK: %[[T3_GEPL:.+]] = llvm.getelementptr %{{.*}}[%{{.*}}] : (!llvm.ptr, i64) -> !llvm.ptr, f32
// CHECK: %[[T3_V0:.+]] = llvm.load %[[T3_GEP0]] {alias_scopes = [#[[SCOPE0]]], noalias_scopes = [#[[SCOPE1]]]} : !llvm.ptr<1> -> f32
// CHECK: %[[T3_VL:.+]] = llvm.load %[[T3_GEPL]] {alias_scopes = [#[[SCOPE1]]], noalias_scopes = [#[[SCOPE0]]]} : !llvm.ptr -> f32

// -----
// Test 4: Same as Test 3, but with memref.alloc() in place of alloca() — the
// local root is still named "local".
// CHECK-DAG: #[[DOMAIN:.+]] = #llvm.alias_scope_domain<{{.*}}, description = "arg_and_memref_alloc">
// CHECK-DAG: #[[SCOPE0:.+]] = #llvm.alias_scope<{{.*}}, domain = #[[DOMAIN]], description = "arg0">
// CHECK-DAG: #[[SCOPE1:.+]] = #llvm.alias_scope<{{.*}}, domain = #[[DOMAIN]], description = "local">
// CHECK-LABEL: @arg_and_memref_alloc
func.func @arg_and_memref_alloc(%arg0: memref<64xf32, 1>) {
  %base0, %offset0, %sizes0, %strides0 = memref.extract_strided_metadata %arg0 : memref<64xf32, 1> -> memref<f32, 1>, index, index, index
  %rc0 = memref.reinterpret_cast %base0 to offset: [0], sizes: [64], strides: [1] : memref<f32, 1> to memref<64xf32, strided<[1]>, 1>
  %cast0 = builtin.unrealized_conversion_cast %rc0 : memref<64xf32, strided<[1]>, 1> to !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
  %ptr0 = llvm.extractvalue %cast0[1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
  %c0 = arith.constant 0 : index
  %c0_i64 = builtin.unrealized_conversion_cast %c0 : index to i64
  %gep0 = llvm.getelementptr %ptr0[%c0_i64] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32

  %local = memref.alloc() : memref<64xf32>
  %castl = builtin.unrealized_conversion_cast %local : memref<64xf32> to !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
  %ptrl = llvm.extractvalue %castl[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
  %c0_i64_2 = builtin.unrealized_conversion_cast %c0 : index to i64
  %gepl = llvm.getelementptr %ptrl[%c0_i64_2] : (!llvm.ptr, i64) -> !llvm.ptr, f32

  %val0 = llvm.load %gep0 : !llvm.ptr<1> -> f32          // load from arg0
  %vall = llvm.load %gepl : !llvm.ptr -> f32             // load from local (memref.alloc)
  return
}
// CHECK: %[[T4_GEP0:.+]] = llvm.getelementptr %{{.*}}[%{{.*}}] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
// CHECK: %[[T4_GEPL:.+]] = llvm.getelementptr %{{.*}}[%{{.*}}] : (!llvm.ptr, i64) -> !llvm.ptr, f32
// CHECK: %[[T4_V0:.+]] = llvm.load %[[T4_GEP0]] {alias_scopes = [#[[SCOPE0]]], noalias_scopes = [#[[SCOPE1]]]} : !llvm.ptr<1> -> f32
// CHECK: %[[T4_VL:.+]] = llvm.load %[[T4_GEPL]] {alias_scopes = [#[[SCOPE1]]], noalias_scopes = [#[[SCOPE0]]]} : !llvm.ptr -> f32

// -----
// Test 5: A memref-typed hexagonmem.alloc() local, confirming the
// hexagonmem::AllocOp root case behaves the same as memref.alloca/alloc.
// CHECK-DAG: #[[DOMAIN:.+]] = #llvm.alias_scope_domain<{{.*}}, description = "arg_and_hexmem_alloc">
// CHECK-DAG: #[[SCOPE0:.+]] = #llvm.alias_scope<{{.*}}, domain = #[[DOMAIN]], description = "arg0">
// CHECK-DAG: #[[SCOPE1:.+]] = #llvm.alias_scope<{{.*}}, domain = #[[DOMAIN]], description = "local">
// CHECK-LABEL: @arg_and_hexmem_alloc
func.func @arg_and_hexmem_alloc(%arg0: memref<64xf32, 1>) {
  %base0, %offset0, %sizes0, %strides0 = memref.extract_strided_metadata %arg0 : memref<64xf32, 1> -> memref<f32, 1>, index, index, index
  %rc0 = memref.reinterpret_cast %base0 to offset: [0], sizes: [64], strides: [1] : memref<f32, 1> to memref<64xf32, strided<[1]>, 1>
  %cast0 = builtin.unrealized_conversion_cast %rc0 : memref<64xf32, strided<[1]>, 1> to !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
  %ptr0 = llvm.extractvalue %cast0[1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
  %c0 = arith.constant 0 : index
  %c0_i64 = builtin.unrealized_conversion_cast %c0 : index to i64
  %gep0 = llvm.getelementptr %ptr0[%c0_i64] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32

  %local = hexagonmem.alloc() {bufferization.manual_deallocation, alignment = 128 : i64} : memref<64xf32>
  %castl = builtin.unrealized_conversion_cast %local : memref<64xf32> to !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
  %ptrl = llvm.extractvalue %castl[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
  %c0_i64_2 = builtin.unrealized_conversion_cast %c0 : index to i64
  %gepl = llvm.getelementptr %ptrl[%c0_i64_2] : (!llvm.ptr, i64) -> !llvm.ptr, f32

  %val0 = llvm.load %gep0 : !llvm.ptr<1> -> f32          // load from arg0
  %vall = llvm.load %gepl : !llvm.ptr -> f32             // load from local (hexagonmem.alloc)
  return
}
// CHECK: %[[T5_GEP0:.+]] = llvm.getelementptr %{{.*}}[%{{.*}}] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
// CHECK: %[[T5_GEPL:.+]] = llvm.getelementptr %{{.*}}[%{{.*}}] : (!llvm.ptr, i64) -> !llvm.ptr, f32
// CHECK: %[[T5_V0:.+]] = llvm.load %[[T5_GEP0]] {alias_scopes = [#[[SCOPE0]]], noalias_scopes = [#[[SCOPE1]]]} : !llvm.ptr<1> -> f32
// CHECK: %[[T5_VL:.+]] = llvm.load %[[T5_GEPL]] {alias_scopes = [#[[SCOPE1]]], noalias_scopes = [#[[SCOPE0]]]} : !llvm.ptr -> f32

// -----
// Test 6: A crouton-typed hexagonmem.alloc reached through
// hexagonmem.crouton_to_memref. traceToRoot() has no case for that op
// (SetAliasScopesPass.cpp's traceToRoot(), recognized-hop list around
// lines 60-113 — CroutonToMemrefOp is absent, so it falls through to
// `return nullptr`), so this chain is left untraced — a known limitation,
// not a bug. Root cause is documented on the op itself: HexagonMemOps.td's
// HexagonMem_CroutonToMemrefOp (lines 218-235) states it "just returns the
// base pointer of the crouton type if the crouton pointers are contiguous"
// but "If the croutons in the crouton table are not contiguous in memory,
// this op would result in a copy of data" — a conditional passthrough-or-copy
// traceToRoot() can't resolve statically, so it's deliberately excluded
// rather than assumed safe. arg0 and arg1 are unaffected and still trace
// normally.
// CHECK-DAG: #[[DOMAIN:.+]] = #llvm.alias_scope_domain<{{.*}}, description = "two_args_and_crouton">
// CHECK-DAG: #[[SCOPE0:.+]] = #llvm.alias_scope<{{.*}}, domain = #[[DOMAIN]], description = "arg0">
// CHECK-DAG: #[[SCOPE1:.+]] = #llvm.alias_scope<{{.*}}, domain = #[[DOMAIN]], description = "arg1">
// CHECK-LABEL: @two_args_and_crouton
func.func @two_args_and_crouton(%arg0: memref<64xf32, 1>, %arg1: memref<64xf32, 1>) {
  %base0, %offset0, %sizes0, %strides0 = memref.extract_strided_metadata %arg0 : memref<64xf32, 1> -> memref<f32, 1>, index, index, index
  %rc0 = memref.reinterpret_cast %base0 to offset: [0], sizes: [64], strides: [1] : memref<f32, 1> to memref<64xf32, strided<[1]>, 1>
  %cast0 = builtin.unrealized_conversion_cast %rc0 : memref<64xf32, strided<[1]>, 1> to !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
  %ptr0 = llvm.extractvalue %cast0[1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
  %c0 = arith.constant 0 : index
  %c0_i64 = builtin.unrealized_conversion_cast %c0 : index to i64
  %gep0 = llvm.getelementptr %ptr0[%c0_i64] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32

  %base1, %offset1, %sizes1, %strides1 = memref.extract_strided_metadata %arg1 : memref<64xf32, 1> -> memref<f32, 1>, index, index, index
  %rc1 = memref.reinterpret_cast %base1 to offset: [0], sizes: [64], strides: [1] : memref<f32, 1> to memref<64xf32, strided<[1]>, 1>
  %cast1 = builtin.unrealized_conversion_cast %rc1 : memref<64xf32, strided<[1]>, 1> to !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
  %ptr1 = llvm.extractvalue %cast1[1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
  %gep1 = llvm.getelementptr %ptr1[%c0_i64] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32

  %crouton = hexagonmem.alloc() {bufferization.manual_deallocation, alignment = 2048 : i64} : !crouton.crouton<64xf32, vtcm>
  %cmemref = hexagonmem.crouton_to_memref (%crouton : !crouton.crouton<64xf32, vtcm>) : memref<64x512xf32, 1>
  %castc = builtin.unrealized_conversion_cast %cmemref : memref<64x512xf32, 1> to !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
  %ptrc = llvm.extractvalue %castc[1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<2 x i64>, array<2 x i64>)>
  %gepc = llvm.getelementptr %ptrc[%c0_i64] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32

  %val0 = llvm.load %gep0 : !llvm.ptr<1> -> f32          // load from arg0
  %val1 = llvm.load %gep1 : !llvm.ptr<1> -> f32          // load from arg1
  %valc = llvm.load %gepc : !llvm.ptr<1> -> f32          // untraced (crouton_to_memref not recognized)
  return
}
// CHECK: %[[T6_GEP0:.+]] = llvm.getelementptr %{{.*}}[%{{.*}}] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
// CHECK: %[[T6_GEP1:.+]] = llvm.getelementptr %{{.*}}[%{{.*}}] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
// CHECK: %[[T6_GEPC:.+]] = llvm.getelementptr %{{.*}}[%{{.*}}] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
// CHECK: %[[T6_V0:.+]] = llvm.load %[[T6_GEP0]] {alias_scopes = [#[[SCOPE0]]], noalias_scopes = [#[[SCOPE1]]]} : !llvm.ptr<1> -> f32
// CHECK: %[[T6_V1:.+]] = llvm.load %[[T6_GEP1]] {alias_scopes = [#[[SCOPE1]]], noalias_scopes = [#[[SCOPE0]]]} : !llvm.ptr<1> -> f32
// CHECK: %[[T6_VC:.+]] = llvm.load %[[T6_GEPC]] : !llvm.ptr<1> -> f32

// -----
// Test 7: A shorter tracing chain — extract_strided_metadata's base_buffer
// feeds a single-operand unrealized_conversion_cast directly, with no
// reinterpret_cast in between. Both loads still trace to their argument
// roots and receive reciprocal alias/noalias scopes.
// CHECK-DAG: #[[DOMAIN:.+]] = #llvm.alias_scope_domain<{{.*}}, description = "extract_strided_metadata_direct">
// CHECK-DAG: #[[SCOPE0:.+]] = #llvm.alias_scope<{{.*}}, domain = #[[DOMAIN]], description = "arg0">
// CHECK-DAG: #[[SCOPE1:.+]] = #llvm.alias_scope<{{.*}}, domain = #[[DOMAIN]], description = "arg1">
// CHECK-LABEL: @extract_strided_metadata_direct
func.func @extract_strided_metadata_direct(%arg0: memref<64xf32, 1>, %arg1: memref<64xf32, 1>) {
  %base0, %offset0, %sizes0, %strides0 = memref.extract_strided_metadata %arg0 : memref<64xf32, 1> -> memref<f32, 1>, index, index, index
  %cast0 = builtin.unrealized_conversion_cast %base0 : memref<f32, 1> to !llvm.struct<(ptr<1>, ptr<1>, i64)>
  %ptr0 = llvm.extractvalue %cast0[1] : !llvm.struct<(ptr<1>, ptr<1>, i64)>
  %c0 = arith.constant 0 : index
  %c0_i64 = builtin.unrealized_conversion_cast %c0 : index to i64
  %gep0 = llvm.getelementptr %ptr0[%c0_i64] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32

  %base1, %offset1, %sizes1, %strides1 = memref.extract_strided_metadata %arg1 : memref<64xf32, 1> -> memref<f32, 1>, index, index, index
  %cast1 = builtin.unrealized_conversion_cast %base1 : memref<f32, 1> to !llvm.struct<(ptr<1>, ptr<1>, i64)>
  %ptr1 = llvm.extractvalue %cast1[1] : !llvm.struct<(ptr<1>, ptr<1>, i64)>
  %gep1 = llvm.getelementptr %ptr1[%c0_i64] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32

  %val0 = llvm.load %gep0 : !llvm.ptr<1> -> f32          // load from arg0
  %val1 = llvm.load %gep1 : !llvm.ptr<1> -> f32          // load from arg1
  return
}
// CHECK: %[[T7_GEP0:.+]] = llvm.getelementptr %{{.*}}[%{{.*}}] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
// CHECK: %[[T7_GEP1:.+]] = llvm.getelementptr %{{.*}}[%{{.*}}] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
// CHECK: %[[T7_V0:.+]] = llvm.load %[[T7_GEP0]] {alias_scopes = [#[[SCOPE0]]], noalias_scopes = [#[[SCOPE1]]]} : !llvm.ptr<1> -> f32
// CHECK: %[[T7_V1:.+]] = llvm.load %[[T7_GEP1]] {alias_scopes = [#[[SCOPE1]]], noalias_scopes = [#[[SCOPE0]]]} : !llvm.ptr<1> -> f32

// -----
// Test 8: A two-operand builtin.unrealized_conversion_cast is not followed
// by traceToRoot() — only single-operand casts are — so arg0's chain is
// left untraced. With only one traceable root (arg1) left, rootToOps.size()
// < 2 and the pass bails out entirely, so arg1 gets no scope either.
// CHECK-LABEL: @multi_operand_cast_untraced
func.func @multi_operand_cast_untraced(%arg0: memref<64xf32, 1>, %arg1: memref<64xf32, 1>) {
  %base0, %offset0, %sizes0, %strides0 = memref.extract_strided_metadata %arg0 : memref<64xf32, 1> -> memref<f32, 1>, index, index, index
  %rc0 = memref.reinterpret_cast %base0 to offset: [0], sizes: [64], strides: [1] : memref<f32, 1> to memref<64xf32, strided<[1]>, 1>
  %cast0 = builtin.unrealized_conversion_cast %rc0 : memref<64xf32, strided<[1]>, 1> to !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
  %c0 = arith.constant 0 : index
  %c0_i64 = builtin.unrealized_conversion_cast %c0 : index to i64
  %multicast = builtin.unrealized_conversion_cast %cast0, %c0_i64 : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>, i64 to !llvm.ptr<1>
  %gep0 = llvm.getelementptr %multicast[%c0_i64] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32

  %base1, %offset1, %sizes1, %strides1 = memref.extract_strided_metadata %arg1 : memref<64xf32, 1> -> memref<f32, 1>, index, index, index
  %rc1 = memref.reinterpret_cast %base1 to offset: [0], sizes: [64], strides: [1] : memref<f32, 1> to memref<64xf32, strided<[1]>, 1>
  %cast1 = builtin.unrealized_conversion_cast %rc1 : memref<64xf32, strided<[1]>, 1> to !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
  %ptr1 = llvm.extractvalue %cast1[1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
  %gep1 = llvm.getelementptr %ptr1[%c0_i64] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32

  %val0 = llvm.load %gep0 : !llvm.ptr<1> -> f32          // untraced (multi-operand cast)
  %val1 = llvm.load %gep1 : !llvm.ptr<1> -> f32          // load from arg1 (but pass bails: <2 traceable roots)
  return
}
// CHECK: %[[T8_GEP0:.+]] = llvm.getelementptr %{{.*}}[%{{.*}}] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
// CHECK: %[[T8_GEP1:.+]] = llvm.getelementptr %{{.*}}[%{{.*}}] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
// CHECK: %[[T8_V0:.+]] = llvm.load %[[T8_GEP0]] : !llvm.ptr<1> -> f32
// CHECK: %[[T8_V1:.+]] = llvm.load %[[T8_GEP1]] : !llvm.ptr<1> -> f32

// -----
// Test 9: llvm.inttoptr is an opaque, unrecognized op, so traceToRoot()
// returns null without crashing. With only one traceable root (arg1) left,
// rootToOps.size() < 2 and the pass bails out entirely.
// CHECK-LABEL: @unknown_defining_op
func.func @unknown_defining_op(%arg0: i64, %arg1: memref<64xf32, 1>) {
  %ptr0 = llvm.inttoptr %arg0 : i64 to !llvm.ptr<1>
  %c0 = arith.constant 0 : index
  %c0_i64 = builtin.unrealized_conversion_cast %c0 : index to i64
  %gep0 = llvm.getelementptr %ptr0[%c0_i64] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32

  %base1, %offset1, %sizes1, %strides1 = memref.extract_strided_metadata %arg1 : memref<64xf32, 1> -> memref<f32, 1>, index, index, index
  %rc1 = memref.reinterpret_cast %base1 to offset: [0], sizes: [64], strides: [1] : memref<f32, 1> to memref<64xf32, strided<[1]>, 1>
  %cast1 = builtin.unrealized_conversion_cast %rc1 : memref<64xf32, strided<[1]>, 1> to !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
  %ptr1 = llvm.extractvalue %cast1[1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
  %gep1 = llvm.getelementptr %ptr1[%c0_i64] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32

  %val0 = llvm.load %gep0 : !llvm.ptr<1> -> f32          // untraced (unrecognized op: llvm.inttoptr)
  %val1 = llvm.load %gep1 : !llvm.ptr<1> -> f32          // load from arg1 (but pass bails: <2 traceable roots)
  return
}
// CHECK: %[[T9_GEP0:.+]] = llvm.getelementptr %{{.*}}[%{{.*}}] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
// CHECK: %[[T9_GEP1:.+]] = llvm.getelementptr %{{.*}}[%{{.*}}] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
// CHECK: %[[T9_V0:.+]] = llvm.load %[[T9_GEP0]] : !llvm.ptr<1> -> f32
// CHECK: %[[T9_V1:.+]] = llvm.load %[[T9_GEP1]] : !llvm.ptr<1> -> f32

// -----
// Test 10: A pointer arriving as a non-entry-block BlockArgument via
// llvm.br is left untraced — traceToRoot() only accepts entry-block
// BlockArguments as roots. With only one traceable root (arg1) left,
// rootToOps.size() < 2 and the pass bails out entirely.
// CHECK-LABEL: @non_entry_block_arg_untraced
func.func @non_entry_block_arg_untraced(%arg0: !llvm.ptr<1>, %arg1: memref<64xf32, 1>) {
  %c0 = arith.constant 0 : index
  %c0_i64 = builtin.unrealized_conversion_cast %c0 : index to i64
  llvm.br ^bb1(%arg0 : !llvm.ptr<1>)
^bb1(%ptr0: !llvm.ptr<1>):
  %gep0 = llvm.getelementptr %ptr0[%c0_i64] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32

  %base1, %offset1, %sizes1, %strides1 = memref.extract_strided_metadata %arg1 : memref<64xf32, 1> -> memref<f32, 1>, index, index, index
  %rc1 = memref.reinterpret_cast %base1 to offset: [0], sizes: [64], strides: [1] : memref<f32, 1> to memref<64xf32, strided<[1]>, 1>
  %cast1 = builtin.unrealized_conversion_cast %rc1 : memref<64xf32, strided<[1]>, 1> to !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
  %ptr1 = llvm.extractvalue %cast1[1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
  %gep1 = llvm.getelementptr %ptr1[%c0_i64] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32

  %val0 = llvm.load %gep0 : !llvm.ptr<1> -> f32          // untraced (non-entry-block arg)
  %val1 = llvm.load %gep1 : !llvm.ptr<1> -> f32          // load from arg1 (but pass bails: <2 traceable roots)
  return
}
// CHECK: %[[T10_GEP0:.+]] = llvm.getelementptr %{{.*}}[%{{.*}}] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
// CHECK: %[[T10_GEP1:.+]] = llvm.getelementptr %{{.*}}[%{{.*}}] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
// CHECK: %[[T10_V0:.+]] = llvm.load %[[T10_GEP0]] : !llvm.ptr<1> -> f32
// CHECK: %[[T10_V1:.+]] = llvm.load %[[T10_GEP1]] : !llvm.ptr<1> -> f32

// -----
// Test 11: The non-entry-block BlockArgument chain from Test 10 stays
// untraced, but a traceable arg root and a traceable alloca root bring
// rootToOps.size() to 2, so the pass still runs. Only the arg/local loads
// get annotated.
// CHECK-DAG: #[[DOMAIN:.+]] = #llvm.alias_scope_domain<{{.*}}, description = "non_entry_block_arg_untraced_with_alloca">
// CHECK-DAG: #[[SCOPE_ARG1:.+]] = #llvm.alias_scope<{{.*}}, domain = #[[DOMAIN]], description = "arg1">
// CHECK-DAG: #[[SCOPE_LOCAL:.+]] = #llvm.alias_scope<{{.*}}, domain = #[[DOMAIN]], description = "local">
// CHECK-LABEL: @non_entry_block_arg_untraced_with_alloca
func.func @non_entry_block_arg_untraced_with_alloca(%arg0: !llvm.ptr<1>, %arg1: memref<64xf32, 1>) {
  %c0 = arith.constant 0 : index
  %c0_i64 = builtin.unrealized_conversion_cast %c0 : index to i64
  llvm.br ^bb1(%arg0 : !llvm.ptr<1>)
^bb1(%ptr0: !llvm.ptr<1>):
  %gep0 = llvm.getelementptr %ptr0[%c0_i64] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32

  %base1, %offset1, %sizes1, %strides1 = memref.extract_strided_metadata %arg1 : memref<64xf32, 1> -> memref<f32, 1>, index, index, index
  %rc1 = memref.reinterpret_cast %base1 to offset: [0], sizes: [64], strides: [1] : memref<f32, 1> to memref<64xf32, strided<[1]>, 1>
  %cast1 = builtin.unrealized_conversion_cast %rc1 : memref<64xf32, strided<[1]>, 1> to !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
  %ptr1 = llvm.extractvalue %cast1[1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
  %gep1 = llvm.getelementptr %ptr1[%c0_i64] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32

  %local = memref.alloca() : memref<64xf32>
  %castl = builtin.unrealized_conversion_cast %local : memref<64xf32> to !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
  %ptrl = llvm.extractvalue %castl[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
  %gepl = llvm.getelementptr %ptrl[%c0_i64] : (!llvm.ptr, i64) -> !llvm.ptr, f32

  %val0 = llvm.load %gep0 : !llvm.ptr<1> -> f32          // untraced (non-entry-block arg)
  %val1 = llvm.load %gep1 : !llvm.ptr<1> -> f32          // load from arg1
  %vall = llvm.load %gepl : !llvm.ptr -> f32             // load from local (alloca)
  return
}
// CHECK: %[[T11_GEP0:.+]] = llvm.getelementptr %{{.*}}[%{{.*}}] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
// CHECK: %[[T11_GEP1:.+]] = llvm.getelementptr %{{.*}}[%{{.*}}] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
// CHECK: %[[T11_GEPL:.+]] = llvm.getelementptr %{{.*}}[%{{.*}}] : (!llvm.ptr, i64) -> !llvm.ptr, f32
// CHECK: %[[T11_V0:.+]] = llvm.load %[[T11_GEP0]] : !llvm.ptr<1> -> f32
// CHECK: %[[T11_V1:.+]] = llvm.load %[[T11_GEP1]] {alias_scopes = [#[[SCOPE_ARG1]]], noalias_scopes = [#[[SCOPE_LOCAL]]]} : !llvm.ptr<1> -> f32
// CHECK: %[[T11_VL:.+]] = llvm.load %[[T11_GEPL]] {alias_scopes = [#[[SCOPE_LOCAL]]], noalias_scopes = [#[[SCOPE_ARG1]]]} : !llvm.ptr -> f32

// -----
// Test 12: A 25-hop GEP chain from a valid root exceeds kMaxSteps=20, so
// the load/store at the end of that chain is left untraced — while two
// other roots (arg0, arg2) in the same function still trace correctly. The
// walk gives up per-chain, not globally.
// CHECK-DAG: #[[DOMAIN:.+]] = #llvm.alias_scope_domain<{{.*}}, description = "exceeds_max_steps">
// CHECK-DAG: #[[SCOPE0:.+]] = #llvm.alias_scope<{{.*}}, domain = #[[DOMAIN]], description = "arg0">
// CHECK-DAG: #[[SCOPE2:.+]] = #llvm.alias_scope<{{.*}}, domain = #[[DOMAIN]], description = "arg2">
// CHECK-LABEL: @exceeds_max_steps
func.func @exceeds_max_steps(%arg0: memref<64xf32, 1>, %arg1: memref<64xf32, 1>, %arg2: memref<64xf32, 1>) {
  %base0, %offset0, %sizes0, %strides0 = memref.extract_strided_metadata %arg0 : memref<64xf32, 1> -> memref<f32, 1>, index, index, index
  %rc0 = memref.reinterpret_cast %base0 to offset: [0], sizes: [64], strides: [1] : memref<f32, 1> to memref<64xf32, strided<[1]>, 1>
  %cast0 = builtin.unrealized_conversion_cast %rc0 : memref<64xf32, strided<[1]>, 1> to !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
  %ptr0 = llvm.extractvalue %cast0[1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
  %c0 = arith.constant 0 : index
  %c0_i64 = builtin.unrealized_conversion_cast %c0 : index to i64
  %gep0 = llvm.getelementptr %ptr0[%c0_i64] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32

  %base2, %offset2, %sizes2, %strides2 = memref.extract_strided_metadata %arg2 : memref<64xf32, 1> -> memref<f32, 1>, index, index, index
  %rc2 = memref.reinterpret_cast %base2 to offset: [0], sizes: [64], strides: [1] : memref<f32, 1> to memref<64xf32, strided<[1]>, 1>
  %cast2 = builtin.unrealized_conversion_cast %rc2 : memref<64xf32, strided<[1]>, 1> to !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
  %ptr2 = llvm.extractvalue %cast2[1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
  %gep2 = llvm.getelementptr %ptr2[%c0_i64] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32

  %base1, %offset1, %sizes1, %strides1 = memref.extract_strided_metadata %arg1 : memref<64xf32, 1> -> memref<f32, 1>, index, index, index
  %rc1 = memref.reinterpret_cast %base1 to offset: [0], sizes: [64], strides: [1] : memref<f32, 1> to memref<64xf32, strided<[1]>, 1>
  %cast1 = builtin.unrealized_conversion_cast %rc1 : memref<64xf32, strided<[1]>, 1> to !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
  %ptr1_0 = llvm.extractvalue %cast1[1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
  %ptr1_1 = llvm.getelementptr %ptr1_0[%c0_i64] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
  %ptr1_2 = llvm.getelementptr %ptr1_1[%c0_i64] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
  %ptr1_3 = llvm.getelementptr %ptr1_2[%c0_i64] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
  %ptr1_4 = llvm.getelementptr %ptr1_3[%c0_i64] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
  %ptr1_5 = llvm.getelementptr %ptr1_4[%c0_i64] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
  %ptr1_6 = llvm.getelementptr %ptr1_5[%c0_i64] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
  %ptr1_7 = llvm.getelementptr %ptr1_6[%c0_i64] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
  %ptr1_8 = llvm.getelementptr %ptr1_7[%c0_i64] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
  %ptr1_9 = llvm.getelementptr %ptr1_8[%c0_i64] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
  %ptr1_10 = llvm.getelementptr %ptr1_9[%c0_i64] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
  %ptr1_11 = llvm.getelementptr %ptr1_10[%c0_i64] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
  %ptr1_12 = llvm.getelementptr %ptr1_11[%c0_i64] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
  %ptr1_13 = llvm.getelementptr %ptr1_12[%c0_i64] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
  %ptr1_14 = llvm.getelementptr %ptr1_13[%c0_i64] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
  %ptr1_15 = llvm.getelementptr %ptr1_14[%c0_i64] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
  %ptr1_16 = llvm.getelementptr %ptr1_15[%c0_i64] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
  %ptr1_17 = llvm.getelementptr %ptr1_16[%c0_i64] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
  %ptr1_18 = llvm.getelementptr %ptr1_17[%c0_i64] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
  %ptr1_19 = llvm.getelementptr %ptr1_18[%c0_i64] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
  %ptr1_20 = llvm.getelementptr %ptr1_19[%c0_i64] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
  %ptr1_21 = llvm.getelementptr %ptr1_20[%c0_i64] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
  %ptr1_22 = llvm.getelementptr %ptr1_21[%c0_i64] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
  %ptr1_23 = llvm.getelementptr %ptr1_22[%c0_i64] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
  %ptr1_24 = llvm.getelementptr %ptr1_23[%c0_i64] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32

  %val0 = llvm.load %gep0 : !llvm.ptr<1> -> f32          // load from arg0
  %val2 = llvm.load %gep2 : !llvm.ptr<1> -> f32          // load from arg2
  llvm.store %val0, %gep2 : f32, !llvm.ptr<1>            // store to arg2
  llvm.store %val2, %gep0 : f32, !llvm.ptr<1>            // store to arg0
  %val1 = llvm.load %ptr1_24 : !llvm.ptr<1> -> f32       // untraced (exceeds kMaxSteps, 25-hop chain from arg1)
  llvm.store %val1, %ptr1_24 : f32, !llvm.ptr<1>         // untraced (exceeds kMaxSteps, 25-hop chain from arg1)
  return
}
// CHECK: %[[T12_GEP0:.+]] = llvm.getelementptr %{{.*}}[%{{.*}}] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
// CHECK: %[[T12_GEP2:.+]] = llvm.getelementptr %{{.*}}[%{{.*}}] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
// CHECK: %[[T12_V0:.+]] = llvm.load %[[T12_GEP0]] {alias_scopes = [#[[SCOPE0]]], noalias_scopes = [#[[SCOPE2]]]} : !llvm.ptr<1> -> f32
// CHECK: %[[T12_V2:.+]] = llvm.load %[[T12_GEP2]] {alias_scopes = [#[[SCOPE2]]], noalias_scopes = [#[[SCOPE0]]]} : !llvm.ptr<1> -> f32
// CHECK: llvm.store %[[T12_V0]], %[[T12_GEP2]] {alias_scopes = [#[[SCOPE2]]], noalias_scopes = [#[[SCOPE0]]]} : f32, !llvm.ptr<1>
// CHECK: llvm.store %[[T12_V2]], %[[T12_GEP0]] {alias_scopes = [#[[SCOPE0]]], noalias_scopes = [#[[SCOPE2]]]} : f32, !llvm.ptr<1>
// CHECK: %[[T12_V1:.+]] = llvm.load %[[T12_LONGPTR:.+]] : !llvm.ptr<1> -> f32
// CHECK: llvm.store %[[T12_V1]], %[[T12_LONGPTR]] : f32, !llvm.ptr<1>

// -----
// Test 13: Existing alias_scopes/noalias_scopes are merged, not
// overwritten. The load starts with [pre_scope] on alias_scopes and
// [pre_noalias_scope] on noalias_scopes; the pass appends arg0 to
// alias_scopes and arg1 to noalias_scopes, preserving the pre-existing
// entries first. The input IR also predefines its own domain (#pre_domain)
// for both — the pass must not reuse it: it always builds a fresh domain
// named after the function, so the pre-existing scopes keep #pre_domain
// while arg0/arg1 get the new one.
// CHECK-DAG: #[[PRE_DOMAIN:.+]] = #llvm.alias_scope_domain<{{.*}}, description = "pre_domain">
// CHECK-DAG: #[[PRE_SCOPE:.+]] = #llvm.alias_scope<{{.*}}, domain = #[[PRE_DOMAIN]], description = "pre_scope">
// CHECK-DAG: #[[PRE_NOALIAS_SCOPE:.+]] = #llvm.alias_scope<{{.*}}, domain = #[[PRE_DOMAIN]], description = "pre_noalias_scope">
// CHECK-DAG: #[[DOMAIN:.+]] = #llvm.alias_scope_domain<{{.*}}, description = "merge_existing_metadata">
// CHECK-DAG: #[[SCOPE0:.+]] = #llvm.alias_scope<{{.*}}, domain = #[[DOMAIN]], description = "arg0">
// CHECK-DAG: #[[SCOPE1:.+]] = #llvm.alias_scope<{{.*}}, domain = #[[DOMAIN]], description = "arg1">
// CHECK-LABEL: @merge_existing_metadata
#pre_domain = #llvm.alias_scope_domain<id = distinct[0]<>, description = "pre_domain">
#pre_scope = #llvm.alias_scope<id = distinct[1]<>, domain = #pre_domain, description = "pre_scope">
#pre_noalias_scope = #llvm.alias_scope<id = distinct[2]<>, domain = #pre_domain, description = "pre_noalias_scope">
func.func @merge_existing_metadata(%arg0: memref<64xf32, 1>, %arg1: memref<64xf32, 1>) {
  %base0, %offset0, %sizes0, %strides0 = memref.extract_strided_metadata %arg0 : memref<64xf32, 1> -> memref<f32, 1>, index, index, index
  %rc0 = memref.reinterpret_cast %base0 to offset: [0], sizes: [64], strides: [1] : memref<f32, 1> to memref<64xf32, strided<[1]>, 1>
  %cast0 = builtin.unrealized_conversion_cast %rc0 : memref<64xf32, strided<[1]>, 1> to !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
  %ptr0 = llvm.extractvalue %cast0[1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>

  %base1, %offset1, %sizes1, %strides1 = memref.extract_strided_metadata %arg1 : memref<64xf32, 1> -> memref<f32, 1>, index, index, index
  %rc1 = memref.reinterpret_cast %base1 to offset: [0], sizes: [64], strides: [1] : memref<f32, 1> to memref<64xf32, strided<[1]>, 1>
  %cast1 = builtin.unrealized_conversion_cast %rc1 : memref<64xf32, strided<[1]>, 1> to !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
  %ptr1 = llvm.extractvalue %cast1[1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>

  %c0 = arith.constant 0 : index
  %c0_i64 = builtin.unrealized_conversion_cast %c0 : index to i64
  %gep0 = llvm.getelementptr %ptr0[%c0_i64] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
  %gep1 = llvm.getelementptr %ptr1[%c0_i64] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32

  %val = llvm.load %gep0 {alias_scopes = [#pre_scope], noalias_scopes = [#pre_noalias_scope]} : !llvm.ptr<1> -> f32   // load from arg0
  llvm.store %val, %gep1 : f32, !llvm.ptr<1>             // store to arg1
  return
}
// CHECK: %[[T13_GEP0:.+]] = llvm.getelementptr %{{.*}}[%{{.*}}] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
// CHECK: %[[T13_GEP1:.+]] = llvm.getelementptr %{{.*}}[%{{.*}}] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
// CHECK: %[[T13_VAL:.+]] = llvm.load %[[T13_GEP0]] {alias_scopes = [#[[PRE_SCOPE]], #[[SCOPE0]]], noalias_scopes = [#[[PRE_NOALIAS_SCOPE]], #[[SCOPE1]]]} : !llvm.ptr<1> -> f32
// CHECK: llvm.store %[[T13_VAL]], %[[T13_GEP1]] {alias_scopes = [#[[SCOPE1]]], noalias_scopes = [#[[SCOPE0]]]} : f32, !llvm.ptr<1>

// -----
// Test 14: Four distinct local allocations are treated as four separate
// roots, but none is stamped noalias against another — a local-allocation
// root's noalias_scopes list omits other local roots, since BasicAA already
// proves two fresh allocations disjoint without this pass's help. Function arguments are present
// but deliberately unused, to isolate local-vs-local behavior; each load's
// alias_scopes still names its own root, but noalias_scopes is empty for
// all four, since every other root here is also local.
// CHECK-DAG: #[[DOMAIN:alias_scope_domain]] = #llvm.alias_scope_domain<{{.*}}, description = "four_alloca_roots_three_args">
// CHECK-DAG: #[[LS0:alias_scope]] = #llvm.alias_scope<{{.*}}, domain = #[[DOMAIN]], description = "local">
// CHECK-DAG: #[[LS1:alias_scope1]] = #llvm.alias_scope<{{.*}}, domain = #[[DOMAIN]], description = "local">
// CHECK-DAG: #[[LS2:alias_scope2]] = #llvm.alias_scope<{{.*}}, domain = #[[DOMAIN]], description = "local">
// CHECK-DAG: #[[LS3:alias_scope3]] = #llvm.alias_scope<{{.*}}, domain = #[[DOMAIN]], description = "local">
// CHECK-LABEL: @four_alloca_roots_three_args
func.func @four_alloca_roots_three_args(%arg0: memref<64xf32, 1>, %arg1: memref<64xf32, 1>, %arg2: memref<64xf32, 1>) {
  %local0 = memref.alloca() : memref<64xf32>
  %local1 = memref.alloca() : memref<64xf32>
  %local2 = memref.alloca() : memref<64xf32>
  %local3 = memref.alloca() : memref<64xf32>
  %cast0 = builtin.unrealized_conversion_cast %local0 : memref<64xf32> to !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
  %cast1 = builtin.unrealized_conversion_cast %local1 : memref<64xf32> to !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
  %cast2 = builtin.unrealized_conversion_cast %local2 : memref<64xf32> to !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
  %cast3 = builtin.unrealized_conversion_cast %local3 : memref<64xf32> to !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
  %ptr0 = llvm.extractvalue %cast0[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
  %ptr1 = llvm.extractvalue %cast1[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
  %ptr2 = llvm.extractvalue %cast2[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
  %ptr3 = llvm.extractvalue %cast3[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
  %c0 = arith.constant 0 : index
  %c0_i64 = builtin.unrealized_conversion_cast %c0 : index to i64
  %gep0 = llvm.getelementptr %ptr0[%c0_i64] : (!llvm.ptr, i64) -> !llvm.ptr, f32
  %gep1 = llvm.getelementptr %ptr1[%c0_i64] : (!llvm.ptr, i64) -> !llvm.ptr, f32
  %gep2 = llvm.getelementptr %ptr2[%c0_i64] : (!llvm.ptr, i64) -> !llvm.ptr, f32
  %gep3 = llvm.getelementptr %ptr3[%c0_i64] : (!llvm.ptr, i64) -> !llvm.ptr, f32
  %val0 = llvm.load %gep0 : !llvm.ptr -> f32             // load from local0 (alloca)
  %val1 = llvm.load %gep1 : !llvm.ptr -> f32             // load from local1 (alloca)
  %val2 = llvm.load %gep2 : !llvm.ptr -> f32             // load from local2 (alloca)
  %val3 = llvm.load %gep3 : !llvm.ptr -> f32             // load from local3 (alloca)
  return
}
// CHECK: %[[T14_GEP0:.+]] = llvm.getelementptr %{{.*}}[%{{.*}}] : (!llvm.ptr, i64) -> !llvm.ptr, f32
// CHECK: %[[T14_GEP1:.+]] = llvm.getelementptr %{{.*}}[%{{.*}}] : (!llvm.ptr, i64) -> !llvm.ptr, f32
// CHECK: %[[T14_GEP2:.+]] = llvm.getelementptr %{{.*}}[%{{.*}}] : (!llvm.ptr, i64) -> !llvm.ptr, f32
// CHECK: %[[T14_GEP3:.+]] = llvm.getelementptr %{{.*}}[%{{.*}}] : (!llvm.ptr, i64) -> !llvm.ptr, f32
// CHECK: llvm.load %[[T14_GEP0]] {alias_scopes = [#[[LS0]]], noalias_scopes = []} : !llvm.ptr -> f32
// CHECK: llvm.load %[[T14_GEP1]] {alias_scopes = [#[[LS1]]], noalias_scopes = []} : !llvm.ptr -> f32
// CHECK: llvm.load %[[T14_GEP2]] {alias_scopes = [#[[LS2]]], noalias_scopes = []} : !llvm.ptr -> f32
// CHECK: llvm.load %[[T14_GEP3]] {alias_scopes = [#[[LS3]]], noalias_scopes = []} : !llvm.ptr -> f32

// -----
// Test 15: Storing an arg-loaded value into local memory does not change
// root provenance — traceToRoot() follows pointer chains, not the data
// flowing through them, so the arg-root and local-root ops stay separately
// scoped.
// CHECK-DAG: #[[DOMAIN:.+]] = #llvm.alias_scope_domain<{{.*}}, description = "arg_value_stored_to_local">
// CHECK-DAG: #[[ARG_SCOPE:.+]] = #llvm.alias_scope<{{.*}}, domain = #[[DOMAIN]], description = "arg0">
// CHECK-DAG: #[[LOCAL_SCOPE:.+]] = #llvm.alias_scope<{{.*}}, domain = #[[DOMAIN]], description = "local">
// CHECK-LABEL: @arg_value_stored_to_local
func.func @arg_value_stored_to_local(%arg0: memref<64xf32, 1>) {
  %base0, %offset0, %sizes0, %strides0 = memref.extract_strided_metadata %arg0 : memref<64xf32, 1> -> memref<f32, 1>, index, index, index
  %rc0 = memref.reinterpret_cast %base0 to offset: [0], sizes: [64], strides: [1] : memref<f32, 1> to memref<64xf32, strided<[1]>, 1>
  %cast0 = builtin.unrealized_conversion_cast %rc0 : memref<64xf32, strided<[1]>, 1> to !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
  %ptr0 = llvm.extractvalue %cast0[1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>

  %local = memref.alloca() : memref<64xf32>
  %castl = builtin.unrealized_conversion_cast %local : memref<64xf32> to !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
  %ptrl = llvm.extractvalue %castl[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>

  %c0 = arith.constant 0 : index
  %c0_i64 = builtin.unrealized_conversion_cast %c0 : index to i64
  %gep0 = llvm.getelementptr %ptr0[%c0_i64] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
  %gepl = llvm.getelementptr %ptrl[%c0_i64] : (!llvm.ptr, i64) -> !llvm.ptr, f32

  %v = llvm.load %gep0 : !llvm.ptr<1> -> f32             // load from arg0
  llvm.store %v, %gepl : f32, !llvm.ptr                  // store to local (alloca)
  %v_local = llvm.load %gepl : !llvm.ptr -> f32          // load from local (alloca)
  return
}
// CHECK: %[[T15_GEP0:.+]] = llvm.getelementptr %{{.*}}[%{{.*}}] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
// CHECK: %[[T15_GEPL:.+]] = llvm.getelementptr %{{.*}}[%{{.*}}] : (!llvm.ptr, i64) -> !llvm.ptr, f32
// CHECK: %[[T15_V:.+]] = llvm.load %[[T15_GEP0]] {alias_scopes = [#[[ARG_SCOPE]]], noalias_scopes = [#[[LOCAL_SCOPE]]]} : !llvm.ptr<1> -> f32
// CHECK: llvm.store %[[T15_V]], %[[T15_GEPL]] {alias_scopes = [#[[LOCAL_SCOPE]]], noalias_scopes = [#[[ARG_SCOPE]]]} : f32, !llvm.ptr
// CHECK: %[[T15_VL:.+]] = llvm.load %[[T15_GEPL]] {alias_scopes = [#[[LOCAL_SCOPE]]], noalias_scopes = [#[[ARG_SCOPE]]]} : !llvm.ptr -> f32

// -----
// Test 16: memref.alloc/alloca results are semantically fresh, so "alloc
// shares storage with an arg or another alloc" has no direct representation
// in typed memref IR. The closest representable stand-in is an opaque
// pointer round-trip (ptrtoint -> inttoptr), which could in principle hide
// real provenance — that chain must stay untraced rather than guessed at.
// CHECK-DAG: #[[T16_DOMAIN:.+]] = #llvm.alias_scope_domain<{{.*}}, description = "opaque_ptr_roundtrip_untraced">
// CHECK-DAG: #[[T16_ARG_SCOPE:.+]] = #llvm.alias_scope<{{.*}}, domain = #[[T16_DOMAIN]], description = "arg0">
// CHECK-DAG: #[[T16_LOCAL_SCOPE:.+]] = #llvm.alias_scope<{{.*}}, domain = #[[T16_DOMAIN]], description = "local">
// CHECK-LABEL: @opaque_ptr_roundtrip_untraced
func.func @opaque_ptr_roundtrip_untraced(%arg0: memref<64xf32, 1>) {
  %base0, %offset0, %sizes0, %strides0 = memref.extract_strided_metadata %arg0 : memref<64xf32, 1> -> memref<f32, 1>, index, index, index
  %rc0 = memref.reinterpret_cast %base0 to offset: [0], sizes: [64], strides: [1] : memref<f32, 1> to memref<64xf32, strided<[1]>, 1>
  %cast0 = builtin.unrealized_conversion_cast %rc0 : memref<64xf32, strided<[1]>, 1> to !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
  %ptr0 = llvm.extractvalue %cast0[1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>

  %local = memref.alloca() : memref<64xf32>
  %castl = builtin.unrealized_conversion_cast %local : memref<64xf32> to !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
  %ptrl = llvm.extractvalue %castl[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>

  %c0 = arith.constant 0 : index
  %c0_i64 = builtin.unrealized_conversion_cast %c0 : index to i64
  %gep_arg = llvm.getelementptr %ptr0[%c0_i64] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
  %gep_local = llvm.getelementptr %ptrl[%c0_i64] : (!llvm.ptr, i64) -> !llvm.ptr, f32

  %p2i = llvm.ptrtoint %ptr0 : !llvm.ptr<1> to i64
  %opaque_ptr = llvm.inttoptr %p2i : i64 to !llvm.ptr<1>
  %gep_opaque = llvm.getelementptr %opaque_ptr[%c0_i64] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32

  %v_arg = llvm.load %gep_arg : !llvm.ptr<1> -> f32      // load from arg0
  %v_local = llvm.load %gep_local : !llvm.ptr -> f32     // load from local (alloca)
  %v_opaque = llvm.load %gep_opaque : !llvm.ptr<1> -> f32  // untraced (unrecognized op: llvm.inttoptr, via ptrtoint/inttoptr roundtrip)
  return
}
// CHECK: %[[T16_GEP_ARG:.+]] = llvm.getelementptr %{{.*}}[%{{.*}}] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
// CHECK: %[[T16_GEP_LOCAL:.+]] = llvm.getelementptr %{{.*}}[%{{.*}}] : (!llvm.ptr, i64) -> !llvm.ptr, f32
// CHECK: %[[T16_GEP_OPAQUE:.+]] = llvm.getelementptr %{{.*}}[%{{.*}}] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
// CHECK: %[[T16_V_ARG:.+]] = llvm.load %[[T16_GEP_ARG]] {alias_scopes = [#[[T16_ARG_SCOPE]]], noalias_scopes = [#[[T16_LOCAL_SCOPE]]]} : !llvm.ptr<1> -> f32
// CHECK: %[[T16_V_LOCAL:.+]] = llvm.load %[[T16_GEP_LOCAL]] {alias_scopes = [#[[T16_LOCAL_SCOPE]]], noalias_scopes = [#[[T16_ARG_SCOPE]]]} : !llvm.ptr -> f32
// CHECK: %[[T16_V_OPAQUE:.+]] = llvm.load %[[T16_GEP_OPAQUE]] : !llvm.ptr<1> -> f32

// -----
// Test 17: A pointer *value* (arg1's own base pointer) is stored as ordinary
// data into arg0's storage, then reloaded and dereferenced. The store itself
// is scoped correctly to arg0 (traceToRoot() only ever inspects the address
// operand via getPointerOperand(), never the value operand — so it never
// looks at *what* is being written, only *where*). The reload is also scoped
// correctly to arg0, since it's a genuine access to arg0's memory. But
// llvm.load is not one of traceToRoot()'s recognized pass-through hops
// (only GEPOp/ExtractValueOp/single-operand casts/reinterpret_cast/
// extract_strided_metadata are), so the GEP built on the *reloaded pointer*
// cannot be traced further back through it — the final dereference stays
// untraced, rather than being (mis)resolved back to arg0's scope. Resolving
// it to arg0 would be a wrong claim: the reloaded pointer's actual
// provenance is arg1, so an access through it can land in arg1's memory.
// CHECK-DAG: #[[T17_DOMAIN:.+]] = #llvm.alias_scope_domain<{{.*}}, description = "ptr_value_roundtrip_through_memory_untraced">
// CHECK-DAG: #[[T17_SCOPE0:.+]] = #llvm.alias_scope<{{.*}}, domain = #[[T17_DOMAIN]], description = "arg0">
// CHECK-DAG: #[[T17_SCOPE1:.+]] = #llvm.alias_scope<{{.*}}, domain = #[[T17_DOMAIN]], description = "arg1">
// CHECK-LABEL: @ptr_value_roundtrip_through_memory_untraced
func.func @ptr_value_roundtrip_through_memory_untraced(%arg0: memref<64xf32, 1>, %arg1: memref<64xf32, 1>) {
  %base0, %offset0, %sizes0, %strides0 = memref.extract_strided_metadata %arg0 : memref<64xf32, 1> -> memref<f32, 1>, index, index, index
  %rc0 = memref.reinterpret_cast %base0 to offset: [0], sizes: [64], strides: [1] : memref<f32, 1> to memref<64xf32, strided<[1]>, 1>
  %cast0 = builtin.unrealized_conversion_cast %rc0 : memref<64xf32, strided<[1]>, 1> to !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
  %ptr0 = llvm.extractvalue %cast0[1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>

  %base1, %offset1, %sizes1, %strides1 = memref.extract_strided_metadata %arg1 : memref<64xf32, 1> -> memref<f32, 1>, index, index, index
  %rc1 = memref.reinterpret_cast %base1 to offset: [0], sizes: [64], strides: [1] : memref<f32, 1> to memref<64xf32, strided<[1]>, 1>
  %cast1 = builtin.unrealized_conversion_cast %rc1 : memref<64xf32, strided<[1]>, 1> to !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
  %ptr1 = llvm.extractvalue %cast1[1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>

  %c0 = arith.constant 0 : index
  %c0_i64 = builtin.unrealized_conversion_cast %c0 : index to i64
  %gep0 = llvm.getelementptr %ptr0[%c0_i64] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
  %gep1 = llvm.getelementptr %ptr1[%c0_i64] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32

  // Ordinary loads establishing both arg0 and arg1 as real, traced roots
  // (without these, arg1 would never enter rootToOps and the pass would
  // bail out on rootToOps.size() < 2 before building any scopes at all).
  %v0 = llvm.load %gep0 : !llvm.ptr<1> -> f32                 // load from arg0
  %v1 = llvm.load %gep1 : !llvm.ptr<1> -> f32                 // load from arg1

  // Write arg1's own base pointer, as plain data, into arg0's storage.
  llvm.store %ptr1, %gep0 : !llvm.ptr<1>, !llvm.ptr<1>        // store to arg0 (writes arg1's pointer value as data)

  // Read that pointer back out of arg0's storage, then dereference it.
  %reloaded = llvm.load %gep0 : !llvm.ptr<1> -> !llvm.ptr<1>  // load from arg0 (yields a pointer value, not dereferenced yet)
  %gepX = llvm.getelementptr %reloaded[%c0_i64] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
  %v_via_reload = llvm.load %gepX : !llvm.ptr<1> -> f32       // untraced (chain passes through llvm.load, not a recognized hop)
  return
}
// CHECK: %[[T17_GEP0:.+]] = llvm.getelementptr %{{.*}}[%{{.*}}] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
// CHECK: %[[T17_GEP1:.+]] = llvm.getelementptr %{{.*}}[%{{.*}}] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
// CHECK: %[[T17_V0:.+]] = llvm.load %[[T17_GEP0]] {alias_scopes = [#[[T17_SCOPE0]]], noalias_scopes = [#[[T17_SCOPE1]]]} : !llvm.ptr<1> -> f32
// CHECK: %[[T17_V1:.+]] = llvm.load %[[T17_GEP1]] {alias_scopes = [#[[T17_SCOPE1]]], noalias_scopes = [#[[T17_SCOPE0]]]} : !llvm.ptr<1> -> f32
// CHECK: llvm.store %{{.*}}, %[[T17_GEP0]] {alias_scopes = [#[[T17_SCOPE0]]], noalias_scopes = [#[[T17_SCOPE1]]]} : !llvm.ptr<1>, !llvm.ptr<1>
// CHECK: %[[T17_RELOADED:.+]] = llvm.load %[[T17_GEP0]] {alias_scopes = [#[[T17_SCOPE0]]], noalias_scopes = [#[[T17_SCOPE1]]]} : !llvm.ptr<1> -> !llvm.ptr<1>
// CHECK: %[[T17_GEPX:.+]] = llvm.getelementptr %[[T17_RELOADED]][%{{.*}}] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
// CHECK: %[[T17_V_RELOAD:.+]] = llvm.load %[[T17_GEPX]] : !llvm.ptr<1> -> f32

// -----
// Test 18: Two arguments and two locals in the same function. Confirms the
// local-vs-local noalias skip (Test 14) composes correctly with arg-vs-arg
// and arg-vs-local claims when both kinds of root coexist: each local's
// noalias_scopes names only the two args, never the other local; each arg's
// noalias_scopes names the other arg AND both locals, unaffected by the
// skip (the skip is keyed on the CURRENT op's root being local, not on
// whether a local root exists anywhere in the function).
// CHECK-DAG: #[[T18_DOMAIN:.+]] = #llvm.alias_scope_domain<{{.*}}, description = "two_args_two_locals_mixed_noalias">
// CHECK-DAG: #[[T18_ARG0:.+]] = #llvm.alias_scope<{{.*}}, domain = #[[T18_DOMAIN]], description = "arg0">
// CHECK-DAG: #[[T18_ARG1:.+]] = #llvm.alias_scope<{{.*}}, domain = #[[T18_DOMAIN]], description = "arg1">
// CHECK-DAG: #[[T18_LS0:.+]] = #llvm.alias_scope<{{.*}}, domain = #[[T18_DOMAIN]], description = "local">
// CHECK-DAG: #[[T18_LS1:.+]] = #llvm.alias_scope<{{.*}}, domain = #[[T18_DOMAIN]], description = "local">
// CHECK-LABEL: @two_args_two_locals_mixed_noalias
func.func @two_args_two_locals_mixed_noalias(%arg0: memref<64xf32, 1>, %arg1: memref<64xf32, 1>) {
  %base0, %offset0, %sizes0, %strides0 = memref.extract_strided_metadata %arg0 : memref<64xf32, 1> -> memref<f32, 1>, index, index, index
  %rc0 = memref.reinterpret_cast %base0 to offset: [0], sizes: [64], strides: [1] : memref<f32, 1> to memref<64xf32, strided<[1]>, 1>
  %cast0 = builtin.unrealized_conversion_cast %rc0 : memref<64xf32, strided<[1]>, 1> to !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
  %ptr0 = llvm.extractvalue %cast0[1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>

  %base1, %offset1, %sizes1, %strides1 = memref.extract_strided_metadata %arg1 : memref<64xf32, 1> -> memref<f32, 1>, index, index, index
  %rc1 = memref.reinterpret_cast %base1 to offset: [0], sizes: [64], strides: [1] : memref<f32, 1> to memref<64xf32, strided<[1]>, 1>
  %cast1 = builtin.unrealized_conversion_cast %rc1 : memref<64xf32, strided<[1]>, 1> to !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>
  %ptr1 = llvm.extractvalue %cast1[1] : !llvm.struct<(ptr<1>, ptr<1>, i64, array<1 x i64>, array<1 x i64>)>

  %local0 = memref.alloca() : memref<64xf32>
  %castl0 = builtin.unrealized_conversion_cast %local0 : memref<64xf32> to !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
  %ptrl0 = llvm.extractvalue %castl0[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>

  %local1 = memref.alloca() : memref<64xf32>
  %castl1 = builtin.unrealized_conversion_cast %local1 : memref<64xf32> to !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
  %ptrl1 = llvm.extractvalue %castl1[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>

  %c0 = arith.constant 0 : index
  %c0_i64 = builtin.unrealized_conversion_cast %c0 : index to i64
  %gep0 = llvm.getelementptr %ptr0[%c0_i64] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
  %gep1 = llvm.getelementptr %ptr1[%c0_i64] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
  %gepl0 = llvm.getelementptr %ptrl0[%c0_i64] : (!llvm.ptr, i64) -> !llvm.ptr, f32
  %gepl1 = llvm.getelementptr %ptrl1[%c0_i64] : (!llvm.ptr, i64) -> !llvm.ptr, f32

  %val0 = llvm.load %gep0 : !llvm.ptr<1> -> f32          // load from arg0
  %val1 = llvm.load %gep1 : !llvm.ptr<1> -> f32          // load from arg1
  %vall0 = llvm.load %gepl0 : !llvm.ptr -> f32           // load from local0 (alloca)
  %vall1 = llvm.load %gepl1 : !llvm.ptr -> f32           // load from local1 (alloca)
  return
}
// CHECK: %[[T18_GEP0:.+]] = llvm.getelementptr %{{.*}}[%{{.*}}] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
// CHECK: %[[T18_GEP1:.+]] = llvm.getelementptr %{{.*}}[%{{.*}}] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
// CHECK: %[[T18_GEPL0:.+]] = llvm.getelementptr %{{.*}}[%{{.*}}] : (!llvm.ptr, i64) -> !llvm.ptr, f32
// CHECK: %[[T18_GEPL1:.+]] = llvm.getelementptr %{{.*}}[%{{.*}}] : (!llvm.ptr, i64) -> !llvm.ptr, f32
// CHECK: llvm.load %[[T18_GEP0]] {alias_scopes = [#[[T18_ARG0]]], noalias_scopes = [#[[T18_ARG1]], #[[T18_LS0]], #[[T18_LS1]]]} : !llvm.ptr<1> -> f32
// CHECK: llvm.load %[[T18_GEP1]] {alias_scopes = [#[[T18_ARG1]]], noalias_scopes = [#[[T18_ARG0]], #[[T18_LS0]], #[[T18_LS1]]]} : !llvm.ptr<1> -> f32
// CHECK: llvm.load %[[T18_GEPL0]] {alias_scopes = [#[[T18_LS0]]], noalias_scopes = [#[[T18_ARG0]], #[[T18_ARG1]]]} : !llvm.ptr -> f32
// CHECK: llvm.load %[[T18_GEPL1]] {alias_scopes = [#[[T18_LS1]]], noalias_scopes = [#[[T18_ARG0]], #[[T18_ARG1]]]} : !llvm.ptr -> f32
