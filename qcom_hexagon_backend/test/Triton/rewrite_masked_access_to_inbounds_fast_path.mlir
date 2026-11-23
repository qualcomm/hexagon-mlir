//===- rewrite_masked_access_to_inbounds_fast_path.mlir -------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

// RUN: %PYTHON %triton_pass_opt hexagon-rewrite-masked-access-to-inbounds-fast-path %s | FileCheck %s
// RUN: %PYTHON %triton_pass_opt hexagon-rewrite-masked-access-to-inbounds-fast-path %s | FileCheck %s --check-prefix=STATS

// Tests for RewriteMaskedAccessToInboundsFastPath.
//
// The pass matches the single-sided boundary mask PyTorch inductor emits on a
// 1-D pointwise tt.load / tt.store:
//
//   %pid    = tt.get_program_id x
//   %xoff   = %pid * XBLOCK
//   %idx    = splat(%xoff) + make_range(0, XBLOCK)
//   %mask   = arith.cmpi slt, %idx, splat(%xnumel)
//   tt.load  %ptr, %mask
//   tt.store %ptr, %val, %mask
//
// and the 2-D inductor idiom:
//
//   %xmask = cmpi slt, expand_dims(xidx, 1), splat(%xnumel) -> [XBLOCK, 1]
//   %ymask = cmpi slt, expand_dims(yidx, 0), splat(%ynumel) -> [1, YBLOCK]
//   %mask  = andi(broadcast(%xmask), broadcast(%ymask))     -> [XBLOCK, YBLOCK]
//
// Strategy:
//   Loads  — mask dropped entirely; all programs load all lanes
//            unconditionally (OOB lanes on the remainder tile read garbage
//            which is discarded by the masked store).
//   Stores — a single scalar in-bounds check per store:
//     1D:    %hi  = %xoff + (XBLOCK - 1)
//            %inb = arith.cmpi slt, %hi, %xnumel
//     2D:    %inb = andi(cmpi slt, xhi, xnumel, cmpi slt, yhi, ynumel)
//            scf.if %inb { <unmasked store> } else { <original masked store> }
//
// so interior "perfect" tiles take the unmasked fast path and only the
// remainder tile pays for the mask. xnumel / ynumel may be runtime i32 args.

// Test 1 (pointwise_runtime_xnumel): the canonical inductor idiom with a
// runtime xnumel. Loads become unconditionally unmasked; the store is wrapped
// in a single scf.if with an unmasked then-block and the original masked store
// in the else-block. The scalar in-bounds check is `xoffset + (XBLOCK-1) <
// xnumel`, all i32.
// CHECK-LABEL: @pointwise_runtime_xnumel
// CHECK:       %[[LOAD:.*]] = tt.load %{{.*}} : tensor<256x!tt.ptr<f32>>
// CHECK-NOT:   scf.if {{.*}} -> (tensor<256xf32>)
// CHECK:       %[[C255:.*]] = arith.constant 255 : i32
// CHECK:       %[[HI:.*]] = arith.addi %{{.*}}, %[[C255]] : i32
// CHECK:       %[[INB:.*]] = arith.cmpi slt, %[[HI]], %{{.*}} : i32
// CHECK:       scf.if %[[INB]] {
// CHECK:         tt.store %{{.*}}, %[[LOAD]] : tensor<256x!tt.ptr<f32>>
// CHECK:       } else {
// CHECK:         tt.store %{{.*}}, %[[LOAD]], %{{.*}} : tensor<256x!tt.ptr<f32>>
// CHECK:       }

// Test 2 (bails_on_non_pid_offset): the offset is pid*XBLOCK + pid (not a
// clean pid*XBLOCK) so the safety predicate fails -> op left untouched.
// CHECK-LABEL: @bails_on_non_pid_offset
// CHECK:       tt.load %{{.*}}, %{{.*}} : tensor<256x!tt.ptr<f32>>
// CHECK-NOT:   scf.if

// Test 3 (bails_on_multi_leaf_mask): a 2-D andi mask where BOTH arms follow
// the clean pid*BLOCK pattern — rewritten by the 2-D extension.
// No store in this function so no scf.if is expected.
// CHECK-LABEL: @bails_on_multi_leaf_mask
// CHECK:       tt.load %{{.*}} : tensor<16x64x!tt.ptr<f32>>
// CHECK-NOT:   scf.if

// STATS: [RewriteMaskedAccessToInboundsFastPath] masked_accesses=17 rewritten=8

module {

tt.func public @pointwise_runtime_xnumel(%base: !tt.ptr<f32>, %out: !tt.ptr<f32>, %xnumel: i32) {
  %c256 = arith.constant 256 : i32
  %pid = tt.get_program_id x : i32
  %xoff = arith.muli %pid, %c256 : i32
  %range = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32>
  %soff = tt.splat %xoff : i32 -> tensor<256xi32>
  %idx = arith.addi %soff, %range : tensor<256xi32>
  %snum = tt.splat %xnumel : i32 -> tensor<256xi32>
  %mask = arith.cmpi slt, %idx, %snum : tensor<256xi32>
  %pbase = tt.splat %base : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
  %p = tt.addptr %pbase, %idx : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
  %v = tt.load %p, %mask : tensor<256x!tt.ptr<f32>>
  %pout = tt.splat %out : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
  %po = tt.addptr %pout, %idx : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
  tt.store %po, %v, %mask : tensor<256x!tt.ptr<f32>>
  tt.return
}

tt.func public @bails_on_non_pid_offset(%base: !tt.ptr<f32>, %xnumel: i32) -> tensor<256xf32> {
  %c256 = arith.constant 256 : i32
  %pid = tt.get_program_id x : i32
  %mul = arith.muli %pid, %c256 : i32
  // xoff = pid*256 + pid  -> not a clean pid*XBLOCK
  %xoff = arith.addi %mul, %pid : i32
  %range = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32>
  %soff = tt.splat %xoff : i32 -> tensor<256xi32>
  %idx = arith.addi %soff, %range : tensor<256xi32>
  %snum = tt.splat %xnumel : i32 -> tensor<256xi32>
  %mask = arith.cmpi slt, %idx, %snum : tensor<256xi32>
  %pbase = tt.splat %base : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
  %p = tt.addptr %pbase, %idx : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
  %v = tt.load %p, %mask : tensor<256x!tt.ptr<f32>>
  tt.return %v : tensor<256xf32>
}

tt.func public @bails_on_multi_leaf_mask(%base: !tt.ptr<f32>, %off0: i32, %off1: i32) -> tensor<16x64xf32> {
  %c16 = arith.constant 16 : i32
  %c64 = arith.constant 64 : i32
  %pid0 = tt.get_program_id x : i32
  %pid1 = tt.get_program_id y : i32
  %xoff0 = arith.muli %pid0, %c16 : i32
  %xoff1 = arith.muli %pid1, %c64 : i32
  %range0 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
  %soff0 = tt.splat %xoff0 : i32 -> tensor<16xi32>
  %idx0 = arith.addi %soff0, %range0 : tensor<16xi32>
  %idx0e = tt.expand_dims %idx0 {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
  %range1 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
  %soff1 = tt.splat %xoff1 : i32 -> tensor<64xi32>
  %idx1 = arith.addi %soff1, %range1 : tensor<64xi32>
  %idx1e = tt.expand_dims %idx1 {axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
  %s0 = arith.constant dense<64> : tensor<16x1xi32>
  %lt0 = arith.cmpi slt, %idx0e, %s0 : tensor<16x1xi32>
  %lt0b = tt.broadcast %lt0 : tensor<16x1xi1> -> tensor<16x64xi1>
  %s1 = arith.constant dense<128> : tensor<1x64xi32>
  %lt1 = arith.cmpi slt, %idx1e, %s1 : tensor<1x64xi32>
  %lt1b = tt.broadcast %lt1 : tensor<1x64xi1> -> tensor<16x64xi1>
  %mask = arith.andi %lt0b, %lt1b : tensor<16x64xi1>
  %pbase = tt.splat %base : !tt.ptr<f32> -> tensor<16x64x!tt.ptr<f32>>
  %idx0b = tt.broadcast %idx0e : tensor<16x1xi32> -> tensor<16x64xi32>
  %p = tt.addptr %pbase, %idx0b : tensor<16x64x!tt.ptr<f32>>, tensor<16x64xi32>
  %v = tt.load %p, %mask : tensor<16x64x!tt.ptr<f32>>
  tt.return %v : tensor<16x64xf32>
}

// CHECK-LABEL: @bails_on_rank2_column_mask
// CHECK-NOT:   scf.if
// In a 2-D kernel (e.g. batch-norm pointwise) inductor emits a column mask
// ymask = yindex < ynumel of shape [YBLOCK, 1] — rank 2 — for loads that only
// depend on the Y coordinate.  Rewriting such loads as unmasked is incorrect
// because the X dimension may still have boundary tiles; the mask covers only
// one axis.  The pass must bail on rank-2 masks even if they look like the 1-D
// slt idiom after broadcast-stripping.
tt.func public @bails_on_rank2_column_mask(%base: !tt.ptr<f32>, %ynumel: i32) -> tensor<16x1xf32> {
  %c16 = arith.constant 16 : i32
  %pid1 = tt.get_program_id y : i32
  %yoff = arith.muli %pid1, %c16 : i32
  %range = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
  %soff = tt.splat %yoff : i32 -> tensor<16xi32>
  %yidx1d = arith.addi %soff, %range : tensor<16xi32>
  // Expand to [16, 1] — this is the [:, None] 2-D column shape.
  %yidx = tt.expand_dims %yidx1d {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
  %snum = tt.splat %ynumel : i32 -> tensor<16x1xi32>
  %ymask = arith.cmpi slt, %yidx, %snum : tensor<16x1xi32>
  %pbase = tt.splat %base : !tt.ptr<f32> -> tensor<16x1x!tt.ptr<f32>>
  %p = tt.addptr %pbase, %yidx : tensor<16x1x!tt.ptr<f32>>, tensor<16x1xi32>
  %v = tt.load %p, %ymask : tensor<16x1x!tt.ptr<f32>>
  tt.return %v : tensor<16x1xf32>
}

// Test 4 (bails_on_indirect_load): the pointer for the second tt.load is
// computed from the result of the first tt.load (indirect / gather indexing).
// CHECK-LABEL: @bails_on_indirect_load
// The direct in_ptr0 load is rewritten (unmasked).
// CHECK:       tt.load %{{.*}} : tensor<32x!tt.ptr<i32>>
// The indirect in_ptr1 load retains its mask.
// CHECK:       tt.load %{{.*}}, %{{.*}} : tensor<32x!tt.ptr<i1>>
// The store is wrapped in scf.if (the single store is rewritten).
// CHECK:       scf.if
tt.func public @bails_on_indirect_load(%in_ptr0: !tt.ptr<i32>, %in_ptr1: !tt.ptr<i1>, %out_ptr0: !tt.ptr<i1>, %ks0: i32, %xnumel: i32) {
  %c32 = arith.constant 32 : i32
  %pid = tt.get_program_id x : i32
  %xoff = arith.muli %pid, %c32 : i32
  %range = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
  %soff = tt.splat %xoff : i32 -> tensor<32xi32>
  %idx = arith.addi %soff, %range : tensor<32xi32>
  %snum = tt.splat %xnumel : i32 -> tensor<32xi32>
  %xmask = arith.cmpi slt, %idx, %snum : tensor<32xi32>
  // Direct load: in_ptr0[x0] — safe to unmask.
  %pbase0 = tt.splat %in_ptr0 : !tt.ptr<i32> -> tensor<32x!tt.ptr<i32>>
  %p0 = tt.addptr %pbase0, %idx : tensor<32x!tt.ptr<i32>>, tensor<32xi32>
  %tmp0 = tt.load %p0, %xmask : tensor<32x!tt.ptr<i32>>
  // Wrap-around: tmp4 = where(tmp0 < 0, tmp0 + ks0, tmp0)
  %zero = arith.constant dense<0> : tensor<32xi32>
  %neg = arith.cmpi slt, %tmp0, %zero : tensor<32xi32>
  %sks0 = tt.splat %ks0 : i32 -> tensor<32xi32>
  %wrapped = arith.addi %tmp0, %sks0 : tensor<32xi32>
  %tmp4 = arith.select %neg, %wrapped, %tmp0 : tensor<32xi1>, tensor<32xi32>
  // Indirect load: in_ptr1[tmp4] — address depends on tmp0 (a load result).
  // The mask must NOT be removed from this load.
  %pbase1 = tt.splat %in_ptr1 : !tt.ptr<i1> -> tensor<32x!tt.ptr<i1>>
  %p1 = tt.addptr %pbase1, %tmp4 : tensor<32x!tt.ptr<i1>>, tensor<32xi32>
  %tmp6 = tt.load %p1, %xmask : tensor<32x!tt.ptr<i1>>
  // Store: out_ptr0[x0] = tmp6 with xmask — this store should be rewritten.
  %pout = tt.splat %out_ptr0 : !tt.ptr<i1> -> tensor<32x!tt.ptr<i1>>
  %po = tt.addptr %pout, %idx : tensor<32x!tt.ptr<i1>>, tensor<32xi32>
  tt.store %po, %tmp6, %xmask : tensor<32x!tt.ptr<i1>>
  tt.return
}

// Test 5 (pointwise_2d_runtime): the canonical 2-D inductor idiom with runtime
// xnumel and ynumel. XBLOCK=16, YBLOCK=32. The mask is:
//   xmask = cmpi slt, expand_dims(xidx, 1), splat(xnumel) -> [16, 1]
//   ymask = cmpi slt, expand_dims(yidx, 0), splat(ynumel) -> [1, 32]
//   mask  = andi(broadcast(xmask), broadcast(ymask))      -> [16, 32]
// Load becomes unmasked. Store is wrapped in scf.if with the AND of both
// per-dimension in-bounds checks.
// CHECK-LABEL: @pointwise_2d_runtime
// CHECK:       %[[LOAD2D:.*]] = tt.load %{{.*}} : tensor<16x32x!tt.ptr<f32>>
// CHECK-NOT:   scf.if {{.*}} -> (tensor<16x32xf32>)
// CHECK:       %[[XHI:.*]] = arith.addi %{{.*}}, %{{.*}} : i32
// CHECK:       %[[XINB:.*]] = arith.cmpi slt, %[[XHI]], %{{.*}} : i32
// CHECK:       %[[YHI:.*]] = arith.addi %{{.*}}, %{{.*}} : i32
// CHECK:       %[[YINB:.*]] = arith.cmpi slt, %[[YHI]], %{{.*}} : i32
// CHECK:       %[[INB2D:.*]] = arith.andi %[[XINB]], %[[YINB]] : i1
// CHECK:       scf.if %[[INB2D]] {
// CHECK:         tt.store %{{.*}}, %[[LOAD2D]] : tensor<16x32x!tt.ptr<f32>>
// CHECK:       } else {
// CHECK:         tt.store %{{.*}}, %[[LOAD2D]], %{{.*}} : tensor<16x32x!tt.ptr<f32>>
// CHECK:       }

// Test 6 (bails_on_2d_non_pid_offset): one arm's offset is pid*XBLOCK+pid
// (not a clean pid*BLOCK) so the 2-D safety predicate fails -> op left untouched.
// CHECK-LABEL: @bails_on_2d_non_pid_offset
// CHECK:       tt.load %{{.*}}, %{{.*}} : tensor<16x32x!tt.ptr<f32>>
// CHECK-NOT:   scf.if

tt.func public @pointwise_2d_runtime(%base: !tt.ptr<f32>, %out: !tt.ptr<f32>,
                                      %xnumel: i32, %ynumel: i32) {
  %c16 = arith.constant 16 : i32
  %c32 = arith.constant 32 : i32
  %pid0 = tt.get_program_id x : i32
  %pid1 = tt.get_program_id y : i32
  %xoff = arith.muli %pid0, %c16 : i32
  %yoff = arith.muli %pid1, %c32 : i32
  // Row index: [16]
  %xrange = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
  %xsoff = tt.splat %xoff : i32 -> tensor<16xi32>
  %xidx = arith.addi %xsoff, %xrange : tensor<16xi32>
  // Col index: [32]
  %yrange = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
  %ysoff = tt.splat %yoff : i32 -> tensor<32xi32>
  %yidx = arith.addi %ysoff, %yrange : tensor<32xi32>
  // Row mask: [16, 1]
  %xidx_e = tt.expand_dims %xidx {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
  %xsnum = tt.splat %xnumel : i32 -> tensor<16x1xi32>
  %xmask = arith.cmpi slt, %xidx_e, %xsnum : tensor<16x1xi32>
  %xmask_bc = tt.broadcast %xmask : tensor<16x1xi1> -> tensor<16x32xi1>
  // Col mask: [1, 32]
  %yidx_e = tt.expand_dims %yidx {axis = 0 : i32} : tensor<32xi32> -> tensor<1x32xi32>
  %ysnum = tt.splat %ynumel : i32 -> tensor<1x32xi32>
  %ymask = arith.cmpi slt, %yidx_e, %ysnum : tensor<1x32xi32>
  %ymask_bc = tt.broadcast %ymask : tensor<1x32xi1> -> tensor<16x32xi1>
  // 2D mask: [16, 32]
  %mask = arith.andi %xmask_bc, %ymask_bc : tensor<16x32xi1>
  // Pointer: row-major indexing via xidx*32 + yidx
  %xidx_b = tt.broadcast %xidx_e : tensor<16x1xi32> -> tensor<16x32xi32>
  %yidx_b = tt.broadcast %yidx_e : tensor<1x32xi32> -> tensor<16x32xi32>
  %c32t = arith.constant dense<32> : tensor<16x32xi32>
  %xscaled = arith.muli %xidx_b, %c32t : tensor<16x32xi32>
  %linidx = arith.addi %xscaled, %yidx_b : tensor<16x32xi32>
  %pbase = tt.splat %base : !tt.ptr<f32> -> tensor<16x32x!tt.ptr<f32>>
  %p = tt.addptr %pbase, %linidx : tensor<16x32x!tt.ptr<f32>>, tensor<16x32xi32>
  %v = tt.load %p, %mask : tensor<16x32x!tt.ptr<f32>>
  %pout = tt.splat %out : !tt.ptr<f32> -> tensor<16x32x!tt.ptr<f32>>
  %po = tt.addptr %pout, %linidx : tensor<16x32x!tt.ptr<f32>>, tensor<16x32xi32>
  tt.store %po, %v, %mask : tensor<16x32x!tt.ptr<f32>>
  tt.return
}

tt.func public @bails_on_2d_non_pid_offset(%base: !tt.ptr<f32>, %out: !tt.ptr<f32>,
                                             %xnumel: i32, %ynumel: i32) {
  %c16 = arith.constant 16 : i32
  %c32 = arith.constant 32 : i32
  %pid0 = tt.get_program_id x : i32
  %pid1 = tt.get_program_id y : i32
  // X offset: pid0 * 16 + pid0 (not a clean pid * BLOCK)
  %xmul = arith.muli %pid0, %c16 : i32
  %xoff = arith.addi %xmul, %pid0 : i32
  %yoff = arith.muli %pid1, %c32 : i32
  // Row index: [16]
  %xrange = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
  %xsoff = tt.splat %xoff : i32 -> tensor<16xi32>
  %xidx = arith.addi %xsoff, %xrange : tensor<16xi32>
  // Col index: [32]
  %yrange = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
  %ysoff = tt.splat %yoff : i32 -> tensor<32xi32>
  %yidx = arith.addi %ysoff, %yrange : tensor<32xi32>
  // Row mask: [16, 1]
  %xidx_e = tt.expand_dims %xidx {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
  %xsnum = tt.splat %xnumel : i32 -> tensor<16x1xi32>
  %xmask = arith.cmpi slt, %xidx_e, %xsnum : tensor<16x1xi32>
  %xmask_bc = tt.broadcast %xmask : tensor<16x1xi1> -> tensor<16x32xi1>
  // Col mask: [1, 32]
  %yidx_e = tt.expand_dims %yidx {axis = 0 : i32} : tensor<32xi32> -> tensor<1x32xi32>
  %ysnum = tt.splat %ynumel : i32 -> tensor<1x32xi32>
  %ymask = arith.cmpi slt, %yidx_e, %ysnum : tensor<1x32xi32>
  %ymask_bc = tt.broadcast %ymask : tensor<1x32xi1> -> tensor<16x32xi1>
  // 2D mask
  %mask = arith.andi %xmask_bc, %ymask_bc : tensor<16x32xi1>
  %xidx_b = tt.broadcast %xidx_e : tensor<16x1xi32> -> tensor<16x32xi32>
  %yidx_b = tt.broadcast %yidx_e : tensor<1x32xi32> -> tensor<16x32xi32>
  %c32t = arith.constant dense<32> : tensor<16x32xi32>
  %xscaled = arith.muli %xidx_b, %c32t : tensor<16x32xi32>
  %linidx = arith.addi %xscaled, %yidx_b : tensor<16x32xi32>
  %pbase = tt.splat %base : !tt.ptr<f32> -> tensor<16x32x!tt.ptr<f32>>
  %p = tt.addptr %pbase, %linidx : tensor<16x32x!tt.ptr<f32>>, tensor<16x32xi32>
  %v = tt.load %p, %mask : tensor<16x32x!tt.ptr<f32>>
  %pout = tt.splat %out : !tt.ptr<f32> -> tensor<16x32x!tt.ptr<f32>>
  %po = tt.addptr %pout, %linidx : tensor<16x32x!tt.ptr<f32>>, tensor<16x32xi32>
  tt.store %po, %v, %mask : tensor<16x32x!tt.ptr<f32>>
  tt.return
}

// Test 7 (bails_on_nonzero_range_start_2d): both arms use make_range with
// start=1 instead of 0.  The structural pattern matches (andi of two
// broadcast(cmpi slt, expand_dims(addi(splat, range), axis), splat(numel)))
// but the in-bounds check `pid*BLOCK + (BLOCK-1) < numel` would be off by
// start (=1), allowing an unmasked store to write past the end of the tensor.
// The pass must bail on both the load and the store.
// CHECK-LABEL: @bails_on_nonzero_range_start_2d
// CHECK:       tt.load %{{.*}}, %{{.*}} : tensor<16x32x!tt.ptr<f32>>
// CHECK-NOT:   scf.if
tt.func public @bails_on_nonzero_range_start_2d(
    %base: !tt.ptr<f32>, %out: !tt.ptr<f32>,
    %xnumel: i32, %ynumel: i32) {
  %c16 = arith.constant 16 : i32
  %c32 = arith.constant 32 : i32
  %pid0 = tt.get_program_id x : i32
  %pid1 = tt.get_program_id y : i32
  %xoff = arith.muli %pid0, %c16 : i32
  %yoff = arith.muli %pid1, %c32 : i32

  // Row: range [1, 17) — extent still 16 but start != 0.
  %xrange = tt.make_range {end = 17 : i32, start = 1 : i32} : tensor<16xi32>
  %xsoff = tt.splat %xoff : i32 -> tensor<16xi32>
  %xidx = arith.addi %xsoff, %xrange : tensor<16xi32>

  // Col: range [1, 33) — extent still 32 but start != 0.
  %yrange = tt.make_range {end = 33 : i32, start = 1 : i32} : tensor<32xi32>
  %ysoff = tt.splat %yoff : i32 -> tensor<32xi32>
  %yidx = arith.addi %ysoff, %yrange : tensor<32xi32>

  // Row mask: [16, 1]
  %xidx_e = tt.expand_dims %xidx {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
  %xsnum = tt.splat %xnumel : i32 -> tensor<16x1xi32>
  %xmask = arith.cmpi slt, %xidx_e, %xsnum : tensor<16x1xi32>
  %xmask_bc = tt.broadcast %xmask : tensor<16x1xi1> -> tensor<16x32xi1>

  // Col mask: [1, 32]
  %yidx_e = tt.expand_dims %yidx {axis = 0 : i32} : tensor<32xi32> -> tensor<1x32xi32>
  %ysnum = tt.splat %ynumel : i32 -> tensor<1x32xi32>
  %ymask = arith.cmpi slt, %yidx_e, %ysnum : tensor<1x32xi32>
  %ymask_bc = tt.broadcast %ymask : tensor<1x32xi1> -> tensor<16x32xi1>

  // 2D mask — same andi(broadcast, broadcast) structure as the canonical idiom.
  %mask = arith.andi %xmask_bc, %ymask_bc : tensor<16x32xi1>

  %xidx_b = tt.broadcast %xidx_e : tensor<16x1xi32> -> tensor<16x32xi32>
  %yidx_b = tt.broadcast %yidx_e : tensor<1x32xi32> -> tensor<16x32xi32>
  %c32t = arith.constant dense<32> : tensor<16x32xi32>
  %xscaled = arith.muli %xidx_b, %c32t : tensor<16x32xi32>
  %linidx = arith.addi %xscaled, %yidx_b : tensor<16x32xi32>
  %pbase = tt.splat %base : !tt.ptr<f32> -> tensor<16x32x!tt.ptr<f32>>
  %p = tt.addptr %pbase, %linidx : tensor<16x32x!tt.ptr<f32>>, tensor<16x32xi32>
  %v = tt.load %p, %mask : tensor<16x32x!tt.ptr<f32>>
  %pout = tt.splat %out : !tt.ptr<f32> -> tensor<16x32x!tt.ptr<f32>>
  %po = tt.addptr %pout, %linidx : tensor<16x32x!tt.ptr<f32>>, tensor<16x32xi32>
  tt.store %po, %v, %mask : tensor<16x32x!tt.ptr<f32>>
  tt.return
}

// Test 8 (bails_on_indirect_store): the pointer for the tt.store is computed
// from the result of a tt.load (indirect / scatter indexing).
// The pass must NOT wrap the store in scf.if.
// CHECK-LABEL: @bails_on_indirect_store
// The direct in_ptr0 load is rewritten (unmasked).
// CHECK:       tt.load %{{.*}} : tensor<32x!tt.ptr<i32>>
// The indirect in_ptr1 load retains its mask.
// CHECK:       tt.load %{{.*}}, %{{.*}} : tensor<32x!tt.ptr<i32>>
// The scatter store retains its mask (no scf.if wrapping).
// CHECK:       tt.store %{{.*}}, %{{.*}}, %{{.*}} : tensor<32x!tt.ptr<i32>>
// CHECK-NOT:   scf.if
tt.func public @bails_on_indirect_store(%in_ptr0: !tt.ptr<i32>, %in_ptr1: !tt.ptr<i32>,
                                         %out_ptr0: !tt.ptr<i32>, %ks0: i32, %xnumel: i32) {
  %c32 = arith.constant 32 : i32
  %pid = tt.get_program_id x : i32
  %xoff = arith.muli %pid, %c32 : i32
  %range = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
  %soff = tt.splat %xoff : i32 -> tensor<32xi32>
  %idx = arith.addi %soff, %range : tensor<32xi32>
  %snum = tt.splat %xnumel : i32 -> tensor<32xi32>
  %xmask = arith.cmpi slt, %idx, %snum : tensor<32xi32>
  // Direct load: in_ptr0[x0] — safe to unmask.
  %pbase0 = tt.splat %in_ptr0 : !tt.ptr<i32> -> tensor<32x!tt.ptr<i32>>
  %p0 = tt.addptr %pbase0, %idx : tensor<32x!tt.ptr<i32>>, tensor<32xi32>
  %tmp0 = tt.load %p0, %xmask : tensor<32x!tt.ptr<i32>>
  // Wrap-around: tmp4 = where(tmp0 < 0, tmp0 + ks0, tmp0)
  %zero = arith.constant dense<0> : tensor<32xi32>
  %neg = arith.cmpi slt, %tmp0, %zero : tensor<32xi32>
  %sks0 = tt.splat %ks0 : i32 -> tensor<32xi32>
  %wrapped = arith.addi %tmp0, %sks0 : tensor<32xi32>
  %tmp4 = arith.select %neg, %wrapped, %tmp0 : tensor<32xi1>, tensor<32xi32>
  // Indirect load: in_ptr1[tmp4] — address depends on tmp0 (a load result).
  %pbase1 = tt.splat %in_ptr1 : !tt.ptr<i32> -> tensor<32x!tt.ptr<i32>>
  %p1 = tt.addptr %pbase1, %tmp4 : tensor<32x!tt.ptr<i32>>, tensor<32xi32>
  %tmp6 = tt.load %p1, %xmask : tensor<32x!tt.ptr<i32>>
  // Scatter store: out_ptr0[tmp4] = tmp6 — store pointer depends on tmp0 (a load result).
  // The mask must NOT be removed from this store.
  %pout = tt.splat %out_ptr0 : !tt.ptr<i32> -> tensor<32x!tt.ptr<i32>>
  %po = tt.addptr %pout, %tmp4 : tensor<32x!tt.ptr<i32>>, tensor<32xi32>
  tt.store %po, %tmp6, %xmask : tensor<32x!tt.ptr<i32>>
  tt.return
}

} // module
