#!/usr/bin/env python3

# ===- rewrite-i64-ptr-args-to-i32-python.py -------------------------------===
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause.
# For more license information:
#   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
#
# ===------------------------------------------------------------------------===


# Please Note: This script holds the input IR for the lit test
# rewrite-i64-ptr-args-to-i32-python.mlir (FileCheck directives live in the
# .mlir file in the same folder). It parses the embedded module, runs the
# rewrite-i64-ptr-args-to-i32 pass via the qcom_hexagon_backend Python binding,
# and prints the result to stdout for FileCheck.
#
# This is done because the pass operates on Triton (tt) dialect IR and is only
# exposed through the Python binding, not via linalg-hexagon-opt. Registering it
# into linalg-hexagon-opt would force that tool to link/register the Triton
# dialect ecosystem (TritonGPU and related deps), which we intentionally keep
# out of the Hexagon opt tool.

from triton._C.libtriton import ir, qcom_hexagon_backend


def main():
    ctx = ir.context()
    qcom_hexagon_backend.load_dialects(ctx)

    ttir = r"""
module {
  tt.func @kernel(%arg0: !tt.ptr<i64>, %arg1: i32) {
    %idx = tt.make_range {start = 0 : i32, end = 4 : i32} : tensor<4xi32>
    %ptrs = tt.splat %arg0 : !tt.ptr<i64> -> tensor<4x!tt.ptr<i64>>
    %ptrs_off = tt.addptr %ptrs, %idx : tensor<4x!tt.ptr<i64>>, tensor<4xi32>
    %vals = tt.load %ptrs_off : tensor<4x!tt.ptr<i64>>
    %asf32 = arith.sitofp %vals : tensor<4xi64> to tensor<4xf32>
    tt.return
  }

  tt.func @triton_poi_fused__to_copy_unsqueeze_1(%in_ptr0: !tt.ptr<i64>, %out_ptr0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %xnumel: i32) attributes {noinline = false} {
    %xmask = arith.constant dense<3> : tensor<4xi32>
    %c4_i32 = arith.constant 4 : i32
    %xoffset = tt.get_program_id x : i32
    %xoffset_0 = arith.muli %xoffset, %c4_i32 : i32
    %xindex = tt.make_range {start = 0 : i32, end = 4 : i32} : tensor<4xi32>
    %xindex_1 = tt.splat %xoffset_0 : i32 -> tensor<4xi32>
    %xindex_2 = arith.addi %xindex_1, %xindex : tensor<4xi32>
    %xmask_3 = arith.cmpi slt, %xindex_2, %xmask : tensor<4xi32>
    %tmp0 = tt.splat %in_ptr0 : !tt.ptr<i64> -> tensor<4x!tt.ptr<i64>>
    %tmp0_4 = tt.addptr %tmp0, %xindex_2 : tensor<4x!tt.ptr<i64>>, tensor<4xi32>
    %tmp0_5 = tt.load %tmp0_4, %xmask_3 : tensor<4x!tt.ptr<i64>>
    %tmp1 = arith.sitofp %tmp0_5 : tensor<4xi64> to tensor<4xf32>
    %0 = tt.splat %out_ptr0 : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %1 = tt.addptr %0, %xindex_2 : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
    tt.store %1, %tmp1, %xmask_3 : tensor<4x!tt.ptr<f32>>
    tt.return
  }

  // triton_poi_fused_add_6 pattern: both in_ptr0 and out_ptr0 are !tt.ptr<i64>.
  // The load value must NOT receive extsi before the store — the store pointer
  // is also rewritten to ptr<i32>, so the value must remain i32.
  tt.func @triton_poi_fused_add_6(%in_ptr0: !tt.ptr<i64> {tt.divisibility = 16 : i32}, %out_ptr0: !tt.ptr<i64> {tt.divisibility = 16 : i32}, %xnumel: i32) attributes {noinline = false} {
    %xmask = arith.constant dense<3> : tensor<4xi32>
    %c4_i32 = arith.constant 4 : i32
    %xoffset = tt.get_program_id x : i32
    %xoffset_0 = arith.muli %xoffset, %c4_i32 : i32
    %xindex = tt.make_range {start = 0 : i32, end = 4 : i32} : tensor<4xi32>
    %xindex_1 = tt.splat %xoffset_0 : i32 -> tensor<4xi32>
    %xindex_2 = arith.addi %xindex_1, %xindex : tensor<4xi32>
    %xmask_3 = arith.cmpi slt, %xindex_2, %xmask : tensor<4xi32>
    %tmp0 = tt.splat %in_ptr0 : !tt.ptr<i64> -> tensor<4x!tt.ptr<i64>>
    %tmp0_4 = tt.addptr %tmp0, %xindex_2 : tensor<4x!tt.ptr<i64>>, tensor<4xi32>
    %tmp0_5 = tt.load %tmp0_4, %xmask_3 : tensor<4x!tt.ptr<i64>>
    %0 = tt.splat %out_ptr0 : !tt.ptr<i64> -> tensor<4x!tt.ptr<i64>>
    %1 = tt.addptr %0, %xindex_2 : tensor<4x!tt.ptr<i64>>, tensor<4xi32>
    tt.store %1, %tmp0_5, %xmask_3 : tensor<4x!tt.ptr<i64>>
    tt.return
  }

  // triton_poi_fused_addindex_7 pattern: both in_ptr0 and out_ptr0 are
  // !tt.ptr<i64>, but i64 arithmetic (addi with a constant) sits between the
  // load and the store. The load result is extsi'd to i64 for the computation,
  // and the i64 result is trunci'd back to i32 at the store to match the
  // narrowed destination pointer.
  tt.func @triton_poi_fused_addindex_7(%in_ptr0: !tt.ptr<i64> {tt.divisibility = 16 : i32}, %out_ptr0: !tt.ptr<i64> {tt.divisibility = 16 : i32}, %xnumel: i32) attributes {noinline = false} {
    %xmask = arith.constant dense<3> : tensor<4xi32>
    %one = arith.constant dense<1> : tensor<4xi64>
    %c4_i32 = arith.constant 4 : i32
    %xoffset = tt.get_program_id x : i32
    %xoffset_0 = arith.muli %xoffset, %c4_i32 : i32
    %xindex = tt.make_range {start = 0 : i32, end = 4 : i32} : tensor<4xi32>
    %xindex_1 = tt.splat %xoffset_0 : i32 -> tensor<4xi32>
    %xindex_2 = arith.addi %xindex_1, %xindex : tensor<4xi32>
    %xmask_3 = arith.cmpi slt, %xindex_2, %xmask : tensor<4xi32>
    %tmp0 = tt.splat %in_ptr0 : !tt.ptr<i64> -> tensor<4x!tt.ptr<i64>>
    %tmp0_4 = tt.addptr %tmp0, %xindex_2 : tensor<4x!tt.ptr<i64>>, tensor<4xi32>
    %tmp0_5 = tt.load %tmp0_4, %xmask_3 : tensor<4x!tt.ptr<i64>>
    %tmp0_6 = arith.addi %tmp0_5, %one : tensor<4xi64>
    %0 = tt.splat %out_ptr0 : !tt.ptr<i64> -> tensor<4x!tt.ptr<i64>>
    %1 = tt.addptr %0, %xindex_2 : tensor<4x!tt.ptr<i64>>, tensor<4xi32>
    tt.store %1, %tmp0_6, %xmask_3 : tensor<4x!tt.ptr<i64>>
    tt.return
  }

  // triton_poi_fused_inplace_8 pattern: a single !tt.ptr<i64> arg is used as
  // BOTH the load source and the store destination (in-place update, common in
  // Inductor with in_out_ptr0). The same pointer closure is shared, so the load
  // narrows to i32, i64 arithmetic stays i64, and the store trunci's back to
  // i32 through the same rewritten pointer.
  tt.func @triton_poi_fused_inplace_8(%inout_ptr0: !tt.ptr<i64> {tt.divisibility = 16 : i32}, %xnumel: i32) attributes {noinline = false} {
    %xmask = arith.constant dense<3> : tensor<4xi32>
    %one = arith.constant dense<1> : tensor<4xi64>
    %c4_i32 = arith.constant 4 : i32
    %xoffset = tt.get_program_id x : i32
    %xoffset_0 = arith.muli %xoffset, %c4_i32 : i32
    %xindex = tt.make_range {start = 0 : i32, end = 4 : i32} : tensor<4xi32>
    %xindex_1 = tt.splat %xoffset_0 : i32 -> tensor<4xi32>
    %xindex_2 = arith.addi %xindex_1, %xindex : tensor<4xi32>
    %xmask_3 = arith.cmpi slt, %xindex_2, %xmask : tensor<4xi32>
    %ptrs = tt.splat %inout_ptr0 : !tt.ptr<i64> -> tensor<4x!tt.ptr<i64>>
    %ptrs_off = tt.addptr %ptrs, %xindex_2 : tensor<4x!tt.ptr<i64>>, tensor<4xi32>
    %loaded = tt.load %ptrs_off, %xmask_3 : tensor<4x!tt.ptr<i64>>
    %updated = arith.addi %loaded, %one : tensor<4xi64>
    tt.store %ptrs_off, %updated, %xmask_3 : tensor<4x!tt.ptr<i64>>
    tt.return
  }
}
"""

    mod = qcom_hexagon_backend.parse_mlir_module_from_str(ttir, ctx)
    changed = qcom_hexagon_backend.rewrite_i64_ptr_args_to_i32(mod)
    print(f"changed={changed}")
    print(mod)


if __name__ == "__main__":
    main()
