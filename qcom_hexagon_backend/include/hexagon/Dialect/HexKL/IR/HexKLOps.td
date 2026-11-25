//===- HexKLOps.td - HexKL Ops -----------------------------*- tablegen -*-===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
// Describes the hexkl dialect ops.
//===----------------------------------------------------------------------===//

#ifndef HEXAGON_DIALECT_HEXKL_IR_HEXKL_OPS_TD
#define HEXAGON_DIALECT_HEXKL_IR_HEXKL_OPS_TD

include "mlir/Interfaces/DestinationStyleOpInterface.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/CommonTypeConstraints.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "hexagon/Dialect/HexKL/IR/HexKLDialect.td"

// Base class for HexKL dialect ops.
class HexKL_Op<string mnemonic, list<Trait> traits> :
    Op<HexKL_Dialect, mnemonic, traits> {

}

def HexKL_MatmulOp
  : HexKL_Op<"matmul",
      [DestinationStyleOpInterface,MemoryEffects<[MemRead, MemWrite]>]> {
  let hasVerifier = 1;

  let summary = "HexKL Matmul op";
  let description = [{
    Performs matrix multiplication using HexKL APIs. Two type combinations are
    supported: F16 lhs/rhs with F16 or F32 outs (the existing HMX F16 path),
    and I8 lhs/rhs with I32 outs (u8 activation x i8 weight -> i32, the HMX
    quantized path). Both operands are the signless `i8` MLIR type; "u8
    activation" vs "i8 weight" is a semantic contract carried by operand
    position, not the MLIR type, matching the rest of this dialect's micro
    ops (e.g. micro_hmx_copy_submatrix_to_8b_activation takes the u8
    activation, micro_hmx_rm_to_wh_i8 the i8 weight). Mixed combinations
    (e.g. F16 lhs with I8 rhs) are rejected by the verifier.
  }];

  let arguments = (ins
    AnyTypeOf<[Non0RankedTensorOf<[F16, I8]>, Non0RankedMemRefOf<[F16, I8]>]>:$lhs,
    AnyTypeOf<[Non0RankedTensorOf<[F16, I8]>, Non0RankedMemRefOf<[F16, I8]>]>:$rhs,
    AnyTypeOf<[Non0RankedTensorOf<[F16, F32, I32]>, Non0RankedMemRefOf<[F16, F32, I32]>]>:$outs
  );

  let results = (outs Variadic<Non0RankedTensorOf<[F16, F32, I32]>>:$result);

  let assemblyFormat = [{
    `ins` `(`  $lhs `,` $rhs`:` type($lhs) `,` type($rhs) `)`
    `outs` `(` $outs `:` type($outs) `)`
    attr-dict
    (`->` type($result)^)?
  }];

  let extraClassDeclaration = [{
    // Implemented as part of DestinationStyleOpInterface
    MutableOperandRange getDpsInitsMutable() { return getOutsMutable(); }
  }];
}

def HexKL_MicroHMXSetupAccReadF16Op
  : HexKL_Op<"micro_hmx_setup_acc_read_f16",
      [MemoryEffects<[MemWrite]>]> {
  let hasVerifier = 1;

  let summary = "Setup accumulator for reading";
  let description = [{
    Sets up the accumulator for reading F16 values. The configuration offset
    is derived from the VTCM block during lowering and is not modeled as an
    operand in the IR.
  }];

  let arguments = (ins
    AnyNon0RankedMemRef:$hmx_block
  );

  let results = (outs);

  let assemblyFormat =
    "`(` $hmx_block `)` `:` type($hmx_block) attr-dict";
}

def HexKL_MicroHMXCopySubmatrixToF16Op
  : HexKL_Op<"micro_hmx_copy_submatrix_to_f16",
      [MemoryEffects<[MemRead, MemWrite]>]> {
  let hasVerifier = 1;

  let summary = "Copy submatrix to F16";
  let description = [{
    Copies a 32x32 tile from a flat F16 matrix in DDR to VTCM at the given
    destination offset.
  }];

  let arguments = (ins
    AnyNon0RankedMemRef:$hmx_block,
    I32:$out_offset,
    AnyNon0RankedMemRef:$src,
    I32:$tile_row,
    I32:$tile_col,
    I32:$input_rows,
    I32:$input_cols
  );

  let results = (outs);

  let assemblyFormat =
    "`(` $hmx_block `,` $out_offset `,` $src `,` $tile_row `,` $tile_col `,` $input_rows `,` $input_cols `)` `:` type($hmx_block) `,` type($out_offset) `,` type($src) `,` type($tile_row) `,` type($tile_col) `,` type($input_rows) `,` type($input_cols) attr-dict";
}

def HexKL_MicroHMXRmToAhF16Op
  : HexKL_Op<"micro_hmx_rm_to_ah_f16",
      [MemoryEffects<[MemRead, MemWrite]>]> {
  let hasVerifier = 1;

  let summary = "Transform row-major to activation-horizontal F16";
  let description = [{
    Transforms a 32x32 fp16 tile from flat row-major layout in VTCM to
    activation layout at the destination offset.
  }];

  let arguments = (ins
    AnyNon0RankedMemRef:$hmx_block,
    I32:$activation_out_offset,
    I32:$flat_in_offset
  );

  let results = (outs);

  let assemblyFormat =
    "`(` $hmx_block `,` $activation_out_offset `,` $flat_in_offset `)` `:` type($hmx_block) `,` type($activation_out_offset) `,` type($flat_in_offset) attr-dict";
}

def HexKL_MicroHMXAccClearF16Op
  : HexKL_Op<"micro_hmx_acc_clear_f16",
      [MemoryEffects<[MemWrite]>]> {

  let summary = "Clear accumulator";
  let description = [{
    Clears the F16 accumulator.
  }];

  let arguments = (ins  );

  let results = (outs);

  let assemblyFormat =
    "attr-dict";
}

def HexKL_MicroHMXRmToWhF16Op
  : HexKL_Op<"micro_hmx_rm_to_wh_f16",
      [MemoryEffects<[MemRead, MemWrite]>]> {
  let hasVerifier = 1;

  let summary = "Transform row-major to weight-horizontal F16";
  let description = [{
    Converts a flat FP16 weight tile in DDR to the HMX weight layout and writes
    it to VTCM at the given destination offset.
  }];

  let arguments = (ins
    AnyNon0RankedMemRef:$hmx_block,
    I32:$weight_offset,
    AnyNon0RankedMemRef:$src,
    I32:$tile_row,
    I32:$tile_col,
    I32:$wt_cols
  );

  let results = (outs);

  let assemblyFormat =
    "`(` $hmx_block `,` $weight_offset `,` $src `,` $tile_row `,` $tile_col `,` $wt_cols `)` `:` type($hmx_block) `,` type($weight_offset) `,` type($src) `,` type($tile_row) `,` type($tile_col) `,` type($wt_cols) attr-dict";
}

def HexKL_MicroHMXMmF16Op
  : HexKL_Op<"micro_hmx_mm_f16",
      [MemoryEffects<[MemRead, MemWrite]>]> {
  let hasVerifier = 1;

  let summary = "Micro HMX matrix multiplication F16";
  let description = [{
    Performs micro GEMM on activation and weight tiles in VTCM using F16 inputs.
  }];

  let arguments = (ins
    AnyNon0RankedMemRef:$hmx_block,
    I32:$activation_offset,
    I32:$weight_offset
  );

  let results = (outs);

  let assemblyFormat =
    "`(` $hmx_block `,` $activation_offset `,` $weight_offset `)` `:` type($hmx_block) `,` type($activation_offset) `,` type($weight_offset) attr-dict";
}

def HexKL_MicroHMXAccReadF16Op
  : HexKL_Op<"micro_hmx_acc_read_f16",
      [MemoryEffects<[MemRead, MemWrite]>]> {
  let hasVerifier = 1;

  let summary = "Read accumulator F16";
  let description = [{
    Reads the F16 accumulator and stores it in VTCM at the given destination offset.
    The configuration offset is derived from the VTCM block during lowering.
  }];

  let arguments = (ins
    AnyNon0RankedMemRef:$hmx_block,
    I32:$out_offset
  );

  let results = (outs);

  let assemblyFormat =
    "`(` $hmx_block `,` $out_offset `)` `:` type($hmx_block) `,` type($out_offset) attr-dict";
}

def HexKL_MicroHMXAhToRmF16Op
  : HexKL_Op<"micro_hmx_ah_to_rm_f16",
      [MemoryEffects<[MemRead, MemWrite]>]> {
  let hasVerifier = 1;

  let summary = "Transform activation-horizontal to row-major F16";
  let description = [{
    Transforms a 32x32 fp16 tile from activation layout in VTCM to flat row-major
    layout at the destination offset.
  }];

  let arguments = (ins
    AnyNon0RankedMemRef:$hmx_block,
    I32:$flat_out_offset,
    I32:$activation_in_offset
  );

  let results = (outs);

  let assemblyFormat =
    "`(` $hmx_block `,` $flat_out_offset `,` $activation_in_offset `)` `:` type($hmx_block) `,` type($flat_out_offset) `,` type($activation_in_offset) attr-dict";
}

//===----------------------------------------------------------------------===//
// HexKL micro HMX quantized (u8 activation x i8 weight) ops
//
// Data-movement ops used by the u8i8 micro-daemon path. Activation load and
// result store fuse layout conversion. Offsets are VTCM byte offsets.
//===----------------------------------------------------------------------===//

def HexKL_MicroHMXCopySubmatrixTo8bActivationOp
  : HexKL_Op<"micro_hmx_copy_submatrix_to_8b_activation",
      [MemoryEffects<[MemRead, MemWrite]>]> {
  let hasVerifier = 1;

  let summary = "Copy submatrix to 8-bit activation-horizontal layout";
  let description = [{
    Copies a 64x32 uint8 DDR tile directly into activation-horizontal VTCM.
    Copy and layout are fused; the 2048-byte destination must be aligned.
  }];

  let arguments = (ins
    AnyNon0RankedMemRef:$hmx_block,
    I32:$out_offset,
    AnyNon0RankedMemRef:$src,
    I32:$tile_row,
    I32:$tile_col,
    I32:$input_rows,
    I32:$input_cols
  );

  let results = (outs);

  let assemblyFormat =
    "`(` $hmx_block `,` $out_offset `,` $src `,` $tile_row `,` $tile_col `,` $input_rows `,` $input_cols `)` `:` type($hmx_block) `,` type($out_offset) `,` type($src) `,` type($tile_row) `,` type($tile_col) `,` type($input_rows) `,` type($input_cols) attr-dict";
}

def HexKL_MicroHMXRmToWhI8Op
  : HexKL_Op<"micro_hmx_rm_to_wh_i8",
      [MemoryEffects<[MemRead, MemWrite]>]> {
  let hasVerifier = 1;

  let summary = "Transform row-major to weight-horizontal I8";
  let description = [{
    Converts a flat int8 weight tile in DDR to the HMX weight-horizontal layout
    and writes it to VTCM at the given destination offset. I8 analog of
    micro_hmx_rm_to_wh_f16; lowers to hexkl_micro_hmx_rm_to_wh_i8.
  }];

  let arguments = (ins
    AnyNon0RankedMemRef:$hmx_block,
    I32:$weight_offset,
    AnyNon0RankedMemRef:$src,
    I32:$tile_row,
    I32:$tile_col,
    I32:$wt_cols
  );

  let results = (outs);

  let assemblyFormat =
    "`(` $hmx_block `,` $weight_offset `,` $src `,` $tile_row `,` $tile_col `,` $wt_cols `)` `:` type($hmx_block) `,` type($weight_offset) `,` type($src) `,` type($tile_row) `,` type($tile_col) `,` type($wt_cols) attr-dict";
}

def HexKL_MicroHMXCopy32bToSubmatrixOp
  : HexKL_Op<"micro_hmx_copy_32b_to_submatrix",
      [MemoryEffects<[MemRead, MemWrite]>]> {
  let hasVerifier = 1;

  let summary = "Copy shuffled 32-bit accumulator tile to a DDR submatrix";
  let description = [{
    Copies a shuffled 64x32 int32 accumulator tile (8192 bytes) from VTCM to a
    submatrix in DDR, unshuffling and writing out in one fused call (no separate
    ah_to_rm/flat-copy step, unlike the F16 path's ah_to_rm_f16 +
    copy_f16_to_f32_submatrix pair). Lowers to
    hexkl_micro_hmx_copy_32b_to_submatrix.
  }];

  let arguments = (ins
    AnyNon0RankedMemRef:$hmx_block,
    I32:$in_offset,
    AnyNon0RankedMemRef:$dst,
    I32:$tile_row,
    I32:$tile_col,
    I32:$output_rows,
    I32:$output_cols
  );

  let results = (outs);

  let assemblyFormat =
    "`(` $hmx_block `,` $in_offset `,` $dst `,` $tile_row `,` $tile_col `,` $output_rows `,` $output_cols `)` `:` type($hmx_block) `,` type($in_offset) `,` type($dst) `,` type($tile_row) `,` type($tile_col) `,` type($output_rows) `,` type($output_cols) attr-dict";
}

def HexKL_MicroHMXCopyF16ToF32SubmatrixOp
  : HexKL_Op<"micro_hmx_copy_f16_to_f32_submatrix",
      [MemoryEffects<[MemRead, MemWrite]>]> {
  let hasVerifier = 1;

  let summary = "Copy F16 to F32 submatrix";
  let description = [{
    Copies a 32x32 tile in flat F16 layout from VTCM to DDR as F32.
  }];

  let arguments = (ins
    AnyNon0RankedMemRef:$hmx_block,
    I32:$in_offset,
    AnyNon0RankedMemRef:$dst,
    I32:$tile_row,
    I32:$tile_col,
    I32:$output_rows,
    I32:$output_cols
  );

  let results = (outs);

  let assemblyFormat =
    "`(` $hmx_block `,` $in_offset `,` $dst `,` $tile_row `,` $tile_col `,` $output_rows `,` $output_cols `)` `:` type($hmx_block) `,` type($in_offset) `,` type($dst) `,` type($tile_row) `,` type($tile_col) `,` type($output_rows) `,` type($output_cols) attr-dict";
}

//===----------------------------------------------------------------------===//
// HexKL micro HMX daemon ops (mobile/DSP dedicated-HMX-thread model)
//
// These pair with the runtime queue in bin/runtime/HMXMicroDaemon. A worker
// thread submits one K-block matmul job (dispatch) and later blocks until the
// daemon has drained its lane (wait_idle). Unlike the inline micro ops above,
// the accumulator-touching work (acc_clear / mm / acc_read) runs on the daemon
// thread, so these ops carry only the GLOBAL VTCM byte offsets the daemon needs
// plus the accumulator-control flags -- the daemon already holds vtcm_base.
//===----------------------------------------------------------------------===//

def HexKL_MicroHMXDispatchOp
  : HexKL_Op<"micro_hmx_dispatch",
      [MemoryEffects<[MemRead, MemWrite]>]> {
  let hasVerifier = 1;

  let summary = "Submit one K-block matmul job to the HMX micro daemon";
  let description = [{
    Enqueues a single K=32 matmul job onto the calling worker's lane in the HMX
    micro daemon queue. `act_offset`/`wgt_offset` are the GLOBAL VTCM byte
    offsets of the AH-laid-out activation tile and WH-laid-out weight tile;
    `out_offset` is the acc-read destination, consumed by the daemon only when
    `write_out` is set. `clear_acc` starts a fresh partial sum (first K-block of
    an output tile) and `write_out` commits it via acc_read (last K-block).
    Non-blocking: the mm runs asynchronously on the daemon thread.

    `job_kind` is "f16" or "u8i8" and selects the accumulator-mode runtime
    functions the daemon runs for this job (mm_f16 vs mm_u8i8, and the matching
    acc_clear / acc_read / setup_acc_read). It travels with the job rather than
    with the queue because the compiler knows the matmul's element type for
    certain while the wrapper creating the queue does not; the daemon programs
    setup_acc_read lazily from it.
  }];

  let arguments = (ins
    AnyType:$queue,
    I32:$worker_id,
    I32:$act_offset,
    I32:$wgt_offset,
    I32:$out_offset,
    I1:$clear_acc,
    I1:$write_out,
    StrAttr:$job_kind
  );

  let results = (outs);

  let assemblyFormat =
    "`(` $queue `,` $worker_id `,` $act_offset `,` $wgt_offset `,` $out_offset `,` $clear_acc `,` $write_out `)` `:` type($queue) `,` type($worker_id) `,` type($act_offset) `,` type($wgt_offset) `,` type($out_offset) `,` type($clear_acc) `,` type($write_out) attr-dict";
}

def HexKL_MicroHMXWaitIdleOp
  : HexKL_Op<"micro_hmx_wait_idle",
      [MemoryEffects<[MemRead, MemWrite]>]> {

  let summary = "Block until the HMX micro daemon has drained this worker's lane";
  let description = [{
    Spin-waits until the daemon has consumed every job the calling worker
    submitted on lane `worker_id`. On return the acc_read output the daemon
    wrote is visible to the worker, so the subsequent ah_to_rm / store reads
    committed data. Models a read+write effect so it is neither reordered past
    the surrounding layout ops nor eliminated as dead.
  }];

  let arguments = (ins
    AnyType:$queue,
    I32:$worker_id
  );

  let results = (outs);

  let assemblyFormat =
    "`(` $queue `,` $worker_id `)` `:` type($queue) `,` type($worker_id) attr-dict";
}

//===----------------------------------------------------------------------===//
// HexKL Macro ops (runtime library modeled in the dialect)
//
// These ops represent the macro runtime calls used by the lowering pipeline.
// Keeping them as dialect ops (instead of lowering directly to LLVM calls)
// enables IR-level optimizations (e.g. fusing back-to-back matmuls, hoisting
// init/lock/unlock/finalize, removing redundant cache operations).
//===----------------------------------------------------------------------===//

def HexKL_MacroInitializeOp
  : HexKL_Op<"macro_initialize", [MemoryEffects<[MemWrite]>]> {
  let summary = "Initialize HexKL macro runtime";
  let results = (outs);
  let assemblyFormat = "attr-dict";
}

def HexKL_MacroFinalizeOp
  : HexKL_Op<"macro_finalize", [MemoryEffects<[MemWrite]>]> {
  let summary = "Finalize HexKL macro runtime";
  let results = (outs);
  let assemblyFormat = "attr-dict";
}

def HexKL_MacroLockHmxOp
  : HexKL_Op<"macro_lock_hmx", [MemoryEffects<[MemWrite]>]> {
  let summary = "Lock HMX for macro runtime";
  let results = (outs);
  let assemblyFormat = "attr-dict";
}

def HexKL_MacroUnlockHmxOp
  : HexKL_Op<"macro_unlock_hmx", [MemoryEffects<[MemWrite]>]> {
  let summary = "Unlock HMX for macro runtime";
  let results = (outs);
  let assemblyFormat = "attr-dict";
}

def HexKL_MacroRmToAhF16InplaceOp
  : HexKL_Op<"macro_rm_to_ah_f16_inplace",
      [MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "Transform activation from RM to AH layout (in-place)";
  let arguments = (ins I32:$m, I32:$k, AnyType:$ptr);
  let results = (outs);
  let assemblyFormat =
    "`(` $m `,` $k `,` $ptr `)` `:` type($m) `,` type($k) `,` type($ptr) attr-dict";
}

def HexKL_MacroAhToRmF16InplaceOp
  : HexKL_Op<"macro_ah_to_rm_f16_inplace",
      [MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "Transform output from AH to RM layout (in-place)";
  let arguments = (ins I32:$m, I32:$n, AnyType:$ptr);
  let results = (outs);
  let assemblyFormat =
    "`(` $m `,` $n `,` $ptr `)` `:` type($m) `,` type($n) `,` type($ptr) attr-dict";
}

def HexKL_MacroMmF16Op
  : HexKL_Op<"macro_mm_f16", [MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "HexKL macro matmul (f16)";
  let arguments = (ins
    I32:$m,
    I32:$n,
    I32:$k,
    AnyType:$out_ptr,
    AnyType:$act_ptr,
    AnyType:$wgt_ptr
  );
  let results = (outs);
  let assemblyFormat =
    "`(` $m `,` $n `,` $k `,` $out_ptr `,` $act_ptr `,` $wgt_ptr `)` `:` type($m) `,` type($n) `,` type($k) `,` type($out_ptr) `,` type($act_ptr) `,` type($wgt_ptr) attr-dict";
}

def HexKL_QurtMemCacheCleanOp
  : HexKL_Op<"qurt_mem_cache_clean", [MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "QURT cache clean/flush/invalidate";
  let arguments = (ins I32:$ptr_i32, I32:$size_bytes_i32, I32:$op, I32:$cache);
  let results = (outs);
  let assemblyFormat =
    "`(` $ptr_i32 `,` $size_bytes_i32 `,` $op `,` $cache `)` `:` type($ptr_i32) `,` type($size_bytes_i32) `,` type($op) `,` type($cache) attr-dict";
}

#endif // HEXAGON_DIALECT_HEXKL_IR_HEXKL_OPS_TD
