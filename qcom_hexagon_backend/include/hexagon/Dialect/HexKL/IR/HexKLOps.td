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

  let summary = "HexKL Matmul op";
  let description = [{
    Performs matrix multiplication on F16 inputs and F32 output using HexKL APIs.
  }];

  let arguments = (ins
    AnyTypeOf<[Non0RankedTensorOf<[F16]>, Non0RankedMemRefOf<[F16]>]>:$lhs,
    AnyTypeOf<[Non0RankedTensorOf<[F16]>, Non0RankedMemRefOf<[F16]>]>:$rhs,
    AnyTypeOf<[Non0RankedTensorOf<[F32]>, Non0RankedMemRefOf<[F32]>]>:$outs
  );

  let results = (outs Variadic<Non0RankedTensorOf<[F32]>>:$result);

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

#endif // HEXAGON_DIALECT_HEXKL_IR_HEXKL_OPS_TD
