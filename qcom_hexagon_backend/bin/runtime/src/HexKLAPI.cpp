//===- HexKLAPI.cpp - implementation file                   ---------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
#include "HexKLAPI.h"
#include "HexagonAPI.h"
#include "HexagonCAPI.h"
#include "hexkl_micro.h"

extern "C" {
int hexkl_matmul_f16f16_f32(int64_t n_row, int64_t n_col, int64_t n_inner,
                            float *restrict outM,
                            const _Float16 *restrict inAct,
                            const _Float16 *restrict inW) {
  int ret = AEE_SUCCESS;

  uint32_t N = n_row;
  uint32_t M = n_col;
  uint32_t A_rows = n_row;
  uint32_t A_cols = n_inner;
  uint32_t X_cols = n_col;
  uint32_t X_rows = n_row;
  uint32_t row_tiles_in_A = (n_inner + (HEXKL_HMX_F16_BLOCK_N_INNER - 1)) /
                            HEXKL_HMX_F16_BLOCK_N_INNER;
  uint32_t weight_offset = HEXKL_HMX_ACTIVATION_ALIGNMENT * row_tiles_in_A;
  const _Float16 *matA = (const _Float16 *)inAct;
  const _Float16 *matW = (const _Float16 *)inW;
  float *matX = outM;

  // Calculate VTCM layout and activation-aligned boundaries
  // Regions:
  //  - Activation tiles:            row_tiles_in_A * ALIGN
  //  - Scratch/flat + accum buffers: (row_tiles_in_A + 2) * ALIGN
  //  - HMX config block:            config_size bytes

  const size_t align = HEXKL_HMX_ACTIVATION_ALIGNMENT;
  const size_t config_bytes = hexkl_micro_hmx_config_size();

  // Layout: [A act | scratch | flat-out | acc-rb] then HMX config in the end
  const size_t data_tiles = row_tiles_in_A   /*A act*/
                            + row_tiles_in_A /*scratch*/
                            + 1              /*flat-out*/
                            + 1;             /*acc-rb*/
  const size_t data_bytes = align * data_tiles;
  const size_t data_bytes_aligned = ((data_bytes + align - 1) / align) * align;

  // Total VTCM (aligned data + config)
  const size_t vtcm_size = data_bytes_aligned + config_bytes;

  uint8_t *vtcm_base =
      (uint8_t *)hexagon_runtime_alloc_1d(vtcm_size, align, true);
  if (!vtcm_base) {
    printf("[HEXKL_MICRO][ERROR] VTCM allocation failed\n");
    return AEE_ENOMEMORY;
  }

  const size_t hmx_config_offset = data_bytes_aligned;
  // Optional sanity checks
  if ((hmx_config_offset + config_bytes) > vtcm_size ||
      (hmx_config_offset % align)) {
    printf("[HEXKL_MICRO][ERROR] HMX config placement invalid\n");
    hexagon_runtime_free_1d(vtcm_base);
    return AEE_EFAILED;
  }

  printf("[HEXKL_MICRO] config_bytes=%zu, vtcm_base=%p, cfg_off=0x%zx\n",
         config_bytes, vtcm_base, hmx_config_offset);

  // HMX config after the aligned data region
  hexkl_micro_hmx_setup_acc_read_f16(vtcm_base, hmx_config_offset);

  // Iterate through rows of X at tile height stride
  for (uint32_t row = 0; row < N; row += HEXKL_HMX_F16_BLOCK_N_ROW) {
    // Load and layout one row of tiles from A. Store row starting at vtcm_base.
    for (int i = 0; i < row_tiles_in_A; i++) {
      // Each fp16 tile of A is 32x32 = 2048 bytes
      hexkl_micro_hmx_copy_submatrix_to_f16(
          vtcm_base,
          /*out_offset=*/HEXKL_HMX_ACTIVATION_ALIGNMENT * (row_tiles_in_A + i),
          /*input_matrix=*/matA,
          /*tile_row=*/row / HEXKL_HMX_F16_BLOCK_N_ROW,
          /*tile_col=*/i,
          /*input_rows=*/A_rows,
          /*input_cols=*/A_cols);
      hexkl_micro_hmx_rm_to_ah_f16(
          vtcm_base,
          /*activation_out_offset=*/HEXKL_HMX_ACTIVATION_ALIGNMENT * i,
          /*flat_in_offset=*/HEXKL_HMX_ACTIVATION_ALIGNMENT *
              (row_tiles_in_A + i));
    }
    // The tiles for one row of A are now loaded and laid out correctly
    // These tiles occupy the first (row_tiles_in_A *
    // HEXKL_HMX_ACTIVATION_ALIGNMENT bytes) of memory from vtcm_base.

    // Iterate through columns of X at tile width stride
    uint32_t col = 0;
    for (; col < M; col += 32) {
      hexkl_micro_hmx_acc_clear_f16();
      // Iterate through (one row of tiles in A) * (one col of tiles in W)
      for (int i = 0; i < row_tiles_in_A; i++) {

        hexkl_micro_hmx_rm_to_wh_f16(vtcm_base,
                                     /*weight_offset=*/weight_offset, matW,
                                     /*row_tile =*/i,
                                     /*col_tile =*/(col) / 32,
                                     /*wt_cols*/ n_col);

        hexkl_micro_hmx_mm_f16(
            vtcm_base,
            /*activation_offset=*/HEXKL_HMX_ACTIVATION_ALIGNMENT * i,
            /*weight_offset=*/weight_offset);
      }
      // Read 32x32 fp16 accumulator
      hexkl_micro_hmx_acc_read_f16(
          vtcm_base, hmx_config_offset,
          /*out_offset=*/HEXKL_HMX_ACTIVATION_ALIGNMENT * (row_tiles_in_A + 1));
      // Change layout to row major
      hexkl_micro_hmx_ah_to_rm_f16(
          vtcm_base,
          /*flat_out_offset=*/HEXKL_HMX_ACTIVATION_ALIGNMENT * row_tiles_in_A,
          /*activation_in_offset=*/HEXKL_HMX_ACTIVATION_ALIGNMENT *
              (row_tiles_in_A + 1));
      // Copy into X
      hexkl_micro_hmx_copy_f16_to_f32_submatrix(
          vtcm_base,
          /*in_offset=*/HEXKL_HMX_ACTIVATION_ALIGNMENT * row_tiles_in_A,
          /*output_matrix=*/matX,
          /*tile_row=*/row / HEXKL_HMX_F16_BLOCK_N_ROW,
          /*tile_col=*/col / HEXKL_HMX_F16_BLOCK_N_COL,
          /*output_rows=*/X_rows,
          /*output_cols=*/X_cols);
    }
  }

  hexagon_runtime_free_1d(vtcm_base);
  return ret;
}
}
