//===-- HMXAPI.h - HMX leaf primitives (runtime module) -------------------===//
//
// Device-side implementation of the HMX leaves that MLIR calls into. This is the
// "final form" of the exp/hmx/primitives scaffold: same semantics, compiled by
// the SDK's hexagon-clang (the only compiler that knows HMX) and linked into the
// kernel by LinkRuntimeModules.
//
// Design (docs/hmx/hmx-system-design.md, docs/hmx/adr-001-hmx-ir-ownership.md):
//   * LEAF granularity: the tile loop, tiling, VTCM layout and pipelining stay in
//     MLIR. This module only provides the innermost operations.
//   * **Addresses, not pointers.** MLIR can only pass integers across the ABI
//     cleanly, and the HMX instructions take addresses in registers anyway, so
//     every buffer argument here is a plain `unsigned` address. The kernel
//     obtains them from the buffer addresses the compiler passes in.
//   * No pointer-typed or `_Float16 *` arguments anywhere.
//
// Measured on device (OnePlus 13, Hexagon v79):
//   * HMX's prerequisites (power up / a compute-resource context carrying the HMX
//     attribute / the lock) are held by the runtime, not by the kernel: the
//     runtime's HexagonAPI constructor acquires them before any kernel runs, and
//     the kernel's VTCM buffers come out of the same runtime pool.
//   * The bias registers MUST be loaded before the read-out, otherwise the
//     accumulator is scaled by the reset value 0 and reads back all zeros.
//   * The read-out must use the FUSED form `mxmem(..):after.hf = acc`; the split
//     form from the (V81) PRM zeroes negative values on v79 silicon.
//===----------------------------------------------------------------------===//

#ifndef HEXAGON_RUNTIME_HMXAPI_H
#define HEXAGON_RUNTIME_HMXAPI_H

#ifdef __cplusplus
extern "C" {
#endif

#define HMX_TILE_BYTES 2048u /* one fp16 32x32 crouton */
#define HMX_TILE_ROWS 32u
#define HMX_TILE_COLS 32u
#define HMX_BIAS_BYTES 256u /* one bias register set */

/* ---- environment --------------------------------------------------------- */
/* The HMX environment is the runtime's, not the kernel's: HexagonAPI acquires and
 * locks the HMX context, and VTCMPool owns the VTCM. The compiler places every
 * crouton array in that VTCM (memory space 1) and hands the leaves the buffer
 * addresses, so there is nothing to acquire here. A separate
 * HAP_compute_res_acquire of our own -- which this module used to make -- takes
 * the megabytes out of the pool instead, and the pool then fails its own
 * "at least 1 MB available" check and kills the DSP process. */

/* ---- bias / convert state (256 B, 256 B aligned) ------------------------- */
/* Identity conversion: scale 1.0, no output bias, identity shaping. */
void hmx_bias_init_unit_f16(unsigned bias_addr);
/* Loads bias register set `set` (0..3) from VTCM. */
void hmx_bias_load_f16(unsigned bias_addr, unsigned set);

/* ---- accumulator -------------------------------------------------------- */
void hmx_acc_clear_f16(void);
/* acc -> memory: one 2 KB fp16 crouton in AR layout, converted with bias set
 * `set` and clearing the accumulator. Fused form (see the file header). */
void hmx_acc_store_f16(unsigned dst_addr, unsigned set);

/* ---- multiply-accumulate ------------------------------------------------- */
/* `n_croutons` activation croutons x weight croutons, accumulated. The two
 * loads must be issued as a pair in one packet, so they live in one function. */
void hmx_mma_f16(unsigned act_addr, unsigned wt_addr, unsigned n_croutons);

/* ---- layout leaves ------------------------------------------------------- */
/* Pack a 32x32 fp16 sub-block of a row-major [rows][cols] fp16 matrix at
 * `src_addr` into the 2 KB crouton at `dst_addr` (out-of-range = zero filled).
 * The source rows are addressed by `src_stride` *elements*: `src_stride ==
 * src_cols` for a dense matrix, but a source that is one N-tile of a wider
 * matrix materialises as a strided view (the store-side twin is a `memref<M x
 * BN, strided<[N, 1]>>` with `N > BN`) and there `src_stride` is the wider row
 * stride N. `src_cols`/`n` stay the column count and are used only for the
 * out-of-range/zero-fill boundary: hard-wiring the row stride to the width
 * reads the wrong element of every row past the first, a wrong result whose
 * error grows with the row index. This is the source-side twin of the unpack
 * `dst_stride`. AH and WH are the same permutation, so both entry points share
 * one implementation. */
void hmx_pack_act_f16(unsigned dst_addr, unsigned src_addr, unsigned src_rows,
                      unsigned src_cols, unsigned src_stride, unsigned tile_row,
                      unsigned tile_col);
void hmx_pack_weight_f16(unsigned dst_addr, unsigned src_addr, unsigned k,
                         unsigned n, unsigned src_stride, unsigned k_tile,
                         unsigned n_tile);
/* Bulk form of the above: one call covers `n_col_tiles` consecutive column
 * (K) tiles of one source row-tile, i.e. exactly calling hmx_pack_act_f16 with
 * tile_col = tile_col_start + t and the crouton at dst_addr + t *
 * HMX_TILE_BYTES, for t = 0..n_col_tiles-1. It runs along the activation
 * crouton array's contiguous axis ([Mt, Kt], K in dim 1), and the per-call
 * fixed cost (the scatter-offsets load) is paid once for the whole call. */
void hmx_pack_act_f16_bulk(unsigned dst_addr, unsigned src_addr,
                           unsigned src_rows, unsigned src_cols,
                           unsigned src_stride, unsigned tile_row,
                           unsigned tile_col_start, unsigned n_col_tiles);
/* Weight twin: the weight crouton array is [Nt, Kt] (K in dim 1 too), but its
 * K tile is the source block's *row* tile, so one call covers `n_k_tiles`
 * consecutive row (K) tiles of one source column-tile: exactly calling
 * hmx_pack_weight_f16 with k_tile = k_tile_start + t and the crouton at
 * dst_addr + t * HMX_TILE_BYTES, for t = 0..n_k_tiles-1. `n_tile` is the fixed
 * source column tile. */
void hmx_pack_weight_f16_bulk(unsigned dst_addr, unsigned src_addr, unsigned k,
                              unsigned n, unsigned src_stride,
                              unsigned k_tile_start, unsigned n_tile,
                              unsigned n_k_tiles);
/* f32 source: the same pack, with the engine's fp16 conversion folded in. An
 * f32 operand is quantised to fp16 by the pack rather than by a separate
 * narrowing pass over DDR (which would cost a whole extra read and write of the
 * operand for a conversion the pack's shuffle performs anyway). The conversion
 * is the shuffle form llama.cpp's transfer_activation_chunk_fp32_to_fp16 uses,
 * whose output IS one 128 B crouton block, so this pack is two 128 B loads and
 * one 128 B store per row-pair -- no scatter.
 *
 * Everything else is the f16 contract unchanged: rows are `src_stride` elements
 * apart (a strided view of a wider matrix threads its real row stride), an
 * out-of-range row or column reads as zero, and the source must be readable for
 * 128 B (32 f32 elements) from each row's tile start. HMX's own contract makes
 * the operand's extents multiples of the tile edge, so the last tile's window
 * ends exactly at the row's last element. */
void hmx_pack_act_f32(unsigned dst_addr, unsigned src_addr, unsigned src_rows,
                      unsigned src_cols, unsigned src_stride, unsigned tile_row,
                      unsigned tile_col);
void hmx_pack_weight_f32(unsigned dst_addr, unsigned src_addr, unsigned k,
                         unsigned n, unsigned src_stride, unsigned k_tile,
                         unsigned n_tile);
/* Bulk forms of the f32 pack, with the same range semantics as their f16 twins
 * (`n_col_tiles` consecutive K tiles at one source row-tile for the activation,
 * `n_k_tiles` consecutive K tiles at one source column-tile for the weight). */
void hmx_pack_act_f32_bulk(unsigned dst_addr, unsigned src_addr,
                           unsigned src_rows, unsigned src_cols,
                           unsigned src_stride, unsigned tile_row,
                           unsigned tile_col_start, unsigned n_col_tiles);
void hmx_pack_weight_f32_bulk(unsigned dst_addr, unsigned src_addr, unsigned k,
                              unsigned n, unsigned src_stride,
                              unsigned k_tile_start, unsigned n_tile,
                              unsigned n_k_tiles);

/* Unpack one AR crouton row-pair into a row-major fp16 block at `dst_addr`
 * (vectorised; the fp32 image is a widening in the compiler). The destination
 * rows are (tile_row*32 + 2*block_j, +1) and are separated by `dst_stride`
 * elements, while only the first `dst_cols` columns of each are written.
 * `dst_stride == dst_cols` for a dense block, but a block that is one N-tile of
 * a wider matrix (the store of an N-split matmul, which materialises as a
 * strided `memref<M x BN, strided<[N, 1]>>`) is a strided view: there
 * `dst_stride` is the wider row stride N. The leaf addresses destination rows
 * by `dst_stride`, so passing the width there writes the block contiguously and
 * an N-split store keeps only its first M*BN elements. Full 64-column chunks are
 * written with 128-byte stores; the leaf picks the aligned VMEM form when the
 * destination base is 128-byte aligned and `dst_stride` is a multiple of 64
 * elements, and otherwise (odd 64-byte rows) uses the unaligned form, so there
 * is still no alignment precondition beyond the source contract. */
void hmx_unpack_acc_f16(unsigned dst_addr, unsigned src_ar_addr,
                        unsigned dst_rows, unsigned dst_cols,
                        unsigned dst_stride, unsigned tile_row,
                        unsigned block_j);
/* Bulk form of the above: one call covers `n_pairs` consecutive row-pairs of
 * tile row `tile_row`, i.e. exactly calling hmx_unpack_acc_f16 with block_j =
 * 0..n_pairs-1 on the same arguments. Rows [tile_row*32, tile_row*32 +
 * 2*n_pairs) are written; the row stride / src / alignment preconditions are
 * unchanged. Exists to move the per-call fixed cost (its measured ~52 cycles)
 * to one call for many pairs. */
void hmx_unpack_acc_f16_bulk(unsigned dst_addr, unsigned src_ar_addr,
                             unsigned dst_rows, unsigned dst_cols,
                             unsigned dst_stride, unsigned tile_row,
                             unsigned n_pairs);
/* Fused tail: unpack one AR crouton row-pair straight into row-major fp32.
 * One call covers destination rows (tile_row*32 + 2*block_j, +1) across the
 * first `dst_cols` columns of rows with `dst_stride` elements: each 32-column
 * tile is one aligned 128 B crouton load, widened via x1.0 (exact, the same
 * idiom as llama.cpp's transfer_output_chunk_fp16_to_fp32), optionally added
 * with the f32 residual row (`res_addr`, stride `res_stride`, is read iff
 * `has_res` is nonzero), and written with aligned 128 B stores. The column
 * tail (`dst_cols % 32`) uses predicated stores; out-of-range rows are
 * skipped. No stack temporaries; one VTCM read per tile serves both rows.
 *
 * Preconditions: `src_ar_addr` is the row's first crouton in VTCM (same as
 * hmx_unpack_acc_f16); every stored 128 B unit is 128-byte aligned, i.e.
 * `dst_addr` is 128-byte aligned and `dst_stride % 32 == 0` (the residual
 * uses unaligned loads, so it has no alignment requirement). The stride is
 * separate from the width on purpose: the valid width may end mid-row while
 * every row still starts aligned -- the same split llama.cpp's chunk transfer
 * uses. Phase 1 is validated on hexagon-sim only; the wiring phase must
 * guarantee the dst alignment before this runs on device. */
void hmx_unpack_acc_f32(unsigned dst_addr, unsigned res_addr, unsigned has_res,
                        unsigned src_ar_addr, unsigned dst_rows,
                        unsigned dst_cols, unsigned dst_stride,
                        unsigned res_stride, unsigned tile_row,
                        unsigned block_j);
/* Bulk form of the fused tail: `n_pairs` consecutive row-pairs of tile row
 * `tile_row` in one call, i.e. block_j = 0..n_pairs-1 on the same arguments.
 * The alignment preconditions are unchanged; out-of-range pairs stop the walk. */
void hmx_unpack_acc_f32_bulk(unsigned dst_addr, unsigned res_addr,
                             unsigned has_res, unsigned src_ar_addr,
                             unsigned dst_rows, unsigned dst_cols,
                             unsigned dst_stride, unsigned res_stride,
                             unsigned tile_row, unsigned n_pairs);

#ifdef __cplusplus
}
#endif
#endif /* HEXAGON_RUNTIME_HMXAPI_H */
