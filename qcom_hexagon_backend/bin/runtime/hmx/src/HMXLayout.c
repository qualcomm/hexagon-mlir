//===-- HMXLayout.c - crouton layout leaves (HVX) -------------------------===//
//
// See include/HMXAPI.h. This is the runtime form of the validated scaffold code
// (exp/hmx/primitives/): same permutation, same contract, HVX instead of scalar
// stores (device-measured ~26x on the whole leaf chain).
//
// Layout (verified byte-for-byte against the PRM figures and llama.cpp's
// producers/consumers by exp/hmx/oracle/):
//   byte(pair, stride) = 128*(pair>>1) + 4*stride + 2*(pair&1)
// AH: pair = row (M),     stride = input channel (K)
// WH: pair = input channel (K), stride = output channel (N)
// AR: pair = row (M),     stride = output channel (N)
// AH and WH are therefore the SAME permutation, which is why one packer serves
// both.
//
// The packer writes the tile with a halfword scatter (llama.cpp's
// hmx_interleave_rows_to_tiles idiom, at halfword granularity because the
// source's non-contiguous dim is the pair dim here, not the stride dim). The
// scatter's 64 halfwords are one 128 B block: element n is the even row's
// column n at byte 4n, element 32+n the odd row's at byte 4n+2. One scatter per
// block replaces the load/vand/vshuff/vshuffe/store chain. Scatter targets VTCM
// only, which the crouton destination always is.
//
// Contract: the source must be readable for 128 B starting at each row's tile
// start (the load is a full unaligned vector load, of which the low 64 B are
// used). Tiles staged in VTCM satisfy this by construction.
//
// All buffer arguments are plain addresses (see HMXAPI.h).
//===----------------------------------------------------------------------===//

#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "HMXAPI.h"

#define HMX_BLOCK_BYTES 128u

/* Scatter offsets for one 128 B block, one per scatter halfword: element n
 * (0..31) is the even row's column n at byte 4n, element 32+n the odd row's at
 * byte 4n+2. The pair (2j, 2j+1) therefore fills byte 128*j of the crouton. */
static const int16_t hmx__pack_offsets[64] __attribute__((aligned(128))) = {
    0,   4,   8,   12,  16,  20,  24,  28,  32,  36,  40,  44,  48,  52,
    56,  60,  64,  68,  72,  76,  80,  84,  88,  92,  96,  100, 104, 108,
    112, 116, 120, 124, 2,   6,   10,  14,  18,  22,  26,  30,  34,  38,
    42,  46,  50,  54,  58,  62,  66,  70,  74,  78,  82,  86,  90,  94,
    98,  102, 106, 110, 114, 118, 122, 126};

static inline HVX_Vector hmx__pack_offsets_vec(void) {
  return *(const HVX_Vector *)hmx__pack_offsets;
}

/* The two source rows as the scatter's 64 halfwords: [row0.lo64 | row1.lo64].
 * vror moves row1's 32 fp16 into the upper half, vmux keeps row0's lower half. */
static inline HVX_Vector hmx__pack_pair(HVX_Vector v0, HVX_Vector v1) {
  return Q6_V_vmux_QVV(Q6_Q_vsetq_R(64), v0, Q6_V_vror_VR(v1, 64));
}

/* Writes one 128 B block. The scatter region is the block itself (127 = the
 * block's last byte), so it can never cross a page. */
static inline void hmx__pack_scatter(unsigned dst_block, HVX_Vector data,
                                     HVX_Vector offs) {
  Q6_vscatter_RMVhV((size_t)(uintptr_t)dst_block, 127u, offs, data);
}

/* Packs a 32x32 fp16 sub-block of a row-major [rows][cols] matrix whose rows
 * are `src_stride` elements apart: 16 blocks, block j holding rows (2j, 2j+1).
 * `cols` is the column count only (the zero-fill/out-of-range boundary).
 * `offs` is hmx__pack_offsets_vec(), hoisted so a bulk call pays for it once. */
static inline __attribute__((always_inline)) void hmx__pack_32x32(
    unsigned dst, unsigned src, unsigned rows, unsigned cols,
    unsigned src_stride, unsigned tile_row, unsigned tile_col, HVX_Vector offs) {
  const unsigned c0 = tile_col * HMX_TILE_COLS;
  const unsigned ncols =
      (c0 >= cols) ? 0u
                   : (cols - c0 >= HMX_TILE_COLS ? HMX_TILE_COLS : cols - c0);
  const unsigned r0_base = tile_row * HMX_TILE_COLS;

  if (ncols == 0) {
    const HVX_Vector z = Q6_V_vzero();
    for (unsigned j = 0; j < 16u; ++j)
      *(HVX_Vector *)(uintptr_t)(dst + j * HMX_BLOCK_BYTES) = z;
    return;
  }

  /* Interior tile: full width and every one of the 16 row pairs inside `rows`,
   * so each block is two row loads and one scatter, with no boundary mask. A
   * matmul whose M/K (act) or K/N (weight) are multiples of 32 hits this on
   * every tile; only the edge tiles take the general path below. */
  if (ncols == HMX_TILE_COLS && r0_base + HMX_TILE_COLS <= rows) {
    const unsigned row_bytes = src_stride * 2u;
    const uint8_t *p0 =
        (const uint8_t *)(uintptr_t)(src + (r0_base * src_stride + c0) * 2u);
    if (src_stride == HMX_TILE_COLS) {
      /* The two rows are adjacent, so one 128 B load already holds the pair. */
      for (unsigned j = 0; j < 16u; ++j) {
        hmx__pack_scatter(dst + j * HMX_BLOCK_BYTES,
                          *(const HVX_UVector *)(const void *)p0, offs);
        p0 += 2u * row_bytes;
      }
      return;
    }
    for (unsigned j = 0; j < 16u; ++j) {
      const HVX_Vector v0 = *(const HVX_UVector *)(const void *)p0;
      const HVX_Vector v1 =
          *(const HVX_UVector *)(const void *)(p0 + row_bytes);
      hmx__pack_scatter(dst + j * HMX_BLOCK_BYTES, hmx__pack_pair(v0, v1),
                        offs);
      p0 += 2u * row_bytes;
    }
    return;
  }

  /* Edge: a row outside `rows`, or a column tail, becomes zeros. Both fold into
   * the same 64-halfword scatter, so the block is still written exactly once
   * (the mask keeps only the first `ncols` halfwords of each row). */
  const HVX_VectorPred keep = Q6_Q_vsetq_R(2u * ncols);
  const HVX_Vector zero = Q6_V_vzero();
  for (unsigned j = 0; j < 16u; ++j) {
    const unsigned r0 = r0_base + 2u * j;
    const unsigned r1 = r0 + 1u;
    HVX_Vector v0 = zero;
    HVX_Vector v1 = zero;
    if (r0 < rows)
      v0 = *(const HVX_UVector *)(const void *)(uintptr_t)(
          src + (r0 * src_stride + c0) * 2u);
    if (r1 < rows)
      v1 = *(const HVX_UVector *)(const void *)(uintptr_t)(
          src + (r1 * src_stride + c0) * 2u);
    if (ncols < HMX_TILE_COLS) {
      v0 = Q6_V_vmux_QVV(keep, v0, zero);
      v1 = Q6_V_vmux_QVV(keep, v1, zero);
    }
    hmx__pack_scatter(dst + j * HMX_BLOCK_BYTES, hmx__pack_pair(v0, v1), offs);
  }
}

void hmx_pack_act_f16(unsigned dst_addr, unsigned src_addr, unsigned src_rows,
                      unsigned src_cols, unsigned src_stride, unsigned tile_row,
                      unsigned tile_col) {
  hmx__pack_32x32(dst_addr, src_addr, src_rows, src_cols, src_stride, tile_row,
                  tile_col, hmx__pack_offsets_vec());
}

void hmx_pack_weight_f16(unsigned dst_addr, unsigned src_addr, unsigned k,
                         unsigned n, unsigned src_stride, unsigned k_tile,
                         unsigned n_tile) {
  /* Same permutation as the activation (see the file header). */
  hmx__pack_32x32(dst_addr, src_addr, k, n, src_stride, k_tile, n_tile,
                  hmx__pack_offsets_vec());
}

/* Pair-scatter offsets: lane i is column i of one loaded source row, and the
 * row's 64 columns are two consecutive K-tiles. Column i<32 goes to byte 4*i of
 * the first crouton, i>=32 to byte 2048 + 4*(i-32) of the second; the odd row's
 * table is the even one + 2. The per-row-pair block offset (128*j) rides on the
 * scatter's base, so these tables are constant. Validated byte-for-byte against
 * the layout oracle in exp/hmx/oracle/pack_pair_scatter_oracle.c. */
static const int16_t hmx__pack_pair_even[64] __attribute__((aligned(128))) = {
    0,   4,   8,   12,  16,  20,  24,  28,  32,  36,  40,  44,  48,  52,  56,
    60,  64,  68,  72,  76,  80,  84,  88,  92,  96,  100, 104, 108, 112, 116,
    120, 124, 2048, 2052, 2056, 2060, 2064, 2068, 2072, 2076, 2080, 2084, 2088,
    2092, 2096, 2100, 2104, 2108, 2112, 2116, 2120, 2124, 2128, 2132, 2136, 2140,
    2144, 2148, 2152, 2156, 2160, 2164, 2168, 2172};
static const int16_t hmx__pack_pair_odd[64] __attribute__((aligned(128))) = {
    2,   6,   10,  14,  18,  22,  26,  30,  34,  38,  42,  46,  50,  54,  58,
    62,  66,  70,  74,  78,  82,  86,  90,  94,  98,  102, 106, 110, 114, 118,
    122, 126, 2050, 2054, 2058, 2062, 2066, 2070, 2074, 2078, 2082, 2086, 2090,
    2094, 2098, 2102, 2106, 2110, 2114, 2118, 2122, 2126, 2130, 2134, 2138, 2142,
    2146, 2150, 2154, 2158, 2162, 2166, 2170, 2174};

/* Packs TWO consecutive column (K) tiles in one pass: per source row-pair, one
 * 128 B load + one scatter per row, 4 ops for 2 KB of croutons against 5 ops for
 * one (the single-tile path). Requires a source row wide enough that the 128 B
 * load stays inside it (src_stride >= 64) and a full 64-column span, so it is
 * the strided interior; everything else falls back to the single-tile path. */
static inline __attribute__((always_inline)) void hmx__pack_2tiles(
    unsigned dst, unsigned src, unsigned rows, unsigned cols,
    unsigned src_stride, unsigned tile_row, unsigned tile_col) {
  const unsigned c0 = tile_col * HMX_TILE_COLS;
  const unsigned r0_base = tile_row * HMX_TILE_COLS;
  if (src_stride >= 2u * HMX_TILE_COLS && cols - c0 >= 2u * HMX_TILE_COLS &&
      r0_base + HMX_TILE_COLS <= rows) {
    const HVX_Vector off_e = *(const HVX_Vector *)hmx__pack_pair_even;
    const HVX_Vector off_o = *(const HVX_Vector *)hmx__pack_pair_odd;
    const unsigned row_bytes = src_stride * 2u;
    const uint8_t *p =
        (const uint8_t *)(uintptr_t)(src + (r0_base * src_stride + c0) * 2u);
    for (unsigned j = 0; j < 16u; ++j) {
      const unsigned base = dst + j * HMX_BLOCK_BYTES;
      Q6_vscatter_RMVhV((size_t)(uintptr_t)base, 4095u, off_e,
                        *(const HVX_UVector *)(const void *)p);
      Q6_vscatter_RMVhV((size_t)(uintptr_t)base, 4095u, off_o,
                        *(const HVX_UVector *)(const void *)(p + row_bytes));
      p += 2u * row_bytes;
    }
    return;
  }
  hmx__pack_32x32(dst, src, rows, cols, src_stride, tile_row, tile_col,
                  hmx__pack_offsets_vec());
  hmx__pack_32x32(dst + HMX_TILE_BYTES, src, rows, cols, src_stride, tile_row,
                  tile_col + 1u, hmx__pack_offsets_vec());
}

/* Bulk pack: `n_col_tiles` consecutive column (K) tiles at one source row-tile
 * in one call, i.e. exactly calling the single entry with tile_col =
 * tile_col_start + t and the crouton at dst_addr + t * HMX_TILE_BYTES, for
 * t = 0..n_col_tiles-1. This walks the activation crouton array's contiguous
 * axis ([Mt, Kt]). The offsets vector is loaded once for the whole call, which
 * is what the per-call fixed cost in the leaf probe is. */
void hmx_pack_act_f16_bulk(unsigned dst_addr, unsigned src_addr,
                           unsigned src_rows, unsigned src_cols,
                           unsigned src_stride, unsigned tile_row,
                           unsigned tile_col_start, unsigned n_col_tiles) {
  const HVX_Vector offs = hmx__pack_offsets_vec();
  unsigned t = 0;
  for (; t + 2u <= n_col_tiles; t += 2u)
    hmx__pack_2tiles(dst_addr + t * HMX_TILE_BYTES, src_addr, src_rows, src_cols,
                     src_stride, tile_row, tile_col_start + t);
  for (; t < n_col_tiles; ++t)
    hmx__pack_32x32(dst_addr + t * HMX_TILE_BYTES, src_addr, src_rows, src_cols,
                    src_stride, tile_row, tile_col_start + t, offs);
}

/* Weight twin: the weight crouton array is [Nt, Kt] as well, but its K tile is
 * the source block's row tile (the packer reads a row-major [K][N] source), so
 * the run steps `k_tile` at a fixed `n_tile` and still writes consecutive
 * croutons. Same permutation as the activation (see the file header). */
void hmx_pack_weight_f16_bulk(unsigned dst_addr, unsigned src_addr, unsigned k,
                              unsigned n, unsigned src_stride,
                              unsigned k_tile_start, unsigned n_tile,
                              unsigned n_k_tiles) {
  /* The pair-scatter needs the source's contiguous axis to BE the crouton tile
   * axis, which holds for the activation ([M][K] -> [Mt][Kt]) but not for the
   * weight ([K][N] with the tile axis on the strided K rows), so the weight
   * stays on the single-tile path. */
  const HVX_Vector offs = hmx__pack_offsets_vec();
  for (unsigned t = 0; t < n_k_tiles; ++t)
    hmx__pack_32x32(dst_addr + t * HMX_TILE_BYTES, src_addr, k, n, src_stride,
                    k_tile_start + t, n_tile, offs);
}

/* ---- f32 sources --------------------------------------------------------- */
//
// An f32 source is quantised to the engine's fp16 in the pack itself. The
// conversion is the shuffle form llama.cpp's transfer_activation_chunk_fp32_to_fp16
// uses (`Q6_W_vcombine_VV` + `Q6_Vhf_equals_Wqf32`), and its output IS one
// 128 B crouton block -- the same block the f16 pack builds from [row0.lo64 |
// row1.lo64] -- so the f32 pack is two 128 B loads and one 128 B store per
// row-pair, with no scatter and no offsets table.
//
// The store is a 128 B aligned vector store: the destination block sits at
// `128 * j` inside a crouton array, whose base is 128 B aligned by construction
// (the pipeline's `buffer-alignment = 128` for the bufferization allocation, and
// the VTCM pool's own 2 KB alignment for anything crouton-sized), so every block
// is aligned. This is the same store llama.cpp's activation pack makes.
static inline HVX_Vector hmx__f32_pair_to_block(HVX_Vector v0, HVX_Vector v1) {
  /* The qf32 add is the v79 form of the sf32 -> qf32 convert (the direct
   * `Q6_Vqf32_equals_Vsf` is V81); adding zero is exact. Argument order matches
   * llama.cpp's helper, so the block's halves land as the engine expects. */
  const HVX_Vector zero = Q6_V_vzero();
  return Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(
      Q6_Vqf32_vadd_VsfVsf(v1, zero), Q6_Vqf32_vadd_VsfVsf(v0, zero)));
}

/* Packs a 32x32 sub-block of a row-major [rows][cols] f32 matrix whose rows are
 * `src_stride` elements apart. Same row-pair walk as hmx__pack_32x32: block j
 * holds rows (2j, 2j+1), out-of-range rows and columns read as zero. */
static inline __attribute__((always_inline)) void hmx__pack_32x32_f32(
    unsigned dst, unsigned src, unsigned rows, unsigned cols,
    unsigned src_stride, unsigned tile_row, unsigned tile_col) {
  const unsigned c0 = tile_col * HMX_TILE_COLS;
  const unsigned ncols =
      (c0 >= cols) ? 0u
                   : (cols - c0 >= HMX_TILE_COLS ? HMX_TILE_COLS : cols - c0);
  const unsigned r0_base = tile_row * HMX_TILE_COLS;

  if (ncols == 0) {
    const HVX_Vector z = Q6_V_vzero();
    for (unsigned j = 0; j < 16u; ++j)
      *(HVX_Vector *)(uintptr_t)(dst + j * HMX_BLOCK_BYTES) = z;
    return;
  }

  /* A partial column tile is masked after the load, exactly as the f16 pack
   * zero-fills: the source's 128 B window is read whole, and the columns past
   * `cols` become zero before the conversion. */
  const HVX_Vector zero = Q6_V_vzero();
  const HVX_VectorPred keep = Q6_Q_vsetq_R(ncols * 4u);
  for (unsigned j = 0; j < 16u; ++j) {
    const unsigned r0 = r0_base + 2u * j;
    const unsigned r1 = r0 + 1u;
    HVX_Vector v0 = zero;
    HVX_Vector v1 = zero;
    if (r0 < rows)
      v0 = *(const HVX_UVector *)(const void *)(uintptr_t)(
          src + (r0 * src_stride + c0) * 4u);
    if (r1 < rows)
      v1 = *(const HVX_UVector *)(const void *)(uintptr_t)(
          src + (r1 * src_stride + c0) * 4u);
    if (ncols < HMX_TILE_COLS) {
      v0 = Q6_V_vmux_QVV(keep, v0, zero);
      v1 = Q6_V_vmux_QVV(keep, v1, zero);
    }
    *(HVX_Vector *)(uintptr_t)(dst + j * HMX_BLOCK_BYTES) =
        hmx__f32_pair_to_block(v0, v1);
  }
}

void hmx_pack_act_f32(unsigned dst_addr, unsigned src_addr, unsigned src_rows,
                      unsigned src_cols, unsigned src_stride, unsigned tile_row,
                      unsigned tile_col) {
  hmx__pack_32x32_f32(dst_addr, src_addr, src_rows, src_cols, src_stride,
                      tile_row, tile_col);
}

void hmx_pack_weight_f32(unsigned dst_addr, unsigned src_addr, unsigned k,
                         unsigned n, unsigned src_stride, unsigned k_tile,
                         unsigned n_tile) {
  /* Same permutation as the activation (see the file header). */
  hmx__pack_32x32_f32(dst_addr, src_addr, k, n, src_stride, k_tile, n_tile);
}

void hmx_pack_act_f32_bulk(unsigned dst_addr, unsigned src_addr,
                           unsigned src_rows, unsigned src_cols,
                           unsigned src_stride, unsigned tile_row,
                           unsigned tile_col_start, unsigned n_col_tiles) {
  for (unsigned t = 0; t < n_col_tiles; ++t)
    hmx__pack_32x32_f32(dst_addr + t * HMX_TILE_BYTES, src_addr, src_rows,
                        src_cols, src_stride, tile_row, tile_col_start + t);
}

void hmx_pack_weight_f32_bulk(unsigned dst_addr, unsigned src_addr, unsigned k,
                              unsigned n, unsigned src_stride,
                              unsigned k_tile_start, unsigned n_tile,
                              unsigned n_k_tiles) {
  for (unsigned t = 0; t < n_k_tiles; ++t)
    hmx__pack_32x32_f32(dst_addr + t * HMX_TILE_BYTES, src_addr, k, n,
                        src_stride, k_tile_start + t, n_tile);
}

/* One chunk: two column tiles (64 columns) of one row-pair, or the 32-column
 * tail. `aligned` is a compile-time constant (see hmx__unpack_acc_f16_body):
 * when it is set the full chunk's 128-byte store addresses are known 128-byte
 * aligned and the aligned VMEM form is used. The PRM is explicit that VMEMU
 * "access[es] multiple L2 cache lines" and borrows the permute network, while an
 * aligned store does neither; the caller only sets `aligned` after proving that
 * the destination base is 128-byte aligned and its row stride is a multiple of
 * 64 elements. When it is unset, the store stays the unaligned form the 64-byte
 * row contract needs (odd destination rows land on 64-byte multiples only). */
static inline void hmx__unpack_chunk(unsigned dst_addr, unsigned src_ar_addr,
                                     unsigned block_j, unsigned col,
                                     unsigned ncols, unsigned r0, unsigned r1,
                                     unsigned dst_rows, unsigned dst_stride,
                                     const int aligned) {
  const uint8_t *cr = (const uint8_t *)(uintptr_t)(
      src_ar_addr + (col / HMX_TILE_COLS) * HMX_TILE_BYTES +
      block_j * HMX_BLOCK_BYTES);

  /* Full chunk: each tile holds the row-pair interleaved as [r0_0, r1_0,
   * r0_1, r1_1, ...], so one two-vector deal over both tiles yields row r0
   * (even lanes) and row r1 (odd lanes), 64 columns each. Each row is then one
   * 128-byte store, no stack temporary. Same idiom as llama.cpp's read-out
   * (flash-attn-ops.c, fa_o_store_thread_f16). */
  if (ncols > HMX_TILE_COLS) {
    const HVX_VectorPair rows = Q6_W_vdeal_VVR(
        *(const HVX_Vector *)(const void *)(cr + HMX_TILE_BYTES),
        *(const HVX_Vector *)(const void *)cr, -2);
    if (r0 < dst_rows) {
      void *dst = (void *)(uintptr_t)(dst_addr + (r0 * dst_stride + col) * 2u);
      if (aligned)
        *(HVX_Vector *)(uintptr_t)dst = Q6_V_lo_W(rows);
      else
        *(HVX_UVector *)(uintptr_t)dst = Q6_V_lo_W(rows);
    }
    if (r1 < dst_rows) {
      void *dst = (void *)(uintptr_t)(dst_addr + (r1 * dst_stride + col) * 2u);
      if (aligned)
        *(HVX_Vector *)(uintptr_t)dst = Q6_V_hi_W(rows);
      else
        *(HVX_UVector *)(uintptr_t)dst = Q6_V_hi_W(rows);
    }
    return;
  }

  /* 32-column tail: one tile, byte-wise store of the (possibly shorter) rows. */
  _Float16 chunk[2u * HMX_TILE_COLS] __attribute__((aligned(128)));
  *(HVX_Vector *)chunk = Q6_Vh_vdeal_Vh(*(const HVX_Vector *)(const void *)cr);
  if (r0 < dst_rows) {
    void *dst = (void *)(uintptr_t)(dst_addr + (r0 * dst_stride + col) * 2u);
    memcpy(dst, chunk, ncols * 2u);
  }
  if (r1 < dst_rows) {
    void *dst = (void *)(uintptr_t)(dst_addr + (r1 * dst_stride + col) * 2u);
    memcpy(dst, chunk + HMX_TILE_COLS, ncols * 2u);
  }
}

/* The read-out conversion, vectorised. One call unpacks a whole row-pair of the
 * read-out: it walks the crouton row, building both destination rows of the block
 * from the same two loads and writing each row as adjacent 128-byte stores.
 * `block_j` selects the 128-byte block of each AR crouton, i.e. rows (2j, 2j+1);
 * `src_ar_addr` is the first crouton of the row. Destination rows are separated
 * by `dst_stride` elements (not `dst_cols`, the width of the block: an N-split
 * store puts the block inside a wider matrix), so `dst_cols` bounds the columns
 * written while `dst_stride` is the row address step.
 *
 * Measured context: the scalar predecessor ran at ~10 MB/s and was 97% of the HMX
 * kernel's runtime; 64-byte stores reached 0.36 GB/s against 2.9 GB/s for the
 * kernel's own DDR streaming, which is what the 128-byte chunks are for, and
 * loading each crouton once for both rows halved the VTCM reads.
 *
 * Two independent general wins live in the loop shape here:
 *
 *   * The chunk body has two arms and `ncols` picks one: `ncols > HMX_TILE_COLS`
 *     is the full 64-column chunk (two crouton tiles, deal + two 128-byte
 *     stores), anything smaller is the 32-column byte-copy tail. With a runtime
 *     `ncols` the compiler cannot drop the tail arm. The old loop only made the
 *     full arm constant for the 4x-unrolled prefix (`dst_cols >= 256`), so every
 *     narrower read-out (N < 256: most decode/attention shapes) fell into the
 *     slow arm. Here *every full 64-column chunk* goes through a
 *     constant-`two_tiles` call and only the final partial chunk keeps the
 *     runtime `ncols`.
 *   * Alignment: for a 128-byte-aligned base and a row stride that is a multiple
 *     of 64 elements, every full-chunk store is 128-byte aligned, so the aligned
 *     VMEM form can be used instead of VMEMU. The test is a per-call invariant,
 *     so the whole body is instantiated twice with a compile-time flag rather
 *     than branching inside the loop (a per-chunk branch measured *slower* than
 *     the aligned store saved). A remainder is a multiple of 32 columns (the
 *     crouton width), i.e. exactly 32 when there is one, and that arm stays the
 *     byte-wise `memcpy`, so the alignment proof only has to cover full chunks.
 *
 * The full-chunk loop is unrolled by four so the next chunk's VTCM loads issue
 * while the current chunk's stores are in flight. */
static inline __attribute__((always_inline)) void hmx__unpack_acc_f16_body(
    unsigned dst_addr, unsigned src_ar_addr, unsigned dst_rows,
    unsigned dst_cols, unsigned dst_stride, unsigned tile_row, unsigned block_j,
    const int aligned) {
  const unsigned r0 = tile_row * HMX_TILE_ROWS + 2u * block_j;
  const unsigned r1 = r0 + 1u;
  const unsigned two_tiles = 2u * HMX_TILE_COLS;
  if (r0 >= dst_rows && r1 >= dst_rows)
    return;

  unsigned col = 0;
  for (; col + 4u * two_tiles <= dst_cols; col += 4u * two_tiles) {
    hmx__unpack_chunk(dst_addr, src_ar_addr, block_j, col, two_tiles, r0, r1,
                      dst_rows, dst_stride, aligned);
    hmx__unpack_chunk(dst_addr, src_ar_addr, block_j, col + two_tiles, two_tiles,
                      r0, r1, dst_rows, dst_stride, aligned);
    hmx__unpack_chunk(dst_addr, src_ar_addr, block_j, col + 2u * two_tiles,
                      two_tiles, r0, r1, dst_rows, dst_stride, aligned);
    hmx__unpack_chunk(dst_addr, src_ar_addr, block_j, col + 3u * two_tiles,
                      two_tiles, r0, r1, dst_rows, dst_stride, aligned);
  }
  /* Every remaining full chunk, still with a constant column count, so the
   * register-resident store arm is used for narrow read-outs too. */
  for (; col + two_tiles <= dst_cols; col += two_tiles)
    hmx__unpack_chunk(dst_addr, src_ar_addr, block_j, col, two_tiles, r0, r1,
                      dst_rows, dst_stride, aligned);
  /* The one partial chunk at the right edge (dst_cols % 64 in 1..63; a multiple
   * of 32 makes it exactly 32 and takes the byte-wise memcpy). */
  if (col < dst_cols) {
    const unsigned ncols = dst_cols - col;
    hmx__unpack_chunk(dst_addr, src_ar_addr, block_j, col, ncols, r0, r1,
                      dst_rows, dst_stride, aligned);
  }
}

void hmx_unpack_acc_f16(unsigned dst_addr, unsigned src_ar_addr,
                        unsigned dst_rows, unsigned dst_cols,
                        unsigned dst_stride, unsigned tile_row,
                        unsigned block_j) {
  /* dst_addr is a byte address and dst_stride a count of f16 elements, so the
   * row step is 2*dst_stride bytes: both are 128-byte multiples for the aligned
   * full-chunk store. The check is per call, not per chunk. */
  const int aligned = (dst_addr & 127u) == 0u && (dst_stride & 63u) == 0u;
  if (aligned)
    hmx__unpack_acc_f16_body(dst_addr, src_ar_addr, dst_rows, dst_cols,
                             dst_stride, tile_row, block_j, 1);
  else
    hmx__unpack_acc_f16_body(dst_addr, src_ar_addr, dst_rows, dst_cols,
                             dst_stride, tile_row, block_j, 0);
}

/* Bulk unpack: `n_pairs` consecutive row-pairs of one 32-row tile row in a
 * single call, i.e. the same as calling hmx_unpack_acc_f16 with block_j =
 * 0..n_pairs-1. Same contract (destination rows `tile_row*32 + 2*block_j`
 * separated by dst_stride, first dst_cols columns written, same alignment
 * rules); the call covers rows [tile_row*32, tile_row*32 + 2*n_pairs).
 *
 * The only difference from the single entry is where the per-call work lands:
 * the aligned test and the call/return are paid once for all pairs, and the
 * always_inline body is reused unchanged so the per-crouton work is identical.
 * This targets the ~52 cycle/call fixed cost the leaf probe measured (it did
 * NOT amortise over separate calls). */
void hmx_unpack_acc_f16_bulk(unsigned dst_addr, unsigned src_ar_addr,
                             unsigned dst_rows, unsigned dst_cols,
                             unsigned dst_stride, unsigned tile_row,
                             unsigned n_pairs) {
  /* Per-call invariant, hoisted out of the pair loop. */
  const int aligned = (dst_addr & 127u) == 0u && (dst_stride & 63u) == 0u;
  /* Rows only grow with block_j, so pairs past the first out-of-range one are
   * out of range too: clamp once instead of letting the inlined body's early
   * return end the whole bulk call. */
  const unsigned r0_base = tile_row * HMX_TILE_ROWS;
  unsigned n = 0u;
  if (r0_base < dst_rows)
    n = (dst_rows - r0_base + 1u) / 2u; /* pairs with r0 = base + 2j < rows */
  if (n > n_pairs)
    n = n_pairs;

  if (aligned) {
    for (unsigned j = 0; j < n; ++j)
      hmx__unpack_acc_f16_body(dst_addr, src_ar_addr, dst_rows, dst_cols,
                               dst_stride, tile_row, j, 1);
  } else {
    for (unsigned j = 0; j < n; ++j)
      hmx__unpack_acc_f16_body(dst_addr, src_ar_addr, dst_rows, dst_cols,
                               dst_stride, tile_row, j, 0);
  }
}

/* One 32-column tile of the fused tail: widen the whole crouton block (both
 * rows), add the residual rows when asked, store both f32 rows. `ncols` is 32
 * on the fast path (aligned stores) or the remainder (predicated stores). */
static inline void hmx__unpack_f32_tile(unsigned dst_addr, unsigned res_addr,
                                        unsigned has_res,
                                        const HVX_Vector *block,
                                        unsigned col, unsigned ncols,
                                        unsigned r0, unsigned r1,
                                        unsigned dst_rows, unsigned dst_stride,
                                        unsigned res_stride, HVX_Vector one) {
  /* Full-block read: one 128 B load holds rows (2j, 2j+1) interleaved as
   * [r0_0, r1_0, r0_1, r1_1, ...]. Widening via x1.0 is exact; .lo is row 0,
   * .hi is row 1 (the same split llama.cpp's transfer_output_chunk relies on).
   */
  const HVX_VectorPair widened = Q6_Wqf32_vmpy_VhfVhf(*block, one);
  HVX_Vector out0 = Q6_Vsf_equals_Vqf32(Q6_V_lo_W(widened));
  HVX_Vector out1 = Q6_Vsf_equals_Vqf32(Q6_V_hi_W(widened));

  if (has_res) {
    if (r0 < dst_rows) {
      const HVX_Vector r =
          *(const HVX_UVector *)(const void *)(uintptr_t)(res_addr +
                                                          (r0 * res_stride +
                                                           col) *
                                                              4u);
      out0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(out0, r));
    }
    if (r1 < dst_rows) {
      const HVX_Vector r =
          *(const HVX_UVector *)(const void *)(uintptr_t)(res_addr +
                                                          (r1 * res_stride +
                                                           col) *
                                                              4u);
      out1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(out1, r));
    }
  }

  if (ncols == HMX_TILE_COLS) {
    if (r0 < dst_rows)
      *(HVX_Vector *)(uintptr_t)(dst_addr + (r0 * dst_stride + col) * 4u) =
          out0;
    if (r1 < dst_rows)
      *(HVX_Vector *)(uintptr_t)(dst_addr + (r1 * dst_stride + col) * 4u) =
          out1;
  } else {
    /* Column tail: the address is still 128 B aligned (row start + k*128),
     * so a first-bytes predicate stores exactly the remainder. */
    const HVX_VectorPred keep = Q6_Q_vsetq_R(ncols * 4u);
    if (r0 < dst_rows)
      Q6_vmem_QRIV(keep,
                   (HVX_Vector *)(uintptr_t)(dst_addr +
                                             (r0 * dst_stride + col) * 4u),
                   out0);
    if (r1 < dst_rows)
      Q6_vmem_QRIV(keep,
                   (HVX_Vector *)(uintptr_t)(dst_addr +
                                             (r1 * dst_stride + col) * 4u),
                   out1);
  }
}

/* One row-pair of the fused tail: see hmx_unpack_acc_f32 below. Shared by the
 * single entry and the bulk loop so the per-pair work is identical. */
static inline __attribute__((always_inline)) void hmx__unpack_acc_f32_body(
    unsigned dst_addr, unsigned res_addr, unsigned has_res,
    unsigned src_ar_addr, unsigned dst_rows, unsigned dst_cols,
    unsigned dst_stride, unsigned res_stride, unsigned tile_row,
    unsigned block_j) {
  const unsigned r0 = tile_row * HMX_TILE_ROWS + 2u * block_j;
  const unsigned r1 = r0 + 1u;
  if (r0 >= dst_rows && r1 >= dst_rows)
    return;

  /* fp16 1.0: the widening multiplier. */
  const HVX_Vector one = Q6_Vh_vsplat_R(0x3C00);
  const unsigned full_tiles = dst_cols / HMX_TILE_COLS;
  const unsigned tail = dst_cols % HMX_TILE_COLS;

  for (unsigned t = 0; t < full_tiles; ++t) {
    const unsigned col = t * HMX_TILE_COLS;
    const HVX_Vector *block = (const HVX_Vector *)(const void *)(uintptr_t)(
        src_ar_addr + t * HMX_TILE_BYTES + block_j * HMX_BLOCK_BYTES);
    hmx__unpack_f32_tile(dst_addr, res_addr, has_res, block, col,
                         HMX_TILE_COLS, r0, r1, dst_rows, dst_stride,
                         res_stride, one);
  }
  if (tail) {
    const unsigned col = full_tiles * HMX_TILE_COLS;
    const HVX_Vector *block = (const HVX_Vector *)(const void *)(uintptr_t)(
        src_ar_addr + full_tiles * HMX_TILE_BYTES + block_j * HMX_BLOCK_BYTES);
    hmx__unpack_f32_tile(dst_addr, res_addr, has_res, block, col, tail, r0, r1,
                         dst_rows, dst_stride, res_stride, one);
  }
}

/* See HMXAPI.h. One call unpacks a whole row-pair of the read-out straight to
 * fp32, folding in the widening and the residual add that today run as two
 * separate DDR passes after hmx_unpack_acc_f16. */
void hmx_unpack_acc_f32(unsigned dst_addr, unsigned res_addr, unsigned has_res,
                        unsigned src_ar_addr, unsigned dst_rows,
                        unsigned dst_cols, unsigned dst_stride,
                        unsigned res_stride, unsigned tile_row,
                        unsigned block_j) {
  hmx__unpack_acc_f32_body(dst_addr, res_addr, has_res, src_ar_addr, dst_rows,
                           dst_cols, dst_stride, res_stride, tile_row, block_j);
}

/* Bulk fused tail: `n_pairs` consecutive row-pairs of one tile row in a single
 * call, i.e. block_j = 0..n_pairs-1. Rows only grow with block_j, so the first
 * fully out-of-range pair stops the walk (the same clamp the fp16 bulk uses).
 * All per-call preconditions are unchanged from the single entry. */
void hmx_unpack_acc_f32_bulk(unsigned dst_addr, unsigned res_addr,
                             unsigned has_res, unsigned src_ar_addr,
                             unsigned dst_rows, unsigned dst_cols,
                             unsigned dst_stride, unsigned res_stride,
                             unsigned tile_row, unsigned n_pairs) {
  const unsigned r0_base = tile_row * HMX_TILE_ROWS;
  for (unsigned j = 0; j < n_pairs; ++j) {
    if (r0_base + 2u * j >= dst_rows)
      break;
    hmx__unpack_acc_f32_body(dst_addr, res_addr, has_res, src_ar_addr, dst_rows,
                             dst_cols, dst_stride, res_stride, tile_row, j);
  }
}
