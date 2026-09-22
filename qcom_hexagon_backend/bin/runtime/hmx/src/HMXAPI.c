//===-- HMXAPI.c - HMX leaves: environment, bias, accumulator, mma --------===//
//
// See include/HMXAPI.h for the design and the device-measured facts. Everything
// here was verified on the phone (Hexagon v79):
//   * the crouton buffers are the runtime's VTCM, placed by the compiler
//   * the bias region layout, and that it must be loaded before the read-out
//   * the read-out must be the fused `mxmem(..):after.hf = acc`
//   * correctness: A=I / W=I / random K=1024 all pass with zero tolerance
//===----------------------------------------------------------------------===//

#include <hmx_hexagon_protos.h>
#include <stdint.h>

#include "HMXAPI.h"


/* ---- bias / convert state ------------------------------------------------ */
// 256 B = 32 channels x 64 bit, split into two 128 B vectors: the first holds the
// lower 32 bits of every channel, the second the upper 32 bits. Identity is
//   lower word = 0x00003C00  (Scale = fp16 1.0 in [15:0], Output bias = 0)
//   upper word = 0           (extras, Shape = identity, Input bias = 0)
void hmx_bias_init_unit_f16(unsigned bias_addr) {
  uint32_t *w = (uint32_t *)(uintptr_t)bias_addr;
  for (unsigned ch = 0; ch < 32u; ++ch)
    w[ch] = 0x00003C00u;
  for (unsigned ch = 32u; ch < 64u; ++ch)
    w[ch] = 0x00000000u;
}

void hmx_bias_load_f16(unsigned bias_addr, unsigned set) {
  /* Operand = address[31:8] | set[1:0]; the address is 256 B aligned. */
  Q6_bias_mxmem2_A((void *)(uintptr_t)(bias_addr | (set & 3u)));
}

/* ---- accumulator -------------------------------------------------------- */
void hmx_acc_clear_f16(void) {
 Q6_mxclracc_hf(); }

void hmx_acc_store_f16(unsigned dst_addr, unsigned set) {
  /* Fused convert+clear store. `set` documents which bias set the caller loaded;
   * the fused form uses the set selected by the cvt path (we always load 0). */
  (void)set;
  Q6_mxmem_AR_after_hf((void *)(uintptr_t)dst_addr, 0);
}

/* ---- multiply-accumulate ------------------------------------------------- */
void hmx_mma_f16(unsigned act_addr, unsigned wt_addr, unsigned n_croutons) {
  /* Activation Rt: [31:11] dC = n-1, [10:7],[1] spatial mask = 11111 (the crouton
   * organization), [6:2] input channel stop = 31. Weight Rt: [31:7] dW =
   * distance to the last 128 B vector, [6:0] reserved = all 1s. */
  unsigned act_rt = (2047u) | ((n_croutons - 1u) << 11);
  unsigned wt_rt = ((16u * n_croutons - 1u) << 7) | 0x7Fu;
  Q6_activation_hf_mxmem_RR_deep(act_addr, act_rt);
  Q6_weight_hf_mxmem_RR(wt_addr, wt_rt);
}
