//===----------------------------------------------------------------------===//
// DMA2DDescLayoutTest.cpp - Host test for the v75+ 24-bit 2D descriptor.
//
// Device independent: it includes only UserDMADescriptors.h (no asm, no HMX)
// and builds descriptors with the same setters UserDMA::copy2D uses. The
// expected words are the `new24` oracle from exp/hmx/dma2d_probe/wrapper_dma2d.cpp
// (build_desc_new), which was confirmed on v79 (logs/dma2d_probe/REPORT.txt).
//
// This file is NOT part of the device gtest suite (test/CMakeLists.txt) and
// must not be added there. Build and run it on the host:
//
//   clang++ -std=c++17 -Wall -Wextra \
//       qcom_hexagon_backend/bin/runtime/test/DMA2DDescLayoutTest.cpp \
//       -o /tmp/dma2d_desc_test && /tmp/dma2d_desc_test
//
//===----------------------------------------------------------------------===//

#include "../UserDMA/UserDMADescriptors.h"

#include <cstdint>
#include <cstdio>
#include <cstring>

using namespace hexagon::userdma;

static int g_failures = 0;

static void checkWord(const char *tag, uint32_t got, uint32_t want) {
  if (got != want) {
    std::printf("FAIL %s: got 0x%08x want 0x%08x\n", tag, got, want);
    ++g_failures;
  }
}

static void checkBool(const char *tag, bool got, bool want) {
  if (got != want) {
    std::printf("FAIL %s: got %d want %d\n", tag, (int)got, (int)want);
    ++g_failures;
  }
}

// Mirrors the geometry writes of UserDMA::copy2D. comp/bypass/order/done are all
// zero for a plain copy, so the oracle's high bits of word1 stay clear.
static void build2D(void *desc, uint32_t src, uint32_t dst, uint32_t width,
                    uint32_t height, uint32_t srcStride, uint32_t dstStride) {
  std::memset(desc, 0, DMA_DESC_2D_SIZE);
  dmaDescSetState(desc, DESC_STATE_READY);
  dmaDescSetDone(desc, DESC_DONE_INCOMPLETE);
  dmaDescSetNext(desc, DMA_NULL_PTR);
  dmaDescSetDescSize(desc, DESC_DESCSIZE_2D);
  dmaDescSetDescType(desc, DESC_TYPE_2D_24BIT);
  dmaDescSetDstComp(desc, DESC_COMP_NONE);
  dmaDescSetSrcComp(desc, DESC_COMP_NONE);
  dmaDescSetRowSize(desc, width);
  dmaDescSetNrows(desc, height);
  dmaDescSetSrcStride(desc, srcStride);
  dmaDescSetDstStride(desc, dstStride);
  dmaDescSetOffset(desc, 0);
  dmaDescSetBypassDst(desc, DESC_BYPASS_OFF);
  dmaDescSetBypassSrc(desc, DESC_BYPASS_OFF);
  dmaDescSetOrder(desc, DESC_ORDER_NOORDER);
  dmaDescSetDone(desc, DESC_DONE_INCOMPLETE);
  dmaDescSetSrc(desc, src);
  dmaDescSetDst(desc, dst);
}

static void testOracleGeometry() {
  static_assert(sizeof(DMADesc2D) == DMA_DESC_2D_SIZE,
                "2D descriptor must be 32 bytes");
  uint32_t desc[8] = {0};
  // Probe A3 case new_ss65600: w=64, h=2, src_stride=65600, dst_stride=64.
  build2D(desc, 0x12345678u, 0x9abcdef0u, 64, 2, 65600, 64);
  checkWord("geom word0", desc[0], 0x00000000u);
  checkWord("geom word1", desc[1], 0x01000040u);
  checkWord("geom word2", desc[2], 0x12345678u);
  checkWord("geom word3", desc[3], 0x9abcdef0u);
  checkWord("geom word4", desc[4], 0x00000009u);
  checkWord("geom word5", desc[5], 0x02000040u);
  checkWord("geom word6", desc[6], 0x01004000u);
  checkWord("geom word7", desc[7], 0x00000000u);
}

static void testWideFieldsNotTruncated() {
  uint32_t desc[8] = {0};
  // Largest representable geometry: 24-bit strides, 16-bit nrows.
  build2D(desc, 1, 2, 0x00FFFFFFu, 0x0000FFFFu, 0x00FFFFFFu, 0x00FFFFFFu);
  checkWord("max word1", desc[1], 0x01FFFFFFu); // dst_stride | desc_size<<24
  checkWord("max word4", desc[4], 0x00000009u); // desc_type
  checkWord("max word5", desc[5], 0xFFFFFFFFu); // row_size | nrows_lo<<24
  checkWord("max word6", desc[6], 0xFFFFFFFFu); // nrows_hi | src_stride<<8
}

static void testRangePredicate() {
  checkBool("fits max",
            dma2DGeometryFits(0x00FFFFFFu, 0x0000FFFFu, 0x00FFFFFFu,
                              0x00FFFFFFu),
            true);
  checkBool("fits probe", dma2DGeometryFits(64, 2, 65600, 64), true);
  checkBool("width too big", dma2DGeometryFits(0x01000000u, 2, 64, 64), false);
  checkBool("height too big", dma2DGeometryFits(64, 0x00010000u, 64, 64),
            false);
  checkBool("src stride too big",
            dma2DGeometryFits(64, 2, 0x01000000u, 64), false);
  checkBool("dst stride too big",
            dma2DGeometryFits(64, 2, 64, 0x01000000u), false);

  // Why the range predicate must run before the setters: a 25-bit row_size is
  // masked down to 0 by the field write, the same silent truncation the v79
  // probe caught for strides above 0xFFFF.
  uint32_t desc[8] = {0};
  dmaDescSetRowSize(desc, 0x01000000u);
  checkWord("truncated row_size", desc[5] & DESC_ROWSIZE_MASK, 0x00000000u);
}

int main() {
  testOracleGeometry();
  testWideFieldsNotTruncated();
  testRangePredicate();
  if (g_failures) {
    std::printf("RESULT: FAIL (%d failed checks)\n", g_failures);
    return 1;
  }
  std::printf("RESULT: PASS\n");
  return 0;
}
