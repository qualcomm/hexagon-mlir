//===-- HmxTarget.h - what the HMX engine can take ------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// The engine's contract, as a value the passes consult -- not as logic buried in
// a rewrite pattern that names one op class.
//
// Upstream Triton has the same shape: an `Accelerate*` pass per op class, each
// asking `TargetInfo` whether the *operation* fits (`getMMAVersionSafe` +
// `supportMMA`). Keeping the capability here is what lets a second op class join
// the engine by writing a thin adapter that extracts its extents and element
// types and asks this table, instead of needing a second pass.
//
// See docs/hmx/hmx-generality.md for what is and is not general today.
//===----------------------------------------------------------------------===//

#ifndef HEXAGON_DIALECT_HMX_TRANSFORMS_HMXTARGET_H
#define HEXAGON_DIALECT_HMX_TRANSFORMS_HMXTARGET_H

#include "hexagon/Dialect/Hmx/IR/HmxCroutonLayout.h"
#include "mlir/IR/BuiltinTypes.h"
#include <algorithm>
#include <cstdint>

namespace mlir {
namespace hmx {

/// The HMX engine as this compiler knows it: the v1 engine on this device.
///
/// The contract is compile-time constants -- one crouton is a 32x32 fp16 block,
/// the device VTCM is 8 MiB and the engine shares it with the rest of the kernel
/// -- so there is no runtime target description to parse. The VTCM budget is the
/// one field a pass may narrow (a test, or a future device); it defaults to the
/// device's.
struct HmxTarget {
  /// One crouton is a 32x32 fp16 block; every extent the engine touches is a
  /// multiple of it.
  static constexpr int64_t tileEdge = crouton::kTileEdge;

  /// The engine needs a few rows before a tile can even be formed. llama.cpp
  /// draws the same line at HTP_MM_HMX_MIN_NROWS = 4.
  static constexpr int64_t minRows = 4;

  /// The device VTCM, in bytes (measured: docs/analysis/hmx-device-prerequisites.md
  /// -- 8 MiB). The crouton arrays the bridge stages, the resident weights and
  /// everything else the kernel puts in VTCM all come out of this one pool.
  static constexpr int64_t defaultVtcmBudget = 8 * 1024 * 1024;

  /// The budget every query is weighed against; defaults to the device's.
  int64_t vtcmBudget = defaultVtcmBudget;

  /// Whether the pipeline provides the allocator the crouton arrays need.
  ///
  /// The engine reads and writes VTCM only, and the crouton arrays are memory
  /// space 1, so a pipeline that does not lower space-1 allocations through the
  /// runtime VTCM pool (`enableConvertToHexagonmem` / the hexagonmem path) would
  /// leave them somewhere the engine cannot reach. Attributing there does not
  /// degrade -- it kills the DSP -- so it is refused instead.
  bool vtcmAllocator = true;

  /// The engine's one contraction contract: an f16 read-out (a wider result is
  /// that image widened afterwards, exactly what llama.cpp's f32 HMX matmul
  /// does), all three extents on the tile grid, and enough rows.
  ///
  /// The operands are the engine's fp16 croutons; a *source* may be fp32, in
  /// which case the pack that materialises the crouton quantises it (see
  /// `hmx.pack_act`). That is why a wider operand element type is admitted here
  /// rather than refused: the engine never sees it.
  ///
  /// Any op whose meaning reduces to such a contraction can ask this -- that is
  /// the whole of the engine's capability today. A second shape (a quantized
  /// operands contract, say) belongs here as another query, not as another pass.
  static bool isContractionOperand(Type elem) {
    return elem.isF16() || elem.isF32();
  }

  bool supportsContraction(int64_t m, int64_t n, int64_t k, Type lhsElem,
                           Type rhsElem, Type outElem) const {
    if (!isContractionOperand(lhsElem) || !isContractionOperand(rhsElem))
      return false;
    if (!outElem.isF16() && !outElem.isF32())
      return false;
    if (m <= minRows)
      return false;
    return m % tileEdge == 0 && n % tileEdge == 0 && k % tileEdge == 0;
  }

  /// How the crouton bridge pays for its residency: the M extent it walks in
  /// one block, and the crouton bytes that one block holds. A contraction too
  /// large to hold whole is walked in M blocks instead of being refused (see
  /// `planBridge`).
  struct BridgePlan {
    /// Rows of M one block covers: a multiple of the tile edge and a divisor of
    /// the contraction's M, so the blocks tile M exactly -- no partial block and
    /// no bounds guard. 0 means no bridge fits.
    int64_t blockM = 0;
    /// The bytes one block holds: its activation block, the whole weight and its
    /// fp16 read-out block. This is the peak VTCM the bridge asks for.
    int64_t bytes = 0;

    /// True when the contraction is walked in several M blocks.
    bool blocked(int64_t m) const { return blockM > 0 && blockM < m; }
    explicit operator bool() const { return blockM > 0; }
  };

  /// The crouton's element size in bytes: the engine's fp16, whatever the
  /// source's element type is (a wider source is quantised by the pack, its
  /// crouton array is still fp16).
  static constexpr int64_t croutonElemBytes = 2;

  /// The crouton footprint of a contraction operated on `rows` rows: both
  /// operands and the engine's fp16 read-out, in bytes. The read-out is fp16
  /// whatever the result's element type (a wider result is widened after the
  /// unpack, outside VTCM).
  static int64_t croutonBytes(int64_t rows, int64_t n, int64_t k) {
    return (rows * k + k * n + rows * n) * croutonElemBytes;
  }

  /// The engine's second question, next to `supportsF16Contraction`: legality
  /// says the engine *can* take this contraction, this says how the crouton
  /// bridge fits what is left of the pool. One hard gate, in the terse shape of
  /// upstream's K threshold (`supportMMA`: `if (k < 256 / bitWidth)`).
  ///
  /// When the whole contraction fits, the plan is the whole M and the bridge is
  /// emitted once. When it does not, M is walked in the largest block that
  /// fits -- shrinking the activation and the read-out, the two arrays that
  /// scale with M -- while the weight stays whole. Only when not even one block
  /// fits is the contraction refused (an empty plan), and the IR is left
  /// untouched: never a half-built hmx region the verifier could see.
  ///
  /// Blocking M rather than N: the weight and the read-out both scale with N,
  /// so a block along N would shrink the read-out but leave the weight whole
  /// either way; a block along M shrinks both the activation and the read-out,
  /// which is where a large MxN product's bytes are. N (the weight) becomes the
  /// binding term at the other end and is blocked the same way when it does.
  BridgePlan planBridge(int64_t m, int64_t n, int64_t k,
                        int64_t vtcmUsed) const {
    int64_t room = vtcmBudget - vtcmUsed;
    // The whole contraction wins when it fits. The strict `<` is the same
    // boundary the pass's budget remark is written against (`footprint >= room`
    // refuses).
    int64_t whole = croutonBytes(m, n, k);
    if (whole < room)
      return BridgePlan{m, whole};

    // Blocking M shrinks the activation and the read-out but not the weight:
    //   F(b) = b * (k + n) * croutonElemBytes + k * n * croutonElemBytes.
    // F is increasing in b, so the largest fitting b is one division away.
    int64_t perRow = (k + n) * croutonElemBytes;
    int64_t weightBytes = k * n * croutonElemBytes;
    if (perRow <= 0)
      return BridgePlan{};
    int64_t maxRows = (room - weightBytes - 1) / perRow;
    int64_t maxBlock = (maxRows / tileEdge) * tileEdge;
    if (maxBlock < tileEdge)
      return BridgePlan{};
    // Round the block down to a divisor of the tile count so every block is the
    // same static shape. One tile always divides, so a block exists whenever
    // `maxBlock` did.
    int64_t maxTiles = maxBlock / tileEdge;
    int64_t mTiles = m / tileEdge;
    int64_t blockTiles = 0;
    for (int64_t d = std::min(maxTiles, mTiles); d >= 1; --d) {
      if (mTiles % d == 0) {
        blockTiles = d;
        break;
      }
    }
    if (blockTiles == 0)
      return BridgePlan{};
    int64_t blockM = blockTiles * tileEdge;
    // A block that spans the whole M would have fit the branch above; a block
    // that does not divide M cannot tile it (only reachable for an M off the
    // tile grid, which the engine's contract already rejects).
    if (blockM >= m || m % blockM != 0)
      return BridgePlan{};
    return BridgePlan{blockM, croutonBytes(blockM, n, k)};
  }
};

} // namespace hmx
} // namespace mlir

#endif // HEXAGON_DIALECT_HMX_TRANSFORMS_HMXTARGET_H
