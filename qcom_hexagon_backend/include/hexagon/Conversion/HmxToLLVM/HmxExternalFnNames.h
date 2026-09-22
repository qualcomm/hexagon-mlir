//===- HmxExternalFnNames.h - Runtime leaf names for hmx ops --------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
// The one table that maps hmx ops to the runtime symbols in libhmxapi.a. Keeping
// it in one place is what makes the ABI auditable: this file and
// bin/runtime/hmx/include/HMXAPI.h are the two halves of the contract.
//===----------------------------------------------------------------------===//
#ifndef HEXAGON_CONVERSION_HMXTOLLVM_HMXEXTERNALFNNAMES_H
#define HEXAGON_CONVERSION_HMXTOLLVM_HMXEXTERNALFNNAMES_H

#include <string>

namespace mlir {
namespace hmx {

std::string getBiasInitUnitF16FnName();
std::string getBiasLoadF16FnName();
std::string getAccClearF16FnName();
std::string getAccStoreF16FnName();
std::string getMmaF16FnName();
std::string getPackActF16FnName();
std::string getPackWeightF16FnName();
// The f32-source twins: the same layout with the engine's fp16 conversion folded
// into the pack, for a source whose element type is f32 (see HMXAPI.h).
std::string getPackActF32FnName();
std::string getPackWeightF32FnName();
std::string getUnpackAccF16FnName();
std::string getUnpackAccF32FnName();
// Ranged (bulk) forms: one call covers `count` croutons, used when an op's
// `count` is above 1 (see HmxOps.td). The single names above stay the
// count==1 degenerate.
std::string getPackActF16BulkFnName();
std::string getPackWeightF16BulkFnName();
std::string getPackActF32BulkFnName();
std::string getPackWeightF32BulkFnName();
std::string getUnpackAccF16BulkFnName();
std::string getUnpackAccF32BulkFnName();

// `hmx.stage` / `hmx.await` deliberately reuse the existing DMA runtime entries
// (the interface plan fixes the cost at "the same one call as today's
// memref.dma_start/dma_wait"), and those symbols live in the DMA table rather
// than in libhmxapi.a. The forwarders below keep this file the single source of
// hmx runtime symbol names and make that reuse explicit instead of having the
// lowering reach across into the DMA table.
std::string getStageDmaStartFnName();
std::string getAwaitDmaWaitFnName();

} // namespace hmx
} // namespace mlir

#endif // HEXAGON_CONVERSION_HMXTOLLVM_HMXEXTERNALFNNAMES_H
