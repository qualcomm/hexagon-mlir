//===- HexagonBufferAlias.cpp - implementation file           -------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

#include "HexagonBufferAlias.h"
#include "HexagonCommon.h"
#include <cassert>

namespace hexagon {

HexagonBufferAlias::HexagonBufferAlias(HexagonBuffer &buffer, size_t nbytes)
    : origBuffer(&buffer) {
  FARF(ALWAYS, "orig: %d, nbytes: %d", origBuffer->TotalBytes(), nbytes);
  assert(origBuffer->TotalBytes() == nbytes &&
         "Incorrect buffer size mentioned");
  assert(origBuffer->GetNdim() == 1 &&
         "Expected buffer to be 1D to create pointer table alias");
  pointerTable.resize(nbytes / CHUNK_SIZE);
  uint8_t *basePtr = reinterpret_cast<uint8_t *>(origBuffer->GetPointer());
  for (int i = 0; i < pointerTable.size(); i++) {
    pointerTable[i] = basePtr;
    basePtr += 2048;
  }
}

void *HexagonBufferAlias::GetPointerTableBase() { return pointerTable.data(); }

std::vector<void *> HexagonBufferAlias::GetPointerTable() {
  return pointerTable;
}

} // namespace hexagon
