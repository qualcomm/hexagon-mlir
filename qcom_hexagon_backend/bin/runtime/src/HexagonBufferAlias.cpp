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
         "Expected buffer to be 1D to create crouton alias");
  croutonTable.resize(nbytes / CROUTON_SIZE);
  uint8_t *basePtr = reinterpret_cast<uint8_t *>(origBuffer->GetPointer());
  for (int i = 0; i < croutonTable.size(); i++) {
    croutonTable[i] = basePtr;
    basePtr += 2048;
  }
}

void *HexagonBufferAlias::GetCroutonTableBase() { return croutonTable.data(); }

std::vector<void *> HexagonBufferAlias::GetCroutonTable() {
  return croutonTable;
}

} // namespace hexagon
