// HexagonResources.h
#ifndef HEXAGON_RESOURCES_H
#define HEXAGON_RESOURCES_H

#include "HexagonAPI.h"

inline void AllocateHexagonResources() {
  if (!HexagonAPI::Global()->hasResources()) {
    HexagonAPI::Global()->AcquireResources();
  }
}

inline void DeallocateHexagonResources() {
  if (HexagonAPI::Global()->hasResources()) {
    HexagonAPI::Global()->ReleaseResources();
  }
}

#endif // HEXAGON_RESOURCES_H
