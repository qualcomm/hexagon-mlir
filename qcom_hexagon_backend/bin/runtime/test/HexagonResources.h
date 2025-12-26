//===- HexagonResources.h - implementation file                  ----------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

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
