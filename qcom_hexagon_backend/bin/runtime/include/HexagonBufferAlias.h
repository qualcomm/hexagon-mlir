//===- HexagonBufferAlias.h - pointer table alias representation ----------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// Represent an alias of an existing HexagonBuffer.
// Used when we create a pointer table alias of a flat HexagonBuffer.
//
//===----------------------------------------------------------------------===//
#ifndef HEXAGON_BIN_RUNTIME_INCLUDE_HEXAGONBUFFERALIAS_H
#define HEXAGON_BIN_RUNTIME_INCLUDE_HEXAGONBUFFERALIAS_H

#include "HexagonBuffer.h"
#include <memory>
#include <vector>

struct Allocation;
namespace hexagon {

/// Represent an alias of an existing HexagonBuffer
///
/// Used when we create a pointer table alias of a flat
/// HexagonBuffer
class HexagonBufferAlias {
public:
  /// Allocate 1d (contiguous) memory within memory scopes.
  HexagonBufferAlias(HexagonBuffer &buffer, size_t nbytes);

  /// Destruction deallocates the underlying allocations.
  // ~HexagonBufferAlias();

  /// Prevent copy/move construction and assignment
  HexagonBufferAlias(const HexagonBufferAlias &) = delete;
  HexagonBufferAlias &operator=(const HexagonBufferAlias &) = delete;
  HexagonBufferAlias(HexagonBufferAlias &&) = delete;
  HexagonBufferAlias &operator=(HexagonBufferAlias &&) = delete;

  void *GetPointerTableBase();
  std::vector<void *> GetPointerTable();
  HexagonBuffer *origBuffer;

private:
  std::vector<void *> pointerTable;
};

} // namespace hexagon

#endif // HEXAGON_BIN_RUNTIME_INCLUDE_HEXAGONBUFFERALIAS_H
