
//===- CRunnerUtils.cpp - Utils for MLIR execution ------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// This file implements basic functions to manipulate structured MLIR types at
// runtime. Entities in this file are meant to be retargetable, including on
// targets without a C++ runtime, and must be kept C compatible.
//
//===----------------------------------------------------------------------===//

#include "CRunnerUtils.h"

#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <string.h>

extern "C" void memrefCopy(int64_t elemSize, UnrankedMemRefType<char> *srcArg,
                           UnrankedMemRefType<char> *dstArg) {
  DynamicMemRefType<char> src(*srcArg);
  DynamicMemRefType<char> dst(*dstArg);

  int64_t rank = src.rank;

  // Handle empty shapes -> nothing to copy.
  for (int rankp = 0; rankp < rank; ++rankp)
    if (src.sizes[rankp] == 0)
      return;

  char *srcPtr = src.data + src.offset * elemSize;
  char *dstPtr = dst.data + dst.offset * elemSize;

  if (rank == 0) {
    memcpy(dstPtr, srcPtr, elemSize);
    return;
  }

  int64_t indices[rank];
  int64_t srcStrides[rank];
  int64_t dstStrides[rank];

  // Initialize index and scale strides.
  for (int rankp = 0; rankp < rank; ++rankp) {
    indices[rankp] = 0;
    srcStrides[rankp] = src.strides[rankp] * elemSize;
    dstStrides[rankp] = dst.strides[rankp] * elemSize;
  }

  // Coalesce the maximal run of trailing dimensions that is contiguous in both
  // operands into one memcpy. A trailing axis joins the block when its byte
  // stride equals the block's current byte size in both src and dst.
  int64_t copyBytes = elemSize;
  int64_t outerRank = rank;
  for (int64_t axis = rank - 1; axis >= 0; --axis) {
    if (srcStrides[axis] != copyBytes || dstStrides[axis] != copyBytes)
      break;
    copyBytes *= src.sizes[axis];
    --outerRank;
  }

  // Fully contiguous in both operands -> a single bulk copy.
  if (outerRank == 0) {
    memcpy(dstPtr, srcPtr, copyBytes);
    return;
  }

  // Iterate only the leading (non-coalesced) axes, copying one contiguous block
  // of copyBytes per step.
  int64_t readIndex = 0, writeIndex = 0;
  for (;;) {
    memcpy(dstPtr + writeIndex, srcPtr + readIndex, copyBytes);
    // Advance index and read position over the outer axes only.
    for (int64_t axis = outerRank - 1; axis >= 0; --axis) {
      // Advance at current axis.
      auto newIndex = ++indices[axis];
      readIndex += srcStrides[axis];
      writeIndex += dstStrides[axis];
      // If this is a valid index, we have our next index, so continue copying.
      if (src.sizes[axis] != newIndex)
        break;
      // We reached the end of this axis. If this is axis 0, we are done.
      if (axis == 0)
        return;
      // Else, reset to 0 and undo the advancement of the linear index that
      // this axis had. Then continue with the axis one outer.
      indices[axis] = 0;
      readIndex -= src.sizes[axis] * srcStrides[axis];
      writeIndex -= dst.sizes[axis] * dstStrides[axis];
    }
  }
}
