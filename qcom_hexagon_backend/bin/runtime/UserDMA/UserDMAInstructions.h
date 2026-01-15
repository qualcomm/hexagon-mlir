//===- UserDMAInstructions.h - Hexagon User DMA Instructions --------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// Reference: Qualcomm Hexagon User DMA Specification 80-V9418-29 Rev. D
//
//===----------------------------------------------------------------------===//

#ifndef HEXAGONBACKEND_BIN_RUNTIME_USERDMA_INSTRUCTIONS_H
#define HEXAGONBACKEND_BIN_RUNTIME_USERDMA_INSTRUCTIONS_H

namespace hexagon {
namespace userdma {
inline unsigned int dmpause() {
  unsigned int dm0 = 0;
  asm volatile(" %0 = dmpause" : "=r"(dm0));
  return dm0;
}

inline void dmstart(void *next) {
  asm volatile(" release(%0):at" : : "r"(next));
  asm volatile(" dmstart(%0)" : : "r"(next));
}

inline void dmlink(void *tail, void *next) {
  asm volatile(" release(%0):at" : : "r"(next));
  asm volatile(" dmlink(%0, %1)" : : "r"(tail), "r"(next));
}

inline unsigned int dmpoll() {
  unsigned int dm0 = 0;
  asm volatile(" %0 = dmpoll" : "=r"(dm0));
  return dm0;
}

inline unsigned int dmwait() {
  unsigned int dm0 = 0;
  asm volatile(" %0 = dmwait" : "=r"(dm0));
  return dm0;
}

inline void dmresume(unsigned int dm0) {
  asm volatile(" dmresume(%0)" : : "r"(dm0));
}

inline unsigned int dmsyncht() {
  unsigned int dm0 = 0;
  asm volatile(" %0 = dmsyncht" : "=r"(dm0));
  return dm0;
}

inline unsigned int dmtlbsynch() {
  unsigned int dm0 = 0;
  asm volatile(" %0 = dmtlbsynch" : "=r"(dm0));
  return dm0;
}

inline unsigned int dmcfgrd(unsigned int dmindex) {
  unsigned int data = 0;
  asm volatile(" %0 = dmcfgrd(%1)" : "=r"(data) : "r"(dmindex));
  return data;
}

inline void dmcfgwr(unsigned int dmindex, unsigned int data) {
  asm volatile(" dmcfgwr(%0, %1)" : : "r"(dmindex), "r"(data));
}
} // namespace userdma
} // namespace hexagon

#endif // USER_DMA_INSTRUCTIONS_H
