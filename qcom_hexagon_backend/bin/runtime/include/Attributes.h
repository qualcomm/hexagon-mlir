//===- Attributes.h -   macros to inline runtime functions  ---------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// Contains Macros to inline runtime functions
//
//===----------------------------------------------------------------------===//

#ifndef ATTRIBUTES_H
#define ATTRIBUTES_H

#ifndef HEXAGON_INTRIN_INLINE
#define HEXAGON_INTRIN_INLINE                                                  \
  inline __attribute__((unused, used, always_inline, visibility("hidden")))
#endif

#ifndef HEXAGON_INTRIN
#define HEXAGON_INTRIN __attribute__((visibility("hidden")))
#endif

#endif // ATTRIBUTES_H
