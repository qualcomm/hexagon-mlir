//===- CRunnerUtils.h - Utils for debugging MLIR execution ----------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// This file declares basic classes and functions to manipulate structured MLIR
// types at runtime. Entities in this file must be compliant with C++11 and be
// retargetable, including on targets without a C++ runtime.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_EXECUTIONENGINE_CRUNNERUTILS_H
#define MLIR_EXECUTIONENGINE_CRUNNERUTILS_H

#include "hexagon/runtime/memref_runtime.h"

#include <array>
#include <cstdint>

template <typename T> using DynamicMemRefType = DynamicMemRefView<T>;

//===----------------------------------------------------------------------===//
// Small runtime support library for memref.copy lowering during codegen.
//===----------------------------------------------------------------------===//
extern "C" void memrefCopy(int64_t elemSize, ::UnrankedMemRefType<char> *src,
                           ::UnrankedMemRefType<char> *dst);

#endif // MLIR_EXECUTIONENGINE_CRUNNERUTILS_H
