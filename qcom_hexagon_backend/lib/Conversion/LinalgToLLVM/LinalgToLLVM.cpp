//===- LinalgToLLVM.cpp - TableGen'd LinalgToLinalg Passes  ---------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

#include "hexagon/Conversion/LinalgToLLVM/LinalgToLLVM.h"

using namespace mlir;
using namespace hexagon;

#define GEN_PASS_CLASSES
#include "hexagon/Conversion/LinalgToLLVM/Passes.h.inc"
