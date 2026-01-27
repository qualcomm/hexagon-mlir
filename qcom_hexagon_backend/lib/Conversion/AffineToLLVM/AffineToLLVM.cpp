
//===----------- AffineToLLVM.cpp - Affine to LLVM Passes -----------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// This file includes registration of passes that lower Affine dialect to
// LLVM dialect.
//
//===----------------------------------------------------------------------===//

#include "hexagon/Conversion/AffineToLLVM/AffineToLLVM.h"

using namespace mlir;
using namespace hexagon;

#define GEN_PASS_CLASSES
#include "hexagon/Conversion/AffineToLLVM/Passes.h.inc"
