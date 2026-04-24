//===-- HexagonMemExternalFnNames.cpp - HexagonMem external fn names ------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// This file defines external function names used in HexagonMem to LLVM
// lowering.
//===----------------------------------------------------------------------===//

#include "hexagon/Conversion/HexagonMemToLLVM/HexagonMemExternalFnNames.h"

using namespace mlir;

std::string hexagonmem::getAllocFnName(bool isCroutonType,
                                       const std::string &deviceType) {
  if (isCroutonType) {
    return "hexagon_runtime_alloc_2d_dsp";
  } else {
    return "hexagon_runtime_alloc_1d_dsp";
  }
}

std::string hexagonmem::getDeallocFnName(bool isCroutonType,
                                         const std::string &deviceType) {
  if (isCroutonType) {
    return "hexagon_runtime_free_2d_dsp";
  } else {
    return "hexagon_runtime_free_1d_dsp";
  }
}

std::string hexagonmem::getCopyFnName(const std::string &deviceType) {
  return "hexagon_runtime_copy_dsp";
}

std::string
hexagonmem::getMemrefToCroutonFnName(const std::string &deviceType) {
  return "hexagon_runtime_build_crouton_dsp";
}

std::string
hexagonmem::getCroutonToMemrefFnName(const std::string &deviceType) {
  return "hexagon_runtime_get_contiguous_memref_dsp";
}
