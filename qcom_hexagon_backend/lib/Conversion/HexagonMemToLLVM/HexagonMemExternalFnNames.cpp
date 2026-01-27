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

std::string hexagonmem::getAllocFnName() {
  static const std::string allocFnName = "hexagon_runtime_alloc_1d";
  return allocFnName;
}

std::string hexagonmem::getDeallocFnName() {
  static const std::string deallocFnName = "hexagon_runtime_free_1d";
  return deallocFnName;
}

std::string hexagonmem::getCopyFnName() { return "hexagon_runtime_copy"; }
