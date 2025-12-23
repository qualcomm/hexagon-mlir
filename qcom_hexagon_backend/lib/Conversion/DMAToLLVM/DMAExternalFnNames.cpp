//===-- DMAExternalFnNames.cpp - DMA external function names --------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// This file defines external function names used in DMA to LLVM lowering.
//===----------------------------------------------------------------------===//

#include "hexagon/Conversion/DMAToLLVM/DMAExternalFnNames.h"

using namespace mlir;

// Helper function to get the name of DMA_Start
std::string hexagon::getDMAStartFnName() {
  static const std::string DMAStartFnName = "hexagon_runtime_dma_start";
  return DMAStartFnName;
}

// Helper function to get the name of 2D DMA_Start
std::string hexagon::getDMA2DStartFnName() {
  static const std::string DMA2DStartFnName = "hexagon_runtime_dma2d_start";
  return DMA2DStartFnName;
}

// Helper function to get the name of DMA_Wait
std::string hexagon::getDMAWaitFnName() {
  static const std::string DMAWaitFnName = "hexagon_runtime_dma_wait";
  return DMAWaitFnName;
}
