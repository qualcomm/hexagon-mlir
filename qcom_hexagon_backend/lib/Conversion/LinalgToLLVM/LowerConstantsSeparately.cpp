//===- LowerConstantsSeparately.cpp - Separate out large constants  ------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
//  This file provides the wrapper createLowerConstantsSeparatelyPass().
//
//===----------------------------------------------------------------------===//

#include "hexagon/Conversion/LinalgToLLVM/LowerConstantsSeparately.h"

#include "mlir/IR/BuiltinOps.h" // For ModuleOp
#include "mlir/Pass/Pass.h"     // For OperationPass

using namespace mlir;
using namespace hexagon;

// Create an instance of the LowerConstantsSeparatelyPass
std::unique_ptr<OperationPass<ModuleOp>>
hexagon::createLowerConstantsSeparatelyPass() {
  return std::make_unique<LowerConstantsSeparatelyPass>();
}
