//===- TmTensorDialect.cpp - TmTensor dialect implementation -------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

#include "hexagon/Dialect/TmTensor/IR/TmTensorDialect.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"

using namespace mlir;
using namespace mlir::tm_tensor;

/// Dialect creation, the instance will be owned by the context. This is the
/// point of registration of custom attrs, types and operations for the dialect.
void TmTensorDialect::initialize() {
  registerAttributes();
  registerTypes();
  registerOperations();
}

//===----------------------------------------------------------------------===//
// ODS-Generated Declarations
//===----------------------------------------------------------------------===//

#include "hexagon/Dialect/TmTensor/IR/TmTensorDialect.cpp.inc"
