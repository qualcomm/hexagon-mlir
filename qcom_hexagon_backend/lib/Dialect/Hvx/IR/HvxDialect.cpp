//===- HvxDialect.cpp - HVX Dialect ----------------------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

#include "hexagon/Dialect/Hvx/IR/HvxDialect.h"

using namespace mlir;
using namespace mlir::hvx;

/// Dialect creation, the instance will be owned by the context. This is the
/// point of registration of custom operations for the dialect.
void HvxDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "hexagon/Dialect/Hvx/IR/HvxOps.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// ODS-Generated Declarations
//===----------------------------------------------------------------------===//

#include "hexagon/Dialect/Hvx/IR/HvxDialect.cpp.inc"
