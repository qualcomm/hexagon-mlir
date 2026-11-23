//===- CompositeDialect.cpp - Composite dialect registration --------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

#include "hexagon/Dialect/Composite/IR/CompositeDialect.h"
#include "hexagon/Dialect/Composite/IR/CompositeOps.h"

using namespace mlir;
using namespace mlir::composite;

void CompositeDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "hexagon/Dialect/Composite/IR/CompositeOps.cpp.inc"
      >();
}

#include "hexagon/Dialect/Composite/IR/CompositeDialect.cpp.inc"
