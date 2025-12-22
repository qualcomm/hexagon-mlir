//===--- TTXTypes.cpp - TTX dialect types -----------------------*- C++ -*-===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// This file implements the TTX dialect types.
//===----------------------------------------------------------------------===//

#include "hexagon/Dialect/TTX/IR/TTXDialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::ttx;

/// Dialect creation, the instance will be owned by the context. This is the
/// point of registration of custom types for the dialect.
void TTXDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "hexagon/Dialect/TTX/IR/TTXTypes.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// ODS-Generated Declarations
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "hexagon/Dialect/TTX/IR/TTXTypes.cpp.inc"
