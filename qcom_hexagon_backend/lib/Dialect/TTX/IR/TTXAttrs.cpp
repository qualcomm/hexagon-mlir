//===-- TTXAttrs.cpp - TTX dialect attributes -----------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// This file implements the TTX dialect attributes.
//===----------------------------------------------------------------------===//
#include "hexagon/Dialect/TTX/IR/TTXDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::ttx;

/// Dialect creation, the instance will be owned by the context. This is the
/// point of registration of custom attributes for the dialect.
void TTXDialect::registerAttributes() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "hexagon/Dialect/TTX/IR/TTXAttrs.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// ODS-Generated Declarations
//===----------------------------------------------------------------------===//

#include "hexagon/Dialect/TTX/IR/TTXEnums.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "hexagon/Dialect/TTX/IR/TTXAttrs.cpp.inc"
