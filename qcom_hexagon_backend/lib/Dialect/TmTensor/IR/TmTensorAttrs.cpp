//===- TmTensorAttrs.cpp - TmTensor dialect attributes implementation ----===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

#include "hexagon/Dialect/TmTensor/IR/TmTensorDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::tm_tensor;

//===----------------------------------------------------------------------===//
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

#define GET_ATTRDEF_CLASSES
#include "hexagon/Dialect/TmTensor/IR/TmTensorAttrs.cpp.inc"

void TmTensorDialect::registerAttributes() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "hexagon/Dialect/TmTensor/IR/TmTensorAttrs.cpp.inc"
      >();
}
