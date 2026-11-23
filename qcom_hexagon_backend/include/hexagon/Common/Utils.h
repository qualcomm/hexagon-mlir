// ===- Utils.h -------------------------------------------------------------===
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
// ===-----------------------------------------------------------------------===

#ifndef HEXAGON_COMMON_UTILS_H
#define HEXAGON_COMMON_UTILS_H

#include <cassert>

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"

namespace mlir {
namespace {

static inline Location CreateNameLoc(StringRef name, Location loc) {
#ifndef NDEBUG
  return NameLoc::get(StringAttr::get(loc.getContext(), name), loc);
#else
  return loc;
#endif
}

static inline Value SetNameLoc(Value value, StringRef name) {
#ifndef NDEBUG
  value.setLoc(CreateNameLoc(name, value.getLoc()));
#endif
  return value;
}

static inline Operation *SetNameLoc(Operation *op, StringRef name) {
#ifndef NDEBUG
  assert(op->getNumResults() == 1 && "Operation must have a single result");
  SetNameLoc(op->getResult(0), name);
#endif
  return op;
}

} // end anonymous namespace

static inline Location operator<<(Location loc, StringRef name) {
  return CreateNameLoc(name, loc);
}

static inline Value operator<<(Value value, StringRef name) {
  return SetNameLoc(value, name);
}

static inline Operation *operator<<(Operation *op, StringRef name) {
  return SetNameLoc(op, name);
}

} // namespace mlir

#endif /* HEXAGON_COMMON_UTILS_H */
