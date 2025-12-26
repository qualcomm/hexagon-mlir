//===-- HexagonTPtrDialect.cpp - Hexagon TPtr dialect ---------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// This file implements the Hexagon TPtr dialect registration.
//===----------------------------------------------------------------------===//

#include "mlir/IR/Builders.h"

#include "hexagon/Dialect/HexagonTPtr/IR/HexagonTPtrDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Ptr/IR/PtrDialect.h"
#include "mlir/Dialect/Ptr/IR/PtrTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::tptr;

namespace {
ParseResult parseIntType(OpAsmParser &parser, Type &ty) {
  if (succeeded(parser.parseOptionalColon()) && parser.parseType(ty))
    return parser.emitError(parser.getNameLoc(), "expected a type");
  if (!ty)
    ty = parser.getBuilder().getIndexType();
  return success();
}
void printIntType(OpAsmPrinter &p, Operation *op, Type ty) {
  if (!ty.isIndex())
    p << " : " << ty;
}
} // namespace

//===----------------------------------------------------------------------===//
// Dialect
//===----------------------------------------------------------------------===//
void mlir::tptr::HexagonTPtrDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "hexagon/Dialect/HexagonTPtr/IR/HexagonTPtrTypes.cpp.inc"
      >();
}

/// Dialect creation, the instance will be owned by the context. This is the
/// point of registration of custom types and operations for the dialect.
void mlir::tptr::HexagonTPtrDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "hexagon/Dialect/HexagonTPtr/IR/HexagonTPtrOps.cpp.inc"
      >();
  registerTypes();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "hexagon/Dialect/HexagonTPtr/IR/HexagonTPtrAttributes.cpp.inc"
      >();
}

bool tptr::HexagonDefaultMemorySpaceAttr::isValidLoad(
    Type type, mlir::ptr::AtomicOrdering ordering, IntegerAttr alignment,
    llvm::function_ref<InFlightDiagnostic()> emitError) const {
  return true;
}

bool tptr::HexagonDefaultMemorySpaceAttr::isValidStore(
    Type type, mlir::ptr::AtomicOrdering ordering, IntegerAttr alignment,
    llvm::function_ref<InFlightDiagnostic()> emitError) const {
  return true;
}

bool tptr::HexagonDefaultMemorySpaceAttr::isValidAtomicOp(
    mlir::ptr::AtomicBinOp binOp, Type type, mlir::ptr::AtomicOrdering ordering,
    IntegerAttr alignment,
    llvm::function_ref<InFlightDiagnostic()> emitError) const {
  return true;
}

bool tptr::HexagonDefaultMemorySpaceAttr::isValidAtomicXchg(
    Type type, mlir::ptr::AtomicOrdering successOrdering,
    mlir::ptr::AtomicOrdering failureOrdering, IntegerAttr alignment,
    llvm::function_ref<InFlightDiagnostic()> emitError) const {
  return true;
}

bool tptr::HexagonDefaultMemorySpaceAttr::isValidAddrSpaceCast(
    Type tgt, Type src,
    llvm::function_ref<InFlightDiagnostic()> emitError) const {
  return true;
}

bool tptr::HexagonDefaultMemorySpaceAttr::isValidPtrIntCast(
    Type intLikeTy, Type ptrLikeTy,
    llvm::function_ref<InFlightDiagnostic()> emitError) const {
  return true;
}
///===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "hexagon/Dialect/HexagonTPtr/IR/HexagonTPtrOps.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "hexagon/Dialect/HexagonTPtr/IR/HexagonTPtrAttributes.cpp.inc"

#include "hexagon/Dialect/HexagonTPtr/IR/HexagonTPtrDialect.cpp.inc"
