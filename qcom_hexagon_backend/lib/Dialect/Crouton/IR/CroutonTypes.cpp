//===-- CroutonTypes.cpp - Crouton dialect types --------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// This file implements the Crouton dialect types.
//===----------------------------------------------------------------------===//

#include "hexagon/Dialect/Crouton/IR/CroutonDialect.h"
#include "mlir/IR/DialectImplementation.h" // required by `CroutonTypes.cpp.inc`
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/TypeSwitch.h" // required by `CroutonTypes.cpp.inc`

using namespace mlir;
using namespace mlir::crouton;

/// Dialect creation, the instance will be owned by the context. This is the
/// point of registration of custom types for the dialect.
void CroutonDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "hexagon/Dialect/Crouton/IR/CroutonTypes.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// ODS-Generated Declarations
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "hexagon/Dialect/Crouton/IR/CroutonTypes.cpp.inc"

static constexpr llvm::StringRef kVtcm = "vtcm";

// Parse CroutonType of the form 'crouton<64x32xi8, vtcm>' or 'crouton<64xi8>'
Type CroutonType::parse(AsmParser &parser) {
  SmallVector<int64_t> dimensions;
  Type elementType;

  if (parser.parseLess() ||
      parser.parseDimensionList(dimensions, /*allowDynamic=*/false) ||
      parser.parseType(elementType))
    return Type();

  bool vtcm = false;
  if (succeeded(parser.parseOptionalComma())) {
    if (parser.parseKeyword(kVtcm))
      return Type();
    vtcm = true;
  }

  if (parser.parseGreater())
    return Type();

  return CroutonType::get(dimensions, elementType, vtcm);
}

void CroutonType::print(AsmPrinter &printer) const {
  printer << "<";
  for (auto dim : getShape())
    printer << dim << "x";
  printer << getElementType();
  if (getVtcm().getValue())
    printer << ", " << kVtcm;
  printer << ">";
}
