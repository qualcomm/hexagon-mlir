//===-- CroutonDialect.cpp - Crouton dialect definition -------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// This file implements the Crouton dialect registration.
//===----------------------------------------------------------------------===//

#include "hexagon/Dialect/Crouton/IR/CroutonDialect.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"

using namespace mlir;
using namespace mlir::crouton;

/// Dialect creation, the instance will be owned by the context. This is the
/// point of registration of custom attrs, types and operations for the dialect.
void CroutonDialect::initialize() { registerTypes(); }

//===----------------------------------------------------------------------===//
// ODS-Generated Declarations
//===----------------------------------------------------------------------===//

#include "hexagon/Dialect/Crouton/IR/CroutonDialect.cpp.inc"
