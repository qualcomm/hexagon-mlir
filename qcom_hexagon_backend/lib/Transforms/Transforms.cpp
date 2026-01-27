//===- Transforms.cpp - TableGen'd Transforms Passes                 ------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// This file defines the registration of all passes in the Transforms
// directory.
//
//===----------------------------------------------------------------------===//

#include "hexagon/Transforms/Transforms.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "llvm/ADT/TypeSwitch.h"

#include <numeric>

using namespace mlir;
using namespace hexagon;

#define GEN_PASS_CLASSES
#include "hexagon/Transforms/Passes.h.inc"
