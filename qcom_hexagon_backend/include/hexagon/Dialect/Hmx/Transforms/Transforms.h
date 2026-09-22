//===- Transforms.h - HMX dialect transform passes -------------*- C++ -*-===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

#ifndef HEXAGON_DIALECT_HMX_TRANSFORMS_TRANSFORMS_H
#define HEXAGON_DIALECT_HMX_TRANSFORMS_TRANSFORMS_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/FunctionImplementation.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace hmx {

#define GEN_PASS_DECL
#include "hexagon/Dialect/Hmx/Transforms/Passes.h.inc"

std::unique_ptr<InterfacePass<FunctionOpInterface>> createMatmulToHmxPass(
    const MatmulToHmxOptions &options = MatmulToHmxOptions());

std::unique_ptr<InterfacePass<FunctionOpInterface>> createHmxPartitionPass(
    const HmxPartitionOptions &options = HmxPartitionOptions());

std::unique_ptr<InterfacePass<FunctionOpInterface>> createWeightResidentPass(
    const WeightResidentOptions &options = WeightResidentOptions());

std::unique_ptr<InterfacePass<FunctionOpInterface>>
createHmxWorkspaceResidentPass();

} // namespace hmx
} // namespace mlir

#endif // HEXAGON_DIALECT_HMX_TRANSFORMS_TRANSFORMS_H
