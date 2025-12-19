//===- MLLVMIRTranslation.h -----------------------------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

#ifndef HEXAGON_TARGET_LINALG_MLLVMIRTRANSLATION_H
#define HEXAGON_TARGET_LINALG_MLLVMIRTRANSLATION_H

#include "hexagon/Conversion/LinalgToLLVM/Passes.h"

#include <string>
#include <unordered_map>
#include <vector>

namespace mlir {
class ModuleOp;
} // namespace mlir

void setLinalgToLLVMOptions(
    mlir::hexagon::LinalgToLLVMOptions &options,
    const std::unordered_map<std::string, std::string> &arch_kwargs);

namespace mlir {

namespace hexagon {

mlir::ModuleOp translateLinalgToLLVMMLIR(
    mlir::ModuleOp module,
    const std::unordered_map<std::string, std::string> &arch_kwargs);

std::vector<ModuleOp> translateLinalgToMultipleLLVMMLIRModules(
    ModuleOp module,
    const std::unordered_map<std::string, std::string> &arch_kwargs);

} // namespace hexagon
} // namespace mlir

#endif // HEXAGON_TARGET_LINALG_MLLVMIRTRANSLATION_H
