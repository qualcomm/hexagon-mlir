//===- Passes.h - Linalg lowering and optimization passes -----------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
#ifndef HEXAGON_TRANSFORMS_PASSES_H
#define HEXAGON_TRANSFORMS_PASSES_H

#include "Transforms.h"

namespace mlir {
namespace hexagon {

std::unique_ptr<mlir::Pass> createVectorAccumulatorForReductionPass();
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createOutlineLinalgGenericPass();
std::unique_ptr<mlir::Pass>
createHexagonDeallocAfterLastUsePass(int64_t chunkBytes = 0);

#define GEN_PASS_REGISTRATION
#include "hexagon/Transforms/Passes.h.inc"

} // namespace hexagon
} // namespace mlir
#endif // HEXAGON_TRANSFORMS_PASSES_H
