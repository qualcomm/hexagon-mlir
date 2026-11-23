//===- Passes.h - Convert Shard to LLVM ops ---------------------*- C++ -*-===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

#ifndef HEXAGON_CONVERSION_SHARDTOLLVM_PASSES_H
#define HEXAGON_CONVERSION_SHARDTOLLVM_PASSES_H

#include "mlir/Dialect/Shard/IR/ShardDialect.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace hexagon {

/// Create the ShardToLLVM conversion pass.
std::unique_ptr<Pass> createShardToLLVMPass();

/// Register ShardToLLVM passes.
#define GEN_PASS_REGISTRATION
#include "hexagon/Conversion/ShardToLLVM/Passes.h.inc"

} // namespace hexagon
} // namespace mlir

#endif // HEXAGON_CONVERSION_SHARDTOLLVM_PASSES_H
