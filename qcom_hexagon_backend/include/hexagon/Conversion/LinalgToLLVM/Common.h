//===- Common.h   - some useful common types and functions ----------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

#ifndef HEXAGON_CONVERSION_HEXAGON_LINALG_TO_LLVM_COMMON_H
#define HEXAGON_CONVERSION_HEXAGON_LINALG_TO_LLVM_COMMON_H
#include "hexagon/Common/Common.h"
#include "mlir/IR/Operation.h"
#include <optional>

namespace mlir {
class TensorType;
class ModuleOp;

namespace hexagon {

inline constexpr unsigned nativeVectorWidthInBytes = 128;
inline constexpr unsigned maxElemSizeInByte = 8;

/// The size of the vtcm memory size in bytes.
inline constexpr unsigned vtcmSizeInBytes = 2 * 1024 * 1024; // 2MB

/// Based on op-operands determines the element type size.
/// If it can't determine returns nullopt.
std::optional<unsigned> computeSmallestOperandTypeSize(Operation *);

/// Returns element type of higher dimensional types.
Type getElementType(Type);

/// Returns element type size in bytes. nullopt if
/// type is not handled currently.
std::optional<unsigned> getElementSizeInBytes(Type);

/// Based on target hardware vector size and operand elemen-type
/// for this op, returns a suitable number of elements in vec.
std::optional<unsigned> computeDataTileSize(Operation *);

// Check whether the loop range is a multiple of tile size
bool isPerfectlyTileable(unsigned LoopRange, unsigned dataTileSize);

/// Checks if inner loop dimension and vector size match
/// for neat vectorization. This is quite a complex decision
/// and currently it decides it with equality.
bool perfectlyVectorizable(unsigned dataTileSize, unsigned innerLoopRange);

/// Hexagon target triple and datalayout
void setTargetTriple(ModuleOp);
void setDataLayout(ModuleOp);

// To check if there functions which return values.
bool doesFuncReturnValue(ModuleOp);

} // namespace hexagon
} // namespace mlir
#endif // HEXAGON_CONVERSION_HEXAGON_LINALG_TO_LLVM_COMMON_H
