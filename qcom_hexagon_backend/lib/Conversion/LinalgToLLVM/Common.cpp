//===- Common.cpp - some useful common types and functions ----------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// This file implements some common types and functions used across
// the Hexagon backend conversion passes.
//===----------------------------------------------------------------------===//

#include "hexagon/Conversion/LinalgToLLVM/Common.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "llvm/ADT/TypeSwitch.h"

namespace mlir {
namespace hexagon {

std::optional<unsigned> computeSmallestOperandTypeSize(Operation *op) {
  // Returns the size of the smallest input operand's type.
  // Choosing the smallest-sized elements results in the largest tile size,
  // which aligns better with Hexagon. We prefer breaking large vectors
  // into multiple vectors over dealing with partial vectors.
  unsigned int minElemSize = maxElemSizeInByte + 1;
  for (OpOperand &opOperand : op->getOpOperands()) {
    Type operandType = opOperand.get().getType();
    if (auto type = dyn_cast<RankedTensorType>(operandType)) {
      // Skip i1 tensors: they represent boolean predicates, not the actual
      // data width being vectorized. Including them would yield a 1-byte
      // minimum and inflate dataTileSize to 128, preventing vectorization of
      // f32/i32 ops whose bodies contain boolean sub-expressions.
      if (type.getElementType().isInteger(1))
        continue;
      auto operandSize = getElementSizeInBytes(type);
      if (operandSize) {
        minElemSize = std::min(minElemSize, *operandSize);
      }
    }
  }

  // For genericOp determine by inspecting body ops.
  if (auto genericOp = dyn_cast<linalg::GenericOp>(op)) {
    for (Operation &op : genericOp.getBody()->getOperations()) {
      if (op.hasTrait<OpTrait::Vectorizable>()) {
        for (auto res : op.getResults()) {
          // Skip i1 results: these are boolean predicates (arith.cmpf,
          // arith.cmpi, arith.andi, arith.ori) that are intermediate values
          // in the computation, not the actual data element size.
          if (getElementType(res.getType()).isInteger(1))
            continue;
          auto resSize = getElementSizeInBytes(res.getType());
          if (resSize)
            minElemSize = std::min(minElemSize, *resSize);
        }
      }
    }
  }

  if (minElemSize < maxElemSizeInByte + 1) {
    return minElemSize;
  }
  return std::nullopt;
}

Type getElementType(Type type) {
  return llvm::TypeSwitch<Type, Type>(type)
      .Case<VectorType, RankedTensorType>(
          [](auto ty) { return ty.getElementType(); })
      .Default([](Type ty) -> Type { return ty; });
}

std::optional<unsigned> getElementSizeInBytes(Type type) {
  Type elemType = getElementType(type);

  if (elemType.isInteger(1) || elemType.isInteger(8) ||
      isa<Float8E4M3FNType, Float8E5M2Type, Float8E4M3FNUZType,
          Float8E5M2FNUZType>(elemType))
    return 1;
  if (elemType.isF16() || elemType.isBF16() || elemType.isInteger(16))
    return 2;
  if (elemType.isF32() || elemType.isInteger(32))
    return 4;
  if (elemType.isF64() || elemType.isInteger(64))
    return 8;
  return std::nullopt;
}

std::optional<unsigned> computeDataTileSize(Operation *op) {
  auto elSize = computeSmallestOperandTypeSize(op);
  if (!elSize)
    return std::nullopt;
  return nativeVectorWidthInBytes / *elSize;
}

bool isPerfectlyTileable(unsigned LoopRange, unsigned dataTileSize) {
  return (LoopRange % dataTileSize == 0);
}

bool perfectlyVectorizable(unsigned dataTileSize, unsigned innerLoopRange) {
  return dataTileSize == innerLoopRange;
}

bool isVectorizableLoopDim(unsigned innerLoopDim, unsigned dataTileSize) {
  // An inner-loop is vectorizable if either of these conditions holds:
  //   1. Its dimension is one HVX vector or smaller:
  //      0 < innerLoopDim <= dataTileSize
  //      (e.g. 32 f16 = 64 bytes <= 128-byte HVX register).
  //      It gets vectorized at its natural width (no masks); and the
  //      LLVM backend is capable of taking care of it even if it is smaller
  //      than the HVX length.
  //   2. Exactly an HVX pair: innerLoopDim == 2 * dataTileSize.
  // Note: The case of dataTileSize < innerLoopDim < 2 * dataTileSize will be
  // later taken care of by splitting it into a head (utilizing full vectors),
  // and a tail (vectorized or scalarized).
  return innerLoopDim != 0 &&
         (innerLoopDim <= dataTileSize || innerLoopDim == 2 * dataTileSize);
}

void setTargetTriple(ModuleOp moduleOp) {
  std::string targetTripleStr = "hexagon";
  moduleOp->setAttr(LLVM::LLVMDialect::getTargetTripleAttrName(),
                    StringAttr::get(moduleOp->getContext(), targetTripleStr));
}

void setDataLayout(ModuleOp moduleOp) {
  std::string dataLayoutStr =
      "e-m:e-p:32:32:32-a:0-n16:32-i64:64:64-i32:32:32"
      "-i16:16:16-i1"
      ":8:8-f32:32:32-f64:64:64-v32:32:32-v64:64:64-v512"
      ":512:512-v1024:1024:1024-v2048:2048:2048";

  // Commented out till index i64/i32 is sorted.
  // moduleOp->setAttr(LLVM::LLVMDialect::getDataLayoutAttrName(),
  //                   StringAttr::get(moduleOp->getContext(), dataLayoutStr));
}

bool doesFuncReturnValue(ModuleOp moduleOp) {
  return moduleOp
      .walk([&](func::FuncOp op) {
        if (!op.getResultTypes().empty())
          return WalkResult::interrupt();
        return WalkResult::advance();
      })
      .wasInterrupted();
}

} // namespace hexagon

} // namespace mlir
