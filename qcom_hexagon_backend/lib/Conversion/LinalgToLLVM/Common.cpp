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
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "llvm/ADT/TypeSwitch.h"

namespace mlir {
namespace hexagon {

// Does the generic body carry an exp-family transcendental?
//
// This gates the i1 handling below, and it is deliberately narrow. A tensor of
// i1 is a predicate (mask), not data, so its 1-byte storage must not drive the
// vector width: counting it makes computeDataTileSize return 128 for an f32
// loop of 64, and perfectlyVectorizable then demands `tile == innerLoop`
// (128 != 64) and blocks vectorization of the entire chain. When that chain is
// a masked feature map -- compare + select around a `math.exp` -- the exp never
// reaches a native-width vector and hexagon-clang later scalarizes it into
// per-element `call expf` plus soft f16<->f32 conversions (`llvm.exp` on
// <32 x float> has no Hexagon lowering; only `llvm.exp2` does).
//
// Other i1-consuming chains (e.g. the causal mask / sum chains of chunked
// linear attention) are *not* stalled this way -- they vectorize fine once the
// surrounding tiling is left alone -- so opening the door for them changes
// their shape for no gain. Gate on the exp family to fix exactly the chain the
// mechanism actually blocks. See hexagon-mlir-local.patch.
static bool bodyHasExpFamilyOp(Operation *op) {
  auto genericOp = dyn_cast_or_null<linalg::GenericOp>(op);
  if (!genericOp)
    return false;
  return llvm::any_of(genericOp.getBody()->getOperations(), [](Operation &op) {
    return isa<math::ExpOp, math::Exp2Op, math::ExpM1Op>(&op);
  });
}

std::optional<unsigned> computeSmallestOperandTypeSize(Operation *op) {
  // Returns the size of the smallest input operand's type.
  // Choosing the smallest-sized elements results in the largest tile size,
  // which aligns better with Hexagon. We prefer breaking large vectors
  // into multiple vectors over dealing with partial vectors.
  //
  // i1 (mask) operands/results are skipped, but only for the exp-family
  // generics described above; every other op keeps the original behaviour.
  const bool skipMaskElements = bodyHasExpFamilyOp(op);

  unsigned int minElemSize = maxElemSizeInByte + 1;
  for (OpOperand &opOperand : op->getOpOperands()) {
    Type operandType = opOperand.get().getType();
    if (auto type = dyn_cast<RankedTensorType>(operandType)) {
      if (skipMaskElements && type.getElementType().isInteger(1))
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
          if (skipMaskElements &&
              getElementType(res.getType()).isInteger(1))
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
  if (elemType.isInteger(1) || elemType.isInteger(8))
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
  if (LoopRange % dataTileSize != 0) {
    return false;
  }
  return true;
}

bool perfectlyVectorizable(unsigned dataTileSize, unsigned innerLoopRange) {
  return dataTileSize == innerLoopRange;
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
