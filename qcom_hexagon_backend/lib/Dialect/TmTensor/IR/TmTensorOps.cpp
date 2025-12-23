//===- TmTensorOps.cpp - TmTensor dialect operations implementation ------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

#include "hexagon/Dialect/TmTensor/IR/TmTensorDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"

using namespace mlir;
using namespace mlir::tm_tensor;

/// Dialect creation, the instance will be owned by the context. This is the
/// point of registration of custom operations for the dialect.
void TmTensorDialect::registerOperations() {
  addOperations<
#define GET_OP_LIST
#include "hexagon/Dialect/TmTensor/IR/TmTensorOps.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// AttentionOp
//===----------------------------------------------------------------------===//

::mlir::LogicalResult AttentionOp::verify() {
  auto queryType = llvm::dyn_cast<RankedTensorType>(getQuery().getType());
  auto keyType = llvm::dyn_cast<RankedTensorType>(getKey().getType());
  auto valueType = llvm::dyn_cast<RankedTensorType>(getValue().getType());
  auto maskType = llvm::dyn_cast<RankedTensorType>(getMask().getType());

  if (!queryType || !keyType || !valueType || !maskType) {
    return emitOpError("query, key, value and mask must be ranked tensors");
  }

  // All tensors must be 3D and have same element type
  if (queryType.getRank() != 3 || keyType.getRank() != 3 ||
      valueType.getRank() != 3 || maskType.getRank() != 3)
    return emitOpError("all tensors must be 3D");

  auto elType = queryType.getElementType();
  if (keyType.getElementType() != elType ||
      valueType.getElementType() != elType ||
      maskType.getElementType() != elType)
    return emitOpError("all tensors must have same element type");

  // Shape constraints:
  //  Q[batch,seq_q,head_dim]
  //  K[batch,seq_kv,head_dim]
  //  V[batch,seq_kv,head_dim]
  //  M[batch,seq_q,seq_kv]
  auto queryShape = queryType.getShape();
  auto keyShape = keyType.getShape();
  auto valueShape = valueType.getShape();
  auto maskShape = maskType.getShape();

  // independent dims
  int64_t batch = queryShape[0];
  int64_t seq_q = queryShape[1];
  int64_t head_dim = queryShape[2];
  int64_t seq_kv = keyShape[1];

  if (ShapedType::isDynamic(batch) || ShapedType::isDynamic(seq_q) ||
      ShapedType::isDynamic(head_dim) || ShapedType::isDynamic(seq_kv))
    return emitOpError("dynamic dimensions not supported");

  // dependendent dims
  if (keyShape[0] != batch || keyShape[2] != head_dim ||
      valueShape[0] != batch || valueShape[1] != seq_kv ||
      valueShape[2] != head_dim || maskShape[0] != batch ||
      maskShape[1] != seq_q || maskShape[2] != seq_kv)
    return emitOpError("tensor shapes don't satisfy attention constraints");

  return success();
}

//===----------------------------------------------------------------------===//
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "hexagon/Dialect/TmTensor/IR/TmTensorOps.cpp.inc"
