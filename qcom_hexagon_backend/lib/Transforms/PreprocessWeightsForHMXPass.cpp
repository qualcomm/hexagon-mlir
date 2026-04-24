//===-- PreprocessWeightsForHMXPass.cpp - Transform weights to WH layout --===//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
#include "hexagon/Dialect/HexKL/IR/HexKLDialect.h"
#include "hexagon/Transforms/Transforms.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectResourceBlobManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>

#define DEBUG_TYPE "preprocess-weights-hmx"

using namespace mlir;

namespace {

/**
 * @brief Block linearize a matrix
 *
 * Reorganizes matrix from 2D (n_row × n_col) into 4D block structure:
 * (n_row/n_block_row) × (n_col/n_block_col) × n_block_row × n_block_col
 *
 * @param n_row Number of rows
 * @param n_col Number of columns
 * @param n_block_row Block height
 * @param n_block_col Block width
 * @param M Matrix data (modified in-place)
 */
static void matrix_block_linearize_f16(size_t n_row, size_t n_col,
                                       size_t n_block_row, size_t n_block_col,
                                       uint16_t *M) {

  // Validate inputs
  if (n_col % n_block_col != 0) {
    return;
  }

  if (n_row % n_block_row != 0) {
    return;
  }

  size_t total_elements = n_row * n_col;
  std::vector<uint16_t> T(total_elements);

  size_t n_col_blocks = n_col / n_block_col;

  for (size_t i = 0; i < n_row; i++) {
    for (size_t j = 0; j < n_col_blocks; j++) {
      for (size_t k = 0; k < n_block_col; k++) {

        size_t old_index =
            i * (n_col_blocks * n_block_col) + j * n_block_col + k;

        size_t new_row = i / n_block_row;
        size_t new_block = i % n_block_row;
        size_t new_index =
            new_row * (n_col_blocks * n_block_row * n_block_col) +
            j * (n_block_row * n_block_col) + new_block * n_block_col + k;

        T[new_index] = M[old_index];
      }
    }
  }

  std::memcpy(M, T.data(), total_elements * sizeof(uint16_t));
}

/**
 * @brief Transform weights from row-major to WH (Weight-HMX) layout
 *
 * This is the complete transformation pipeline for weights:
 * 1. Block linearize with 2×1 blocks
 * 2. Block linearize with (N/2)×64 blocks
 *
 * @param n_row Original number of rows (K)
 * @param n_col Original number of columns (N)
 * @param W Weight matrix data
 * @return 0 on success, -1 on failure
 */
static int transform_rm_to_wh_f16(size_t n_row, size_t n_col, uint16_t *W) {

  // Validate dimensions
  if (n_row % 2 != 0 || n_col % 2 != 0) {
    return -1;
  }

  size_t current_rows = n_row;
  size_t current_cols = n_col;

  // Step 1: Block linearize with 2×1 blocks
  matrix_block_linearize_f16(current_rows, current_cols, 2, 1, W);

  // Step 2: Block linearize with (N/2)×64 blocks
  if (current_cols >= 64 && current_rows >= 2) {
    size_t step3_rows = current_rows / 2;
    size_t step3_cols = current_cols * 2;
    size_t block_rows = current_rows / 2;
    size_t block_cols = 64;

    if (step3_cols >= block_cols && step3_rows >= block_rows) {
      matrix_block_linearize_f16(step3_rows, step3_cols, block_rows, block_cols,
                                 W);
    }
  }

  return 0;
}

// Helper to check if a value is used as RHS (weight) in a matmul
bool isMatmulWeight(Value value) {
  for (Operation *user : value.getUsers()) {
    if (auto matmulOp = dyn_cast<hexkl::MatmulOp>(user)) {
      if (matmulOp.getRhs() == value) {
        return true;
      }
    }
  }
  return false;
}

struct PreprocessWeightsForHMX : public OpRewritePattern<arith::ConstantOp> {
  PreprocessWeightsForHMX(MLIRContext *ctx) : OpRewritePattern(ctx) {}

  LogicalResult matchAndRewrite(arith::ConstantOp op,
                                PatternRewriter &rewriter) const override {

    auto resourceAttr = dyn_cast<DenseResourceElementsAttr>(op.getValue());
    if (!resourceAttr)
      return failure();

    std::string handleName = resourceAttr.getRawHandle().getKey().str();
    if (handleName.find("_wh_layout") != std::string::npos) {
      return failure();
    }

    auto type = cast<RankedTensorType>(resourceAttr.getType());
    if (type.getRank() != 2 || !type.getElementType().isF16())
      return failure();

    if (!isMatmulWeight(op.getResult()))
      return failure();

    int64_t K = type.getDimSize(0);
    int64_t N = type.getDimSize(1);

    if ((K % 2) != 0 || (N % 2) != 0) {
      return failure();
    }

    auto blob = resourceAttr.getRawHandle().getBlob();
    if (!blob)
      return failure();

    llvm::ArrayRef<char> rawData = blob->getData();
    if (rawData.size() != type.getNumElements() * 2)
      return failure();

    // Copy data to mutable buffer
    std::vector<uint16_t> transformedData(type.getNumElements());
    const uint16_t *src = reinterpret_cast<const uint16_t *>(rawData.data());
    std::memcpy(transformedData.data(), src,
                type.getNumElements() * sizeof(uint16_t));

    // Apply WH transform
    int ret = transform_rm_to_wh_f16(K, N, transformedData.data());
    if (ret != 0) {
      return rewriter.notifyMatchFailure(op, "weight transformation failed");
    }

    // Create new blob with 128-byte alignment
    size_t byteSize = transformedData.size() * sizeof(uint16_t);
    void *alignedPtr = nullptr;
    if (posix_memalign(&alignedPtr, 128, byteSize) != 0) {
      return failure();
    }
    char *buffer = static_cast<char *>(alignedPtr);
    std::memcpy(buffer, transformedData.data(), byteSize);

    auto deleter = [](void *ptr, size_t, size_t) { std::free(ptr); };
    auto newBlob = AsmResourceBlob(llvm::ArrayRef<char>(buffer, byteSize), 128,
                                   deleter, false);

    std::string newHandleName = handleName + "_wh_layout";
    auto newResourceAttr =
        DenseResourceElementsAttr::get(type, newHandleName, std::move(newBlob));

    rewriter.replaceOpWithNewOp<arith::ConstantOp>(op, newResourceAttr);

    return success();
  }
};

struct PreprocessWeightsForHMXPass
    : public PassWrapper<PreprocessWeightsForHMXPass,
                         InterfacePass<FunctionOpInterface>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PreprocessWeightsForHMXPass)

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, hexkl::HexKLDialect>();
  }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<PreprocessWeightsForHMX>(&getContext());

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }

  StringRef getArgument() const final { return "preprocess-weights-hmx"; }

  StringRef getDescription() const final {
    return "Transform constant weights to HexKL macro API WH layout at compile "
           "time";
  }
};

} // namespace

namespace mlir {
namespace hexagon {

std::unique_ptr<InterfacePass<FunctionOpInterface>>
createPreprocessWeightsForHMXPass() {
  return std::make_unique<PreprocessWeightsForHMXPass>();
}

} // namespace hexagon
} // namespace mlir
