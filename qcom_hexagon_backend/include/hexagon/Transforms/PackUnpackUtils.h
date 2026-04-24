//===- PackUnpackUtils.h - Utils for generating pack/unpack ops------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
#ifndef HEXAGON_TRANSFORMS_PACKUNPACKUTILS_H
#define HEXAGON_TRANSFORMS_PACKUNPACKUTILS_H

#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/BuiltinTypes.h"

// Generalized candidate type check for any Conv2D op and element type
template <typename Conv2DOpT>
bool isCandidateType(Conv2DOpT &conv2DOp, mlir::Type expectedType);

// Utility to check if a RankedTensorType matches the expected element type and
// shape
bool hasValidTensorType(mlir::Value value, mlir::Type expectedType,
                        int expectedRank = 4);

// Utility to check if strides and dilations are unit (i.e., [1, 1])
template <typename Conv2DOpT>
bool hasUnitStrideAndDilation(Conv2DOpT &conv2DOp);

// Checks if Conv2DNhwcFhwcOp is a candidate for wrapping with layout
// conversion ops.
bool isCandidate16BitElements(mlir::linalg::Conv2DNhwcFhwcOp &conv2DOp);

// Checks if Conv2DNhwcHwcfQOp is a candidate for wrapping with layout
// conversion ops.
bool isCandidate8BitElements(mlir::linalg::Conv2DNhwcHwcfQOp &conv2DOp);

// Layout conversions for 16-bit image elements require two pack operations
// The first one effects this mapping: (n, h, w, c) -> (n, h/8, w/4, c/32, 8, 4,
// 32) This function returns the tensor type of the first pack.
mlir::RankedTensorType
getTempPacked16BitElementType(const mlir::RankedTensorType &origType);

// This function returns the tensor type after layout conversion for 16-bit
// elements For an (n, h, w, c) tensor type it returns (n, h/8, w/4, c/32, 8, 2,
// 32, 2)
mlir::RankedTensorType
getPacked16BitElementType(const mlir::RankedTensorType &origType);

// This function returns the filter tensor type after layout conversion for
// 16-bit elements For an (f, h, w, c) filter it returns (f/32, c/32, h, w, 16,
// 32, 2)
mlir::RankedTensorType
getPackedFilter16BitElementType(const mlir::RankedTensorType &origType);

// This function returns the tensor type after layout conversion for 8-bit
// elements For an (n, h, w, c) tensor type it returns (n, h/8, w/8, c/32, 8, 8,
// 32)
mlir::RankedTensorType
getPacked8BitElementType(const mlir::RankedTensorType &origType);

int ceildiv(int dividend, int divisor);

// This helper function generates code for layout conversions for an i8
// tensor.
mlir::linalg::PackOp
buildI8PackOps(mlir::RewriterBase &rewriter, mlir::Value image,
               mlir::RankedTensorType &imageRTType, mlir::Location &loc,
               mlir::Value padValue, llvm::SmallVector<int64_t> &innerDimsPos,
               llvm::SmallVector<mlir::OpFoldResult> &innerTiles);

// Helper function for unpacking the output of a I8 op.
mlir::linalg::UnPackOp
unpackI8Output(mlir::RewriterBase &rewriter, mlir::Value conv2DOutput,
               mlir::Location &loc, llvm::SmallVector<int64_t> &innerDimsPos,
               llvm::SmallVector<mlir::OpFoldResult> &innerTiles,
               mlir::RankedTensorType &outRTTy,
               llvm::ArrayRef<int64_t> &outShape);

// This helper function generates code for layout conversions for an FP16
// tensor.
mlir::linalg::PackOp
buildF16PackOps(mlir::RewriterBase &rewriter, mlir::Value image,
                mlir::RankedTensorType &imageRTType, mlir::Location &loc,
                mlir::Value padValue,
                llvm::SmallVector<int64_t> &innerDimsPosTemp,
                llvm::SmallVector<mlir::OpFoldResult> &innerTilesTemp,
                llvm::SmallVector<int64_t> &innerDimsPos,
                llvm::SmallVector<mlir::OpFoldResult> &innerTiles);
// Helper function for unpacking the output of a packed FP16 op.
mlir::linalg::UnPackOp
unpackF16Output(mlir::RewriterBase &rewriter, mlir::Value firstUnpackInput,
                mlir::Location &loc, llvm::SmallVector<int64_t> &innerDimsPos,
                llvm::SmallVector<mlir::OpFoldResult> &innerTiles,
                llvm::SmallVector<int64_t> &innerDimsPosTemp,
                llvm::SmallVector<mlir::OpFoldResult> &innerTilesTemp,
                mlir::RankedTensorType &outRTTy,
                llvm::ArrayRef<int64_t> &outShape);

#endif
