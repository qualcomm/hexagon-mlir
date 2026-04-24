//===-- LowerHexKLMatmulToMacroPass.cpp - Lower to hexkl_macro_mm_f16 ----===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//===----------------------------------------------------------------------===//
//
// Lowers hexkl.matmul operations to hexkl_macro_mm_f16 library calls.
//
// This pass assumes:
// - Weights are constant and preprocessed to WH layout at compile-time
// - Activations are dynamic and need runtime transformation to AH layout
// - Output needs transformation back from AH to row-major layout
//
// - The pass can be enhanced later to work with layout propagation to eliminate
// - the layout transformations to and from row major everytime.
//
// Memory Safety:
// - Activations are copied to temporary buffers before in-place transformation
// - Weights are used directly (already in correct layout, read-only)
// - Output buffer is safe (allocated for this operation)
//
// HMX Resource Management:
// - initialize/lock_hmx called before first matmul in a loop or group
// - unlock_hmx/finalize called after last matmul in a loop or group
// - Consecutive matmuls share the same HMX lock (optimization)
//
//===----------------------------------------------------------------------===//

#include "hexagon/Dialect/HexKL/IR/HexKLDialect.h"
#include "hexagon/Transforms/Transforms.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace hexagon;

namespace {

struct LowerHexKLMatmulToMacro : public OpRewritePattern<hexkl::MatmulOp> {
  LowerHexKLMatmulToMacro(MLIRContext *ctx) : OpRewritePattern(ctx) {}

  LogicalResult matchAndRewrite(hexkl::MatmulOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value lhs = op.getLhs();   // Activation (M×K)
    Value rhs = op.getRhs();   // Weight (K×N) - preprocessed to WH layout
    Value outs = op.getOuts(); // Output (M×N)

    auto lhsType = cast<MemRefType>(lhs.getType());
    auto rhsType = cast<MemRefType>(rhs.getType());
    auto outType = cast<MemRefType>(outs.getType());

    // Validate element types (only FP16 supported)
    if (!lhsType.getElementType().isF16() ||
        !rhsType.getElementType().isF16() ||
        !outType.getElementType().isF16()) {
      return rewriter.notifyMatchFailure(op, "only f16 matmul supported");
    }

    // Extract matrix dimensions
    Value idx0 = arith::ConstantIndexOp::create(rewriter, loc, 0);
    Value idx1 = arith::ConstantIndexOp::create(rewriter, loc, 1);

    Value dimM = memref::DimOp::create(rewriter, loc, lhs, idx0);
    Value dimK = memref::DimOp::create(rewriter, loc, lhs, idx1);
    Value dimN = memref::DimOp::create(rewriter, loc, rhs, idx1);

    Type i32Ty = rewriter.getI32Type();
    Type i64Ty = rewriter.getI64Type();
    auto ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());

    // Convert dimensions to i32 (required by C API)
    Value MI32 = arith::IndexCastOp::create(rewriter, loc, i32Ty, dimM);
    Value KI32 = arith::IndexCastOp::create(rewriter, loc, i32Ty, dimK);
    Value NI32 = arith::IndexCastOp::create(rewriter, loc, i32Ty, dimN);

    // Calculate buffer sizes for cache operations
    Value two =
        arith::ConstantIndexOp::create(rewriter, loc, 2); // FP16 = 2 bytes

    // Activation buffer size (M × K × 2 bytes)
    Value actSize = arith::MulIOp::create(rewriter, loc, dimM, dimK);
    Value actSizeBytes = arith::MulIOp::create(rewriter, loc, actSize, two);
    Value actSizeBytesI32 =
        arith::IndexCastOp::create(rewriter, loc, i32Ty, actSizeBytes);

    // Weight buffer size (K × N × 2 bytes)
    Value dimKRhs = memref::DimOp::create(rewriter, loc, rhs, idx0);
    Value wgtSize = arith::MulIOp::create(rewriter, loc, dimKRhs, dimN);
    Value wgtSizeBytes = arith::MulIOp::create(rewriter, loc, wgtSize, two);
    Value wgtSizeBytesI32 =
        arith::IndexCastOp::create(rewriter, loc, i32Ty, wgtSizeBytes);

    // Output buffer size (M × N × 2 bytes)
    Value outSize = arith::MulIOp::create(rewriter, loc, dimM, dimN);
    Value outSizeBytes = arith::MulIOp::create(rewriter, loc, outSize, two);
    Value outSizeBytesI32 =
        arith::IndexCastOp::create(rewriter, loc, i32Ty, outSizeBytes);

    // ========================================================================
    // ACTIVATION BUFFER PREPARATION
    // ========================================================================
    // Allocate temporary buffer and copy activation data
    // This is necessary because rm_to_ah_f16_inplace modifies the buffer
    // in-place, and we must not corrupt the original input buffer
    auto actTempType =
        MemRefType::get(lhsType.getShape(), lhsType.getElementType());
    Value actTemp = memref::AllocOp::create(rewriter, loc, actTempType);
    memref::CopyOp::create(rewriter, loc, lhs, actTemp);

    // Extract pointer to activation buffer
    Value actPtr = memref::ExtractAlignedPointerAsIndexOp::create(
        rewriter, loc, rewriter.getIndexType(), actTemp);
    Value actPtrI32 = arith::IndexCastOp::create(rewriter, loc, i32Ty, actPtr);
    Value actPtrI64 = arith::IndexCastOp::create(rewriter, loc, i64Ty, actPtr);
    Value actPtrCast =
        LLVM::IntToPtrOp::create(rewriter, loc, ptrType, actPtrI64);

    // ========================================================================
    // WEIGHT BUFFER PREPARATION
    // ========================================================================
    // Weights are preprocessed at compile-time and stored in WH layout
    // We use them directly (read-only, no copy needed)
    Value rhsPtr = memref::ExtractAlignedPointerAsIndexOp::create(
        rewriter, loc, rewriter.getIndexType(), rhs);
    Value rhsPtrI32 = arith::IndexCastOp::create(rewriter, loc, i32Ty, rhsPtr);
    Value rhsPtrI64 = arith::IndexCastOp::create(rewriter, loc, i64Ty, rhsPtr);
    Value rhsPtrCast =
        LLVM::IntToPtrOp::create(rewriter, loc, ptrType, rhsPtrI64);

    // ========================================================================
    // OUTPUT BUFFER PREPARATION
    // ========================================================================
    // Output buffer is allocated for this operation, safe to use directly
    Value outPtr = memref::ExtractAlignedPointerAsIndexOp::create(
        rewriter, loc, rewriter.getIndexType(), outs);
    Value outPtrI32 = arith::IndexCastOp::create(rewriter, loc, i32Ty, outPtr);
    Value outPtrI64 = arith::IndexCastOp::create(rewriter, loc, i64Ty, outPtr);
    Value outPtrCast =
        LLVM::IntToPtrOp::create(rewriter, loc, ptrType, outPtrI64);

    // ========================================================================
    // CACHE OPERATION CONSTANTS
    // ========================================================================
    // QURT cache operation types
    Value QURT_MEM_CACHE_FLUSH = arith::ConstantOp::create(
        rewriter, loc, i32Ty, rewriter.getI32IntegerAttr(0)); // Flush only
    Value QURT_MEM_CACHE_FLUSH_INVALIDATE = arith::ConstantOp::create(
        rewriter, loc, i32Ty,
        rewriter.getI32IntegerAttr(2)); // Flush and invalidate
    Value QURT_MEM_DCACHE = arith::ConstantOp::create(
        rewriter, loc, i32Ty, rewriter.getI32IntegerAttr(1)); // Data cache

    // ========================================================================
    // ACTIVATION TRANSFORMATION: Row-Major → AH Layout
    // ========================================================================
    // Transform activation from row-major to AH (Activation-HMX) layout
    // This is done in-place on the temporary buffer
    hexkl::MacroRmToAhF16InplaceOp::create(rewriter, loc, MI32, KI32,
                                           actPtrCast);

    // Flush activation buffer to ensure HMX sees the transformed data
    hexkl::QurtMemCacheCleanOp::create(rewriter, loc, actPtrI32,
                                       actSizeBytesI32, QURT_MEM_CACHE_FLUSH,
                                       QURT_MEM_DCACHE);

    // ========================================================================
    // WEIGHT CACHE FLUSH
    // ========================================================================
    // Flush weight buffer to ensure HMX sees the preprocessed data
    // (Weights are already in WH layout, no transformation needed)
    hexkl::QurtMemCacheCleanOp::create(rewriter, loc, rhsPtrI32,
                                       wgtSizeBytesI32, QURT_MEM_CACHE_FLUSH,
                                       QURT_MEM_DCACHE);

    // ========================================================================
    // OUTPUT CACHE INVALIDATE
    // ========================================================================
    // Invalidate output buffer to ensure we see fresh data after HMX writes
    hexkl::QurtMemCacheCleanOp::create(
        rewriter, loc, outPtrI32, outSizeBytesI32,
        QURT_MEM_CACHE_FLUSH_INVALIDATE, QURT_MEM_DCACHE);

    // ========================================================================
    // MATRIX MULTIPLICATION: HMX Execution
    // ========================================================================
    // Perform the actual matrix multiplication using HMX hardware
    // Inputs are in AH and WH layouts, output is in AH layout
    hexkl::MacroMmF16Op::create(rewriter, loc, MI32, NI32, KI32, outPtrCast,
                                actPtrCast, rhsPtrCast);

    // ========================================================================
    // OUTPUT TRANSFORMATION: AH Layout → Row-Major
    // ========================================================================
    // Transform output from AH layout back to row-major layout
    // This is done in-place on the output buffer
    hexkl::MacroAhToRmF16InplaceOp::create(rewriter, loc, MI32, NI32,
                                           outPtrCast);

    // Deallocate temporary activation buffer
    memref::DeallocOp::create(rewriter, loc, actTemp);

    // Remove the original hexkl.matmul operation
    rewriter.eraseOp(op);
    return success();
  }
};

struct LowerHexKLMatmulToMacroPass
    : public PassWrapper<LowerHexKLMatmulToMacroPass,
                         InterfacePass<FunctionOpInterface>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerHexKLMatmulToMacroPass)

  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<hexkl::HexKLDialect, arith::ArithDialect, memref::MemRefDialect,
                LLVM::LLVMDialect, scf::SCFDialect>();
  }

  // Find the outermost loop containing a matmul operation
  // Used to determine where to place HMX init/lock calls
  scf::ForOp getOutermostLoop(Operation *op) {
    scf::ForOp outermostLoop = nullptr;
    Operation *current = op->getParentOp();

    while (current && !isa<FunctionOpInterface>(current)) {
      if (auto forOp = dyn_cast<scf::ForOp>(current)) {
        outermostLoop = forOp;
      }
      current = current->getParentOp();
    }

    return outermostLoop;
  }

  bool areConsecutive(hexkl::MatmulOp op1, hexkl::MatmulOp op2) {
    Operation *current = op1->getNextNode();
    while (current && current != op2) {
      if (isa<hexkl::MatmulOp>(current))
        return false;
      if (!current->hasTrait<OpTrait::IsTerminator>() &&
          current->getNumResults() > 0) {
        return false;
      }
      current = current->getNextNode();
    }
    return current == op2;
  }

  void runOnOperation() override {
    auto funcOp = getOperation();

    llvm::DenseSet<Operation *> handledLoops;
    SmallVector<hexkl::MatmulOp> standaloneMatmuls;
    SmallVector<std::pair<Operation *, Operation *>> standaloneGroups;

    funcOp.walk([&](hexkl::MatmulOp matmulOp) {
      if (auto outermostLoop = getOutermostLoop(matmulOp)) {
        handledLoops.insert(outermostLoop.getOperation());
        return;
      }
      standaloneMatmuls.push_back(matmulOp);
    });

    llvm::DenseSet<Operation *> processedStandaloneMatmuls;
    for (hexkl::MatmulOp matmulOp : standaloneMatmuls) {
      if (processedStandaloneMatmuls.contains(matmulOp.getOperation()))
        continue;

      Operation *groupStart = matmulOp.getOperation();
      Operation *groupEnd = matmulOp.getOperation();
      processedStandaloneMatmuls.insert(matmulOp.getOperation());

      Operation *current = matmulOp->getNextNode();
      hexkl::MatmulOp currentMatmul = matmulOp;
      while (current) {
        auto nextMatmul = dyn_cast<hexkl::MatmulOp>(current);
        if (!nextMatmul)
          break;
        if (getOutermostLoop(nextMatmul) ||
            !areConsecutive(currentMatmul, nextMatmul))
          break;
        groupEnd = nextMatmul.getOperation();
        processedStandaloneMatmuls.insert(nextMatmul.getOperation());
        currentMatmul = nextMatmul;
        current = nextMatmul->getNextNode();
      }

      standaloneGroups.push_back({groupStart, groupEnd});
    }

    OpBuilder builder(funcOp.getContext());
    for (Operation *loopOp : handledLoops) {
      auto forOp = cast<scf::ForOp>(loopOp);
      builder.setInsertionPoint(forOp);
      hexkl::MacroInitializeOp::create(builder, forOp.getLoc());
      hexkl::MacroLockHmxOp::create(builder, forOp.getLoc());

      builder.setInsertionPointAfter(forOp);
      hexkl::MacroUnlockHmxOp::create(builder, forOp.getLoc());
      hexkl::MacroFinalizeOp::create(builder, forOp.getLoc());
    }

    for (auto [groupStart, groupEnd] : standaloneGroups) {
      builder.setInsertionPoint(groupStart);
      hexkl::MacroInitializeOp::create(builder, groupStart->getLoc());
      hexkl::MacroLockHmxOp::create(builder, groupStart->getLoc());

      builder.setInsertionPointAfter(groupEnd);
      hexkl::MacroUnlockHmxOp::create(builder, groupEnd->getLoc());
      hexkl::MacroFinalizeOp::create(builder, groupEnd->getLoc());
    }

    RewritePatternSet patterns(&getContext());
    patterns.add<LowerHexKLMatmulToMacro>(&getContext());

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
      return;
    }
  }

  StringRef getArgument() const final { return "lower-hexkl-matmul-to-macro"; }

  StringRef getDescription() const final {
    return "Lower hexkl.matmul to hexkl_macro_mm_f16 (constant weights only)";
  }
};

} // namespace

namespace mlir {
namespace hexagon {

std::unique_ptr<InterfacePass<FunctionOpInterface>>
createLowerHexKLMatmulToMacroPass() {
  return std::make_unique<LowerHexKLMatmulToMacroPass>();
}

} // namespace hexagon
} // namespace mlir
