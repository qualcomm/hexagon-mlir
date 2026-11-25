//===--- VTCMTilingPass.cpp - implement a basic tiling for VTCM pass  ----====//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
// This pass tiles a (potentially fused) linalg.generic-on-tensors.
// It assumes that the tensor ins/outs are in DDR address space,
// the tiles need to be brought into VTCM, and the result stored back into DDR.
// Small tensors are completely prefetched (copied) to VTCM before the tiling.
//
// The tiling-option is external and is queried by this pass.
// This pass tiles and rewrites the IR so that after bufferization:
//    - space is allocated on VTCM.
//    - slices (or entire small tensors) are copied from DDR to VTCM.
//    - slices in VTCM are passed to the tiled generic.
//    - results copied back to DDR (by bufferizer or explicitly for small
//      tensors).
// Example:
// ```
//    %.. = linalg.generic {
//            indexing_maps = [#map, ...], iterator_types = [..]}
//            ins(%... : tensor<1024x256xf32>, ...)
//            outs(%... : tensor<1024x256xf32>, ...) { ... }
// ```
// will be re-written as:
// ```
//   % = scf.for %.. = %c0 to %c1024 step %c32_0 iter_args(%arg4 = %..)
//          -> (tensor<1024x256xf32>) {
//     % = scf.for %.. = %c0_1 to %c256 step %c64_2 iter_args(%arg6 = %..)
//            -> (tensor<1024x256xf32>) {
//          ...
//          %vtctm_tensor = bufferization.alloc_tensor()
//              copy(%ddr_tensor_slice) {memory_space = 1 : i64}
//              : tensor<32x64xf32>
//          ...
//          %.. = linalg.generic
//                 {indexing_maps = [#map, ...], iterator_types = [..]}
//                 ins(%vtcm_tensor, ... : tensor<32x64xf32>, ...)
//                 outs(%vtcm_tensor2 : tensor<32x64xf32>) { ... }
// ```
// which after bufferization, deallocation and hoisting will be:
// ```
//    %vtcm_alloc = memref.alloc() {alignment = 64 : i64} : memref<32x64xf32, 1>
//    scf.for %.. = %c0 to %c1024 step %c32 {
//      scf.for %.. = %c0 to %c256 step %c64 {
//        ...
//         memref.copy %ddr_subview, %vtcm_alloc
//               : memref<32x64xf32, strided<[256, 1], offset: ?>>
//                 to memref<32x64xf32, 1>
//         ...
//         %tile_result_on_vtcm  = linalg.generic {indexing_maps = }
//                  ins(%vtcm_alloc, ... : memref<32x64xf32, 1>, ...)
//                  outs(%vtcm_alloc2 : memref<32x64xf32, 1>) {  ... }
//         ...
//         memref.copy %tile_result_on_vtcm, %full_result_ddr
//                  : memref<32x64xf32, 1>
//                    to memref<32x64xf32, strided<[256, 1], offset: ?>>
//      }
//    }
//    memref.dealloc %vtcm_alloc : memref<32x64xf32, 1>
//```

#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/BufferViewFlowAnalysis.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"

#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/Support/Debug.h"
#include <algorithm>

#include "hexagon/Conversion/LinalgToLLVM/Common.h"
#include "hexagon/Conversion/LinalgToLLVM/LinalgToLLVM.h"
#include "hexagon/Conversion/LinalgToLLVM/VTCMTilingOptions.h"
#include "hexagon/Transforms/OptionsParsing.h"

#define DEBUG_TYPE "vtcm-tiling"
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define DBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
using namespace mlir::linalg;
using namespace hexagon;

#define GEN_PASS_DEF_VTCMTILING
#include "hexagon/Conversion/LinalgToLLVM/Passes.h.inc"

namespace {

struct VTCMTilingPass : public ::impl::VTCMTilingBase<VTCMTilingPass> {
  explicit VTCMTilingPass(const VTCMTilingOptions &options) : Base(options) {}
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<bufferization::BufferizationDialect>();
  }
  void runOnOperation() override;
};

// Returns true if `op` is a pure elementwise generic: every iterator is
// "parallel" (no reduction) and every indexing map is an identity
// permutation of the iteration space (no broadcast, no transpose/gather).
bool isPureElementwise(linalg::GenericOp op) {
  for (auto iterType : op.getIteratorTypesArray())
    if (iterType != utils::IteratorType::parallel)
      return false;
  for (AffineMap map : op.getIndexingMapsArray())
    if (!map.isIdentity())
      return false;
  return true;
}

// Annotate tiled IR for later passes (e.g. double buffering)
void annotateOp(Operation *op) {
  op->setAttr("all_parallel", mlir::UnitAttr::get(op->getContext()));
  op->setAttr("tiled_generic", mlir::UnitAttr::get(op->getContext()));
}

void annotateTiledLoop(linalg::GenericOp op, linalg::TiledLinalgOp tiledOp) {
  MLIRContext *ctx = op.getContext();
  bool allParallel = true;
  for (auto iterType : op.getIteratorTypesArray())
    if (iterType != utils::IteratorType::parallel)
      allParallel = false;

  if (allParallel && tiledOp.loops.size() > 0 &&
      isa<scf::ForOp>(tiledOp.loops[0]))
    annotateOp(tiledOp.loops[0]);
}

/// Returns true if the outs operand at the given DPS init index is write-only.
bool isOutsWriteOnly(linalg::GenericOp op, unsigned outsIdx) {
  return op.getRegionOutputArgs()[outsIdx].use_empty();
}

/// Allocate a tensor on the specified memory space without copying.
/// Used this for write-only outputs to avoid an unnecessary DDR read.
Value allocOnMemorySpace(IRRewriter &rewriter, Value rankedTensor, Location loc,
                         unsigned memorySpace) {
  auto sourceType = ::llvm::cast<RankedTensorType>(rankedTensor.getType());
  SmallVector<Value> dynamicSizes;
  for (auto [dim, size] : llvm::enumerate(sourceType.getShape()))
    if (ShapedType::isDynamic(size))
      dynamicSizes.push_back(
          tensor::DimOp::create(rewriter, loc, rankedTensor, dim));
  rewriter.setInsertionPointAfterValue(rankedTensor);
  auto allocOp = bufferization::AllocTensorOp::create(rewriter, loc, sourceType,
                                                      dynamicSizes);
  allocOp.setMemorySpaceAttr(
      rewriter.getIntegerAttr(rewriter.getI64Type(), memorySpace));
  return allocOp.getResult();
}

/// Allocate a fresh tensor on VTCM without copying content.
inline Value allocOnVTCM(IRRewriter &rewriter, Value rankedTensor,
                         Location loc) {
  return allocOnMemorySpace(rewriter, rankedTensor, loc, VTCM_ADDRESS_SPACE);
}

/// Takes a 'rankedTensor' value and returns a tensor on the specified memory
/// space, optionally copying data.
Value copyToMemorySpace(IRRewriter &rewriter, Value rankedTensor, Location loc,
                        unsigned memorySpace) {
  SmallVector<Value> dynamicSizes;
  auto sourceType = ::llvm::cast<RankedTensorType>(rankedTensor.getType());

  rewriter.setInsertionPointAfterValue(rankedTensor);
  // For a static-shaped tensor.empty() the contents are undef by definition,
  // so drop the copy operand — OneShotBufferize will not emit a DDR alloc +
  // memref.copy for the source.  Dynamic shapes are left unchanged for now:
  // dropping the copy operand there would require forwarding dynamic sizes to
  // alloc_tensor, which adds complexity without a known use-case.
  bool isStaticEmpty = rankedTensor.getDefiningOp<tensor::EmptyOp>() &&
                       sourceType.hasStaticShape();
  Value copySource = isStaticEmpty ? Value{} : rankedTensor;
  auto copyOp = bufferization::AllocTensorOp::create(rewriter, loc, sourceType,
                                                     dynamicSizes, copySource);
  copyOp.setMemorySpaceAttr(
      rewriter.getIntegerAttr(rewriter.getI64Type(), memorySpace));
  return copyOp.getResult();
}

/// Takes a 'rankedTensor' value and returns a tensor on DDR.
inline Value copyToDDR(IRRewriter &rewriter, Value rankedTensor, Location loc) {
  return copyToMemorySpace(rewriter, rankedTensor, loc,
                           DEFAULT_DDR_ADDRESS_SPACE);
}

/// Takes a 'rankedTensor' value and returns a tensor on VTCM.
inline Value copyToVTCM(IRRewriter &rewriter, Value rankedTensor,
                        Location loc) {
  return copyToMemorySpace(rewriter, rankedTensor, loc, VTCM_ADDRESS_SPACE);
}

/// Return a cloned generic with `newIns` and `newOuts`.
GenericOp getGenericWithNewOperands(IRRewriter &rewriter, GenericOp op,
                                    SmallVector<Value> &newIns,
                                    SmallVector<Value> &newOuts) {
  rewriter.setInsertionPointAfter(op);
  GenericOp newOp = GenericOp::create(
      rewriter, op.getLoc(), op.getResultTypes(), newIns, newOuts,
      op.getIndexingMapsArray(), op.getIteratorTypesArray(),
      /*bodyBuild=*/nullptr);
  rewriter.inlineRegionBefore(op->getRegion(0), newOp.getRegion(),
                              newOp.getRegion().begin());
  return newOp;
}

/// Return a linalg generic where `prefetch` tensors are VTCM copies
/// of data on DDR. Other operands are as in original.
GenericOp replaceGenericWithPrefetchedOperands(IRRewriter &rewriter,
                                               GenericOp op,
                                               SmallVector<bool> prefetch) {
  SmallVector<Value> newIns;
  SmallVector<Value> newOuts;

  for (OpOperand &opOperand : op->getOpOperands()) {
    auto idx = opOperand.getOperandNumber();
    Value globalTensor = op->getOperand(idx);
    Value newTensor = globalTensor;

    if (prefetch[idx]) {
      bool writeOnly = op.isDpsInit(&opOperand) &&
                       isOutsWriteOnly(op, idx - op.getNumDpsInputs());
      // allocate space on VTCM for the prefetched. Copy only if 'read'.
      newTensor = writeOnly ? allocOnVTCM(rewriter, globalTensor, op.getLoc())
                            : copyToVTCM(rewriter, globalTensor, op.getLoc());
    }
    op.isDpsInit(&opOperand) ? newOuts.push_back(newTensor)
                             : newIns.push_back(newTensor);
  }
  auto newOp = getGenericWithNewOperands(rewriter, op, newIns, newOuts);
  rewriter.replaceOp(op, newOp->getResults());
  return newOp;
}

/// Replace tiled generic that operates on slices from DDR,
/// to a new generic that operates on copies on VTCM.
LogicalResult replaceTiledGenericWithVTCMSlices(IRRewriter &rewriter,
                                                GenericOp top,
                                                SmallVector<bool> prefetch) {
  SmallVector<Value> newIns, newOuts;
  for (OpOperand &opOperand : top->getOpOperands()) {
    auto idx = opOperand.getOperandNumber();
    Value globalTensor = top->getOperand(idx);
    Value newTensor = globalTensor;

    if (!prefetch[idx]) {
      if (!globalTensor.template getDefiningOp<tensor::ExtractSliceOp>())
        return failure();
      bool writeOnly = top.isDpsInit(&opOperand) &&
                       isOutsWriteOnly(top, idx - top.getNumDpsInputs());
      if (writeOnly) {
        // Write-only out: allocate on VTCM space; without the copy.
        // The slice become dead and will be removed by canonicalize.
        newTensor = allocOnVTCM(rewriter, globalTensor, top.getLoc());
      } else {
        newTensor = copyToVTCM(rewriter, globalTensor, top.getLoc());
      }
    }
    top.isDpsInit(&opOperand) ? newOuts.push_back(newTensor)
                              : newIns.push_back(newTensor);
  }
  GenericOp newOp = getGenericWithNewOperands(rewriter, top, newIns, newOuts);
  rewriter.replaceOp(top, newOp->getResults());
  return success();
}

/// Copying the results corresponding to the operands to be "prefetched" for a
/// linalg op to DDR and replacing the uses to the copied tensor on DDR.
void copyResultsToDDR(IRRewriter &rewriter, GenericOp op,
                      SmallVector<bool> prefetch) {
  for (int idx = 0; idx < op.getNumDpsInits(); ++idx) {
    auto operandIdx = op.getNumDpsInputs() + idx;
    if (prefetch[operandIdx]) {
      Value resultTensor = op.getResult(idx);
      Value newTensor = copyToDDR(rewriter, resultTensor, op.getLoc());
      rewriter.replaceAllUsesExcept(resultTensor, newTensor,
                                    newTensor.getDefiningOp());
    }
  }
}

void VTCMTilingPass::runOnOperation() {
  auto userProvidedTileSizes = parseTileSizes(tileSizes);
  auto funcOp = getOperation();

  funcOp.walk([&](linalg::GenericOp op) {
    IRRewriter rewriter(op.getContext());

    // Skip generics already enclosed by a tiled scf.for annotated with
    // `tiled_generic` (e.g. produced by PromoteTiledSCFLoopToVTCMPass) so we
    // don't re-tile something that has already been tiled upstream.
    for (Operation *parent = op->getParentOp(); parent;
         parent = parent->getParentOp()) {
      if (isa<scf::ForOp>(parent) && parent->hasAttr("tiled_generic"))
        return WalkResult::advance();
    }

    // Pure elementwise ops never benefit from VTCM residency -- every
    // operand element is touched exactly once regardless of tiling,
    // so staging only adds alloc+DMA-copy+free overhead on both the
    // way in and the way out. Leave the op untouched; it bufferizes to
    // plain memrefs operating directly on DDR. Respect an explicit
    // user-provided tile size (e.g. from a test or tool invocation) as an
    // intentional override of this default.
    if (!userProvidedTileSizes && isPureElementwise(op)) {
      DBG("skipping VTCM staging for pure elementwise op: " << op);
      return WalkResult::advance();
    }

    SmallVector<bool> prefetch(op.getNumOperands(), false);
    FailureOr<linalg::LinalgTilingOptions> vtcmTilingOptions =
        getVTCMTilingOptions(op, userProvidedTileSizes, prefetch, vtcmBudget);
    if (failed(vtcmTilingOptions))
      return WalkResult::advance();

    // Replace with a new generic where operands which fit completely
    // into VTCM (prefetch 'set') are copied to VTCM before tiling, then copy
    // the corresponding results back to DDR.
    linalg::GenericOp prefetchOp =
        replaceGenericWithPrefetchedOperands(rewriter, op, prefetch);
    copyResultsToDDR(rewriter, prefetchOp, prefetch);

    rewriter.setInsertionPointAfter(prefetchOp);
    FailureOr<linalg::TiledLinalgOp> tiledOp =
        linalg::tileLinalgOp(rewriter, prefetchOp, *vtcmTilingOptions);
    if (failed(tiledOp))
      return WalkResult::advance();

    // annotate generated loop for reference.
    annotateTiledLoop(prefetchOp, *tiledOp);

    auto top = llvm::dyn_cast<GenericOp>(tiledOp->op.getOperation());
    if (failed(replaceTiledGenericWithVTCMSlices(rewriter, top, prefetch)))
      return WalkResult::advance();

    rewriter.replaceOp(prefetchOp, tiledOp->tensorResults);
    return WalkResult::advance();
  });
}
} // namespace

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
hexagon::createVTCMTilingPass(const VTCMTilingOptions &options) {
  return std::make_unique<VTCMTilingPass>(options);
}
