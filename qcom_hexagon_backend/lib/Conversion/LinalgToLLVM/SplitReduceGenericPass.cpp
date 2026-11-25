//===- SplitReduceGenericPass.cpp : Split Reduce Generic Operations ------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// This pass splits linalg generic reduction ops with static shapes into
// parallel and reduction phases for vectorization. It preserves the original
// reduction identity constant, ensuring correct semantics for operations
// like max reductions that require specific identity (e.g., -inf for maxnumf).
//
//===----------------------------------------------------------------------===//

#include "hexagon/Conversion/LinalgToLLVM/Common.h"
#include "hexagon/Conversion/LinalgToLLVM/LinalgToLLVM.h"
#include "hexagon/Transforms/OptionsParsing.h"
#include "hexagon/Transforms/Passes.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"
#include <algorithm>

#define DEBUG_TYPE "split-reduce-generic"
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define DBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
using namespace mlir::linalg;
using namespace hexagon;

#define GEN_PASS_DEF_SPLITREDUCEGENERIC
#include "hexagon/Conversion/LinalgToLLVM/Passes.h.inc"

namespace {

LogicalResult splitGenericReduce(RewriterBase &rewriter,
                                 linalg::GenericOp genericOp) {
  DBG("Processing generic op for reduction splitting");

  // Split one reduction dimension while preserving any surrounding parallel
  // dimensions.  Attention row reductions, for example, are represented as
  // [parallel, reduction]: the query-row loop must remain parallel while the
  // key loop is split for vectorization.
  auto iteratorTypes = genericOp.getIteratorTypesArray();
  unsigned reductionLoopCount =
      llvm::count(iteratorTypes, utils::IteratorType::reduction);
  bool hasOneReductionLoop = reductionLoopCount == 1;
  bool hasOnlyParallelAndReductionLoops =
      llvm::all_of(iteratorTypes, [](utils::IteratorType iteratorType) {
        return iteratorType == utils::IteratorType::parallel ||
               iteratorType == utils::IteratorType::reduction;
      });

  // This is to avoid successive split-reduction.
  if (!hasOneReductionLoop || !hasOnlyParallelAndReductionLoops) {
    return rewriter.notifyMatchFailure(
        genericOp,
        "generic op does not have exactly one reduction loop and only "
        "parallel surrounding loops");
  }

  // This pass targets reductions in the innermost loop. Leave column
  // reductions such as [reduction, parallel] unchanged.
  if (iteratorTypes.back() != utils::IteratorType::reduction) {
    return rewriter.notifyMatchFailure(genericOp,
                                       "reduction loop is not innermost");
  }

  // Restrict this transformation to associative floating-point reductions.
  // ExtFOp is a per-element input promotion, not a combiner, and is used by
  // f16-to-f32 sum reductions such as RMSNorm and LayerNorm.
  for (Operation &bodyOp : genericOp.getBody()->getOperations()) {
    if (!isa<arith::AddFOp, arith::ExtFOp, arith::MaxNumFOp, arith::MinNumFOp,
             arith::MulFOp, linalg::YieldOp>(bodyOp)) {
      return rewriter.notifyMatchFailure(
          genericOp, "generic op is not a supported floating-point reduction");
    }
  }

  // Reject ops with dynamic shapes; splitReduction only handles static shapes.
  for (auto operand : genericOp->getOperands()) {
    auto shapedType = dyn_cast<ShapedType>(operand.getType());
    if (shapedType && !shapedType.hasStaticShape()) {
      return rewriter.notifyMatchFailure(genericOp,
                                         "generic op has dynamic shapes");
    }
  }

  // This is to overcome a bug in upstream MLIR split-reduce-generic.
  // Check if the genericOp has one outs whose defining op is tensor.insert,
  // tensor.from_elements, or linalg.fill with a constant value. preserve that
  // value.
  TypedAttr preservedConstant = nullptr;
  if (genericOp.getNumDpsInits() == 1) {
    Value output = genericOp.getDpsInitOperand(0)->get();

    // Bufferization can wrap a filled tensor in alloc_tensor copy(...).
    // Look through copies to recover the original initializer.
    while (auto allocTensor =
               output.getDefiningOp<bufferization::AllocTensorOp>()) {
      Value copy = allocTensor.getCopy();
      if (!copy)
        break;
      output = copy;
    }

    // Check for tensor.insert
    if (auto insertOp = output.getDefiningOp<tensor::InsertOp>()) {
      Value insertedValue = insertOp.getScalar();
      if (auto constOp = insertedValue.getDefiningOp<arith::ConstantOp>()) {
        // Preserve the constant value (could be f32 or f16)
        preservedConstant = constOp.getValue();
        DBG("Preserved constant value from tensor.insert: "
            << preservedConstant);
      }
    }
    // Check for tensor.from_elements
    else if (auto fromElementsOp =
                 output.getDefiningOp<tensor::FromElementsOp>()) {
      if (fromElementsOp.getElements().size() == 1) {
        Value element = fromElementsOp.getElements()[0];
        if (auto constOp = element.getDefiningOp<arith::ConstantOp>()) {
          // Preserve the constant value (could be f32 or f16)
          preservedConstant = constOp.getValue();
          DBG("Preserved constant value from tensor.from_elements: "
              << preservedConstant);
        }
      }
    }
    // Preserve the source fill value for the generated partial reduction.
    else if (auto fillOp = output.getDefiningOp<linalg::FillOp>()) {
      if (auto constOp = fillOp.value().getDefiningOp<arith::ConstantOp>()) {
        preservedConstant = constOp.getValue();
        DBG("Preserved constant value from linalg.fill: " << preservedConstant);
      }
    }
  }

  // Create control function for split reduction
  auto controlFn = [](LinalgOp op) -> SplitReductionOptions {
    SplitReductionOptions options;
    auto tileSize = computeDataTileSize(op); // vector length.
    options.ratio = tileSize.has_value() ? *tileSize : 32;
    // Split the reduction loop, not always loop 0.
    auto iteratorTypes = op.getIteratorTypesArray();
    options.index = llvm::find(iteratorTypes, utils::IteratorType::reduction) -
                    iteratorTypes.begin();
    options.innerParallel = true; // Inner dimension is parallel
    return options;
  };

  // This utility expands the selected reduction dimension, preserving the
  // surrounding parallel loops. For [query-row, key], it produces
  // [query-row, key/32, lane] followed by a final reduction over lane.
  FailureOr<SplitReductionResult> result =
      linalg::splitReduction(rewriter, cast<LinalgOp>(genericOp.getOperation()),
                             controlFn, /*useAlloc=*/false);

  if (failed(result)) {
    return rewriter.notifyMatchFailure(genericOp,
                                       "failed to split reduction operation");
  }

  // If we preserved a constant, replace the fillOp's value with it
  if (preservedConstant && result->fillOp) {
    Value fillValue = result->fillOp.value();
    if (auto fillConstOp = fillValue.getDefiningOp<arith::ConstantOp>()) {
      rewriter.setInsertionPoint(fillConstOp);
      Type constType = preservedConstant.getType();
      Value newConstOp = arith::ConstantOp::create(
          rewriter, fillConstOp.getLoc(), constType, preservedConstant);
      rewriter.replaceAllUsesWith(fillConstOp.getResult(), newConstOp);
      DBG("Replaced fillOp constant with preserved value");
    }
  }

  DBG("Successfully split reduction operation");
  return success();
}

struct SplitReduceGenericPass
    : public ::impl::SplitReduceGenericBase<SplitReduceGenericPass> {

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, tensor::TensorDialect,
                    arith::ArithDialect, scf::SCFDialect>();
  }

  void runOnOperation() override {
    auto funcOp = getOperation();
    IRRewriter rewriter(funcOp.getContext());

    funcOp.walk([&](linalg::GenericOp op) {
      if (failed(splitGenericReduce(rewriter, op))) {
        DBG("Failed to split generic reduce for op");
        // Don't signal pass failure, just skip this op
      }
    });
  }
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
hexagon::createSplitReduceGenericPass() {
  return std::make_unique<SplitReduceGenericPass>();
}
