//===- HoistScalarOps.cpp - Hoist scalar ops from linalg.generic ----------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// This pass hoists loop-invariant scalar arithmetic and math operations from
// linalg.generic regions to just before the operation. Each hoisted operation
// is converted to a rank-0 tensor input with a corresponding scalar indexing
// map and block argument. If not hoisted, this could lead to these scalar
// ops to be vectorized unecessarily, hurting performance in inner loops. This
// transformation prepares the IR for subsequent loop-invariant code motion
// (LICM) passes.
//
// The pass handles:
// - Chains of dependent scalar operations
// - Type-changing operations (arith.extf, arith.truncf)
// - Operations that depend on rank-0 tensor inputs
//
// Example transformation:
//   %scalar = ... : tensor<f32>        // rank-0 scalar tensor
//   %vector = ... : tensor<128xf32>    // rank-1 vector tensor
//
//   %result = linalg.generic
//       ins(%scalar, %vector : tensor<f32>, tensor<128xf32>)
//       outs(%output : tensor<128xf32>) {
//     ^bb0(%s: f32, %v: f32, %out: f32):
//       %rsqrt = math.rsqrt %s : f32           // scalar op, loop-invariant
//       %mul = arith.mulf %v, %rsqrt : f32     // uses vector element, not
//       hoistable linalg.yield %mul : f32
//   }
//
// becomes:
//   %scalar = ... : tensor<f32>
//   %vector = ... : tensor<128xf32>
//
//   %extracted = tensor.extract %scalar[] : tensor<f32>
//   %rsqrt = math.rsqrt %extracted : f32       // hoisted outside
//   %rsqrt_tensor = tensor.from_elements %rsqrt : tensor<f32>
//
//   %result = linalg.generic
//       ins(%scalar, %rsqrt_tensor, %vector : tensor<f32>, tensor<f32>,
//       tensor<128xf32>) outs(%output : tensor<128xf32>) {
//     ^bb0(%s: f32, %r: f32, %v: f32, %out: f32):
//       %mul = arith.mulf %v, %r : f32         // now uses hoisted value
//       linalg.yield %mul : f32
//   }
//
//===----------------------------------------------------------------------===//

#include "hexagon/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"

#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "hoist-scalar-ops"

using namespace mlir;

namespace {

struct HoistScalarOpsPass
    : public PassWrapper<HoistScalarOpsPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(HoistScalarOpsPass)

  StringRef getArgument() const final {
    return "hoist-scalar-ops-in-linalg-generic";
  }

  StringRef getDescription() const final {
    return "Hoists loop-invariant scalar arith/math ops out of linalg.generic";
  }

  void runOnOperation() override;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, arith::ArithDialect,
                    math::MathDialect, tensor::TensorDialect>();
  }

private:
  /// Identify all hoistable scalar operations in the generic's region.
  ///
  /// We maintain two data structures:
  /// - hoistableOps (vector): Preserves the order of operations for correct
  /// hoisting
  /// - canHoistSet (set): Enables O(1) lookup during invariance analysis
  void findHoistableOps(Block &regionBlock, unsigned oldNumInputs,
                        linalg::GenericOp gop,
                        SmallVectorImpl<Operation *> &hoistableOps,
                        DenseSet<Operation *> &canHoistSet);

  /// Hoist the identified operations outside the generic and wrap them in
  /// rank-0 tensors
  void hoistOperations(linalg::GenericOp gop, OpBuilder &builder,
                       ArrayRef<Operation *> hoistableOps,
                       unsigned oldNumInputs,
                       SmallVectorImpl<Value> &newHoistedTensors,
                       DenseMap<Operation *, Value> &hoistedOpResultMap);

  /// Build new generic op with updated inputs, indexing maps, and region
  linalg::GenericOp
  buildNewGenericOp(linalg::GenericOp gop, OpBuilder &builder, MLIRContext *ctx,
                    ArrayRef<Value> newHoistedTensors, unsigned oldNumInputs,
                    unsigned oldNumOutputs, ArrayRef<Operation *> hoistableOps,
                    const DenseSet<Operation *> &canHoistSet,
                    Block &regionBlock);

  /// Process a single linalg.generic operation and hoist invariant scalar ops.
  /// Returns true if any changes were made to the IR.
  bool runOnGenericOp(linalg::GenericOp gop, MLIRContext *ctx);
};

void HoistScalarOpsPass::runOnOperation() {
  func::FuncOp func = getOperation();
  MLIRContext *ctx = func.getContext();

  LLVM_DEBUG(llvm::dbgs() << "=== Running HoistScalarOpsPass ===\n");

  // Iterate until no more changes occur. This is necessary to handle chains
  // of dependent operations where hoisting one operation makes another
  // operation hoistable in the next iteration.
  for (bool changed = true; changed;) {
    // Collect all linalg.generic operations before modifying the IR.
    // We cannot modify the IR while walking it, as that would invalidate
    // the walk iterator.
    SmallVector<linalg::GenericOp> genericOps;
    func.walk([&](linalg::GenericOp gop) { genericOps.push_back(gop); });

    LLVM_DEBUG(llvm::dbgs() << "Found " << genericOps.size()
                            << " linalg.generic operations\n");

    changed = false;
    for (linalg::GenericOp gop : genericOps) {
      if (runOnGenericOp(gop, ctx)) {
        // Break and restart the walk after each modification to ensure
        // we're working with valid IR
        changed = true;
        break;
      }
    }
  }

  LLVM_DEBUG(llvm::dbgs() << "=== HoistScalarOpsPass Complete ===\n");
}

void HoistScalarOpsPass::findHoistableOps(
    Block &regionBlock, unsigned oldNumInputs, linalg::GenericOp gop,
    SmallVectorImpl<Operation *> &hoistableOps,
    DenseSet<Operation *> &canHoistSet) {
  for (Operation &op : regionBlock.getOperations()) {
    // Stop at the terminator (linalg.yield)
    if (isa<linalg::YieldOp>(op))
      break;

    // Only consider arith and math dialect operations
    auto *dialect = op.getDialect();
    if (!dialect || !(dialect->getNamespace() == "arith" ||
                      dialect->getNamespace() == "math"))
      continue;

    // Only hoist scalar operations (single result with scalar type)
    if (op.getNumResults() != 1 ||
        !isa<FloatType, IntegerType>(op.getResult(0).getType()))
      continue;

    // Check if all operands are loop-invariant
    bool isInvariant = llvm::all_of(op.getOperands(), [&](Value v) {
      // Case 1: Operand defined outside the linalg.generic region
      if (v.getParentBlock() != &regionBlock)
        return true;

      // Case 2: Operand is a constant
      if (isa_and_nonnull<arith::ConstantOp>(v.getDefiningOp()))
        return true;

      // Case 3: Operand is a block argument from a rank-0 tensor input
      if (auto barg = dyn_cast<BlockArgument>(v)) {
        if (barg.getArgNumber() < oldNumInputs) {
          auto tensorType = dyn_cast<RankedTensorType>(
              gop.getDpsInputOperand(barg.getArgNumber())->get().getType());
          return tensorType && tensorType.getRank() == 0;
        }
      }

      return false;
    });

    if (isInvariant) {
      hoistableOps.push_back(&op);
      canHoistSet.insert(&op);
      LLVM_DEBUG(llvm::dbgs()
                 << "  Found hoistable op: " << op.getName() << "\n");
    }
  }
}

void HoistScalarOpsPass::hoistOperations(
    linalg::GenericOp gop, OpBuilder &builder,
    ArrayRef<Operation *> hoistableOps, unsigned oldNumInputs,
    SmallVectorImpl<Value> &newHoistedTensors,
    DenseMap<Operation *, Value> &hoistedOpResultMap) {
  for (Operation *op : hoistableOps) {
    IRMapping bvm;

    // Remap operands for the cloned operation
    for (Value operand : op->getOperands()) {
      // If operand comes from a previously hoisted operation, use the
      // hoisted scalar value
      if (auto defOp = operand.getDefiningOp()) {
        if (hoistedOpResultMap.count(defOp)) {
          bvm.map(operand, hoistedOpResultMap[defOp]);
          continue;
        }
      }

      // If operand is a block argument from a rank-0 tensor input,
      // extract the scalar value from the tensor
      if (auto barg = dyn_cast<BlockArgument>(operand)) {
        if (barg.getArgNumber() < oldNumInputs) {
          Value inputTensor =
              gop.getDpsInputOperand(barg.getArgNumber())->get();
          Value extracted =
              builder.create<tensor::ExtractOp>(gop.getLoc(), inputTensor);
          bvm.map(operand, extracted);
          continue;
        }
      }

      // Otherwise, the operand is from outside and needs no mapping
    }

    // Clone the operation outside the linalg.generic
    Operation *hoistedOp = builder.clone(*op, bvm);
    Value scalarResult = hoistedOp->getResult(0);
    hoistedOpResultMap[op] = scalarResult;

    // Wrap the scalar result in a rank-0 tensor to pass as input
    auto tensorType = RankedTensorType::get({}, scalarResult.getType());
    Value tensorInput = builder.create<tensor::FromElementsOp>(
        gop.getLoc(), tensorType, scalarResult);
    newHoistedTensors.push_back(tensorInput);
  }
}

linalg::GenericOp HoistScalarOpsPass::buildNewGenericOp(
    linalg::GenericOp gop, OpBuilder &builder, MLIRContext *ctx,
    ArrayRef<Value> newHoistedTensors, unsigned oldNumInputs,
    unsigned oldNumOutputs, ArrayRef<Operation *> hoistableOps,
    const DenseSet<Operation *> &canHoistSet, Block &regionBlock) {
  // Build new inputs list
  SmallVector<Value> newInputs(gop.getInputs());
  newInputs.append(newHoistedTensors.begin(), newHoistedTensors.end());
  unsigned newNumInputs = newInputs.size();

  // Build new indexing maps: [old_input_maps, new_scalar_maps,
  // old_output_maps]
  SmallVector<AffineMap> oldMaps = gop.getIndexingMapsArray();
  SmallVector<AffineMap> newMaps;

  // Copy old input maps
  for (unsigned i = 0; i < oldNumInputs; ++i) {
    newMaps.push_back(oldMaps[i]);
  }

  // Add scalar maps for hoisted tensors (rank-0, no dimensions accessed)
  unsigned numLoops = gop.getNumLoops();
  for (size_t i = 0; i < newHoistedTensors.size(); ++i) {
    newMaps.push_back(AffineMap::get(numLoops, 0, {}, ctx));
  }

  // Copy old output maps
  for (unsigned i = oldNumInputs; i < oldMaps.size(); ++i) {
    newMaps.push_back(oldMaps[i]);
  }

  // Create the new linalg.generic operation
  auto newGeneric = builder.create<linalg::GenericOp>(
      gop.getLoc(), gop->getResultTypes(), newInputs, gop.getOutputs(), newMaps,
      gop.getIteratorTypesArray());

  // Build the region for the new linalg.generic
  Region &region = newGeneric.getRegion();
  Block *newBlock = builder.createBlock(&region);

  // Add block arguments for all inputs (old + hoisted)
  for (size_t i = 0; i < newInputs.size(); ++i) {
    Type elemType = cast<ShapedType>(newInputs[i].getType()).getElementType();
    newBlock->addArgument(elemType, gop.getLoc());
  }

  // Add block arguments for all outputs
  SmallVector<Value> outputs(gop.getOutputs());
  for (size_t i = 0; i < outputs.size(); ++i) {
    Type elemType = cast<ShapedType>(outputs[i].getType()).getElementType();
    newBlock->addArgument(elemType, gop.getLoc());
  }

  IRMapping bvm;

  // Map old input arguments to new input arguments (same positions)
  for (unsigned i = 0; i < oldNumInputs; ++i) {
    bvm.map(regionBlock.getArgument(i), newBlock->getArgument(i));
  }

  // Map old output arguments to new output arguments
  // (shifted by the number of hoisted inputs)
  for (unsigned i = 0; i < oldNumOutputs; ++i) {
    unsigned oldArgIdx = oldNumInputs + i;
    unsigned newArgIdx = newNumInputs + i;
    bvm.map(regionBlock.getArgument(oldArgIdx),
            newBlock->getArgument(newArgIdx));
  }

  // Map hoisted operation results to their corresponding new block arguments
  for (size_t i = 0; i < hoistableOps.size(); ++i) {
    unsigned newArgIdx = oldNumInputs + i;
    bvm.map(hoistableOps[i]->getResult(0), newBlock->getArgument(newArgIdx));
  }

  // Clone all non-hoisted operations into the new region
  builder.setInsertionPointToEnd(newBlock);
  for (Operation &op : regionBlock.getOperations()) {
    if (canHoistSet.count(&op) == 0) {
      builder.clone(op, bvm);
    }
  }

  return newGeneric;
}

bool HoistScalarOpsPass::runOnGenericOp(linalg::GenericOp gop,
                                        MLIRContext *ctx) {
  if (!gop || gop.getRegion().empty() || gop.getRegion().front().empty())
    return false;

  Block &regionBlock = gop.getRegion().front();
  unsigned oldNumInputs = gop.getNumDpsInputs();
  unsigned oldNumOutputs = gop.getOutputs().size();

  LLVM_DEBUG({
    llvm::dbgs() << "Processing linalg.generic with " << oldNumInputs
                 << " inputs and " << oldNumOutputs << " outputs\n";
  });

  SmallVector<Operation *> hoistableOps;
  DenseSet<Operation *> canHoistSet;
  findHoistableOps(regionBlock, oldNumInputs, gop, hoistableOps, canHoistSet);

  if (hoistableOps.empty()) {
    LLVM_DEBUG(llvm::dbgs() << "  No hoistable operations found\n");
    return false;
  }

  LLVM_DEBUG(llvm::dbgs() << "  Hoisting " << hoistableOps.size()
                          << " operations\n");

  OpBuilder builder(gop);
  SmallVector<Value> newHoistedTensors;
  DenseMap<Operation *, Value> hoistedOpResultMap;
  hoistOperations(gop, builder, hoistableOps, oldNumInputs, newHoistedTensors,
                  hoistedOpResultMap);

  auto newGeneric =
      buildNewGenericOp(gop, builder, ctx, newHoistedTensors, oldNumInputs,
                        oldNumOutputs, hoistableOps, canHoistSet, regionBlock);

  LLVM_DEBUG(llvm::dbgs() << "  Creating new linalg.generic with "
                          << (oldNumInputs + newHoistedTensors.size())
                          << " inputs\n");

  gop.replaceAllUsesWith(newGeneric.getResults());
  gop.erase();

  LLVM_DEBUG(llvm::dbgs() << "  Successfully hoisted operations\n");
  return true;
}

} // namespace

namespace mlir {
namespace hexagon {

std::unique_ptr<Pass> createHoistScalarOpsPass() {
  return std::make_unique<HoistScalarOpsPass>();
}

} // namespace hexagon
} // namespace mlir
