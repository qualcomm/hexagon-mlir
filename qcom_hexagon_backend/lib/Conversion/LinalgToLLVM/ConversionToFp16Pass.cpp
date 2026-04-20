//===- ConversionToFp16Pass.cpp - Convert fp64/fp32 to fp16  //-===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===---------------------------------------------------------------------===//
//
// This function level pass tries to `reduce-precision`
// for some of the computations and stores (tensor-type) from fp32/fp64 to fp16.
//
// Computations and storage in lower precision (fp16) improves computation speed
// and lowers storage and memory transfers. However, it cannot be blindly
// applied everywhere. Some precision lifting is in fact required and performed
// during model conversion (e.g. by PyTorch/Triton as a precision-policy).
// This pass enables further precision reduction beyond framework defaults,
// allowing users to trade accuracy for performance.
//
// This pass iterates over the operations in a fp16 model
// and converts all the fp64/fp32 operations to fp16 operations except few
// excluded ops. The excluded ops are certain operations which if converted to
// fp16 might cause accuracy issues or those which require special handling. It
// uses op walk and pattern matching to replace op with fp16 version while
// maintaining IR validity through strategic insertion of truncf and extf
// conversion operations.
//
// The pass works by:
// 1. Identifying operations with fp64/fp32 types
//
// 2. Updating the op with :
//  - Operands conversion: Convert the operands to fp16 version,
//    by inserting truncf operations if required  to convert the input to fp16.
//  - Operation clone: Clone the op with the updated operands and result type.
//  - Create newRegion and newBlock in the cloned op: Update the block arguments
//    with new datatype and strategically insert extf/truncf operations for the
//    block arguments into the newBlock to ensure IR validity inside the block.
//  - Copy all ops in the basic block from oldRegion to newRegion and
//    recursively call the update function on the ops which changes the
//    datatype.
//
// 3. Inserting extf operations for output to ensure the consumer op gets the
// expected data-type
//
// 4. Replace the old op with: (input)truncfOp -> fp16 Op -> (output)extfOp.
//
//===---------------------------------------------------------------------===//

#include "hexagon/Conversion/LinalgToLLVM/LinalgToLLVM.h"
#include "hexagon/Conversion/LinalgToLLVM/Passes.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/ErrorHandling.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/Visitors.h>
#include <mlir/Support/LLVM.h>

#define DEBUG_TYPE "conversion-to-fp16"

#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define DBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
using namespace mlir::func;
using namespace hexagon;

#define GEN_PASS_DEF_CONVERSIONTOFP16
#include "hexagon/Conversion/LinalgToLLVM/Passes.h.inc"

namespace {

Type getScalarOrElementType(Type ty) {
  if (auto tensorTy = dyn_cast<TensorType>(ty))
    return tensorTy.getElementType();
  if (auto memrefTy = dyn_cast<BaseMemRefType>(ty))
    return memrefTy.getElementType();
  return ty;
}

// This consists of a set of excluded operations
// which if converted to fp16 might cause accuracy issues or
// those which require special handling.
static const llvm::DenseSet<llvm::StringRef> excludedOp{
    "linalg.matmul", "arith.constant", "linalg.yield", "linalg.reduce",
    "arith.divf"};

// Checks if the operation is in the excluded set
bool opToExclude(Operation *op) {
  auto &set = excludedOp;
  return set.contains(op->getName().getStringRef());
}

// Returns true if target is lower precision.
bool canDowncastToTargetType(Type oldtype, Type targetType) {
  Type oldScalarType = getScalarOrElementType(oldtype);
  Type targetScalarType = getScalarOrElementType(targetType);

  if (targetScalarType.isF16())
    return oldScalarType.isF32() || oldScalarType.isF64();
  else if (targetScalarType.isF32())
    return oldScalarType.isF64();
  return false;
}

// Returns true if any operand or result can be downcast to targetType.
bool canDowncastToTargetType(Operation *op, Type targetType) {
  auto canDowncast = [&](Type type) {
    return canDowncastToTargetType(type, targetType);
  };

  bool hasDowncastOperand = llvm::any_of(op->getOperands(), [&](Value operand) {
    return canDowncast(operand.getType());
  });
  bool hasDowncastResult = llvm::any_of(op->getResults(), [&](Value result) {
    return canDowncast(result.getType());
  });
  return hasDowncastOperand || hasDowncastResult;
}

// Always call canDowncastToTargetType() first before calling this function
// so as to ensure that it is valid conversion of data type.
// Returns Type() (null) if conversion is not possible.
Type downcastToTargetType(Type inputType, Type targetType) {
  if (auto tensorType = dyn_cast<TensorType>(inputType)) {
    return RankedTensorType::get(tensorType.getShape(), targetType);
  } else if (auto memrefType = dyn_cast<MemRefType>(inputType)) {
    return MemRefType::get(memrefType.getShape(), targetType,
                           memrefType.getLayout(), memrefType.getMemorySpace());
  } else if (isa<mlir::FloatType>(inputType)) {
    return targetType;
  }
  // Return null type to indicate failure - caller should handle gracefully
  return Type();
}

// Returns true if its a generic op with desired op (extf/truncf) followed by
// yield.
bool isLinalgGenericWithUnary(
    Operation *op, llvm::function_ref<bool(Operation &)> isDesiredOp) {
  if (!isa<linalg::GenericOp>(op))
    return false;

  auto genericOp = dyn_cast<linalg::GenericOp>(op);
  if (!genericOp || genericOp.getNumResults() != 1)
    return false;

  // Check that there is exactly one region and has exactly one block
  if (genericOp->getRegions().size() != 1 ||
      genericOp.getRegion().getBlocks().size() != 1)
    return false;

  Block *body = genericOp.getBody();
  if (!body || body->getOperations().size() != 2)
    return false;

  auto &ops = body->getOperations();
  return isa<linalg::YieldOp>(ops.back()) && isDesiredOp(ops.front());
}

bool isLinalgGenericWithExtF(Operation *op) {
  return isLinalgGenericWithUnary(
      op, [](Operation &firstOp) { return isa<arith::ExtFOp>(firstOp); });
}

bool isLinalgGenericWithTruncF(Operation *op) {
  return isLinalgGenericWithUnary(
      op, [](Operation &firstOp) { return isa<arith::TruncFOp>(firstOp); });
}

// --------- Helper functions to Insert Extf/Truncf Operations --------- //
std::pair<Type, Value> createOutputTypeAndValue(Location loc, Value input,
                                                Type convertToType,
                                                IRRewriter &rewriter) {
  Type inputType = input.getType();
  if (auto tensorType = dyn_cast<TensorType>(inputType)) {
    auto outputType =
        RankedTensorType::get(tensorType.getShape(), convertToType);
    auto emptyTensor = tensor::EmptyOp::create(
        rewriter, loc, outputType.getShape(), outputType.getElementType());
    return {outputType, emptyTensor};
  } else if (auto memrefType = dyn_cast<MemRefType>(inputType)) {
    auto outputType =
        MemRefType::get(memrefType.getShape(), convertToType,
                        memrefType.getLayout(), memrefType.getMemorySpace());
    Value output = memref::AllocOp::create(rewriter, loc, outputType);
    return {outputType, output};
  }
  // Return nulls to indicate unsupported type - caller should handle gracefully
  return {Type(), Value()};
}

// Insert linalg.generic op which does extf/truncf conversion
Value insertConversionOp(Location loc, Value input, Type convertToType,
                         bool isExtf, IRRewriter &rewriter) {
  OpBuilder::InsertionGuard guard(rewriter);
  // Create output type and value
  auto [outputType, output] =
      createOutputTypeAndValue(loc, input, convertToType, rewriter);

  // Check if creation failed
  if (!outputType || !output)
    return Value();

  // Get rank
  int64_t rank = 0;
  if (auto tensorType = dyn_cast<TensorType>(input.getType()))
    rank = tensorType.getRank();
  else if (auto memrefType = dyn_cast<MemRefType>(input.getType()))
    rank = memrefType.getRank();

  // Create indexing maps and iterator typefor the generic op
  SmallVector<AffineMap> indexingMaps;
  indexingMaps.push_back(
      AffineMap::getMultiDimIdentityMap(rank, rewriter.getContext()));
  indexingMaps.push_back(
      AffineMap::getMultiDimIdentityMap(rank, rewriter.getContext()));
  SmallVector<utils::IteratorType> iteratorTypes(rank,
                                                 utils::IteratorType::parallel);

  // create linalg.generic op
  auto genericOp =
      linalg::GenericOp::create(rewriter, loc,
                                /*resultTensorTypes=*/TypeRange{outputType},
                                /*inputs=*/ValueRange{input},
                                /*outputs=*/ValueRange{output},
                                /*indexingMaps=*/indexingMaps,
                                /*iteratorTypes=*/iteratorTypes);

  // Build the body
  Type inputElemType = getScalarOrElementType(input.getType());
  Type outputElemType = getScalarOrElementType(outputType);
  Block *body = rewriter.createBlock(
      &genericOp.getRegion(), genericOp.getRegion().begin(),
      {inputElemType, outputElemType}, {loc, loc});

  rewriter.setInsertionPointToEnd(body);

  // Create extf and yield operations in the region
  Value convertedVal =
      isExtf ? arith::ExtFOp ::create(rewriter, loc, outputElemType,
                                      body->getArgument(0))
                   .getResult()
             : arith::TruncFOp ::create(rewriter, loc, outputElemType,
                                        body->getArgument(0))
                   .getResult();
  linalg::YieldOp::create(rewriter, loc, convertedVal);

  return genericOp.getResult(0);
}

// --------- Insert Extf Operations --------- //
/*
Once the operation is converted to fp16 datatype (target type),
then a new linalg.generic consisting of extf is inserted at the end of the
operation.
Ensuring that consumer op gets the expected data-type and the IR is valid.
Incase of scalar data type arith::ExtfOp are inserted.

For example :------------------------------------
%fp32_out = linalg.transpose ins(%input : tensor<1024x64xf32)
                              outs(%output : tensor<64x1024xf32>)
                              permutation = [1, 0]

converts to :------------------------------------
%fp16_out = linalg.transpose ins(%input : tensor<1024x64xf16>)
                              outs(%transposed : tensor<64x1024xf16>)
                              permutation = [1, 0]
%fp32_out_extf = linalg.generic {indexing_maps =
                                    [affine_map<(d0, d1) -> (d0, d1)>,
                                    affine_map<(d0, d1) -> (d0, d1)>],
                                    iterator_types = ["parallel","parallel"]}
                                    ins(%fp16_out : tensor<64x1024xf16>)
                                    outs(%output : tensor<64x1024xf32>) {
                    ^bb0(%in: f16, %out: f32):
                      %1 = arith.extf %in : f16 to f32
                      linalg.yield %1 : f32
                  } -> tensor<64x1024xf32>
*/

// Returns created op : `arith.extf : targetType to oldType`
// Returns Value() (null) if conversion is not possible.
Value createExtfOp(Location loc, Value input, Type convertToType,
                   IRRewriter &rewriter) {
  OpBuilder::InsertionGuard guard(rewriter);
  if (isa<FloatType>(input.getType())) {
    return arith::ExtFOp::create(rewriter, loc, convertToType, input)
        .getResult();
  } else if (isa<TensorType, MemRefType>(input.getType())) {
    return insertConversionOp(loc, input, convertToType, true, rewriter);
  }

  // Return null to indicate unsupported type - skip this conversion
  return Value();
}

// Since the consumer op of the current op has not changed its data type,
// hence, after changing the output data type of the current op we should insert
// extf operations to restore the data type and pass to the consumer op.
// Returns Value() (null) if conversion is not possible.
Value callFunctionToCreateExtfOps(Type convertfrom, Type convertto, Value input,
                                  Location loc, IRRewriter &rewriter) {
  Type elementType16 = Float16Type::get(rewriter.getContext());
  Type elementType32 = Float32Type::get(rewriter.getContext());
  Type elementType64 = Float64Type::get(rewriter.getContext());

  Type fromElemType = getScalarOrElementType(convertfrom);
  Type toElemType = getScalarOrElementType(convertto);

  if (fromElemType.isF16() && toElemType.isF32()) {
    return createExtfOp(loc, input, elementType32, rewriter);
  } else if (fromElemType.isF32() && toElemType.isF64()) {
    return createExtfOp(loc, input, elementType64, rewriter);
  } else if (fromElemType.isF16() && toElemType.isF64()) {
    // Insert two extf ops since the consumer expects fp64 and the current ops
    // output is fp16. extf(fp16 --> fp32) followed by extf(fp32 --> fp64)
    auto newExtfInserted = createExtfOp(loc, input, elementType32, rewriter);
    if (!newExtfInserted)
      return Value();
    return createExtfOp(loc, newExtfInserted, elementType64, rewriter);
  }
  // Return null to indicate unsupported conversion - skip this optimization
  return Value();
}

// --------- Insert Truncf Operations --------- //
/*
While converting an op to fp16, if the input does not come from an operation
that was computed in fp16, then for a tensor/memref data type a new
linalg.generic consisting of truncf is inserted which convert the input to fp16
datatype. This is to ensure that the IR is valid when an is converted to fp16.
Incase of scalar operands arith::TruncfOp are inserted.
The matmul operation is not converted to fp16 datatype, assuming that it may
lead to accuracy failure.

For example :------------------------------------
%fp32_inp = linalg.matmul ins(%1, %2 : tensor<1024x64xf32>, tensor<64x64xf32>)
                          outs(%3 : tensor<1024x64xf32>) ->tensor<1024x64xf32>
%output = tensor.empty() : tensor<64x1024xf32>
%fp32_out = linalg.transpose ins(%fp32_inp : tensor<1024x64xf32>)
                              outs(%output : tensor<64x1024xf32>)
                              permutation = [1, 0]

converts to :------------------------------------
%fp32_inp = linalg.matmul ins(%1, %2 : tensor<1024x64xf32>, tensor<64x64xf32>)
                            outs(%3 : tensor<1024x64xf32>)
                          -> tensor<1024x64xf32>
%tmp = tensor.empty() : tensor<1024x64xf16>
%fp16_truncf_out = linalg.generic {indexing_maps =
                                [affine_map<(d0, d1) -> (d0, d1)>,
                                affine_map<(d0, d1) -> (d0,d1)>],
                                iterator_types = ["parallel", "parallel"]}
                                ins(%fp32_inp :tensor<1024x64xf32>)
                                outs(%tmp : tensor<1024x64xf16>) {
                      ^bb0(%in: f32, %out: f16):
                        %1 = arith.truncf %in : f32 to f16
                        linalg.yield %1 : f16
                    } -> tensor<1024x64xf16>
%output = tensor.empty() : tensor<64x1024xf16>
%fp16_out = linalg.transpose ins(%fp16_truncf_out : tensor<1024x64xf16>)
                              outs(%output : tensor<64x1024xf16>)
                              permutation = [1, 0]
*/

// Returns created op : `arith.truncf : input to targetType`
// Returns Value() (null) if conversion is not possible.
Value createTruncfOp(Location loc, Value input, Type convertToType,
                     IRRewriter &rewriter) {
  OpBuilder::InsertionGuard guard(rewriter);
  if (isa<FloatType>(input.getType())) {
    return arith::TruncFOp::create(rewriter, loc, convertToType, input)
        .getResult();
  } else if (isa<TensorType, MemRefType>(input.getType())) {
    return insertConversionOp(loc, input, convertToType, false, rewriter);
  }
  // Return null to indicate unsupported type - skip this conversion
  return Value();
}

// This function is called when we intend to convert the operands of the current
// operation to fp16. This function tries to insert the truncf operations, to
// convert the input to fp16 datatype and update the current ops input operands.
// Returns Value() (null) if conversion is not possible.
Value callFunctionToCreateTruncfOps(Type convertfrom, Type convertto,
                                    Value input, Location loc,
                                    IRRewriter &rewriter) {
  Type elementType16 = Float16Type::get(rewriter.getContext());
  Type elementType32 = Float32Type::get(rewriter.getContext());
  Type elementType64 = Float64Type::get(rewriter.getContext());

  Type fromElemType = getScalarOrElementType(convertfrom);
  Type toElemType = getScalarOrElementType(convertto);

  if (fromElemType.isF32() && toElemType.isF16()) {
    return createTruncfOp(loc, input, elementType16, rewriter);
  } else if (fromElemType.isF64() && toElemType.isF32()) {
    return createTruncfOp(loc, input, elementType32, rewriter);
  } else if (fromElemType.isF64() && toElemType.isF16()) {
    // Insert two truncf ops since the current op expects fp16 input and the
    // defining ops output is fp64. truncf(fp64 --> fp32) followed by
    // truncf(fp32 --> fp16)
    auto asfp32 = createTruncfOp(loc, input, elementType32, rewriter);
    if (!asfp32)
      return Value();
    return createTruncfOp(loc, asfp32, elementType16, rewriter);
  }
  // Return null to indicate unsupported conversion - skip this optimization
  return Value();
}

// --------- Insert Extf Operations After Cloned New Op --------- //
// For any fp16 converted op add a wrapper function (extf),
// to convert it back to old data type and use in the following consumer ops.
// Returns empty vector if conversion fails - caller should handle gracefully.
SmallVector<Value> InsertExtfOpsAfterClonedNewOp(Operation *op,
                                                 Operation *clonedOp,
                                                 Location loc,
                                                 IRRewriter &rewriter) {
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointAfter(clonedOp);

  SmallVector<Value> finalResults;
  finalResults.reserve(clonedOp->getNumResults());

  for (auto [oldResult, clonedOpResult] :
       llvm::zip(op->getResults(), clonedOp->getResults())) {
    Type oldTy = oldResult.getType();
    Type clonedTy = clonedOpResult.getType();

    if (oldTy != clonedTy) {
      Value extfResult = callFunctionToCreateExtfOps(
          clonedTy, oldTy, clonedOpResult, loc, rewriter);

      // Conversion failed - return empty to indicate failure
      if (!extfResult)
        return SmallVector<Value>();

      finalResults.push_back(extfResult);
    } else {
      finalResults.push_back(clonedOpResult);
    }
  }
  return finalResults;
}

void runFp16Conversion(Operation *operation, IRRewriter &rewriter);

//===---------------------------------------------------------------------===//
// Helper Functions for Region Conversion
//===---------------------------------------------------------------------===//

// Creates new argument types by converting to targetType where applicable
SmallVector<Type> createNewArgTypes(Block &oldBlock, Type targetType) {
  SmallVector<Type> newArgTypes;
  newArgTypes.reserve(oldBlock.getNumArguments());
  for (BlockArgument oldArg : oldBlock.getArguments()) {
    Type oldType = oldArg.getType();
    if (canDowncastToTargetType(oldType, targetType)) {
      newArgTypes.push_back(downcastToTargetType(oldType, targetType));
    } else {
      newArgTypes.push_back(oldType);
    }
  }
  return newArgTypes;
}

// Creates location vector for new block arguments
SmallVector<Location> createNewArgLocations(Block &oldBlock) {
  SmallVector<Location> newArgLocs;
  newArgLocs.reserve(oldBlock.getNumArguments());
  for (BlockArgument oldArg : oldBlock.getArguments()) {
    newArgLocs.push_back(oldArg.getLoc());
  }
  return newArgLocs;
}

// Since the block arguments are now of targetType,
// we need to map them to the operations inside the cloned block.
// This results into introducing extf ops which converts
// the argument to the old data type and pass it to the consumers,
// to ensure that the IR is valid.
// Returns false if conversion fails.
bool mapBlockArguments(Block &oldBlock, Block *clonedBlock,
                       mlir::IRMapping &mapper, Location loc,
                       IRRewriter &rewriter) {
  for (auto [oldArg, clonedArg] :
       llvm::zip(oldBlock.getArguments(), clonedBlock->getArguments())) {
    if (oldArg.getType() != clonedArg.getType()) {
      // Insert extf to convert clonedArg back to old type for consumers
      auto newClonedArg = callFunctionToCreateExtfOps(
          clonedArg.getType(), oldArg.getType(), clonedArg, loc, rewriter);
      // Conversion failed - cannot proceed with region conversion
      if (!newClonedArg)
        return false;
      mapper.map(oldArg, newClonedArg);
    } else {
      mapper.map(oldArg, clonedArg);
    }
  }
  return true;
}

// Clones all operations from old block to cloned block using the mapper
void cloneBlockOperations(Block &oldBlock, Block *clonedBlock,
                          mlir::IRMapping &mapper, IRRewriter &rewriter) {
  for (Operation &oldInnerOp : oldBlock) {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToEnd(clonedBlock);
    (void)rewriter.clone(oldInnerOp, mapper);
  }
}

// Converts yield operands to target type by inserting truncf ops
// Returns empty vector if conversion fails.
SmallVector<Value> convertYieldOperands(linalg::YieldOp yieldOp,
                                        Type targetType, Location loc,
                                        IRRewriter &rewriter) {
  SmallVector<Value> newOperands;
  newOperands.reserve(yieldOp.getNumOperands());
  for (auto oldOperand : yieldOp.getOperands()) {
    Type oldType = oldOperand.getType();
    if (canDowncastToTargetType(oldType, targetType)) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPoint(yieldOp);
      Value converted = callFunctionToCreateTruncfOps(
          oldOperand.getType(), targetType, oldOperand, loc, rewriter);

      // Conversion failed - return empty to indicate failure
      if (!converted)
        return SmallVector<Value>();

      newOperands.push_back(converted);
    } else {
      newOperands.push_back(oldOperand);
    }
  }
  return newOperands;
}

// Converts operations in the cloned block to target type.
// Returns false if conversion fails.
bool convertClonedBlockOperations(Block *clonedBlock, Type targetType,
                                  Location loc, IRRewriter &rewriter) {
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(clonedBlock);

  for (Operation &newInnerOp : *clonedBlock) {
    if (auto yieldOp = dyn_cast<linalg::YieldOp>(newInnerOp)) {
      auto newOperands =
          convertYieldOperands(yieldOp, targetType, loc, rewriter);

      // Conversion failed - return failure
      if (newOperands.empty() && yieldOp.getNumOperands() > 0)
        return false;

      // Update operands only if conversion succeeded
      if (!newOperands.empty())
        newInnerOp.setOperands(newOperands);
    } else {
      // Recursively convert nested operations
      runFp16Conversion(&newInnerOp, rewriter);
    }
  }

  return true;
}

// Processes a single block:
// creates new block, maps arguments, clones ops, converts types.
// Returns false if conversion fails.
bool processBlock(Block &oldBlock, Region &clonedRegion, Type targetType,
                  Location loc, IRRewriter &rewriter) {
  // 1. Create new argument types with target types
  SmallVector<Type> newArgTypes = createNewArgTypes(oldBlock, targetType);
  SmallVector<Location> newArgLocs = createNewArgLocations(oldBlock);

  // 2. Create new block in the cloned region
  Block *clonedBlock = rewriter.createBlock(&clonedRegion, clonedRegion.end(),
                                            newArgTypes, newArgLocs);

  // 3. Map old arguments to new arguments (with extf ops where needed)
  mlir::IRMapping mapper;
  bool mappingSuccess =
      mapBlockArguments(oldBlock, clonedBlock, mapper, loc, rewriter);
  if (!mappingSuccess) {
    rewriter.eraseBlock(clonedBlock);
    return false;
  }

  // 4. Clone all operations from old block to new block
  cloneBlockOperations(oldBlock, clonedBlock, mapper, rewriter);

  // 5. Convert operations in the cloned block to target type
  bool clonedBlockOpsConversionSuccess =
      convertClonedBlockOperations(clonedBlock, targetType, loc, rewriter);
  if (!clonedBlockOpsConversionSuccess) {
    rewriter.eraseBlock(clonedBlock);
    return false;
  }

  return true;
}

// After the op is cloned,
// convert the datatype of the region and the block operations.
// Returns false if conversion fails.
bool convertRegionsToTargetType(Operation *oldOp, Operation *clonedOp,
                                Type targetType, IRRewriter &rewriter) {
  if (oldOp->getNumRegions() == 0)
    return true;
  Location loc = clonedOp->getLoc();
  // Iterate over each region of the oldOp and clonedOp
  for (auto [oldRegion, clonedRegion] :
       llvm::zip(oldOp->getRegions(), clonedOp->getRegions())) {
    if (oldRegion.empty())
      continue;
    // Process each block in the region
    for (Block &oldBlock : oldRegion) {
      bool processBlockSuccess =
          processBlock(oldBlock, clonedRegion, targetType, loc, rewriter);
      if (!processBlockSuccess) {
        return false;
      }
    }
  }

  return true;
}

//===---------------------------------------------------------------------===//
// Rewrite Patterns for Different Dialects
//===---------------------------------------------------------------------===//
WalkResult rewriteOp(Operation *op, Type targetType, IRRewriter &rewriter) {
  OpBuilder::InsertionGuard guard(rewriter);
  Location loc = op->getLoc();
  rewriter.setInsertionPoint(op);

  // 1. Convert operands: f32/fp64 -> f16 (where applicable)
  SmallVector<Value> newOperands;
  newOperands.reserve(op->getNumOperands());
  for (auto operand : op->getOperands()) {
    if (canDowncastToTargetType(operand.getType(), targetType)) {
      Value converted = callFunctionToCreateTruncfOps(
          operand.getType(), targetType, operand, loc, rewriter);
      if (!converted)
        return WalkResult::skip();
      newOperands.push_back(converted);
    } else {
      newOperands.push_back(operand);
    }
  }

  // 2. Compute new result types (convert f32/fp64 to f16)
  SmallVector<Type> newResultTypes;
  for (auto resultType : op->getResultTypes()) {
    if (canDowncastToTargetType(resultType, targetType)) {
      Type newType = downcastToTargetType(resultType, targetType);
      newResultTypes.push_back(newType);
    } else {
      newResultTypes.push_back(resultType);
    }
  }

  // 3. Clone the Linalg op with fp16 operands/results
  OperationState state(loc, op->getName().getStringRef());
  state.addOperands(newOperands);
  state.addTypes(newResultTypes);
  state.addAttributes(op->getAttrs());

  // Handle regions if the operation has any
  for (unsigned i = 0; i < op->getNumRegions(); ++i) {
    state.addRegion();
  }

  Operation *clonedOp = rewriter.create(state);

  // 4. Clone the region in linalg op
  // If the op had regions, clone them into the new op and
  // convert their internal ops to targetType as needed.
  if (op->getNumRegions() > 0) {
    bool regionConversionSuccess =
        convertRegionsToTargetType(op, clonedOp, targetType, rewriter);
    if (!regionConversionSuccess) {
      // Region conversion failed - clean up and skip this op
      rewriter.eraseOp(clonedOp);
      return WalkResult::skip();
    }
  }

  // 5. Wrap fp16 results with extf so consumers still see old datatype
  SmallVector<Value> finalResults =
      InsertExtfOpsAfterClonedNewOp(op, clonedOp, loc, rewriter);

  // Conversion failed - clean up and skip this op
  if (finalResults.empty() && clonedOp->getNumResults() > 0) {
    rewriter.eraseOp(clonedOp);
    return WalkResult::skip();
  }

  op->replaceAllUsesWith(finalResults);

  // Since the ops within the linalg.generic op will be already converted via
  // recursion hence, we should return "skip" to avoid walking on the ops inside
  // the old version of the linalg.generic op. As "Advance" will iterate over
  // the ops in the regions of the old version of linalg.generic op.
  return WalkResult::skip();
}

// Arith dialect ops
WalkResult rewriteArithOp(Operation *op, Type targetType,
                          IRRewriter &rewriter) {
  // Skip truncf and extf ops
  if (isa<arith::TruncFOp, arith::ExtFOp>(op)) {
    return WalkResult::advance();
  }

  // This will skip the op if it does not have a single fp32 or fp64 operand
  // which can be downscasted to fp16
  if (!canDowncastToTargetType(op, targetType))
    return WalkResult::skip();

  OpBuilder::InsertionGuard guard(rewriter);
  Location loc = op->getLoc();
  rewriter.setInsertionPoint(op);

  // Handle Constant op separately to ensure that,
  // along with data type the value attributes are changed.
  if (auto cst = dyn_cast<arith::ConstantOp>(op)) {
    OpBuilder::InsertionGuard guard(rewriter);
    auto valueAttr = dyn_cast<FloatAttr>(cst.getValue());
    if (!valueAttr)
      return WalkResult::skip();

    Location loc = cst.getLoc();
    rewriter.setInsertionPoint(cst);

    // Initialize the value and attribute for the new constant op
    // and insert the new constant op.
    double val = valueAttr.getValueAsDouble();
    auto targetAttr = mlir::FloatAttr::get(targetType, val);
    auto newCst =
        arith::ConstantOp::create(rewriter, loc, targetType, targetAttr);

    SmallVector<Value> finalResults =
        InsertExtfOpsAfterClonedNewOp(op, newCst, loc, rewriter);

    // Conversion failed - skip this constant
    if (finalResults.empty() && newCst->getNumResults() > 0) {
      rewriter.eraseOp(newCst);
      return WalkResult::skip();
    }

    cst->replaceAllUsesWith(finalResults);
    return WalkResult::skip();
  }
  return rewriteOp(op, targetType, rewriter);
};

// Linalg dialect ops
WalkResult rewriteLinalgOp(Operation *op, Type targetType,
                           IRRewriter &rewriter) {
  // Skip linalg.generic ops with extf/truncf inside
  if (isLinalgGenericWithExtF(op) || isLinalgGenericWithTruncF(op))
    return WalkResult::skip();

  // This will move to the next op if the current op does not have
  // a single fp32 or fp64 operand which can be downscasted to fp16
  if (!canDowncastToTargetType(op, targetType))
    return WalkResult::skip();

  return rewriteOp(op, targetType, rewriter);
}

// Tensor dialect ops
WalkResult rewriteTensorOp(Operation *op, Type targetType,
                           IRRewriter &rewriter) {
  // This will skip the op if it does not have a single fp32 or fp64 operand
  // which can be downscasted to fp16
  if (!canDowncastToTargetType(op, targetType))
    return WalkResult::skip();

  return rewriteOp(op, targetType, rewriter);
};

//===---------------------------------------------------------------------===//
// fp32/fp64 to fp16 Conversion Pass
//===---------------------------------------------------------------------===//

// Sets target datatype type in which all the operations are to be converted.
Type getTargetDataType(Operation *op) {
  return Float16Type::get(op->getContext());
}

// This function calls the respective rewrite functions for each dialect
void runFp16Conversion(Operation *operation, IRRewriter &rewriter) {
  operation->walk<WalkOrder::PreOrder>([&](Operation *op_walk) {
    if (opToExclude(op_walk))
      return WalkResult::skip();

    Type targetType = getTargetDataType(op_walk);
    Dialect *dialect = op_walk->getDialect();

    if (isa<linalg::LinalgDialect>(dialect))
      return rewriteLinalgOp(op_walk, targetType, rewriter);
    else if (isa<arith::ArithDialect>(dialect))
      return rewriteArithOp(op_walk, targetType, rewriter);
    else if (isa<tensor::TensorDialect>(dialect))
      return rewriteTensorOp(op_walk, targetType, rewriter);

    return WalkResult::advance();
  });
}

struct ConversionToFp16Pass
    : public ::impl::ConversionToFp16Base<ConversionToFp16Pass> {
  void runOnOperation() override {
    Operation *operation = getOperation();
    auto funcOp = dyn_cast<func::FuncOp>(operation);
    if (!funcOp)
      return;

    // Check if any input or output is FP16
    auto isFp16 = [](Type type) {
      return getScalarOrElementType(type).isF16();
    };

    bool hasFp16Input = llvm::any_of(funcOp.getArgumentTypes(), isFp16);
    bool hasFp16Output = llvm::any_of(funcOp.getResultTypes(), isFp16);

    // Run conversion only for FP16 models
    if (hasFp16Input || hasFp16Output) {
      mlir::IRRewriter rewriter(funcOp.getContext());
      runFp16Conversion(funcOp, rewriter);
    }
  }
};

} // namespace

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
hexagon::createConversionToFp16Pass() {
  return std::make_unique<ConversionToFp16Pass>();
}
