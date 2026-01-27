//===- EnableHexagonRoutinesPass.cpp - replace llvm ops with custom funcs
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// This file replaces llvm ops with custom hand optimized hvx routines.
// This is useful for functions like exp which get scalarized if left as is.
// The pass needs to run after convert-to-llvm pass.
//
//===----------------------------------------------------------------------===//

#include "hexagon/Conversion/LinalgToLLVM/Common.h"
#include "hexagon/Transforms/Transforms.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/TypeSwitch.h"
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#define DEBUG_TYPE "hexagon-llvm-use-hexagon-routines"

#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define DBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
using namespace hexagon;

#define GEN_PASS_CLASSES
#include "hexagon/Transforms/Passes.h.inc"

namespace {

// Get a SymbolRefAttr containing the fnName of the LLVM::FuncOp.
// If the function does not exist, insert a declaration, with a type if
// provided.
static FlatSymbolRefAttr
getOrCreateFunc(Operation *op, std::string fnName, PatternRewriter &rewriter,
                LLVM::LLVMFunctionType fnType = LLVM::LLVMFunctionType()) {
  // fnName is a dynamic std::string, unique it via a SymbolRefAttr.
  FlatSymbolRefAttr fnNameAttr =
      SymbolRefAttr::get(rewriter.getContext(), fnName);
  auto module = op->getParentOfType<ModuleOp>();
  if (module.lookupSymbol(fnNameAttr.getAttr()))
    return fnNameAttr;

  if (fnType == LLVM::LLVMFunctionType()) {
    SmallVector<Type, 8> operandTypes;
    llvm::append_range(operandTypes, op->getOperandTypes());
    assert(op->getNumResults() == 1);
    fnType =
        LLVM::LLVMFunctionType::get(op->getResultTypes().front(), operandTypes);
  }

  OpBuilder::InsertionGuard guard(rewriter);
  // Insert before module terminator.
  rewriter.setInsertionPoint(module.getBody(),
                             std::prev(module.getBody()->end()));
  LLVM::LLVMFuncOp funcOp = rewriter.create<LLVM::LLVMFuncOp>(
      op->getLoc(), fnNameAttr.getValue(), fnType);
  funcOp.setPrivate();
  return fnNameAttr;
}

// Map of LLVM ops to Custom routine names.

static FailureOr<std::string> LLVMOpToHexagonRountineName(const Operation *op) {
  return TypeSwitch<const Operation *, FailureOr<std::string>>(op)
      .Case<LLVM::ExpOp>(
          [](auto ty) { return FailureOr<std::string>(std::string("exp")); })
      .Case<LLVM::Exp2Op>(
          [](auto ty) { return FailureOr<std::string>(std::string("pow")); })
      .Case<LLVM::SinOp>(
          [](auto ty) { return FailureOr<std::string>(std::string("sin")); })
      .Case<LLVM::CosOp>(
          [](auto ty) { return FailureOr<std::string>(std::string("cos")); })
      .Case<LLVM::TanOp>(
          [](auto ty) { return FailureOr<std::string>(std::string("tan")); })
      .Case<LLVM::TanhOp>(
          [](auto ty) { return FailureOr<std::string>(std::string("tanh")); })
      .Case<LLVM::ASinOp>(
          [](auto ty) { return FailureOr<std::string>(std::string("asin")); })
      .Case<LLVM::ACosOp>(
          [](auto ty) { return FailureOr<std::string>(std::string("acos")); })
      .Case<LLVM::ATanOp>(
          [](auto ty) { return FailureOr<std::string>(std::string("atan")); })
      .Case<LLVM::SqrtOp>(
          [](auto ty) { return FailureOr<std::string>(std::string("sqrt")); })
      .Case<LLVM::FFloorOp>(
          [](auto ty) { return FailureOr<std::string>(std::string("floor")); })
      .Case<LLVM::FCeilOp>(
          [](auto ty) { return FailureOr<std::string>(std::string("ceil")); })
      .Default([](auto ty) { return FailureOr<std::string>(failure()); });
}

// Generate suffix for an operand type.
static std::string getSuffix(Type t) {
  std::string suffix = "";
  if (isa<VectorType>(t)) {
    suffix += "v";
  }
  Type elemType = getElementType(t);
  // Elem Type can be only F16 and F32 here
  if (elemType.isF16())
    suffix += "hf";
  else if (elemType.isF32())
    suffix += "f";
  else
    llvm_unreachable("Expecting only F16 or F32 type!");
  return suffix;
}

// Generate suffix for each operand type in the opertion.
static std::string getSuffix(Operation *op) {
  std::string suffix = "";
  for (const auto &operandType : op->getOperandTypes())
    suffix += getSuffix(operandType);
  return suffix;
}

// Generate function name using the LLVM ops to routine name map and the
// operand types in the operation.
static FailureOr<std::string> getFnName(Operation *op) {
  FailureOr<std::string> fnName = LLVMOpToHexagonRountineName(op);
  if (failed(fnName))
    return failure();
  return "_hexagon_runtime_" + fnName.value() + "__" + getSuffix(op);
}

static std::optional<std::pair<Type, unsigned>>
getVectorElementInfo(const Type &type) {

  if (auto vecTy = dyn_cast<VectorType>(type)) {
    return std::make_pair(vecTy.getElementType(), vecTy.getNumElements());
  } else {
    return std::nullopt;
  }
}

static bool isNativeVectorFactorFloat(const Type &type) {

  auto elementInfo = getVectorElementInfo(type);
  if (elementInfo) {
    Type elemType = elementInfo->first;
    unsigned elemCount = elementInfo->second;
    if ((elemType.isF16() && elemCount % 64 == 0) ||
        (elemType.isF32() && elemCount % 32 == 0))
      return true;
    return false;
  }
  return false;
}

// Check if all operands and results are Hexagon native vectors( or have size
// which is a factor of native vector width) with f16 or f32 element types.
static bool allNativeVectorFactorTypes(Operation *op) {
  bool nativeVectorOperands = true;
  bool nativeVectorResults = true;
  for (const auto &operandType : op->getOperandTypes())
    nativeVectorOperands &= isNativeVectorFactorFloat(operandType);
  for (const auto &resultType : op->getResultTypes())
    nativeVectorResults &= isNativeVectorFactorFloat(resultType);
  return nativeVectorOperands && nativeVectorResults;
}

Value replaceWithHexRTCall(Operation *op, PatternRewriter &rewriter,
                           Value operand1, Type nativeVectorType,
                           std::string fnName, Type elemType) {

  Value Result = nullptr;
  if (isa<LLVM::Exp2Op>(op)) {
    // Logic to handle Exp2Op

    // Before Transformation:
    //  %0 = llvm.intr.exp2(%arg0) : (vector<64xf16>) -> vector<64xf16>
    //
    // After Transformation:
    //  %0 = llvm.mlir.constant(dense<2.000000e+00> : vector<64xf16>) :
    //       vector<64xf16>
    //  %1 = llvm.bitcast %0 : vector<64xf16> to vector<32xi32>
    //  %2 = llvm.bitcast %arg0 : vector<64xf16> to vector<32xi32>
    //  %3 = llvm.call @_hexagon_runtime_pow__vhf(%1, %2) : (vector<32xi32>,
    //       vector<32xi32>) -> vector<32xi32>
    //  %4 = llvm.bitcast %3 : vector<32xi32> to vector<64xf16>
    //
    //  Note: Although bitcast f16 to i32 sounds semantically wrong as
    //  conversion does not happen, this is the expectation/interpretation
    //  from the library side.

    auto shapedType = mlir::cast<mlir::ShapedType>(nativeVectorType);
    auto denseAttr = DenseFPElementsAttr::get(
        shapedType, FloatAttr::get(elemType, 2.0f).getValue());
    auto splatVector = rewriter.create<LLVM::ConstantOp>(
        op->getLoc(), nativeVectorType, denseAttr);

    auto intVectorType = mlir::VectorType::get({32}, rewriter.getI32Type());
    // Bitcast the splat vector to <32xi32>
    auto bitcastSplatVector = rewriter.create<LLVM::BitcastOp>(
        op->getLoc(), intVectorType, splatVector);

    // Create the function type with two operands
    auto fnType = LLVM::LLVMFunctionType::get(intVectorType,
                                              {intVectorType, intVectorType});
    auto fnNameRefAttr = getOrCreateFunc(op, fnName, rewriter, fnType);

    // Bitcast the input operand to <32xi32>
    auto bitcastInput =
        rewriter.create<LLVM::BitcastOp>(op->getLoc(), intVectorType, operand1);

    // Create the call operation
    auto callOp = rewriter.create<LLVM::CallOp>(
        op->getLoc(), intVectorType, fnNameRefAttr,
        ValueRange{bitcastSplatVector, bitcastInput});

    // Bitcast the result back to natrive vector type
    auto bitcastResult = rewriter.create<LLVM::BitcastOp>(
        op->getLoc(), nativeVectorType, callOp.getResults());
    Result = bitcastResult.getResult();
  } else {
    auto intVectorType = mlir::VectorType::get({32}, rewriter.getI32Type());
    // Create the function type with single operand
    auto fnType = LLVM::LLVMFunctionType::get(intVectorType, {intVectorType});
    auto fnNameRefAttr = getOrCreateFunc(op, fnName, rewriter, fnType);

    // Bitcast the input operand to <32xi32>
    auto bitcastInput =
        rewriter.create<LLVM::BitcastOp>(op->getLoc(), intVectorType, operand1);

    // Create the call operation
    auto callOp = rewriter.create<LLVM::CallOp>(
        op->getLoc(), intVectorType, fnNameRefAttr, ValueRange{bitcastInput});

    // Bitcast the result back to natrive vector type
    auto bitcastResult = rewriter.create<LLVM::BitcastOp>(
        op->getLoc(), nativeVectorType, callOp.getResults());
    Result = bitcastResult.getResult();
  }
  return Result;
}

Value multiVectorHandle(Operation *op, PatternRewriter &rewriter,
                        Value operand1, Value destOp, Type nativeVectorType,
                        std::string fnName, Type elemType, int idx) {
  Value bitCastOp = replaceWithHexRTCall(op, rewriter, operand1,
                                         nativeVectorType, fnName, elemType);

  Value insertOp = rewriter.create<LLVM::vector_insert>(op->getLoc(), destOp,
                                                        bitCastOp, idx);

  return insertOp;
}

Value singleVectorHandle(Operation *op, PatternRewriter &rewriter,
                         Type nativeVectorType, std::string fnName,
                         Type elemType) {
  Value bitCastOp = replaceWithHexRTCall(op, rewriter, op->getOperand(0),
                                         nativeVectorType, fnName, elemType);
  return bitCastOp;
}

// Helper to check if a value is constant 1.0 or a splat of 1.0
static bool isConstantOne(Value val) {
  // Check for scalar constant
  if (auto constOp = val.getDefiningOp<LLVM::ConstantOp>()) {
    if (auto floatAttr = dyn_cast<FloatAttr>(constOp.getValue())) {
      return floatAttr.getValueAsDouble() == 1.0;
    }
    // Check for vector splat
    if (auto denseAttr = dyn_cast<DenseElementsAttr>(constOp.getValue())) {
      if (denseAttr.isSplat()) {
        auto splatValue = denseAttr.getSplatValue<APFloat>();
        return splatValue.convertToDouble() == 1.0;
      }
    }
  }
  return false;
}

// Common helper to replace an operation with Hexagon runtime calls
// Handles both single and multi-vector cases
static LogicalResult
replaceOpWithHexagonRuntimeCall(Operation *op, PatternRewriter &rewriter,
                                Value inputOperand, Type resultType,
                                const std::string &fnName) {

  // Get element info
  auto returnTyElementInfo = getVectorElementInfo(resultType);
  if (!returnTyElementInfo)
    return failure();

  Type elemType = returnTyElementInfo->first;
  unsigned elemCount = returnTyElementInfo->second;
  unsigned nativeVectorWidth = elemType.isF16() ? 64 : 32;
  Type nativeVectorType = LLVM::getVectorType(elemType, nativeVectorWidth);

  auto factor = elemCount / nativeVectorWidth;
  Location loc = op->getLoc();

  if (factor > 1) {
    // Handle large vectors by splitting
    LLVM::UndefOp undefOp = rewriter.create<LLVM::UndefOp>(loc, resultType);
    Value destOp = undefOp;

    for (int idx = 0; idx < elemCount; idx += nativeVectorWidth) {
      LLVM::vector_extract extractOp = rewriter.create<LLVM::vector_extract>(
          loc, nativeVectorType, inputOperand, idx);

      Value insertOp =
          multiVectorHandle(op, rewriter, extractOp.getResult(), destOp,
                            nativeVectorType, fnName, elemType, idx);

      destOp = insertOp;
    }

    rewriter.replaceOp(op, destOp);
  } else {
    // Single native vector case
    Value bitCastOp = replaceWithHexRTCall(op, rewriter, inputOperand,
                                           nativeVectorType, fnName, elemType);

    rewriter.replaceOp(op, bitCastOp);
  }

  return success();
}

// Pattern to detect 1.0 / sqrt(x) and replace directly with Hexagon rsqrt
// runtime call
class FDivSqrtToHexagonRsqrtPattern : public OpRewritePattern<LLVM::FDivOp> {
public:
  using OpRewritePattern<LLVM::FDivOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(LLVM::FDivOp fdivOp,
                                PatternRewriter &rewriter) const override {
    // Get the operands of fdiv
    Value numerator = fdivOp.getLhs();
    Value denominator = fdivOp.getRhs();

    // Check if denominator is a sqrt operation
    auto sqrtOp = denominator.getDefiningOp<LLVM::SqrtOp>();
    if (!sqrtOp)
      return failure();

    // Check if numerator is a constant 1.0 (or splat of 1.0 for vectors)
    if (!isConstantOne(numerator))
      return failure();

    // Check if the types are native vector factor types
    if (!isNativeVectorFactorFloat(fdivOp.getType()))
      return failure();

    DBG("Detected 1.0/sqrt(x) pattern, converting directly to Hexagon rsqrt "
        "runtime call");

    // Get the sqrt's input
    Value sqrtInput = sqrtOp.getOperand();
    Type resultType = fdivOp.getType();

    // Get element type to generate function name
    auto elementInfo = getVectorElementInfo(resultType);
    if (!elementInfo)
      return failure();

    Type elemType = elementInfo->first;

    // Generate the rsqrt function name
    std::string fnName = "_hexagon_runtime_rsqrt__v";
    if (elemType.isF16())
      fnName += "hf";
    else if (elemType.isF32())
      fnName += "f";
    else
      return failure();

    // Use common helper to perform the replacement
    if (failed(replaceOpWithHexagonRuntimeCall(fdivOp.getOperation(), rewriter,
                                               sqrtInput, resultType, fnName)))
      return failure();

    // If sqrt has no other uses, erase it
    if (sqrtOp.getResult().use_empty())
      rewriter.eraseOp(sqrtOp);

    return success();
  }
};

template <typename LLVMOp>
class NativeVectorFloatOps : public OpRewritePattern<LLVMOp> {
public:
  using OpRewritePattern<LLVMOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(LLVMOp op,
                                PatternRewriter &rewriter) const override {
    // Check if we have custom Hexagon implementation for this routine.
    // Only process ops with native vector( or size that is factor of native
    // vector width) operands and results and with #results and #operands == 1.
    FailureOr<std::string> fnName = getFnName(op);
    if (failed(fnName) || op->getNumResults() != 1 ||
        op->getNumOperands() != 1 || !allNativeVectorFactorTypes(op)) {
      return failure();
    }

    auto returnTyElementInfo = getVectorElementInfo(op->getResult(0).getType());
    auto operandTyElementInfo =
        getVectorElementInfo(op->getOperand(0).getType());

    if (!returnTyElementInfo || !operandTyElementInfo) {
      return failure();
    } else if (returnTyElementInfo != operandTyElementInfo) {
      DBG(" -> operand and return type of operation is not the same. Aborting");
      return failure();
    }

    // Use common helper to perform the replacement
    return replaceOpWithHexagonRuntimeCall(
        op.getOperation(), rewriter, op->getOperand(0),
        op->getResult(0).getType(), fnName.value());
  }
};

void populateEnableHexagonRoutinesPattern(RewritePatternSet &patterns) {
  // Add pattern to detect 1.0/sqrt(x) and convert directly to Hexagon rsqrt
  patterns.insert<FDivSqrtToHexagonRsqrtPattern>(patterns.getContext());

  // Add patterns to convert intrinsics to Hexagon runtime calls
  patterns.insert<NativeVectorFloatOps<LLVM::ExpOp>>(patterns.getContext());
  patterns.insert<NativeVectorFloatOps<LLVM::Exp2Op>>(patterns.getContext());
  patterns.insert<NativeVectorFloatOps<LLVM::SinOp>>(patterns.getContext());
  patterns.insert<NativeVectorFloatOps<LLVM::ASinOp>>(patterns.getContext());
  patterns.insert<NativeVectorFloatOps<LLVM::CosOp>>(patterns.getContext());
  patterns.insert<NativeVectorFloatOps<LLVM::ACosOp>>(patterns.getContext());
  patterns.insert<NativeVectorFloatOps<LLVM::TanOp>>(patterns.getContext());
  patterns.insert<NativeVectorFloatOps<LLVM::ATanOp>>(patterns.getContext());
  patterns.insert<NativeVectorFloatOps<LLVM::TanhOp>>(patterns.getContext());
  patterns.insert<NativeVectorFloatOps<LLVM::SqrtOp>>(patterns.getContext());
  patterns.insert<NativeVectorFloatOps<LLVM::FFloorOp>>(patterns.getContext());
  patterns.insert<NativeVectorFloatOps<LLVM::FCeilOp>>(patterns.getContext());
}

struct HexagonLLVMEnableHexagonRoutinesPass
    : public HexagonLLVMEnableHexagonRoutinesBase<
          HexagonLLVMEnableHexagonRoutinesPass> {
public:
  void runOnOperation() override {

    auto moduleOp = getOperation();
    RewritePatternSet patterns(moduleOp.getContext());
    populateEnableHexagonRoutinesPattern(patterns);
    if (failed(applyPatternsGreedily(moduleOp, std::move(patterns)))) {
      return signalPassFailure();
    }
  }

  /// Register the dialects that must be loaded in the context before this pass.
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<LLVM::LLVMDialect>();
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
hexagon::createHexagonLLVMEnableHexagonRoutinesPass() {
  return std::make_unique<HexagonLLVMEnableHexagonRoutinesPass>();
}
