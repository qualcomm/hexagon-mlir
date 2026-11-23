//===- LowerLibdevicePass.cpp - lower libdev triton calls -----------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// Patterns to transform a external library call, specifically a libdevice
// call which lowers to a tt.extern_elementwise op, to a QHL qhmath call.
//
//===----------------------------------------------------------------------===//

#include "hexagon/Transforms/Transforms.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/TransformOps/TensorTransformOps.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "triton/Dialect/Triton/IR/Dialect.h"

using namespace mlir;
using namespace hexagon;

#define GEN_PASS_DEF_LOWERLIBDEVICE
#include "hexagon/Transforms/Passes.h.inc"

namespace {

struct LowerLibdevicePattern final
    : public OpRewritePattern<triton::ExternElementwiseOp> {
  LowerLibdevicePattern(MLIRContext *ctx) : OpRewritePattern(ctx) {}

  LogicalResult matchAndRewrite(triton::ExternElementwiseOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    if (op.getSrcs().empty() || op.getSrcs().size() > 2)
      return rewriter.notifyMatchFailure(
          op, "only unary and binary QHL extern_elementwise are supported");
    Value inputValue = op.getSrcs()[0];
    Value outputValue = op.getResult();

    // Mark the parent function so OptimizeAllocationLivenessPass skips it:
    // the libdevice call takes raw llvm.ptr args invisible to SSA liveness.
    if (auto funcOp = op->getParentOfType<func::FuncOp>())
      funcOp->setAttr("hexagon.uses_libdevice", rewriter.getUnitAttr());

    auto inputTy = inputValue.getType();
    auto outputTy = outputValue.getType();
    auto llvmPtrTy = mlir::LLVM::LLVMPointerType::get(rewriter.getContext());
    auto inTensorType = mlir::dyn_cast<RankedTensorType>(inputTy);
    auto outTensorType = mlir::dyn_cast<RankedTensorType>(outputTy);

    bool scalarOperand = false;
    if (!inTensorType && !outTensorType) {
      scalarOperand = true;
      if (!inputTy.isIntOrIndexOrFloat() || inputTy != outputTy)
        return rewriter.notifyMatchFailure(
            op, "scalar extern_elementwise must have matching elemental "
                "input/result types");
    } else if (!inTensorType || !outTensorType) {
      return rewriter.notifyMatchFailure(
          op, "only ranked tensor types or matching scalar elemental types "
              "are supported");
    }

    for (Value extraSrc : op.getSrcs().drop_front()) {
      if (extraSrc.getType() != inputTy)
        return rewriter.notifyMatchFailure(
            op, "all extern_elementwise sources must have identical types");
    }

    int64_t numElems = 0;
    MemRefType memrefType;
    // For scalar libdevice calls the QHL batch symbols (qhmath_hvx_*_af /
    // _ahf) operate on f32 / f16 buffers. Narrow float types must therefore be
    // widened to a supported compute type for the call and converted back
    // afterwards: bf16 -> f32 (_af), fp8 e4m3fn/fnuz -> f16 (_ahf).
    Type scalarComputeTy = outputTy;
    if (scalarOperand) {
      if (auto floatTy = mlir::dyn_cast<FloatType>(outputTy)) {
        if (floatTy.isBF16())
          scalarComputeTy = Float32Type::get(rewriter.getContext());
        else if (isa<Float8E4M3FNType, Float8E4M3FNUZType, Float8E5M2Type,
                     Float8E5M2FNUZType>(floatTy))
          scalarComputeTy = Float16Type::get(rewriter.getContext());
      }
    }
    if (scalarOperand) {
      numElems = 1;
      memrefType = MemRefType::get({1}, scalarComputeTy);
    } else {
      if (inTensorType.getShape() != outTensorType.getShape())
        return rewriter.notifyMatchFailure(
            op, "input and output tensor shapes must match");

      if (!inTensorType.hasStaticShape() || !outTensorType.hasStaticShape())
        return rewriter.notifyMatchFailure(
            op, "only static shaped tensors are supported");

      numElems = inTensorType.getNumElements();
      memrefType = MemRefType::get(outTensorType.getShape(),
                                   outTensorType.getElementType());
    }
    auto i64Ty = IntegerType::get(rewriter.getContext(), 64);

    Value outputMemref = memref::AllocOp::create(rewriter, loc, memrefType);
    Value outputPtrIndex = memref::ExtractAlignedPointerAsIndexOp::create(
        rewriter, loc, outputMemref);
    Value outputPtrInt =
        arith::IndexCastOp::create(rewriter, loc, i64Ty, outputPtrIndex);
    Value outputPtr =
        LLVM::IntToPtrOp::create(rewriter, loc, llvmPtrTy, outputPtrInt)
            .getResult();

    // Materialize one input buffer per source, in operand order.
    SmallVector<Value> inputPtrs;
    inputPtrs.reserve(op.getSrcs().size());
    for (Value src : op.getSrcs()) {
      Value inputMemref;
      if (scalarOperand) {
        inputMemref = memref::AllocOp::create(rewriter, loc, memrefType);
        Value zero = arith::ConstantIndexOp::create(rewriter, loc, 0);
        Value storeVal = src;
        if (scalarComputeTy != inputTy)
          storeVal = arith::ExtFOp::create(rewriter, loc, scalarComputeTy, src);
        memref::StoreOp::create(rewriter, loc, storeVal, inputMemref,
                                ValueRange{zero});
      } else {
        inputMemref =
            bufferization::ToBufferOp::create(rewriter, loc, memrefType, src);
      }
      Value inputPtrIndex = memref::ExtractAlignedPointerAsIndexOp::create(
          rewriter, loc, inputMemref);
      Value inputPtrInt =
          arith::IndexCastOp::create(rewriter, loc, i64Ty, inputPtrIndex);
      inputPtrs.push_back(
          LLVM::IntToPtrOp::create(rewriter, loc, llvmPtrTy, inputPtrInt)
              .getResult());
    }

    // Size should be uint32_t for libdevice functions (QHL)
    auto i32Ty = rewriter.getI32Type();
    Value size =
        arith::ConstantIntOp::create(rewriter, loc, numElems, i32Ty.getWidth());
    if (size.getType() != i32Ty)
      size = arith::IndexCastUIOp::create(rewriter, loc, i32Ty, size);

    SmallVector<Value> inputs(inputPtrs);
    inputs.push_back(outputPtr);
    inputs.push_back(size);
    mlir::ValueRange callOperands(inputs);

    ModuleOp module = op->getParentOfType<ModuleOp>();
    StringRef fName = op.getSymbol();

    SmallVector<Type> llvmArgTys(inputPtrs.size(), llvmPtrTy);
    llvmArgTys.push_back(outputPtr.getType());
    llvmArgTys.push_back(size.getType());
    auto llvmFnTy = LLVM::LLVMFunctionType::get(i32Ty, llvmArgTys,
                                                /*isVarArg=*/false);

    auto llvmFunc = module.lookupSymbol<LLVM::LLVMFuncOp>(fName);
    if (!llvmFunc) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(module.getBody());

      llvmFunc = LLVM::LLVMFuncOp::create(rewriter, loc, fName, llvmFnTy);
      llvmFunc.setVisibility(SymbolTable::Visibility::Private);
    }
    rewriter.setInsertionPoint(op);
    LLVM::CallOp::create(rewriter, loc, TypeRange{i32Ty},
                         SymbolRefAttr::get(llvmFunc), callOperands);
    if (scalarOperand) {
      Value zero = arith::ConstantIndexOp::create(rewriter, loc, 0);
      Value scalarRes =
          memref::LoadOp::create(rewriter, loc, outputMemref, ValueRange{zero});
      if (scalarComputeTy != outputTy)
        scalarRes = arith::TruncFOp::create(rewriter, loc, outputTy, scalarRes);
      rewriter.replaceOp(op, ValueRange{scalarRes});
    } else {
      Value res = bufferization::ToTensorOp::create(
          rewriter, loc, outTensorType, outputMemref, true /* restrict */,
          true /* writable */);
      rewriter.replaceOp(op, res);
    }
    return success();
  }
};

template <typename MathOp>
struct LowerScalarMathPattern final : public OpRewritePattern<MathOp> {
  std::string qhlFuncF32;
  std::string qhlFuncF16;
  LowerScalarMathPattern(MLIRContext *ctx, StringRef fnF32, StringRef fnF16)
      : OpRewritePattern<MathOp>(ctx), qhlFuncF32(fnF32), qhlFuncF16(fnF16) {}

  struct QhlChoice {
    StringRef fn;
    FloatType computeTy;
  };
  static std::optional<QhlChoice> chooseQhl(FloatType t, StringRef f32,
                                            StringRef f16) {
    MLIRContext *ctx = t.getContext();
    if (t.isF32())
      return QhlChoice{f32, t};
    if (t.isF16())
      return QhlChoice{f16, t};
    if (t.isBF16())
      return QhlChoice{f32, Float32Type::get(ctx)};
    if (isa<Float8E4M3FNType, Float8E4M3FNUZType, Float8E5M2Type,
            Float8E5M2FNUZType>(t))
      return QhlChoice{f16, Float16Type::get(ctx)};
    return std::nullopt;
  }

  LogicalResult matchAndRewrite(MathOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value inputScalar = op->getOperand(0);
    auto inputTy = inputScalar.getType();

    if (!isa<FloatType>(inputTy))
      return rewriter.notifyMatchFailure(op, "not a scalar float type");

    if (op->template getParentOfType<linalg::LinalgOp>())
      return rewriter.notifyMatchFailure(
          op, "math op inside a linalg op — handled by UseHexagonRoutines");

    auto floatTy = cast<FloatType>(inputTy);

    // Select QHL function and compute type.
    auto choice = chooseQhl(floatTy, qhlFuncF32, qhlFuncF16);
    if (!choice)
      return rewriter.notifyMatchFailure(op,
                                         "unsupported element type for QHL");
    StringRef qhlFuncName = choice->fn;
    FloatType computeTy = choice->computeTy;

    auto llvmPtrTy = LLVM::LLVMPointerType::get(rewriter.getContext());
    auto i32Ty = rewriter.getI32Type();
    auto i64Ty = IntegerType::get(rewriter.getContext(), 64);

    // Upconvert input scalar if compute type differs (bf16→f32, fp8→f16).
    Value computeInput = inputScalar;
    if (computeTy != floatTy)
      computeInput =
          arith::ExtFOp::create(rewriter, loc, computeTy, inputScalar);

    // Allocate two 1-element stack buffers in compute type.
    auto memrefType = MemRefType::get({1}, computeTy);
    Value inputBuf = memref::AllocaOp::create(rewriter, loc, memrefType);
    Value outputBuf = memref::AllocaOp::create(rewriter, loc, memrefType);

    // Store scalar into input buffer.
    Value zero = arith::ConstantIndexOp::create(rewriter, loc, 0);
    memref::StoreOp::create(rewriter, loc, computeInput, inputBuf,
                            ValueRange{zero});

    // Extract aligned pointers.
    auto toPtr = [&](Value buf) -> Value {
      Value idx =
          memref::ExtractAlignedPointerAsIndexOp::create(rewriter, loc, buf);
      Value asI64 = arith::IndexCastOp::create(rewriter, loc, i64Ty, idx);
      return LLVM::IntToPtrOp::create(rewriter, loc, llvmPtrTy, asI64)
          .getResult();
    };
    Value inputPtr = toPtr(inputBuf);
    Value outputPtr = toPtr(outputBuf);

    Value size =
        arith::ConstantIntOp::create(rewriter, loc, 1, i32Ty.getWidth());

    // Declare or look up the QHL batch function.
    ModuleOp module = op->template getParentOfType<ModuleOp>();
    auto llvmFnTy = LLVM::LLVMFunctionType::get(
        i32Ty, {llvmPtrTy, llvmPtrTy, i32Ty}, /*isVarArg=*/false);

    auto llvmFunc = module.lookupSymbol<LLVM::LLVMFuncOp>(qhlFuncName);
    if (!llvmFunc) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(module.getBody());
      llvmFunc = LLVM::LLVMFuncOp::create(rewriter, loc, qhlFuncName, llvmFnTy);
      llvmFunc.setVisibility(SymbolTable::Visibility::Private);
    }

    rewriter.setInsertionPoint(op);
    LLVM::CallOp::create(rewriter, loc, TypeRange{i32Ty},
                         SymbolRefAttr::get(llvmFunc),
                         ValueRange{inputPtr, outputPtr, size});

    // Load result and downconvert if needed.
    Value result =
        memref::LoadOp::create(rewriter, loc, outputBuf, ValueRange{zero});
    if (computeTy != floatTy)
      result = arith::TruncFOp::create(rewriter, loc, floatTy, result);

    rewriter.replaceOp(op, result);
    return success();
  }
};

struct LowerLibdevicePass
    : public ::impl::LowerLibdeviceBase<LowerLibdevicePass> {
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add<LowerLibdevicePattern>(context);

    // Handle scalar math.* ops
    patterns.add<LowerScalarMathPattern<math::CosOp>>(
        context, "qhmath_hvx_cos_af", "qhmath_hvx_cos_ahf");
    patterns.add<LowerScalarMathPattern<math::SinOp>>(
        context, "qhmath_hvx_sin_af", "qhmath_hvx_sin_ahf");
    patterns.add<LowerScalarMathPattern<math::ExpOp>>(
        context, "qhmath_hvx_exp_af_ieee", "qhmath_hvx_exp_ahf_ieee");
    patterns.add<LowerScalarMathPattern<math::Exp2Op>>(
        context, "qhmath_hvx_exp2_af", "qhmath_hvx_exp2_ahf");
    patterns.add<LowerScalarMathPattern<math::TanhOp>>(
        context, "qhmath_hvx_tanh_af", "qhmath_hvx_tanh_ahf");
    patterns.add<LowerScalarMathPattern<math::TanOp>>(
        context, "qhmath_hvx_tan_af", "qhmath_hvx_tan_ahf");
    patterns.add<LowerScalarMathPattern<math::AsinOp>>(
        context, "qhmath_hvx_asin_af", "qhmath_hvx_asin_ahf");
    patterns.add<LowerScalarMathPattern<math::AcosOp>>(
        context, "qhmath_hvx_acos_af", "qhmath_hvx_acos_ahf");
    patterns.add<LowerScalarMathPattern<math::AtanOp>>(
        context, "qhmath_hvx_atan_af", "qhmath_hvx_atan_ahf");
    patterns.add<LowerScalarMathPattern<math::FloorOp>>(
        context, "qhmath_hvx_floor_af", "qhmath_hvx_floor_ahf");
    patterns.add<LowerScalarMathPattern<math::CeilOp>>(
        context, "qhmath_hvx_ceil_af", "qhmath_hvx_ceil_ahf");

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>> hexagon::createLowerLibdevicePass() {
  return std::make_unique<LowerLibdevicePass>();
}
