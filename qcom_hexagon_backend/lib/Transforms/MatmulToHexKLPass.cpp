//===-- MatmulToHexKLPass.cpp - linalg.matmul to hexkl ops --------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// Converts linalg.matmul with FP16 inputs to hexkl.matmul.
//
// Macro mode: Uses hexkl_macro_* API (f16 output only, upcast to f32 if needed)
//             Requires constant weights (preprocessed at compile-time)
// Micro mode: Uses hexkl_micro_* API (more flexible I/O dtype, memory
// allocation)
//
//
//===----------------------------------------------------------------------===//

#include "hexagon/Dialect/HexKL/IR/HexKLDialect.h"
#include "hexagon/Transforms/BatchMatmulToSCF.h"
#include "hexagon/Transforms/LinalgUtils.h"
#include "hexagon/Transforms/Transforms.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "matmul-to-hexkl"

using namespace mlir;
using namespace hexagon;

namespace mlir {
namespace hexagon {
#define GEN_PASS_DEF_MATMULTOHEXKL
#include "hexagon/Transforms/Passes.h.inc"
} // namespace hexagon
} // namespace mlir

namespace {

static bool isI32Fill(Value value, int64_t expected);

static bool isAddReduction(linalg::ReduceOp reduce, int64_t columns) {
  if (reduce.getInputs().size() != 1 || reduce.getInits().size() != 1)
    return false;

  auto inputType = dyn_cast<RankedTensorType>(reduce.getInputs()[0].getType());
  auto resultType = dyn_cast<RankedTensorType>(reduce.getResult(0).getType());
  if (!inputType || !resultType || inputType.getRank() != 2 ||
      resultType.getRank() != 1 || resultType.getShape()[0] != columns ||
      !resultType.getElementType().isInteger(32))
    return false;

  auto dimensions = reduce.getDimensions();
  if (dimensions.size() != 1 || dimensions[0] != 0)
    return false;

  // The correction assumes a plain column sum, not a sum with a bias.
  if (!isI32Fill(reduce.getInits()[0], 0))
    return false;

  Block &body = reduce.getRegion().front();
  auto yield = dyn_cast<linalg::YieldOp>(body.getTerminator());
  if (!yield || yield.getNumOperands() != 1 || body.getNumArguments() != 2)
    return false;

  auto add = yield.getOperand(0).getDefiningOp<arith::AddIOp>();
  if (!add)
    return false;

  return (add.getLhs() == body.getArgument(0) &&
          add.getRhs() == body.getArgument(1)) ||
         (add.getLhs() == body.getArgument(1) &&
          add.getRhs() == body.getArgument(0));
}

static bool isElementwiseRank2Generic(linalg::GenericOp generic,
                                      unsigned numInputs) {
  if (generic.getDpsInputOperands().size() != numInputs ||
      generic.getDpsInits().size() != 1 || generic.getNumLoops() != 2)
    return false;

  auto maps = generic.getIndexingMapsArray();
  if (maps.size() != numInputs + 1 ||
      !llvm::all_of(maps, [](AffineMap map) { return map.isIdentity(); }))
    return false;

  auto iterators = generic.getIteratorTypesArray();
  return iterators.size() == 2 &&
         llvm::all_of(iterators, [](utils::IteratorType iterator) {
           return iterator == utils::IteratorType::parallel;
         });
}

static bool isI32Fill(Value value, int64_t expected) {
  auto fill = value.getDefiningOp<linalg::FillOp>();
  if (!fill || fill.getInputs().size() != 1)
    return false;

  auto constant = fill.getInputs()[0].getDefiningOp<arith::ConstantOp>();
  if (!constant)
    return false;

  auto integer = dyn_cast<IntegerAttr>(constant.getValue());
  return integer && integer.getType().isInteger(32) &&
         integer.getInt() == expected;
}

static bool isExactCenteredI8Producer(Value narrow, Value &wide) {
  auto narrowGeneric = narrow.getDefiningOp<linalg::GenericOp>();
  if (!narrowGeneric || !isElementwiseRank2Generic(narrowGeneric, 1))
    return false;

  auto narrowType = dyn_cast<RankedTensorType>(narrow.getType());
  auto wideType = dyn_cast<RankedTensorType>(
      narrowGeneric.getDpsInputOperand(0)->get().getType());
  if (!narrowType || !wideType || narrowType.getRank() != 2 ||
      wideType.getRank() != 2 || !narrowType.getElementType().isInteger(8) ||
      !wideType.getElementType().isInteger(32) ||
      narrowType.getShape() != wideType.getShape())
    return false;

  Block &narrowBody = narrowGeneric.getRegion().front();
  auto narrowYield = dyn_cast<linalg::YieldOp>(narrowBody.getTerminator());
  if (!narrowYield || narrowYield.getNumOperands() != 1 ||
      narrowBody.getNumArguments() != 2)
    return false;
  auto trunc = narrowYield.getOperand(0).getDefiningOp<arith::TruncIOp>();
  if (!trunc || trunc.getIn() != narrowBody.getArgument(0))
    return false;

  wide = narrowGeneric.getDpsInputOperand(0)->get();
  auto centerGeneric = wide.getDefiningOp<linalg::GenericOp>();
  if (!centerGeneric || !isElementwiseRank2Generic(centerGeneric, 2))
    return false;

  Block &centerBody = centerGeneric.getRegion().front();
  auto centerYield = dyn_cast<linalg::YieldOp>(centerBody.getTerminator());
  if (!centerYield || centerYield.getNumOperands() != 1 ||
      centerBody.getNumArguments() != 3)
    return false;
  auto subtract = centerYield.getOperand(0).getDefiningOp<arith::SubIOp>();
  if (!subtract || subtract.getLhs() != centerBody.getArgument(0) ||
      subtract.getRhs() != centerBody.getArgument(1))
    return false;

  Value unsignedInput = centerGeneric.getDpsInputOperand(0)->get();
  auto unsignedGeneric = unsignedInput.getDefiningOp<linalg::GenericOp>();
  if (!unsignedGeneric || !isElementwiseRank2Generic(unsignedGeneric, 1))
    return false;

  Block &unsignedBody = unsignedGeneric.getRegion().front();
  auto unsignedYield = dyn_cast<linalg::YieldOp>(unsignedBody.getTerminator());
  if (!unsignedYield || unsignedYield.getNumOperands() != 1 ||
      unsignedBody.getNumArguments() != 2)
    return false;
  auto extui = unsignedYield.getOperand(0).getDefiningOp<arith::ExtUIOp>();
  if (!extui || extui.getIn() != unsignedBody.getArgument(0) ||
      !extui.getIn().getType().isInteger(8))
    return false;

  return isI32Fill(centerGeneric.getDpsInputOperand(1)->get(), 128);
}

static Value findExistingColumnSum(Value weight, Operation *matmul) {
  auto weightType = dyn_cast<RankedTensorType>(weight.getType());
  if (!weightType || weightType.getRank() != 2)
    return {};

  int64_t columns = weightType.getShape()[1];
  auto findReduction = [&](Value input) -> Value {
    for (Operation *user : input.getUsers()) {
      auto reduce = dyn_cast<linalg::ReduceOp>(user);
      if (reduce && reduce.getInputs()[0] == input &&
          reduce->getBlock() == matmul->getBlock() &&
          reduce->isBeforeInBlock(matmul) && isAddReduction(reduce, columns))
        return reduce.getResult(0);
    }
    return {};
  };

  if (Value result = findReduction(weight))
    return result;

  Value wide;
  if (isExactCenteredI8Producer(weight, wide))
    return findReduction(wide);

  return {};
}

// HMX mm_u8i8 uses unsigned activations and signed weights. Convert signed i8
// dot exactly as mm_u8i8(A ^ 0x80, B) - 128 * colsum(B). The XOR bias cancels
// an existing producer-side shift; K-edge weight padding must be zero.
struct U8I8Emitter {
  PatternRewriter &rewriter;
  Location loc;

  // %biased = xori(%src, 0x80), elementwise over a rank-2 i8 tensor.
  Value emitBias(TypedValue<RankedTensorType> src) const {
    auto srcType = src.getType();
    Value init = tensor::EmptyOp::create(rewriter, loc, srcType.getShape(),
                                         srcType.getElementType());
    AffineMap id = rewriter.getMultiDimIdentityMap(2);
    SmallVector<utils::IteratorType> iters(2, utils::IteratorType::parallel);
    auto op = linalg::GenericOp::create(
        rewriter, loc, srcType, ValueRange{src}, ValueRange{init},
        SmallVector<AffineMap>{id, id}, iters,
        [&](OpBuilder &b, Location loc, ValueRange args) {
          // 0x80 as i8 is -128; xor flips the sign bit, mapping the signed
          // range onto the unsigned one that mm_u8i8 expects.
          Value signBit = arith::ConstantOp::create(
              b, loc, b.getIntegerAttr(b.getI8Type(), -128));
          linalg::YieldOp::create(
              b, loc,
              arith::XOrIOp::create(b, loc, args[0], signBit).getResult());
        });
    return op.getResult(0);
  }

  // %colsum[n] = sum over k of sext(%weight[k][n]) : tensor<Nxi32>
  Value emitColSum(TypedValue<RankedTensorType> weight, Type i32Ty) const {
    int64_t n = weight.getType().getShape()[1];
    auto vecType = RankedTensorType::get({n}, i32Ty);
    Value zero = arith::ConstantOp::create(rewriter, loc,
                                           rewriter.getIntegerAttr(i32Ty, 0));
    Value empty =
        tensor::EmptyOp::create(rewriter, loc, vecType.getShape(), i32Ty);
    Value init = linalg::FillOp::create(rewriter, loc, ValueRange{zero},
                                        ValueRange{empty})
                     .getResult(0);

    // Iteration space (n, k): n parallel, k reduction.  Weight is indexed
    // [k][n], so its map is (d0, d1) -> (d1, d0).
    MLIRContext *ctx = rewriter.getContext();
    AffineExpr d0 = getAffineDimExpr(0, ctx), d1 = getAffineDimExpr(1, ctx);
    SmallVector<AffineMap> maps = {
        AffineMap::get(2, 0, {d1, d0}, ctx), // weight[k][n]
        AffineMap::get(2, 0, {d0}, ctx),     // colsum[n]
    };
    SmallVector<utils::IteratorType> iters = {utils::IteratorType::parallel,
                                              utils::IteratorType::reduction};
    auto op = linalg::GenericOp::create(
        rewriter, loc, vecType, ValueRange{weight}, ValueRange{init}, maps,
        iters, [&](OpBuilder &b, Location loc, ValueRange args) {
          // Widen before accumulating: K can be large enough that an i8 sum
          // wraps.
          Value wide = arith::ExtSIOp::create(b, loc, i32Ty, args[0]);
          linalg::YieldOp::create(
              b, loc, arith::AddIOp::create(b, loc, args[1], wide).getResult());
        });
    return op.getResult(0);
  }

  // %out[m,n] = %raw[m,n] - 128 * %colsum[n]
  Value emitCorrection(Value raw, Value colsum,
                       RankedTensorType outType) const {
    Value init = tensor::EmptyOp::create(rewriter, loc, outType.getShape(),
                                         outType.getElementType());
    MLIRContext *ctx = rewriter.getContext();
    AffineExpr d0 = getAffineDimExpr(0, ctx), d1 = getAffineDimExpr(1, ctx);
    SmallVector<AffineMap> maps = {
        AffineMap::get(2, 0, {d0, d1}, ctx), // raw[m][n]
        AffineMap::get(2, 0, {d1}, ctx),     // colsum[n] broadcast over m
        AffineMap::get(2, 0, {d0, d1}, ctx), // out[m][n]
    };
    SmallVector<utils::IteratorType> iters(2, utils::IteratorType::parallel);
    auto op = linalg::GenericOp::create(
        rewriter, loc, outType, ValueRange{raw, colsum}, ValueRange{init}, maps,
        iters, [&](OpBuilder &b, Location loc, ValueRange args) {
          Value c128 = arith::ConstantOp::create(
              b, loc, b.getIntegerAttr(outType.getElementType(), 128));
          Value scaled = arith::MulIOp::create(b, loc, args[1], c128);
          linalg::YieldOp::create(
              b, loc,
              arith::SubIOp::create(b, loc, args[0], scaled).getResult());
        });
    return op.getResult(0);
  }
};

struct MatmulToHexKL final : public OpRewritePattern<linalg::MatmulOp> {
  bool isMacroMode;

  MatmulToHexKL(MLIRContext *ctx, bool macroMode)
      : OpRewritePattern(ctx), isMacroMode(macroMode) {}

  bool isConstantWeight(Value weight) const {
    return weight.getDefiningOp<arith::ConstantOp>() != nullptr;
  }

  LogicalResult matchAndRewrite(linalg::MatmulOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    const auto rank = 2;
    // Macro mode has no u8i8 entry point (hexkl_macro_* is f16-only), so only
    // micro mode admits the (i8, i8) -> i32 contract.
    HMXMatmulKind kind = HMXMatmulKind::F16;
    auto check = isHMXTypeAndRankCompatible(op, rewriter, rank,
                                            /*allowU8I8=*/!isMacroMode, &kind);
    if (failed(check))
      return failure();

    auto [A, B, C] = *check;

    auto aShape = A.getType().getShape();
    auto bShape = B.getType().getShape();
    if (aShape.size() < 2 || bShape.size() < 2)
      return failure();

    int64_t M = aShape[0];
    int64_t K = aShape[1];
    int64_t N = bShape[1];

    // The u8i8 weight-layout API processes complete K=32 blocks and has no
    // K-bound argument for zero-padding a tail tile.
    if (kind == HMXMatmulKind::U8I8 && (K <= 0 || K % 32 != 0)) {
      return rewriter.notifyMatchFailure(
          op, "u8i8 micro mode requires a positive static K multiple of 32");
    }

    // rm_to_wh_i8 has no N bound and reads full 32-column tiles.
    if (kind == HMXMatmulKind::U8I8 && (N <= 0 || N % 32 != 0)) {
      return rewriter.notifyMatchFailure(
          op, "u8i8 micro mode requires a positive static N multiple of 32");
    }

    if (isMacroMode) {
      // Macro mode requires constant weights
      if (!isConstantWeight(B)) {
        return rewriter.notifyMatchFailure(
            op, "macro mode requires constant weights (weight must be "
                "arith.constant)");
      }

      const int64_t MAX_N_ROW = 1600;
      const int64_t MAX_N_COL = 5120;
      const int64_t MAX_N_INNER = 76030;

      if (M > MAX_N_ROW) {
        return rewriter.notifyMatchFailure(
            op, "macro mode: M=" + std::to_string(M) + " exceeds max " +
                    std::to_string(MAX_N_ROW));
      }

      if (N > MAX_N_COL) {
        return rewriter.notifyMatchFailure(
            op, "macro mode: N=" + std::to_string(N) + " exceeds max " +
                    std::to_string(MAX_N_COL));
      }

      if (K > MAX_N_INNER) {
        return rewriter.notifyMatchFailure(
            op, "macro mode: K=" + std::to_string(K) + " exceeds max " +
                    std::to_string(MAX_N_INNER));
      }

      const int64_t BLOCK_SIZE = 32;

      if ((M % BLOCK_SIZE) != 0) {
        return rewriter.notifyMatchFailure(
            op, "macro mode: M=" + std::to_string(M) + " must be multiple of " +
                    std::to_string(BLOCK_SIZE));
      }

      if (N % BLOCK_SIZE != 0) {
        return rewriter.notifyMatchFailure(
            op, "macro mode: N=" + std::to_string(N) + " must be multiple of " +
                    std::to_string(BLOCK_SIZE));
      }

      if (K % BLOCK_SIZE != 0) {
        return rewriter.notifyMatchFailure(
            op, "macro mode: K=" + std::to_string(K) + " must be multiple of " +
                    std::to_string(BLOCK_SIZE));
      }
    }

    LLVM_DEBUG({
      llvm::dbgs() << "[" << DEBUG_TYPE
                   << "] Converting matmul to hexkl.matmul ("
                   << (isMacroMode ? "macro" : "micro") << " mode):\n";
      llvm::dbgs() << "  LHS: " << A.getType() << " (M=" << M << ", K=" << K
                   << ")\n";
      llvm::dbgs() << "  RHS: " << B.getType() << " (K=" << K << ", N=" << N
                   << ")\n";
      llvm::dbgs() << "  Output: " << C.getType() << " (M=" << M << ", N=" << N
                   << ")\n";
      if (isMacroMode) {
        llvm::dbgs() << "  Weight is constant: "
                     << (isConstantWeight(B) ? "yes" : "no") << "\n";
      }
    });

    auto cType = C.getType();
    if (isMacroMode) {
      auto f16Shape = cType.getShape();
      auto f16Type = RankedTensorType::get(f16Shape, rewriter.getF16Type());
      Value f16Output = tensor::EmptyOp::create(rewriter, loc, f16Shape,
                                                rewriter.getF16Type());

      auto hexklMatmul =
          hexkl::MatmulOp::create(rewriter, loc, f16Type, A, B, f16Output);

      Value result = hexklMatmul->getResult(0);

      if (cType.getElementType().isF32()) {
        Value f32Output = tensor::EmptyOp::create(rewriter, loc, f16Shape,
                                                  rewriter.getF32Type());

        AffineMap identityMap = rewriter.getMultiDimIdentityMap(2);
        SmallVector<AffineMap> indexingMaps = {identityMap, identityMap};
        SmallVector<utils::IteratorType> iteratorTypes = {
            utils::IteratorType::parallel, utils::IteratorType::parallel};

        auto castOp = linalg::GenericOp::create(
            rewriter, loc, cType, result, f32Output, indexingMaps,
            iteratorTypes, [&](OpBuilder &b, Location loc, ValueRange args) {
              Value f16Val = args[0];
              Value f32Val =
                  arith::ExtFOp::create(b, loc, rewriter.getF32Type(), f16Val);
              linalg::YieldOp::create(b, loc, f32Val);
            });

        result = castOp.getResult(0);
      }

      rewriter.replaceOp(op, result);

    } else if (kind == HMXMatmulKind::U8I8) {
      // Signed i8 x i8 -> i32 via HMX's unsigned-activation MAC. See
      // U8I8Emitter for the identity; the result is bit-exact.
      //
      // Note this reuses the f16 path's pre-existing assumption that
      // linalg.matmul's incoming C is dead (triton-shared emits a linalg.fill 0
      // into it): hexkl.matmul overwrites its destination rather than
      // accumulating, so a nonzero C would be dropped.
      U8I8Emitter emit{rewriter, loc};
      Type i32Ty = cType.getElementType();

      Value biasedA = emit.emitBias(A);
      Value colSum = findExistingColumnSum(B, op);
      if (!colSum)
        colSum = emit.emitColSum(B, i32Ty);

      Value rawOut =
          tensor::EmptyOp::create(rewriter, loc, cType.getShape(), i32Ty);
      auto hexklMatmul =
          hexkl::MatmulOp::create(rewriter, loc, cType, biasedA, B, rawOut);

      Value corrected =
          emit.emitCorrection(hexklMatmul->getResult(0), colSum, cType);
      rewriter.replaceOp(op, corrected);

    } else {
      Value outputOperand = tensor::EmptyOp::create(
          rewriter, loc, cType.getShape(), cType.getElementType());

      auto hexklMatmul =
          hexkl::MatmulOp::create(rewriter, loc, cType, A, B, outputOperand);

      rewriter.replaceOp(op, hexklMatmul->getResult(0));
    }

    return success();
  }
};

struct MatmulToHexKLPass
    : public hexagon::impl::MatmulToHexKLBase<MatmulToHexKLPass> {
  using hexagon::impl::MatmulToHexKLBase<MatmulToHexKLPass>::MatmulToHexKLBase;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<hexkl::HexKLDialect, arith::ArithDialect, linalg::LinalgDialect,
                tensor::TensorDialect, scf::SCFDialect>();
  }

  void runOnOperation() override {
    bool isMacroMode = (mode == "macro");

    if (mode != "macro" && mode != "micro") {
      emitError(getOperation()->getLoc(),
                "Invalid mode '" + mode + "'. Must be 'macro' or 'micro'");
      return signalPassFailure();
    }

    LLVM_DEBUG(llvm::dbgs()
               << "[" << DEBUG_TYPE << "] Running MatmulToHexKLPass (" << mode
               << " mode)\n");

    RewritePatternSet patterns(&getContext());
    patterns.add<MatmulToHexKL>(&getContext(), isMacroMode);
    if (!isMacroMode)
      patterns.add<LoopedHMXMatmul>(&getContext());

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      LLVM_DEBUG(llvm::dbgs()
                 << "[" << DEBUG_TYPE << "] Pattern application failed\n");
      return signalPassFailure();
    }

    LLVM_DEBUG(llvm::dbgs()
               << "[" << DEBUG_TYPE << "] MatmulToHexKLPass complete\n");
  }
};

} // namespace

namespace mlir {
namespace hexagon {

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createMatmulToHexKLPass(const MatmulToHexKLOptions &options) {
  return std::make_unique<MatmulToHexKLPass>(options);
}

} // namespace hexagon
} // namespace mlir
