//===- SpecalizeTranscendentalsPass.cpp - Identify transcendental ops -----===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// Recognizes transcendental functions in their expanded arithmetic form and
// replaces them with equivalent named-op implementations, so that downstream
// passes can substitute custom library implementations.
//
// Recognize tanh patterns and rewrite them to linalg.tanh:
//   (1 - exp(-2x)) / (1 + exp(-2x))
//   (exp(2x) - 1)  / (exp(2x) + 1)
//   (exp(x) - exp(-x)) / (exp(x) + exp(-x))
//
// Recognize SiLU and sigmoid patterns, and use tanh.
// sigmoid: 1 / (1 + exp(-x))  ->  0.5 * (1 + tanh(x/2))
// silu:    x / (1 + exp(-x))  ->  x * sigmoid(x)
//
//===----------------------------------------------------------------------===//

#include "hexagon/Transforms/Transforms.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"

#include <functional>

#define DEBUG_TYPE "specialize-transcendentals"

#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define DBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
using namespace hexagon;

#define GEN_PASS_DEF_SPECIALIZETRANSCENDENTALS
#include "hexagon/Transforms/Passes.h.inc"

namespace {

/// Returns true iff `genericOp` is a single-input, single-output op
/// op whose body instruction is an arith cast.
static bool isCastOnlyGeneric(linalg::GenericOp genericOp) {
  if (genericOp.getNumDpsInputs() != 1 || genericOp.getNumDpsInits() != 1)
    return false;
  Block &body = genericOp.getRegion().front();
  if (body.getOperations().size() != 2)
    return false;
  Operation &castOp = body.front();
  if (!isa<arith::ExtFOp, arith::TruncFOp, arith::BitcastOp, arith::FPToUIOp,
           arith::FPToSIOp, arith::UIToFPOp, arith::SIToFPOp>(castOp))
    return false;
  auto yieldOp = cast<linalg::YieldOp>(body.back());
  return yieldOp.getNumOperands() == 1 &&
         yieldOp.getOperands()[0] == castOp.getResult(0);
}

/// Follows a chain of cast-only linalg.generic wrappers and returns the
//  first value whose defining op is not such a wrapper. This lets matcher
/// navigate through precision-changing casts inserted by the front-end.
static Value lookThroughCastGenerics(Value v) {
  while (auto g = v.getDefiningOp<linalg::GenericOp>()) {
    if (!isCastOnlyGeneric(g))
      break;
    v = g.getDpsInputOperand(0)->get();
  }
  return v;
}

/// If v's element type differs from targetElemType, inserts a
/// linalg.generic that casts each element (extf or truncf) and returns the
/// result.  Returns v unchanged when types already match.
static Value maybeCastToElemType(IRRewriter &rewriter, Location loc, Value v,
                                 Type targetElemType) {
  auto tensorType = cast<RankedTensorType>(v.getType());
  if (tensorType.getElementType() == targetElemType)
    return v;

  auto dstType = tensorType.clone(targetElemType);
  Value empty = tensor::EmptyOp::create(rewriter, loc, dstType.getShape(),
                                        targetElemType);

  unsigned rank = dstType.getRank();
  AffineMap id = AffineMap::getMultiDimIdentityMap(rank, rewriter.getContext());
  SmallVector<utils::IteratorType> iters(rank, utils::IteratorType::parallel);
  auto srcElemType = tensorType.getElementType();

  auto castGeneric = linalg::GenericOp::create(
      rewriter, loc, TypeRange{dstType}, ValueRange{v}, ValueRange{empty},
      SmallVector<AffineMap>{id, id}, iters,
      [&](OpBuilder &b, Location loc, ValueRange args) {
        auto srcFloat = dyn_cast<FloatType>(srcElemType);
        auto dstFloat = dyn_cast<FloatType>(targetElemType);
        Value casted;
        if (srcFloat && dstFloat && srcFloat.getWidth() < dstFloat.getWidth())
          casted = arith::ExtFOp::create(b, loc, targetElemType, args[0]);
        else
          casted = arith::TruncFOp::create(b, loc, targetElemType, args[0]);
        linalg::YieldOp::create(b, loc, casted);
      });
  return castGeneric.getResult(0);
}

/// Returns true if `val` is produced by linalg.fill(arith.constant(`c`)).
static bool isFromFillWithConstant(Value val, float c) {
  if (auto fillOp = val.getDefiningOp<linalg::FillOp>()) {
    if (auto constOp =
            fillOp.getInputs()[0].getDefiningOp<arith::ConstantOp>()) {
      if (auto floatAttr = dyn_cast<FloatAttr>(constOp.getValue())) {
        return floatAttr.getValueAsDouble() == c;
      }
    }
  }
  return false;
}

/// If `v` is a negation of some value (via `linalg.negf(u)`,
/// `linalg.mul(u, -1)`, or `linalg.sub(0, u)` in either operand order where
/// applicable), returns `u` (looked through cast-only generics).  Returns
/// nullptr when `v` is not a recognized negation.
static Value matchNegation(Value v) {
  if (auto negfOp = v.getDefiningOp<linalg::NegFOp>())
    return lookThroughCastGenerics(negfOp.getDpsInputOperand(0)->get());

  if (auto mulOp = v.getDefiningOp<linalg::MulOp>()) {
    Value m0 = mulOp.getDpsInputOperand(0)->get();
    Value m1 = mulOp.getDpsInputOperand(1)->get();
    if (isFromFillWithConstant(m1, -1.0))
      return lookThroughCastGenerics(m0);
    if (isFromFillWithConstant(m0, -1.0))
      return lookThroughCastGenerics(m1);
  }

  // 0 - u  (the front-end lowers `-x` to a subtraction from a fill(0)).
  // Only the first operand may be the zero: `u - 0` is not a negation.
  if (auto subOp = v.getDefiningOp<linalg::SubOp>()) {
    Value s0 = subOp.getDpsInputOperand(0)->get();
    Value s1 = subOp.getDpsInputOperand(1)->get();
    if (isFromFillWithConstant(s0, 0.0))
      return lookThroughCastGenerics(s1);
  }
  return nullptr;
}

/// Matches the two tanh forms, which share the same structure:
/// one operand of sub/add is fill(1), the other is exp(±2x):
///   (1 - exp(-2x)) / (1 + exp(-2x))   -- fill(1) is s0, exp arg must be -2x.
///   (exp(2x) - 1)  / (exp(2x) + 1)    -- fill(1) is s1, exp arg must be +2x.
/// Returns x on match, nullptr otherwise.
static Value matchTanhScalarExpForm(linalg::SubOp subOp, linalg::AddOp addOp) {
  Value s0 = subOp.getDpsInputOperand(0)->get();
  Value s1 = subOp.getDpsInputOperand(1)->get();

  // Determine which sub operand is fill(1) and which is the exp.
  Value expVal = nullptr;
  float requiredMul = 0.0f;
  if (isFromFillWithConstant(s0, 1.0)) {
    expVal = s1;
    requiredMul = -2.0f;
  } else if (isFromFillWithConstant(s1, 1.0)) {
    expVal = s0;
    requiredMul = 2.0f;
  } else {
    return nullptr;
  }

  // add must contain the same fill(1) and expVal (either order).
  Value a0 = addOp.getDpsInputOperand(0)->get();
  Value a1 = addOp.getDpsInputOperand(1)->get();
  if ((a0 != expVal && a1 != expVal) ||
      (!isFromFillWithConstant(a0, 1.0) && !isFromFillWithConstant(a1, 1.0)))
    return nullptr;

  // exp argument must be requiredMul * x.  Look through any cast-only generics.
  auto expOp = lookThroughCastGenerics(expVal).getDefiningOp<linalg::ExpOp>();
  if (!expOp)
    return nullptr;
  auto mulOp = lookThroughCastGenerics(expOp.getDpsInputOperand(0)->get())
                   .getDefiningOp<linalg::MulOp>();
  if (!mulOp)
    return nullptr;
  Value m0 = mulOp.getDpsInputOperand(0)->get();
  Value m1 = mulOp.getDpsInputOperand(1)->get();
  if (isFromFillWithConstant(m1, requiredMul))
    return m0;
  if (isFromFillWithConstant(m0, requiredMul))
    return m1;
  return nullptr;
}

/// Pattern: tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x)).
static Value matchTanhDualExpForm(linalg::SubOp subOp, linalg::AddOp addOp) {
  Value expPosVal = subOp.getDpsInputOperand(0)->get();
  Value expNegVal = subOp.getDpsInputOperand(1)->get();
  auto expPosOp =
      lookThroughCastGenerics(expPosVal).getDefiningOp<linalg::ExpOp>();
  auto expNegOp =
      lookThroughCastGenerics(expNegVal).getDefiningOp<linalg::ExpOp>();
  if (!expPosOp || !expNegOp)
    return nullptr;

  // e^x and e^(-x) are the same SSA values in sub and add (either order).
  Value a0 = addOp.getDpsInputOperand(0)->get();
  Value a1 = addOp.getDpsInputOperand(1)->get();
  if (!((a0 == expPosVal && a1 == expNegVal) ||
        (a0 == expNegVal && a1 == expPosVal)))
    return nullptr;

  // Look through any cast-only wrappers on the exp inputs.
  Value expPosInput =
      lookThroughCastGenerics(expPosOp.getDpsInputOperand(0)->get());
  Value expNegInput =
      lookThroughCastGenerics(expNegOp.getDpsInputOperand(0)->get());

  // exp arguments are x and -x: via linalg.negf(x) or linalg.mul(x, -1).
  Value negSource = matchNegation(expNegInput);
  if (!negSource || negSource != expPosInput)
    return nullptr;
  return expPosInput;
}

/// Match tanh expansions: returns the tanh input `x` on match, nullptr
/// otherwise.  Recognizes the scalar-exp forms and
/// the dual-exp form.
static Value matchTanh(linalg::DivOp divOp) {
  auto subOp = lookThroughCastGenerics(divOp.getDpsInputOperand(0)->get())
                   .getDefiningOp<linalg::SubOp>();
  auto addOp = lookThroughCastGenerics(divOp.getDpsInputOperand(1)->get())
                   .getDefiningOp<linalg::AddOp>();
  if (!subOp || !addOp)
    return nullptr;

  if (Value x = matchTanhScalarExpForm(subOp, addOp))
    return x;
  return matchTanhDualExpForm(subOp, addOp);
}

/// Matches the silu / sigmoid denominator `1 + exp(-x)` and returns `x`
/// (nullptr on no match).  The exp argument is `-x` via negf, mul(x,-1), or
/// sub(0,x).
static Value matchSigmoidDenominator(linalg::DivOp divOp) {
  auto addOp = lookThroughCastGenerics(divOp.getDpsInputOperand(1)->get())
                   .getDefiningOp<linalg::AddOp>();
  if (!addOp)
    return nullptr;

  Value d0 = addOp.getDpsInputOperand(0)->get();
  Value d1 = addOp.getDpsInputOperand(1)->get();
  Value expVal = nullptr;
  if (isFromFillWithConstant(d0, 1.0))
    expVal = d1;
  else if (isFromFillWithConstant(d1, 1.0))
    expVal = d0;
  else
    return nullptr;

  auto expOp = lookThroughCastGenerics(expVal).getDefiningOp<linalg::ExpOp>();
  if (!expOp)
    return nullptr;

  Value expInput = lookThroughCastGenerics(expOp.getDpsInputOperand(0)->get());
  return matchNegation(expInput);
}

//===----------------------------------------------------------------------===//
// Rewrite builders.  Each takes the matched input and emits named linalg ops
// in `outputType`, returning the replacement value for the div.
//===----------------------------------------------------------------------===//

/// Emit tanh(x), casting `x` to the output element type if needed.
static Value buildTanh(IRRewriter &rewriter, Location loc, Value x,
                       RankedTensorType outputType) {
  Value in = maybeCastToElemType(rewriter, loc, x, outputType.getElementType());
  Value empty = tensor::EmptyOp::create(rewriter, loc, outputType.getShape(),
                                        outputType.getElementType());
  return linalg::TanhOp::create(rewriter, loc, outputType, ValueRange{in},
                                ValueRange{empty})
      .getResult(0);
}

/// Emit sigmoid(x) = 0.5 * (1 + tanh(x/2)) using named linalg ops.  `x` is
/// assumed already cast to the output element type.  No math.sigmoid op
/// exists, so this reuses the tanh library route.
static Value buildSigmoid(IRRewriter &rewriter, Location loc, Value x,
                          RankedTensorType outputType) {
  Type elemType = outputType.getElementType();
  ArrayRef<int64_t> shape = outputType.getShape();
  auto emptyLike = [&]() -> Value {
    return tensor::EmptyOp::create(rewriter, loc, shape, elemType);
  };
  auto fillConst = [&](double c) -> Value {
    Value cst = arith::ConstantOp::create(rewriter, loc,
                                          rewriter.getFloatAttr(elemType, c));
    return linalg::FillOp::create(rewriter, loc, ValueRange{cst},
                                  ValueRange{emptyLike()})
        .getResult(0);
  };
  auto mul = [&](Value a, Value b) -> Value {
    return linalg::MulOp::create(rewriter, loc, TypeRange{outputType},
                                 ValueRange{a, b}, ValueRange{emptyLike()})
        .getResult(0);
  };

  Value half = fillConst(0.5);
  Value one = fillConst(1.0);
  Value xHalf = mul(x, half);                             // x/2
  Value th = buildTanh(rewriter, loc, xHalf, outputType); // tanh(x/2)
  Value onePlus =
      linalg::AddOp::create(rewriter, loc, TypeRange{outputType},
                            ValueRange{one, th}, ValueRange{emptyLike()})
          .getResult(0);     // 1 + tanh(x/2)
  return mul(half, onePlus); // 0.5 * (1 + tanh(x/2))
}

//===----------------------------------------------------------------------===//
// Uniform driver.  A SpecializePattern matches a linalg.div and, on success,
// returns the replacement value (built in terms of the div's output type);
// nullptr means "no match".  applySpecializations walks every div once per
// pattern and replaces the matches.
//===----------------------------------------------------------------------===//

using SpecializePattern =
    std::function<Value(IRRewriter &, linalg::DivOp, RankedTensorType)>;

static void applySpecialization(func::FuncOp funcOp, IRRewriter &rewriter,
                                const SpecializePattern &pattern) {
  // Collect the divs first so we never build/replace IR mid-walk.
  SmallVector<linalg::DivOp> divs;
  funcOp.walk([&](linalg::DivOp divOp) { divs.push_back(divOp); });

  for (linalg::DivOp divOp : divs) {
    rewriter.setInsertionPoint(divOp);
    auto outputType = cast<RankedTensorType>(divOp.getResult(0).getType());
    if (Value repl = pattern(rewriter, divOp, outputType))
      rewriter.replaceOp(divOp, repl);
  }
}

/// tanh:    (exp forms) / (exp forms)  ->  tanh(x)
static Value specializeTanhPattern(IRRewriter &rewriter, linalg::DivOp divOp,
                                   RankedTensorType outputType) {
  Value x = matchTanh(divOp);
  if (!x)
    return nullptr;
  DBG("Specializing tanh at " << divOp.getLoc());
  return buildTanh(rewriter, divOp.getLoc(), x, outputType);
}

/// sigmoid: 1 / (1 + exp(-x))  ->  0.5 * (1 + tanh(x/2))
static Value specializeSigmoidPattern(IRRewriter &rewriter, linalg::DivOp divOp,
                                      RankedTensorType outputType) {
  // Numerator must be fill(1).
  if (!isFromFillWithConstant(divOp.getDpsInputOperand(0)->get(), 1.0))
    return nullptr;
  Value x = matchSigmoidDenominator(divOp);
  if (!x)
    return nullptr;
  DBG("Specializing sigmoid at " << divOp.getLoc());
  Location loc = divOp.getLoc();
  Value xc = maybeCastToElemType(rewriter, loc, x, outputType.getElementType());
  return buildSigmoid(rewriter, loc, xc, outputType);
}

/// silu:    x / (1 + exp(-x))  ->  x * sigmoid(x)
static Value specializeSiluPattern(IRRewriter &rewriter, linalg::DivOp divOp,
                                   RankedTensorType outputType) {
  Value x = matchSigmoidDenominator(divOp);
  if (!x)
    return nullptr;
  // Numerator must be the same value the denominator negates (silu, not
  // sigmoid, which has a fill(1) numerator handled by its own pattern).
  if (lookThroughCastGenerics(divOp.getDpsInputOperand(0)->get()) != x)
    return nullptr;
  DBG("Specializing silu at " << divOp.getLoc());
  Location loc = divOp.getLoc();
  Value xc = maybeCastToElemType(rewriter, loc, x, outputType.getElementType());
  Value sig = buildSigmoid(rewriter, loc, xc, outputType);
  return linalg::MulOp::create(rewriter, loc, TypeRange{outputType},
                               ValueRange{xc, sig},
                               ValueRange{tensor::EmptyOp::create(
                                   rewriter, loc, outputType.getShape(),
                                   outputType.getElementType())})
      .getResult(0);
}

/// Rewrites recognized transcendental patterns to linalg named math ops.
struct SpecializeTranscendentalsPass
    : public ::impl::SpecializeTranscendentalsBase<
          SpecializeTranscendentalsPass> {

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect>();
    registry.insert<arith::ArithDialect>();
  }

  void runOnOperation() override {
    auto funcOp = getOperation();
    MLIRContext *context = &getContext();

    // Snapshot pre-existing named ops so we never generalize them.
    DenseSet<Operation *> preExistingNamedOps;
    funcOp.walk([&](linalg::LinalgOp op) {
      if (!isa<linalg::GenericOp>(op))
        preExistingNamedOps.insert(op);
    });

    // Lower `linalg.generic` to `linalg.named` to expose per-op structure.
    IRRewriter rewriter(context);
    SmallVector<linalg::GenericOp> genericOps;
    funcOp.walk(
        [&](linalg::GenericOp genericOp) { genericOps.push_back(genericOp); });
    for (auto genericOp : genericOps) {
      rewriter.setInsertionPoint(genericOp);
      (void)linalg::specializeGenericOp(rewriter, genericOp);
    }

    // Do the match and replacements, working on named ops.  Order matters:
    // silu before sigmoid is irrelevant (their numerators are disjoint), but
    // each pattern walks all divs independently.
    applySpecialization(funcOp, rewriter, specializeTanhPattern);
    applySpecialization(funcOp, rewriter, specializeSiluPattern);
    applySpecialization(funcOp, rewriter, specializeSigmoidPattern);

    // Lower back to linalg.generic — but only ops we created.
    SmallVector<linalg::LinalgOp> namedOps;
    funcOp.walk([&](linalg::LinalgOp op) {
      if (!isa<linalg::GenericOp>(op) && !preExistingNamedOps.count(op))
        namedOps.push_back(op);
    });
    for (auto namedOp : namedOps) {
      rewriter.setInsertionPoint(namedOp);
      (void)linalg::generalizeNamedOp(rewriter, namedOp);
    }
  }
};
} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
hexagon::createSpecializeTranscendentalsPass() {
  return std::make_unique<SpecializeTranscendentalsPass>();
}
