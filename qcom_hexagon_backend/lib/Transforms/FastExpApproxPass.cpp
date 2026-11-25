//===- FastExpApproxPass.cpp - Replace math.exp with polynomial -----------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// Replaces math.exp (and math.exp2) on f32 with a parametric polynomial
// approximation that lowers to pure HVX arithmetic.
//
//   exp(x)  = 2^(x * 1/ln 2)
//   exp2(x) = 2^x
//
// In both cases we factor as 2^k * 2^f  (k = trunc(y), f = y - k):
//
//   2^f ≈ 1 + c_1*f + c_2*f^2 + ... + c_n*f^n   (degree-n minimax on [-1,0])
//   2^k = bitcast<f32>((k + 127) << 23)          (exact IEEE-754 exponent)
//
// Coefficients are precomputed as the minimax polynomial for 2^f on [-1, 0]
// with fixed constant 1.0, solved via LP (scipy/HiGHS). See
// compute_minimax_coefficients.py for the computation.
//
// Error budget (max |p(f) - 2^f| over f in [-1,0]):
//   degree 1: 3.01e-02    degree 2: 1.54e-03    degree 3: 6.30e-05
//   degree 4: 2.17e-06    degree 5: 1.11e-07    degree 6+: <1e-7 (f32 eps)
//
// The default degree is 3.
//
// Supported operand types: scalar f32, statically-shaped tensor<...xf32>,
// and vector<...xf32>.  Shaped types use splat dense constants; all arith
// ops naturally broadcast over elements.
//
//===----------------------------------------------------------------------===//

#include "hexagon/Transforms/Transforms.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "fast-exp-approx"

#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define DBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
using namespace hexagon;

#define GEN_PASS_DEF_FASTEXPAPPROX
#include "hexagon/Transforms/Passes.h.inc"

namespace {

// ---------------------------------------------------------------------------
// Minimax polynomial coefficients for 2^f on [-1, 0] with fixed c_0 = 1.0.
// kTwoPowCoeffs[n-1][k] = c_{k+1}, i.e. the coefficient of f^{k+1}.
// Polynomial: p(f) = 1 + c_1*f + c_2*f^2 + ... + c_n*f^n  (Horner form).
// Unused slots (k >= n) are 0.0.
// ---------------------------------------------------------------------------

static constexpr float kTwoPowCoeffs[10][10] = {
    // degree 1, max_err=3.012e-02
    {0.5301239122f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f},
    // degree 2, max_err=1.544e-03
    {0.6748326295f, 0.1763761572f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
     0.0f},
    // degree 3, max_err=6.304e-05
    {0.6918343431f, 0.2319281473f, 0.0401568189f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
     0.0f, 0.0f},
    // degree 4, max_err=2.172e-06
    {0.6930792525f, 0.2395465468f, 0.0533732306f, 0.0069080229f, 0.0f, 0.0f,
     0.0f, 0.0f, 0.0f, 0.0f},
    // degree 5, max_err=1.109e-07
    {0.6931445853f, 0.2401849028f, 0.0553070664f, 0.0092143433f, 0.0009476349f,
     0.0f, 0.0f, 0.0f, 0.0f, 0.0f},
    // degree 6, max_err=7.702e-09
    {0.6931471791f, 0.2402263704f, 0.0555009017f, 0.0096004666f, 0.0012944694f,
     0.0001157131f, 0.0f, 0.0f, 0.0f, 0.0f},
    // degree 7, max_err=1.985e-08
    {0.6931471800f, 0.2402264594f, 0.0555029578f, 0.0096083205f, 0.0013048583f,
     0.0001202163f, 0.0f, 0.0f, 0.0f, 0.0f},
    // degree 8, max_err=1.429e-10
    {0.6931471804f, 0.2402264956f, 0.0555038780f, 0.0096166880f, 0.0013293146f,
     0.0001482749f, 0.0000110856f, 0.0f, 0.0f, 0.0f},
    // degree 9, max_err=1.142e-08
    {0.6931471709f, 0.2402257323f, 0.0554896705f, 0.0095426971f, 0.0011671982f,
     0.0f, 0.0f, 0.0000642961f, 0.0000286859f, 0.0f},
    // degree 10, max_err=9.477e-09
    {0.6931471637f, 0.2402252693f, 0.0554868059f, 0.0095363804f, 0.0011621987f,
     0.0f, 0.0f, 0.0000618880f, 0.0000273693f, 0.0f},
};

// Returns true if v is scalar f32 or a statically-shaped tensor/vector of f32.
static bool isF32Like(Value v) {
  Type ty = v.getType();
  if (auto fty = dyn_cast<FloatType>(ty))
    return fty.getWidth() == 32;
  if (auto shaped = dyn_cast<ShapedType>(ty))
    if (shaped.hasStaticShape())
      if (auto fty = dyn_cast<FloatType>(shaped.getElementType()))
        return fty.getWidth() == 32;
  return false;
}

// Build the polynomial approximation of 2^y for f32 `y` of given degree.
// Accepts scalar f32, tensor<...xf32>, and vector<...xf32>.
//
// For finite in-range y the integer exponent k = trunc(y) is exact and the
// fractional part f = y - k is in (-1, 0].  The polynomial and bit-trick then
// give the correct result.  Out-of-range and non-finite inputs produce garbage
// intermediates, but three integer selects at the end always override those:
//   y = -inf or y <= -127  =>  select → 0.0
//   y = +inf or y >= 128   =>  select → +inf
static Value buildTwoPowApprox(OpBuilder &b, Location loc, Value y,
                               unsigned deg) {
  assert(deg >= 1 && deg <= 10 && "degree must be in [1, 10]");

  Type yTy = y.getType();
  Type f32 = b.getF32Type();
  Type i32 = b.getI32Type();
  // For shaped types, derive the matching integer shaped type.
  Type iTy = isa<ShapedType>(yTy) ? cast<ShapedType>(yTy).clone(i32) : i32;

  const float *c = kTwoPowCoeffs[deg - 1];

  // Create a float constant of the same type as y (splat for shaped types).
  auto floatCst = [&](float v) -> Value {
    if (auto shapedTy = dyn_cast<ShapedType>(yTy)) {
      auto attr = DenseElementsAttr::get(
          shapedTy, b.getFloatAttr(f32, static_cast<double>(v)));
      return arith::ConstantOp::create(b, loc, attr);
    }
    return arith::ConstantOp::create(b, loc, b.getF32FloatAttr(v));
  };

  // Create an integer constant of the integer-counterpart type.
  auto intCst = [&](int32_t v) -> Value {
    if (auto intShapedTy = dyn_cast<ShapedType>(iTy)) {
      auto attr = DenseElementsAttr::get(
          intShapedTy, b.getIntegerAttr(i32, static_cast<int64_t>(v)));
      return arith::ConstantOp::create(b, loc, attr);
    }
    return arith::ConstantOp::create(b, loc, b.getI32IntegerAttr(v));
  };

  // kIRaw = trunc(y) as i32.  For finite in-range y this is exact.  For ±inf
  // or NaN fptosi saturates to INT_MAX/INT_MIN on Hexagon — the result feeds
  // only the overflow/underflow selects below and is never used for the
  // polynomial itself when those selects fire.
  Value kIRaw = arith::FPToSIOp::create(b, loc, iTy, y);

  Value kF = arith::SIToFPOp::create(b, loc, yTy, kIRaw);
  Value f = arith::SubFOp::create(b, loc, y, kF);

  // 2^f via Horner starting from the highest-degree term:
  //   acc = c_n
  //   acc = acc * f + c_{n-1}
  //   ...
  //   acc = acc * f + c_1
  //   twoPowF = acc * f + 1.0
  Value acc = floatCst(c[deg - 1]);
  for (int k = (int)deg - 2; k >= 0; --k) {
    acc = arith::MulFOp::create(b, loc, acc, f);
    acc = arith::AddFOp::create(b, loc, acc, floatCst(c[k]));
  }
  Value twoPowF = arith::AddFOp::create(
      b, loc, arith::MulFOp::create(b, loc, acc, f), floatCst(1.0f));

  // 2^k = bitcast<f32>((kIRaw + 127) << 23).
  Value kBias = arith::AddIOp::create(b, loc, kIRaw, intCst(127));
  Value kShift = arith::ShLIOp::create(b, loc, kBias, intCst(23));
  Value twoPowK = arith::BitcastOp::create(b, loc, yTy, kShift);

  // 2^y = twoPowF * twoPowK is mathematically > 0.
  // Mark so sign-sensitive downstream rewrites (FastInversePass) can treat
  // an exp approximation as non-negative. See kHexagonNonNegativeAttrName.
  auto resultOp = arith::MulFOp::create(b, loc, twoPowF, twoPowK);
  resultOp->setAttr(hexagon::kHexagonNonNegativeAttrName, b.getUnitAttr());
  Value result = resultOp.getResult();

  // Fix up ±inf inputs via integer selects — float ops acquire fastmath<fast>
  // downstream which permits assuming no infinities, so we can't rely on
  // float clamps to fix up the polynomial for out-of-range inputs.
  // All comparisons are on kIRaw (the unclamped integer exponent):
  //   kIRaw >= 128  <=>  input was >= 128.0 or +inf  =>  output = +inf
  //   kIRaw <= -127 <=>  input was <= -127.0 or -inf  =>  output = 0.0
  // We build the +inf and 0.0 values purely from integer ops to avoid float
  // inf in the IR entirely.
  // +inf as a float: bitcast(0x7F800000) = bitcast(255 << 23).
  Value infBits = arith::ShLIOp::create(b, loc, intCst(255), intCst(23));
  Value posInf = arith::BitcastOp::create(b, loc, yTy, infBits);

  Value isOverflow = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::sge,
                                           kIRaw, intCst(128));
  result = arith::SelectOp::create(b, loc, isOverflow, posInf, result);

  Value isUnderflow = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::sle,
                                            kIRaw, intCst(-127));
  Value zero = floatCst(0.0f);
  result = arith::SelectOp::create(b, loc, isUnderflow, zero, result);

  return result;
}

struct FastExpApproxPass : public ::impl::FastExpApproxBase<FastExpApproxPass> {
  explicit FastExpApproxPass(FastExpApproxOptions options = {})
      : FastExpApproxBase(options) {}

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<math::MathDialect>();
    registry.insert<arith::ArithDialect>();
  }

  void runOnOperation() override {
    unsigned deg = degree;
    if (deg < 1 || deg > 10) {
      getOperation().emitError()
          << "fast-exp-approx: degree must be in [1, 10], got " << deg;
      return signalPassFailure();
    }

    auto funcOp = getOperation();

    SmallVector<Operation *> toErase;
    funcOp.walk([&](Operation *op) {
      OpBuilder b(op);
      Location loc = op->getLoc();

      if (auto expOp = dyn_cast<math::ExpOp>(op)) {
        Value x = expOp.getOperand();
        if (!isF32Like(x))
          return;
        Value invLn2 = arith::ConstantOp::create(
            b, loc, b.getF32FloatAttr(1.4426950408889634f));
        // For shaped types, splat the scalar constant to match x's type.
        if (isa<ShapedType>(x.getType())) {
          auto shapedTy = cast<ShapedType>(x.getType());
          auto attr = DenseElementsAttr::get(
              shapedTy, b.getFloatAttr(b.getF32Type(), 1.4426950408889634));
          invLn2 = arith::ConstantOp::create(b, loc, attr);
        }
        Value y = arith::MulFOp::create(b, loc, x, invLn2);
        Value result = buildTwoPowApprox(b, loc, y, deg);
        expOp.replaceAllUsesWith(result);
        toErase.push_back(expOp);
        DBG("Replaced math.exp (degree=" << deg << ") at " << loc);
      } else if (auto exp2Op = dyn_cast<math::Exp2Op>(op)) {
        Value x = exp2Op.getOperand();
        if (!isF32Like(x))
          return;
        Value result = buildTwoPowApprox(b, loc, x, deg);
        exp2Op.replaceAllUsesWith(result);
        toErase.push_back(exp2Op);
        DBG("Replaced math.exp2 (degree=" << deg << ") at " << loc);
      }
    });

    for (Operation *op : toErase)
      op->erase();
  }
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
hexagon::createFastExpApproxPass(FastExpApproxOptions options) {
  return std::make_unique<FastExpApproxPass>(options);
}
