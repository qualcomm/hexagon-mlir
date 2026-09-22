//===-- HmxToLLVMPass.cpp - Lower hmx ops to LLVM -------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// hmx.* -> llvm.call on the runtime leaves.
//
// The runtime is compiled by the SDK's clang and speaks in plain integers, so
// every buffer argument becomes the integer image of the memref's aligned
// pointer, and the calls are plain `llvm.call`s on `llvm.func` declarations (a
// `func.func` would grow an `_mlir_ciface_*` wrapper around the symbol).
//
// The assembly of the pass -- LLVMConversionTarget, DataLayoutAnalysis, memref
// descriptors -- follows HexKLToLLVM, which is the local precedent for lowering
// to this runtime.
//===----------------------------------------------------------------------===//

#include "hexagon/Conversion/HmxToLLVM/HmxExternalFnNames.h"
#include "hexagon/Conversion/HmxToLLVM/HmxToLLVM.h"
#include "hexagon/Dialect/Hmx/IR/HmxDialect.h"

#include "mlir/Analysis/DataLayoutAnalysis.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/MemRefBuilder.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/FunctionCallUtils.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/ADT/StringSet.h"

using namespace mlir;
using namespace mlir::hmx;

// At global scope, like HexKLToLLVM: the generated pass base lands in ::impl.
#define GEN_PASS_DEF_HMXTOLLVM
#include "hexagon/Conversion/HmxToLLVM/Passes.h.inc"

namespace {

/// Declare (once) a runtime leaf with the given argument types and no result.
FailureOr<LLVM::LLVMFuncOp> getVoidLeaf(ModuleOp module, StringRef name,
                                        ArrayRef<Type> argTys,
                                        ConversionPatternRewriter &rewriter) {
  auto voidTy = LLVM::LLVMVoidType::get(module->getContext());
  return LLVM::lookupOrCreateFn(rewriter, module, name, argTys, voidTy);
}

/// Declare (once) a runtime leaf that returns a value.
FailureOr<LLVM::LLVMFuncOp> getLeaf(ModuleOp module, StringRef name,
                                    ArrayRef<Type> argTys, Type resultTy,
                                    ConversionPatternRewriter &rewriter) {
  return LLVM::lookupOrCreateFn(rewriter, module, name, argTys, resultTy);
}

/// Runtime entry that constructs the runtime's global singleton, i.e. powers HMX
/// up and acquires it. It allocates nothing, so it is the only way for a kernel
/// that never calls a runtime allocation to satisfy the engine precondition.
static constexpr const char *kHmxEnsureFn = "hexagon_runtime_hmx_ensure_dsp";
/// Runtime entry that releases this thread's HMX lock at the end of the span
/// that kHmxEnsureFn opened; see the invariant on ensureHmxEngine.
static constexpr const char *kHmxUnlockFn = "hexagon_runtime_hmx_unlock_dsp";

/// Whether `fn` will execute on the HMX engine: its body contains an op of the
/// HMX dialect that lowers to an HMX engine leaf (mma / acc / bias / pack /
/// unpack) -- exactly the functions the ensure/unlock pair guards.
///
/// The test is on the dialect ops themselves, and it has to run *before* the
/// conversion below erases them; it never looks at callee names. A callee-name
/// prefix decides wrongly as soon as a leaf is renamed or wrapped, or when any
/// unrelated symbol happens to carry the prefix, and the failure is silent: a
/// missing unlock parks the next thread in HAP_compute_res_hmx_lock forever
/// (bin/runtime/src/HexagonAPI.cpp, EnsureHmxLockForThisThread), while a
/// wrongly-added pair only costs a lock round-trip the function releases itself.
///
/// `hmx.stage` / `hmx.await` are dialect ops but lower to the plain DMA runtime
/// entries (`hexagon_runtime_dma_*`), execute no HMX instruction and need no
/// engine (see HmxExternalFnNames.cpp); they are excluded *by name here* rather
/// than the engine ops being enumerated as a whitelist, so that a future dialect
/// op counts as engine until proven otherwise: a wrongly-included op adds a
/// pair the function releases itself, a wrongly-excluded one hangs the device.
///
/// The exclusion was reviewed and ratified: these two ops are excluded **because
/// they lower to the DMA runtime, not to an engine leaf** -- the contract is
/// pinned by test/Conversion/HmxToLLVM/hmx-to-llvm.mlir @stage_await ("a
/// function that only stages and awaits issues no HMX instruction, so it needs
/// no engine ensure/unlock"). Do not "simplify" this exclusion away: dropping it
/// would insert a pair where today's `hmx_*`-call test inserts none, and adding
/// a whitelist of engine ops instead would silently drop the pair of any future
/// engine op and hang the device.
static bool issuesHmxEngineLeaves(Operation *fn) {
  bool found = false;
  fn->walk([&](Operation *op) {
    Dialect *dialect = op->getDialect();
    if (dialect &&
        dialect->getNamespace() == HmxDialect::getDialectNamespace() &&
        !isa<StageOp, AwaitOp>(op))
      found = true;
  });
  return found;
}

/// The symbol prefix of the runtime HMX leaves (all of HmxExternalFnNames.cpp).
/// Used only by verifyHmxLeafCallers below -- never to decide where a pair goes.
static constexpr const char *kHmxLeafPrefix = "hmx_";

/// Post-conversion invariant check, deliberately run *before* ensureHmxEngine
/// so a diagnostic sees IR the insertion has not touched yet. A function that
/// calls an HMX leaf but is absent from `engineKernels` would run HMX
/// instructions with no ensure/unlock pair, and the next thread would block
/// forever in HAP_compute_res_hmx_lock (bin/runtime/src/HexagonAPI.cpp,
/// EnsureHmxLockForThisThread) -- so that state must fail the build loudly
/// instead of shipping a kernel that hangs the device.
///
/// The `hmx_` callee prefix appears here **for verification only, never for the
/// decision**: which functions get the pair is decided solely by
/// issuesHmxEngineLeaves, on the hmx dialect ops, before the conversion. The
/// dialect test is strictly narrower than the old prefix test -- it cannot see
/// a leaf call that exists without a dialect op behind it -- and this check
/// turns exactly that one direction, which used to be a *silent* missing pair,
/// into a compile-time error. It never selects a function for insertion.
static void verifyHmxLeafCallers(ModuleOp moduleOp,
                                 const llvm::StringSet<> &engineKernels) {
  auto verify = [&](auto fn) {
    if (fn.isDeclaration() || engineKernels.count(fn.getName()))
      return;
    bool found = false;
    fn->walk([&](LLVM::CallOp call) {
      if (std::optional<StringRef> callee = call.getCallee())
        found |= callee->starts_with(kHmxLeafPrefix);
    });
    if (found)
      fn.emitError()
          << "function '" << fn.getName()
          << "' issues HMX leaf calls ('" << kHmxLeafPrefix
          << "...') but issuesHmxEngineLeaves did not recognise it from its "
             "hmx dialect ops, so it gets no "
             "hexagon_runtime_hmx_ensure_dsp/hexagon_runtime_hmx_unlock_dsp "
             "pair: it would execute HMX instructions without the engine "
             "brought up, and the next thread would block forever in "
             "HAP_compute_res_hmx_lock";
  };
  moduleOp.walk([&](LLVM::LLVMFuncOp fn) { verify(fn); });
  moduleOp.walk([&](func::FuncOp fn) { verify(fn); });
}

/// Bring the engine up before every HMX accumulator sequence and release it
/// after. Without the ensure calls the DSP aborts as soon as an HMX instruction
/// executes: a kernel whose HMX ops never go through a runtime allocation has
/// nothing else that would power the engine up. Without the unlock calls a
/// second thread's NON_SHARED lock blocks forever (qurt_hmx.h: the waiter
/// suspends until the unit is available), hanging the device as soon as two
/// threads run HMX kernels. The lock is one pairing per kernel: one ensure at
/// entry, one unlock before each return.
///
/// `engineKernels` is collected before the conversion (see runOnOperation),
/// keyed by symbol name: names are unique in a module's symbol table and this
/// pass never renames a function, so the key identifies the same function after
/// the conversion, whereas an `Operation *` held across it could dangle if a
/// future lowering rebuilds a function. Every name still present matches (a
/// miss would need a rename, which does not happen here); a stale name whose
/// function is gone inserts nowhere, and that function executes nothing.
static void ensureHmxEngine(ModuleOp moduleOp,
                            const llvm::StringSet<> &engineKernels) {
  // Both function forms: in the full LinalgToLLVM pipeline func-to-llvm runs
  // before this pass, so kernels are llvm.func; in the standalone lit tests
  // they are still func.func. The engine precondition applies either way.
  SmallVector<Operation *> kernels;
  moduleOp.walk([&](LLVM::LLVMFuncOp fn) {
    if (!fn.isDeclaration() && engineKernels.count(fn.getName()))
      kernels.push_back(fn);
  });
  moduleOp.walk([&](func::FuncOp fn) {
    if (!fn.isDeclaration() && engineKernels.count(fn.getName()))
      kernels.push_back(fn);
  });
  if (kernels.empty())
    return;
  OpBuilder builder(moduleOp.getContext());
  FailureOr<LLVM::LLVMFuncOp> ensureFn = LLVM::lookupOrCreateFn(
      builder, moduleOp, kHmxEnsureFn, /*paramTypes=*/ArrayRef<Type>{},
      LLVM::LLVMVoidType::get(moduleOp.getContext()));
  if (failed(ensureFn))
    return;
  FailureOr<LLVM::LLVMFuncOp> unlockFn = LLVM::lookupOrCreateFn(
      builder, moduleOp, kHmxUnlockFn, /*paramTypes=*/ArrayRef<Type>{},
      LLVM::LLVMVoidType::get(moduleOp.getContext()));
  if (failed(unlockFn))
    return;
  FlatSymbolRefAttr ensureCallee =
      FlatSymbolRefAttr::get(ensureFn->getOperation());
  FlatSymbolRefAttr unlockCallee =
      FlatSymbolRefAttr::get(unlockFn->getOperation());
  for (Operation *fn : kernels) {
    builder.setInsertionPointToStart(&fn->getRegion(0).front());
    LLVM::CallOp::create(builder, fn->getLoc(), TypeRange{}, ensureCallee,
                         ValueRange{});
    // Collect first: inserting while walking invalidates the walk. Both
    // return forms: llvm.return in llvm.func, func.return in func.func.
    SmallVector<Operation *> returns;
    fn->walk([&](Operation *op) {
      if (isa<LLVM::ReturnOp, func::ReturnOp>(op))
        returns.push_back(op);
    });
    for (Operation *ret : returns) {
      builder.setInsertionPoint(ret);
      LLVM::CallOp::create(builder, fn->getLoc(), TypeRange{}, unlockCallee,
                           ValueRange{});
    }
  }
}

/// The runtime ABI is i32, so an index operand has to come down to that.
Value toI32(ConversionPatternRewriter &rewriter, Location loc, Value v) {
  if (v.getType().isInteger(32))
    return v;
  return LLVM::TruncOp::create(rewriter, loc, rewriter.getI32Type(), v);
}

/// A buffer argument is an address: the integer image of the memref's aligned
/// pointer *plus the descriptor's offset*. The offset is not optional: a pack
/// source is often a view into the middle of a buffer (the K block of an
/// attention matmul is a `reinterpret_cast` with a per-iteration offset), and
/// dropping it makes every iteration pack the first block -- measured as a wrong
/// f16 attention result that grows with the number of K blocks.
/// `elemBytes` is the size of one element of the *viewed* array, so callers pass
/// the crouton size for crouton arrays and the element size otherwise.
Value asAddress(ConversionPatternRewriter &rewriter, Location loc,
                Value memrefDesc, int64_t elemBytes) {
  auto i32Ty = rewriter.getI32Type();
  MemRefDescriptor desc(memrefDesc);
  Value addr = LLVM::PtrToIntOp::create(rewriter, loc, i32Ty,
                                        desc.alignedPtr(rewriter, loc));
  Value bytes = LLVM::MulOp::create(
      rewriter, loc, i32Ty, toI32(rewriter, loc, desc.offset(rewriter, loc)),
      LLVM::ConstantOp::create(rewriter, loc, i32Ty,
                               rewriter.getI32IntegerAttr(elemBytes)));
  return LLVM::AddOp::create(rewriter, loc, i32Ty, addr, bytes);
}

/// The stride, in croutons, of tile-grid dimension `dim` (0 = K/M tiles, 1 = N
/// tiles) of a crouton array, read from the operand's own layout. A *slice* of a
/// larger crouton array -- a `memref.subview` over the N tiles, which is how a
/// whole resident weight is read one N block at a time -- keeps dim 0's stride
/// at the whole array's `N/32` croutons, not at its own `dimSize(1)`; addressing
/// it with `dimSize(1)` walks into the wrong K tile for every row past the
/// first. A dense array returns `dimSize(1)` (dim 0) and 1 (dim 1) -- the exact
/// constants the old arithmetic hardcoded -- and a dynamic, non-positive or
/// non-crouton-aligned stride falls back to that dense contract rather than
/// guessing (same policy as `rowStride`).
static int64_t croutonTileStride(MemRefType type, int64_t dim) {
  int64_t dense = dim == 0 ? type.getDimSize(1) : 1;
  SmallVector<int64_t, 5> strides;
  int64_t offset;
  if (failed(type.getStridesAndOffset(strides, offset)) ||
      dim >= static_cast<int64_t>(strides.size()))
    return dense;
  int64_t stride = strides[dim];
  if (ShapedType::isDynamic(stride) || stride <= 0 ||
      stride % crouton::kCroutonElements != 0)
    return dense;
  return stride / crouton::kCroutonElements;
}

/// The address of crouton `(row, col)` of a crouton array: the array's base plus
/// `(row * stride(0) + col * stride(1))` croutons. The tile-grid strides come
/// from the operand's layout, so a crouton array sliced out of a larger one is
/// addressed at its real position; a dense array emits exactly the old
/// `(row * dimSize(1) + col) * 2048` byte arithmetic (the column step stays 1,
/// so the old `+ col` form is kept verbatim).
Value croutonAddr(ConversionPatternRewriter &rewriter, Location loc,
                  Value memrefDesc, MemRefType type, Value row, Value col,
                  int64_t elemBytes) {
  auto i32Ty = rewriter.getI32Type();
  // The descriptor's offset is counted in *elements*, not in croutons, so the
  // base uses the element size while the tile term below is in croutons.
  Value base = asAddress(rewriter, loc, memrefDesc, elemBytes);
  auto cst = [&](int64_t v) -> Value {
    return LLVM::ConstantOp::create(rewriter, loc, i32Ty,
                                    rewriter.getI32IntegerAttr(v));
  };
  // Materialise the row-stride constant before the row index's cast so the
  // dense emission order (constant, trunc, mul) is exactly the old one.
  Value rowStride = cst(croutonTileStride(type, 0));
  Value rowOff = LLVM::MulOp::create(rewriter, loc, i32Ty,
                                     toI32(rewriter, loc, row), rowStride);
  // The column step is 1 for every crouton array the pipeline builds, so keep
  // the old `row*cols + col` form verbatim; a wider step (a layout that packs
  // several N tiles per crouton stride) scales the column index instead.
  Value colIdx = toI32(rewriter, loc, col);
  int64_t colStep = croutonTileStride(type, 1);
  if (colStep != 1)
    colIdx = LLVM::MulOp::create(rewriter, loc, i32Ty, colIdx, cst(colStep));
  Value tile = LLVM::AddOp::create(rewriter, loc, i32Ty, rowOff, colIdx);
  Value bytes = LLVM::MulOp::create(
      rewriter, loc, i32Ty, tile,
      LLVM::ConstantOp::create(rewriter, loc, i32Ty,
                               rewriter.getI32IntegerAttr(crouton::kCroutonBytes)));
  return LLVM::AddOp::create(rewriter, loc, i32Ty, base, bytes);
}

/// The row stride of a rank-2 row-major buffer, in elements. Both a pack source
/// and an unpack destination are addressed row-by-row by the leaf using this
/// stride: the width for a dense buffer, but larger when the buffer is one
/// N-tile of a wider matrix (`memref<M x BN, strided<[N, 1]>>` with `N > BN`) --
/// the K blocks of an attention matmul (source side) and the destination of an
/// N-split store (destination side). Threading the width there instead of the
/// stride reads/writes the wrong columns of every row past the first.
///
/// `width` is the already-materialised column-count value, returned unchanged
/// when the buffer is dense (stride(rank-2) == width) or has an identity layout,
/// so the dense path emits exactly the IR it did before. A dynamic stride cannot
/// be read here, so it falls back to `width` (the dense contract) rather than
/// mis-addressing.
Value rowStride(ConversionPatternRewriter &rewriter, Location loc,
                MemRefType type, Value width) {
  SmallVector<int64_t, 2> strides;
  int64_t offset;
  int64_t widthC = type.getDimSize(type.getRank() - 1);
  if (succeeded(type.getStridesAndOffset(strides, offset)) &&
      strides.size() >= 2) {
    int64_t stride = strides[strides.size() - 2];
    if (!ShapedType::isDynamic(stride) && stride != widthC)
      return LLVM::ConstantOp::create(rewriter, loc, rewriter.getI32Type(),
                                      rewriter.getI32IntegerAttr(stride));
  }
  return width;
}

/// The element count of a memref as an i32. A static shape is a constant; a
/// dynamic one is the product of the descriptor's sizes.
static Value memrefNumElements(ConversionPatternRewriter &rewriter,
                               Location loc, Value memrefDesc,
                               MemRefType type) {
  auto i32Ty = rewriter.getI32Type();
  if (type.hasStaticShape())
    return LLVM::ConstantOp::create(rewriter, loc, i32Ty,
                                    rewriter.getI32IntegerAttr(
                                        type.getNumElements()));
  MemRefDescriptor desc(memrefDesc);
  Value n = LLVM::ConstantOp::create(rewriter, loc, i32Ty,
                                     rewriter.getI32IntegerAttr(1));
  for (unsigned i = 0; i < type.getRank(); ++i)
    n = LLVM::MulOp::create(rewriter, loc, i32Ty, n,
                            toI32(rewriter, loc, desc.size(rewriter, loc, i)));
  return n;
}

/// The runtime's `AddrSpace` for a memref: the memref's own memory space (0 =
/// DDR, 1 = VTCM), which is exactly the enum. `fallback` is used when the memref
/// has no space attribute, so the lowering never silently invents a space.
static int64_t memrefAddressSpace(MemRefType type, int64_t fallback) {
  if (auto space = dyn_cast_or_null<IntegerAttr>(type.getMemorySpace()))
    return space.getInt();
  return fallback;
}

/// Declare (once) the DMA start entry the `hmx.stage` op lowers to. The signature
/// is the one DMAToLLVMPass declares for the same symbol: the pointer arguments
/// are real `!llvm.ptr`s (the host data layout widens them), while the address
/// spaces and the two bypass flags travel as i32.
static FailureOr<LLVM::LLVMFuncOp>
getDmaStartLeaf(ModuleOp module, ConversionPatternRewriter &rewriter) {
  MLIRContext *context = module->getContext();
  auto ptrTy = LLVM::LLVMPointerType::get(context);
  auto i32Ty = rewriter.getI32Type();
  return getLeaf(module, getStageDmaStartFnName(),
                 {ptrTy, i32Ty, ptrTy, i32Ty, i32Ty, i32Ty, i32Ty, ptrTy}, i32Ty,
                 rewriter);
}

/// The DMA wait entry `hmx.await` lowers to: `void dma_wait(i32 token)`.
static FailureOr<LLVM::LLVMFuncOp>
getDmaWaitLeaf(ModuleOp module, ConversionPatternRewriter &rewriter) {
  return getVoidLeaf(module, getAwaitDmaWaitFnName(), {rewriter.getI32Type()},
                     rewriter);
}

/// Emit a runtime leaf call and bind the op's result. The memref-form pack/unpack
/// ops are DPS and return the buffer they wrote (the interface plan's C3), so the
/// result IS the `dst` operand's address: the replacement is the converted `dst`
/// descriptor itself -- no copy, no descriptor rebuild, zero cost. An op with no
/// result (the pre-DPS form) just erases.
static void replaceWithLeafCall(ConversionPatternRewriter &rewriter,
                                Location loc, Operation *op, Value result,
                                LLVM::LLVMFuncOp fn, ValueRange args) {
  LLVM::CallOp::create(rewriter, loc, TypeRange{},
                       FlatSymbolRefAttr::get(fn.getOperation()), args);
  if (op->getNumResults() == 0)
    rewriter.eraseOp(op);
  else
    rewriter.replaceOp(op, result);
}

/// `hmx.mma` -> `hmx_mma_f16(act_at(m,k), wt_at(n,k), n_croutons)`. The weight
/// crouton grid is `[Nt, Kt, ...]` (logical `[N,K]`, K contiguous in dim1, see
/// `docs/hmx/hmx-weight-layout-plan.md` §0), so its tile index pair is `(n_tile, k_tile)`
/// -- dim0 is N, dim1 is K -- not the `(k,n)` of the activation's `[Mt,Kt]`.
struct LowerMma : public ConvertOpToLLVMPattern<MmaOp> {
  using ConvertOpToLLVMPattern<MmaOp>::ConvertOpToLLVMPattern;

  LogicalResult matchAndRewrite(MmaOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    auto i32Ty = rewriter.getI32Type();

    auto fn = getVoidLeaf(module, getMmaF16FnName(), {i32Ty, i32Ty, i32Ty}, rewriter);
    if (failed(fn))
      return failure();

    auto actType = cast<MemRefType>(op.getAct().getType());
    auto wtType = cast<MemRefType>(op.getWt().getType());

    Value actAddr = croutonAddr(rewriter, loc, adaptor.getAct(), actType,
                                adaptor.getM(), adaptor.getK(),
                                actType.getElementTypeBitWidth() / 8);
    // weight grid is [Nt, Kt]: dim0 = N, dim1 = K (layout A), so row = n_tile,
    // col = k_tile.
    Value wtAddr = croutonAddr(rewriter, loc, adaptor.getWt(), wtType,
                               adaptor.getN(), adaptor.getK(),
                               wtType.getElementTypeBitWidth() / 8);
    Value nCroutons = LLVM::ConstantOp::create(
        rewriter, loc, i32Ty, rewriter.getI32IntegerAttr(op.getNCroutons()));

    SmallVector<Value> args{actAddr, wtAddr, nCroutons};
    rewriter.replaceOpWithNewOp<LLVM::CallOp>(
        op, TypeRange{}, FlatSymbolRefAttr::get((*fn).getOperation()), args);
    return success();
  }
};

/// `hmx.acc_read` -> `hmx_bias_load_f16(bias, set)` then
/// `hmx_acc_store_f16(dst, set)`. The bias registers have to be loaded for the
/// selected set before the read-out, and that pairing is this op's contract.
struct LowerAccRead : public ConvertOpToLLVMPattern<AccReadOp> {
  using ConvertOpToLLVMPattern<AccReadOp>::ConvertOpToLLVMPattern;

  LogicalResult matchAndRewrite(AccReadOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    auto i32Ty = rewriter.getI32Type();

    auto loadFn =
        getVoidLeaf(module, getBiasLoadF16FnName(), {i32Ty, i32Ty}, rewriter);
    auto storeFn =
        getVoidLeaf(module, getAccStoreF16FnName(), {i32Ty, i32Ty}, rewriter);
    if (failed(loadFn) || failed(storeFn))
      return failure();

    Value set = LLVM::ConstantOp::create(
        rewriter, loc, i32Ty, rewriter.getI32IntegerAttr(op.getBiasSet()));
    // The bias block is an I8 memref, so the offset is already in bytes.
    Value bias = asAddress(rewriter, loc, adaptor.getBias(), 1);
    auto dstType = cast<MemRefType>(op.getDst().getType());
    Value dst = croutonAddr(rewriter, loc, adaptor.getDst(), dstType,
                            adaptor.getM(), adaptor.getN(),
                            dstType.getElementTypeBitWidth() / 8);

    LLVM::CallOp::create(
        rewriter, loc, TypeRange{},
        FlatSymbolRefAttr::get((*loadFn).getOperation()),
        SmallVector<Value>{bias, set});
    rewriter.replaceOpWithNewOp<LLVM::CallOp>(
        op, TypeRange{}, FlatSymbolRefAttr::get((*storeFn).getOperation()),
        SmallVector<Value>{dst, set});
    return success();
  }
};

/// `hmx.bias_init` -> `hmx_bias_init_unit_f16(bias)`.
struct LowerBiasInit : public ConvertOpToLLVMPattern<BiasInitOp> {
  using ConvertOpToLLVMPattern<BiasInitOp>::ConvertOpToLLVMPattern;

  LogicalResult matchAndRewrite(BiasInitOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    auto i32Ty = rewriter.getI32Type();

    auto fn = getVoidLeaf(module, getBiasInitUnitF16FnName(), {i32Ty}, rewriter);
    if (failed(fn))
      return failure();

    rewriter.replaceOpWithNewOp<LLVM::CallOp>(
        op, TypeRange{}, FlatSymbolRefAttr::get((*fn).getOperation()),
        SmallVector<Value>{asAddress(rewriter, loc, adaptor.getBias(), 1)});
    return success();
  }
};

/// `hmx.acc_clear` -> `hmx_acc_clear_f16()`.
struct LowerAccClear : public ConvertOpToLLVMPattern<AccClearOp> {
  using ConvertOpToLLVMPattern<AccClearOp>::ConvertOpToLLVMPattern;

  LogicalResult matchAndRewrite(AccClearOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();

    auto fn = getVoidLeaf(module, getAccClearF16FnName(), {}, rewriter);
    if (failed(fn))
      return failure();

    rewriter.replaceOpWithNewOp<LLVM::CallOp>(
        op, TypeRange{}, FlatSymbolRefAttr::get((*fn).getOperation()),
        ValueRange{});
    return success();
  }
};

/// `hmx.pack_act` -> `hmx_pack_act_f16(...)` for an f16 source and
/// `hmx_pack_act_f32(...)` for an f32 one (which quantises to the engine's fp16
/// inside the pack), each in the single-block form or the ranged `_bulk` form
/// covering `count` consecutive K tiles in one call (chosen by the op's `count`;
/// its count of 1 is the same single block).
/// `src_stride` is the source's own row stride, so a source that is one N-tile
/// of a wider matrix (a strided view) is read from the right columns instead of
/// from the first `cols` elements of every row.
struct LowerPackAct : public ConvertOpToLLVMPattern<PackActOp> {
  using ConvertOpToLLVMPattern<PackActOp>::ConvertOpToLLVMPattern;

  LogicalResult matchAndRewrite(PackActOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    auto i32Ty = rewriter.getI32Type();

    auto dstType = cast<MemRefType>(op.getDst().getType());
    auto srcType = cast<MemRefType>(op.getSrc().getType());

    int64_t count = op.getCount().value_or(1);
    bool bulk = count > 1;
    SmallVector<Type> argTys(7, i32Ty);
    if (bulk)
      argTys.push_back(i32Ty);
    // The source's element type picks the leaf: an f32 source is quantised to
    // the engine's fp16 inside the pack, so it has its own entry rather than a
    // narrowing op of its own ahead of the pack.
    bool srcIsF32 = srcType.getElementType().isF32();
    auto fn = getVoidLeaf(module,
                          srcIsF32 ? (bulk ? getPackActF32BulkFnName()
                                           : getPackActF32FnName())
                                   : (bulk ? getPackActF16BulkFnName()
                                           : getPackActF16FnName()),
                          argTys, rewriter);
    if (failed(fn))
      return failure();

    Value dst = croutonAddr(rewriter, loc, adaptor.getDst(), dstType,
                            adaptor.getRow(), adaptor.getCol(),
                            dstType.getElementTypeBitWidth() / 8);
    Value src = asAddress(rewriter, loc, adaptor.getSrc(),
                          srcType.getElementTypeBitWidth() / 8);
    auto dimCst = [&](int64_t v) {
      return LLVM::ConstantOp::create(rewriter, loc, i32Ty,
                                      rewriter.getI32IntegerAttr(v))
          .getResult();
    };
    Value rows = dimCst(srcType.getDimSize(0));
    Value cols = dimCst(srcType.getDimSize(1));
    Value srcStride = rowStride(rewriter, loc, srcType, cols);

    SmallVector<Value> args{dst, src, rows, cols, srcStride,
                            toI32(rewriter, loc, adaptor.getRow()),
                            toI32(rewriter, loc, adaptor.getCol())};
    if (bulk)
      args.push_back(dimCst(count));
    replaceWithLeafCall(rewriter, loc, op, adaptor.getDst(), *fn, args);
    return success();
  }
};

/// `hmx.pack_weight` -> `hmx_pack_weight_f16(...)` for an f16 source and
/// `hmx_pack_weight_f32(...)` for an f32 one, each in the single-block form or
/// the ranged `_bulk` form covering `count` consecutive K tiles in one call. The
/// dst crouton grid is `[Nt, Kt, ...]`, so the address takes `(n_tile, k_tile)`
/// even though the source block coordinates the leaf receives stay `(k, n)`.
/// `src_stride` is the source's own row stride, the same contract as
/// `hmx.pack_act`: a K block that is one tile of a wider matrix is read with its
/// real row stride, not the width.
struct LowerPackWeight : public ConvertOpToLLVMPattern<PackWeightOp> {
  using ConvertOpToLLVMPattern<PackWeightOp>::ConvertOpToLLVMPattern;

  LogicalResult matchAndRewrite(PackWeightOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    auto i32Ty = rewriter.getI32Type();

    auto dstType = cast<MemRefType>(op.getDst().getType());
    auto srcType = cast<MemRefType>(op.getSrc().getType());

    int64_t count = op.getCount().value_or(1);
    bool bulk = count > 1;
    SmallVector<Type> argTys(7, i32Ty);
    if (bulk)
      argTys.push_back(i32Ty);
    bool srcIsF32 = srcType.getElementType().isF32();
    auto fn = getVoidLeaf(module,
                          srcIsF32 ? (bulk ? getPackWeightF32BulkFnName()
                                           : getPackWeightF32FnName())
                                   : (bulk ? getPackWeightF16BulkFnName()
                                           : getPackWeightF16FnName()),
                          argTys, rewriter);
    if (failed(fn))
      return failure();

    // weight grid is [Nt, Kt]: dim0 = N, dim1 = K (layout A), so row = n_tile,
    // col = k_tile.
    Value dst = croutonAddr(rewriter, loc, adaptor.getDst(), dstType,
                            adaptor.getNTile(), adaptor.getKTile(),
                            dstType.getElementTypeBitWidth() / 8);
    Value src = asAddress(rewriter, loc, adaptor.getSrc(),
                          srcType.getElementTypeBitWidth() / 8);
    auto dimCst = [&](int64_t v) {
      return LLVM::ConstantOp::create(rewriter, loc, i32Ty,
                                      rewriter.getI32IntegerAttr(v))
          .getResult();
    };
    Value k = dimCst(srcType.getDimSize(0));
    Value n = dimCst(srcType.getDimSize(1));
    Value srcStride = rowStride(rewriter, loc, srcType, n);

    SmallVector<Value> args{dst, src, k, n, srcStride,
                            toI32(rewriter, loc, adaptor.getKTile()),
                            toI32(rewriter, loc, adaptor.getNTile())};
    if (bulk)
      args.push_back(dimCst(count));
    replaceWithLeafCall(rewriter, loc, op, adaptor.getDst(), *fn, args);
    return success();
  }
};

/// `hmx.unpack_acc` -> `hmx_unpack_acc_f16(dst, crouton(src,row,col), rows,
/// cols, dst_stride, row, col)` for one row-pair, or the ranged
/// `hmx_unpack_acc_f16_bulk(..., row, n_pairs)` covering `count` consecutive
/// row-pairs in one call. The leaf writes fp16 row-major; a wider result is
/// widened by the pipeline afterwards, which keeps one unpack implementation.
/// `dst_stride` is the destination's own row stride, so an N-split block (a
/// strided `memref<M x BN, strided<[N, 1]>>`) lands in the right columns instead
/// of being written contiguously.
struct LowerUnpackAcc : public ConvertOpToLLVMPattern<UnpackAccOp> {
  using ConvertOpToLLVMPattern<UnpackAccOp>::ConvertOpToLLVMPattern;

  LogicalResult matchAndRewrite(UnpackAccOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    auto i32Ty = rewriter.getI32Type();

    int64_t count = op.getCount().value_or(1);
    bool bulk = count > 1;
    // The bulk leaf has the same arity as the single one: it replaces the single
    // form's `col` (always 0 here, the leaf walks the crouton row itself) with
    // the row-pair count. Appending it instead would shift the ABI and the leaf
    // would read n_pairs = 0, writing nothing.
    SmallVector<Type> argTys(7, i32Ty);
    auto fn = getVoidLeaf(module,
                          bulk ? getUnpackAccF16BulkFnName()
                               : getUnpackAccF16FnName(),
                          argTys, rewriter);
    if (failed(fn))
      return failure();

    auto srcType = cast<MemRefType>(op.getSrc().getType());
    auto dstType = cast<MemRefType>(op.getDst().getType());

    Value dst = asAddress(rewriter, loc, adaptor.getDst(),
                          dstType.getElementTypeBitWidth() / 8);
    // The leaf unpacks a whole row-pair and walks the crouton row itself, so the
    // address it takes is the row's first crouton.
    Value zeroCol = LLVM::ConstantOp::create(rewriter, loc, i32Ty,
                                             rewriter.getI32IntegerAttr(0));
    Value src = croutonAddr(rewriter, loc, adaptor.getSrc(), srcType,
                            adaptor.getRow(), zeroCol,
                            srcType.getElementTypeBitWidth() / 8);
    auto dimCst = [&](int64_t v) {
      return LLVM::ConstantOp::create(rewriter, loc, i32Ty,
                                      rewriter.getI32IntegerAttr(v))
          .getResult();
    };
    Value rows = dimCst(dstType.getDimSize(0));
    Value cols = dimCst(dstType.getDimSize(1));
    Value dstStride = rowStride(rewriter, loc, dstType, cols);

    SmallVector<Value> args{dst, src, rows, cols, dstStride,
                            toI32(rewriter, loc, adaptor.getRow())};
    if (bulk)
      args.push_back(dimCst(count));
    else
      args.push_back(toI32(rewriter, loc, adaptor.getCol()));
    replaceWithLeafCall(rewriter, loc, op, adaptor.getDst(), *fn, args);
    return success();
  }
};


/// `hmx.unpack_acc_f32` -> `hmx_unpack_acc_f32(dst, res, has_res,
/// crouton(src,row,0), rows, cols, dst_stride, res_stride, row, col)`, or its
/// ranged form `..._bulk(..., row, n_pairs)` covering `count` row-pairs. The
/// fused tail: the leaf unpacks the row-pairs to fp32, adds the residual when
/// present, and stores, in one call. The residual travels as (address, flag):
/// absent is (0, 0), so the same symbol serves both forms and the wiring
/// phase can thread C through without a second op. `dst_stride` is the
/// destination's own row stride (the width for a dense destination, larger for a
/// strided view); the residual is dense by construction, so `res_stride` is its
/// column count.
struct LowerUnpackAccF32 : public ConvertOpToLLVMPattern<UnpackAccF32Op> {
  using ConvertOpToLLVMPattern<UnpackAccF32Op>::ConvertOpToLLVMPattern;

  LogicalResult matchAndRewrite(UnpackAccF32Op op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    auto i32Ty = rewriter.getI32Type();

    int64_t count = op.getCount().value_or(1);
    bool bulk = count > 1;
    // Same arity rule as the fp16 unpack: `n_pairs` replaces `col`, it is not an
    // extra argument.
    SmallVector<Type> argTys(10, i32Ty);
    auto fn = getVoidLeaf(module,
                          bulk ? getUnpackAccF32BulkFnName()
                               : getUnpackAccF32FnName(),
                          argTys, rewriter);
    if (failed(fn))
      return failure();

    auto srcType = cast<MemRefType>(op.getSrc().getType());
    auto dstType = cast<MemRefType>(op.getDst().getType());

    Value dst = asAddress(rewriter, loc, adaptor.getDst(),
                          dstType.getElementTypeBitWidth() / 8);
    Value res;
    Value hasRes;
    Value resStride;
    if (Value residual = adaptor.getResidual()) {
      auto resType = cast<MemRefType>(op.getResidual().getType());
      res = asAddress(rewriter, loc, residual,
                      resType.getElementTypeBitWidth() / 8);
      hasRes = LLVM::ConstantOp::create(rewriter, loc, i32Ty,
                                        rewriter.getI32IntegerAttr(1));
      // The residual is a row-major block like the destination; its row stride
      // is stride(rank-2), not the width. A strided residual (an N-tile view)
      // read with the width would walk the wrong rows, exactly like the pack
      // `src_stride` / unpack `dst_stride` cases. Dense falls back to the width.
      Value resWidth = LLVM::ConstantOp::create(
          rewriter, loc, i32Ty,
          rewriter.getI32IntegerAttr(resType.getDimSize(1)));
      resStride = rowStride(rewriter, loc, resType, resWidth);
    } else {
      res = LLVM::ConstantOp::create(rewriter, loc, i32Ty,
                                     rewriter.getI32IntegerAttr(0));
      hasRes = LLVM::ConstantOp::create(rewriter, loc, i32Ty,
                                        rewriter.getI32IntegerAttr(0));
      resStride = LLVM::ConstantOp::create(rewriter, loc, i32Ty,
                                           rewriter.getI32IntegerAttr(0));
    }
    // Like the fp16 unpack, the leaf walks the crouton row itself, so the
    // address it takes is the row's first crouton.
    Value zeroCol = LLVM::ConstantOp::create(rewriter, loc, i32Ty,
                                             rewriter.getI32IntegerAttr(0));
    Value src = croutonAddr(rewriter, loc, adaptor.getSrc(), srcType,
                            adaptor.getRow(), zeroCol,
                            srcType.getElementTypeBitWidth() / 8);
    auto dimCst = [&](int64_t v) {
      return LLVM::ConstantOp::create(rewriter, loc, i32Ty,
                                      rewriter.getI32IntegerAttr(v))
          .getResult();
    };
    Value rows = dimCst(dstType.getDimSize(0));
    Value cols = dimCst(dstType.getDimSize(1));
    Value dstStride = rowStride(rewriter, loc, dstType, cols);

    SmallVector<Value> args{dst,
                            res,
                            hasRes,
                            src,
                            rows,
                            cols,
                            dstStride,
                            resStride,
                            toI32(rewriter, loc, adaptor.getRow())};
    if (bulk)
      args.push_back(dimCst(count));
    else
      args.push_back(toI32(rewriter, loc, adaptor.getCol()));
    replaceWithLeafCall(rewriter, loc, op, adaptor.getDst(), *fn, args);
    return success();
  }
};


/// `hmx.stage` -> `hexagon_runtime_dma_start(src_row, DDR, slot, VTCM,
/// slot_bytes, 0, 0, status)`, returning the DMA token. The slot is filled by one
/// 1D DMA, so the source is addressed as `base + row * stride(0) * elemBytes`:
/// the memref's *real* row stride, not its width, otherwise a source that is one
/// tile of a wider matrix starts at the wrong column. The length is the
/// destination's element count in bytes (the whole slot is filled).
///
/// `row` is an element row offset into `src` (not a tile index); the verifier
/// pins `src` to a static rank-2 f16 memref. The op issues its transfer
/// unconditionally: the partition pass keeps every row in range (its pipelined
/// kernel stops one iteration short per pipeline stage and the peeled tail only
/// awaits and computes, staging nothing), so there is no out-of-range case to
/// guard here. A token sentinel is not an option either -- the runtime's tokens
/// start at 0, so 0 cannot mean "no transfer".
struct LowerStage : public ConvertOpToLLVMPattern<StageOp> {
  using ConvertOpToLLVMPattern<StageOp>::ConvertOpToLLVMPattern;

  LogicalResult matchAndRewrite(StageOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    MLIRContext *context = module->getContext();
    auto i32Ty = rewriter.getI32Type();
    auto ptrTy = LLVM::LLVMPointerType::get(context);

    auto fn = getDmaStartLeaf(module, rewriter);
    if (failed(fn))
      return failure();

    auto srcType = cast<MemRefType>(op.getSrc().getType());
    auto dstType = cast<MemRefType>(op.getDst().getType());
    int64_t srcElemBytes = srcType.getElementTypeBitWidth() / 8;
    int64_t dstElemBytes = dstType.getElementTypeBitWidth() / 8;

    auto cst = [&](int64_t v) -> Value {
      return LLVM::ConstantOp::create(rewriter, loc, i32Ty,
                                      rewriter.getI32IntegerAttr(v));
    };

    // Source address: descriptor base+offset, then the row's byte offset. The
    // row stride comes from the operand's own layout (dense falls back to the
    // width); the verifier pins a static source, so the column count is a
    // constant and the dense `memref<MxK>` emits stride(0) == K == width.
    Value row = toI32(rewriter, loc, adaptor.getRow());
    Value srcCols = cst(srcType.getDimSize(1));
    Value srcStride = rowStride(rewriter, loc, srcType, srcCols);
    Value rowBytes = LLVM::MulOp::create(
        rewriter, loc, i32Ty,
        LLVM::MulOp::create(rewriter, loc, i32Ty, row, srcStride),
        cst(srcElemBytes));
    Value srcAddr = LLVM::AddOp::create(
        rewriter, loc, i32Ty,
        asAddress(rewriter, loc, adaptor.getSrc(), srcElemBytes), rowBytes);

    // Destination and status are write-through pointers. The DMA address spaces
    // are the memrefs' own memory spaces (0 = DDR, 1 = VTCM), which is exactly
    // the runtime's AddrSpace enum.
    Value dstAddr = asAddress(rewriter, loc, adaptor.getDst(), dstElemBytes);
    int64_t statusElemBytes =
        cast<MemRefType>(op.getStatus().getType()).getElementTypeBitWidth() / 8;
    Value statusAddr =
        asAddress(rewriter, loc, adaptor.getStatus(), statusElemBytes);
    Value lengthBytes = LLVM::MulOp::create(
        rewriter, loc, i32Ty,
        memrefNumElements(rewriter, loc, adaptor.getDst(), dstType),
        cst(dstElemBytes));

    SmallVector<Value> args{
        LLVM::IntToPtrOp::create(rewriter, loc, ptrTy, srcAddr),
        cst(memrefAddressSpace(srcType, 0)),
        LLVM::IntToPtrOp::create(rewriter, loc, ptrTy, dstAddr),
        cst(memrefAddressSpace(dstType, 1)),
        lengthBytes,
        cst(0), // bypassCacheSrc: the source may be cached DDR
        cst(0), // bypassCacheDst
        LLVM::IntToPtrOp::create(rewriter, loc, ptrTy, statusAddr)};

    // The call's i32 token is the loop-carried handle: the value an external
    // scheduler versions and rotates.
    rewriter.replaceOpWithNewOp<LLVM::CallOp>(
        op, TypeRange{i32Ty}, FlatSymbolRefAttr::get((*fn).getOperation()), args);
    return success();
  }
};

/// `hmx.await` -> `dma_wait(token)`, then the result is the staged slot itself.
/// The wait is unconditional (the matching `hmx.stage` always issued its
/// transfer), and the op is a pure alias: the result IS the `dst` descriptor, so
/// the consumer reads exactly the buffer the DMA wrote -- no copy, no descriptor
/// rebuild.
struct LowerAwait : public ConvertOpToLLVMPattern<AwaitOp> {
  using ConvertOpToLLVMPattern<AwaitOp>::ConvertOpToLLVMPattern;

  LogicalResult matchAndRewrite(AwaitOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();

    auto fn = getDmaWaitLeaf(module, rewriter);
    if (failed(fn))
      return failure();

    Value token = toI32(rewriter, loc, adaptor.getToken());
    replaceWithLeafCall(rewriter, loc, op, adaptor.getDst(), *fn,
                        ValueRange{token});
    return success();
  }
};


struct HmxToLLVMPass : public ::impl::HmxToLLVMBase<HmxToLLVMPass> {
  using Base::Base;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<memref::MemRefDialect, LLVM::LLVMDialect, HmxDialect,
                    func::FuncDialect>();
  }

  void runOnOperation() override {
    auto moduleOp = getOperation();
    MLIRContext *context = moduleOp->getContext();

    // Decide *before* converting which functions issue HMX leaves: the decision
    // is made on the hmx dialect ops, and the conversion below erases every one
    // of them, so it cannot be made afterwards -- and must not be made on callee
    // names (see issuesHmxEngineLeaves). The conversion registers patterns for
    // hmx ops only, so it neither renames nor rebuilds a function; recording
    // the symbol names here identifies the same functions after it, even if a
    // future lowering were to rebuild a function while keeping its name.
    llvm::StringSet<> engineKernels;
    moduleOp.walk([&](LLVM::LLVMFuncOp fn) {
      if (!fn.isDeclaration() && issuesHmxEngineLeaves(fn))
        engineKernels.insert(fn.getName());
    });
    moduleOp.walk([&](func::FuncOp fn) {
      if (!fn.isDeclaration() && issuesHmxEngineLeaves(fn))
        engineKernels.insert(fn.getName());
    });

    const auto &dataLayoutAnalysis = getAnalysis<DataLayoutAnalysis>();

    LLVMConversionTarget target(*context);
    RewritePatternSet patterns(context);
    LowerToLLVMOptions options(context,
                               dataLayoutAnalysis.getAtOrAbove(moduleOp));
    LLVMTypeConverter typeConverter(context, options);

    target.addLegalDialect<memref::MemRefDialect>();
    target.addIllegalDialect<HmxDialect>();

    hmx::populateHmxToLLVMConversionPatterns(typeConverter, patterns);

    if (failed(applyPartialConversion(moduleOp, target, std::move(patterns))))
      signalPassFailure();

    // Invariant check before anything is inserted: a leaf caller the dialect
    // test declined must fail the build loudly, not ship without a pair (see
    // verifyHmxLeafCallers). Existing control flow is untouched: after
    // signalPassFailure() this still runs, exactly as it always has.
    verifyHmxLeafCallers(moduleOp, engineKernels);

    // Every function collected above as issuing HMX leaves has to power the
    // engine on and release it (one ensure/unlock pair per kernel); no other
    // pass knows whether the engine is needed. The decision was made while the
    // hmx ops it is based on still existed -- by now they are leaf calls.
    ensureHmxEngine(moduleOp, engineKernels);
  }
};

} // namespace

void hmx::populateHmxToLLVMConversionPatterns(LLVMTypeConverter &typeConverter,
                                              RewritePatternSet &patterns) {
  patterns.add<LowerMma, LowerAccRead, LowerBiasInit, LowerAccClear, LowerPackAct,
               LowerPackWeight, LowerUnpackAcc, LowerUnpackAccF32, LowerStage,
               LowerAwait>(typeConverter);
}

std::unique_ptr<OperationPass<ModuleOp>> mlir::hmx::createHmxToLLVMPass() {
  return std::make_unique<HmxToLLVMPass>();
}
