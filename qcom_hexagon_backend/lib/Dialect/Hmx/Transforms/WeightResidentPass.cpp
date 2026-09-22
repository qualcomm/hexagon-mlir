//===-- WeightResidentPass.cpp - weights become resident VTCM -------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// Two sources of an HMX weight, one residency mechanism.
//
//   * A weight that is constant at compile time reaches the tile level as the
//     prepacked constant of `matmul-to-hmx`: one-shot bufferization turns it into
//     a `memref.global` plus a `memref.get_global`, and copy canonicalization
//     forwards that global straight into `hmx.matmul` -- the engine, however,
//     reads its weight from VTCM, so the tile level rejects a DDR operand.
//     This pass gives such an operand a VTCM buffer and records where its
//     contents come from. It deliberately does not emit the copy: the buffer's
//     lowering calls the runtime, which allocates the buffer on the first
//     launch, copies the weight in once, pins it against the per-launch
//     deallocation and returns the same address forever after.
//
//   * A *runtime* weight (a function argument) has no compile-time image, so
//     `matmul-to-hmx` bridges it with a `hmx.pack_weight` loop -- paid on every
//     launch. When the `prepackRuntimeWeights` option is on, the host pre-packer
//     (the launcher) writes that argument's bytes already in crouton order; the
//     kernel then only has to get those bytes into VTCM, which is exactly the
//     runtime's one-copy resident entry -- byte-identical to the constant path's
//     copy, so the same residency mechanism covers both. This pass replaces the
//     pack loop with the resident declaration and publishes the weight's slot and
//     layout on the module so the host packs the same permutation the compiler
//     would have (`hmx.weight_prepack`).
//
//     The option is off by default: dropping the pack is only correct when the
//     caller honours the published prepack contract (the launcher does; a raw
//     caller passing row-major bytes does not), so the default keeps the IR
//     byte-identical to before.
//
//   * An N-slice of a runtime weight: a decode kernel splits N across programs,
//     so the bridge packs one column block of a wider `[K, N]` weight. The
//     view's row stride is the whole N, which is what proves the argument's
//     bytes are the whole weight's bytes -- the resident therefore holds the
//     whole `[K, N]` and each `hmx.matmul` reads its block through a
//     `memref.subview`. A view whose whole shape cannot be pinned (an
//     offset that is not provably tile-aligned, or a non-row-major/derived
//     source) keeps the per-launch bridge rather than guessing.
//
// Residency is declared once, in the module attribute
// `hmx.weight_resident_bytes` (the aggregate of the per-buffer byte counts the
// pass computes). Both static VTCM-budget readers (`matmul-to-hmx`,
// `hmx-partition`) consult it, and the runtime receives the same number on the
// lowering's call, so the static budget and the resident footprint cannot
// drift apart.
//
//===----------------------------------------------------------------------===//

#include "hexagon/Common/Common.h"
#include "hexagon/Dialect/HexagonMem/IR/HexagonMemDialect.h"
#include "hexagon/Dialect/Hmx/IR/HmxDialect.h"
#include "hexagon/Dialect/Hmx/Transforms/Transforms.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include <string>

#define DEBUG_TYPE "weight-resident"

using namespace mlir;
using namespace mlir::hmx;

namespace mlir {
namespace hmx {
#define GEN_PASS_DEF_WEIGHTRESIDENT
#include "hexagon/Dialect/Hmx/Transforms/Passes.h.inc"
} // namespace hmx
} // namespace mlir

namespace {

/// Per-buffer residency record on the `hexagonmem.alloc`: where the contents
/// come from and how many bytes. The constant path names a global symbol; the
/// runtime path names the function argument slot. The lowering reads this
/// dictionary; the module attribute below is its aggregate for the static
/// budget readers.
constexpr const char *kResidentAttr = "hmx.weight_resident";
constexpr const char *kResidentKeyGlobal = "global";
constexpr const char *kResidentKeyAddress = "address";
constexpr const char *kResidentKeyBytes = "bytes";

/// Aggregate of every resident buffer in the module, in bytes.
constexpr const char *kResidentBytesAttr = "hmx.weight_resident_bytes";

/// JSON array of the runtime weights the host must pre-pack: each entry names
/// the function, the argument slot, the logical shape and the crouton shape.
constexpr const char *kPrepackAttr = "hmx.weight_prepack";

/// The module attribute carrying the permutation the host must apply.
constexpr const char *kPrepackLayoutAttr = "hmx.weight_prepack_layout";

/// The crouton permutation the compiler applies, as a JSON coefficient map. A
/// weight grid is [Nt, Kt, 16, 32, 2] with logical [N, K], so a physical index
/// (d0=n_tile, d1=k_tile, d2=j, d3=c, d4=h) maps to logical
/// (tile*d1 + half*d2 + d4, tile*d0 + d3). Published so the host packer derives
/// the permutation from compiler metadata instead of re-deriving it, and can
/// fail loudly if the two ever disagree; built from the one tile-edge constant
/// so it cannot drift from the layout.
static std::string prepackLayoutJson() {
  return std::string("{\"ndims\":5,\"results\":[[[1,") +
         std::to_string(hmx::crouton::kTileEdge) + "],[2," +
         std::to_string(hmx::crouton::kCroutonHalf) + "],[4,1]],[[0," +
         std::to_string(hmx::crouton::kTileEdge) + "],[3,1]]]}";
}

/// The prepacked constant a value is loaded from, or null. Only the direct
/// `memref.get_global` form is handled: that is what bufferization produces for
/// the constant fast path of `matmul-to-hmx`.
memref::GlobalOp prepackedSource(Value v) {
  auto getGlobal = v.getDefiningOp<memref::GetGlobalOp>();
  if (!getGlobal)
    return {};
  return SymbolTable::lookupNearestSymbolFrom<memref::GlobalOp>(
      getGlobal, getGlobal.getNameAttr());
}

static int64_t byteSize(MemRefType type) {
  return type.getNumElements() * (type.getElementTypeBitWidth() / 8);
}

/// A runtime weight's pack bridge: the crouton array a `hmx.pack_weight` loop
/// fills, and the writers/loops that can be dropped with it.
struct WeightPack {
  memref::AllocOp array;
  SmallVector<PackWeightOp> packs;
  SmallVector<scf::ForOp> loops;
};

/// Every `hmx.pack_weight` that writes `array` as its destination.
static SmallVector<PackWeightOp> packWeightWriters(Value array) {
  SmallVector<PackWeightOp> packs;
  for (OpOperand &u : array.getUses())
    if (auto p = dyn_cast<PackWeightOp>(u.getOwner()))
      if (p.getDst() == array)
        packs.push_back(p);
  return packs;
}

/// True when `loop`'s body is the weight bridge and nothing else: index
/// arithmetic and pack writes. Erasing such a loop removes exactly the bridge.
static bool isWeightBridgeLoop(scf::ForOp loop) {
  for (Operation &inner : loop.getBody()->without_terminator())
    if (!isa<arith::ConstantIndexOp, arith::DivUIOp, arith::RemUIOp,
             arith::AddIOp, arith::MulIOp, arith::IndexCastOp, PackWeightOp>(
            inner))
      return false;
  return true;
}

/// The runtime-weight bridge behind `v`, or nullopt. Both bufferization shapes
/// are handled: the pack loop writes the allocation in place (the common
/// canonicalized form), or the allocation is the loop's carried init.
static std::optional<WeightPack> findWeightPack(Value v) {
  WeightPack pack;
  if (auto alloc = v.getDefiningOp<memref::AllocOp>()) {
    pack.array = alloc;
    pack.packs = packWeightWriters(v);
  } else if (auto loop = v.getDefiningOp<scf::ForOp>()) {
    if (loop.getNumResults() != 1 || loop.getResult(0) != v ||
        loop.getInitArgs().empty())
      return std::nullopt;
    pack.array = loop.getInitArgs()[0].getDefiningOp<memref::AllocOp>();
    if (!pack.array)
      return std::nullopt;
    for (Operation &inner : loop.getBody()->without_terminator())
      if (auto p = dyn_cast<PackWeightOp>(inner))
        if (p.getDst() == loop.getRegionIterArg(0))
          pack.packs.push_back(p);
    if (!isWeightBridgeLoop(loop))
      return std::nullopt;
    pack.loops.push_back(loop);
  } else {
    return std::nullopt;
  }
  if (!pack.array || pack.packs.empty())
    return std::nullopt;
  // The alloc form keeps its writers inside the same bridge loop; record it so
  // it can be erased with the bridge.
  if (pack.loops.empty()) {
    llvm::DenseSet<scf::ForOp> seen;
    for (PackWeightOp p : pack.packs) {
      scf::ForOp loop = p->getParentOfType<scf::ForOp>();
      if (!loop || !isWeightBridgeLoop(loop))
        return std::nullopt;
      if (seen.insert(loop).second)
        pack.loops.push_back(loop);
    }
  }
  return pack;
}

static std::string jsonArray(ArrayRef<int64_t> values) {
  std::string out = "[";
  for (size_t i = 0; i < values.size(); ++i) {
    if (i)
      out += ",";
    out += std::to_string(values[i]);
  }
  out += "]";
  return out;
}

/// Append one prepack entry (a JSON object) to the module's JSON array.
static void appendPrepackEntry(ModuleOp module, StringRef entry) {
  auto existing = module->getAttrOfType<StringAttr>(kPrepackAttr);
  if (!existing) {
    module->setAttr(kPrepackAttr,
                    StringAttr::get(module.getContext(), "[" + entry.str() + "]"));
    return;
  }
  std::string merged = existing.getValue().str();
  // Strip the closing bracket, add the separator only when non-empty, re-close.
  if (!merged.empty() && merged.back() == ']')
    merged.pop_back();
  if (merged.size() > 1)
    merged += ",";
  merged += entry.str();
  merged += "]";
  module->setAttr(kPrepackAttr, StringAttr::get(module.getContext(), merged));
}

/// Add `bytes` to the module's resident footprint.
static void addResidentBytes(ModuleOp module, int64_t bytes) {
  int64_t total = bytes;
  if (auto existing = module->getAttrOfType<IntegerAttr>(kResidentBytesAttr))
    total += existing.getInt();
  module->setAttr(
      kResidentBytesAttr,
      IntegerAttr::get(IntegerType::get(module.getContext(), 64), total));
}

/// Follow a chain of layout-only views to the underlying function argument.
/// A view qualifies only when it provably covers the whole argument: offset 0
/// and dense row-major strides for its static shape, so the host pre-pack of
/// the whole argument feeds exactly what the bridge would read. The source may
/// be:
///
///   * a ranked memref -- it must have the same static shape and element type
///     as the view, so the view is the whole argument; or
///   * an unranked memref (`memref<*xf16>`) -- how a real Triton kernel receives
///     a weight. There is no static source shape to compare, so the view's
///     static shape *is* the published contract shape. This is only sound
///     because offset 0 + dense strides pin the view to the buffer base; a view
///     that merely asserted a shape over a larger buffer would still be rejected
///     by the stride check unless it covered the buffer densely.
///
/// Anything else returns null and keeps the old bridge: guessing would silently
/// compute on the wrong data. In particular a nonzero *or dynamic* offset is a
/// real slice, so it keeps the per-launch bridge (see the note next to the
/// offset check). Real kernels wrap pack sources in `reinterpret_cast` with a
/// dense-equivalent strided layout, which the bare-BlockArgument check below
/// would otherwise miss entirely.
static BlockArgument underlyingDenseArgument(Value v) {
  for (int depth = 0; depth < 8; ++depth) {
    if (auto arg = dyn_cast<BlockArgument>(v))
      return arg;
    auto reinterpret = v.getDefiningOp<memref::ReinterpretCastOp>();
    if (!reinterpret)
      return {};
    auto viewType = dyn_cast<MemRefType>(v.getType());
    if (!viewType || !viewType.hasStaticShape())
      return {};
    Type srcType = reinterpret.getSource().getType();
    if (auto baseType = dyn_cast<MemRefType>(srcType)) {
      if (!baseType.hasStaticShape() ||
          baseType.getElementType() != viewType.getElementType() ||
          baseType.getShape() != viewType.getShape())
        return {};
    } else if (auto baseType = dyn_cast<UnrankedMemRefType>(srcType)) {
      if (baseType.getElementType() != viewType.getElementType())
        return {};
    } else {
      return {};
    }
    // The view must start at the buffer base. A nonzero *static* offset is a
    // real slice. A *dynamic* offset is the reason P2 does not fire on a real
    // decode kernel: the source view is `offset: [%n]` (the program-id N
    // offset), so the argument's bytes are not the weight's bytes. Pre-packing
    // the whole argument there would silently compute on the wrong data, so it
    // must keep the old bridge. Lifting that is a contract decision, not a
    // local one; the candidates are to hand the kernel the whole weight (no
    // N-offset view), to loop over N inside the kernel, or to carry the offset
    // in the host prepack contract. None is chosen here. Check both the op
    // operands and the view layout so an omitted-operand form cannot slip
    // through.
    for (int64_t off : reinterpret.getStaticOffsets())
      if (off != 0)
        return {};
    if (auto strided = dyn_cast<StridedLayoutAttr>(viewType.getLayout()))
      if (strided.getOffset() != 0)
        return {};
    // Dense row-major strides for the (static) shape.
    int64_t rank = viewType.getRank();
    SmallVector<int64_t> dense(rank, 0);
    int64_t stride = 1;
    for (int64_t i = rank - 1; i >= 0; --i) {
      dense[i] = stride;
      stride *= viewType.getDimSize(i);
    }
    if (auto strided = dyn_cast<StridedLayoutAttr>(viewType.getLayout())) {
      auto strides = strided.getStrides();
      for (int64_t i = 0; i < rank; ++i) {
        if (strides[i] == ShapedType::kDynamic || strides[i] != dense[i])
          return {};
      }
    } else if (!viewType.getLayout().isIdentity()) {
      return {};
    }
    v = reinterpret.getSource();
  }
  return {};
}

/// True when `v` is provably a multiple of `factor`, for the small affine forms
/// a program-id offset takes (`pid * BN`, sums of such). The subview that reads
/// one N block of the resident weight is indexed in croutons, so the block
/// offset has to land on a tile edge; anything not provably a multiple keeps the
/// per-launch bridge rather than truncating a division.
static bool isMultipleOf(Value v, int64_t factor, int depth = 0) {
  if (depth > 8)
    return false;
  if (auto cst = v.getDefiningOp<arith::ConstantIndexOp>())
    return cst.value() % factor == 0;
  if (auto cst = v.getDefiningOp<arith::ConstantOp>())
    if (auto intAttr = dyn_cast<IntegerAttr>(cst.getValue()))
      return intAttr.getInt() % factor == 0;
  if (auto cast = v.getDefiningOp<arith::IndexCastOp>())
    return isMultipleOf(cast.getIn(), factor, depth + 1);
  if (auto cast = v.getDefiningOp<arith::IndexCastUIOp>())
    return isMultipleOf(cast.getIn(), factor, depth + 1);
  if (auto add = v.getDefiningOp<arith::AddIOp>())
    return isMultipleOf(add.getLhs(), factor, depth + 1) &&
           isMultipleOf(add.getRhs(), factor, depth + 1);
  if (auto mul = v.getDefiningOp<arith::MulIOp>())
    return isMultipleOf(mul.getLhs(), factor, depth + 1) ||
           isMultipleOf(mul.getRhs(), factor, depth + 1);
  return false;
}

/// An N-slice of a wider runtime weight (B2): the pack bridge covers one N block
/// of a `[K, N]` matrix. The whole N is the view's row stride, so the argument's
/// bytes *are* the whole weight's bytes and one resident copy serves every
/// program.
struct WeightSlice {
  BlockArgument arg;
  int64_t n = 0;          // whole logical columns (N)
  Value dynamicOffset;    // N block offset in elements, when dynamic
  std::optional<int64_t> staticOffset; // N block offset in elements, when static
};

/// Match the bridge source as `reinterpret_cast(arg, offset=[n0], sizes=[K, BN],
/// strides=[N, 1])` over an entry argument: one N block of a `[K, N]` weight.
/// Returns null for anything whose whole weight cannot be pinned -- not a 2D
/// row-major view, a non-argument source, a dynamic/unaligned offset, a view the
/// crouton grid disagrees with, or a block that does not tile N. The caller then
/// keeps the old per-launch bridge: guessing would silently pack the wrong bytes.
static std::optional<WeightSlice> underlyingSliceArgument(Value v,
                                                          MemRefType crouton) {
  auto reinterpret = v.getDefiningOp<memref::ReinterpretCastOp>();
  if (!reinterpret)
    return std::nullopt;
  auto viewType = dyn_cast<MemRefType>(v.getType());
  if (!viewType || viewType.getRank() != 2 || !viewType.hasStaticShape())
    return std::nullopt;
  // The pre-pack contract is a function-argument contract: the source must be
  // the entry argument itself, not an internal buffer.
  auto arg = dyn_cast<BlockArgument>(reinterpret.getSource());
  if (!arg)
    return std::nullopt;
  // A row-major view with unit inner stride: the underlying matrix is [K, N]
  // with N = stride(0), and the view is one N block of it.
  auto strided = dyn_cast<StridedLayoutAttr>(viewType.getLayout());
  if (!strided)
    return std::nullopt;
  auto strides = strided.getStrides();
  if (strides.size() != 2 || strides[1] != 1 ||
      ShapedType::isDynamic(strides[0]) || strides[0] <= 0)
    return std::nullopt;
  int64_t n = strides[0];
  int64_t k = viewType.getDimSize(0);
  int64_t bn = viewType.getDimSize(1);
  // The crouton bridge already fixed the tile grid; the view has to describe the
  // same [K, BN] block, the whole grid has to be whole croutons, and the block
  // has to tile N exactly so every program's slice lies inside the resident.
  // A weight grid is [Nt, Kt, ...]: dim0 is N, dim1 is K.
  if (crouton.getRank() != 5 || crouton.getDimSize(1) * hmx::crouton::kTileEdge != k ||
      crouton.getDimSize(0) * hmx::crouton::kTileEdge != bn)
    return std::nullopt;
  // Strictly narrower than the whole N: the model is "the view is *one* N block
  // of a wider weight", so a view as wide as the whole N is a block of nothing
  // (there are no other blocks for an offset to select) and any offset into it
  // is not an N offset -- it is the K-block case the offset check below rejects.
  // One N block means at least two.
  if (n % hmx::crouton::kTileEdge != 0 || bn % hmx::crouton::kTileEdge != 0 || n <= bn ||
      n % bn != 0)
    return std::nullopt;
  // The offset is the descriptor's element offset (a one-element list), i.e. the
  // N block this program owns. A static offset is checked directly; a dynamic
  // one has to be provably tile-aligned (`pid * BN`) *and* provably a column
  // offset: in a row-major [K, N] matrix every whole number of rows is a
  // multiple of N, so an offset that is provably a multiple of N is a row (K)
  // offset -- a loop-varying block of an activation consumed as a weight, whose
  // resident holds one block while the offset walks past its end. A column
  // offset is never such a multiple (offset 0 is the dense path's business).
  if (reinterpret.getStaticOffsets().size() != 1)
    return std::nullopt;
  WeightSlice slice;
  slice.arg = arg;
  slice.n = n;
  int64_t staticOffset = reinterpret.getStaticOffsets().front();
  if (staticOffset != ShapedType::kDynamic) {
    if (staticOffset < 0 || staticOffset % hmx::crouton::kTileEdge != 0 ||
        staticOffset + bn > n)
      return std::nullopt;
    slice.staticOffset = staticOffset;
  } else {
    if (reinterpret.getOffsets().size() != 1)
      return std::nullopt;
    Value offset = reinterpret.getOffsets().front();
    if (!isMultipleOf(offset, hmx::crouton::kTileEdge) || isMultipleOf(offset, n))
      return std::nullopt;
    slice.dynamicOffset = offset;
  }
  return slice;
}

struct WeightResidentPass
    : public mlir::hmx::impl::WeightResidentBase<WeightResidentPass> {
  using WeightResidentBase::WeightResidentBase;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, hexagonmem::HexagonMemDialect,
                    hmx::HmxDialect, memref::MemRefDialect, scf::SCFDialect>();
  }

  void runOnOperation() override {
    func::FuncOp func = cast<func::FuncOp>(getOperation());
    ModuleOp module = func->getParentOfType<ModuleOp>();
    if (!module)
      return;

    SmallVector<MatmulOp> matmuls;
    func.walk([&](MatmulOp op) { matmuls.push_back(op); });
    if (matmuls.empty())
      return;

    IRRewriter rewriter(func.getContext());
    // One resident buffer per distinct constant, no matter how many matmuls
    // read it (the runtime keys on the same source address anyway).
    llvm::DenseMap<Attribute, Value> residentBySource;
    // One resident buffer per runtime weight slot.
    llvm::DenseMap<int64_t, Value> residentBySlot;
    int64_t addedBytes = 0;

    for (MatmulOp op : matmuls) {
      for (OpOperand &operand : op->getOpOperands()) {
        MemRefType type = dyn_cast<MemRefType>(operand.get().getType());
        if (!type)
          continue;
        // Only the inputs carry a prepacked weight; the output is written.
        if (operand.getOperandNumber() >= 2)
          continue;

        // --- Constant path: the prepacked `memref.global`. ---
        if (memref::GlobalOp source = prepackedSource(operand.get())) {
          // The constant has to keep existing until the lowering reads its
          // address, so make it a public symbol rather than letting symbol DCE
          // drop an operand-less private global.
          source->setAttr(SymbolTable::getVisibilityAttrName(),
                          rewriter.getStringAttr("public"));

          FlatSymbolRefAttr symbol =
              FlatSymbolRefAttr::get(source.getSymNameAttr());
          Value resident = residentBySource.lookup(symbol);
          if (!resident) {
            int64_t bytes = byteSize(type);
            auto vtcmType =
                MemRefType::get(type.getShape(), type.getElementType(),
                                AffineMap{}, hexagon::VTCM_ADDRESS_SPACE);
            rewriter.setInsertionPoint(op);
            auto alloc = hexagonmem::AllocOp::create(
                rewriter, op.getLoc(), vtcmType, ValueRange{},
                rewriter.getI64IntegerAttr(128));
            alloc->setAttr(
                kResidentAttr,
                rewriter.getDictionaryAttr(
                    {rewriter.getNamedAttr(
                         kResidentKeyGlobal, FlatSymbolRefAttr::get(
                                                 source.getSymNameAttr())),
                     rewriter.getNamedAttr(
                         kResidentKeyBytes,
                         rewriter.getI64IntegerAttr(bytes))}));
            resident = alloc.getResult();
            residentBySource.insert({symbol, resident});
            addedBytes += bytes;
            LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE "] resident " << symbol
                                    << " (" << bytes << " bytes)\n");
          }

          rewriter.modifyOpInPlace(op, [&]() { operand.set(resident); });
          continue;
        }

        // --- Runtime path: the host pre-packs a function argument. ---
        // Only the weight (rhs) is a resident candidate; the activation (lhs) is
        // packed fresh every launch and stays on the existing bridge.
        if (!prepackRuntimeWeights || operand.getOperandNumber() != 1)
          continue;

        std::optional<WeightPack> pack = findWeightPack(operand.get());
        if (!pack)
          continue;
        // The bridge must pack one runtime input, not an internal buffer. It
        // usually reads layout-only views (`reinterpret_cast` from
        // bufferization); resolve those to the entry argument and to the *whole*
        // weight the resident buffer has to hold.
        //
        // Two forms reach here. A dense view covers the whole argument, so the
        // resident is the argument's own crouton array. An N-slice (B2) is one
        // column block of a wider `[K, N]` weight -- the view's row stride is the
        // whole N -- so the argument's bytes *are* the whole weight's bytes and a
        // single resident copy serves every program; the matmul reads its block
        // through a subview. A view that proves neither keeps the per-launch
        // bridge: pre-packing the whole argument for a partial view would
        // silently compute on the wrong bytes.
        Value src = pack->packs.front().getSrc();
        if (llvm::any_of(pack->packs,
                         [&](PackWeightOp p) { return p.getSrc() != src; }))
          continue;
        // The pre-pack contract is defined on an fp16 weight: the host permutes
        // the weight's own bytes, so the resident holds exactly what this
        // device-side pack would have written. A wider weight would need the
        // host to quantise it to the same fp16 the crouton holds, which is a
        // different contract -- leave such a weight on the per-launch bridge,
        // whose pack does quantise (see hmx.pack_weight).
        auto srcMemref = dyn_cast<MemRefType>(src.getType());
        if (!srcMemref || !srcMemref.getElementType().isF16())
          continue;

        BlockArgument arg;
        MemRefType residentType;
        int64_t logicalN = type.getDimSize(0) * hmx::crouton::kTileEdge;
        std::optional<WeightSlice> slice;
        if (BlockArgument denseArg = underlyingDenseArgument(src)) {
          arg = denseArg;
          residentType = MemRefType::get(type.getShape(), type.getElementType(),
                                         AffineMap{},
                                         hexagon::VTCM_ADDRESS_SPACE);
        } else if ((slice = underlyingSliceArgument(src, type))) {
          arg = slice->arg;
          logicalN = slice->n;
          SmallVector<int64_t> wholeShape(type.getShape().begin(),
                                          type.getShape().end());
          wholeShape[0] = slice->n / hmx::crouton::kTileEdge;
          residentType = MemRefType::get(wholeShape, type.getElementType(),
                                         AffineMap{},
                                         hexagon::VTCM_ADDRESS_SPACE);
        } else {
          continue;
        }
        if (arg.getOwner() != &func.getBody().front())
          continue;
        int64_t slot = arg.getArgNumber();

        Value resident = residentBySlot.lookup(slot);
        // One resident per slot: if the same argument resolved to a different
        // whole shape here the two would alias, so keep this op's bridge.
        if (resident && cast<MemRefType>(resident.getType()) != residentType)
          continue;
        if (!resident) {
          int64_t bytes = byteSize(residentType);
          rewriter.setInsertionPoint(op);
          // The residency key: the argument's aligned pointer. It is passed as
          // an operand (the allocation itself stays static; the verifier allows
          // this one resident-only operand) so the lowering reads it after the
          // function conversion, when a block argument is no longer a single
          // descriptor. Read it off the argument itself: the view proved
          // identical above, and the argument survives lowering more robustly.
          Value address = memref::ExtractAlignedPointerAsIndexOp::create(
              rewriter, op.getLoc(), arg);
          auto alloc = hexagonmem::AllocOp::create(
              rewriter, op.getLoc(), residentType, ValueRange{address},
              rewriter.getI64IntegerAttr(128));
          alloc->setAttr(
              kResidentAttr,
              rewriter.getDictionaryAttr(
                  {rewriter.getNamedAttr(kResidentKeyAddress,
                                         rewriter.getUnitAttr()),
                   rewriter.getNamedAttr(kResidentKeyBytes,
                                         rewriter.getI64IntegerAttr(bytes))}));
          resident = alloc.getResult();
          residentBySlot.insert({slot, resident});
          addedBytes += bytes;

          // Publish the pre-pack contract: the host packs the whole argument,
          // and the permutation comes from the compiler's own layout map. The
          // logical shape is reconstructed from the crouton grid -- the whole
          // `[K, N]` for an N-slice, the argument exactly for a dense weight.
          // A weight grid is [Nt, Kt, ...], so K is dim1 and N is `logicalN`.
          SmallVector<int64_t> logical{residentType.getDimSize(1) * hmx::crouton::kTileEdge,
                                       logicalN};
          std::string entry =
              "{\"func\":\"" + func.getSymName().str() + "\",\"slot\":" +
              std::to_string(slot) + ",\"shape\":" + jsonArray(logical) +
              ",\"crouton\":" + jsonArray(residentType.getShape()) +
              ",\"dtype\":\"f16\"}";
          appendPrepackEntry(module, entry);
          if (!module->getAttrOfType<StringAttr>(kPrepackLayoutAttr))
            module->setAttr(kPrepackLayoutAttr,
                            rewriter.getStringAttr(prepackLayoutJson()));
          LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE "] resident runtime slot "
                                  << slot << " (" << bytes << " bytes)\n");
        }

        // The engine reads the block this program owns: the resident itself for a
        // dense whole argument, or a subview over the crouton grid's N tiles for
        // an N-slice. A weight grid is [Nt, Kt, ...] (dim0 = N, dim1 = K), so the
        // N block offset lands on dim0.
        Value rhs = resident;
        if (slice) {
          rewriter.setInsertionPoint(op);
          // The block's N offset in croutons: a static offset was proven
          // tile-aligned when it matched, a dynamic one is divided (exactly, by
          // the same proof).
          Value n0Crouton;
          if (slice->dynamicOffset) {
            Value edge = arith::ConstantIndexOp::create(rewriter, op.getLoc(),
                                                        hmx::crouton::kTileEdge);
            n0Crouton = arith::DivUIOp::create(rewriter, op.getLoc(),
                                               slice->dynamicOffset, edge);
          } else {
            n0Crouton = arith::ConstantIndexOp::create(
                rewriter, op.getLoc(), *slice->staticOffset / hmx::crouton::kTileEdge);
          }
          SmallVector<OpFoldResult> offsets{n0Crouton, rewriter.getIndexAttr(0),
                                            rewriter.getIndexAttr(0),
                                            rewriter.getIndexAttr(0),
                                            rewriter.getIndexAttr(0)};
          SmallVector<OpFoldResult> sizes{
              rewriter.getIndexAttr(type.getDimSize(0)),
              rewriter.getIndexAttr(type.getDimSize(1)),
              rewriter.getIndexAttr(type.getDimSize(2)),
              rewriter.getIndexAttr(type.getDimSize(3)),
              rewriter.getIndexAttr(type.getDimSize(4))};
          SmallVector<OpFoldResult> strides(5, rewriter.getIndexAttr(1));
          rhs = memref::SubViewOp::create(rewriter, op.getLoc(), resident,
                                          offsets, sizes, strides);
        }
        rewriter.modifyOpInPlace(op, [&]() { operand.set(rhs); });

        // The bridge is now dead: drop the pack loop, the crouton array and its
        // deallocation. Everything was verified to be the bridge before this
        // point, so nothing else can observe the buffer. The pack ops go with
        // their loop when it is the bridge and nothing else.
        for (scf::ForOp loop : pack->loops)
          if (loop.use_empty())
            rewriter.eraseOp(loop);
        SmallVector<memref::DeallocOp> deallocs;
        for (Operation *user : pack->array->getUsers())
          if (auto d = dyn_cast<memref::DeallocOp>(user))
            deallocs.push_back(d);
        for (memref::DeallocOp d : deallocs)
          rewriter.eraseOp(d);
        if (pack->array->use_empty())
          rewriter.eraseOp(pack->array);
      }
    }

    if (addedBytes)
      addResidentBytes(module, addedBytes);
  }
};

} // namespace

std::unique_ptr<InterfacePass<FunctionOpInterface>>
mlir::hmx::createWeightResidentPass(
    const mlir::hmx::WeightResidentOptions &options) {
  return std::make_unique<WeightResidentPass>(options);
}
