//===- HmxOps.cpp - HMX Ops  ----------------------------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

#include "hexagon/Common/Common.h"

#include "hexagon/Dialect/Hmx/IR/HmxDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "llvm/ADT/SmallVector.h"

using namespace mlir;
using namespace mlir::hmx;

/// Dialect creation, the instance will be owned by the context. This is the
/// point of registration of custom operations for the dialect.
void HmxDialect::registerOperations() {
  addOperations<
#define GET_OP_LIST
#include "hexagon/Dialect/Hmx/IR/HmxOps.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// ODS-Generated Declarations
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "hexagon/Dialect/Hmx/IR/HmxOps.cpp.inc"

namespace {

/// One crouton as it appears in the IR: 16 pairs x 32 columns x 2 halves of a
/// pair. It is what `linalg.pack` produces from a 32x32 fp16 tile
/// (see docs/hmx/hmx-system-design.md §10.1).
constexpr int64_t kCroutonDims[3] = {crouton::kCroutonPair, crouton::kCroutonCol,
                                     crouton::kCroutonHalf};

/// The conversion-state block the bias registers are loaded from: 256 B, which
/// is also the HMX_BIAS_BYTES the runtime uses.
constexpr int64_t kConvStateBytes = 256;

LogicalResult verifyVtcm(Operation *op, Value v, StringRef name) {
  auto memrefType = dyn_cast<MemRefType>(v.getType());
  if (!memrefType)
    return op->emitOpError() << name << " must be a memref";

  if (memrefType.getMemorySpaceAsInt() != hexagon::VTCM_ADDRESS_SPACE)
    return op->emitOpError() << name << " must be in VTCM (memory space "
                             << hexagon::VTCM_ADDRESS_SPACE << ")";

  return success();
}

LogicalResult verifyCroutonArray(Operation *op, Value v, StringRef name) {
  auto memrefType = dyn_cast<MemRefType>(v.getType());
  if (!memrefType)
    return op->emitOpError() << name << " must be a memref";

  if (memrefType.getMemorySpaceAsInt() != hexagon::VTCM_ADDRESS_SPACE)
    return op->emitOpError() << name << " must be in VTCM (memory space "
                             << hexagon::VTCM_ADDRESS_SPACE << ")";

  if (!memrefType.getElementType().isF16())
    return op->emitOpError() << name << " must have f16 elements";

  if (!memrefType.hasStaticShape() || memrefType.getRank() != 5)
    return op->emitOpError()
           << name << " must be a 2D grid of croutons: memref<Mt x Kt x 16 x 32 "
                      "x 2 x f16>";

  unsigned last = memrefType.getRank() - 3;
  if (memrefType.getDimSize(last) != kCroutonDims[0] ||
      memrefType.getDimSize(last + 1) != kCroutonDims[1] ||
      memrefType.getDimSize(last + 2) != kCroutonDims[2])
    return op->emitOpError() << name << " must end in a crouton: ..., 16, 32, 2";

  // When the layout survived bufferization (`#hmx.crouton_memref_layout`), it
  // carries the logical matrix and must agree with the grid it is attached to
  // (the same cross-check the tensor encoding makes at construction). Without
  // the layout the shape alone is the contract, exactly as before.
  if (auto layout =
          dyn_cast_or_null<CroutonMemRefLayoutAttr>(memrefType.getLayout())) {
    if (layout.getLogical()[0] !=
            memrefType.getDimSize(0) * crouton::kTileEdge ||
        layout.getLogical()[1] !=
            memrefType.getDimSize(1) * crouton::kTileEdge)
      return op->emitOpError()
             << name << " carries #hmx.crouton_memref_layout with logical ["
             << layout.getLogical()[0] << ", " << layout.getLogical()[1]
             << "] but the grid is [" << memrefType.getDimSize(0) << ", "
             << memrefType.getDimSize(1) << "]";
  }

  return success();
}

LogicalResult verifyConvState(Operation *op, Value v, StringRef name) {
  if (failed(verifyVtcm(op, v, name)))
    return failure();

  auto memrefType = cast<MemRefType>(v.getType());
  if (!memrefType.getElementType().isInteger(8) ||
      !memrefType.hasStaticShape() || memrefType.getRank() != 1 ||
      memrefType.getDimSize(0) != kConvStateBytes)
    return op->emitOpError() << name
                             << " must be the 256-byte conversion state: "
                                "memref<256xi8> in VTCM";

  return success();
}

/// The direction of a layout materialisation: `crouton` is the side that is (or
/// will be) a crouton array, `rowMajor` the side that is (or will be) an
/// ordinary rank-2 matrix.
///
/// `#hmx.crouton` is a tensor-stage annotation: after bufferization the memrefs
/// carry only the rank-5 shape, so the memref form is deliberately not
/// constrained here. The check is likewise inert while neither side carries the
/// encoding -- that is the pre-migration state, where the layout is still
/// expressed by the shape alone. As soon as an encoding is present it must be on
/// the crouton side, and its `logical` shape must agree with the row-major side:
/// the two views have to describe the same matrix.
///
/// `transposed` selects the orientation of that agreement. It is false when the
/// crouton `logical` shape *is* the row-major shape (activation, read-out): the
/// engine walks the grid's second dim and that dim is the row-major second dim.
/// It is true for the weight, which is stored as Wᵀ (`logical = [N, K]` for a
/// row-major `[K, N]` source) so that K lands on the grid's second dim too; see
/// docs/hmx/hmx-weight-layout-plan.md section 0.
LogicalResult verifyCroutonDirection(Operation *op, Value crouton,
                                     Value rowMajor, StringRef croutonName,
                                     StringRef rowMajorName,
                                     bool transposed = false) {
  if (getCroutonEncoding(rowMajor.getType()))
    return op->emitOpError()
           << rowMajorName
           << " is the row-major side and must not carry #hmx.crouton";

  CroutonLayoutAttr encoding = getCroutonEncoding(crouton.getType());
  if (!encoding)
    return success();

  auto rowMajorType = dyn_cast<RankedTensorType>(rowMajor.getType());
  if (!rowMajorType)
    return op->emitOpError() << croutonName
                             << " carries #hmx.crouton but " << rowMajorName
                             << " is not a tensor";
  if (!rowMajorType.hasStaticShape() || rowMajorType.getRank() != 2)
    return op->emitOpError()
           << rowMajorName << " must be a static rank-2 matrix next to "
           << croutonName << " in crouton layout";
  int64_t expectedRows = rowMajorType.getDimSize(transposed ? 1 : 0);
  int64_t expectedCols = rowMajorType.getDimSize(transposed ? 0 : 1);
  if (encoding.getLogical()[0] != expectedRows ||
      encoding.getLogical()[1] != expectedCols)
    return op->emitOpError()
           << "crouton logical shape [" << encoding.getLogical()[0] << ", "
           << encoding.getLogical()[1] << "] must match "
           << (transposed ? "the transpose of " : "") << rowMajorName
           << " shape [" << rowMajorType.getDimSize(0) << ", "
           << rowMajorType.getDimSize(1) << "]";

  return success();
}

/// Upper bound of an op's optional bulk range: the leaf walks `count` crouton
/// units in one call, so a range past the axis's extent would write past the
/// array. `IntPositive` already rejects zero and negatives.
LogicalResult verifyCount(Operation *op, std::optional<uint64_t> count,
                          int64_t limit, StringRef axis) {
  if (count && *count > static_cast<uint64_t>(limit))
    return op->emitOpError() << "count " << *count << " exceeds the " << axis
                             << " extent " << limit;
  return success();
}

/// Structural legality of a pack source: the op contracts a *row-major* matrix,
/// so its rows must be at least as far apart as it is wide. This is the
/// invariant the lowering preserves by threading `stride(rank-2)` into the leaf
/// as `src_stride`: a source that is one tile of a wider matrix is a strided
/// view with a *larger* row stride (`memref<M x BN, strided<[N, 1]>>` with `N >=
/// BN`), and a lowering that hard-wires the width (`cols`/`n`) reads the wrong
/// element of every row past the first. Tensors (pre-bufferization), dynamic
/// shapes and dynamic strides are not constrained here.
LogicalResult verifyPackSource(Operation *op, Value rowMajor, StringRef name) {
  auto memrefType = dyn_cast<MemRefType>(rowMajor.getType());
  if (!memrefType || !memrefType.hasStaticShape() || memrefType.getRank() < 2)
    return success();

  SmallVector<int64_t, 2> strides;
  int64_t offset;
  if (failed(memrefType.getStridesAndOffset(strides, offset)))
    return success();

  int64_t rowStride = strides[strides.size() - 2];
  int64_t width = memrefType.getDimSize(memrefType.getRank() - 1);
  if (!ShapedType::isDynamic(rowStride) && rowStride < width)
    return op->emitOpError()
           << name << " row stride " << rowStride << " is smaller than its "
           << width
           << " columns: a row-major pack source cannot have overlapping rows";

  return success();
}

/// Structural legality of `hmx.matmul`: the crouton grid. The element type is
/// already enforced by the type constraints, so this checks that the operands are
/// croutons and that they actually form a product:
///
///   lhs: [Mt, Kt, 16, 32, 2]   rhs: [Nt, Kt, 16, 32, 2]   out: [Mt, Nt, 16, 32, 2]
///
/// `rhs` holds Wᵀ: its `logical` shape is [N, K], so the contraction is
/// `lhs[1] == rhs[1]` (K) and the output tile grid is `[lhs[0], rhs[0]]`.
///
/// Whether the op is *worth* using on a given target is decided by the
/// `matmul-to-hmx` pass, which also weighs the VTCM budget -- that is a different
/// question from what the op means.
LogicalResult verifyMatmul(Operation *op, Value lhs, Value rhs, Value out) {
  auto lhsType = dyn_cast<ShapedType>(lhs.getType());
  auto rhsType = dyn_cast<ShapedType>(rhs.getType());
  auto outType = dyn_cast<ShapedType>(out.getType());
  if (!lhsType || !rhsType || !outType)
    return op->emitOpError("expects shaped operands");

  auto isCrouton = [](ShapedType t) {
    return t.hasStaticShape() && t.getRank() == 5 &&
           t.getDimSize(2) == kCroutonDims[0] &&
           t.getDimSize(3) == kCroutonDims[1] &&
           t.getDimSize(4) == kCroutonDims[2];
  };
  if (!isCrouton(lhsType) || !isCrouton(rhsType) || !isCrouton(outType))
    return op->emitOpError()
           << "expects croutons: [., ., " << kCroutonDims[0] << ", "
           << kCroutonDims[1] << ", " << kCroutonDims[2] << "]";

  // rhs stores Wᵀ, so K is its second grid dim (see weightKTiles).
  if (lhsType.getDimSize(1) != rhsType.getDimSize(1))
    return op->emitOpError("inner tile counts must agree");
  if (outType.getDimSize(0) != lhsType.getDimSize(0) ||
      outType.getDimSize(1) != rhsType.getDimSize(0))
    return op->emitOpError("output tile count must be [Mt, Nt]");

  return success();
}

} // namespace

LogicalResult BiasInitOp::verify() {
  return verifyConvState(getOperation(), getBias(), "bias");
}

LogicalResult MatmulOp::verify() {
  Operation *op = getOperation();
  return verifyMatmul(op, getLhs(), getRhs(), getOuts());
}

LogicalResult AllocCroutonOp::verify() {
  auto type = dyn_cast<RankedTensorType>(getResult().getType());
  if (!type || !type.hasStaticShape() || type.getRank() != 5)
    return emitOpError() << "result must be a static rank-5 crouton array";
  if (type.getDimSize(2) != kCroutonDims[0] ||
      type.getDimSize(3) != kCroutonDims[1] ||
      type.getDimSize(4) != kCroutonDims[2])
    return emitOpError() << "result must end in a crouton: ..., 16, 32, 2";

  // When the `#hmx.crouton` encoding is present it self-verifies at type
  // construction (VerifiableTensorEncoding), including `logical == grid * 32`;
  // only its identity needs pinning here. When it was erased (`drop-encodings`,
  // the default) the shape above is the whole contract -- exactly the state the
  // pipeline was in before the encoding existed.
  if (type.getEncoding() && !getCroutonEncoding(type))
    return emitOpError() << "result encoding must be #hmx.crouton";

  return success();
}

LogicalResult MmaOp::verify() {
  Operation *op = getOperation();
  if (failed(verifyCroutonArray(op, getAct(), "act")) ||
      failed(verifyCroutonArray(op, getWt(), "wt")))
    return failure();

  auto actType = cast<MemRefType>(getAct().getType());
  auto wtType = cast<MemRefType>(getWt().getType());
  // K is the grid's second dim on both sides (the weight is stored as Wᵀ).
  if (actType.getDimSize(1) != weightKTiles(wtType))
    return op->emitOpError("the K tile counts must agree");

  if (getNCroutons() < 1)
    return op->emitOpError("n_croutons must be at least 1");
  // The engine's Rt[dC] is five bits (V81 PRM 4.2.1: at most 32 croutons = 1024
  // input channels). A larger count would silently overflow into the spatial
  // mask in the runtime leaf, so reject it here.
  if (getNCroutons() > 32)
    return op->emitOpError("n_croutons must be at most 32");

  return success();
}

LogicalResult AccReadOp::verify() {
  Operation *op = getOperation();
  if (failed(verifyConvState(op, getBias(), "bias")) ||
      failed(verifyCroutonArray(op, getDst(), "dst")))
    return failure();

  if (getBiasSet() < 0 || getBiasSet() > 3)
    return op->emitOpError("bias_set must be in [0, 3]");

  return success();
}

//===----------------------------------------------------------------------===//
// The layout materialisation boundary
//===----------------------------------------------------------------------===//
//
// These three ops are the only conversions between row-major and crouton
// layouts (docs/hmx/layout-representation-survey.md section 4.4), so the
// direction is part of their contract: pack goes row-major -> crouton, unpack
// goes crouton -> row-major. See `verifyCroutonDirection` for why the check is
// gated on the encoding being present.

LogicalResult PackActOp::verify() {
  if (failed(verifyCroutonDirection(getOperation(), getDst(), getSrc(), "dst",
                                    "src")))
    return failure();
  if (failed(verifyPackSource(getOperation(), getSrc(), "src")))
    return failure();
  // The bulk range runs along the activation grid's contiguous (K) axis.
  auto dstType = dyn_cast<ShapedType>(getDst().getType());
  return verifyCount(getOperation(), getCount(),
                     dstType ? dstType.getDimSize(1) : 1, "K tile");
}

LogicalResult PackWeightOp::verify() {
  // The weight is stored as Wᵀ: the crouton `dst` has logical [N, K] while the
  // row-major `src` is [K, N], so the logical shape is the source's transpose
  // (unlike `pack_act`, whose logical shape is the source shape).
  if (failed(verifyCroutonDirection(getOperation(), getDst(), getSrc(), "dst",
                                    "src", /*transposed=*/true)))
    return failure();
  if (failed(verifyPackSource(getOperation(), getSrc(), "src")))
    return failure();
  // The bulk range runs along the weight grid's contiguous (K) axis, dim 1 of
  // the [Nt, Kt] array.
  auto dstType = dyn_cast<ShapedType>(getDst().getType());
  return verifyCount(getOperation(), getCount(),
                     dstType ? dstType.getDimSize(1) : 1, "K tile");
}

LogicalResult UnpackAccOp::verify() {
  if (failed(verifyCroutonDirection(getOperation(), getSrc(), getDst(), "src",
                                    "dst")))
    return failure();
  // A tile row holds exactly 16 row-pairs; `col` selects the first.
  return verifyCount(getOperation(), getCount(), crouton::kCroutonPair,
                     "row-pair");
}

LogicalResult UnpackAccF32Op::verify() {
  Operation *op = getOperation();
  // Bufferized form is the VTCM crouton array; tensor form (what the bridge
  // emits) is the rank-5 f16 tensor. `hmx.unpack_acc` accepts the latter
  // through the encoding-gated direction check alone; the f32 side here needs
  // its shape pinned in both forms, so the check is form-aware.
  if (isa<MemRefType>(getSrc().getType())) {
    if (failed(verifyCroutonArray(op, getSrc(), "src")))
      return failure();
  } else if (auto tensor = dyn_cast<RankedTensorType>(getSrc().getType())) {
    if (!tensor.getElementType().isF16() || !tensor.hasStaticShape() ||
        tensor.getRank() != 5 || tensor.getDimSize(2) != 16 ||
        tensor.getDimSize(3) != 32 || tensor.getDimSize(4) != 2)
      return op->emitOpError()
             << "src must be a crouton tensor: [., ., 16, 32, 2] x f16";
  } else {
    return op->emitOpError() << "src must be a memref or ranked tensor";
  }
  if (failed(verifyCroutonDirection(op, getSrc(), getDst(), "src", "dst")))
    return failure();

  auto isF32Matrix = [&](Value v, StringRef name) -> LogicalResult {
    auto shaped = dyn_cast<ShapedType>(v.getType());
    if (!shaped || !shaped.getElementType().isF32()) {
      op->emitOpError() << name << " must have f32 elements";
      return failure();
    }
    if (auto memref = dyn_cast<MemRefType>(v.getType())) {
      if (!memref.hasStaticShape() || memref.getRank() != 2) {
        op->emitOpError()
            << name << " must be a static rank-2 matrix in memref form";
        return failure();
      }
    } else if (auto tensor = dyn_cast<RankedTensorType>(v.getType())) {
      if (!tensor.hasStaticShape() || tensor.getRank() != 2) {
        op->emitOpError()
            << name << " must be a static rank-2 matrix in tensor form";
        return failure();
      }
    } else {
      op->emitOpError() << name << " must be a memref or ranked tensor";
      return failure();
    }
    return success();
  };
  if (failed(isF32Matrix(getDst(), "dst")))
    return failure();

  // The residual is an elementwise add into the result: same type, same shape.
  if (Value res = getResidual())
    if (res.getType() != getDst().getType())
      return op->emitOpError()
             << "residual type " << res.getType() << " must match dst type "
             << getDst().getType();

  // Same 16-row-pair tile row as `hmx.unpack_acc`.
  if (failed(verifyCount(op, getCount(), crouton::kCroutonPair, "row-pair")))
    return failure();

  return success();
}

//===----------------------------------------------------------------------===//
// Asynchronous staging
//===----------------------------------------------------------------------===//
//
// `hmx.stage`/`hmx.await` expose one tile transfer as an SSA edge
// (docs/hmx/hmx-scheduling-interface-plan.md section 9.2). The verifiers pin the
// two buffers to the shapes the lowering assumes: a rank-2 row-major source and
// a VTCM slot one crouton tile tall and exactly as wide.

LogicalResult StageOp::verify() {
  Operation *op = getOperation();

  // The source is an ordinary row-major matrix. Its row stride is what the
  // lowering threads to the DMA, so a tile cut out of a wider matrix is just a
  // strided view; nothing beyond "static rank-2 f16/f32" is needed here.
  auto srcType = dyn_cast<MemRefType>(getSrc().getType());
  if (!srcType || srcType.getRank() != 2 || !srcType.hasStaticShape() ||
      !(srcType.getElementType().isF16() || srcType.getElementType().isF32()))
    return op->emitOpError()
           << "src must be a static rank-2 f16/f32 memref (row-major source)";

  // The slot is the VTCM buffer the DMA writes and the pack reads: one crouton
  // tile tall (32 rows) and as wide as the source, in the source's own element
  // type. A different width would make the staged tile disagree with the
  // source row it was copied from.
  if (failed(verifyVtcm(op, getDst(), "dst")))
    return failure();
  auto dstType = cast<MemRefType>(getDst().getType());
  if (!dstType.getElementType().isF16() && !dstType.getElementType().isF32())
    return op->emitOpError()
           << "dst must be a static VTCM f16/f32 slot: memref<32xKxelem, 1>";
  if (dstType.getElementType() != srcType.getElementType())
    return op->emitOpError()
           << "dst element type must match the source element type";
  if (dstType.getDimSize(0) != crouton::kTileEdge)
    return op->emitOpError()
           << "dst must be one crouton tile tall (" << crouton::kTileEdge
           << " rows), got " << dstType.getDimSize(0);
  if (dstType.getDimSize(1) != srcType.getDimSize(1))
    return op->emitOpError()
           << "dst columns (" << dstType.getDimSize(1)
           << ") must match the source columns (" << srcType.getDimSize(1)
           << ")";

  // The status word is exactly one i32 the runtime DMA writes the completion
  // token into.
  auto statusType = dyn_cast<MemRefType>(getStatus().getType());
  if (!statusType || !statusType.getElementType().isInteger(32) ||
      !statusType.hasStaticShape() || statusType.getRank() != 1 ||
      statusType.getDimSize(0) != 1)
    return op->emitOpError() << "status must be one i32 word: memref<1xi32>";

  return success();
}

LogicalResult AwaitOp::verify() {
  Operation *op = getOperation();
  if (failed(verifyVtcm(op, getDst(), "dst")))
    return failure();

  auto dstType = cast<MemRefType>(getDst().getType());
  if (!dstType.getElementType().isF16() && !dstType.getElementType().isF32())
    return op->emitOpError() << "dst must have f16 or f32 elements";

  // The result is the slot itself: that value edge is what the compute side
  // consumes, so it must describe the same buffer.
  if (getResult().getType() != getDst().getType())
    return op->emitOpError()
           << "result type " << getResult().getType() << " must match dst type "
           << getDst().getType();

  return success();
}
