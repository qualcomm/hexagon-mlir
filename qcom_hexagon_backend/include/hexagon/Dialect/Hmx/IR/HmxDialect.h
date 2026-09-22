//===- HmxDialect.h - HMX Dialect  ----------------------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

#ifndef HEXAGON_DIALECT_HMX_IR_HMX_DIALECT_H
#define HEXAGON_DIALECT_HMX_IR_HMX_DIALECT_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/TensorEncoding.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "hexagon/Dialect/Hmx/IR/HmxCroutonLayout.h"

//===----------------------------------------------------------------------===//
// HMX Dialect
//===----------------------------------------------------------------------===//
#include "hexagon/Dialect/Hmx/IR/HmxDialect.h.inc"

//===----------------------------------------------------------------------===//
// HMX Attributes
//===----------------------------------------------------------------------===//
#define GET_ATTRDEF_CLASSES
#include "hexagon/Dialect/Hmx/IR/HmxAttrs.h.inc"

//===----------------------------------------------------------------------===//
// Layout contract
//===----------------------------------------------------------------------===//
//
// The HMX engine's data layout as one contract, not three conventions.
//
// A crouton array is a 2D grid of 32x32 fp16 croutons whose trailing dims are
// exactly {16, 32, 2}, living in VTCM. The three roles the engine uses it in --
// activation (AH), weight (WH) and read-out (AR) -- are the *same* permutation,
// verified bit-exactly on device: a matmul chained off another matmul reads the
// producer's AR array as its activation with no conversion, and the two agree on
// every element (see docs/hmx/hmx-generality.md and STATE-OF-PLAY section 4).
// So compatibility is "same logical shape and element type", and the role is a
// property of the *use*, not of the type.
//
// The layout is now carried **in the type**, as the `#hmx.crouton` tensor
// encoding (a `VerifiableTensorEncoding`, checked automatically by
// `RankedTensorType::verify`). That is what makes the layout propagatable: any
// producer and any consumer of a crouton array are compatible by construction,
// and the only question left is where a conversion has to be materialised.
//
// See docs/hmx/layout-representation-survey.md for the mechanism and the
// reasoning (tensor encoding rather than memref layout; no role in the type).
//
// The layout is named by the `#hmx.crouton` encoding, not inferred from a rank-5
// shape, so a bare shape is not a crouton.
namespace mlir {
namespace hmx {

/// The implicit single hardware resource of the HMX path: the engine's
/// accumulator / bias-register state and its staging pipeline. There is one
/// such unit, so every op that mutates it conflicts with every other one
/// (`hmx.mma` cannot be reordered around `hmx.acc_clear` / `hmx.acc_read`).
///
/// Non-addressable: this is register state, not pointer-based memory, so a
/// value-based memory access is never told it aliases the engine. The
/// resource's parent stays the default one (the `SideEffects::Resource`
/// contract), so it is *not* disjoint from ordinary memory effects either --
/// effects that also touch real memory (e.g. `hmx.acc_read`, `hmx.stage`) keep
/// their plain `MemRead`/`MemWrite` on the default resource next to their
/// engine instance, while an engine-only op (`hmx.acc_clear`, whose write was
/// never a memory write) no longer claims to mutate all memory.
///
/// Named from the `.td` side by `Hmx_EngineResource` (HmxOps.td); no
/// registration is needed -- resources are CRTP singletons.
struct HmxEngineResource
    : public SideEffects::Resource::Base<HmxEngineResource> {
  llvm::StringRef getName() const final { return "HmxEngine"; }
  bool isAddressable() const final { return false; }
};

/// The crouton encoding on `type`, or a null attribute when `type` is not a
/// crouton-encoded ranked tensor.
CroutonLayoutAttr getCroutonEncoding(Type type);

/// The memref-side layout (`#hmx.crouton_memref_layout`) for the crouton
/// encoding on `type`, or a null attribute when `type` is not a
/// crouton-encoded ranked tensor. This is the encoding → layout mapping the
/// bufferization seam has to apply; see CroutonMemRefLayoutAttr.
CroutonMemRefLayoutAttr croutonMemRefLayoutOf(Type type);

/// True when `type` is a ranked tensor carrying the `#hmx.crouton` encoding.
bool hasCroutonEncoding(Type type);

/// True when `a` and `b` are interchangeable crouton layouts: same logical
/// shape and element type. The role (AH/WH/AR) is not part of the comparison --
/// the three are one permutation.
bool sameCroutonEncoding(Type a, Type b);

/// The physical type of the crouton array holding a row-major `[M, N]` matrix:
/// `tensor<M/32, N/32, 16, 32, 2 x f16, #hmx.crouton<logical=[M,N]>>`. `logical`
/// must be 32-aligned in both dims; its element type is only the *source's* --
/// the crouton is always the engine's fp16 image, a wider source being quantised
/// by the pack that materialises the array.
RankedTensorType croutonLayoutType(RankedTensorType logical);

/// The physical type of the *weight* crouton array for a row-major `[K, N]`
/// weight. The engine always walks the grid's second dim, so the weight grid is
/// `[Nt, Kt, 16, 32, 2]` with `logical = [N, K]`: the weight is stored as Wᵀ, and
/// this is the same `#hmx.crouton` encoding as any other crouton array, just
/// attached to the transposed logical shape. `weightRowMajorKN` must be
/// 32-aligned in both dims (enforced inside `croutonLayoutType`).
RankedTensorType weightCroutonType(RankedTensorType weightRowMajorKN);

/// The K extent (in 32-wide croutons) of a weight crouton array: the grid's
/// *second* dim. This is the single named source for "which weight grid dim is
/// K" -- do not spell out `getDimSize(0/1)` for a weight elsewhere.
int64_t weightKTiles(ShapedType weightCrouton);

/// The N extent (in 32-wide croutons) of a weight crouton array: the grid's
/// *first* dim.
int64_t weightNTiles(ShapedType weightCrouton);

} // namespace hmx
} // namespace mlir

//===----------------------------------------------------------------------===//
// HMX Ops
//===----------------------------------------------------------------------===//
#define GET_OP_CLASSES
#include "hexagon/Dialect/Hmx/IR/HmxOps.h.inc"

#endif // HEXAGON_DIALECT_HMX_IR_HMX_DIALECT_H
