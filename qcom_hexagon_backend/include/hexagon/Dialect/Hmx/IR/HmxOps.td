//===- HmxOps.td - HMX Ops ---------------------------------*- tablegen -*-===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
// Describes the hmx dialect ops.
//
// These are the leaves of the HMX path. Their operands are typed views, not
// addresses: the crouton `memref<16x32x2xf16, 1>` (the layout produced by
// `hexagonmem.convert_layout`) and ordinary memory-space-1 buffers for the
// conversion state. HmxToLLVM is what turns them into the address-only calls the
// runtime leaves actually take.
//
// The pack/unpack ops return their destination as a result once bufferized:
// the written buffer must be an SSA value so that a consumer or an `scf.yield`
// can depend on the producer through a value, which is what exposes the
// dependency to an external scheduler
// (docs/hmx/hmx-scheduling-interface-plan.md section 9.3). The tensor form's
// result is unchanged, and in memref form the result is the `dst` operand
// itself, so the lowering emits no copy.
//===----------------------------------------------------------------------===//

#ifndef HEXAGON_DIALECT_HMX_IR_HMX_OPS_TD
#define HEXAGON_DIALECT_HMX_IR_HMX_OPS_TD

include "mlir/IR/OpBase.td"
include "mlir/IR/CommonTypeConstraints.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/DestinationStyleOpInterface.td"
include "hexagon/Dialect/Hmx/IR/HmxDialect.td"

// The implicit single hardware resource of the HMX path (defined in
// HmxDialect.h): the engine's accumulator / bias-register state and its
// staging pipeline. Ops carrying effects on it conflict with each other by
// resource identity, instead of claiming -- as an effect on the default
// resource would -- that they mutate all memory.
def Hmx_EngineResource : Resource<"::mlir::hmx::HmxEngineResource">;

// Base class for HMX dialect ops.
class Hmx_Op<string mnemonic, list<Trait> traits = []> :
    Op<Hmx_Dialect, mnemonic, traits>;

def Hmx_MatmulOp
  : Hmx_Op<"matmul",
      [DestinationStyleOpInterface, MemoryEffects<[MemRead, MemWrite]>]> {
  let hasVerifier = 1;

  let summary = "Matrix multiplication scheduled for the HMX engine";

  let description = [{
    A `linalg.matmul` that the compiler has decided belongs to the HMX engine.
    **Engine attribution and layout are the same statement here**, because the
    operands are already in the layout the engine reads:

    ```
    lhs: tensor<Mt, Kt, 16, 32, 2>     // Mt = M/32, Kt = K/32
    rhs: tensor<Nt, Kt, 16, 32, 2>     // W stored as Wᵀ, logical [N][K]
    out: tensor<Mt, Nt, 16, 32, 2>     // AR read-out, pair on the M dim
    ```
    The last three dimensions are one crouton: 16 pairs x 32 columns x 2 halves of
    the pair. That is byte-for-byte the order the engine reads (locked down by
    `exp/hmx/oracle/`).

    Keeping the layout in the type is what makes a matmul chain free: the read-out
    layout AR and the activation layout AH are the same permutation, so the result
    of one `hmx.matmul` **is** a valid `lhs` of the next one -- the types are equal
    by construction, so no repack can appear between them. A row-major producer
    (norm, embedding, kernel argument) is bridged by an explicit
    `linalg.pack`/`unpack`, which is also where `fold-pack-unpack-constants` turns a
    constant weight into a prepacked constant at compile time.

    The op is write-only on `outs` (like `linalg.fill`): the hardware accumulator
    is cleared before the tiles are summed, so the result does NOT accumulate into
    an incoming value. Whoever creates this op from a general matmul must prove
    that the original accumulator was empty.

    Whether a given matmul is *worth* converting is a separate decision, taken by
    the `matmul-to-hmx` pass against the target and the VTCM budget.
  }];

  let arguments = (ins
    AnyTypeOf<[Non0RankedTensorOf<[F16]>, Non0RankedMemRefOf<[F16]>]>:$lhs,
    AnyTypeOf<[Non0RankedTensorOf<[F16]>, Non0RankedMemRefOf<[F16]>]>:$rhs,
    AnyTypeOf<[Non0RankedTensorOf<[F16]>, Non0RankedMemRefOf<[F16]>]>:$outs
  );

  let results = (outs Variadic<Non0RankedTensorOf<[F16]>>:$result);

  let assemblyFormat = [{
    `ins` `(` $lhs `,` $rhs `:` type($lhs) `,` type($rhs) `)`
    `outs` `(` $outs `:` type($outs) `)`
    attr-dict
    (`->` type($result)^)?
  }];

  let extraClassDeclaration = [{
    // Implemented as part of DestinationStyleOpInterface
    MutableOperandRange getDpsInitsMutable() { return getOutsMutable(); }
  }];
}

def Hmx_AllocCroutonOp : Hmx_Op<"alloc_crouton", [MemoryEffects<[MemAlloc]>]> {
  let hasVerifier = 1;

  let summary = "Allocate the VTCM crouton array an HMX matmul reads and writes";

  let description = [{
    The one legal allocation point for a crouton array; `matmul-to-hmx` is its
    only emitter. `bufferization.alloc_tensor` cannot carry the crouton layout
    across bufferization: the pinned one-shot bufferize pass cannot be given a
    custom type converter, and `alloc_tensor` infers its own static-identity
    buffer type (hmx-interface-gaps.md section 2.2). This op owns the seam
    instead: its `BufferizableOpInterface` maps the `#hmx.crouton` encoding to
    the identity-map `#hmx.crouton_memref_layout` (via
    `croutonMemRefLayoutOf`), so an encoded result bufferizes to

    ```
    memref<M/32, N/32, 16, 32, 2 x f16,
           #hmx.crouton_memref_layout<logical = [M, N]>, 1>
    ```

    while a result whose encoding was erased (`drop-encodings`, the default)
    bufferizes to the same layout-less `memref<... x f16, 1>` a
    `bufferization.alloc_tensor` with `memory_space = 1` produces -- there the
    two are byte-identical.

    The allocation is always in VTCM (memory space 1): that is the op's
    meaning, not an attribute, because the engine reads and writes VTCM only.
    The result is fully static (the tile grid is fixed at attribution time), so
    unlike `alloc_tensor` there are no dynamic sizes and no `copy` operand:
    the contents are undefined, and the pack bridge that fills the array is
    what makes them defined.
  }];

  let results = (outs RankedTensorOf<[F16]>:$result);

  let assemblyFormat = "attr-dict `->` type($result)";

  let builders = [
    OpBuilder<(ins "RankedTensorType":$type), [{
      $_state.addTypes(type);
    }]>
  ];
}

def Hmx_PackActOp
  : Hmx_Op<"pack_act",
      [DestinationStyleOpInterface, MemoryEffects<[MemRead, MemWrite]>]> {
  let hasVerifier = 1;

  let summary = "Pack one row-major 32x32 block into an AH crouton";

  let description = [{
    Packs the 32x32 fp16 block at (`row`, `col`) of the row-major `src` into
    crouton (`row`, `col`) of the activation crouton array `dst`, in the AH
    layout the engine reads. Out-of-range source elements read as zero.

    The source may be fp32 as well as fp16; the engine's operand is fp16 either
    way, so an fp32 source is quantised by the pack itself (the engine's
    conversion, folded into the same pass that does the layout -- narrowing the
    source first would cost a whole extra read and write of it). The semantics of
    an fp32 source are therefore "this matrix, rounded to fp16".

    This is the engine's layout, not a general transpose: the work is one
    vectorised runtime leaf (`hmx_pack_act_f16`). Expressing it as
    `linalg.pack` instead leaves the generic lowering to synthesise an
    element-wise data movement (measured: 99.3% of the HMX kernel's runtime).

    The source is addressed row-by-row by its own `stride(rank-2)` in elements:
    `src_cols` for a dense matrix, but larger when the source is one tile of a
    wider matrix (a strided view, `memref<M x BN, strided<[N, 1]>>` with `N >=
    BN`). The lowering threads that stride to the leaf as `src_stride`;
    addressing rows by the width instead reads the wrong element of every row
    past the first.

    The optional `count` is a range along the crouton array's contiguous axis.
    The activation array is `[Mt, Kt]`, so `count` column tiles are packed in
    one call: croutons `(row, col) .. (row, col + count - 1)`, all from source
    blocks of row `row`. Absent means one block (`count == 1`), which is the
    single-crouton form; the range exists so one runtime call (one whole vector
    of scatter offsets) covers a whole K run.
  }];

  let arguments = (ins
    Arg<AnyTypeOf<[Non0RankedTensorOf<[F16]>, Non0RankedMemRefOf<[F16]>]>,
        "activation crouton array in VTCM">:$dst,
    Arg<AnyTypeOf<[Non0RankedTensorOf<[F16, F32]>,
                   Non0RankedMemRefOf<[F16, F32]>]>,
        "row-major source">:$src,
    Index:$row,
    Index:$col,
    OptionalAttr<ConfinedAttr<I64Attr, [IntPositive]>>:$count
  );
  // Tensor form: the crouton array. Memref form (DPS): the same `dst` buffer
  // that was written, so the destination is an SSA value.
  let results = (outs
    Variadic<AnyTypeOf<[Non0RankedTensorOf<[F16]>,
                        Non0RankedMemRefOf<[F16]>]>>:$result);

  let assemblyFormat = [{
    `ins` `(` $src `,` $row `,` $col `:` type($src) `)`
    `outs` `(` $dst `:` type($dst) `)`
    attr-dict
    (`->` type($result)^)?
  }];

  let extraClassDeclaration = [{
    MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
}

def Hmx_PackWeightOp
  : Hmx_Op<"pack_weight",
      [DestinationStyleOpInterface, MemoryEffects<[MemRead, MemWrite]>]> {
  let hasVerifier = 1;

  let summary = "Pack one row-major 32x32 block into a WH crouton";

  let description = [{
    Same as `hmx.pack_act` for the weight operand: packs the 32x32 block at
    (`k_tile`, `n_tile`) of the row-major [K][N] `src` into crouton
    (`k_tile`, `n_tile`) of the weight crouton array `dst`. WH is the same
    permutation as AH (locked down in the crouton oracle), so the two ops share
    one runtime leaf layout; they are separate ops because the engine indexes the
    two arrays along different dims. As in `hmx.pack_act`, an fp32 source is
    quantised to the engine's fp16 by the pack.

    As in `hmx.pack_act`, the source rows are addressed by `stride(rank-2)`: `N`
    for a dense matrix, larger when the source is one tile of a wider matrix.
    The lowering threads it to the leaf as `src_stride`.

    The optional `count` ranges over the weight crouton array's contiguous axis
    the same way: the grid is `[Nt, Kt]`, so `count` K tiles at a fixed `n_tile`
    are packed in one call -- croutons `(n_tile, k_tile) .. (n_tile, k_tile +
    count - 1)` -- with absent meaning one block.
  }];

  let arguments = (ins
    Arg<AnyTypeOf<[Non0RankedTensorOf<[F16]>, Non0RankedMemRefOf<[F16]>]>,
        "weight crouton array in VTCM">:$dst,
    Arg<AnyTypeOf<[Non0RankedTensorOf<[F16, F32]>,
                   Non0RankedMemRefOf<[F16, F32]>]>,
        "row-major source">:$src,
    Index:$k_tile,
    Index:$n_tile,
    OptionalAttr<ConfinedAttr<I64Attr, [IntPositive]>>:$count
  );
  // Tensor form: the crouton array. Memref form (DPS): the same `dst` buffer
  // that was written, so the destination is an SSA value.
  let results = (outs
    Variadic<AnyTypeOf<[Non0RankedTensorOf<[F16]>,
                        Non0RankedMemRefOf<[F16]>]>>:$result);

  let assemblyFormat = [{
    `ins` `(` $src `,` $k_tile `,` $n_tile `:` type($src) `)`
    `outs` `(` $dst `:` type($dst) `)`
    attr-dict
    (`->` type($result)^)?
  }];

  let extraClassDeclaration = [{
    MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
}

def Hmx_UnpackAccOp
  : Hmx_Op<"unpack_acc",
      [DestinationStyleOpInterface, MemoryEffects<[MemRead, MemWrite]>]> {
  let hasVerifier = 1;

  let summary = "Unpack one AR crouton into a row-major fp16 block";

  let description = [{
    Unpacks crouton (`row`, `col`) of the AR crouton array `src` into the 32x32
    fp16 block at (`row`, `col`) of the row-major `dst`, with one vectorised
    runtime call. A wider result is the fp16 image widened afterwards, the same
    way `hmx.matmul` reads its accumulator out as fp16.

    The read-out of the accumulator and its conversion to row-major are one
    runtime step, the same way the conversion-state load and the accumulator
    store are one step in `hmx.acc_read`.

    The optional `count` is a range of row-pairs at a fixed tile `row`: the leaf
    always walks a whole crouton row itself, so `count` pairs start at `col`
    (block `col`) and cover `count` consecutive 128-byte blocks. The whole tile
    row is `count == 16`, and absent means one pair (`count == 1`).
  }];

  let arguments = (ins
    Arg<AnyTypeOf<[Non0RankedTensorOf<[F16]>, Non0RankedMemRefOf<[F16]>]>,
        "AR crouton array in VTCM">:$src,
    Arg<AnyTypeOf<[Non0RankedTensorOf<[F16]>, Non0RankedMemRefOf<[F16]>]>,
        "row-major fp16 destination">:$dst,
    Index:$row,
    Index:$col,
    OptionalAttr<ConfinedAttr<I64Attr, [IntPositive]>>:$count
  );
  // Tensor form: the row-major matrix. Memref form (DPS): the same `dst` buffer
  // that was written, so the destination is an SSA value.
  let results = (outs
    Variadic<AnyTypeOf<[Non0RankedTensorOf<[F16]>,
                        Non0RankedMemRefOf<[F16]>]>>:$result);

  let assemblyFormat = [{
    `ins` `(` $src `,` $row `,` $col `:` type($src) `)`
    `outs` `(` $dst `:` type($dst) `)`
    attr-dict
    (`->` type($result)^)?
  }];

  let extraClassDeclaration = [{
    MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
}

def Hmx_UnpackAccF32Op
  : Hmx_Op<"unpack_acc_f32",
      [DestinationStyleOpInterface, MemoryEffects<[MemRead, MemWrite]>]> {
  let hasVerifier = 1;

  let summary = "Unpack one AR crouton row-pair into row-major f32, optionally adding a residual";

  let description = [{
    The fused tail of an HMX matmul: unpacks crouton row (`row`, `block_j`)
    of the AR crouton array `src` straight into the f32 rows of the row-major
    `dst`, folding in the fp16->fp32 widening and, when `residual` is present,
    the add of the original accumulator (`linalg.matmul`'s C term, which the
    engine overwrites instead of accumulating into).

    Without a residual this replaces `hmx.unpack_acc` plus the widening
    `linalg.generic`; with one it additionally replaces the `addInto`
    `linalg.generic`. All three today run as separate DDR passes; the runtime
    leaf (`hmx_unpack_acc_f32`) does them in one: one aligned crouton load per
    32-column tile, widened via x1.0, residual added, aligned f32 stores.

    `residual`, when present, must have exactly `dst`'s type. The leaf writes
    fp32 row-major, so unlike `hmx.unpack_acc` there is no wider result to
    widen afterwards: this op only ever produces f32.

    The optional `count` ranges over row-pairs exactly as in `hmx.unpack_acc`:
    `count` pairs start at `col` at tile `row`, whole tile row = 16, absent = 1.
  }];

  let arguments = (ins
    Arg<AnyTypeOf<[Non0RankedTensorOf<[F16]>, Non0RankedMemRefOf<[F16]>]>,
        "AR crouton array in VTCM">:$src,
    Arg<AnyTypeOf<[Non0RankedTensorOf<[F32]>, Non0RankedMemRefOf<[F32]>]>,
        "row-major f32 destination">:$dst,
    Index:$row,
    Index:$col,
    Optional<AnyTypeOf<[Non0RankedTensorOf<[F32]>, Non0RankedMemRefOf<[F32]>]>>:$residual,
    OptionalAttr<ConfinedAttr<I64Attr, [IntPositive]>>:$count
  );
  // Tensor form: the row-major f32 matrix. Memref form (DPS): the same `dst`
  // buffer that was written, so the destination is an SSA value.
  let results = (outs
    Variadic<AnyTypeOf<[Non0RankedTensorOf<[F32]>,
                        Non0RankedMemRefOf<[F32]>]>>:$result);

  let assemblyFormat = [{
    `ins` `(` $src `,` $row `,` $col (`,` $residual^)? `:`
      type($src) (`,` type($residual)^)? `)`
    `outs` `(` $dst `:` type($dst) `)`
    attr-dict
    (`->` type($result)^)?
  }];

  let extraClassDeclaration = [{
    MutableOperandRange getDpsInitsMutable() { return getDstMutable(); }
  }];
}

def Hmx_BiasInitOp
  : Hmx_Op<"bias_init", [MemoryEffects<[MemWrite, MemWrite<Hmx_EngineResource>]>]> {
  let hasVerifier = 1;

  let summary = "Write the identity conversion state the read-out loads";

  let description = [{
    Fills one 256-byte conversion-state block with the identity: scale 1.0 and
    no output bias.

    The bias registers must be loaded from such a block before the accumulator
    can be read out. With an uninitialised block the scale is zero and the
    read-out comes back all zeros, which is what made the first device
    experiments look like a broken read-out instruction.

    This is per-kernel setup, not per-tile work, so it is its own op rather than
    part of `hmx.acc_read`.

    The block is real memory, so the `MemWrite` on the default resource stays;
    the `HmxEngineResource` instance records that the block is engine-owned
    read-out state established by the HMX sequence.
  }];

  let arguments = (ins Non0RankedMemRefOf<[I8]>:$bias);

  let assemblyFormat = "$bias attr-dict `:` type($bias)";
}

def Hmx_AccClearOp
  : Hmx_Op<"acc_clear", [MemoryEffects<[MemWrite<Hmx_EngineResource>]>]> {
  let summary = "Clear the HMX accumulator";

  let description = [{
    Clears the accumulator. The accumulator is an implicit single hardware
    resource with no SSA value, so the clear is a write on the dialect's
    `HmxEngineResource`: it still conflicts with every other op touching that
    resource -- `hmx.mma` and `hmx.acc_read` cannot be reordered around it --
    while operations that only touch memory, which this op never does, no
    longer serialise against it.
  }];

  let arguments = (ins);
  let results = (outs);

  let assemblyFormat = "attr-dict";
}

def Hmx_MmaOp
  : Hmx_Op<"mma",
      [MemoryEffects<[MemRead, MemRead<Hmx_EngineResource>,
                      MemWrite<Hmx_EngineResource>]>]> {
  let hasVerifier = 1;

  let summary = "Multiply-accumulate one position of two crouton arrays";

  let description = [{
    Accumulates `n_croutons` activation croutons against weight croutons, starting
    at position (`m`, `k`) of the activation array and (`n`, `k`) of the weight
    array, and walking forward from there. The weight is stored as Wᵀ, so k is
    the second grid dim on both operands.

    The operands are the **whole crouton arrays addressed by tile indices**, not a
    crouton sliced out of them: the engine takes an address plus a range, so this
    is the shape the hardware actually has, and it keeps the offset arithmetic in
    one place (HmxToLLVM's linearisation) instead of scattering subviews through
    the tile loop.

    ```
    act: memref<Mt, Kt, 16, 32, 2 x f16, 1>     wt: memref<Nt, Kt, 16, 32, 2 x f16, 1>
    ```

    The last three dimensions are one crouton (16 pairs x 32 columns x 2 halves of
    a pair) and the leading two are the tile grid. Both arrays must live in VTCM --
    the engine only reads from there. The activation and weight loads are issued as
    a pair in one packet by the runtime leaf, so they cannot be split into two ops.

    The accumulator is implicit: the op's read-modify-write of it is carried on
    `HmxEngineResource` (conflicting with `hmx.acc_clear`/`hmx.acc_read` by
    resource identity), while its `MemRead` on the default resource covers the
    crouton arrays, which are real memory. The op writes no memory, so it
    claims no `MemWrite` on the default resource.
  }];

  let arguments = (ins
    Arg<AnyNon0RankedMemRef, "activation crouton array in VTCM">:$act,
    Arg<AnyNon0RankedMemRef, "weight crouton array in VTCM">:$wt,
    Index:$m,
    Index:$n,
    Index:$k,
    I32Attr:$n_croutons
  );
  let results = (outs);

  let assemblyFormat = [{
    $act `,` $wt `,` $m `,` $n `,` $k attr-dict `:` type($act) `,` type($wt)
  }];
}

def Hmx_AccReadOp
  : Hmx_Op<"acc_read",
      [MemoryEffects<[MemRead, MemWrite, MemRead<Hmx_EngineResource>,
                      MemWrite<Hmx_EngineResource>]>]> {
  let hasVerifier = 1;

  let summary = "Load the bias registers and convert the accumulator out";

  let description = [{
    Loads conversion state `bias` into bias register set `bias_set`, converts the
    accumulator into position (`m`, `n`) of the AR crouton array `dst` in one fused
    step, and clears the accumulator.

    The fused read-out is required: the split form documented in the (V81) PRM
    zeroes negative values on v79 silicon. `bias_set` selects one of the four
    register sets, which is what lets several tiles share one state block.

    Turning `dst` into row-major values is the epilogue's job, not this op's.

    `bias` and `dst` are real memory and keep `MemRead`/`MemWrite` on the
    default resource; the accumulator read-out and the clear that follows it
    are engine state, carried on `HmxEngineResource` -- which is what conflicts
    with `hmx.mma` and `hmx.acc_clear`.
  }];

  let arguments = (ins
    Arg<AnyNon0RankedMemRef, "conversion state block in VTCM">:$bias,
    Arg<AnyNon0RankedMemRef, "AR crouton array in VTCM">:$dst,
    Index:$m,
    Index:$n,
    I32Attr:$bias_set
  );
  let results = (outs);

  let assemblyFormat = [{
    $bias `,` $dst `,` $m `,` $n attr-dict `:` type($bias) `,` type($dst)
  }];
}

//===----------------------------------------------------------------------===//
// Asynchronous staging
//===----------------------------------------------------------------------===//
//
// `hmx.stage` / `hmx.await` turn one DDR -> VTCM tile transfer into an SSA
// edge. The DMA token is the handle: it is address-width (an i32 on this
// target), not a new dialect type, because the runtime's `dma_start` already
// returns exactly that. `await` yields the staged slot, so the compute side
// depends on the transfer through a value -- which is what an external
// scheduler needs to see
// (docs/hmx/hmx-scheduling-interface-plan.md section 9.2/9.3).

def Hmx_StageOp
  : Hmx_Op<"stage",
      [MemoryEffects<[MemRead, MemWrite, MemWrite<Hmx_EngineResource>]>]> {
  let hasVerifier = 1;

  let summary = "Start an asynchronous DDR -> VTCM transfer of one tile row";

  let description = [{
    Starts the runtime DMA transfer of the 32-row x K tile beginning at source
    row `row` of the row-major `src` into the VTCM slot `dst`, and returns the
    transfer's token. `row` is a 0-based *element row offset* into `src` (the
    lowering scales it by the source's own row stride), not a tile index: tile
    `t` of a 32-row tile grid starts at row `t * 32`. The token is the handle a
    later `hmx.await` waits on; it is an i32 because that is the width of an
    address on this target and the type the runtime already returns.

    The transfer is issued unconditionally. There is no out-of-range case and no
    token sentinel: runtime tokens start at 0, so 0 cannot mean "no transfer".
    Keeping every `row` in range is the caller's job, and the tile level does it
    by construction -- the software-pipelined kernel stops one iteration short
    per pipeline stage and the peeled tail only awaits and computes, staging
    nothing.

    `status` is the single DMA status word the runtime writes the completion
    into. It is per-slot state, not shared.

    The op is deliberately not destination-style: its result is the token, not
    the written slot -- the slot is updated asynchronously and only becomes
    visible at the matching `hmx.await`.

    `src`, `dst` and `status` are real memory and keep the default-resource
    effects; the in-flight transfer is pipeline state, recorded as the
    `HmxEngineResource` write.
  }];

  let arguments = (ins
    Arg<Non0RankedMemRefOf<[F16, F32]>, "row-major source (DDR)">:$src,
    Index:$row,
    Arg<Non0RankedMemRefOf<[F16, F32]>, "VTCM destination slot">:$dst,
    Arg<Non0RankedMemRefOf<[I32]>, "DMA status word">:$status
  );
  let results = (outs I32:$token);

  let assemblyFormat = [{
    `ins` `(` $src `,` $row `:` type($src) `)`
    `outs` `(` $dst `,` $status `:` type($dst) `,` type($status) `)`
    attr-dict
    `->` type($token)
  }];
}

def Hmx_AwaitOp
  : Hmx_Op<"await",
      [MemoryEffects<[MemRead, MemWrite, MemRead<Hmx_EngineResource>,
                      MemWrite<Hmx_EngineResource>]>]> {
  let hasVerifier = 1;

  let summary = "Wait for a staged transfer and return its ready slot";

  let description = [{
    Waits for the transfer `token` to complete and returns the staged VTCM slot
    `dst`. Returning `dst` is what makes the transfer a value the compute side
    consumes: a `hmx.pack_act` reading the result then depends on the transfer
    through SSA, not through a side effect an external scheduler cannot see
    (docs/hmx/hmx-scheduling-interface-plan.md section 9.3, requirement C3).

    The result type is always `dst`'s type: this op names the slot, it does not
    allocate one.

    `dst` and `status` are real memory -- the staged write becomes visible at
    this op -- and keep the default-resource effects; observing completion and
    handing the slot to the compute side is pipeline state on
    `HmxEngineResource`.
  }];

  let arguments = (ins
    I32:$token,
    Arg<Non0RankedMemRefOf<[F16, F32]>, "staged VTCM slot">:$dst
  );
  let results = (outs Non0RankedMemRefOf<[F16, F32]>:$result);

  let assemblyFormat = [{
    `ins` `(` $token `:` type($token) `)`
    `outs` `(` $dst `:` type($dst) `)`
    attr-dict
    `->` type($result)
  }];
}

#endif // HEXAGON_DIALECT_HMX_IR_HMX_OPS_TD
