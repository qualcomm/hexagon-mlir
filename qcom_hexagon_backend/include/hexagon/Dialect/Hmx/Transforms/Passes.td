//===- Passes.td - HMX dialect passes ----------------------*- tablegen -*-===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
// The hmx dialect's own transform passes: engine attribution, the tile level,
// and weight residency. They live with the dialect so the HMX feature has one
// home; `hmx-to-llvm` is a conversion and stays under Conversion/HmxToLLVM.
//===----------------------------------------------------------------------===//

#ifndef HEXAGON_DIALECT_HMX_TRANSFORMS_PASSES_TD
#define HEXAGON_DIALECT_HMX_TRANSFORMS_PASSES_TD

include "mlir/Pass/PassBase.td"

def MatmulToHmx : InterfacePass<"matmul-to-hmx", "mlir::FunctionOpInterface"> {
  let summary = "Convert HMX-eligible linalg.matmul to hmx.matmul";

  let description = [{
    Engine attribution for the HMX matrix engine, modelled on Triton's
    AccelerateMatmul: the entire decision is a legality predicate (`hmxEligible`,
    asking `HmxTarget`) followed by a residency check
    (`HmxTarget::planBridge`: the crouton arrays of one M block must fit VTCM; a
    contraction too large to hold whole is walked in M blocks, and only one that
    fits no block at all is left alone). There is no enable/disable switch --
    only `vtcm-budget`, which caps the residency the bridge may claim.

    Version 1 (plain HMX) requires f16 or f32 inputs (an f32 source is quantised
    to the engine's fp16 by the bridge pack), an f16 or f32 output, 2D static
    shapes, M/N/K multiples of the 32-wide tile and M > 4. A matmul that does not
    qualify is left untouched -- the existing HVX and linalg paths keep handling
    it, because "this op is not for HMX" is an engine choice rather than a
    failure. The pass therefore never errors; it emits a remark, so a near-miss
    is explainable.

    Its operands live in VTCM, so the pass needs the pipeline to provide the
    VTCM allocator (`vtcm-allocator`); without it the attribution is refused
    rather than made on buffers the engine cannot read.

    Seeding and propagating the crouton layout is deliberately NOT this pass's
    job: that is standard `linalg.pack`/`unpack` (inner_dims_pos = [0], inner_tiles = [2]) work for
    the existing layout machinery.
  }];

  let options = [
    Option<"vtcmBudgetBytes", "vtcm-budget", "int64_t", /*default=*/"0",
           "VTCM budget in bytes the bridge must fit; 0 means the device "
           "default (HmxTarget::defaultVtcmBudget).">,
    Option<"vtcmAllocator", "vtcm-allocator", "bool", /*default=*/"true",
           "Whether the pipeline provides the VTCM allocator for the crouton "
           "arrays (the hexagonmem path that lowers space-1 allocations). The "
           "engine reads and writes VTCM only, so without it an attributed "
           "matmul would hand the leaves addresses the engine cannot use.">,
    Option<"dropEncodings", "drop-encodings", "bool", /*default=*/"true",
           "Erase every #hmx.crouton tensor encoding once the pass is done, "
           "restoring the bare rank-5 physical types the rest of the pipeline "
           "knows (default bufferization drops the encoding anyway and emits "
           "no layout). Turn off to keep the encodings: the one-shot "
           "bufferization maps them to the identity-map "
           "#hmx.crouton_memref_layout through the dialect's own allocation op "
           "hmx.alloc_crouton (what this pass emits instead of "
           "bufferization.alloc_tensor), whose BufferizableOpInterface carries "
           "the layout into the memref types.">
  ];

  let constructor = "mlir::hmx::createMatmulToHmxPass()";
}

def HmxPartition : InterfacePass<"hmx-partition", "mlir::FunctionOpInterface"> {
  let summary = "Partition hmx.matmul into the tile loop";

  let description = [{
    The tile level of the HMX path (post-bufferization): an `hmx.matmul` on
    crouton arrays becomes a bias init, an (m, n) tile loop whose body is a K
    loop of mmas, and a read-out per tile.

    VTCM ownership (docs/hmx/hmx-system-design.md §10.12): the crouton arrays
    stay ordinary memory-space-1 buffers, and this pass deliberately does not
    carve a region out for them. The VTCM the kernel runs in is the one the
    runtime already owns, and the existing space-1 machinery is what places them
    there, together with every other VTCM buffer of the kernel.

    The software-pipelined form stages one 32-row activation tile at a time
    through a static ring of `hmx.stage` slots (`depth` 2 when the budget and
    the tile count allow, else 1): the pass emits the serial source loop
    (issue -> await -> compute per tile) and, at depth 2, hands it to the SCF
    software pipeliner (`scf::pipelineForLoop`) with the schedule "issue in
    stage 0, await and compute in stage 1"; the pipeliner generates the
    prologue, the steady kernel and the peeled epilogue and versions the
    cross-stage (token, slot) pair as its `iter_args`. Depth 1 is the same
    source loop left unpipelined. Keeping every stage row in range is by
    construction (the kernel stops one iteration short per stage and the
    epilogue stages nothing), so the transfer needs neither a guard nor a token
    sentinel. When staging is declined the plain tile loop is emitted unchanged
    and a remark names the reason.

    The staged path packs into one crouton-row scratch, not the whole activation
    array: the pack and its mmas are in the same iteration, so one row is live
    at a time. The array is retired whole, so a staged loop never carries a
    second copy of the activation.

    `pipeline-depth` selects the form: the staged ring at some depth, or the
    unstaged serial tile loop. The serial arm skips staging entirely -- no
    `hmx.stage`/`hmx.await`, the activation bridge and its array are kept -- so
    it is the same clear -> K mmas -> read nest the pass emits when the pipeline
    does not apply, reached by construction rather than by declining.
  }];

  let options = [
    Option<"vtcmBudgetBytes", "vtcm-budget", "int64_t", /*default=*/"0",
           "VTCM budget in bytes the pipeline ring must fit; 0 means the device "
           "default (HmxTarget::defaultVtcmBudget).">,
    Option<"pipelineDepth", "pipeline-depth", "int64_t", /*default=*/"0",
           "Activation-staging ring depth: 0 = auto (the deepest ring the VTCM "
           "budget and the tile count allow), 1 = force the serial ring, 2 = "
           "ask for the double ring (narrowed to the deepest ring that fits, "
           "with a remark, when the budget cannot pay for it), 3 = skip the "
           "staging rewrite and emit the unstaged serial tile loop (no "
           "hmx.stage/hmx.await, the activation array is kept).">
  ];

  let constructor = "mlir::hmx::createHmxPartitionPass()";
}

def WeightResident : InterfacePass<"weight-resident", "mlir::FunctionOpInterface"> {
  let summary = "Make constant HMX weights a resident VTCM buffer";

  let description = [{
    A constant weight reaches the HMX tile level as a `memref.get_global` of a
    prepacked `memref.global` (the compile-time pack of `matmul-to-hmx`), which
    the engine cannot read: `hmx.mma` requires its weight in VTCM. This pass
    turns each such operand into a VTCM `hexagonmem.alloc` tagged
    `hmx.weight_resident`, recording the source symbol and byte count, and
    declares the total in the module attribute `hmx.weight_resident_bytes`.

    The buffer is not filled by the kernel. Its lowering emits one call to
    `hexagon_runtime_weight_resident(src, bytes)`: the runtime allocates the
    buffer on the first launch, copies the weight in once, pins it (the
    per-launch deallocation is swallowed) and returns the same address on every
    later launch. So the weight is packed into VTCM exactly once per process.

    Must run after bufferization (the operand is a `memref.get_global`) and
    before `hmx-partition` (which needs the weight in VTCM).

    With `prepackRuntimeWeights` on, the same residency is applied to a runtime
    weight: the `hmx.pack_weight` bridge of a function argument is replaced by a
    resident buffer whose contents the runtime copies once, and the argument's
    slot and crouton layout are published on the module (`hmx.weight_prepack`)
    for the host pre-packer. The option is off by default because dropping the
    pack is only correct when the caller actually pre-packs; the default leaves
    the IR byte-identical to before.
  }];

  let options = [
    Option<"prepackRuntimeWeights", "prepack-runtime-weights", "bool",
           /*default=*/"false",
           "Replace a runtime weight's per-launch pack bridge with a resident "
           "buffer the host pre-packs (see hmx.weight_prepack).">,
  ];

  let constructor = "mlir::hmx::createWeightResidentPass()";
}

def HmxWorkspaceResident : InterfacePass<"hmx-workspace-resident",
                                         "mlir::FunctionOpInterface"> {
  let summary = "Make the per-launch HMX VTCM workspace a resident buffer";

  let description = [{
    Every HMX kernel's prologue allocates (and epilogue frees) its VTCM
    workspace: the crouton arrays of `matmul-to-hmx`, the `hmx-partition`
    conversion state, the activation-staging ring slots and statuses, and the
    crouton-row scratch. The runtime's allocator charges a size-independent
    best-fit scan plus bookkeeping for each, measured at ~6.3 us per alloc/free
    pair, so on a small shape the sequence costs more than the compute.

    A workspace has no compile-time contents -- the kernel refills it every
    launch -- so all it needs is a stable buffer. This pass tags each static
    VTCM `memref.alloc` with `hmx.workspace_resident` and drops its
    deallocation. The lowering emits one `hexagon_runtime_workspace_resident`
    call: the first launch allocates and pins the buffer, every later launch
    gets the same address back with no allocation, no free and no copy.

    The key is `(hash(function symbol) << 32) | allocation index` with the top
    bit set, so it is stable across launches, unique per (function, buffer) and
    disjoint from the weight path's address keys.

    Correctness boundary: a resident buffer is shared by every launch in the
    process. Sequential launches are safe (each launch overwrites the whole
    buffer before reading it); a grid>1 launch runs several instances in one
    process and they would clobber each other. The pass is therefore only added
    by the caller's opt-in (`enable-workspace-resident`), which promises
    single-instance execution.

    Runs after `hmx-partition` (so the partition pass's buffers exist) and
    before `convert-to-hexagonmem` (which carries the tag to the
    `hexagonmem.alloc` whose lowering reads it).
  }];

  let constructor = "mlir::hmx::createHmxWorkspaceResidentPass()";
}

#endif // HEXAGON_DIALECT_HMX_TRANSFORMS_PASSES_TD
