# ===- hexagon_options.py ---------------------------------------------------===
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause.
# For more license information:
#   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
#
# ===------------------------------------------------------------------------===

import hashlib
from dataclasses import dataclass

from . import knobs

# Per-operand multicast modes and input aliases. "broadcast" is RHS-only --
# LHS multicast has no leader, only "allgather" (each core reads a disjoint
# DDR slice in parallel).
_MC_CANONICAL = ("off", "broadcast", "allgather")
_MC_ALIASES = {"v0": "broadcast", "v1": "allgather"}

# Modes each operand accepts.  Used to reject e.g. lhsMulticastMode=broadcast
# rather than silently accepting a mode with no implementation.
_MC_ALLOWED = {
    "rhs": ("off", "broadcast", "allgather"),
    "lhs": ("off", "allgather"),
}


def normalize_multicast_mode(mode, operand: str = "rhs") -> str:
    """Canonicalize a multicast mode for `operand` ("lhs" or "rhs").

    Accepts the canonical names and the aliases in _MC_ALIASES
    (case-insensitive).  Raises ValueError on anything the operand does not
    support.
    """
    if operand not in _MC_ALLOWED:
        raise ValueError(
            f"Invalid multicast operand {operand!r}; expected one of "
            f"{tuple(_MC_ALLOWED)}"
        )
    m = str(mode).strip().lower()
    m = _MC_ALIASES.get(m, m)
    allowed = _MC_ALLOWED[operand]
    if m not in allowed:
        raise ValueError(
            f"Invalid {operand}MulticastMode {mode!r}; expected one of "
            f"{allowed} (or aliases {tuple(_MC_ALIASES)})"
        )
    return m


def normalize_rhs_multicast_mode(mode) -> str:
    """Backwards-compatible wrapper for normalize_multicast_mode(mode, "rhs")."""
    return normalize_multicast_mode(mode, "rhs")


def is_mlir_nsp_sharded(
    enable_nsp_sharding: bool, nsp_sharding_semantics: bool
) -> bool:
    """Whether the kernel is sharded across NSPs by the MLIR compiler.

    enableNSPSharding runs NSPShardPlanner/NSPLocalize in LinalgToLLVM, while
    nspShardingSemantics makes the launcher hand each NSP a contiguous chunk
    of the Triton grid. When both are set, the launcher's chunked grid drives
    the sharding; only enableNSPSharding alone yields kernels that compute
    their own shard from (threadID, coreID).
    """
    return enable_nsp_sharding and not nsp_sharding_semantics


def derive_multicast_mode(lhs_mode: str, rhs_mode: str) -> str:
    """Collapse per-operand modes into the multicast shape: off|lhs|rhs|lhs_rhs.

    This selects the launcher template/grid mapping, distinct from each
    operand's own broadcast/allgather mode.
    """
    lhs_on = lhs_mode != "off"
    rhs_on = rhs_mode != "off"
    if lhs_on and rhs_on:
        return "lhs_rhs"
    if lhs_on:
        return "lhs"
    if rhs_on:
        return "rhs"
    return "off"


@dataclass(frozen=True)
class HexagonOptions:
    allow_fp8e4nv: bool = False
    allowed_dot_input_precisions: tuple[str] = ("ieee",)
    arch_triple: str = "hexagon"
    arch_features: str = f"+hvxv{knobs.hexagon.arch_version},+hvx-length128b"
    device_type: str = "hexagon"
    vectorize: int = 1
    vector_length: int = 32
    num_threads: int = 4
    data_layout: str = (
        "e-m:e-p:32:32:32-a:0-n16:32-i64:64:64-i32:32:32-i16:16:16-i1:8:8-f32:32:"
        "32-f64:64:64-v32:32:32-v64:64:64-v512:512:512-v1024:1024:1024-v2048:2048:2048"
    )
    # TODO: Next 7 options not used currently but kept to be consistent with other backends, remove later if not required
    num_warps: int = 1
    num_stages: int = 1
    num_ctas: int = 1
    shared: bool = False
    cluster_dims: tuple = (1, 1, 1)
    supported_fp8_dtypes: tuple[str, ...] = ()
    sanitize_overflow: bool = True
    debug: bool = False
    # TODO: Currently set name here temporarily. Need to set correctly to the name of the compiled kernel
    # when driver code is completed.
    name: str = "Hexagon"
    instrumentation_mode: str = ""
    htp_kernel_gen: bool = False
    target_artifact: str = "o"
    iterations: int = 10  # Triton specific benchmarking iteration count

    # Hexagon Linalg Options
    fusion: bool = True
    fusionAllowRecompute: bool = False
    fusionDoMultiUse: bool = True

    enableBufferization: bool = True  # Used to disable for some dma testing
    enableCollapseAddressSpace: bool = True  # lower llvm.ptr<1> if hexagonmem did not
    enableConvTiling: bool = False
    enableDoubleBuffering: bool = False  # enable double buffering optimization
    convTileSizes: str = ""
    enableConvertToHexagonmem: bool = True  # rewrites memref.alloc/copy to hexagonmem.*
    enableHexagonmemCopyToDMA: bool = False  # rewrites hexmem.copy to memref.dma_*
    enableHexKL: bool = False  # use HexKL to lower matmul and convolutions
    relaxFP32MatmulToFP16: bool = (
        False  # relax fp32 matmul inputs to fp16 to support HexKL matmul lowering
    )
    hexKLMode: str = "micro"  # possible options "macro", "micro", "micro-daemon"
    enableMultiThreading: bool = (
        False  # linalg-generic based multi-threading (FormVirtualThreadsPass)
    )
    enableThreadedDispatch: bool = (
        False  # use tm.exec() (real qurt threads) for SPMD grid dispatch; implicitly enabled when enableMultiThreading=True; when False, tm.exec_serial() is used
    )
    enableSCFThreading: bool = False  # scf based multi-threading
    enableSplitReduction: bool = False  # split-reduction optimization
    enableSplitReduceGeneric: bool = False  # split-reduce-generic optimization
    enableVectorization: bool = True  # enable HVX vectorization.
    enablePackedGenericVectorization: bool = (
        False  # gate HexagonVectorizePackedGeneric.
    )
    enablePackedGenericLoopSplitting: bool = (
        True  # loop splitting for Nx32 repacked inputs
    )
    enablePackedGenericEagerVectorization: bool = (
        False  # eagerly vectorize all operands upfront
    )
    enableVTCMTiling: bool = (
        True  # tile linalg-generic and introduce vtcm address-space
    )
    # External VTCM scratch buffer size in bytes per program instance.
    # scratch=0 (default): disabled, each instance allocates VTCM internally.
    # scratch=N (N>0): enables external VTCM flow. The wrapper allocates
    # N * prod(grid) bytes total, passes each instance a memref<Nxi8, 1>
    # {hexagon.scratch} slice. The compiler tiles and plans within N bytes.
    scratch: int = 0
    enablePromoteTiledSCFLoopToVTCM: bool = False  # promote tiled SCF loops to VTCM
    enableVectorAccumulatorForReduction: bool = (
        False  # vector accumulator optimization for reductions
    )
    enableHVXInlining: bool = False
    enableSCFLoopUnroll: bool = False
    # Outer-loop unroll-and-jam of reduction loops (linalg level, pre-vectorize)
    # to expose independent reduction chains and improve ILP. Opt-in.
    enableUnrollAndJam: bool = False
    unrollAndJamFactor: int = 2
    enableFastExpApprox: bool = False  # math.exp -> polynomial+IEEE-bit approx
    fastExpApproxDegree: int = 3  # polynomial degree for 2^f (1..10)
    # erf/gelu implementation selection (see libdevice.py). True (default) picks
    # the fast tanh-composite; False picks the accurate polynomial variant. Both
    # are set to False together by HEXAGON_ENABLE_ACCURACY in parse_options; the
    # trace-time behavior is driven by the same env read in libdevice.py, these
    # fields keep the option surface consistent and fold into the compile hash.
    enableErfTanhApprox: bool = True
    enableGeluTanhApprox: bool = True
    enableConversionToFp16: bool = False

    # Run the llvm-alignment-propagation pass to specialize the kernel to the
    # alignment of its pointer arguments. Disable when callers cannot
    # guarantee the alignment of input/output buffers.
    specializeKernelToArgAlignment: bool = False

    # Run set-alias-scopes in linalg-to-llvm. This pass assumes distinct kernel
    # pointer arguments satisfy the no-alias contract.
    enableSetAliasScopes: bool = False

    # This option enables 'seeding' of layout conversion ops around conv2d ops.
    # This introduces some builtin.unrealized_conversion_cast ops, that are expected
    # to be eliminated by conv -> hmx.conv2d pass. Till then, this pass could be
    # disabled (or there will be errors).
    enableSeedLayoutConversions: bool = False

    tileSizes: str = ""  # User defined tile sizes - for debugging purposes

    # Separate out constants (dense_resource) into separate shared objects
    lowerConstantsInSeparateSharedObjects: bool = False

    # light weight profiling using hardware instructions
    enableLWP: bool = False

    # Eliminate boundary_check on tt.load/tt.store when tensor_shape % block == 0.
    # ASSUMPTION: caller launches with grid[dim] = cdiv(shape[dim], block[dim]).
    # Default: False — user must explicitly opt in.
    eliminateSafeBoundaryChecks: bool = False

    # Emit an in-bounds scalar check (scf.if) in RewriteTensorDescriptorToPointer
    # so fully in-bounds tiles take an unmasked fast path and only boundary tiles
    # pay for mask computation + a masked load/store. Only affects descriptors
    # lowered by that pass, i.e. when the frontend TMA emulation
    # (TRITON_HEXAGON_TMA_INBOUNDS_FAST_PATH) is off. Default: False.
    rewriteDescriptorInboundsFastPath: bool = False

    # Rewrite inductor's single-sided runtime mask (xindex < xnumel) on
    # tt.load/tt.store: loads have their mask dropped entirely (unmasked);
    # stores are guarded by a per-program scalar in-bounds scf.if so only
    # the remainder tile pays the masked-store cost.
    rewriteMaskedAccessToInboundsFastPath: bool = False

    enableBroadcastedReductionToButterfly: bool = False

    enableCompositeOps: bool = False  # recognize and lower composite ops (e.g. softmax)

    # Run FastInversePass (div → Quake_rsqrt multiply). Default True.
    # Set False to keep the scalar sfrecipa path for benchmarking.
    enableFastInverse: bool = True

    outlineLinalgGeneric: bool = (
        False  # outline linalg.generic into noinline funcs to reduce compile time
    )

    # By default, loops are instrumented along with the function body.
    # To turn off loop level instrumentation, set it to True.
    disableLWPLoop: bool = False
    # By default, outer loop and nested sibling loops at LWPloopDepth 1 are instrumented.
    # Increasing LWPloopDepth may cause overhead.
    LWPloopDepth: int = 1

    # By default, delete all artifacts pushed to device for this kernel's execution after it runs.
    # This solely applies to execution on the standalone launcher.
    deviceCleanup: bool = True

    # DMA masked HMX matmul operands directly from the source buffer instead
    # of a staging copy.
    enableElideMaskedStaging: bool = False

    # NSP sharding options
    shardingAllowCollectives: bool = False
    # HVX workers per core that share the VTCM budget (0 = all of them).
    hmxNumWorkers: int = 0
    # Number of NSP cores available for dispatch (0 = backend default).
    # The MLIR attribute nsp.num_cores takes precedence when present.
    nspCores: int = 0
    # With HMX, shard small matmuls over fewer than nspCores cores so that each
    # core keeps at least this many multiply-accumulates (0 = all cores).
    nspMatmulMinMacsPerCore: int = 0

    # deallocAfterLastUse is mutually exclusive with the standard pipeline: when
    # true, HexagonDeallocAfterLastUse runs instead (threshold = deallocChunkBytes);
    # when false, the standard buffer-deallocation pipeline runs.
    deallocAfterLastUse: bool = False
    deallocChunkBytes: int = 0

    def __post_init__(self):
        # Validate target_artifact
        valid_artifacts = {"ttir", "ttsharedir", "llir", "o", "so"}
        if self.target_artifact not in valid_artifacts:
            raise ValueError(
                f"Invalid target_artifact '{self.target_artifact}'. "
                f"Must be one of: {', '.join(sorted(valid_artifacts))}"
            )

    def hash(self):
        key = "_".join([f"{name}-{val}" for name, val in self.__dict__.items()])
        # HEXAGON_LLVM_ARGS changes codegen but isn't a dataclass field;
        # fold it into the key so different flag settings don't share cache
        # slots.
        key += "_env:HEXAGON_LLVM_ARGS=" + knobs.hexagon.llvm_args
        return hashlib.md5(key.encode("utf-8")).hexdigest()
