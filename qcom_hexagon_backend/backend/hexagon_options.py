# ===- hexagon_options.py ---------------------------------------------------===
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause.
# For more license information:
#   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
#
# ===------------------------------------------------------------------------===

import os
from dataclasses import dataclass
from typing import Tuple
import hashlib


@dataclass(frozen=True)
class HexagonOptions:
    allow_fp8e4nv: bool = False
    allowed_dot_input_precisions: Tuple[str] = ("ieee",)
    arch_triple: str = "hexagon"
    arch_features: str = f'+hvxv{os.getenv("HEXAGON_ARCH_VERSION")},+hvx-length128b'
    device_type: str = "hexagon"
    vectorize: int = 1
    vector_length: int = 32
    num_threads: int = 4
    data_layout: str = (
        "e-m:e-p:32:32:32-a:0-n16:32-i64:64:64-i32:32:32-i16:16:16-i1:8:8-f32:32:"
        "32-f64:64:64-v32:32:32-v64:64:64-v512:512:512-v1024:1024:1024-v2048:2048:2048"
    )
    # Part of the launch-option contract with Triton core, not tuning knobs of
    # this backend: JITFunction._pack_args (triton/python/triton/runtime/jit.py)
    # raises KeyError for any launch kwarg that is absent from
    # `options.__dict__`, and core injects these as launch defaults. Deleting
    # them here (they are never branched on locally) makes every launch die with
    # "Keyword argument <name> was specified but unrecognised" -- measured on
    # device: all 7 tests failed in <2s each. Keep them.
    num_warps: int = 1
    num_stages: int = 1
    num_ctas: int = 1
    shared: bool = False
    cluster_dims: tuple = (1, 1, 1)
    supported_fp8_dtypes: Tuple[str, ...] = ()
    sanitize_overflow: bool = True
    debug: bool = False
    # Fallback only: a real compilation overwrites metadata["name"] with the
    # function name extracted from the MLIR module (backend/compiler.py), and
    # that is the symbol pack_metadata() hands to the launcher.
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
    hexKLMode: str = "micro"  # possible options "macro", "micro"
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
    enableVTCMTiling: bool = (
        True  # tile linalg-generic and introduce vtcm address-space
    )
    # External VTCM scratch buffer size in bytes per program instance.
    # scratch=0 (default): disabled, each instance allocates VTCM internally.
    # scratch=N (N>0): enables external VTCM flow. The wrapper allocates
    # N * prod(grid) bytes total, passes each instance a memref<Nxi8, 1>
    # {hexagon.scratch} slice. The compiler tiles and plans within N bytes.
    scratch: int = 0
    enableHVXInlining: bool = False
    enableSCFLoopUnroll: bool = False
    enableConversionToFp16: bool = False

    # Runtime-weight residency (P2). When on, the compiler drops a runtime
    # weight's per-launch `hmx.pack_weight` bridge and instead reads it from a
    # resident VTCM buffer; the launcher pre-packs the weight once per process
    # using the module's `hmx.weight_prepack` metadata. On by default: the
    # generated launcher always pre-packs, and a weight the pass cannot prove
    # dense (a dynamic-offset N-split view) keeps its old bridge rather than
    # being mis-packed. Measured at steady state: S1 -8.5%, S2 -19.8%, S3 -25.0%.
    enableWeightResident: bool = True

    # 2-D last-dim linalg.reduce (f16/f32 max/add) lowered to per-vector
    # elementwise folding plus an hvx.vror butterfly (vector-row-reduce pass)
    # instead of the scalarized per-lane chain the Hexagon backend produces for
    # vector.reduce.fmax. Off by default: the device A/B switch.
    enableVectorRowReduce: bool = False

    # HMX tile-level software-pipeline depth (hmx-partition). 0 = auto (the
    # deepest activation-staging ring the VTCM budget and the tile count allow),
    # 1 = force the serial ring, 2 = request the double ring (narrowed to the
    # deepest ring that fits, with a remark, when the budget cannot pay for it),
    # 3 = skip staging and emit the unstaged serial tile loop (the third A/B arm:
    # no hmx.stage/hmx.await, the activation bridge is kept).
    enableHmxPipelineDepth: int = 0

    # Per-launch VTCM workspace residency (hmx-workspace-resident). When on, the
    # crouton arrays, conversion state, staging ring/scratch and statuses of an
    # HMX kernel are allocated once per process and reused by every launch
    # instead of being allocated/freed per launch. Off by default: a resident
    # buffer is shared by all launches in the process, which is only correct for
    # single-instance execution -- a grid>1 launch would run several kernel
    # instances over the same buffers.
    enableWorkspaceResident: bool = False

    # Unit-test-only: seeds layout conversion ops around conv2d ops, which
    # introduces builtin.unrealized_conversion_cast ops. It is wired behind
    # `enableMatmulToConv && enableSeedLayoutConversions` in
    # LinalgToLLVMPass.cpp, and `enableMatmulToConv` has no field here, so from
    # this backend the option alone is a no-op (it does not error). Nothing
    # eliminates those casts either: the hmx dialect declares matmul-to-hmx /
    # hmx-partition / weight-resident / hmx-workspace-resident only -- there is
    # no conv -> hmx pass yet.
    enableSeedLayoutConversions: bool = False

    # Upstream crouton/pack machinery. The pack frontier extension is on by
    # default upstream; the HVX croutonization pass is not, and it is what turns
    # a crouton layout conversion into HVX permutes instead of element-wise moves.
    extendPackUpperFrontier: bool = True
    extendPackLowerFrontier: bool = True
    forceHVXCroutonization: bool = False

    tileSizes: str = ""  # User defined tile sizes - for debugging purposes

    # Separate out constants (dense_resource) into separate shared objects
    lowerConstantsInSeparateSharedObjects: bool = False

    # light weight profiling using hardware instructions
    enableLWP: bool = False

    # By default, loops are instrumented along with the function body.
    # To turn off loop level instrumentation, set it to True.
    disableLWPLoop: bool = False
    # By default, outer loop and nested sibling loops at LWPloopDepth 1 are instrumented.
    # Increasing LWPloopDepth may cause overhead.
    LWPloopDepth: int = 1

    # By default, delete all artifacts pushed to device for this kernel's execution after it runs.
    # This solely applies to execution on the standalone launcher.
    deviceCleanup: bool = True

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
        return hashlib.md5(key.encode("utf-8")).hexdigest()
