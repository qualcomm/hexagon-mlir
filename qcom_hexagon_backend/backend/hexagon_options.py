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
    supported_fp8_dtypes: Tuple[str, ...] = ()
    sanitize_overflow: bool = True
    debug: bool = False
    # TODO: Currently set name here temporarily. Need to set correctly to the name of the compiled kernel
    # when driver code is completed.
    name: str = "Hexagon"
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
    convTilingFactor: int = -1
    convTileHeightDim: bool = False
    convTileOCDim: bool = False
    enableConvertToHexagonmem: bool = True  # rewrites memref.alloc/copy to hexagonmem.*
    enableHexagonmemCopyToDMA: bool = False  # rewrites hexmem.copy to memref.dma_*
    enableHexKL: bool = False  # use HexKL to lower matmul and convolutions
    enableMultiThreading: bool = False  # linalg-generic based multi-threading
    enableSCFThreading: bool = False  # scf based multi-threading
    enableSplitReduction: bool = False  # split-reduction optimization
    enableVectorization: bool = True  # enable HVX vectorization.
    enableVTCMTiling: bool = (
        True  # tile linalg-generic and introduce vtcm address-space
    )

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

    # By default, loops are instrumented along with the function body.
    # To turn off loop level instrumentation, set it to True.
    disableLWPLoop: bool = False
    # By default, outer loop and nested sibling loops at LWPloopDepth 1 are instrumented.
    # Increasing LWPloopDepth may cause overhead.
    LWPloopDepth: int = 1

    def __post_init__(self):
        # Validate target_artifact
        valid_artifacts = {"o", "llir", "so"}
        if self.target_artifact not in valid_artifacts:
            raise ValueError(
                f"Invalid target_artifact '{self.target_artifact}'. "
                f"Must be one of: {', '.join(sorted(valid_artifacts))}"
            )

    def hash(self):
        key = "_".join([f"{name}-{val}" for name, val in self.__dict__.items()])
        return hashlib.md5(key.encode("utf-8")).hexdigest()
