# ===- compiler.py ----------------------------------------------------------===
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause.
# For more license information:
#   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
#
# ===------------------------------------------------------------------------===

import functools
import os
import re
import tempfile
from pathlib import Path
import subprocess
from dataclasses import dataclass, asdict
from typing import Any, Dict, Tuple, no_type_check
from types import ModuleType
import hashlib

from triton._C.libtriton import ir, passes, qcom_hexagon_backend  # type: ignore
from triton.backends.compiler import BaseBackend, GPUTarget
from triton.backends.qcom_hexagon_backend.utils import parse_return_types

# Temporary measure to compile .so for HTP without calling the Triton driver
from triton.backends.qcom_hexagon_backend.hexagon_executor import HexagonExecutor

# This file is part of a small subset of python files that uses some type-annotations
# and it passes type-verification with mypy (a type checker).
# To typecheck this set of files, do:
# mypy compiler.py hexagon_executor.py hexagon_launcher_base.py torch_mlir_hexagon_launcher.py triton_hexagon_launcher.py --follow-untyped-imports --check-untyped-defs


def _get_triton_shared_opt_path() -> str:
    path = os.getenv("TRITON_SHARED_OPT_PATH", "")
    if path == "":
        raise Exception("TRITON_SHARED_OPT_PATH is not set.")
    return path


def ttir_to_ttsharedir(mod):
    # Get Triton-MLIR as string
    ttir_code = str(mod)
    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = os.path.join(tmpdir, "tt.mlir")
        dst_path = os.path.join(tmpdir, "ttshared.mlir")
        Path(src_path).write_text(ttir_code)
        triton_shared_opt_path = _get_triton_shared_opt_path()
        subprocess.check_call(
            [
                triton_shared_opt_path,
                src_path,
                "--triton-to-linalg-experimental",
                "-o",
                dst_path,
            ]
        )

        return Path(dst_path).read_text()


# NOTE: This func has significant overlap with ttsharedir_to_obj(). This is intentional.
# This is a stopgap solution to allow more control over compiling kernels for QNN custom ops.
# When we later have a better way to pass compilation flags to ttsharedir_to_obj(),
# this function can be deprecated- we'd use ttsharedir_to_obj() in all cases.
def ttsharedir_to_llir(mod: str, options, metadata={}):
    context = ir.context()
    qcom_hexagon_backend.load_dialects(context)
    mlir_mod = qcom_hexagon_backend.parse_mlir_module_from_str(mod, context)

    metadata["name"] = qcom_hexagon_backend.extract_func_name_from_mlir_module(mlir_mod)
    return_values = qcom_hexagon_backend.get_return_list(mlir_mod, metadata["name"])
    metadata["return_types"] = parse_return_types(return_values)

    options_map = {k: str(v) for k, v in (options.__dict__).items()}
    return qcom_hexagon_backend.translate_linalg_to_llvmir(mlir_mod, options_map)


def ttsharedir_to_obj(mod: str, options, metadata={}) -> bytes:
    context = ir.context()
    qcom_hexagon_backend.load_dialects(context)
    # Temporary regex substitution to lower tt.scan
    mod = re.sub(r"\btt\.scan\b", "ttx.scan", mod)
    mlir_mod = qcom_hexagon_backend.parse_mlir_module_from_str(mod, context)

    # Parses kernel name and return signature from mlir module
    metadata["name"] = qcom_hexagon_backend.extract_func_name_from_mlir_module(mlir_mod)
    return_values = qcom_hexagon_backend.get_return_list(mlir_mod, metadata["name"])
    metadata["return_types"] = parse_return_types(return_values)

    options_map = {k: str(v) for k, v in (options.__dict__).items()}
    # TODO: Move setting benchmarking iterations when additional stage for shared object creation is part of compilation pipeline.
    metadata["iterations"] = options_map["iterations"]

    # TODO: The lowering pipeline needs to be refactored similar to other Triton backends to
    # have a dynamic pipeline filtered by options with each pass represented by a pybind function.
    # See make_ttgir() in nvidia backend as an example.
    mods_llvmir_bytes = qcom_hexagon_backend.translate_linalg_to_obj(
        mlir_mod, options_map
    )
    # Note: translate_linalg_to_obj() now returns a collection of object codes in general,
    # which in the case of the triton flow will only contain one element (i.e. one object code)
    # since there is no separation of constants for the triton flow.
    # So we need to get the first and unique one here
    principal_mod_bytes = mods_llvmir_bytes[0]
    return principal_mod_bytes


# Temporary measure for .so generation, only for Triton-HTP integration
def obj_to_so(mod, metadata={}) -> str:
    tmpdir = tempfile.mkdtemp()
    kernel_code = os.path.join(tmpdir, metadata["name"] + ".o")
    Path(kernel_code).write_bytes(mod)
    so_path = HexagonExecutor(compile_only=True).generate_shared_object(
        "", kernel_code, htp_kernel_gen=True
    )
    return so_path


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


class HexagonBackend(BaseBackend):
    # TODO: Setting the binary extension of the final executable as .o,
    # to be updated to .so if additional stages are added.
    binary_ext = "o"

    def __init__(self, target: GPUTarget) -> None:
        super().__init__(target)
        self.version_key = None

    @no_type_check  # Forced to ignore typing since base class uses deprecated annotation
    @staticmethod
    def supports_target(target: GPUTarget):
        return target.backend == "hexagon"

    # TODO: set version to hexagon version
    #       compute_core_version_key() seems to deprecated, need new approach
    @functools.lru_cache()
    def hash(self):
        version = "TODO"
        return f"{version}-{self.target}"

    def parse_options(self, opts) -> Any:
        args = {
            k: opts[k] for k in HexagonOptions.__dataclass_fields__.keys() if k in opts
        }
        return HexagonOptions(**args)

    @staticmethod
    def make_ttir(mod, metadata, opt):
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        passes.common.add_inliner(pm)
        passes.ttir.add_rewrite_tensor_pointer(pm)
        passes.ttir.add_rewrite_tensor_descriptor_to_pointer(pm)
        passes.common.add_canonicalizer(pm)
        passes.ttir.add_combine(pm)
        passes.ttir.add_reorder_broadcast(pm)
        passes.common.add_cse(pm)
        passes.ttir.add_triton_licm(pm)
        passes.common.add_symbol_dce(pm)
        passes.ttir.add_loop_unroll(pm)
        passes.common.add_cse(pm)
        pm.run(mod)
        return mod

    # May need to add num_warps
    def add_stages(self, stages, options, language):
        stages["ttir"] = lambda src, metadata: self.make_ttir(src, metadata, options)
        stages["ttsharedir"] = lambda src, metadata: ttir_to_ttsharedir(src)
        if options.htp_kernel_gen:
            if options.target_artifact == "llir":
                stages["llir"] = lambda src, metadata: ttsharedir_to_llir(
                    src, options, metadata
                )
                # A dummy "o" stage is necessary because Triton requires a stage corresponding to the default binary extention (".o")
                # There is no hook we can use to change the binary extension after HexagonBackend is instantiated but before
                # self.binary_ext is used. We cannot create the .o and the .llir file in the same pipeline, as the
                # output of one stage must directly feed into the next stage (and foregoing the llir stage leaves us
                # without a direct way to return the llir). The only way around this hack is to use the output of the llir
                # stage as the input of the .o stage, which we explicitly work around to support the torch-mlir workflow.
                stages["o"] = lambda src, metadata: b"__DUMMY_OBJ_FILE_PLACEHOLDER__"
            elif options.target_artifact == "so":
                stages["o"] = lambda src, metadata: ttsharedir_to_obj(
                    src, options, metadata
                )
                stages["so"] = lambda src, metadata: obj_to_so(src, metadata)
            else:  # target_artifact == "o"
                stages["o"] = lambda src, metadata: ttsharedir_to_obj(
                    src, options, metadata
                )
        else:  # Default compilation pipeline
            stages["o"] = lambda src, metadata: ttsharedir_to_obj(
                src, options, metadata
            )

    # Skipping those methods below for now

    def add_meta_info(self, ir, cur_module, next_module, metadata, asm):
        metadata["name"] = "hexagon_backend_name"
        metadata["shared"] = "0"

    def get_stream(self):
        return ""

    def get_current_device(self):
        return ""

    def set_current_device(self, device):
        pass

    def get_kernel_bin(self):
        return "tthir"

    def get_version_key(self):
        if self.version_key is None:
            self.version_key = compute_core_version_key()  # type: ignore[name-defined]
        return self.version_key

    # Loading dialects for each layer seperately
    def load_dialects(self, context):
        return

    def get_codegen_implementation(self, options):
        # We don't have limitations for minimum dot size for our architecture, so we just
        # add a placeholder with the minimum size values.
        codegen_fns = {"min_dot_size": lambda lhsType, rhsType: (1, 1, 1)}
        return codegen_fns

    def pack_metadata(self, metadata):
        # Putting these in so we're
        # consistent with other backends
        return (
            metadata.num_warps,
            metadata.num_ctas,
            metadata.shared,
            metadata.cluster_dims[0],
            metadata.cluster_dims[1],
            metadata.cluster_dims[2],
            metadata.name,
            metadata.return_types,
            metadata.iterations,
        )

    def get_module_map(self) -> Dict[str, ModuleType]:
        from triton.backends.qcom_hexagon_backend.hexagon_extern.hexagon import (
            libdevice,
        )

        return {"triton.language.extra.libdevice": libdevice}
