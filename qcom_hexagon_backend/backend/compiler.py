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
from typing import Any, Dict, no_type_check
from types import ModuleType

from triton._C.libtriton import ir, passes, qcom_hexagon_backend  # type: ignore
from triton.backends.compiler import BaseBackend, GPUTarget
from triton.backends.qcom_hexagon_backend.utils import parse_return_types

# Temporary measure to compile .so for HTP without calling the Triton driver
from triton.backends.qcom_hexagon_backend.hexagon_executor import HexagonExecutor
from .hexagon_options import HexagonOptions

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
      assert self.target.backend == "hexagon"
      args = {
        k: opts[k]
        for k in HexagonOptions.__dataclass_fields__.keys()
        if k in opts
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
