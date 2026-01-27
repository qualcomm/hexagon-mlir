# ===- mlir_launcher.py -----------------------------------------------------===
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause.
# For more license information:
#   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
#
# ===------------------------------------------------------------------------===

import os
from pathlib import Path
from triton.backends.qcom_hexagon_backend.hexagon_executor import HexagonExecutor
from triton.backends.qcom_hexagon_backend.hexagon_launcher_base import (
    create_timestamped_folder,
)
from triton._C.libtriton import qcom_hexagon_backend, ir  # type: ignore


"""
This launcher is only intended for unit testing small MLIR examples and/or
primiarily to be used for debugging only and not to run large models.

The main difference from other launchers is that reading and creating inputs
is deferred to the C++ stub along with validation of the result, if required.
This is done intentionally to keep the launcher to be as lean as possible.
"""


class MLIRHexagonLauncher:
    """
    Attributes:
        mlir_input_path - Name of external MLIR file
        cpp_wrapper_path - Path to external C++ stub
        validate_result - Boolean to extract and print perf report if it is generated in C++ stub
        func_name - Identifier for the kernel being executed.
    """

    @staticmethod
    def run_mlir_with_custom_cpp_wrapper(
        mlir_bytecode_path: str,
        cpp_wrapper_path: str,
        validate_result: bool,
        options: dict[str, str],
        func_name: str = "kernel",
    ):
        context = ir.context()
        qcom_hexagon_backend.load_dialects(context)
        mlir_mod = qcom_hexagon_backend.parse_mlir_module_from_file(
            mlir_bytecode_path, context
        )
        obj_modules: list[str] = qcom_hexagon_backend.translate_linalg_to_obj(
            mlir_mod, options
        )
        assert len(obj_modules) == 1, "Multiple modules not supported"

        local_dir_path = create_timestamped_folder(func_name)
        print(f"==> cpp wrapper path: {cpp_wrapper_path}")
        obj_src_path = os.path.join(local_dir_path, func_name + ".o")
        Path(obj_src_path).write_bytes(obj_modules[0])
        print(f"==> kernel obj saved in: {obj_src_path}")

        hexec = HexagonExecutor(options["enableLWP"])
        so_path = hexec.generate_shared_object(cpp_wrapper_path, obj_src_path)
        print("==> Shared object generated: ", so_path)
        hexec.run(
            paths_to_shared_libs_local=[so_path],
            input_tensor_paths=[],
            output_paths=[],
            generatePerf=validate_result,
        )
        if hexec.final_result != "Pass":
            raise RuntimeError(
                f"MLIR kernel {func_name} failed with result: {hexec.final_result}"
            )
