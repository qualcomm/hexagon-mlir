# ===- stablehlo_hexagon_launcher.py ---------------------------------------===
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause.
# For more license information:
#   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
#
# ===------------------------------------------------------------------------===

import shutil
import time
from pathlib import Path

from torch import Tensor  # For type annotations
from triton._C.libtriton import ir, qcom_hexagon_backend  # type: ignore
from triton.backends.qcom_hexagon_backend.compiler import HexagonOptions
from triton.backends.qcom_hexagon_backend.hexagon_executor import HexagonExecutor
from triton.backends.qcom_hexagon_backend.hexagon_launcher_base import (
    HexagonLauncherBase,
    HexagonWrapperGenerator,
    MLIRHexagonWrapperGenerator,
    create_timestamped_folder,
)
from triton.backends.qcom_hexagon_backend.utils import (
    parse_return_types,
    profile_inputs,
)


class StableHLOHexagonLauncher(HexagonLauncherBase):
    """Link and run a precompiled StableHLO-derived Hexagon object."""

    def run_from_obj(
        self,
        obj_path: str | Path,
        mlir_meta_path: str | Path,
        inputs: list[Tensor],
        func_name: str,
        base_dir_for_artifacts: str | Path | None = None,
        iterations: int = 1,
        options: dict | None = None,
        enable_etm: bool = False,
        cleanup_device_post_exec: bool = True,
        compile_only: bool = False,
        kernel_run_id: str | None = None,
    ) -> list[Tensor] | None:
        """Link a prebuilt object using MLIR metadata only for its return types."""
        if options is None:
            options = HexagonOptions().__dict__

        local_dir_path = create_timestamped_folder(func_name, base_dir_for_artifacts)
        run_id = kernel_run_id or local_dir_path.name
        hexec = HexagonExecutor(
            kernel_run_id=run_id,
            enable_lwp=options["enableLWP"],
            enable_etm=enable_etm,
            enable_hexkl=options["enableHexKL"],
            cleanup_device_post_exec=cleanup_device_post_exec,
            compile_only=compile_only,
        )

        context = ir.context()
        qcom_hexagon_backend.load_dialects(context)
        meta_mod = qcom_hexagon_backend.parse_mlir_module_from_file(
            str(mlir_meta_path), context
        )
        result_types = qcom_hexagon_backend.get_return_list(meta_mod, func_name)
        return_types = parse_return_types(result_types)

        func_name_with_ciface = "_mlir_ciface_" + func_name
        obj_dst_path = local_dir_path / f"{func_name_with_ciface}.o"
        shutil.copy(obj_path, obj_dst_path)
        print(f"==> kernel obj staged at: {obj_dst_path}", flush=True)

        if func_name == "main":
            self.rename_kernel_main_symbol(obj_dst_path, hexec)

        wrapper_generator = MLIRHexagonWrapperGenerator(
            profile_inputs(inputs, rank_zero_tensors_as_scalars=False),
            iterations,
            func_name_with_ciface,
            return_types,
            options,
        )
        exec_dir = hexec.get_Executable_Path() or str(local_dir_path)
        cpp_wrapper_path = self.generate_and_dump_wrapper(
            wrapper_generator,
            local_dir_path,
            func_name_with_ciface,
            exec_dir,
        )
        print(f"==> Generated cpp wrapper: {cpp_wrapper_path}", flush=True)

        start_time = time.time()
        so_path = hexec.generate_shared_object(cpp_wrapper_path, obj_dst_path, [])
        print(
            f"obj->so (with wrapper) took {time.time() - start_time:.3f}s  (-> {so_path})",
            flush=True,
        )

        if compile_only:
            print("Compile-only mode: skipping execution.", flush=True)
            return None

        return self.execute_kernel(
            hexec,
            local_dir_path,
            func_name_with_ciface,
            [so_path],
            wrapper_generator,
        )

    @staticmethod
    def get_output_tensor_path_count(
        wrapper_generator: HexagonWrapperGenerator,
    ) -> int:
        return sum(1 for output in wrapper_generator.output_profs if output.rank)
