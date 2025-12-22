# ===- qaic_executor.py -----------------------------------------------------===
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause.
# For more license information:
#   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
#
# ===------------------------------------------------------------------------===

import hashlib
import logging
import os
import subprocess
import sys
from collections import namedtuple
from typing import Optional  # For type annotations
from torch import Tensor  # For type annotations
from triton.backends.qcom_hexagon_backend.utils import (  # type: ignore
    get_env_var,
    split_path,
)

from triton.backends.qcom_hexagon_backend.hexagon_executor import HexagonExecutor

# Get log level from environment variable, default to WARNING
log_level_str = os.getenv("LOG_LEVEL", "WARNING").upper()
log_level = getattr(logging, log_level_str, logging.WARNING)

# Configure logging
logging.basicConfig(
    level=log_level, format="%(asctime)s - %(levelname)s - %(message)s", force=True
)


class QAicExecutor(HexagonExecutor):
    def __init__(self, enable_lwp=False, enable_etm=False, compile_only=False):
        super().__init__(enable_lwp, enable_etm, compile_only)

    def get_config(self):
        """Configures HexagonExecutor with environment variables, tools paths, and Q6 version."""
        config = namedtuple("config", ("env_vars", "HEX_TOOLS", "Q6_VERSION"))

        # Define environment variables based on the execution mode
        env_vars = {
            "HEXAGON_TOOLS": "HEXAGON_TOOLS",
            "HEXAGON_MLIR_ROOT": "HEXAGON_MLIR_ROOT",
            "HEXAGON_SDK_ROOT": "HEXAGON_SDK_ROOT",
            "Q6_VERSION": "HEXAGON_ARCH_VERSION",
        }
        # Retrieve environment variables
        env = {key: get_env_var(val) for key, val in env_vars.items()}

        env["QAIC_HEXAGON_INCLUDE_DIR"] = get_env_var(
            "QAIC_HEXAGON_INCLUDE_DIR",
            "/opt/qti-aic/dev/inc/jit/",
        )
        env["QAIC_PLATFORM_JIT_INCLUDE_DIR"] = get_env_var(
            "QAIC_PLATFORM_JIT_INCLUDE_DIR",
            "/opt/qti-aic/dev/inc/jit/",
        )
        env["QAIC_TRITON_INCLUDE_DIR"] = get_env_var("QAIC_TRITON_INCLUDE_DIR", "")

        # List of tools to join with HEXAGON_TOOLS
        tools = ["hexagon-clang", "hexagon-clang++", "hexagon-sim"]
        HEX_TOOLS = {
            tool: os.path.join(env["HEXAGON_TOOLS"], "bin", tool) for tool in tools
        }

        Q6_VERSION = env["Q6_VERSION"]
        return config(env_vars=env, HEX_TOOLS=HEX_TOOLS, Q6_VERSION=Q6_VERSION)

    def run(
        self, func_name, paths_to_shared_libs_local: list[str], inputs: list[Tensor]
    ):
        if self.exec_mode == "simulator":
            self.run_kernel_on_simulator(func_name, paths_to_shared_libs_local, inputs)
        else:
            self.run_kernel_on_device(func_name, paths_to_shared_libs_local, inputs)

    def run_kernel_on_simulator(
        self, func_name, paths_to_shared_libs_local: list[str], inputs: list[Tensor]
    ):
        raise NotImplementedError("run_kernel_on_simulator() not implemented yet ")

    def run_kernel_on_device(
        self, func_name, paths_to_shared_libs_local: list[str], inputs: list[Tensor]
    ):
        from torch_qaic._C._custom_ops import _dispatch_kernel

        so_path = paths_to_shared_libs_local[0]
        entry_function = "kernel_" + hashlib.md5(func_name.encode("utf-8")).hexdigest()
        try:
            # TODO: Currently runs with single NSP and thread; will use metadata soon.
            _dispatch_kernel(so_path, entry_function, 1, 1, *inputs)
            logging.info(
                f"✅ Successfully executed {entry_function} kernel on QAic device!"
            )
        except Exception as e:
            logging.warning(
                f"❌ Failed to execute {entry_function} kernel on QAic device with error: {e}"
            )
            raise

    def generate_shared_object(
        self,
        cpp_wrapper_path: Optional[str],
        kernel_obj_path: str,
        lib_dependencies_paths: list[str] = [],
    ) -> str:
        logging.info(f"wrapper_path =  {cpp_wrapper_path}")
        logging.info(f"kernel_code =  {kernel_obj_path}")
        hexagon_compile_include_string = """-I{QAIC_TRITON_INCLUDE_DIR} -I{QAIC_HEXAGON_INCLUDE_DIR} -I{QAIC_PLATFORM_JIT_INCLUDE_DIR}"""
        runtime_libs = ["qhmath_hvx"]

        HEX_CXX_FLAGS = f"-O3 -mv{self.config.Q6_VERSION} -mhvx"
        INCLUDES = hexagon_compile_include_string.format(
            QAIC_TRITON_INCLUDE_DIR=self.config.env_vars["QAIC_TRITON_INCLUDE_DIR"],
            QAIC_HEXAGON_INCLUDE_DIR=self.config.env_vars["QAIC_HEXAGON_INCLUDE_DIR"],
            QAIC_PLATFORM_JIT_INCLUDE_DIR=self.config.env_vars[
                "QAIC_PLATFORM_JIT_INCLUDE_DIR"
            ],
        )

        QHL_LINK_DIR = """{HEXAGON_SDK_ROOT}/libs/qhl_hvx/prebuilt/hexagon_toolv87_v{Q6_VERSION}""".format(
            HEXAGON_SDK_ROOT=self.config.env_vars["HEXAGON_SDK_ROOT"],
            Q6_VERSION=self.config.Q6_VERSION,
        )

        if not (
            os.path.exists(QHL_LINK_DIR)
            and os.path.exists(QHL_LINK_DIR + "/libqhmath_hvx.a")
        ):
            print(
                f"QHL_HVX Library was not found. Check that {QHL_LINK_DIR} exists in your machine and that libqhmath_hvx.a exists within {QHL_LINK_DIR}."
            )
            sys.exit(1)

        LINK_DIRS = """ -L{}""".format(
            QHL_LINK_DIR,
        )
        runtime_libs_str = " ".join([f"-l{lib}" for lib in runtime_libs])
        logging.info("==> Compiling shared object file ...")
        dir, kernel_file_without_ext, _ = split_path(kernel_obj_path)
        so_name = "lib" + kernel_file_without_ext
        kernel_path = os.path.join(dir, f"{so_name}.so")
        # If the shared object we're generating needs to be linked with some dependencies, extract the base directory
        # of those libs and their names for latter providing them to the -L and -l options of the link command
        if lib_dependencies_paths:
            (
                dependencies_base_dir,
                lib_names_extracted,
            ) = self.validate_and_extract_lib_names(lib_dependencies_paths)

        command = "{} -nostdinc++ -std=c++17 -ffast-math -fno-finite-math-only -fno-exceptions -fno-threadsafe-statics -fno-builtin -fvisibility=default -ffreestanding -O3 -mllvm -hexagon-commgep=false -mllvm -hexagon-postinc-opt -mhvx-ieee-fp {} {} {} {} {} {} {} -shared -fPIC -G0 -nostdlib++ -nostdlib -o {}".format(
            self.config.HEX_TOOLS["hexagon-clang++"],
            HEX_CXX_FLAGS,
            INCLUDES,
            LINK_DIRS,
            cpp_wrapper_path,
            kernel_obj_path,
            runtime_libs_str,
            (
                ""
                if not lib_dependencies_paths
                else "-L{} -l{}".format(
                    dependencies_base_dir, " -l".join(lib_names_extracted)
                )
            ),
            kernel_path,
        )
        logging.info(command)
        try:
            subprocess.run(
                command,
                shell=True,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.STDOUT,
            )
        except subprocess.CalledProcessError as e:
            logging.warning(f"Error executing the command: {e}")
            sys.exit(1)
        return kernel_path
