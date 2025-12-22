# ===- triton_qaic_launcher.py ----------------------------------------------===
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
import shutil
from collections import namedtuple
from typing import Optional  # For type annotations
from torch import Tensor  # For type annotations
from triton.backends.qcom_hexagon_backend.utils import (  # type: ignore
    get_env_var,
    split_path,
)

from pathlib import Path
from triton.backends.qcom_hexagon_backend.hexagon_executor import HexagonExecutor
from triton.backends.qcom_hexagon_backend.qaic_executor import QAicExecutor
from triton.backends.qcom_hexagon_backend.triton_hexagon_launcher import (
    TritonWrapperGeneratorStrings,
    TritonHexagonWrapperGenerator,
)
from triton.backends.qcom_hexagon_backend.compiler import (
    HexagonOptions,
)
from triton.backends.qcom_hexagon_backend.hexagon_launcher_base import (
    HexagonLauncherBase,
    HexagonWrapperGenerator,
    create_timestamped_folder,
)
from triton.backends.qcom_hexagon_backend.utils import (
    profile_triton_inputs,
)


# Get log level from environment variable, default to WARNING
log_level_str = os.getenv("LOG_LEVEL", "WARNING").upper()
log_level = getattr(logging, log_level_str, logging.WARNING)

# Configure logging
logging.basicConfig(
    level=log_level, format="%(asctime)s - %(levelname)s - %(message)s", force=True
)


class TritonQAicWrapperGeneratorString(TritonWrapperGeneratorStrings):
    def __init__(self):
        super().__init__()
        self.tensor_init_template = """
        Tensor<{tensor_ctype}, {tensor_rank}> {input_tensor_prefix}{i}({{{sizes}, {strides}, 128}}, MemType::HEAP);\nMemRefDescriptor<{tensor_ctype}, {tensor_rank}> *{memrefdesc_name}{i} = {input_tensor_prefix}{i}.toMemRefDesc();\n {memrefdesc_name}{i}->aligned = extractArg<{tensor_ctype}*, {i}>(pointerArray); \n """
        # Codegen string for the data structures and functions defined before the main body.
        self.code_define = """
    {llvm_func_sign}
    {input_wrapper_struct_def}
    {result_struct_def}
    """

        # Codegen string for the contents of main body.
        self.code_body = """
    {entry_function_signature} {{
    {tensor_definition_str}
    {input_wrapper_structs_init}
    {result_struct_init}
    {llvm_function_call}
    return 0;
    }}
    """
        self.code_headers = """
    #include <string.h>
    #include <stdint.h>
    #include "common.h"
    #include "tensor.h"
    #include "jit_dev_exe_function.h"

    // Traits class definition
    template <typename T>
    struct ExtractArgTraits {
      static T extract(void* ptr) {
        return static_cast<T>(ptr);
      }
    };
    // Specialization for const long
    template <>
    struct ExtractArgTraits<const long> {
      static long extract(void* ptr) {
        return *reinterpret_cast<const long*>(ptr);
      }
    };
    // Specialization for const int32_t
    template <>
    struct ExtractArgTraits<const int> {
      static int extract(void* ptr) {
        return *reinterpret_cast<const int*>(ptr);
      }
    };
    template <typename T, uint32_t Index_>
    static inline T extractArg(const AicJitPointerArray* pointerArray) {
      return ExtractArgTraits<T>::extract(pointerArray->pointers[Index_]);
    }
    """


class TritonQAicWrapperGenerator(TritonHexagonWrapperGenerator):
    def __init__(
        self,
        input_profs,
        iterations,
        func_name,
        output_profs,
        launch_grid,
        options: dict,
    ):
        super().__init__(
            input_profs, iterations, func_name, output_profs, launch_grid, options
        )
        self.common_strings = TritonQAicWrapperGeneratorString()

    def generate_entry_function_signature(self):
        """Generates the entry function signature for the kernel"""
        entry_function = (
            "kernel_" + hashlib.md5(self.func_name.encode("utf-8")).hexdigest()
        )
        return f"uint32_t {entry_function}(const AicJitEntryPointConfig *entryConfig, const AicJitPointerArray *pointerArray)"

    def generate_cpp_code_headers(self) -> str:
        """Generates C++ code headers"""
        code_headers = self.common_strings.code_headers
        return code_headers

    def generate_cpp_code_define(self) -> str:
        """Generates C++ code defines including LLVM function signature and structs"""
        code_define = self.common_strings.code_define.format(
            llvm_func_sign=self.generate_llvm_function_signature(),
            input_wrapper_struct_def=self.generate_input_wrapper_struct_def(),
            result_struct_def=self.generate_result_struct(),
        )
        return code_define

    def generate_cpp_code_body(self, file_name, exec_dir) -> str:
        """Generates the main body of the C++ wrapper code"""
        code_body = self.common_strings.code_body.format(
            entry_function_signature=self.generate_entry_function_signature(),
            tensor_definition_str=self.generate_input_declarations(),
            input_wrapper_structs_init=self.generate_input_wrapper_structs_init(),
            result_struct_init=self.generate_result_struct_init(),
            llvm_function_call=self.generate_llvm_function_call(),
        )
        return code_body

    def generate_cpp_wrapper(self, file_name, exec_dir) -> str:
        """Generates the complete C++ wrapper file including headers, defines, and body"""
        code_headers = self.generate_cpp_code_headers()
        code_define = self.generate_cpp_code_define()
        code_body = self.generate_cpp_code_body(file_name, exec_dir)
        code_body = f"""
        extern "C" {{
        {code_body}
        }}
        """
        return self.common_strings.code_string.format(
            code_headers=code_headers, code_define=code_define, code_body=code_body
        )


class TritonQAicLauncher(HexagonLauncherBase):
    def execute_kernel(
        self,
        hexec: HexagonExecutor,
        inputs: list[Tensor],
        local_dir: str,
        func_name: str,
        paths_to_shared_libs_generated: list[str],
        wrapper_generator: HexagonWrapperGenerator,
    ):
        # Run the kernel
        hexec.run(func_name, paths_to_shared_libs_generated, inputs)

    def _exec_kernel(
        self,
        kernel_obj_as_bytes: bytes,
        iterations: int,
        func_name: str,
        inputs: list[Tensor],
        output_profs: list,
        launch_grid: tuple,
    ) -> list[Tensor]:
        # Getting the input metadata for effective wrapper codegen.
        input_profs = profile_triton_inputs(inputs)

        local_dir_path = create_timestamped_folder(func_name)
        # 1 - Writting the object code for the kernel to disk
        obj_src_path = os.path.join(local_dir_path, func_name + ".o")
        Path(obj_src_path).write_bytes(kernel_obj_as_bytes)
        logging.info(f"==> kernel obj saved in: {obj_src_path}")

        # Get HexagonOptions at this point for triton pipeline.
        options = HexagonOptions().__dict__

        hexec = QAicExecutor()

        # Pass options to the wrapper generator.
        # TritonHexagonWrapperGenerator will assert for lwp and multithreading conflict.
        # HexagonWrapperGenerator will create the call to WriteLWPOutput() if lwp is enabled.
        wrapper_generator = TritonQAicWrapperGenerator(
            input_profs, iterations, func_name, output_profs, launch_grid, options
        )
        logging.info("==> Wrapper generator correctly instantiated")

        # The directory path used for execution is given by the executor, and if it's the empty string (meaning running on device),
        # then the execution directory path will be set to the local directory path
        exec_dir = hexec.get_Executable_Path()
        # If no executable path specified, using local directory.
        if not exec_dir:
            exec_dir = local_dir_path

        # 2 - Generating and writing the wrapper to disk
        cpp_wrapper_path = self.generate_and_dump_wrapper(
            wrapper_generator, local_dir_path, func_name, exec_dir
        )

        # 3 - Generate the corresponding shared object for the current object file
        so_path = hexec.generate_shared_object(cpp_wrapper_path, obj_src_path)
        logging.info(f"==> Shared object generated:  {so_path}")
        self.execute_kernel(
            hexec, inputs, local_dir_path, func_name, [so_path], wrapper_generator
        )
        return inputs
