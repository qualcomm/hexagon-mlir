# ===- triton_hexagon_launcher.py -------------------------------------------===
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause.
# For more license information:
#   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
#
# ===------------------------------------------------------------------------===

import os
import sys
from pathlib import Path
from math import prod
from torch import Tensor  # For type annotations
from triton.backends.qcom_hexagon_backend.compiler import (
    HexagonOptions,
)
from triton.backends.qcom_hexagon_backend.hexagon_executor import HexagonExecutor
from triton.backends.qcom_hexagon_backend.hexagon_launcher_base import (
    HexagonLauncherBase,
    HexagonWrapperGenerator,
    WrapperGeneratorStrings,
    create_timestamped_folder,
)
from triton.backends.qcom_hexagon_backend.utils import (
    parse_triton_llvm_kernel_signature,
    profile_triton_inputs,
)

# This file is part of a small subset of python files that uses some type-annotations
# and it passes type-verification with mypy (a type checker).
# To typecheck this set of files, do:
# mypy compiler.py hexagon_executor.py hexagon_launcher_base.py torch_mlir_hexagon_launcher.py triton_hexagon_launcher.py --follow-untyped-imports --check-untyped-defs


class TritonWrapperGeneratorStrings(WrapperGeneratorStrings):
    def __init__(self):
        super().__init__()

        self.closure_struct_def = """
// Packs all arguments for the function.
struct Closure {{
    {closure_fields}
    int num_programs_X;
    int num_programs_Y;
    int num_programs_Z;
    int pid_X;
    int pid_Y;
    int pid_Z;
}};
"""
        self.multithread_helper = """
void multithread_helper(void *arg) {{
    Closure *closure = (Closure *)arg;
    {func_name}({closure_args_string}
               closure->num_programs_X,
               closure->num_programs_Y,
               closure->num_programs_Z,
               closure->pid_X,
               closure->pid_Y,
               closure->pid_Z);
}}
"""
        self.thread_pool_setup = """
ThreadManager<Closure> tm({num_threads});
std::vector<Closure> closures(tm.thread_cnt());
for (int pid_X = 0; pid_X < {grid[0]}; pid_X++) {{
    for (int pid_Y = 0; pid_Y < {grid[1]}; pid_Y++) {{
        for (int pid_Z = 0; pid_Z < {grid[2]}; pid_Z++) {{
            int flat_pid = pid_X * {strides[0]} +
                       pid_Y * {strides[1]} +
                       pid_Z * {strides[2]};
            closures[flat_pid] = {{{closure_args_string}
                                  {grid[0]}, {grid[1]}, {grid[2]},
                                  pid_X, pid_Y, pid_Z}};
        }}
    }}
}}
"""
        self.thread_pool_exec = """tm.exec(multithread_helper, closures.data());"""

        # Wrapper for inputs when there are >0 direct return values (changes the calling convention)
        self.input_struct_template = """
template<typename T, int N>
struct FuncInput {
    int64_t rank;
    MemRefDescriptor<T, N> *p0;
};
"""

        # Initialization for inputs when there are >0 direct return values
        self.input_struct_init = """
FuncInput<{tensor_ctype}, {tensor_rank}> {input_wrapper_name}{i} = {{{tensor_rank}, {memrefdesc_name}{i}}};
FuncInput<{tensor_ctype}, {tensor_rank}> *{input_wrapper_ptr_name}{i} = &{input_wrapper_name}{i};\n
"""

        # Additional headers needed for multithreading
        self.triton_code_headers = """
#include "multithreading.h"
"""

        # Additional functions and structs for Multithreading
        self.triton_code_define = """
{closure_struct_def}
{multithread_helper}
"""

        # Main triton cpp code body.
        self.triton_code_body = """
int main() {{
{tensor_definition_str}
{read_from_file_calls}
{setup_thread_pool}
{benchmarking_and_reporting}
{update_tensor}
{write_to_file_calls}
return 0;
}}
"""


"""
    Wrapper generator that contains the Triton related codegen utilities used to generate the CPP launcher.
    In this implementation, the codegen is based on the assumption that the inputs tensors are converted to
    Unranked Memrefs in the MLIR -> LLVM IR lowering process.
"""


class TritonHexagonWrapperGenerator(HexagonWrapperGenerator):
    def __init__(
        self,
        input_profs,
        iterations,
        func_name,
        output_profs,
        launch_grid,
        options: dict,
    ):

        # Setting the grid and grid_strides which correspond to the Triton kernel launch grid.
        self.grid = launch_grid
        self.grid_strides = (launch_grid[1] * launch_grid[2], launch_grid[2], 1)
        # Set a flag to check if multithreading codegen should be enabled.
        self.multithreading_enabled = prod(self.grid) > 1

        assert not (
            self.multithreading_enabled and options["enableLWP"]
        ), "LWP is not supported with multithreading. Please disable LWP."
        super().__init__(
            input_profs,
            iterations,
            func_name,
            output_profs,
            TritonWrapperGeneratorStrings(),
            options,
        )

    def generate_llvm_function_signature(self):
        """Generate LLVM function definition for Triton"""
        if len(self.output_profs) == 0:
            function_arg_string = self.generate_llvm_function_signature_arg_string()
        else:
            function_arg_string = self.generate_llvm_direct_returns_function_signature()
        # Represents the num_programs(x,y,z) and program_id(x,y,z) arguments
        function_arg_string += ", int, int, int, int, int, int"
        return self.common_strings.extern_llvm_func_defn.format(
            kernel_name=self.func_name, function_arg_string=function_arg_string
        )

    def generate_llvm_direct_returns_function_signature(self):
        """Generates LLVM func def. for Triton with >0 return values- the input calling convention changes"""
        function_arg_string = "void *, "
        for inp in self.input_profs:
            if inp.input_type == "tensor":
                extern_args = "void *, "
            else:
                extern_args = self.common_strings.extern_llvm_func_args_scalar.format(
                    scalar_ctype=inp.dtype
                )
            function_arg_string += extern_args
        return function_arg_string[:-2]

    def generate_llvm_function_call(self):
        """Generates actual function call to lowered LLVM function call"""
        if len(self.output_profs) == 0:
            function_call_descriptor_string = (
                self.generate_llvm_function_call_arg_string()
            )
        else:
            function_call_descriptor_string = (
                self.generate_llvm_direct_returns_function_call_arg_string()
            )
        # Since this is used for the single threaded case only, the num_programs is (1,1,1) and program_ids is (0,0,0).
        function_call_descriptor_string += ", 1, 1, 1, 0, 0, 0"
        return self.common_strings.extern_llvm_func_call.format(
            func_name=self.func_name, descriptor_string=function_call_descriptor_string
        )

    def generate_llvm_direct_returns_function_call_arg_string(self):
        # The function call contains the pointer to the Result as the first parameter
        function_call_descriptor_string = "r,"
        for inp in self.input_profs:
            if inp.input_type == "tensor":
                tensor_idx = inp.idx
                function_call_descriptor_string += (
                    f"{self.common_strings.input_wrapper_ptr_name}{tensor_idx}, "
                )
            else:
                scalar_index = inp.idx
                function_call_descriptor_string += (
                    f"{self.common_strings.scalar_name}{scalar_index}, "
                )
        return function_call_descriptor_string[:-2]
        """Generates LLVM func call args for Triton with >0 return values- the input calling convention changes"""

    def generate_input_wrapper_struct_def(self):
        """Generates template for input wrapper structs (only used by Triton)"""
        if len(self.output_profs) > 0:
            return self.common_strings.input_struct_template
        else:
            return ""

    def generate_input_wrapper_structs_init(self):
        """Generates initializations for input wrapper structs (only be used by Triton)"""
        formatted_tensor_string = ""
        if len(self.output_profs) > 0:
            for inp in self.input_profs:
                # This indirect calling convention should only be required for tensors
                if inp.input_type == "tensor":
                    tensor_declaration = self.common_strings.input_struct_init.format(
                        tensor_ctype=inp.dtype,
                        tensor_rank=inp.rank,
                        i=inp.idx,
                        memrefdesc_name=self.common_strings.memrefdesc_name,
                        input_wrapper_name=self.common_strings.input_wrapper_name,
                        input_wrapper_ptr_name=self.common_strings.input_wrapper_ptr_name,
                    ).lstrip()
                    formatted_tensor_string += tensor_declaration
        return formatted_tensor_string

    def generate_closure_definition(self):
        """
        Generates string for closure struct.
        Only used when multithreading.
        The 6 previously unused args now map to:
            - 3 args for num_programs (1 for each axis)
            - 3 args for program_id (1 for each exis)
        """
        closure_fields_string = ""
        indent = " " * 4
        for inp in self.input_profs:
            if inp.input_type == "tensor":
                closure_fields_string += f"{indent}int64_t t{str(inp.idx)}Rank;\n"
                closure_fields_string += f"{indent}MemRefDescriptor<{inp.dtype}, {inp.rank}> *t{str(inp.idx)}MemRef;\n"
            else:
                closure_fields_string += f"{indent}{inp.dtype} S{str(inp.idx)};\n"

        return self.common_strings.closure_struct_def.format(
            closure_fields=closure_fields_string.strip()
        )

    def generate_multithread_helper(self):
        """
        Generates closure cast and call to LLVM function.
        Only used when multithreading.
        """
        closure_helper_string = ""
        indent = " " * 15
        for inp in self.input_profs:
            if inp.input_type == "tensor":
                closure_helper_string += f"{indent}closure->t{str(inp.idx)}Rank,\n"
                closure_helper_string += f"{indent}closure->t{str(inp.idx)}MemRef,\n"
            else:
                closure_helper_string += f"{indent}closure->S{str(inp.idx)},\n"
        return self.common_strings.multithread_helper.format(
            func_name=self.func_name, closure_args_string=closure_helper_string.strip()
        )

    def generate_thread_pool_setup(self):
        """
        Generates calls to set up thread pool and map threads to closures.
        Only used while multithreading.
        We iterate over the SPMD grid, and assign each thread to one closure.
        To stay consistent with the interface provided by ThreadManager.exec(),
        we flatten each 3D index in the grid to map to a scalar index in the
        std::vector of closures.
        """
        closure_args_string = ""
        indent = " " * 34
        for inp in self.input_profs:
            if inp.input_type == "tensor":
                closure_args_string += f"{indent}{inp.rank}, dt{inp.idx},\n"
            else:
                closure_args_string += f"{indent}S{inp.idx},\n"
        return self.common_strings.thread_pool_setup.format(
            grid=self.grid,
            strides=self.grid_strides,
            num_threads=prod(self.grid),
            closure_args_string=closure_args_string.strip(),
        )

    def generate_thread_pool_exec(self):
        """Generates the calls to execute each closure with a thread"""
        return self.common_strings.thread_pool_exec

    def generate_cpp_code_headers(self) -> str:
        code_headers = (
            self.common_strings.code_headers + self.common_strings.triton_code_headers
        )
        return code_headers

    def generate_cpp_code_define(self) -> str:
        # Note: return values for multithreaded kernels are not supported when executed by the
        #       Triton/Hexagon driver
        code_define = self.common_strings.code_define.format(
            llvm_func_sign=self.generate_llvm_function_signature(),
            input_wrapper_struct_def="",
            result_struct_def="",
        ) + self.common_strings.triton_code_define.format(
            closure_struct_def=self.generate_closure_definition(),
            multithread_helper=self.generate_multithread_helper(),
        )
        return code_define

    def generate_cpp_code_body(self, file_name, exec_dir) -> str:
        code_body = self.common_strings.triton_code_body.format(
            tensor_definition_str=self.generate_input_declarations(),
            read_from_file_calls=self.generate_tensor_read_from_file_calls(
                file_name, exec_dir
            ),
            write_to_file_calls=self.generate_tensor_write_to_file_calls(
                file_name, exec_dir
            ),
            benchmarking_and_reporting=self.generate_benchmarking_and_reporting(
                self.generate_thread_pool_exec()
            ),
            update_tensor=self.generate_update_tensor_calls(),
            setup_thread_pool=self.generate_thread_pool_setup(),
        )
        return code_body

    def generate_cpp_wrapper(self, file_name, exec_dir):
        """
        Generates skeleton cpp file which launches triton kernel.
        """
        if self.multithreading_enabled:
            code_headers = self.generate_cpp_code_headers()
            code_define = self.generate_cpp_code_define()
            code_body = self.generate_cpp_code_body(file_name, exec_dir)
            return self.common_strings.code_string.format(
                code_headers=code_headers, code_define=code_define, code_body=code_body
            )
        else:
            return super().generate_cpp_wrapper(file_name, exec_dir)


class TritonHexagonLauncher(HexagonLauncherBase):
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
        input_tensor_count = sum(
            [1 for inp in input_profs if inp.input_type == "tensor"]
        )

        local_dir_path = create_timestamped_folder(func_name)
        # 1 - Writting the object code for the kernel to disk
        obj_src_path = os.path.join(local_dir_path, func_name + ".o")
        Path(obj_src_path).write_bytes(kernel_obj_as_bytes)
        print(f"==> kernel obj saved in: {obj_src_path}")

        # Get HexagonOptions at this point for triton pipeline.
        options = HexagonOptions().__dict__

        # Pass lwp related info to HexagonExecutor() for creating shared obj and pulling lwp.json file if enabled.
        hexec = HexagonExecutor(options["enableLWP"])

        # Pass options to the wrapper generator.
        # TritonHexagonWrapperGenerator will assert for lwp and multithreading conflict.
        # HexagonWrapperGenerator will create the call to WriteLWPOutput() if lwp is enabled.
        wrapper_generator = TritonHexagonWrapperGenerator(
            input_profs, iterations, func_name, output_profs, launch_grid, options
        )
        print("==> Wrapper generator correctly instantiated")

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
        print("==> Shared object generated: ", so_path)

        # 4 - Running the shared object located at `so_path`
        results = super().execute_kernel(
            hexec, local_dir_path, func_name, [so_path], wrapper_generator
        )

        # In Triton the inputs arguments to a kernel function, can also contain output pointers
        # and essentially each input can be an output as well, so we
        # overwrite each tensor inputs with the execution results.
        # TODO: This isn't a valid assumption to make when a kernel both returns values
        #       and writes back to the input ptrs. Need to refactor to support both simultaneously.
        output_tensor_count = self.get_output_tensor_path_count(wrapper_generator)
        assert len(results), output_tensor_count
        profs = (
            wrapper_generator.output_profs
            if (len(wrapper_generator.output_profs) > 0)
            else wrapper_generator.input_profs
        )
        res_idx = 0
        for out in profs:
            if out.rank:
                inputs[out.input_id].copy_(results[res_idx])
                res_idx += 1
        return results

    # TODO: Replace generate_input_output_paths() with a method that allows triton kernels
    #       to both write back ptrs and return values directly from the kernel
    @staticmethod
    def get_output_tensor_path_count(wrapper_generator):
        if len(wrapper_generator.output_profs) > 0:
            profs = wrapper_generator.output_profs
        else:
            profs = wrapper_generator.input_profs
        return sum([1 for out in profs if out.rank])


# TODO: Specifically related to Triton driver workflow. Currenlty unused and retained
# for similarly to other backends. Need to investigate usage.
class HexagonUtils:
    def __new__(cls):
        if not hasattr(cls, "instance"):
            cls.instance = super(HexagonUtils, cls).__new__(cls)
        return cls.instance

    def __init__(self):
        self.device_id: tuple = ("hexagon", 0)

    @staticmethod
    def get_device_properties(device):
        return {
            "max_shared_mem": 2**20,
            "multiprocessor_count": None,
            "sm_clock_rate": None,
            "mem_clock_rate": None,
            "mem_bus_width": None,
        }

    @staticmethod
    def load_binary(name, kernel_asm, shared, device):
        return (
            None,  # module
            kernel_asm,  # function
            None,  # n_regs
            None,  # n_spills
            sys.maxsize,  # n_max_threads
        )
