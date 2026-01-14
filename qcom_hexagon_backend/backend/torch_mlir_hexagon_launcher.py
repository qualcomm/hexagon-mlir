# ===- torch_mlir_hexagon_launcher.py ---------------------------------------===
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause.
# For more license information:
#   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
#
# ===------------------------------------------------------------------------===

import os
import shutil
from pathlib import Path
from typing import Optional
import time
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
    parse_return_types,
    profile_torch_mlir_inputs,
)
from triton._C.libtriton import qcom_hexagon_backend, ir  # type: ignore

# This file is part of a small subset of python files that uses some type-annotations
# and it passes type-verification with mypy (a type checker).
# To typecheck this set of files, do:
#   mypy compiler.py hexagon_executor.py hexagon_launcher_base.py torch_mlir_hexagon_launcher.py \
#   triton_hexagon_launcher.py --follow-untyped-imports --check-untyped-defs


class TorchMLIRWrapperGeneratorStrings(WrapperGeneratorStrings):
    def __init__(self):
        super().__init__()

        self.torch_mlir_code_body = """
int main() {{
{tensor_definition_str}
{result_struct_init}
{read_from_file_calls}
{benchmarking_and_reporting}
{update_tensor}
{write_to_file_calls}
{lwp}
return 0;
}}
"""


class TorchMlirHexagonWrapperGenerator(HexagonWrapperGenerator):
    def __init__(self, input_profs, iterations, func_name, output_profs, options: dict):
        super().__init__(
            input_profs,
            iterations,
            func_name,
            output_profs,
            TorchMLIRWrapperGeneratorStrings(),
            options,
        )

    def generate_llvm_function_signature(self):
        """Generate lowered LLVM function definition to be called from CPP launcher"""
        # The first argument in the function signature is pointer to the return value struct
        function_arg_string = "void *,"
        function_arg_string += self.generate_llvm_function_signature_arg_string()
        return self.common_strings.extern_llvm_func_defn.format(
            kernel_name=self.func_name, function_arg_string=function_arg_string
        )

    def generate_llvm_function_call(self):
        """Generates actual function call to lowered LLVM function call"""
        # The function call contains the pointer to the Result as the first parameter
        function_call_descriptor_string = "r,"
        function_call_descriptor_string += self.generate_llvm_function_call_arg_string()
        return self.common_strings.extern_llvm_func_call.format(
            func_name=self.func_name, descriptor_string=function_call_descriptor_string
        )

    def generate_input_wrapper_struct_def(self):
        """Generates template for input wrapper structs"""
        return ""

    def generate_input_wrapper_structs_init(self):
        """Generates initializations for input wrapper structs"""
        return ""

    def generate_cpp_wrapper(self, file_name, exec_dir):
        """
        Generates skeleton cpp file which launches the kernel.
        """
        code_headers = self.common_strings.code_headers

        code_define = self.common_strings.code_define.format(
            llvm_func_sign=self.generate_llvm_function_signature(),
            input_wrapper_struct_def="",
            result_struct_def=self.generate_result_struct(),
        )

        code_body = self.common_strings.torch_mlir_code_body.format(
            tensor_definition_str=self.generate_input_declarations(),
            input_wrapper_structs_init="",
            result_struct_init=self.generate_result_struct_init(),
            read_from_file_calls=self.generate_tensor_read_from_file_calls(
                file_name, exec_dir
            ),
            benchmarking_and_reporting=self.generate_benchmarking_and_reporting(
                self.generate_llvm_function_call()
            ),
            write_to_file_calls=self.generate_tensor_write_to_file_calls(
                file_name, exec_dir
            ),
            update_tensor=self.generate_update_tensor_calls(),
            lwp=self.generate_lwp_call(),
        )

        return self.common_strings.code_string.format(
            code_headers=code_headers, code_define=code_define, code_body=code_body
        )


class TorchMLIRHexagonLauncher(HexagonLauncherBase):
    # Lower mlir module to (potentially several) object codes
    # which are returned in a vector, each as vector<char>
    def mlir_to_obj(self, mlir_mod: ir.module, options: dict) -> list[bytes]:
        # Must cast options to strings before passing to backend
        options = {k: str(v) for k, v in options.items()}
        modules_compiled_as_bytes = qcom_hexagon_backend.translate_linalg_to_obj(
            mlir_mod, options
        )
        return modules_compiled_as_bytes

    # Compile the mlir bytecode from file `mlir_bytecode_path` that lives in `local_dir`,
    # whose principal function to call is `func_name`, using the HexagonExecutor `hexec`.
    # Returns the TorchMlirHexagonWrapperGenerator used, and the collection of paths to
    # the shared libraries generated.
    # TODO: the HexagonExecutor `hexec` is passed because that's still the class that
    #        implements the method generate_shared_object(). Change that!
    def compile_torch_mlir(
        self,
        hexec: HexagonExecutor,
        local_dir: str,
        mlir_bytecode_path: str,
        inputs: list[Tensor],
        func_name: str,
        options: dict,
        iterations: int = 1,
    ) -> tuple[TorchMlirHexagonWrapperGenerator, list[str]]:
        # Context is being initialized through triton source code - python/src/ir.cc
        # This creates a dependency on triton code and
        # will cause issue when we separate torch-mlir workflow from triton.
        # Todo: We need to define the context in hexagon backend.
        context = ir.context()
        if options.get("lowerConstantsInSeparateSharedObjects", False):
            context.disable_multithreading()
        qcom_hexagon_backend.load_dialects(context)
        mlir_mod = qcom_hexagon_backend.parse_mlir_module_from_file(
            mlir_bytecode_path, context
        )
        # Careful, this needs to be done before calling mlir_to_obj()
        result_types = qcom_hexagon_backend.get_return_list(mlir_mod, func_name)
        func_name_with_ciface = "_mlir_ciface_" + func_name

        # MLIR to (potentially several) object files (as strings)
        obj_modules: list[bytes] = self.mlir_to_obj(mlir_mod, options)
        # We now need to link each object code obtained (separately!), to obtain a collection of shared objects (.so)

        # We will populate this collection with the path of the generated shared libraries
        # (there will be only 1 .so if options["lowerConstantsInSeparateSharedObjects"] is False,
        # and multiple ones otherwise)
        paths_to_shared_libs_generated: list[str] = []
        # Every shared object we will build has no dependencies (hence the default to []) except for the principal module
        lib_dependencies_paths = []

        nb_modules_compiled = len(obj_modules)
        print(
            "We have",
            nb_modules_compiled,
            "object files obtained from the MLIR->obj compilation",
            (
                "that we now need to link independently into their own shared object (.so)."
                if nb_modules_compiled > 1
                else ""
            ),
        )
        # Dealing with each object code that has been generated by mlir_to_obj()
        # Note: we reverse obj_modules to start with the "constants-only" modules and to finish with the principal module
        # since the principal one will need some -l annotations
        for i, kernel_obj_as_bytes in enumerate(reversed(obj_modules)):
            # Convert the index since we've had to reverse obj_modules to finish with the principal module
            i = nb_modules_compiled - 1 - i
            print("------ Starting work on compiled module number", i + 1, "------")
            # 1 - Writting the object code for the kernel to disk
            # The principal module containg the code will keep the name of the function
            # but the modules containg only constants will be called with a suffix ([...]-consts-1.o, [...]-consts-2.o, etc)
            filename_obj = (
                func_name_with_ciface
                if (i == 0)
                else func_name_with_ciface + "-consts-" + str(i)
            )
            obj_src_path = os.path.join(local_dir, filename_obj + ".o")
            Path(obj_src_path).write_bytes(kernel_obj_as_bytes)
            print(f"==> kernel obj saved in: {obj_src_path}")

            cpp_wrapper_path = None
            # 2 - If it's the principal module only, deal with the wrapper code that we need to generate and dump
            if i == 0:
                # Creating the wrapper generator for the principal module
                return_types = parse_return_types(result_types)
                input_profs = profile_torch_mlir_inputs(inputs)

                # Pass options to the wrapper generator.
                # HexagonWrapperGenerator will create the call to WriteLWPOutput() if lwp is enabled.
                wrapper_generator = TorchMlirHexagonWrapperGenerator(
                    input_profs,
                    iterations,
                    func_name_with_ciface,
                    return_types,
                    options,
                )
                print("==> Wrapper generator correctly instanciated")

                # The directory path used for execution is given by the executor, and if it's the empty string (meaning running on device),
                # then the execution directory path will be set to the local directory path
                exec_dir = hexec.get_Executable_Path()
                # If no executable path specified, using local directory.
                if not exec_dir:
                    exec_dir = local_dir

                # Generating and writing the wrapper to disk
                cpp_wrapper_path = self.generate_and_dump_wrapper(
                    wrapper_generator, local_dir, func_name_with_ciface, exec_dir
                )
                print(
                    f"==> Generated a cpp wrapper to act as kernel starter: {cpp_wrapper_path}"
                )

                # The main module has for dependencies all the .so that we just built up to this point (from nb_modules_compiled-1 to 1)
                lib_dependencies_paths = paths_to_shared_libs_generated

            # 3 - Generate the corresponding shared object for the current object file
            so_path = hexec.generate_shared_object(
                cpp_wrapper_path, obj_src_path, lib_dependencies_paths
            )  # cpp_wrapper_path is None if that's not the principal module
            print("==> Shared object generated: ", so_path)
            # Add it to the collection of paths to the shared libraries that have been generated
            paths_to_shared_libs_generated.append(so_path)
        # We reverse the paths_to_shared_libs_generated to have them back in the normal order: principal first, then constants-only ones
        return (wrapper_generator, list(reversed(paths_to_shared_libs_generated)))

    # Toplevel function that compiles the mlir and execute the result.
    # Executes either on device or on simulator depending on the env var RUN_ON_SIM.
    def run_torch_mlir(
        self,
        mlir_bytecode_path: str,
        inputs: list[Tensor],
        func_name: str,
        base_dir_for_artifacts: Optional[str] = None,
        iterations: int = 1,
        options: dict = None,
        enable_etm=False,
    ) -> list[Tensor]:
        if options is None:
            options = HexagonOptions().__dict__

        # Pass lwp related info to HexagonExecutor() for creating shared obj and pulling lwp.json file if enabled.
        hexec = HexagonExecutor(options["enableLWP"], enable_etm)
        local_dir_path = create_timestamped_folder(func_name, base_dir_for_artifacts)

        filename = os.path.basename(mlir_bytecode_path)
        destination_path = os.path.join(local_dir_path, filename)
        # Copy the initial MLIR code to the timestamped folder
        shutil.copy(mlir_bytecode_path, destination_path)

        start_time = time.time()
        # Compile the mlir bytecode from file `mlir_bytecode_path`
        (wrapper_generator, paths_to_shared_libs_generated) = self.compile_torch_mlir(
            hexec,
            local_dir_path,
            mlir_bytecode_path,
            inputs,
            func_name,
            options,
            iterations,
        )
        end_time = time.time()
        print(
            f"Compilation from initial MLIR to .so took {end_time - start_time:.4f} seconds",
            flush=True,
        )

        func_name_with_ciface = "_mlir_ciface_" + func_name
        # Execute the kernel using the HexagonExecutor `hexec`
        results = self.execute_kernel(
            hexec,
            local_dir_path,
            func_name_with_ciface,
            paths_to_shared_libs_generated,
            wrapper_generator,
        )
        return results

    @staticmethod
    def get_output_tensor_path_count(wrapper_generator: HexagonWrapperGenerator):
        """Returns number of output paths by checking rank (dim)"""
        return sum([1 for out in wrapper_generator.output_profs if out.rank])
