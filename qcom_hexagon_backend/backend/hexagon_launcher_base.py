# ===- hexagon_launcher_base.py ---------------------------------------------===
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause.
# For more license information:
#   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
#
# ===------------------------------------------------------------------------===

import hashlib
import logging
import subprocess
from datetime import datetime
from pathlib import Path

try:
    from torch import Tensor  # For type annotations
except ImportError:
    Tensor = object  # type: ignore
from triton.backends.qcom_hexagon_backend import knobs
from triton.backends.qcom_hexagon_backend.hexagon_executor import HexagonExecutor

# This file is part of a small subset of python files that uses some type-annotations
# and it passes type-verification with mypy (a type checker).
# To typecheck this set of files, do:
# mypy compiler.py hexagon_executor.py hexagon_launcher_base.py \
#   torch_mlir_hexagon_launcher.py stablehlo_hexagon_launcher.py \
#   triton_hexagon_launcher.py --follow-untyped-imports --check-untyped-defs

log = logging.getLogger(__name__)


def _fs_name(func_name: str) -> str:
    """Return a filename-safe alias (≤ 150 chars) for func_name.

    Used only for filesystem artifacts (.o, .so, _wrapper.cpp, build dir).
    The full func_name is still used for C symbol linkage and runtime dispatch.
    Matches the upstream Triton [:150] guard in triton/compiler/compiler.py.
    """
    if len(func_name) <= 150:
        return func_name
    suffix = hashlib.md5(func_name.encode()).hexdigest()[:16]
    return func_name[:100] + "_" + suffix


class WrapperGeneratorStrings:
    def __init__(self):
        self.input_tensor_name = "T"
        self.memrefdesc_name = "dt"
        self.scalar_name = "S"
        self.output_tensor_name = "O"

        self.input_wrapper_name = "wrDt"
        self.input_wrapper_ptr_name = "pDt"

        self.tensor_init_template = """
    Tensor<{tensor_ctype}, {tensor_rank}> {input_tensor_prefix}{i}({{{sizes}, {strides}, 128}}, MemType::HEAP);\nMemRefDescriptor<{tensor_ctype}, {tensor_rank}> *{memrefdesc_name}{i} = {input_tensor_prefix}{i}.toMemRefDesc(); \n"""
        self.tensor_init_template_rank_0 = """
    Tensor<{tensor_ctype}, 0> {input_tensor_prefix}{i};\nMemRefDescriptor<{tensor_ctype}, 0> *{memrefdesc_name}{i} = {input_tensor_prefix}{i}.toMemRefDesc(); \n"""
        self.scalar_init_template = (
            """{scalar_ctype} {scalar_prefix}{i} = {scalar_val}; \n"""
        )

        self.result_tensor_init_template = """
    Tensor<{tensor_ctype}, {tensor_rank}> {output_tensor_prefix}{i}(r->r{i}, MemType::HEAP, 128);\n"""
        self.extern_llvm_func_no_return_args = (
            """int64_t, MemRefDescriptor<{tensor_ctype}, {tensor_rank}> *, """
        )
        self.extern_llvm_func_with_return_args = (
            """ MemRefDescriptor<{tensor_ctype}, {tensor_rank}> *, """
        )
        self.result_struct_init = """FuncResult *r = new FuncResult;\n"""
        self.result_struct_def = """
struct FuncResult
{{{result_fields}
}};
"""
        self.extern_llvm_func_args_tensor = """ {tensor_pointer_ctype} *, """
        self.extern_llvm_func_args_scalar = """{scalar_ctype}, """
        self.extern_llvm_func_defn = (
            """extern "C" void {kernel_name}({function_arg_string});"""
        )
        self.extern_llvm_func_call = """{func_name}({descriptor_string});\n"""
        self.load_from_file_string = """{input_tensor_prefix}{idx}.load_from_file("{path}/{file_name}_t{idx}.raw");\n"""
        self.load_from_file_chunked_string = """{input_tensor_prefix}{idx}.load_from_file_chunked("{path}/{file_name}_t{idx}.raw");\n"""
        self.dump_to_file_string = """{output_tensor_prefix}{idx}.dump_to_file("{path}/{file_name}_o{idx}.raw");\n"""
        self.call_lwp = """WriteLWPOutput("{path}/{fname}.json");\n"""

        self.func_call_and_benchmarking = """
uint64_t avg_time_us = benchmark_time_us({iterations}, [&]() {{
    {function_call}
}});
TestReport tr("{func_name}", avg_time_us, "us", Result::Pass, "{save_path}");
tr.save();
"""

        # Codegen string for the Headers in the generated CPP file.
        self.code_headers = """
#define FARF_ALWAYS 1
#include <HAP_farf.h>
#include <limits>
#include <stdio.h>
#include <stdlib.h>
#include <cstdint>
#include "common.h"
#include "tensor.h"
#include "hexagon_benchmark.h"
#include "test_report.h"
#include "CRunnerUtils.cpp"
#include "debug.h"
#include "prof_utils.h"
"""

        # Codegen string for the data structures and functions defined before the main body.
        self.code_define = """
{llvm_func_sign}
{input_wrapper_struct_def}
{result_struct_def}
"""

        # Codegen string for the contents of main body.
        self.code_body = """
int main() {{
{tensor_definition_str}
{input_wrapper_structs_init}
{result_struct_init}
{read_from_file_calls}
{benchmarking_and_reporting}
{update_tensor}
{write_to_file_calls}
{lwp}
return 0;
}}
"""

        # Codegen string for the entire CPP file.
        self.code_string = """
{code_headers}
{code_define}
{code_body}
"""


class HexagonWrapperGenerator:
    def __init__(
        self, input_profs, iterations, func_name, output_profs, common_strings, options
    ):
        """
        Initialize the HexagonWrapperGenerator instance.

        Parameters:
            input_profs (list): A list of input tensors and scalars for the kernel.
            func_name (str): The name of the LLVM IR function to be defined/called.
            output_profs (list): A list of output profiles.
            common_strings(WrapperGeneratorStrings): Set of strings with placeholder to be updated.
        """
        self.input_profs = input_profs
        self.iterations = iterations
        self.func_name = func_name
        self.output_profs = output_profs
        self.common_strings = common_strings
        self.enable_lwp = options["enableLWP"]
        self.chunked_file_loads = options.get("chunkedFileLoads", False)

    def generate_input_declarations(self):
        """
        Generate declaration for each input
        for torch tensors, write them in Tensor class form,
        for scalars, declare and intialize a variable with scalar input's value.
        """
        formatted_tensor_string = ""

        for inp in self.input_profs:
            if inp.input_type == "tensor":
                if inp.rank == 0:
                    tensor_declaration = (
                        self.common_strings.tensor_init_template_rank_0.format(
                            tensor_ctype=inp.dtype,
                            i=inp.idx,
                            memrefdesc_name=self.common_strings.memrefdesc_name,
                            input_tensor_prefix=self.common_strings.input_tensor_name,
                        ).lstrip()
                    )
                else:
                    tensor_declaration = (
                        self.common_strings.tensor_init_template.format(
                            tensor_ctype=inp.dtype,
                            tensor_rank=inp.rank,
                            i=inp.idx,
                            sizes=inp.shape[0],
                            strides=inp.shape[1],
                            memrefdesc_name=self.common_strings.memrefdesc_name,
                            input_tensor_prefix=self.common_strings.input_tensor_name,
                        ).lstrip()
                    )
                formatted_tensor_string += tensor_declaration
            else:
                scalar_declaration = self.common_strings.scalar_init_template.format(
                    i=inp.idx,
                    scalar_ctype=inp.dtype,
                    scalar_val=inp.value,
                    scalar_prefix=self.common_strings.scalar_name,
                ).lstrip()
                formatted_tensor_string += scalar_declaration
        return formatted_tensor_string

    def generate_llvm_function_signature_arg_string(self):
        """Generate common function signature args"""
        function_arg_string = ""
        for inp in self.input_profs:
            if inp.input_type == "tensor":
                if len(self.output_profs) > 0:
                    extern_args = (
                        self.common_strings.extern_llvm_func_with_return_args.format(
                            tensor_ctype=inp.dtype, tensor_rank=inp.rank
                        )
                    )
                else:
                    extern_args = (
                        self.common_strings.extern_llvm_func_no_return_args.format(
                            tensor_ctype=inp.dtype, tensor_rank=inp.rank
                        )
                    )
            else:
                extern_args = self.common_strings.extern_llvm_func_args_scalar.format(
                    scalar_ctype=inp.dtype
                )
            function_arg_string += extern_args
        return function_arg_string[:-2]

    def generate_llvm_function_signature(self):
        """Generate lowered LLVM function definition to be called from CPP launcher"""
        raise NotImplementedError("LLVM function signature generation not implemented.")

    def generate_llvm_function_call_arg_string(self):
        """Generate common function call args"""
        function_call_descriptor_string = ""
        for inp in self.input_profs:
            if inp.input_type == "tensor":
                if len(self.output_profs) == 0:
                    function_call_descriptor_string += f"{inp.rank}, "
                function_call_descriptor_string += (
                    f"{self.common_strings.memrefdesc_name}{inp.idx}, "
                )
            else:
                function_call_descriptor_string += (
                    f"{self.common_strings.scalar_name}{inp.idx}, "
                )
        return function_call_descriptor_string[:-2]

    def generate_llvm_function_call(self):
        """Generates actual function call to lowered LLVM function call"""
        raise NotImplementedError("LLVM function call generation not implemented.")

    def generate_input_wrapper_struct_def(self):
        """Generates template for input wrapper structs"""
        raise NotImplementedError("Unimplemented- requires definition by frontend.")

    def generate_input_wrapper_structs_init(self):
        """Generates initializations for input wrapper structs"""
        raise NotImplementedError("Unimplemented- requires definition by frontend.")

    def generate_benchmarking_and_reporting(self, function_call, exec_dir):
        return self.common_strings.func_call_and_benchmarking.format(
            iterations=self.iterations,
            function_call=function_call,
            func_name=self.func_name,
            save_path=f"{exec_dir}/perf.txt",
        )

    def generate_update_tensor_calls(self):
        """Generates a function call to update the tensor with the result of the output"""

        formatted_result_string = ""

        # TODO: Currently skipping returning scalar values, since writing mechanism is not in place.
        # To be updated later with the usecase which returns scalar values.
        for idx, ret in enumerate(self.output_profs):
            if ret.rank:
                tensor_declaration = (
                    self.common_strings.result_tensor_init_template.format(
                        output_tensor_prefix=self.common_strings.output_tensor_name,
                        tensor_ctype=ret.dtype,
                        tensor_rank=ret.rank,
                        i=idx,
                    ).lstrip()
                )
                formatted_result_string += tensor_declaration
        return formatted_result_string

    def generate_result_struct(self):
        """Generates the Result string"""
        if len(self.output_profs) > 0:
            result_fields_string = ""
            for idx, res in enumerate(self.output_profs):
                if res.rank:
                    result_fields_string += (
                        f"\n\tMemRefDescriptor<{res.dtype},{res.rank}> r{idx};"
                    )
                else:
                    result_fields_string += f"{res.dtype} r{idx};\n"
            return self.common_strings.result_struct_def.format(
                result_fields=result_fields_string
            )
        return ""

    def generate_result_struct_init(self):
        if len(self.output_profs) > 0:
            return self.common_strings.result_struct_init
        return ""

    def generate_tensor_read_from_file_calls(self, file_name, exec_dir):
        """Generates calls to read tensors from file"""
        read_from_file_string = ""

        for inp in self.input_profs:
            if inp.input_type == "tensor":
                tensor_idx = inp.idx
                load_string = (
                    self.common_strings.load_from_file_chunked_string
                    if self.chunked_file_loads
                    else self.common_strings.load_from_file_string
                )
                write_call = load_string.format(
                    input_tensor_prefix=self.common_strings.input_tensor_name,
                    idx=tensor_idx,
                    path=exec_dir,
                    file_name=file_name,
                )
                read_from_file_string += write_call
        return read_from_file_string

    def generate_lwp_call(self):
        if self.enable_lwp:
            return self.common_strings.call_lwp.format(
                path="/data/local/tmp", fname="lwp"
            )
        return ""

    def generate_tensor_write_to_file_calls(self, fname, exec_dir):
        """Generates calls to dump tensors to file"""
        if len(self.output_profs) > 0:
            profs = self.output_profs
            prefix = self.common_strings.output_tensor_name
        else:
            profs = self.input_profs
            prefix = self.common_strings.input_tensor_name

        write_to_file_string = ""
        for out in profs:
            # TODO: add scalar/bool support
            if out.rank:
                write_to_file_string += self.common_strings.dump_to_file_string.format(
                    output_tensor_prefix=prefix,
                    idx=out.idx,
                    path=exec_dir,
                    file_name=fname,
                )
        return write_to_file_string

    def generate_cpp_code_headers(self) -> str:
        code_headers = self.common_strings.code_headers
        return code_headers

    def generate_cpp_code_define(self) -> str:
        code_define = self.common_strings.code_define.format(
            llvm_func_sign=self.generate_llvm_function_signature(),
            input_wrapper_struct_def=self.generate_input_wrapper_struct_def(),
            result_struct_def=self.generate_result_struct(),
        )
        return code_define

    def generate_cpp_code_body(self, file_name, exec_dir) -> str:
        code_body = self.common_strings.code_body.format(
            tensor_definition_str=self.generate_input_declarations(),
            input_wrapper_structs_init=self.generate_input_wrapper_structs_init(),
            result_struct_init=self.generate_result_struct_init(),
            read_from_file_calls=self.generate_tensor_read_from_file_calls(
                file_name, exec_dir
            ),
            benchmarking_and_reporting=self.generate_benchmarking_and_reporting(
                self.generate_llvm_function_call(), exec_dir
            ),
            update_tensor=self.generate_update_tensor_calls(),
            write_to_file_calls=self.generate_tensor_write_to_file_calls(
                file_name, exec_dir
            ),
            lwp=self.generate_lwp_call(),
        )
        return code_body

    def generate_cpp_wrapper(self, file_name, exec_dir) -> str:
        """
        Generates skeleton cpp file which calls a function.
        """
        code_headers = HexagonWrapperGenerator.generate_cpp_code_headers(self)
        code_define = HexagonWrapperGenerator.generate_cpp_code_define(self)
        code_body = HexagonWrapperGenerator.generate_cpp_code_body(
            self, file_name, exec_dir
        )

        return self.common_strings.code_string.format(
            code_headers=code_headers, code_define=code_define, code_body=code_body
        )


class MLIRWrapperGeneratorStrings(WrapperGeneratorStrings):
    def __init__(self):
        super().__init__()

        self.mlir_code_body = """
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


class MLIRHexagonWrapperGenerator(HexagonWrapperGenerator):
    def __init__(self, input_profs, iterations, func_name, output_profs, options: dict):
        super().__init__(
            input_profs,
            iterations,
            func_name,
            output_profs,
            MLIRWrapperGeneratorStrings(),
            options,
        )

    def generate_llvm_function_signature(self):
        """Generate lowered LLVM function definition to be called from CPP launcher."""
        function_arg_string = "void *,"
        function_arg_string += self.generate_llvm_function_signature_arg_string()
        return self.common_strings.extern_llvm_func_defn.format(
            kernel_name=self.func_name, function_arg_string=function_arg_string
        )

    def generate_llvm_function_call(self):
        """Generate call to the lowered LLVM function."""
        function_call_descriptor_string = "r,"
        function_call_descriptor_string += self.generate_llvm_function_call_arg_string()
        return self.common_strings.extern_llvm_func_call.format(
            func_name=self.func_name, descriptor_string=function_call_descriptor_string
        )

    def generate_input_wrapper_struct_def(self):
        return ""

    def generate_input_wrapper_structs_init(self):
        return ""

    def generate_cpp_wrapper(self, file_name, exec_dir):
        code_headers = self.common_strings.code_headers
        code_define = self.common_strings.code_define.format(
            llvm_func_sign=self.generate_llvm_function_signature(),
            input_wrapper_struct_def="",
            result_struct_def=self.generate_result_struct(),
        )
        code_body = self.common_strings.mlir_code_body.format(
            tensor_definition_str=self.generate_input_declarations(),
            result_struct_init=self.generate_result_struct_init(),
            read_from_file_calls=self.generate_tensor_read_from_file_calls(
                file_name, exec_dir
            ),
            benchmarking_and_reporting=self.generate_benchmarking_and_reporting(
                self.generate_llvm_function_call(), exec_dir
            ),
            update_tensor=self.generate_update_tensor_calls(),
            write_to_file_calls=self.generate_tensor_write_to_file_calls(
                file_name, exec_dir
            ),
            lwp=self.generate_lwp_call(),
        )
        return self.common_strings.code_string.format(
            code_headers=code_headers, code_define=code_define, code_body=code_body
        )


def create_timestamped_folder(
    model_name: str,
    base_dir: str | Path | None = None,
) -> Path:
    """Create a timestamp-based directory for a launcher invocation.

    The directory is created below ``base_dir`` when provided; otherwise, it
    is created below the configured Hexagon MLIR dump directory.  It is named
    using a filesystem-safe version of ``model_name`` followed by the current
    local timestamp.  Parent directories are created as needed.
    The directory is never reused: if the generated name already exists, the
    creation fails rather than silently sharing the existing directory.

    Args:
        model_name: Name of the model or kernel used to form the directory
            name.  Long names are shortened by :func:`_fs_name`.
        base_dir: Optional directory in which to create the folder.  If not
            provided, the configured Hexagon MLIR dump directory is used.

    Returns:
        The path to the newly created directory.  Use ``Path.name`` to obtain
        the directory name when it is needed as a run identifier.

    Raises:
        FileExistsError: If the generated timestamped directory already
            exists, for example because of a timestamp collision.
        OSError: If the directory or one of its parents cannot be created.
    """
    dump_dir = knobs.hexagon_mlir.dump_dir
    if base_dir is not None:
        dump_dir = Path(base_dir)

    name = _fs_name(model_name)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    full_path = dump_dir / f"{name}-{timestamp}"
    full_path.mkdir(parents=True, exist_ok=False)

    log.debug("Created timestamped folder: %s", full_path.name)
    return full_path


def create_hashed_folder(
    model_name: str,
    content_hash: str,
    base_dir: str | Path | None = None,
) -> tuple[Path, bool]:
    """Create or reuse a content-addressed directory for build artifacts.

    The directory is created below ``base_dir`` when provided; otherwise, it
    is created below the configured Hexagon MLIR dump directory.  It is named
    using a filesystem-safe version of ``model_name`` followed by
    ``content_hash``.  Repeated calls with the same arguments therefore refer
    to the same directory, allowing cached artifacts to be reused.
    ``content_hash`` is supplied by the caller and is not computed by this
    function.

    Args:
        model_name: Name of the model or kernel used to form the directory
            name.  Long names are shortened by :func:`_fs_name`.
        content_hash: Hash identifying the contents or build configuration
            associated with the artifacts in the directory.
        base_dir: Optional directory in which to create the folder.  If not
            provided, the configured Hexagon MLIR dump directory is used.

    Returns:
        A tuple containing the directory and a boolean indicating whether the
        directory already existed before this call.  ``False`` indicates that
        this call created the directory.

    Raises:
        OSError: If the directory or one of its parents cannot be created.
    """
    dump_dir = knobs.hexagon_mlir.dump_dir
    if base_dir is not None:
        dump_dir = Path(base_dir)

    name = _fs_name(model_name)
    full_path = dump_dir / f"{name}-{content_hash}"
    cache_hit = full_path.is_dir()
    full_path.mkdir(parents=True, exist_ok=True)

    log.debug(
        "%s hashed folder: %s",
        "Reusing" if cache_hit else "Created",
        full_path.name,
    )

    return full_path, cache_hit


class HexagonLauncherBase:
    @staticmethod
    def rename_kernel_main_symbol(obj_path: str | Path, hexec: HexagonExecutor) -> None:
        """Avoid collision between an MLIR `main` symbol and the C++ wrapper."""
        objcopy = (
            Path(hexec.config.HEX_TOOLS["hexagon-clang++"]).parent
            / "hexagon-llvm-objcopy"
        )
        if not objcopy.exists():
            return
        subprocess.run(
            [
                str(objcopy),
                "--redefine-sym",
                "main=mlir_kernel_main",
                str(obj_path),
                str(obj_path),
            ],
            check=True,
        )
        print("==> renamed kernel symbol main -> mlir_kernel_main", flush=True)

    def generate_input_output_paths(
        self, directory: Path, file_name: str, wrapper: HexagonWrapperGenerator
    ) -> tuple[list[Path], list[Path]]:
        input_paths = []
        output_paths = []

        for inp in wrapper.input_profs:
            if inp.input_type == "tensor":
                i = inp.idx
                data = inp.value
                input_path = directory / f"{file_name}_t{i}.raw"
                with open(input_path, "wb") as file:
                    file.write(data.numpy().tobytes())
                input_paths.append(input_path)

        # The output path count is provided by the frontend. There are
        # several cases (writing back ptrs, return values, or both)
        # that must be handled accordingly
        output_tensor_path_count = self.get_output_tensor_path_count(wrapper)
        for i in range(output_tensor_path_count):
            output_path = directory / f"{file_name}_o{i}.raw"
            output_paths.append(output_path)

        return input_paths, output_paths

    @staticmethod
    def get_output_tensor_path_count(wrapper_generator: HexagonWrapperGenerator):
        raise NotImplementedError("Unimplemented- requires definition by frontend.")

    # Ask the HexagonWrapperGenerator `wrapper_generator` to generate the wrapper for `file_name`, which is then written to disk,
    # knowing that the execution folder (either on device or local) will be `exec_dir`
    def generate_and_dump_wrapper(
        self,
        wrapper_generator: HexagonWrapperGenerator,
        local_dir: Path,
        file_name: str,
        exec_dir: str,
    ) -> Path:
        """Generates and dumps CPP wrapper file using a given wrapper generator"""
        cpp_wrapper_path = local_dir / f"{file_name}_wrapper.cpp"
        with open(cpp_wrapper_path, "w") as file:
            cpp_wrapper = wrapper_generator.generate_cpp_wrapper(file_name, exec_dir)
            file.write(cpp_wrapper)
        return cpp_wrapper_path

    def execute_kernel(
        self,
        hexec: HexagonExecutor,
        local_dir: Path,
        filename_without_ext: str,
        paths_to_shared_libs_generated: list[str],
        wrapper_generator: HexagonWrapperGenerator,
    ) -> list[Tensor]:
        """
        Generates the input and output paths and runs the kernel
        """
        # Generate input/output paths
        input_tensor_paths, output_tensor_paths = self.generate_input_output_paths(
            local_dir, filename_without_ext, wrapper_generator
        )
        # Run the kernel
        results = hexec.run(
            paths_to_shared_libs_generated,
            input_tensor_paths,
            output_tensor_paths,
            generatePerf=True,
        )
        return results
