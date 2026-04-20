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
import warnings
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
            {vtcm_setup_str}
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
{cleanup_str}
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
        # NOTE: For Triton SPMD, grid>1 still needs closure/thread-pool scaffolding
        # to iterate over pids. Actual execution can be threaded (tm.exec) or
        # serialized (tm.exec_serial) depending on enableThreadedDispatch.
        self.multithreading_enabled = prod(self.grid) > 1
        # enableThreadedDispatch controls tm.exec() (real qurt threads) vs
        # tm.exec_serial() (serial loop over pids). For new code, set
        # enableThreadedDispatch directly; parse_options() auto-sets it when
        # scratch > 0.
        #
        # enableMultiThreading also enables threaded dispatch for backward
        # compatibility: before enableThreadedDispatch existed, enableMultiThreading
        # was the only flag that controlled both FormVirtualThreadsPass (compiler-side
        # async DMA pipelining) and tm.exec() (launcher-side thread dispatch).
        # Removing this OR would silently change dispatch behavior for existing
        # callers of enableMultiThreading=True (e.g. test_double_buffer_gelu.py).
        self.enable_threaded_spmd = bool(options["enableThreadedDispatch"]) or bool(
            options["enableMultiThreading"]
        )

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
        self.scratch = options["scratch"]
        self._init_vtcm_resources()

    def _init_vtcm_resources(self):
        """Compute per-instance VTCM scratch sizing.

        Only computes vtcm_per_instance_bytes (the aligned per-thread
        slice).  Total VTCM required is computed at code-gen time as
        vtcm_per_instance_bytes * prod(grid), since an oversubscription
        guard in _exec_kernel guarantees prod(grid) <= num_threads.
        Subclasses may override to change the alignment or add
        additional validation.
        """
        _vtcm_alignment = 2048
        if self.scratch == 0:
            self.vtcm_per_instance_bytes = 0
        else:
            self.vtcm_per_instance_bytes = self.scratch & ~(_vtcm_alignment - 1)
            if self.vtcm_per_instance_bytes <= 0:
                raise ValueError(
                    f"Per-instance VTCM scratch is zero after alignment "
                    f"(scratch={self.scratch}, alignment={_vtcm_alignment}). "
                    f"Increase scratch size."
                )

    def _vtcm_signature_arg(self):
        if not self.scratch:
            return ""
        # When output_profs is empty the kernel has no return-value profiling
        # and the VTCM buffer is passed directly as a MemRefDescriptor.
        # When output_profs is non-empty the calling convention wraps inputs in
        # FuncInput<> structs and passes a void* to the wrapper struct instead.
        if len(self.output_profs) == 0:
            return ", int8_t *, int8_t *, int64_t, int64_t, int64_t"
        return ", void *"

    def _vtcm_call_arg(self):
        if not self.scratch:
            return ""
        # Matches the asymmetry in _vtcm_signature_arg above.
        if len(self.output_profs) == 0:
            return (
                ", vtcmScratchDesc.allocated"
                ", vtcmScratchDesc.aligned"
                ", vtcmScratchDesc.offset"
                ", vtcmScratchDesc.sizes[0]"
                ", vtcmScratchDesc.strides[0]"
            )
        return ", pVtcmScratch"

    def generate_llvm_function_signature(self):
        """Generate LLVM function definition for Triton"""
        if len(self.output_profs) == 0:
            function_arg_string = self.generate_llvm_function_signature_arg_string()
        else:
            function_arg_string = self.generate_llvm_direct_returns_function_signature()
        # Represents the num_programs(x,y,z) and program_id(x,y,z) arguments
        function_arg_string += self._vtcm_signature_arg()
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
        function_call_descriptor_string += self._vtcm_call_arg()
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

    def generate_input_declarations(self):
        formatted_tensor_string = super().generate_input_declarations()
        if self.scratch:
            grid_size = prod(self.grid)
            total_scratch = self.vtcm_per_instance_bytes * grid_size
            formatted_tensor_string += f"""constexpr int64_t vtcmPerInstanceBytes = {self.vtcm_per_instance_bytes};
constexpr int64_t totalRequiredScratch = {total_scratch};
compute_res_attr_t vtcmRes;
HAP_compute_res_attr_init(&vtcmRes);
HAP_compute_res_attr_set_vtcm_param(&vtcmRes, totalRequiredScratch, ENFORCE_SINGLE_PAGE_VTCM_ALLOC);
uint32_t vtcmContextID = HAP_compute_res_acquire(&vtcmRes, HAP_REQUEST_TIMEOUT_US);
if (vtcmContextID == 0) {{
  FARF(ERROR, "Failed to acquire VTCM scratch (bytes=%lld)", (long long)totalRequiredScratch);
  return -1;
}}
int8_t *vtcmBase = (int8_t *)HAP_compute_res_attr_get_vtcm_ptr(&vtcmRes);
if (vtcmBase == nullptr) {{
  FARF(ERROR, "VTCM acquire returned null pointer");
  HAP_compute_res_release(vtcmContextID);
  return -1;
}}
// For the non-MT direct-call path (flat_pid == 0), construct a descriptor
// covering the first (and only) instance's slice.
MemRefDescriptor<int8_t, 1> vtcmScratchDesc = {{vtcmBase, vtcmBase, 0, {{vtcmPerInstanceBytes}}, {{1}}}};
"""
        return formatted_tensor_string

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
            if self.scratch:
                formatted_tensor_string += f"""FuncInput<int8_t, 1> wrVtcmScratch = {{1, &vtcmScratchDesc}};
FuncInput<int8_t, 1> *pVtcmScratch = &wrVtcmScratch;
"""
        return formatted_tensor_string

    def generate_cleanup(self):
        """Generates cleanup code for resources allocated in the wrapper."""
        cleanup_str = ""
        if self.scratch:
            cleanup_str += "HAP_compute_res_release(vtcmContextID);\n"
        return cleanup_str

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
        if self.scratch:
            if len(self.output_profs) == 0:
                closure_fields_string += (
                    # Store by value so each closure owns its own per-instance
                    # slice descriptor — storing a pointer would alias all
                    # closures to the same descriptor.
                    f"{indent}MemRefDescriptor<int8_t, 1> vtcmScratchMemRef;\n"
                )
            else:
                closure_fields_string += f"{indent}void *vtcmScratch;\n"

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
        if self.scratch:
            if len(self.output_profs) == 0:
                closure_helper_string += (
                    f"{indent}closure->vtcmScratchMemRef.allocated,\n"
                )
                closure_helper_string += (
                    f"{indent}closure->vtcmScratchMemRef.aligned,\n"
                )
                closure_helper_string += f"{indent}closure->vtcmScratchMemRef.offset,\n"
                closure_helper_string += (
                    f"{indent}closure->vtcmScratchMemRef.sizes[0],\n"
                )
                closure_helper_string += (
                    f"{indent}closure->vtcmScratchMemRef.strides[0],\n"
                )
            else:
                closure_helper_string += f"{indent}closure->vtcmScratch,\n"
        return self.common_strings.multithread_helper.format(
            func_name=self.func_name, closure_args_string=closure_helper_string.strip()
        )

    def generate_vtcm_closure_setup(self):
        if not self.scratch:
            return ""
        if len(self.output_profs) == 0:
            return """closures[flat_pid].vtcmScratchMemRef = {vtcmBase,
                                                    vtcmBase + flat_pid * vtcmPerInstanceBytes,
                                                    0,
                                                    {vtcmPerInstanceBytes},
                                                    {1}};"""
        return "closures[flat_pid].vtcmScratch = vtcmBase + flat_pid * vtcmPerInstanceBytes;"

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
        if self.scratch:
            if len(self.output_profs) == 0:
                # Placeholder values keep aggregate-initializer field ordering
                # intact; per-instance descriptor is assigned right after.
                closure_args_string += f"{indent}{{}},\n"
            else:
                closure_args_string += f"{indent}nullptr,\n"
        return self.common_strings.thread_pool_setup.format(
            grid=self.grid,
            strides=self.grid_strides,
            num_threads=prod(self.grid),
            closure_args_string=closure_args_string.strip(),
            vtcm_setup_str=self.generate_vtcm_closure_setup(),
        )

    def generate_thread_pool_exec(self):
        """Generates the calls to execute each closure with a thread"""
        if self.enable_threaded_spmd:
            return self.common_strings.thread_pool_exec
        return "tm.exec_serial(multithread_helper, closures.data());"

    def generate_cpp_code_headers(self) -> str:
        if self.multithreading_enabled:
            code_headers = (
                self.common_strings.code_headers
                + self.common_strings.triton_code_headers
            )
        else:
            code_headers = self.common_strings.code_headers
        return code_headers

    def generate_cpp_code_define(self) -> str:
        if self.multithreading_enabled:
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
        else:
            code_define = self.common_strings.code_define.format(
                llvm_func_sign=self.generate_llvm_function_signature(),
                input_wrapper_struct_def=self.generate_input_wrapper_struct_def(),
                result_struct_def=self.generate_result_struct(),
            )
        return code_define

    def generate_cpp_code_body(self, file_name, exec_dir) -> str:
        if self.multithreading_enabled:
            func_call = self.generate_thread_pool_exec()
            setup_thread_pool = self.generate_thread_pool_setup()
        else:
            func_call = self.generate_llvm_function_call()
            setup_thread_pool = ""

        code_body = self.common_strings.triton_code_body.format(
            tensor_definition_str=self.generate_input_declarations()
            + self.generate_input_wrapper_structs_init(),
            read_from_file_calls=self.generate_tensor_read_from_file_calls(
                file_name, exec_dir
            ),
            write_to_file_calls=self.generate_tensor_write_to_file_calls(
                file_name, exec_dir
            ),
            benchmarking_and_reporting=self.generate_benchmarking_and_reporting(
                func_call, exec_dir
            ),
            update_tensor=self.generate_update_tensor_calls(),
            setup_thread_pool=setup_thread_pool,
            cleanup_str=self.generate_cleanup(),
        )
        return code_body

    def generate_cpp_wrapper(self, file_name, exec_dir):
        """
        Generates skeleton cpp file which launches triton kernel.
        Always uses the Triton code body template (which supports cleanup).
        """
        code_headers = self.generate_cpp_code_headers()
        code_define = self.generate_cpp_code_define()
        code_body = self.generate_cpp_code_body(file_name, exec_dir)
        return self.common_strings.code_string.format(
            code_headers=code_headers, code_define=code_define, code_body=code_body
        )


class TritonHexagonLauncher(HexagonLauncherBase):
    def _exec_kernel(
        self,
        kernel_obj_as_bytes: bytes,
        iterations: int,
        func_name: str,
        inputs: list[Tensor],
        output_profs: list,
        launch_grid: tuple,
        compiled_scratch: int | str | None = None,
        compiled_enable_multithreading: bool | str | None = None,
        compiled_enable_threaded_dispatch: bool | str | None = None,
        runtime_options: dict | None = None,
    ) -> list[Tensor]:
        # Getting the input metadata for effective wrapper codegen.
        input_profs = profile_triton_inputs(inputs)
        input_tensor_count = sum(
            [1 for inp in input_profs if inp.input_type == "tensor"]
        )

        local_dir_path, kernel_run_id = create_timestamped_folder(func_name)
        # 1 - Writting the object code for the kernel to disk
        obj_src_path = os.path.join(local_dir_path, func_name + ".o")
        Path(obj_src_path).write_bytes(kernel_obj_as_bytes)
        print(f"==> kernel obj saved in: {obj_src_path}")

        # Get HexagonOptions at this point for triton pipeline.
        options = HexagonOptions().__dict__.copy()
        if compiled_scratch is not None:
            options["scratch"] = int(compiled_scratch)
        if compiled_enable_multithreading is not None:
            if isinstance(compiled_enable_multithreading, str):
                options["enableMultiThreading"] = (
                    compiled_enable_multithreading.lower() == "true"
                )
            else:
                options["enableMultiThreading"] = bool(compiled_enable_multithreading)
        if compiled_enable_threaded_dispatch is not None:
            if isinstance(compiled_enable_threaded_dispatch, str):
                options["enableThreadedDispatch"] = (
                    compiled_enable_threaded_dispatch.lower() == "true"
                )
            else:
                options["enableThreadedDispatch"] = bool(
                    compiled_enable_threaded_dispatch
                )
        if runtime_options:
            # Use the union of cached option keys and current HexagonOptions fields
            # so that newly added flags (e.g. enableThreadedDispatch) are never silently
            # dropped when the kernel is served from a cache built before the flag existed.
            supported_keys = set(options.keys()) | set(HexagonOptions().__dict__.keys())
            for key, value in runtime_options.items():
                if key in supported_keys:
                    # Coerce types to match the compiled-time option types.
                    if key == "scratch":
                        value = int(value)
                    options[key] = value
            if (
                compiled_scratch is not None
                and "scratch" in runtime_options
                and int(runtime_options["scratch"]) != int(compiled_scratch)
            ):
                raise ValueError(
                    "Runtime scratch option does not match compile-time scratch "
                    f"(runtime={runtime_options['scratch']}, "
                    f"compiled={compiled_scratch})."
                )
            unsupported_keys = sorted(set(runtime_options.keys()) - supported_keys)
            # Keep compatibility with Triton call kwargs (e.g. constexpr args)
            # by ignoring unknown keys. Warn only for option-like names that
            # likely indicate a typo in backend/runtime options.
            suspicious = [
                k
                for k in unsupported_keys
                if k == "scratch" or k.startswith("enable") or k.endswith("_scratch")
            ]
            if suspicious:
                warnings.warn(
                    "Unsupported runtime options ignored: " + ", ".join(suspicious),
                    stacklevel=2,
                )

        # Safety gate: VTCM scratch with oversubscription is not supported on
        # the Hexagon launcher.  The ThreadManager spawns prod(grid) qurt
        # threads (1:1 with closures) and VTCM is sliced per flat_pid.  When
        # prod(grid) > num_threads, closures with flat_pid >= num_threads
        # would access VTCM out-of-bounds.  Full oversubscription support
        # (cap threads + loop) is deferred.
        if options["scratch"] > 0 and prod(launch_grid) > options["num_threads"]:
            raise ValueError(
                f"VTCM scratch with oversubscription "
                f"(grid={prod(launch_grid)} > "
                f"num_threads={options['num_threads']}) is not supported on "
                f"the Hexagon launcher. Reduce grid size."
            )

        # Safety gate: enableMultiThreading causes FormVirtualThreadsPass to
        # emit scf.forall (parallel tiling loops) which rely on the async
        # runtime. The single-instance wrapper uses a direct call with no async
        # setup, so scf.forall iterations beyond the first never execute and
        # the output is garbage. Disable for grid==1.
        if prod(launch_grid) == 1 and options.get("enableMultiThreading", False):
            warnings.warn(
                "Disabling enableMultiThreading for single-instance launch (grid==1) "
                "to avoid incorrect results from unexecuted scf.forall iterations.",
                stacklevel=2,
            )
            options["enableMultiThreading"] = False

        # Safety gate: enableThreadedDispatch has no effect for grid==1 since
        # the single-instance path uses a direct call (no ThreadManager).
        if prod(launch_grid) == 1 and options.get("enableThreadedDispatch", False):
            warnings.warn(
                "Disabling enableThreadedDispatch for single-instance launch (grid==1) "
                "since ThreadManager is not used for single-instance kernels.",
                stacklevel=2,
            )
            options["enableThreadedDispatch"] = False

        # Temporary safety gate:
        # UserDMA-backed hexagonmem-copy lowering is unstable with Triton SPMD
        # multithreaded wrappers at higher grid sizes (observed DSP abort in
        # UserDMA ring buffer handling). Keep SPMD functional by disabling DMA
        # copy lowering when more than one program instance is launched.
        if prod(launch_grid) > 1 and options.get("enableHexagonmemCopyToDMA", False):
            warnings.warn(
                "Disabling enableHexagonmemCopyToDMA for SPMD launch (grid>1) "
                "to avoid known UserDMA runtime instability.",
                stacklevel=2,
            )
            options["enableHexagonmemCopyToDMA"] = False

        # Pass lwp related info to HexagonExecutor() for creating shared obj and pulling lwp.json file if enabled.
        hexec = HexagonExecutor(
            kernel_run_id=kernel_run_id,
            enable_lwp=options["enableLWP"],
            enable_hexkl=options["enableHexKL"],
            cleanup_device_post_exec=options["deviceCleanup"],
        )

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
