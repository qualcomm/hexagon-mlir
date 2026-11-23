# ===- triton_hexagon_launcher.py -------------------------------------------===
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause.
# For more license information:
#   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
#
# ===------------------------------------------------------------------------===

from __future__ import annotations

import sys
import warnings
from math import prod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor
from triton.backends.qcom_hexagon_backend.compiler import (
    HexagonOptions,
)
from triton.backends.qcom_hexagon_backend.hexagon_executor import HexagonExecutor
from triton.backends.qcom_hexagon_backend.hexagon_launcher_base import (
    HexagonLauncherBase,
    HexagonWrapperGenerator,
    WrapperGeneratorStrings,
    _fs_name,
    create_timestamped_folder,
)
from triton.backends.qcom_hexagon_backend.utils import (
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
    int thread_id;
    int total_threads;
}};
"""
        # multithread_helper calls hexagon_runtime_set_dma_worker_id, but the
        # kernel only links a strong definition when it emits DMA ops. Emit a
        # weak no-op so DMA-free kernels don't crash on an unresolved call; the
        # kernel's strong def overrides it when present.
        self.multithread_helper = """
extern "C" __attribute__((weak)) void
hexagon_runtime_set_dma_worker_id(int id, int total) {{}}

void multithread_helper(void *arg) {{
    Closure *closure = (Closure *)arg;
    hexagon_runtime_set_dma_worker_id(closure->thread_id, closure->total_threads);
    // Oversubscription loop: when prod(grid) > total_threads, each thread
    // handles multiple grid points by striding through the flat grid space.
    // flat_pid is decomposed to 3D (pid_X, pid_Y, pid_Z) using row-major
    // order: X outermost, Z innermost.
    //
    // Example: grid=(3,4,5), total_threads=7 → total_grid=60.
    // 60 grid points are distributed across 7 threads with stride 7:
    //   Thread 0 handles flat_pid = 0, 7, 14, 21, …
    //   Thread 1 handles flat_pid = 1, 8, 15, 22, …
    //   Thread 2 handles flat_pid = 2, 9, 16, 23, …
    // flat_pid linearizes the 3D grid in row-major order (X outermost, Z innermost):
    //   flat_pid = pid_X * (4*5) + pid_Y * 5 + pid_Z
    // The inverse decomposition recovers 3D coordinates, e.g. flat_pid=23:
    //   pid_Z = 23 % 5           = 3
    //   pid_Y = (23 / 5) % 4     = 0
    //   pid_X = (23 / 5) / 4     = 1
    //   → kernel invoked at (pid_X=1, pid_Y=0, pid_Z=3)
    int total_grid = closure->num_programs_X *
                     closure->num_programs_Y *
                     closure->num_programs_Z;
    for (int flat_pid = closure->thread_id;
         flat_pid < total_grid;
         flat_pid += closure->total_threads) {{
        int pid_Z = flat_pid % closure->num_programs_Z;
        int tmp   = flat_pid / closure->num_programs_Z;
        int pid_Y = tmp % closure->num_programs_Y;
        int pid_X = tmp / closure->num_programs_Y;
        {func_name}({closure_args_string}
                   closure->num_programs_X,
                   closure->num_programs_Y,
                   closure->num_programs_Z,
                   pid_X, pid_Y, pid_Z);
    }}
}}
"""
        self.thread_pool_setup = """
// One closure per thread. Each carries thread_id and total_threads
// so multithread_helper can stride over grid points.
// VTCM scratch is sliced per thread_id.
ThreadManager<Closure> tm({actual_threads});
std::vector<Closure> closures(tm.thread_cnt());
for (int tid = 0; tid < {actual_threads}; tid++) {{
    closures[tid] = {{{closure_args_string}
                      {grid[0]}, {grid[1]}, {grid[2]},
                      tid, {actual_threads}}};
    {vtcm_setup_str}
}}
"""
        self.thread_pool_exec = """tm.exec(multithread_helper, closures.data());"""

        # ---- HMX micro-daemon (hexKLMode="micro-daemon") codegen ----
        # One dedicated thread owns HMX and drains a lock-free queue; N worker
        # threads submit K-block jobs via hexkl.micro_hmx_dispatch and never
        # touch the HMX engine. These templates replace the plain SPMD ones
        # above when self.micro_daemon is set.

        # Helper run by every thread. The extra thread (thread_id ==
        # total_threads) is the daemon; the rest are workers striding the grid.
        # Workers pass the injected daemon args (thread_id, un-sliced vtcm_base
        # memref, queue) after grid/pid, matching the arg tail read by
        # DecomposeHexKLMatmulToDaemonPass (thread_id=n-3, vtcm_base=n-2,
        # queue=n-1).
        self.micro_daemon_helper = """
// micro_daemon_helper replaces the standard multithread_helper, so it carries
// its own weak no-op for hexagon_runtime_set_dma_worker_id: the kernel links a
// strong def only when it emits DMA ops (the decompose pass's A-tile DMA does),
// and this weak fallback keeps a DMA-free daemon kernel linkable.
extern "C" __attribute__((weak)) void
hexagon_runtime_set_dma_worker_id(int id, int total) {{}}

void multithread_helper(void *arg) {{
    Closure *closure = (Closure *)arg;
    // Dedicated HMX daemon: drains dispatched jobs until exit_signal is set and
    // all lanes are empty. HMX was acquired/locked once by
    // micro_daemon_setup_helper and stays locked across iterations; workers
    // never touch the HMX engine.
    if (closure->thread_id >= closure->total_threads) {{
        hmx_micro_daemon_serve((HMXMicroQueue *)closure->hmx_queue);
        return;
    }}
    // Worker: claim a private DMA engine (worker N -> slot N+1) so the per-tile
    // A-tile DMA does not contend with sibling workers on one shared slot-0
    // engine. Latent until now -- the A-tile load used a core copy, not the DMA
    // engine. The daemon thread returns above, so only workers set an id.
    hexagon_runtime_set_dma_worker_id(closure->thread_id, closure->total_threads);
    int total_grid = closure->num_programs_X *
                     closure->num_programs_Y *
                     closure->num_programs_Z;
    for (int flat_pid = closure->thread_id;
         flat_pid < total_grid;
         flat_pid += closure->total_threads) {{
        int pid_Z = flat_pid % closure->num_programs_Z;
        int tmp   = flat_pid / closure->num_programs_Z;
        int pid_Y = tmp % closure->num_programs_Y;
        int pid_X = tmp / closure->num_programs_Y;
        {func_name}({closure_args_string}
                   {scratch_memref}closure->num_programs_X,
                   closure->num_programs_Y,
                   closure->num_programs_Z,
                   pid_X, pid_Y, pid_Z,
                   closure->thread_id,
                   closure->vtcmBase, closure->vtcmBase, 0,
                   {hmx_slice_bytes_total}, 1,
                   (void *)closure->hmx_queue);
    }}
    // Last worker out signals the daemon to stop. __atomic_fetch_sub returns
    // the pre-decrement value, so ==1 identifies the final worker; the daemon's
    // serve() then drains and returns for this iteration (HMX stays locked until
    // micro_daemon_teardown_helper).
    if (__atomic_fetch_sub(&hmx_micro_workers_remaining, 1,
                           __ATOMIC_ACQ_REL) == 1) {{
        hmx_micro_daemon_exit((HMXMicroQueue *)closure->hmx_queue);
    }}
}}
// Run once before the benchmark loop: only the daemon thread acquires+locks HMX
// and programs the acc-read config; workers no-op. Hoisted out of the timed loop
// so the HMX acquire/lock is paid once, not per iteration.
void micro_daemon_setup_helper(void *arg) {{
    Closure *closure = (Closure *)arg;
    if (closure->thread_id >= closure->total_threads)
        hmx_micro_daemon_setup((HMXMicroQueue *)closure->hmx_queue);
}}
// Run once after the benchmark loop: only the daemon thread unlocks+releases
// HMX; workers no-op. Must run before the thread pool is destroyed so release
// happens on the acquiring thread.
void micro_daemon_teardown_helper(void *arg) {{
    Closure *closure = (Closure *)arg;
    if (closure->thread_id >= closure->total_threads)
        hmx_micro_daemon_teardown((HMXMicroQueue *)closure->hmx_queue);
}}
"""
        # Thread-pool setup (runs once). Allocates the queue in ordinary cached
        # memory (NOT VTCM) and creates actual_threads + 1 threads (the extra one
        # is the daemon). hmx_queue_bytes is declared in the VTCM-acquire block.
        self.micro_daemon_thread_pool_setup = """
// VTCM layout: [ HMX worker slices ({hmx_slice_bytes_total} B) | HMX config
// (_HMX_DAEMON_CONFIG_BYTES bytes) | HVX worker slices ].
// The queue lives in ordinary cached memory: its scalar atomic metadata is
// touched only by CPU threads (not HMX), and scalar VTCM accesses are expensive.
// Over-allocate + 64-byte-align so the struct's cache-line padding
// for head/tail and adjacent slots avoids false sharing as intended.
std::vector<int8_t> hmx_queue_backing(hmx_queue_bytes + 63);
int8_t *hmx_queue_storage =
    (int8_t *)(((uintptr_t)hmx_queue_backing.data() + 63) & ~(uintptr_t)63);
HMXMicroQueue *hmx_queue = (HMXMicroQueue *)hmx_queue_storage;
// 2048-aligned config offset (see the VTCM-acquire block).
uint32_t hmx_config_offset = (uint32_t)hmx_config_offset_bytes;
// One extra thread (index {actual_threads}) is the dedicated HMX daemon.
ThreadManager<Closure> tm({actual_threads} + 1);
std::vector<Closure> closures(tm.thread_cnt());
for (int tid = 0; tid < {actual_threads}; tid++) {{
    closures[tid] = {{{closure_args_string}
                      vtcmBase, (void *)hmx_queue,
                      {hvx_field_init}{grid[0]}, {grid[1]}, {grid[2]},
                      tid, {actual_threads}}};
    {hvx_closure_setup}
}}
// Daemon closure: thread_id == total_threads selects the daemon branch.
closures[{actual_threads}] = {{{closure_args_string}
                              vtcmBase, (void *)hmx_queue,
                              {hvx_field_init}{grid[0]}, {grid[1]}, {grid[2]},
                              {actual_threads}, {actual_threads}}};
// One-time (hoisted out of the benchmark loop): full queue init, then the
// daemon thread acquires+locks HMX and programs the acc-read config. The
// per-iteration path only rearms exit_signal (see micro_daemon_exec).
hmx_micro_queue_init((void *)hmx_queue_storage, (void *)vtcmBase,
                     hmx_config_offset, {actual_threads});
tm.exec(micro_daemon_setup_helper, closures.data());
"""
        # Executed each benchmark iteration. tm.exec is a full barrier, so
        # iteration N is fully drained before the rearm for N+1 runs.
        self.micro_daemon_exec = """// Per iteration: clear exit_signal only; the queue is already drained (head ==
// tail) and HMX stays locked from micro_daemon_setup_helper.
hmx_micro_daemon_rearm(hmx_queue);
__atomic_store_n(&hmx_micro_workers_remaining, {actual_threads},
                 __ATOMIC_RELAXED);
tm.exec(multithread_helper, closures.data());"""

        # Extra headers/decls for the daemon path. The HMXMicroDaemon runtime
        # header lives in a bin/runtime module dir that is NOT on the JIT
        # wrapper's include path, so forward-declare the C externs the wrapper
        # calls instead of #including it. The
        # kernel-side hmx_micro_dispatch / _wait_idle are emitted into the kernel
        # object by the HexKL lowering and resolved at link time, so they are not
        # declared here.
        self.micro_daemon_code_headers = """
#include <stddef.h>
#include <stdint.h>
extern "C" {
    typedef struct HMXMicroQueue HMXMicroQueue;
    size_t hmx_micro_queue_size_bytes(void);
    HMXMicroQueue *hmx_micro_queue_init(void *mem, void *vtcm_base,
                                        uint32_t config_offset,
                                        uint32_t num_lanes);
    void hmx_micro_daemon_setup(HMXMicroQueue *q);
    void hmx_micro_daemon_serve(HMXMicroQueue *q);
    void hmx_micro_daemon_teardown(HMXMicroQueue *q);
    void hmx_micro_daemon_rearm(HMXMicroQueue *q);
    void hmx_micro_daemon_exit(HMXMicroQueue *q);
    uint32_t hexkl_micro_hmx_config_size(void);
}
// Live-worker countdown; the last worker to finish signals the daemon to exit.
static int hmx_micro_workers_remaining;
"""

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
{lwp}
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
    # MLIR memref descriptor ABI: allocated ptr, aligned ptr, offset, size,
    # stride. Emitted whenever a raw VTCM memref is passed by value.
    _MEMREF_DESC_SIG = ", int8_t *, int8_t *, int64_t, int64_t, int64_t"

    # Must match kF16Layout in DecomposeHexKLMatmulToDaemonPass.cpp:
    # kRing activation tiles, kRing weight tiles, and one each of flat, acc,
    # and scratch. Each f16 32x32 tile occupies 2048 B.
    _HMX_DAEMON_RING_SLOTS = 8
    _HMX_DAEMON_F16_TILE_BYTES = 32 * 32 * 2
    _HMX_DAEMON_PER_INSTANCE_BYTES = (
        2 * _HMX_DAEMON_RING_SLOTS + 3
    ) * _HMX_DAEMON_F16_TILE_BYTES
    # Shared HMX config upper bound; the wrapper needs it at code-generation time.
    _HMX_DAEMON_CONFIG_BYTES = 16384

    def __init__(
        self,
        input_profs,
        iterations,
        func_name,
        output_profs,
        launch_grid,
        options: dict,
    ):

        # Setting the grid which corresponds to the Triton kernel launch grid.
        self.grid = launch_grid

        # The wrapper's worker count is bounded by both the launch grid and
        # num_threads. A single HVX worker is safe for LWP even when threaded
        # dispatch was requested (for example, scratch-backed launches force
        # that option on).
        effective_hvx_workers = min(prod(self.grid), int(options["num_threads"]))

        # Guard LWP only when multiple HVX workers can interleave timestamps.
        if options.get("enableLWP") and effective_hvx_workers > 1:
            raise ValueError(
                "LWP can't be enabled with multiple HVX workers: effective"
                f" HVX worker count is {effective_hvx_workers}; set num_threads=1"
            )
        # Grid > 1 needs thread-pool scaffolding even with serial dispatch.
        self.multithreading_enabled = prod(self.grid) > 1
        # enableThreadedDispatch selects tm.exec() vs tm.exec_serial(). Preserve
        # enableMultiThreading as an alias for existing callers.
        self.enable_threaded_spmd = bool(options["enableThreadedDispatch"]) or bool(
            options["enableMultiThreading"]
        )

        # HMX micro-daemon path: enableHexKL + hexKLMode="micro-daemon".
        # This always needs the thread-pool scaffolding (one HMX daemon thread +
        # N workers) with real threaded dispatch, regardless of grid size, and a
        # dedicated code path that injects the daemon args into the kernel call.
        self.micro_daemon = bool(options.get("enableHexKL")) and (
            options.get("hexKLMode") == "micro-daemon"
        )
        if self.micro_daemon:
            if len(output_profs) > 0:
                raise ValueError(
                    "hexKLMode='micro-daemon' does not support kernels with "
                    "direct return values; kernel outputs must be written "
                    "through a pointer argument."
                )
            self.multithreading_enabled = True
            self.enable_threaded_spmd = True


        super().__init__(
            input_profs,
            iterations,
            func_name,
            output_profs,
            TritonWrapperGeneratorStrings(),
            options,
        )
        self.scratch = options["scratch"]
        self.actual_threads = effective_hvx_workers
        self._init_vtcm_resources()

    def _init_vtcm_resources(self):
        """Compute per-instance VTCM scratch sizing.

        Only computes vtcm_per_instance_bytes (the aligned per-thread
        slice).  Total VTCM required is computed at code-gen time as
        vtcm_per_instance_bytes * actual_threads, where actual_threads
        is min(prod(grid), num_threads).  When prod(grid) > num_threads,
        each thread loops over multiple grid points (oversubscription).
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
            return self._MEMREF_DESC_SIG
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

    def _build_function_arg_string(self):
        """Build the argument string for the LLVM extern declaration.

        Handles both output-less kernels and kernels with direct returns,
        appending the VTCM arg and the six Triton grid/pid integers.
        """
        if len(self.output_profs) == 0:
            arg_string = self.generate_llvm_function_signature_arg_string()
        else:
            arg_string = self.generate_llvm_direct_returns_function_signature()
        if self.micro_daemon:
            # Scratch precedes grid/pid; daemon arguments are appended:
            #   [triton_args, {hexagon.scratch} HVX memref?, grid/pid (6 ints),
            #    thread_id (i32), vtcm_base (memref: alloc, align, off, size,
            #    stride), queue (ptr)].
            arg_string += self._vtcm_signature_arg()
            arg_string += ", int, int, int, int, int, int"
            arg_string += ", int"
            arg_string += self._MEMREF_DESC_SIG
            arg_string += ", void *"
            return arg_string
        # VTCM pointer, then num_programs(x,y,z) and program_id(x,y,z)
        arg_string += self._vtcm_signature_arg()
        arg_string += ", int, int, int, int, int, int"
        return arg_string

    def generate_llvm_function_signature(self):
        """Generate LLVM function definition for Triton"""
        return self.common_strings.extern_llvm_func_defn.format(
            kernel_name=self.func_name,
            function_arg_string=self._build_function_arg_string(),
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
        if self.micro_daemon:
            hmx_slice_bytes_total = (
                self._HMX_DAEMON_PER_INSTANCE_BYTES * self.actual_threads
            )
            hvx_slice_bytes_total = self.vtcm_per_instance_bytes * self.actual_threads
            # VTCM layout:
            #   [ HMX worker slices | HMX config | HVX worker slices ]
            formatted_tensor_string += f"""size_t hmx_queue_bytes = hmx_micro_queue_size_bytes();
uint32_t hmx_config_bytes = hexkl_micro_hmx_config_size();
if (hmx_config_bytes > {self._HMX_DAEMON_CONFIG_BYTES}u) {{
  FARF(ERROR, "HMX micro-daemon config exceeds reserved VTCM (bytes=%u, reserved=%u)",
       (unsigned)hmx_config_bytes, (unsigned){self._HMX_DAEMON_CONFIG_BYTES}u);
  return -1;
}}
// acc_read_int32 requires the config offset to be 2048-byte aligned.
int64_t hmx_config_offset_bytes =
    (((int64_t){hmx_slice_bytes_total}) + 2047) & ~(int64_t)2047;
"""
            if self.scratch:
                # HVX region follows the config block at 2048-byte alignment.
                formatted_tensor_string += f"""int64_t hvx_start_bytes =
    ((hmx_config_offset_bytes + {self._HMX_DAEMON_CONFIG_BYTES}) + 2047) & ~(int64_t)2047;
int64_t totalRequiredScratch = hvx_start_bytes + (int64_t){hvx_slice_bytes_total};
"""
            else:
                # HMX-only daemon needs slices and config only.
                formatted_tensor_string += f"""int64_t totalRequiredScratch =
    hmx_config_offset_bytes + (int64_t){self._HMX_DAEMON_CONFIG_BYTES};
"""
            formatted_tensor_string += """compute_res_attr_t vtcmRes;
HAP_compute_res_attr_init(&vtcmRes);
// HMX cannot span VTCM pages. Use the v1 API with b_single_page=1; v2 is absent
// from the target DSP image.
HAP_compute_res_attr_set_vtcm_param(&vtcmRes, (unsigned int)totalRequiredScratch, 1);
uint32_t vtcmContextID = HAP_compute_res_acquire(&vtcmRes, HAP_REQUEST_TIMEOUT_US);
if (vtcmContextID == 0) {
  FARF(ERROR, "Failed to acquire VTCM scratch (bytes=%lld)", (long long)totalRequiredScratch);
  return -1;
}
int8_t *vtcmBase = (int8_t *)HAP_compute_res_attr_get_vtcm_ptr(&vtcmRes);
if (vtcmBase == nullptr) {
  FARF(ERROR, "VTCM acquire returned null pointer");
  HAP_compute_res_release(vtcmContextID);
  return -1;
}
"""
            if self.scratch:
                # Base of the HVX sub-region; per-instance slices are carved
                # from here in the thread-pool setup.
                formatted_tensor_string += (
                    "int8_t *hvxBase = vtcmBase + hvx_start_bytes;\n"
                )
            return formatted_tensor_string
        if self.scratch:
            total_scratch = self.vtcm_per_instance_bytes * self.actual_threads
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
                formatted_tensor_string += """FuncInput<int8_t, 1> wrVtcmScratch = {1, &vtcmScratchDesc};
FuncInput<int8_t, 1> *pVtcmScratch = &wrVtcmScratch;
"""
        return formatted_tensor_string

    def generate_cleanup(self):
        """Generates cleanup code for resources allocated in the wrapper."""
        cleanup_str = ""
        if self.micro_daemon:
            # Release HMX on the daemon thread once, after the benchmark loop and
            # before the thread pool (and thus the daemon thread) is destroyed.
            cleanup_str += "tm.exec(micro_daemon_teardown_helper, closures.data());\n"
        if self.scratch or self.micro_daemon:
            cleanup_str += "HAP_compute_res_release(vtcmContextID);\n"
        return cleanup_str

    def generate_closure_definition(self):
        """
        Generates string for closure struct.
        Only used when multithreading.
        The 5 trailing fields are:
            - 3 for num_programs (1 for each axis)
            - thread_id and total_threads (for oversubscription loop)
        """
        closure_fields_string = ""
        indent = " " * 4
        for inp in self.input_profs:
            if inp.input_type == "tensor":
                closure_fields_string += f"{indent}int64_t t{inp.idx!s}Rank;\n"
                closure_fields_string += f"{indent}MemRefDescriptor<{inp.dtype}, {inp.rank}> *t{inp.idx!s}MemRef;\n"
            else:
                closure_fields_string += f"{indent}{inp.dtype} S{inp.idx!s};\n"
        if self.micro_daemon:
            # Un-sliced global VTCM base (the pass adds tid*perInstanceBytes)
            # and the shared daemon queue.
            closure_fields_string += f"{indent}int8_t *vtcmBase;\n"
            closure_fields_string += f"{indent}void *hmx_queue;\n"
            if self.scratch:
                closure_fields_string += (
                    f"{indent}MemRefDescriptor<int8_t, 1> vtcmScratchMemRef;\n"
                )
        elif self.scratch:
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
                closure_helper_string += f"{indent}closure->t{inp.idx!s}Rank,\n"
                closure_helper_string += f"{indent}closure->t{inp.idx!s}MemRef,\n"
            else:
                closure_helper_string += f"{indent}closure->S{inp.idx!s},\n"
        if self.micro_daemon:
            hmx_slice_bytes_total = (
                self._HMX_DAEMON_PER_INSTANCE_BYTES * self.actual_threads
            )
            hvx_indent = " " * 19
            scratch_memref = ""
            if self.scratch:
                scratch_memref = (
                    "closure->vtcmScratchMemRef.allocated,\n"
                    f"{hvx_indent}closure->vtcmScratchMemRef.aligned,\n"
                    f"{hvx_indent}closure->vtcmScratchMemRef.offset,\n"
                    f"{hvx_indent}closure->vtcmScratchMemRef.sizes[0],\n"
                    f"{hvx_indent}closure->vtcmScratchMemRef.strides[0],\n"
                    f"{hvx_indent}"
                )
            return self.common_strings.micro_daemon_helper.format(
                func_name=self.func_name,
                closure_args_string=closure_helper_string.strip(),
                hmx_slice_bytes_total=hmx_slice_bytes_total,
                scratch_memref=scratch_memref,
            )
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
            return """closures[tid].vtcmScratchMemRef = {vtcmBase,
                                                    vtcmBase + tid * vtcmPerInstanceBytes,
                                                    0,
                                                    {vtcmPerInstanceBytes},
                                                    {1}};"""
        return "closures[tid].vtcmScratch = vtcmBase + tid * vtcmPerInstanceBytes;"

    def generate_thread_pool_setup(self):
        """
        Generates calls to set up thread pool and map threads to closures.
        Only used while multithreading.
        Each thread is assigned a thread_id and loops over grid points with
        stride == total_threads (oversubscription support).
        """
        closure_args_string = ""
        indent = " " * 22
        for inp in self.input_profs:
            if inp.input_type == "tensor":
                closure_args_string += f"{indent}{inp.rank}, dt{inp.idx},\n"
            else:
                closure_args_string += f"{indent}S{inp.idx},\n"
        if self.micro_daemon:
            # vtcmBase/hmx_queue are appended by the template; grid/thread_id
            # too. Only tensor args come from here. The daemon base size operand
            # is the HMX region total (stride 38912 * workers).
            hmx_slice_bytes_total = (
                self._HMX_DAEMON_PER_INSTANCE_BYTES * self.actual_threads
            )
            # When scratch>0, add a value-init placeholder for the
            # vtcmScratchMemRef field (keeps aggregate-init field ordering) and
            # a per-instance assignment carving the HVX slice from hvxBase.
            hvx_field_init = ""
            hvx_closure_setup = ""
            if self.scratch:
                hvx_per_inst = str(self.vtcm_per_instance_bytes)
                hvx_field_init = "{}, "
                hvx_closure_setup = (
                    "closures[tid].vtcmScratchMemRef = {vtcmBase, hvxBase + tid * "
                    + hvx_per_inst
                    + ", 0, {"
                    + hvx_per_inst
                    + "}, {1}};"
                )
            return self.common_strings.micro_daemon_thread_pool_setup.format(
                grid=self.grid,
                actual_threads=self.actual_threads,
                hmx_slice_bytes_total=hmx_slice_bytes_total,
                closure_args_string=closure_args_string.strip(),
                hvx_field_init=hvx_field_init,
                hvx_closure_setup=hvx_closure_setup,
            )
        if self.scratch:
            if len(self.output_profs) == 0:
                # Placeholder values keep aggregate-initializer field ordering
                # intact; per-instance descriptor is assigned right after.
                closure_args_string += f"{indent}{{}},\n"
            else:
                closure_args_string += f"{indent}nullptr,\n"
        return self.common_strings.thread_pool_setup.format(
            grid=self.grid,
            actual_threads=self.actual_threads,
            closure_args_string=closure_args_string.strip(),
            vtcm_setup_str=self.generate_vtcm_closure_setup(),
        )

    def generate_thread_pool_exec(self):
        """Generates the calls to execute each closure with a thread"""
        if self.micro_daemon:
            return self.common_strings.micro_daemon_exec.format(
                actual_threads=self.actual_threads
            )
        if self.enable_threaded_spmd:
            return self.common_strings.thread_pool_exec
        return "tm.exec_serial(multithread_helper, closures.data());"

    def generate_cpp_code_headers(self) -> str:
        if self.multithreading_enabled:
            code_headers = (
                self.common_strings.code_headers
                + self.common_strings.triton_code_headers
            )
            if self.micro_daemon:
                code_headers += self.common_strings.micro_daemon_code_headers
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
            lwp=self.generate_lwp_call(),
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
        compiled_enable_lwp: bool | str | None = None,
        compiled_enable_hexkl: bool | str | None = False,
        compiled_hexkl_mode: str | None = None,
        compiled_num_threads: int | str | None = None,
        runtime_options: dict | None = None,
        input_arg_types: list | None = None,
    ) -> list[Tensor]:
        # Getting the input metadata for effective wrapper codegen.
        input_profs = profile_triton_inputs(inputs, input_arg_types)
        input_tensor_count = sum(
            [1 for inp in input_profs if inp.input_type == "tensor"]
        )

        local_dir_path = create_timestamped_folder(func_name)
        kernel_run_id = local_dir_path.name
        fs_name = _fs_name(func_name)
        # 1 - Writting the object code for the kernel to disk
        obj_src_path = local_dir_path / f"{fs_name}.o"
        obj_src_path.write_bytes(kernel_obj_as_bytes)
        print(f"==> kernel obj saved in: {obj_src_path}")

        # Get HexagonOptions at this point for triton pipeline.
        # NOTE: this starts from DEFAULTS, so any compile-time option that is not
        # restored below is silently replaced by its default here. Anything the
        # generated wrapper depends on must therefore travel through
        # compiler.py's pack_metadata and be reapplied in this block.
        options = HexagonOptions().__dict__.copy()
        if compiled_scratch is not None:
            options["scratch"] = int(compiled_scratch)
        # Drives actual_threads = min(prod(grid), num_threads), i.e. the worker
        # count baked into the wrapper (plus one for the HMX micro-daemon). Absent
        # for kernels served from a cache packed before the field was appended, in
        # which case the default stands.
        if compiled_num_threads is not None:
            options["num_threads"] = int(compiled_num_threads)
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
        if compiled_enable_lwp is not None:
            if isinstance(compiled_enable_lwp, str):
                options["enableLWP"] = compiled_enable_lwp.lower() == "true"
            else:
                options["enableLWP"] = bool(compiled_enable_lwp)
        if compiled_enable_hexkl is not None:
            if isinstance(compiled_enable_hexkl, str):
                options["enableHexKL"] = compiled_enable_hexkl.lower() == "true"
            else:
                options["enableHexKL"] = bool(compiled_enable_hexkl)
        if compiled_hexkl_mode is not None:
            options["hexKLMode"] = str(compiled_hexkl_mode)
        if runtime_options:
            # Use the union of cached option keys and current HexagonOptions fields
            # so that newly added flags (e.g. enableThreadedDispatch) are never silently
            # dropped when the kernel is served from a cache built before the flag existed.
            supported_keys = set(options.keys()) | set(HexagonOptions().__dict__.keys())
            for key, value in runtime_options.items():
                if key in supported_keys:
                    # Coerce types to match the compiled-time option types.
                    if key in ("scratch", "num_threads"):
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

        options["num_threads"] = int(options["num_threads"])
        if options["num_threads"] <= 0:
            raise ValueError(f"num_threads must be > 0, got {options['num_threads']}")

        # Note: enableMultiThreading at grid==1 is valid — FormVirtualThreadsPass
        # creates scf.forall ops which FormAsyncThreadsPass converts to async
        # intra-kernel threads. This does not require ThreadManager/SPMD dispatch.
        #
        # enableThreadedDispatch (SPMD tm.exec() vs tm.exec_serial()) is a no-op
        # at grid==1 since multithreading_enabled is already False.
        if prod(launch_grid) == 1 and options.get("enableThreadedDispatch", False):
            options["enableThreadedDispatch"] = False

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
            wrapper_generator, local_dir_path, fs_name, exec_dir
        )

        # 3 - Generate the corresponding shared object for the current object file
        so_path = hexec.generate_shared_object(cpp_wrapper_path, obj_src_path)
        print("==> Shared object generated: ", so_path)

        # 4 - Running the shared object located at `so_path`
        results = super().execute_kernel(
            hexec, local_dir_path, fs_name, [so_path], wrapper_generator
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
            cls.instance = super().__new__(cls)
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
