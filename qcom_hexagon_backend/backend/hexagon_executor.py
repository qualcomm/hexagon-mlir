# ===- hexagon_executor.py --------------------------------------------------===
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause.
# For more license information:
#   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
#
# ===------------------------------------------------------------------------===

import atexit
import os
import shlex
import struct
import subprocess
import sys
import threading
import time
from collections import namedtuple
from pathlib import Path

import numpy as np
try:
    import torch
    from torch import Tensor  # For type annotations
except ImportError:
    torch = None  # type: ignore
    Tensor = object  # type: ignore
from triton.backends.qcom_hexagon_backend import knobs
from triton.backends.qcom_hexagon_backend.hexagon_profiler import HexagonProfiler
from triton.backends.qcom_hexagon_backend.utils import (  # type: ignore
    get_exec_mode,
    split_path,
    to_torch_type,
)

_BACKEND_DIR = Path(__file__).resolve().parent
_PACKAGED_ROOT = _BACKEND_DIR / "hexagon_mlir_root"
_RUN_MAIN_ON_HEXAGON_RPC_TIMEOUT_MS = 30 * 60 * 1000

# Common-lib device dirs (see HexagonExecutor._ensure_common_libs_pushed) for
# which a process-exit cleanup has already been registered, so repeated pushes
# within the same process don't stack up duplicate atexit handlers. Guarded by
# a lock because the check-then-add is not atomic across concurrent threads.
_common_lib_cleanup_registered: set[str] = set()
_common_lib_cleanup_lock = threading.Lock()

# Common-lib device dirs whose libs have already been confirmed present on the
# device in THIS process. Once a dir is in here, _ensure_common_libs_pushed
# short-circuits with no adb round-trip, so at most one presence check + push
# happens per process (normally warmed up by push_common_libs_for_session at
# pytest startup, so no test ever pays the check). Shares the lock above.
_common_libs_pushed: set[str] = set()


def _sdk_tool_version(q6_version: str) -> str:
    """Return the SDK prebuilt tool version for the given Hexagon arch version.

    The SDK 6.6.0.0 v81 assets use the toolv19 directory naming, while v79
    assets use toolv88. Older architectures use toolv87.
    """
    arch_version = int(q6_version.lstrip("v"))
    if arch_version >= 81:
        return "v19"
    if arch_version >= 79:
        return "v88"
    return "v87"


# This file is part of a small subset of python files that uses some type-annotations
# and it passes type-verification with mypy (a type checker).
# To typecheck this set of files, do:
# mypy compiler.py hexagon_executor.py hexagon_launcher_base.py torch_mlir_hexagon_launcher.py triton_hexagon_launcher.py --follow-untyped-imports --check-untyped-defs


class HexagonExecutor:
    def __init__(
        self,
        kernel_run_id: str,
        enable_lwp=False,
        enable_etm=False,
        compile_only=False,
        enable_hexkl=False,
        cleanup_device_post_exec=True,
        perf_path=None,
    ):
        """
        Initializes the HexagonExecutor instance.

        Attributes:
            exec_mode (str): Stores the provided mode of execution.
            device_path (str): Stores the base path where build artifacts are pushed to on device. Leaf node dir is named after 'kernel_run_id'.
            lib_path (str): Stores the path where dynamic libraries are pushed to on device. Must be child dir of 'device_path'.
            config (namedtuple): Holds the configuration for the HexagonExecutor, including
                                 environment variables, tools paths, and Q6 version, set by
                                 calling the `get_config` method.
        """
        if not compile_only:
            if (not isinstance(kernel_run_id, str)) or (len(kernel_run_id) == 0):
                raise ValueError(
                    "kernel_run_id must be well-formed, non-empty string for execution via standalone launcher"
                )
        self.exec_mode = get_exec_mode() if not compile_only else "compile_only"
        self.device_path = f"/data/local/tmp/{kernel_run_id}"
        self.lib_path = f"{self.device_path}/lib"
        self.alt_perf_path = (
            perf_path  # Only needed in contexts where the .cpp wrapper is hardcoded
        )
        self.config = self.get_config()
        self.enable_lwp = enable_lwp
        self.enable_etm = enable_etm  # When set to True, etm traces will be collected and processed with pyetm.
        self.enable_hexkl = enable_hexkl
        self.final_result = "Pass"
        self.cleanup_device_post_exec = cleanup_device_post_exec

    # Cases were encountered where on-device execution
    # performance report reported a failure, but
    # the PyTest would still pass. This function
    # is a temporary workaround to make sure
    # that the final result is correctly captured.
    def get_test_result(self, perf_local_path: Path):
        with open(perf_local_path) as file:
            for line in file:
                if "Result" in line:
                    return line.split("Result:")[-1].strip()

    # Returns the executable path if running on device.
    def get_Executable_Path(self):
        return self.device_path if self.exec_mode == "device" else ""

    @staticmethod
    def parse_ribbon_stack_size():
        ribbon_stack_size = os.environ.get("HEXAGON_RIBBON_STACK_SIZE", "").strip()
        if ribbon_stack_size:
            try:
                ribbon_stack_size = hex(int(ribbon_stack_size, 0))
            except ValueError:
                raise ValueError(
                    f"HEXAGON_RIBBON_STACK_SIZE must be an integer (decimal or 0x-hex), got: {ribbon_stack_size!r}"
                )
        return f" stack_size={ribbon_stack_size}" if ribbon_stack_size else ""

    def get_config(self):
        """Configures HexagonExecutor with environment variables, tools paths, and Q6 version."""
        config = namedtuple("config", ("env_vars", "HEX_TOOLS", "Q6_VERSION"))

        if knobs.hexagon.sdk_version is None:
            raise ValueError("HEXAGON_SDK_VERSION variable needed in the environment!")

        env = {
            "HEXAGON_TOOLS": str(knobs.hexagon.tools),
            "HEXAGON_MLIR_ROOT": (
                str(knobs.hexagon.mlir_root) if knobs.hexagon.mlir_root else None
            ),
            "HEXAGON_SDK_ROOT": str(knobs.hexagon.sdk_root),
            "HEXAGON_SDK_VERSION": knobs.hexagon.sdk_version,
            "Q6_VERSION": knobs.hexagon.arch_version,
            "HEXKL_ROOT": str(knobs.hexagon.hexkl_root or ""),
        }

        if self.exec_mode == "device":
            env["ANDROID_HOST"] = knobs.hexagon.android_host
            env["ANDROID_SERIAL"] = knobs.hexagon.android_serial

        if env.get("ANDROID_HOST"):
            env["ANDROID_HOST"] = "-H " + env["ANDROID_HOST"]

        # Get the directory for runtime libs like libhex_nn_v3.a
        env["HEXAGON_RUNTIME_LIBS_DIR"] = str(knobs.hexagon.runtime_libs_dir or "")

        # List of tools to join with HEXAGON_TOOLS
        tools = ["hexagon-clang", "hexagon-clang++", "hexagon-sim"]
        HEX_TOOLS = {
            tool: os.path.join(env["HEXAGON_TOOLS"], "bin", tool) for tool in tools
        }

        Q6_VERSION = env["Q6_VERSION"]
        return config(env_vars=env, HEX_TOOLS=HEX_TOOLS, Q6_VERSION=Q6_VERSION)

    '''
        Executing the shared objects on the device or simulator (depending on the env var RUN_ON_SIM).

        Attributes:
            paths_to_shared_libs_local (list<Path>): Paths to the .so in the local folder (the first one
                is expected to be the principal one, containing the main() function introduced by the wrapper).
            input_tensor_paths (list<Path>): Paths to the input tensors in the local directory.
            output_paths (list<Path>): Paths to the local directory where output tensors files need to be generated.
            generatePerf (bool): Boolean to specify whether performance report should be generated while running on device.
        """
    '''

    def run(
        self,
        paths_to_shared_libs_local: list[Path],
        input_tensor_paths: list[Path],
        output_paths: list[Path],
        generatePerf: bool = False,
    ) -> list[Tensor]:
        if self.exec_mode == "simulator":
            self.run_kernel_on_simulator(paths_to_shared_libs_local)
        else:
            self.run_kernel_on_device(
                paths_to_shared_libs_local,
                input_tensor_paths,
                output_paths,
                generatePerf,
            )

        print("==> Ran successfully, collecting results")
        results = []
        for output_path in output_paths:
            with open(output_path, "rb") as file:
                out_dtype_id = struct.unpack("I", file.read(4))[0]
                out_dtype = to_torch_type(out_dtype_id)
                out_rank = struct.unpack("q", file.read(8))[0]
                out_shape = [
                    struct.unpack("q", file.read(8))[0] for _ in range(out_rank)
                ]
                out_data = file.read()
                output_torch_tensor = torch.frombuffer(
                    np.frombuffer(out_data, dtype=np.uint8).copy(), dtype=out_dtype
                )
                output_torch_tensor = output_torch_tensor.reshape(out_shape)
                results.append(output_torch_tensor)
        return results

    # Auxiliary function used by generate_shared_object()
    # Returns the base directory common to all the libraries, and a list of their extracted names
    # (without the 'lib' prefix and without the '.so' extension) which will be used for linking
    # Assumption: The argument lib_dependencies_paths should not be empty
    def validate_and_extract_lib_names(
        self, lib_dependencies_paths: list[Path]
    ) -> tuple[Path, list[str]]:
        assert lib_dependencies_paths, "The list of library paths is empty."

        # Use split_path on the first item to get the reference directory
        base_dir, _, _ = split_path(lib_dependencies_paths[0])
        lib_names = []

        for path in lib_dependencies_paths:
            dir_path, filename, ext = split_path(path)

            # Check that all files are in the same directory
            assert (
                dir_path == base_dir
            ), f"Library {path} is not in the same directory as others (base_dir seemed to be: {base_dir})."
            # Check that all files have .so extension
            assert ext == ".so", f"Library {path} does not have a .so extension."
            # Remove 'lib' prefix if present
            assert filename.startswith(
                "lib"
            ), f"Library {path} does not start with 'lib'"
            filename = filename[3:]

            lib_names.append(filename)

        return base_dir, lib_names

    '''
        Generate a shared object (.so) for the kernel `kernel_obj_path` using the cpp wrapper `cpp_wrapper_path`
        If generating a shared object containing only constants (intended to be used by a principal shared object),
        then pass None for the `cpp_wrapper_path` as there is no need to construct a main() function around it.

        Attributes:
            cpp_wrapper_path (str): Path to the CPP wrapper file path.
            kernel_obj_path (str): Path to the kernel object code (.o).
        """
    '''

    def generate_shared_object(
        self,
        cpp_wrapper_path: str | Path | None,
        kernel_obj_path: Path,
        lib_dependencies_paths: list[Path] = [],
        htp_kernel_gen: bool = False,
    ) -> Path:
        print("wrapper_path = ", cpp_wrapper_path)
        print("kernel_code = ", kernel_obj_path)
        hexagon_compile_include_string = """-I{HEXAGON_MLIR_ROOT}/test/utils/dsp/include -I{HEXAGON_RUNTIME_INCLUDE_DIR} -I{HEXAGON_SDK_ROOT}/incs -I{HEXAGON_SDK_ROOT}/incs/stddef -I{HEXAGON_SDK_ROOT}/rtos/qurt/computev{Q6_VERSION}/include/qurt -I{HEXAGON_SDK_ROOT}/rtos/qurt/computev{Q6_VERSION}/include/posix"""
        runtime_libs = []

        HEX_CXX_FLAGS = f"-O3 -mv{self.config.Q6_VERSION} -mhvx"
        INCLUDES = hexagon_compile_include_string.format(
            HEXAGON_MLIR_ROOT=self.config.env_vars["HEXAGON_MLIR_ROOT"],
            HEXAGON_RUNTIME_INCLUDE_DIR=str(_BACKEND_DIR / "include"),
            HEXAGON_SDK_ROOT=self.config.env_vars["HEXAGON_SDK_ROOT"],
            Q6_VERSION=self.config.Q6_VERSION,
        )
        LINK_DIRS = ""

        # Qhmath library include
        qhmath_libs = {"qhmath": "qhl", "qhmath_hvx": "qhl_hvx"}
        for lib, parent_dir in qhmath_libs.items():
            qhmath_path = (
                Path(self.config.env_vars["HEXAGON_SDK_ROOT"])
                / "libs"
                / parent_dir
                / f"prebuilt/hexagon_tool{_sdk_tool_version(self.config.Q6_VERSION)}_v{self.config.Q6_VERSION}"
                / f"lib{lib}.a"
            )
            if not qhmath_path.exists():
                print(
                    f"Library lib{lib}.a was not found. Ensure that {qhmath_path.parent} exists on your system and contains lib{lib}.a"
                )
                sys.exit(1)

            LINK_DIRS += f" -L{qhmath_path.parent}"
            runtime_libs.append(lib)

        if self.enable_hexkl:
            hexkl_root = self.config.env_vars["HEXKL_ROOT"]
            if not hexkl_root:
                print("HEXKL_ROOT is required when HexKL is enabled")
                sys.exit(1)
            hexkl_dir = (
                Path(hexkl_root)
                / "lib"
                / self.config.env_vars["HEXAGON_SDK_VERSION"]
                / f"hexagon_toolv19_v{self.config.Q6_VERSION}"
            )
            kl_libs = ["libhexkl_micro.a", "libhexkl_macro.a"]
            for lib in kl_libs:
                hexkl_path = hexkl_dir / lib
                if not hexkl_path.exists():
                    print(
                        f"Library {lib} was not found. Ensure that {hexkl_dir} exists on your system and contains {lib}"
                    )
                    sys.exit(1)
                runtime_libs.append(str(hexkl_path))
            LINK_DIRS += f" -L{hexkl_dir}"

        # Check if HEXAGON_RUNTIME_LIBS_DIR + "/multithreading exists and "libhexagon_mlir_async_runtime.a" inside it
        runtime_libs_dir = self.config.env_vars["HEXAGON_RUNTIME_LIBS_DIR"]
        multithreading_dir = (
            Path(runtime_libs_dir) / "multithreading" if runtime_libs_dir else None
        )
        if (
            multithreading_dir is not None
            and multithreading_dir.exists()
            and (multithreading_dir / "libhexagon_mlir_async_runtime.a").exists()
        ):
            LINK_DIRS += f" -L{multithreading_dir}"
            runtime_libs.append("hexagon_mlir_async_runtime")
        # Allow both implicit (-> -lfoo) and explicit archive paths (-> /path/to/libfoo.a).
        runtime_libs_str = " ".join(
            [lib if lib.endswith(".a") else f"-l{lib}" for lib in runtime_libs]
        )
        print("==> Compiling shared object file ...")
        kernel_dir, kernel_file_without_ext, _ = split_path(kernel_obj_path)
        so_name = "libTriton" if htp_kernel_gen else "lib" + kernel_file_without_ext
        kernel_path = kernel_dir / f"{so_name}.so"
        # If the shared object we're generating needs to be linked with some dependencies, extract the base directory
        # of those libs and their names for latter providing them to the -L and -l options of the link command
        if lib_dependencies_paths:
            (
                dependencies_base_dir,
                lib_names_extracted,
            ) = self.validate_and_extract_lib_names(lib_dependencies_paths)

        # This function returns the empty string if `htp_kernel_gen` or if it's a module containing only constants
        # (which is tested with cpp_wrapper_path being None). Otherwise it returns its argument.
        keep_or_skip_string = lambda arg: (
            arg if (not htp_kernel_gen) and (cpp_wrapper_path != None) else ""
        )

        prof_utils_path = "{}/test/utils/dsp/include/prof_utils.cc".format(
            self.config.env_vars["HEXAGON_MLIR_ROOT"]
        )
        lwp_handler_path = (
            "{}/qcom_hexagon_backend/bin/runtime/profiler/lwp_handler.S".format(
                self.config.env_vars["HEXAGON_MLIR_ROOT"]
            )
        )

        command = "{} {} {} {} {} {} {} {} -shared -fPIC -G0 {} -o {}".format(
            self.config.HEX_TOOLS["hexagon-clang++"],
            HEX_CXX_FLAGS,
            INCLUDES,
            LINK_DIRS,
            keep_or_skip_string(cpp_wrapper_path),
            kernel_obj_path,
            runtime_libs_str,
            ("" if not self.enable_lwp else f"{prof_utils_path} {lwp_handler_path}"),
            (
                ""
                if not lib_dependencies_paths
                else "-L{} -l{}".format(
                    dependencies_base_dir, " -l".join(lib_names_extracted)
                )
            ),
            kernel_path,
        )
        print(command)
        try:
            subprocess.run(
                shlex.split(command),
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.STDOUT,
            )
        except subprocess.CalledProcessError as e:
            print(f"Error executing the command: {e}")
            sys.exit(1)
        return kernel_path

    def _chunked_push_commands(self, local_paths, chunk_size=48):
        """Return bounded adb push commands for large input sets."""
        host = self.config.env_vars["ANDROID_HOST"]
        serial = self.config.env_vars["ANDROID_SERIAL"]
        commands = []
        for start in range(0, len(local_paths), chunk_size):
            chunk = local_paths[start : start + chunk_size]
            commands.append(
                (
                    ["adb", *shlex.split(host), "-s", serial,
                     "push", *(str(p) for p in chunk), self.device_path],
                    True,
                )
            )
        return commands

    def _common_lib_device_dir(self) -> str:
        """Device dir for artifacts identical across every kernel run at this
        (tool_version, Q6_VERSION) pair, shared instead of re-pushed per test."""
        return f"/data/local/tmp/hexagon_mlir_common_libs/{_sdk_tool_version(self.config.Q6_VERSION)}_{self.config.Q6_VERSION}"

    def _common_lib_local_paths(self) -> list[Path]:
        """Local paths of the version-locked libs shared across every kernel
        run: the run_main_on_hexagon skel and the C++ runtime. Derived purely
        from config (Q6_VERSION, SDK/tools roots), so they can be pushed once at
        session start without any kernel context. Exits if a lib is missing."""
        librun_main_on_hexagon_skel_path = (
            Path(self.config.env_vars["HEXAGON_SDK_ROOT"])
            / f"libs/run_main_on_hexagon/ship/hexagon_tool{_sdk_tool_version(self.config.Q6_VERSION)}_v{self.config.Q6_VERSION}/librun_main_on_hexagon_skel.so"
        )
        libcpp_path = (
            Path(self.config.env_vars["HEXAGON_TOOLS"])
            / f"target/hexagon/lib/v{self.config.Q6_VERSION}/G0/pic/libc++.so.1"
        )
        libcppabi_path = (
            Path(self.config.env_vars["HEXAGON_TOOLS"])
            / f"target/hexagon/lib/v{self.config.Q6_VERSION}/G0/pic/libc++abi.so.1"
        )
        for path in (librun_main_on_hexagon_skel_path, libcpp_path, libcppabi_path):
            if not path.exists():
                print("Path does not exist:", path)
                sys.exit(1)
        return [librun_main_on_hexagon_skel_path, libcpp_path, libcppabi_path]

    def _ensure_common_libs_pushed(self, local_paths: list[Path]) -> str:
        """Push version-locked shared libs to a common device dir exactly once,
        instead of into every test's private lib_path. The presence check + push
        runs at most once per process: the first caller does it under the lock
        and records the dir in _common_libs_pushed, and every later caller
        (i.e. every subsequent test) short-circuits with no adb round-trip.
        Returns the common device dir."""
        common_dir = self._common_lib_device_dir()
        self._register_common_lib_cleanup(common_dir)
        with _common_lib_cleanup_lock:
            if common_dir in _common_libs_pushed:
                return common_dir
            self._push_common_libs(common_dir, local_paths)
            _common_libs_pushed.add(common_dir)
        return common_dir

    def _push_common_libs(self, common_dir: str, local_paths: list[Path]) -> None:
        """Create the common dir and push any libs not already on the device."""
        check_command = [
            "adb",
            *shlex.split(self.config.env_vars["ANDROID_HOST"]),
            "-s", self.config.env_vars["ANDROID_SERIAL"],
            "shell", f"mkdir -p {common_dir} && ls {common_dir}",
        ]
        print(" ".join(check_command), flush=True)
        start_time = time.time()
        # A failed/garbled presence check must not crash the run: fall back to
        # an empty "present" set so we simply attempt to (re)push every lib.
        try:
            output = subprocess.check_output(
                check_command, shell=False, stderr=subprocess.STDOUT
            )
            present = set(output.decode("utf-8", errors="replace").split())
        except subprocess.CalledProcessError as e:
            print(f"Failed to check common lib directory: {e}", flush=True)
            print(
                "Attempting to create directory and push all libraries...", flush=True
            )
            present = set()
        except Exception as e:
            print(f"Unexpected error checking common libs: {e}", flush=True)
            present = set()
        print(f"Command took {time.time() - start_time:.4f} seconds.", flush=True)
        missing = [p for p in local_paths if p.name not in present]
        if missing:
            push_command = [
                "adb",
                *shlex.split(self.config.env_vars["ANDROID_HOST"]),
                "-s", self.config.env_vars["ANDROID_SERIAL"],
                "push",
                *[str(p) for p in missing],
                common_dir,
            ]
            print(" ".join(push_command), flush=True)
            start_time = time.time()
            try:
                subprocess.run(
                    push_command,
                    shell=False,
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.STDOUT,
                )
            except subprocess.CalledProcessError as e:
                print(f"Error pushing common libraries: {e}", flush=True)
                sys.exit(1)
            print(f"Command took {time.time() - start_time:.4f} seconds.", flush=True)
        else:
            print(
                f"Common libs already present in {common_dir}, skipping push.",
                flush=True,
            )

    def _register_common_lib_cleanup(self, common_dir: str) -> None:
        """Remove the common lib dir once the process exits, so it doesn't
        outlive the test run it was pushed for."""
        android_host = self.config.env_vars["ANDROID_HOST"]
        android_serial = self.config.env_vars["ANDROID_SERIAL"]

        def cleanup():
            # argv form (no shell=True): common_dir/android_* never pass
            # through a shell, so there is no injection surface even if their
            # construction changes later. android_host is "" or "-H <host>".
            # (cleanup runs at interpreter exit via atexit.)
            command = ["adb"]
            if android_host:
                command.extend(android_host.split())
            command.extend(["-s", android_serial, "shell", "rm", "-rf", common_dir])
            print(" ".join(command), flush=True)
            start_time = time.time()
            try:
                subprocess.run(
                    command,
                    check=False,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.STDOUT,
                )
            except Exception as e:
                print(f"Cleanup failed for {common_dir}: {e}", flush=True)
            print(f"Command took {time.time() - start_time:.4f} seconds.", flush=True)

        # Mark-and-register under the lock so the set membership and the
        # atexit handler stay in lockstep: at most one handler per dir, and
        # "in the set" always implies a handler is actually registered.
        with _common_lib_cleanup_lock:
            if common_dir in _common_lib_cleanup_registered:
                return
            _common_lib_cleanup_registered.add(common_dir)
            atexit.register(cleanup)

    def run_kernel_on_device(
        self,
        paths_to_shared_libs_local: list[Path],
        input_tensor_paths: list[Path],
        output_paths: list[Path],
        generatePerf: bool = False,
    ):
        """Push files to device, and run on DSP"""
        print("==> Running the kernel on device ...")
        run_main_on_hexagon_path = (
            Path(self.config.env_vars["HEXAGON_SDK_ROOT"])
            / "libs/run_main_on_hexagon/ship/android_aarch64/run_main_on_hexagon"
        )
        # The skel and C++ runtime libs live in a shared device dir (pushed once
        # per process, normally at pytest startup); the skel/libc++ locals are
        # still needed below for the optional ETM app_bins copy.
        (
            librun_main_on_hexagon_skel_path,
            libcpp_path,
            libcppabi_path,
        ) = self._common_lib_local_paths()

        common_lib_device_dir = self._ensure_common_libs_pushed(
            [librun_main_on_hexagon_skel_path, libcpp_path, libcppabi_path]
        )

        # Construct all the paths to the shared libs on device, using the `device_path` and the name of the .so in the local folder
        paths_to_shared_libs_on_device = [
            f"{self.device_path}/{split_path(path_to_current_shared_lib_local)[1]}.so"
            for path_to_current_shared_lib_local in paths_to_shared_libs_local
        ]
        # The first lib is the principal lib, which we will need to provide later as an argument to run_main_on_hexagon
        path_to_principal_lib_local = paths_to_shared_libs_local[0]
        local_dir, principal_lib_without_ext, _ = split_path(
            path_to_principal_lib_local
        )
        path_to_principal_lib_on_device = paths_to_shared_libs_on_device[0]

        run_main_on_hexagon_debugconfig_local_path = os.path.join(
            local_dir, "run_main_on_hexagon.debugconfig"
        )
        Path(run_main_on_hexagon_debugconfig_local_path).write_text(
            f"rpctimeout={_RUN_MAIN_ON_HEXAGON_RPC_TIMEOUT_MS}\n", encoding="utf-8"
        )

        if generatePerf:
            if self.alt_perf_path:
                perf_device_path = self.alt_perf_path
            else:
                perf_device_path = f"{self.device_path}/perf.txt"
        else:
            perf_device_path = ""
        perf_local_path = (
            local_dir / f"{principal_lib_without_ext}_perf.txt" if generatePerf else ""
        )
        output_device_paths = [
            f"{self.device_path}/{Path(fname).name}" for fname in output_paths
        ]

        # WriteLWPOutput() in all wrappers writes to /data/local/tmp/lwp.json;
        # pull from that fixed path regardless of where kernel artifacts live.
        lwp_device_path = "/data/local/tmp/lwp.json"
        lwp_local_path = local_dir / "lwp.json"

        etm_local_dir = local_dir / "etm_pyetm"
        ribbon_stack_size_arg = self.parse_ribbon_stack_size()
        logcat_local_path = local_dir / "logcat.txt"

        # Initializing results list for the previous dumped tensor paths
        result = []

        # Removing all previous dumped tensors (necessary for proper output)
        try:
            all_dumps = [
                "adb",
                *shlex.split(self.config.env_vars["ANDROID_HOST"]),
                "-s", self.config.env_vars["ANDROID_SERIAL"],
                "shell", f"ls {self.device_path}/tensor_dump*.txt",
            ]
            result = (
                subprocess.check_output(all_dumps, shell=False, stderr=subprocess.DEVNULL).decode().splitlines()
            )

        except subprocess.CalledProcessError:
            print("No previous tensors to destroy.")

        # Try actually destroying the previously found dumps
        try:
            if len(result) != 0:
                for file in result:
                    name = file.strip().split("/")[-1]
                    dump_device_path = f"{self.device_path}/{name}"

                    subprocess.run(
                        ["adb", *shlex.split(self.config.env_vars["ANDROID_HOST"]),
                         "-s", self.config.env_vars["ANDROID_SERIAL"],
                         "shell", f"rm -rf {dump_device_path}"],
                        check=True,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.STDOUT,
                    )

                print("All previous dumped tensors destroyed.")
        except subprocess.CalledProcessError:
            print("Error destroying previous tensors.")
            sys.exit(1)

        # Commands are (command_list, should_run[, ignore_errors[, redirect_path]]).
        # redirect_path: when set, stdout+stderr are written to that file instead of DEVNULL.
        _adb = ["adb", *shlex.split(self.config.env_vars["ANDROID_HOST"]),
                "-s", self.config.env_vars["ANDROID_SERIAL"]]
        commands = [
            # Remove the output tensors and kernel.so from any previous run
            (
                _adb + ["shell", "rm -rf {} || true".format(
                    " ".join(output_device_paths + paths_to_shared_libs_on_device
                             + [perf_device_path] + [lwp_device_path])
                )],
                True, True,
            ),
            # Create base directory for this kernel's execution on device
            (_adb + ["shell", f"mkdir -p {self.device_path}"], True),
            # Create dynamic library directory for this kernel's execution on device
            (_adb + ["shell", f"mkdir -p {self.lib_path}"], True),
            # Batch pushes to avoid host argv limits for large input sets.
            *self._chunked_push_commands(
                input_tensor_paths
                + [run_main_on_hexagon_path]
                + [run_main_on_hexagon_debugconfig_local_path]
                + paths_to_shared_libs_local
            ),
            # Cleanup and copy all the necessary files related to the test
            (["mkdir", "-p", f"{etm_local_dir}/app_bins"], self.enable_etm),
            (
                ["cp",
                 *(str(p) for p in paths_to_shared_libs_local
                   + [librun_main_on_hexagon_skel_path] + [libcpp_path]),
                 f"{etm_local_dir}/app_bins"],
                self.enable_etm,
            ),
            # Run the kernel using run_main_on_hexagon
            (
                _adb + ["shell",
                    'cd {}; touch /vendor/lib/rfsa/adsp/run_main_on_hexagon.farf;'
                    ' export DSP_LIBRARY_PATH="{};{}";'
                    ' export ADSP_LIBRARY_PATH="{};{};/vendor/lib/rfsa/adsp/";'
                    ' ./run_main_on_hexagon 3 {}{}'.format(
                        self.device_path,
                        self.lib_path, common_lib_device_dir,
                        self.lib_path, common_lib_device_dir,
                        path_to_principal_lib_on_device,
                        ribbon_stack_size_arg,
                    )],
                True,
            ),
            # Save DSP output for diagnostics and perf recording.
            # redirect_path causes stdout+stderr to be written to the file.
            (_adb + ["logcat", "-d"], True, True, logcat_local_path),
            # Pull all the output tensors from device
            (
                _adb + ["pull", *output_device_paths,
                        str(Path((output_paths + ["/invalid/dir"])[0]).parent)],
                bool(output_paths),
            ),
            # Pull the perf report from device
            (
                _adb + ["pull", perf_device_path, str(perf_local_path)],
                generatePerf, True,
            ),
            # Pull the lwp report from device if it is enabled.
            (_adb + ["pull", lwp_device_path, str(lwp_local_path)], self.enable_lwp),
            (["cp", str(lwp_local_path), "/tmp/lwp.json"], self.enable_lwp),
            # Finally, tear down directories for this kernel's execution
            (
                _adb + ["shell", f"rm -rf {self.device_path} {self.lib_path} || true"],
                self.cleanup_device_post_exec,
            ),
        ]

        try:
            if self.enable_etm:
                hex_prof = HexagonProfiler(
                    etm_local_dir,
                    profiling_mode="etm",
                    device_lib_path=path_to_principal_lib_on_device,
                    local_lib_filename=f"{principal_lib_without_ext}.so",
                )
            for command, should_run, *rest in commands:
                ignore_errors = rest[0] if rest else False
                redirect_path = rest[1] if len(rest) > 1 else None
                if not should_run:
                    continue
                print(shlex.join(command), flush=True)
                start_time = time.time()
                if redirect_path:
                    with open(redirect_path, "w") as _rf:
                        subprocess.run(command, check=not ignore_errors,
                                       stdout=_rf, stderr=subprocess.STDOUT)
                else:
                    subprocess.run(command, check=not ignore_errors,
                                   stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
                end_time = time.time()
                elapsed_time = end_time - start_time
                print(f"Command took {elapsed_time:.4f} seconds.", flush=True)
                if any("run_main_on_hexagon.farf" in arg for arg in command) and self.enable_etm:
                    hex_prof.analyze_trace()
        except subprocess.CalledProcessError as e:
            print(f"Error executing the command: {e}")
            with open(logcat_local_path, "w") as _lf:
                subprocess.run(
                    ["adb", *shlex.split(self.config.env_vars["ANDROID_HOST"]),
                     "-s", self.config.env_vars["ANDROID_SERIAL"], "logcat", "-d"],
                    check=False, stdout=_lf, stderr=subprocess.STDOUT,
                )
            sys.exit(1)

        # Copy mlirbc files to /tmp to facilitate post-processing (replaces
        # the shell `cp *.mlirbc /tmp/initial-linalg.mlir || echo ...` idiom).
        if self.enable_lwp:
            mlirbc_files = list(Path(local_dir).glob("*.mlirbc"))
            if mlirbc_files:
                subprocess.run(["cp", *(str(f) for f in mlirbc_files),
                                "/tmp/initial-linalg.mlir"], check=False)
            else:
                print("No MLIRBC files to copy")

        # Initializing our result for the next two try blocks
        result = []
        if generatePerf and perf_local_path.exists():
            self.final_result = self.get_test_result(perf_local_path)

        # Finding all of our dumped tensors
        try:
            # Stores command to find all the dumps we generated using wildcard
            all_dumps = [
                "adb",
                *shlex.split(self.config.env_vars["ANDROID_HOST"]),
                "-s", self.config.env_vars["ANDROID_SERIAL"],
                "shell", f"ls {self.device_path}/tensor_dump_*.txt",
            ]

            # Finds all dumps generated by running earlier command
            result = (
                subprocess.check_output(all_dumps, shell=False, stderr=subprocess.DEVNULL).decode().splitlines()
            )

            prev_len = len(output_paths)
            empty = [""] * len(result)
            output_paths += empty
        except subprocess.CalledProcessError:
            print("No dumps generated.")
        except ValueError as v:
            print(v)

        # This portion places all the tensors into our output_paths list in order
        # so that we can compare them to the x86 dumps in order
        try:
            for file in result:
                name = file.strip().split("/")[-1]
                tensor_num = int(name.split(".")[0].split("_")[-1])
                dump_device_path = f"{self.device_path}/{name}"
                dump_local_path = local_dir / f"{principal_lib_without_ext}_{name}"

                # Placing this path in the correct spot
                output_paths[tensor_num + prev_len] = dump_local_path

                subprocess.run(
                    ["adb", *shlex.split(self.config.env_vars["ANDROID_HOST"]),
                     "-s", self.config.env_vars["ANDROID_SERIAL"],
                     "pull", dump_device_path, str(dump_local_path)],
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.STDOUT,
                )
        except subprocess.CalledProcessError as e:
            print(f"Error pulling dumped tensors: {e}")

        if generatePerf:
            try:
                with open(perf_local_path) as f:
                    print(f.read().strip())
            except Exception as e:
                print(f"Error opening perf report: {e}")

    def run_kernel_on_simulator(self, paths_to_shared_libs_local: list[Path]):
        """Run the kernel on hexagon simulator"""
        print("==> Running the kernel on simulator ...")
        # The first lib is the principal lib, which we will need to provide later
        path_to_principal_lib_local = paths_to_shared_libs_local[0]
        local_dir, _, _ = split_path(path_to_principal_lib_local)

        SIM_OSM_PATH = local_dir / "osm.cfg"
        SIM_Q6SS_PATH = local_dir / "q6ss.cfg"

        # Write cosim config files directly instead of via shell echo redirection.
        SIM_OSM_PATH.write_text(
            "{}/rtos/qurt/computev{}/debugger/lnx64/qurt_model.so\n".format(
                self.config.env_vars["HEXAGON_SDK_ROOT"], self.config.Q6_VERSION
            )
        )
        SIM_Q6SS_PATH.write_text(
            "{}/lib/iss/qtimer.so --csr_base=0xFC900000 --irq_p=1 --freq=19200000 --cnttid=1\n".format(
                self.config.env_vars["HEXAGON_TOOLS"]
            )
        )
        with open(SIM_Q6SS_PATH, "a") as _cfg:
            _cfg.write("{}/lib/iss/l2vic.so 32 0xFC910000\n".format(
                self.config.env_vars["HEXAGON_TOOLS"]
            ))

        sim_command = shlex.split(
            "{} -mv{}"
            " --usefs={}/target/hexagon/lib/v{}/G0/pic"
            " --simulated_returnval"
            " --cosim_file {}"
            " --l2tcm_base 0xd800"
            " --rtos {}"
            " {}/rtos/qurt/computev{}/sdksim_bin/runelf.pbn --"
            " {}/libs/run_main_on_hexagon/ship/hexagon_tool{}_v{}/run_main_on_hexagon_sim"
            " stack_size=0x400000 --"
            " {}".format(
                self.config.HEX_TOOLS["hexagon-sim"],
                self.config.Q6_VERSION,
                self.config.env_vars["HEXAGON_TOOLS"],
                self.config.Q6_VERSION,
                SIM_Q6SS_PATH,
                SIM_OSM_PATH,
                self.config.env_vars["HEXAGON_SDK_ROOT"],
                self.config.Q6_VERSION,
                self.config.env_vars["HEXAGON_SDK_ROOT"],
                _sdk_tool_version(self.config.Q6_VERSION),
                self.config.Q6_VERSION,
                path_to_principal_lib_local,
            )
        )
        try:
            print(shlex.join(sim_command))
            result = subprocess.run(
                sim_command, check=True, capture_output=True, text=True
            )
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
        except subprocess.CalledProcessError as e:
            print(f"Error executing the command: {e}")
            sys.exit(1)


def push_common_libs_for_session() -> None:
    """Push the version-locked shared libs (skel + C++ runtime) to the device
    once, up front, so no individual test has to check for or push them. Called
    from the shared test conftest at pytest startup. No-op when not executing on
    a device (e.g. simulator runs), and best-effort: any failure is reported but
    left for the first real kernel run to surface, so it never blocks collection."""
    if get_exec_mode() != "device":
        return
    try:
        executor = HexagonExecutor(kernel_run_id="common_lib_session_push")
        executor._ensure_common_libs_pushed(executor._common_lib_local_paths())
    except (Exception, SystemExit) as e:
        # A missing lib makes _common_lib_local_paths sys.exit; swallow it here
        # so collection proceeds and the first real kernel run surfaces it.
        print(f"Warning: could not pre-push common libs at session start: {e}")
