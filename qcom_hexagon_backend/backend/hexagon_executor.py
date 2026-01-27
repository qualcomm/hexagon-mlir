# ===- hexagon_executor.py --------------------------------------------------===
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause.
# For more license information:
#   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
#
# ===------------------------------------------------------------------------===

import os, subprocess, struct, sys, shutil
import time
import torch
import numpy as np
from collections import namedtuple
from typing import Optional  # For type annotations
from torch import Tensor  # For type annotations
from triton.backends.qcom_hexagon_backend.utils import (  # type: ignore
    get_env_var,
    get_exec_mode,
    to_torch_type,
    split_path,
)
from triton.backends.qcom_hexagon_backend.hexagon_profiler import HexagonProfiler

# This file is part of a small subset of python files that uses some type-annotations
# and it passes type-verification with mypy (a type checker).
# To typecheck this set of files, do:
# mypy compiler.py hexagon_executor.py hexagon_launcher_base.py torch_mlir_hexagon_launcher.py triton_hexagon_launcher.py --follow-untyped-imports --check-untyped-defs


class HexagonExecutor:
    def __init__(self, enable_lwp=False, enable_etm=False, compile_only=False):
        """
        Initializes the HexagonExecutor instance.

        Attributes:
            exec_mode (str): Stores the provided mode of execution.
            device_path (str): Stores the path where the executables will be placed on the device.
            config (namedtuple): Holds the configuration for the HexagonExecutor, including
                                 environment variables, tools paths, and Q6 version, set by
                                 calling the `get_config` method.
        """
        self.exec_mode = get_exec_mode() if not compile_only else "compile_only"
        self.device_path = "/vendor/bin"
        self.config = self.get_config()
        self.enable_lwp = enable_lwp
        self.enable_etm = enable_etm  # When set to True, etm traces will be collected and processed with pyetm.
        self.final_result = "Pass"

    # Cases were encountered where on-device execution
    # performance report reported a failure, but
    # the PyTest would still pass. This function
    # is a temporary workaround to make sure
    # that the final result is correctly captured.
    def get_test_result(self, perf_local_path: str):
        with open(perf_local_path, "r") as file:
            for line in file:
                if "Result" in line:
                    return line.split("Result:")[-1].strip()

    # Returns the executable path if running on device.
    def get_Executable_Path(self):
        return self.device_path if self.exec_mode == "device" else ""

    def get_config(self):
        """Configures HexagonExecutor with environment variables, tools paths, and Q6 version."""
        config = namedtuple("config", ("env_vars", "HEX_TOOLS", "Q6_VERSION"))

        # Define environment variables based on the execution mode
        env_vars = {
            "HEXAGON_TOOLS": "HEXAGON_TOOLS",
            "HEXAGON_MLIR_ROOT": "HEXAGON_MLIR_ROOT",
            "HEXAGON_SDK_ROOT": "HEXAGON_SDK_ROOT",
            "Q6_VERSION": "HEXAGON_ARCH_VERSION",
            "HEXKL_ROOT": "HEXKL_ROOT",
        }
        if self.exec_mode == "device":
            env_vars.update(
                {"ANDROID_HOST": "ANDROID_HOST", "ANDROID_SERIAL": "ANDROID_SERIAL"}
            )
        # Retrieve environment variables
        env = {key: get_env_var(val) for key, val in env_vars.items()}
        if "ANDROID_HOST" in env and env["ANDROID_HOST"]:
            env["ANDROID_HOST"] = "-H " + env["ANDROID_HOST"]

        # Get the directory for runtime libs like libhex_nn_v3.a
        # 1. First try HEXAGON_RUNTIME_LIBS_DIR env variable.
        # 2. If (1) is not set, then try this path:
        #       which(linalg-hexagon-opt)/../runtime
        linalg_hexagon_opt_path = shutil.which("linalg-hexagon-opt")
        default_runtime_libs_dir = ""
        if linalg_hexagon_opt_path is not None:
            linalg_hexagon_opt_dir = os.path.dirname(linalg_hexagon_opt_path)
            default_runtime_libs_dir = os.path.join(linalg_hexagon_opt_dir, "runtime")
        env["HEXAGON_RUNTIME_LIBS_DIR"] = get_env_var(
            "HEXAGON_RUNTIME_LIBS_DIR", default_runtime_libs_dir
        )

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
            paths_to_shared_libs_local (list<str>): Paths to the .so in the local folder (the first one
                is expected to be the principal one, containing the main() function introduced by the wrapper).
            input_tensor_paths (list<str>): Paths to the input tensors in the local directory.
            output_paths (list<str>): Paths to the local directory where output tensors files need to be generated.
            generatePerf (bool): Boolean to specify whether performance report should be generated while running on device.
        """
    '''

    def run(
        self,
        paths_to_shared_libs_local: list[str],
        input_tensor_paths: list[str],
        output_paths: list[str],
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

        print(f"==> Ran successfully, collecting results")
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
                    np.array(out_data), dtype=out_dtype
                )  #  cast to numpy and then passed to torch to suppress torch warning
                output_torch_tensor = output_torch_tensor.reshape(out_shape)
                results.append(output_torch_tensor)
        return results

    # Auxiliary function used by generate_shared_object()
    # Returns the base directory common to all the libraries, and a list of their extracted names
    # (without the 'lib' prefix and without the '.so' extension) which will be used for linking
    # Assumption: The argument lib_dependencies_paths should not be empty
    def validate_and_extract_lib_names(
        self, lib_dependencies_paths: list[str]
    ) -> tuple[str, list[str]]:
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
        cpp_wrapper_path: Optional[str],
        kernel_obj_path: str,
        lib_dependencies_paths: list[str] = [],
        htp_kernel_gen: bool = False,
    ) -> str:
        print("wrapper_path = ", cpp_wrapper_path)
        print("kernel_code = ", kernel_obj_path)
        hexagon_compile_include_string = """-I{HEXAGON_MLIR_ROOT}/test/utils/dsp/include -I{HEXAGON_SDK_ROOT}/incs -I{HEXAGON_SDK_ROOT}/incs/stddef -I{HEXAGON_SDK_ROOT}/rtos/qurt/computev{Q6_VERSION}/include/qurt -I{HEXAGON_SDK_ROOT}/rtos/qurt/computev{Q6_VERSION}/include/posix"""
        runtime_libs = ["qhmath_hvx"]

        HEX_CXX_FLAGS = f"-O3 -mv{self.config.Q6_VERSION} -mhvx"
        INCLUDES = hexagon_compile_include_string.format(
            HEXAGON_MLIR_ROOT=self.config.env_vars["HEXAGON_MLIR_ROOT"],
            HEXAGON_SDK_ROOT=self.config.env_vars["HEXAGON_SDK_ROOT"],
            Q6_VERSION=self.config.Q6_VERSION,
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

        LINK_DIRS = """-L{} -L{}""".format(
            self.config.env_vars["HEXAGON_RUNTIME_LIBS_DIR"],
            QHL_LINK_DIR,
        )

        hexkl_dir = """{HEXKL_ROOT}/lib/hexagon_toolv19_v{Q6_VERSION}""".format(
            HEXKL_ROOT=self.config.env_vars["HEXKL_ROOT"],
            Q6_VERSION=self.config.Q6_VERSION,
        )
        if os.path.exists(hexkl_dir) and os.path.exists(
            os.path.join(hexkl_dir, "libhexkl_micro.a")
        ):
            LINK_DIRS += f" -L{hexkl_dir}"
            runtime_libs.append("hexkl_micro")

        # Check if HEXAGON_RUNTIME_LIBS_DIR + "/multithreading exists and "libhexagon_mlir_async_runtime.a" inside it
        multithreading_dir = os.path.join(
            self.config.env_vars["HEXAGON_RUNTIME_LIBS_DIR"], "multithreading"
        )
        if os.path.exists(multithreading_dir) and os.path.exists(
            os.path.join(multithreading_dir, "libhexagon_mlir_async_runtime.a")
        ):
            LINK_DIRS += f" -L{multithreading_dir}"
            runtime_libs.append("hexagon_mlir_async_runtime")
        runtime_libs_str = " ".join([f"-l{lib}" for lib in runtime_libs])
        print("==> Compiling shared object file ...")
        dir, kernel_file_without_ext, _ = split_path(kernel_obj_path)
        so_name = "libTriton" if htp_kernel_gen else "lib" + kernel_file_without_ext
        kernel_path = os.path.join(dir, f"{so_name}.so")
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
            keep_or_skip_string(INCLUDES),
            keep_or_skip_string(LINK_DIRS),
            keep_or_skip_string(cpp_wrapper_path),
            kernel_obj_path,
            keep_or_skip_string(runtime_libs_str),
            (
                ""
                if not self.enable_lwp
                else "{} {}".format(prof_utils_path, lwp_handler_path)
            ),
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
                command,
                shell=True,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.STDOUT,
            )
        except subprocess.CalledProcessError as e:
            print(f"Error executing the command: {e}")
            sys.exit(1)
        return kernel_path

    def run_kernel_on_device(
        self,
        paths_to_shared_libs_local: list[str],
        input_tensor_paths: list[str],
        output_paths: list[str],
        generatePerf: bool = False,
    ):
        """Push files to device, and run on DSP"""
        print("==> Running the kernel on device ...")
        run_main_on_hexagon_path = os.path.join(
            self.config.env_vars["HEXAGON_SDK_ROOT"],
            "libs/run_main_on_hexagon/ship/android_aarch64/run_main_on_hexagon",
        )
        librun_main_on_hexagon_skel_path = os.path.join(
            self.config.env_vars["HEXAGON_SDK_ROOT"],
            "libs/run_main_on_hexagon/ship/hexagon_toolv87_v{}/librun_main_on_hexagon_skel.so".format(
                self.config.Q6_VERSION
            ),
        )

        libcpp_path = os.path.join(
            self.config.env_vars["HEXAGON_TOOLS"],
            "target/hexagon/lib/v{}/G0/pic/libc++.so.1".format(self.config.Q6_VERSION),
        )

        if not os.path.exists(libcpp_path):
            print("Path does not exist:", libcpp_path)
            sys.exit(1)

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

        perf_device_path = f"{self.device_path}/perf.txt" if generatePerf else ""
        perf_local_path = (
            os.path.join(local_dir, f"{principal_lib_without_ext}_perf.txt")
            if generatePerf
            else ""
        )
        output_device_paths = [
            f"{self.device_path}/{os.path.basename(fname)}" for fname in output_paths
        ]

        lwp_device_path = f"{self.device_path}/lwp.json"
        lwp_local_path = os.path.join(local_dir, "lwp.json")

        etm_local_dir = os.path.join(local_dir, "etm_pyetm")

        # Initializing results list for the previous dumped tensor paths
        result = []

        # Removing all previous dumped tensors (necessary for proper output)
        try:
            all_dumps = "adb {} -s {} shell ls {}/tensor_dump*.txt 2>/dev/null".format(
                self.config.env_vars["ANDROID_HOST"],
                self.config.env_vars["ANDROID_SERIAL"],
                self.device_path,
            )
            result = (
                subprocess.check_output(all_dumps, shell=True).decode().splitlines()
            )

        except subprocess.CalledProcessError as e:
            print("No previous tensors to destroy.")

        # Try actually destroying the previously found dumps
        try:
            if len(result) != 0:
                for file in result:
                    name = file.strip().split("/")[-1]
                    dump_device_path = f"{self.device_path}/{name}"

                    command = "adb {} -s {} shell 'rm -rf {}'".format(
                        self.config.env_vars["ANDROID_HOST"],
                        self.config.env_vars["ANDROID_SERIAL"],
                        dump_device_path,
                    )

                    subprocess.run(
                        command,
                        shell=True,
                        check=True,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.STDOUT,
                    )

                print("All previous dumped tensors destroyed.")
        except subprocess.CalledProcessError as e:
            print("Error destroying previous tensors.")
            sys.exit(1)

        # Commands is a list of tuple of the form (command, should_run)
        #     command - Command to possible run
        #     should_run - Boolean to check if we should run the command
        commands = [
            # Remove the output tensors and kernel.so from any previous run
            (
                "adb {} -s {} shell 'rm -rf {}'".format(
                    self.config.env_vars["ANDROID_HOST"],
                    self.config.env_vars["ANDROID_SERIAL"],
                    " ".join(
                        output_device_paths
                        + paths_to_shared_libs_on_device
                        + [perf_device_path]
                        + [lwp_device_path]
                    ),
                ),
                True,
            ),
            # Push run_main_on_hexagon binary, input_tensors and all the shared libs
            (
                "adb {} -s {} push {} {}".format(
                    self.config.env_vars["ANDROID_HOST"],
                    self.config.env_vars["ANDROID_SERIAL"],
                    " ".join(
                        input_tensor_paths
                        + [run_main_on_hexagon_path]
                        + paths_to_shared_libs_local
                    ),
                    self.device_path,
                ),
                True,
            ),
            # Push run_main_on_hexagon skel library
            (
                "adb {} -s {} push {} /vendor/lib/rfsa/adsp".format(
                    self.config.env_vars["ANDROID_HOST"],
                    self.config.env_vars["ANDROID_SERIAL"],
                    librun_main_on_hexagon_skel_path,
                ),
                True,
            ),
            # Push libc++.so.1 library specific to hexagon-tools and Q6 version
            (
                "adb {} -s {} push {} /vendor/lib/rfsa/adsp".format(
                    self.config.env_vars["ANDROID_HOST"],
                    self.config.env_vars["ANDROID_SERIAL"],
                    libcpp_path,
                ),
                True,
            ),
            # Cleanup and copy all the necessary files related to the test
            (
                "mkdir -p {dir}/app_bins && cp {} {dir}/app_bins".format(
                    " ".join(
                        paths_to_shared_libs_local
                        + [librun_main_on_hexagon_skel_path]
                        + [libcpp_path]
                    ),
                    dir=etm_local_dir,
                ),
                self.enable_etm,
            ),
            # Run the kernel using run_main_on_hexagon
            (
                "adb {} -s {} shell 'cd {}; touch /vendor/lib/rfsa/adsp/run_main_on_hexagon.farf; ./run_main_on_hexagon 3 {}'".format(
                    self.config.env_vars["ANDROID_HOST"],
                    self.config.env_vars["ANDROID_SERIAL"],
                    self.device_path,
                    path_to_principal_lib_on_device,
                ),
                True,
            ),
            # Pull all the output tensors from device
            (
                "adb {} -s {} pull {} {}".format(
                    self.config.env_vars["ANDROID_HOST"],
                    self.config.env_vars["ANDROID_SERIAL"],
                    " ".join(output_device_paths),
                    # Assumption: All output_paths have same dirname
                    # Append a random entry to make sure the list in not empty.
                    os.path.dirname((output_paths + ["/invalid/dir"])[0]),
                ),
                bool(output_paths),
            ),
            # Pull the perf report from device
            (
                "adb {} -s {} pull {} {}".format(
                    self.config.env_vars["ANDROID_HOST"],
                    self.config.env_vars["ANDROID_SERIAL"],
                    perf_device_path,
                    perf_local_path,
                ),
                generatePerf,
            ),
            # Pull the lwp report from device if it is enabled.
            # "Copy the relevant files to /tmp to facilitate post-processing.
            (
                "adb {} -s {} pull {} {} && "
                "cp {} /tmp/lwp.json && "
                "cp {}/*.mlirbc /tmp/initial-linalg.mlir || echo 'No MLIRBC files to copy' ".format(
                    self.config.env_vars["ANDROID_HOST"],
                    self.config.env_vars["ANDROID_SERIAL"],
                    lwp_device_path,
                    lwp_local_path,
                    lwp_local_path,
                    local_dir,
                ),
                self.enable_lwp,
            ),
        ]

        try:
            if self.enable_etm:
                if "mlir_ciface" in principal_lib_without_ext:
                    kernel_name = ""  # torch-mlir flow -> not just one kernel
                else:
                    kernel_name = principal_lib_without_ext[3:]  # remove lib prefix

                hex_prof = HexagonProfiler(
                    etm_local_dir, profiling_mode="etm", kernel_name=kernel_name
                )
            for command, should_run in commands:
                if not should_run:
                    continue
                print(command, flush=True)
                start_time = time.time()
                subprocess.run(
                    command,
                    shell=True,
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.STDOUT,
                )
                end_time = time.time()
                elapsed_time = end_time - start_time
                print(f"Command took {elapsed_time:.4f} seconds.", flush=True)
                if "run_main_on_hexagon.farf" in command and self.enable_etm:
                    hex_prof.analyze_trace()
        except subprocess.CalledProcessError as e:
            print(f"Error executing the command: {e}")
            sys.exit(1)

        # Initializing our result for the next two try blocks
        result = []
        if generatePerf and os.path.exists(perf_local_path):
            self.final_result = self.get_test_result(perf_local_path)

        # Finding all of our dumped tensors
        try:
            # Stores command to find all the dumps we generated using wildcard
            all_dumps = "adb {} -s {} shell ls {}/tensor_dump_*.txt 2>/dev/null".format(
                self.config.env_vars["ANDROID_HOST"],
                self.config.env_vars["ANDROID_SERIAL"],
                self.device_path,
            )

            # Finds all dumps generated by running earlier command
            result = (
                subprocess.check_output(all_dumps, shell=True).decode().splitlines()
            )

            prev_len = len(output_paths)
            empty = [""] * len(result)
            output_paths += empty
        except subprocess.CalledProcessError as e:
            print(f"No dumps generated.")
        except ValueError as v:
            print(v)

        # This portion places all the tensors into our output_paths list in order
        # so that we can compare them to the x86 dumps in order
        try:
            for file in result:
                name = file.strip().split("/")[-1]
                tensor_num = int(name.split(".")[0].split("_")[-1])
                dump_device_path = f"{self.device_path}/{name}"
                dump_local_path = os.path.join(
                    local_dir, f"{principal_lib_without_ext}_{name}"
                )

                command = "adb {} -s {} pull {} {}".format(
                    self.config.env_vars["ANDROID_HOST"],
                    self.config.env_vars["ANDROID_SERIAL"],
                    dump_device_path,
                    dump_local_path,
                )

                # Placing this path in the correct spot
                output_paths[tensor_num + prev_len] = dump_local_path

                subprocess.run(
                    command,
                    shell=True,
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.STDOUT,
                )
        except subprocess.CalledProcessError as e:
            print(f"Error pulling dumped tensors: {e}")

        if generatePerf:
            try:
                with open(perf_local_path, "r") as f:
                    print(f.read().strip())
            except Exception as e:
                print(f"Error opening perf report: {e}")

    def run_kernel_on_simulator(self, paths_to_shared_libs_local: list[str]):
        """Run the kernel on hexagon simulator"""
        print("==> Running the kernel on simulator ...")
        # The first lib is the principal lib, which we will need to provide later
        path_to_principal_lib_local = paths_to_shared_libs_local[0]
        local_dir, _, _ = split_path(path_to_principal_lib_local)

        SIM_OSM_PATH = os.path.join(local_dir, f"osm.cfg")
        SIM_Q6SS_PATH = os.path.join(local_dir, f"q6ss.cfg")

        commands = [
            (
                'echo "{}/rtos/qurt/computev{}/debugger/lnx64/qurt_model.so" > {}'.format(
                    self.config.env_vars["HEXAGON_SDK_ROOT"],
                    self.config.Q6_VERSION,
                    SIM_OSM_PATH,
                )
            ),
            (
                'echo "{}/lib/iss/qtimer.so --csr_base=0xFC900000 --irq_p=1 --freq=19200000 --cnttid=1" > {}'.format(
                    self.config.env_vars["HEXAGON_TOOLS"], SIM_Q6SS_PATH
                )
            ),
            (
                'echo "{}/lib/iss/l2vic.so 32 0xFC910000" >> {}'.format(
                    self.config.env_vars["HEXAGON_TOOLS"], SIM_Q6SS_PATH
                )
            ),
            (
                "{} -mv{} \
                --usefs={}/../Tools/target/hexagon/lib/v{}/G0/pic \
                --simulated_returnval \
                --cosim_file {} \
                --l2tcm_base 0xd800 \
                --rtos {} \
                {}/rtos/qurt/computev{}/sdksim_bin/runelf.pbn -- \
                {}/libs/run_main_on_hexagon/ship/hexagon_toolv87_v{}/run_main_on_hexagon_sim \
                stack_size=0x400000 -- \
                {}"
            ).format(
                self.config.HEX_TOOLS["hexagon-sim"],
                self.config.Q6_VERSION,
                self.config.env_vars["HEXAGON_TOOLS"],
                self.config.Q6_VERSION,
                SIM_Q6SS_PATH,
                SIM_OSM_PATH,
                self.config.env_vars["HEXAGON_SDK_ROOT"],
                self.config.Q6_VERSION,
                self.config.env_vars["HEXAGON_SDK_ROOT"],
                self.config.Q6_VERSION,
                path_to_principal_lib_local,
            ),
        ]
        try:
            for cmd_idx, command in enumerate(commands):
                print(command)
                result = subprocess.run(
                    command, shell=True, check=True, capture_output=True, text=True
                )
                if cmd_idx == len(commands) - 1:
                    print(f"STDOUT: {result.stdout}")
                    print(f"STDERR: {result.stderr}")
        except subprocess.CalledProcessError as e:
            print(f"Error executing the command: {e}")
            sys.exit(1)
