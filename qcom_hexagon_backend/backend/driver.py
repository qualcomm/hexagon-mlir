# ===- driver.py ------------------------------------------------------------===
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause.
# For more license information:
#   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
#
# ===------------------------------------------------------------------------===

from math import prod
from triton.backends.driver import DriverBase
from triton.backends.compiler import GPUTarget
from triton.backends.qcom_hexagon_backend.triton_hexagon_launcher import (
    TritonHexagonLauncher,
    HexagonUtils,
)
from triton.backends.qcom_hexagon_backend.utils import make_profiled_return


def getHexagonLauncherClass(device_type="dsp"):
    class HexagonLauncher:
        def __init__(self, src, metadata):
            if not hasattr(self, "_initialized"):
                raise RuntimeError(
                    "HexagonLauncher can only be instantialized by HexagonDriver."
                )
            cst_key = lambda i: src.fn.arg_names.index(i) if isinstance(i, str) else i
            self.input_type_list = {
                cst_key(key): value for key, value in src.signature.items()
            }
            self.launcher = TritonHexagonLauncher()
            pass

        def __call__(self, *args, **kwargs):
            # args =  grid_0, grid_1, grid_2, stream, kernel.function,
            #         kernel.packed_metadata, launch_metadata,
            #         CompiledKernel.launch_enter_hook,
            #         self.CompiledKernel.launch_exit_hook, *args
            kernel_llir = args[4]
            pack_metadata = args[5]
            unstructured_return_types = pack_metadata[7]
            return_profs = [
                make_profiled_return(ret) for ret in unstructured_return_types
            ]
            # Extract metadata["name"] field which has the function name.
            # Add "_mlir_ciface_" prefix if kernel has >0 returns (this changes the calling conv.)
            func_name = (
                "_mlir_ciface_" if len(return_profs) > 0 else ""
            ) + pack_metadata[6]
            iterations = pack_metadata[8]
            num_fixed_args = 9
            inputs_with_constants = list(args[num_fixed_args:])
            inputs = [
                inp
                for idx, inp in enumerate(inputs_with_constants)
                if not self.input_type_list[idx] == "constexpr"
            ]
            launch_grid = (args[0], args[1], args[2])
            if prod(launch_grid) < 1:
                raise ValueError(
                    """
                    Must set at least 1 thread in SPMD launch grid.
                    To launch singlethreaded, invoke kernel as follows:
                    your_kernel[(1,)](...)
                    """
                )
            self.launcher._exec_kernel(
                kernel_llir, iterations, func_name, inputs, return_profs, launch_grid
            )
            # TODO: There seems to be no way to propogate the call returns upward, because
            #    - The call result is not used by the caller
            #    - The packed_metadata field (or any input arg) is immutable, because the args are passed as a tuple
            #    - This HexagonLauncher class doesn't inherit from the caller (CompiledKernel)
            # To support direct returns, we'd need to make very simple changes to the upstream compiler.py

    return HexagonLauncher


def ty_to_cpp(ty):
    return {
        "i1": "int32_t",
        "i8": "int8_t",
        "i16": "int16_t",
        "i32": "int32_t",
        "i64": "int64_t",
        "u1": "uint32_t",
        "u8": "uint8_t",
        "u16": "uint16_t",
        "u32": "uint32_t",
        "u64": "uint64_t",
        "fp16": "float",
        "bf16": "float",
        "fp32": "float",
        "f32": "float",
        "fp64": "double",
    }[ty]


class HexagonDriver(DriverBase):
    def __init__(self, device_type="dsp"):
        self.utils = HexagonUtils()
        self.backend = "HEXAGON"
        self.device_type = device_type
        instance = getHexagonLauncherClass(device_type)
        instance._initialized = True
        self.launcher_cls = instance
        # Dummy executable/binary set, since there is no binary for our usecase.
        self.binary_ext = "hex"

    @staticmethod
    def is_active():
        return True

    def map_python_to_cpp_type(self, ty: str) -> str:
        return ty_to_cpp(ty)

    def get_benchmarker(self):
        from triton.testing import do_bench

        return do_bench

    def get_current_target(self):
        return GPUTarget("hexagon", 0, 0)

    def get_current_device(self):
        return GPUTarget("hexagon", 0, 0)

    def get_active_torch_device(self):
        import torch

        return torch.device("cpu")

    def get_current_stream(self, device):
        return None
