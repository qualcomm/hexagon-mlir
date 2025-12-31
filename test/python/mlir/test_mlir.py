# ===- test_mlir.py ---------------------------------------------------------===
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause.
# For more license information:
#   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
#
# ===------------------------------------------------------------------------===

import argparse
import os
import subprocess

from triton.backends.qcom_hexagon_backend.mlir_launcher import (
    MLIRHexagonLauncher,
)
from triton.backends.qcom_hexagon_backend.compiler import (
    HexagonOptions,
)


def get_options(kernel_name: str, args) -> dict[str, str]:
    """Generate backend options with stringified values."""
    options = HexagonOptions().__dict__.copy()

    # Apply command-line overrides for commonly used options
    if args.enable_multi_threading is not None:
        options["enableMultiThreading"] = args.enable_multi_threading
    if args.enable_vtcm_tiling is not None:
        options["enableVTCMTiling"] = args.enable_vtcm_tiling
    if args.enable_convert_to_hexagonmem is not None:
        options["enableConvertToHexagonmem"] = args.enable_convert_to_hexagonmem
    if args.enable_hexagonmem_copy_to_dma is not None:
        options["enableHexagonmemCopyToDMA"] = args.enable_hexagonmem_copy_to_dma
    if args.enable_bufferization is not None:
        options["enableBufferization"] = args.enable_bufferization
    if args.enable_hexkl is not None:
        options["enableHexKL"] = args.enable_hexkl

    # Apply generic option overrides
    if args.option:
        for opt in args.option:
            try:
                key, value = opt.split("=", 1)
                # Convert string value to appropriate type
                if value.lower() in ("true", "false"):
                    options[key] = value.lower() == "true"
                elif value.isdigit():
                    options[key] = int(value)
                else:
                    options[key] = value
            except ValueError:
                raise ValueError(
                    f"Invalid option format: '{opt}'. Expected 'key=value'"
                )

            # Validate that the option exists in HexagonOptions
            if key not in HexagonOptions().__dict__:
                raise ValueError(
                    f"Unknown option: '{key}'. Must be a valid HexagonOptions field."
                )

    return {k: str(v) for k, v in options.items()}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run MLIR kernels with configurable Hexagon backend options",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python test_mlir.py --mlir_file_name input.mlir --cpp_wrapper_path wrapper.cpp
  
  # Enable DMA conversion
  python test_mlir.py --mlir_file_name input.mlir --cpp_wrapper_path wrapper.cpp \\
      --enable-hexagonmem-copy-to-dma
  
  # Disable bufferization
  python test_mlir.py --mlir_file_name input.mlir --cpp_wrapper_path wrapper.cpp \\
      --no-enable-bufferization
  
  # Use generic option flag for less common options
  python test_mlir.py --mlir_file_name input.mlir --cpp_wrapper_path wrapper.cpp \\
      --option fusion=false --option tileSizes="64,64"
  
  # Combine explicit and generic options
  python test_mlir.py --mlir_file_name input.mlir --cpp_wrapper_path wrapper.cpp \\
      --enable-multi-threading --option enableLWP=true
        """,
    )

    # Required arguments
    parser.add_argument(
        "--mlir_file_name", type=str, help="Name of external MLIR file", required=True
    )
    parser.add_argument(
        "--cpp_wrapper_path", type=str, help="Path to external C++ stub", required=True
    )
    parser.add_argument(
        "--func_name",
        type=str,
        help="Identifier for the kernel being executed",
        default="kernel",
    )
    parser.add_argument(
        "--validate_result",
        help="To extract and print perf report",
        action="store_true",
    )

    # Commonly used Hexagon backend options (explicit flags)
    options_group = parser.add_argument_group("Hexagon Backend Options (Commonly Used)")
    options_group.add_argument(
        "--enable-multi-threading",
        dest="enable_multi_threading",
        action="store_true",
        default=None,
        help="Enable async multi-threading (default: False)",
    )
    options_group.add_argument(
        "--enable-vtcm-tiling",
        dest="enable_vtcm_tiling",
        action="store_true",
        default=None,
        help="Enable lowering linalg-on-tensors (DDR) to tiled-compute on VTCM (default: True)",
    )
    options_group.add_argument(
        "--enable-convert-to-hexagonmem",
        dest="enable_convert_to_hexagonmem",
        action="store_true",
        default=None,
        help="Enable conversion from memref to hexagonmem ops (default: True)",
    )
    options_group.add_argument(
        "--enable-hexagonmem-copy-to-dma",
        dest="enable_hexagonmem_copy_to_dma",
        action="store_true",
        default=None,
        help="Enable conversion from hexagonmem copy to DMA ops (default: False)",
    )
    options_group.add_argument(
        "--enable-bufferization",
        dest="enable_bufferization",
        action="store_true",
        default=None,
        help="Enable bufferization passes (default: True)",
    )
    options_group.add_argument(
        "--enable-hexkl",
        dest="enable_hexkl",
        action="store_true",
        default=None,
        help="Enable HexKL to lower matmul and convolutions to HMX ops (default: False)",
    )

    # Generic option flag for all other HexagonOptions
    generic_group = parser.add_argument_group("Generic Options")
    generic_group.add_argument(
        "--option",
        action="append",
        metavar="KEY=VALUE",
        help="Set any HexagonOptions field (can be used multiple times). "
        'Example: --option fusion=false --option tileSizes="64,64"',
    )

    return parser.parse_args()


# Enabling direct execution to quickly test mlir files
def run_mlir():
    args = parse_args()
    options = get_options(args.func_name, args)
    MLIRHexagonLauncher.run_mlir_with_custom_cpp_wrapper(
        args.mlir_file_name,
        args.cpp_wrapper_path,
        args.validate_result,
        options,
        args.func_name,
    )


if __name__ == "__main__":
    run_mlir()
