# ===- mlir_test_utils.py ---------------------------------------------------===
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause.
# For more license information:
#   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
#
# ===------------------------------------------------------------------------===

"""
Shared utilities for MLIR kernel tests.
"""

import os
from typing import Dict

from triton.backends.qcom_hexagon_backend.mlir_launcher import (
    MLIRHexagonLauncher,
)
from triton.backends.qcom_hexagon_backend.compiler import (
    HexagonOptions,
)


def run_mlir_kernel_test(
    kernel_name: str,
    current_file_dir: str,
    options_overrides: Dict[str, bool] = None,
    validate_result: bool = True,
):
    """
    Run an MLIR kernel test with the specified parameters.

    Args:
        kernel_name: Name of the kernel to test
        current_file_dir: Directory containing the test files (input.mlir and wrapper.cpp)
        options_overrides: Dictionary of HexagonOptions to override (str->bool)
        validate_result: Whether to validate the result
    """
    mlir_input_path = f"{current_file_dir}/input.mlir"
    cpp_wrapper_path = f"{current_file_dir}/wrapper.cpp"

    # Get default options
    options = HexagonOptions().__dict__.copy()

    # Apply overrides if provided
    if options_overrides:
        options.update(options_overrides)

    # Convert all values to strings
    options = {k: str(v) for k, v in options.items()}

    MLIRHexagonLauncher.run_mlir_with_custom_cpp_wrapper(
        mlir_input_path, cpp_wrapper_path, validate_result, options, kernel_name
    )
