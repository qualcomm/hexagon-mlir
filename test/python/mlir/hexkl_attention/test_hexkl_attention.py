import pytest
import os
import sys

# Add parent directory to path to import mlir_test_utils
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from mlir_test_utils import run_mlir_kernel_test


def test_mlir_suite():
    kernel_name = "hexkl_attention"
    current_file_dir = os.path.dirname(os.path.abspath(__file__))

    run_mlir_kernel_test(
        kernel_name=kernel_name,
        current_file_dir=current_file_dir,
        options_overrides={"enableHexKL": True},
        validate_result=True,
    )
