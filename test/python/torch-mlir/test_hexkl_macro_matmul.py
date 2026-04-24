# ===- test_hexkl_macro_matmul.py ------------------------------------------===
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause.
# For more license information:
#   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
#
# ===------------------------------------------------------------------------===

import torch
import torch.nn as nn
import utils
from triton.backends.qcom_hexagon_backend.compiler import HexagonOptions


class MatmulModel(nn.Module):
    """Generic matmul model with configurable weight shape and transpose."""

    def __init__(self, weight_shape, transpose=False):
        super(MatmulModel, self).__init__()
        self.weight = nn.Parameter(torch.randn(*weight_shape, dtype=torch.float16))
        self.transpose = transpose

    def forward(self, x):
        if self.transpose:
            return torch.matmul(x, self.weight.T)
        else:
            return torch.matmul(x, self.weight)


def test_large_square_1024x1024():
    """Test 1: Large square matmul (1024x1024x1024)"""
    print("\n=== Test: Large Square 1024x1024x1024 ===")

    torch.manual_seed(42)
    M, K, N = 1024, 1024, 1024

    model = MatmulModel((K, N), transpose=False)
    input_tensor = torch.randn(M, K, dtype=torch.float16)

    options = HexagonOptions().__dict__
    options["enableHexKL"] = True
    options["hexKLMode"] = "macro"

    manager = utils.ModelManager(model, input_tensor)
    manager.create_linalg_module()
    manager.write_bytecode_to_file()
    manager.execute_and_compare(options=options, rtol=1e-2, atol=1e-2)


def test_rectangular_tall_transposed():
    """Test 2: Rectangular tall with transposed weights (1024x512x256)"""
    print("\n=== Test: Rectangular Tall Transposed 1024x512x256 ===")

    torch.manual_seed(42)
    M, K, N = 1024, 512, 256

    model = MatmulModel((N, K), transpose=True)
    input_tensor = torch.randn(M, K, dtype=torch.float16)

    options = HexagonOptions().__dict__
    options["enableHexKL"] = True
    options["hexKLMode"] = "macro"

    manager = utils.ModelManager(model, input_tensor)
    manager.create_linalg_module()
    manager.write_bytecode_to_file()
    manager.execute_and_compare(options=options, rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    test_large_square_1024x1024()
    test_rectangular_tall_transposed()
