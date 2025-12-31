# ===- test_add_matmul_softmax.py -------------------------------------------===
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause.
# For more license information:
#   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
#
# ===------------------------------------------------------------------------===

import torch
import utils
import pytest
from triton.backends.qcom_hexagon_backend.compiler import (
    HexagonOptions,
)
import subprocess
import os


# Define the computation graph
class AddMatmulSoftmax(torch.nn.Module):
    def __init__(self):
        super(AddMatmulSoftmax, self).__init__()

    def forward(self, x, y, z):
        add_result = x + y
        matmul_result = torch.matmul(add_result, z)
        softmax_result = torch.nn.functional.softmax(matmul_result, dim=-1)
        return softmax_result


@pytest.mark.parametrize("enablelwp", [False])
def test_add_matmul_softmax_torch(enablelwp):
    # Create dummy 2D tensors
    matrix_size = 32
    shape = (matrix_size, matrix_size)

    # Instantiate the model and run the forward pass
    model = AddMatmulSoftmax()
    x = torch.randn(shape)
    y = torch.randn(shape)
    z = torch.randn(shape)

    if enablelwp:
        options = HexagonOptions().__dict__
        options["enableLWP"] = True
        # options['disableLWPLoop'] = True # Uncomment to turn off loop level instrumentation.
    else:
        options = None

    manager = utils.ModelManager(model, x, y, z)
    manager.create_linalg_module()
    manager.write_bytecode_to_file()
    # Set rtol, atol as needed.
    # Pass dump_outputs=True to print the outputs.
    manager.execute_and_compare(rtol=1e-05, atol=1e-06)

    if enablelwp:
        utils.process_lwp()


if __name__ == "__main__":
    test_add_matmul_softmax_torch()
