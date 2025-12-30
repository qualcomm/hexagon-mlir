# ===- utils.py -------------------------------------------------------------===
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause.
# For more license information:
#   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
#
# ===------------------------------------------------------------------------===

import torch
import torch_mlir.fx as fx_mlir
from pathlib import Path
from triton.backends.qcom_hexagon_backend.torch_mlir_hexagon_launcher import (
    TorchMLIRHexagonLauncher,
)
import subprocess
import os


class ModelManager:
    def __init__(self, model, *inputs):
        self.model = model
        self.func_name = model.__class__.__name__
        self.linalg_filename = Path(__file__).parent / (str(self.func_name) + ".mlirbc")
        self.inputs = inputs

    # Convert the model to MLIR
    def create_linalg_module(self):
        self.mlir_module = fx_mlir.export_and_import(
            self.model,
            *self.inputs,
            output_type="linalg-on-tensors",
            func_name=self.func_name,
        )

    # Save the linalg module to a bytecode file
    def write_bytecode_to_file(self):
        bytecode = self.mlir_module.operation.get_asm(binary=True)
        # Save the linalg module as a bytecode in a file
        with open(self.linalg_filename, "wb") as f:
            f.write(bytecode)

    # Execute on x86 and hexagon and compare the results
    def execute_and_compare(
        self,
        rtol=1e-05,
        atol=1e-08,
        dump_outputs=False,
        iterations=1,
        options: dict = None,
        enable_etm=False,
    ):
        # Generate reference output
        reference = self.model(*self.inputs)

        # Hexagon execution
        output = TorchMLIRHexagonLauncher().run_torch_mlir(
            str(self.linalg_filename),
            self.inputs,
            self.func_name,
            base_dir_for_artifacts=None,
            iterations=iterations,
            options=options,
            enable_etm=enable_etm,
        )

        # Compare results
        if len(output) == 1:
            if dump_outputs:
                print("\nReference output:\n", reference)
                print("\nHexagon output:\n", output[0])

            assert torch.allclose(output[0], reference, rtol, atol)

        elif len(output) > 1:
            for index in range(len(output)):
                if dump_outputs:
                    print("\nReference output:\n", reference[index])
                    print("\nHexagon output:\n", output[index])

                assert torch.allclose(output[index], reference[index], rtol, atol)


def process_lwp():
    HEXAGON_MLIR_ROOT = os.environ.get("HEXAGON_MLIR_ROOT")

    if not HEXAGON_MLIR_ROOT:
        print("Cannot process lwp data as path to process_lwp.py is unknown")
        return

    subprocess.run(
        [
            "python3",
            f"{HEXAGON_MLIR_ROOT}/test/python/process_lwp.py",
            "/tmp/lwp.json",
            "/tmp/lwp_infodump.txt",
            "/tmp/initial-linalg.mlir",
        ],
        check=True,
    )
