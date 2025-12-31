# ===- utils.py -------------------------------------------------------------===
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause.
# For more license information:
#   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
#
# ===------------------------------------------------------------------------===

import torch


def parameterize_func_name(*parameters):
    def decorator(func):

        # Currently supported torch types
        torch_dtype_str = {
            torch.float32: "float32",
            torch.float: "float32",
            torch.float16: "float16",
            torch.float64: "float64",
            torch.double: "float64",
            torch.uint8: "uint8",
            torch.uint16: "uint16",
            torch.uint32: "uint32",
            torch.uint64: "uint64",
            torch.int8: "int8",
            torch.int16: "int16",
            torch.short: "int16",
            torch.int32: "int32",
            torch.int: "int32",
            torch.int64: "int64",
            torch.long: "int64",
            torch.bool: "bool",
        }

        parameterized_func_name = func.__name__
        for p in parameters:

            # Checking the valid parameter types, which can be extended for other parameters as well.
            assert isinstance(p, (torch.dtype,)), "Parameter not supported"
            if isinstance(p, torch.dtype):
                assert p in torch_dtype_str, (
                    str(p) + " is not supported as torch parameter type"
                )
                parameterized_func_name += "_" + torch_dtype_str[p]

        func.__name__ = parameterized_func_name
        return func

    return decorator
