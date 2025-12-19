# ===- utils.py -------------------------------------------------------------===
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause.
# For more license information:
#   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
#
# ===------------------------------------------------------------------------===

import os, re
import torch
from collections import namedtuple
from enum import Enum


class MissingEnvironmentVariable(Exception):
    pass


def get_env_var(name, default=None):
    if default is not None:
        return os.environ.get(name, default)
    try:
        return os.environ[name]
    except KeyError:
        raise MissingEnvironmentVariable(f"{name} variable needed in the environment!")


def split_path(path):
    filename = os.path.basename(path)
    directory = os.path.dirname(path)
    filename_without_ext, extension = os.path.splitext(filename)
    return (directory, filename_without_ext, extension)


def to_torch_type(type_id):
    cpp_to_torch_map = {
        1: torch.float16,
        2: torch.float32,
        3: torch.float64,
        4: torch.int8,
        5: torch.int16,
        6: torch.int32,
        7: torch.int64,
        8: torch.uint8,
    }
    # Old torch versions do not support torch.uint16, torch.uint32, torch.uint64
    try:
        cpp_to_torch_map[9] = torch.uint16
        cpp_to_torch_map[10] = torch.uint32
    except:
        pass
    return cpp_to_torch_map[type_id]


def replace_list_brackets(dimension_list):
    """Replaces a list surrounded by square brackets with curly braces"""
    dimension_list = str(dimension_list)
    if "[" in str(dimension_list) and "]" in dimension_list:
        return dimension_list.translate(str.maketrans({"[": "{", "]": "}"}))


def get_shape(shape):
    """Converts torch sizes to conventional list of sizes/strides"""
    sizes = [
        dimension for dimension in shape
    ]  # shape here will be of form torch.Size[...]
    # Now, we want the strides. NOTE: We can skip the first dimension and keep inserting and multiplying with val at first index
    strides = [1]
    for dim_shape in reversed(sizes[1:]):
        strides.insert(0, strides[0] * dim_shape)
    sizes = replace_list_brackets(sizes)
    strides = replace_list_brackets(strides)
    return sizes, strides


def get_ctype(dtype):
    """Maps a torch data type to hexagon-clang compatible data types"""
    mapping = {
        torch.double: "double",
        torch.float: "float",
        torch.float64: "double",
        torch.float32: "float",
        torch.float16: "_Float16",
        torch.long: "int32_t",
        torch.int: "int32_t",
        torch.int64: "int64_t",
        torch.int32: "int32_t",
        torch.int16: "int16_t",
        torch.int8: "int8_t",
        torch.uint8: "uint8_t",
        torch.bool: "bool",
    }
    # Old torch versions do not support torch.uint16, torch.uint32 and
    # torch.uint64. Add entry into the map only if there are no errors
    try:
        mapping[torch.uint64] = "uint64_t"
        mapping[torch.uint32] = "uint32_t"
        mapping[torch.uint16] = "uint16_t"
    except:
        pass
    return mapping.get(
        dtype, "unknown_type"
    )  # returns value of key-value pair if found in mapping, else returns "unknown_type"


def get_mlir_to_ctype(dtype):
    """Maps a MLIR data type to hexagon-clang compatible data types"""
    mapping = {
        "f64": "double",
        "f32": "float",
        "f16": "_Float16",
        "i32": "int32_t",
        "i64": "int64_t",
        "i16": "int16_t",
        "i8": "int8_t",
        "b8": "bool",
    }

    return mapping.get(
        dtype, "unknown_type"
    )  # returns value of key-value pair if found in mapping, else returns "unknown_type"


def profile_triton_inputs(inputs):
    return profile_inputs(inputs)


def profile_torch_mlir_inputs(inputs):
    return profile_inputs(inputs)


# Given an list of input tensors, to return the profile inputs by categorizing the inputs
# into various inputs types and deriving related metadata.
def profile_inputs(inputs):
    Profiled_input = namedtuple(
        "InputProfile",
        (
            "idx",  # Index for tensor or scalar input
            "input_id",  # Corresponding index in the input list
            "input_type",  # Type of the input (tensor or scalar)
            "value",  # Actual value of the input
            "dtype",  # CPP data type of the input value
            "rank",  # Rank of the input (None for scalars)
            "shape",  # Shape of the input (None for scalars)
        ),
    )
    profiled = []
    tensor_index = 0
    scalar_index = 0
    for index, inp in enumerate(inputs):
        if isinstance(inp, torch.Tensor):
            profiled.append(
                Profiled_input(
                    input_type="tensor",
                    idx=tensor_index,
                    input_id=index,
                    value=inp.data,
                    dtype=get_ctype(inp.dtype),
                    rank=inp.dim(),
                    shape=get_shape(inp.shape),
                )
            )
            tensor_index += 1
        elif isinstance(inp, (int, float)):
            profiled.append(
                Profiled_input(
                    input_type="scalar",
                    idx=scalar_index,
                    input_id=index,
                    value=inp,
                    dtype=type(inp).__name__,
                    rank=None,
                    shape=None,
                )
            )
            scalar_index += 1
        else:
            raise ValueError(f"Unsupported input type {type(inp.__name__)}")
    return profiled


# TODO: Parses the LLVM IR to ge the kernel_name. This is limited to use cases where there is only one function
# hence, need to directly pass the filename.
def parse_triton_llvm_kernel_signature(llir, num_tensors, num_inputs):

    pattern = r"define (.+) @(\w+)\((.+)\)"
    matches = re.findall(pattern, llir)
    assert len(matches) == 1

    # Parse return type
    return_type = matches[0][0]
    assert return_type == "void"

    # Get kernel name
    kernel_name = matches[0][1]

    kernel_args = matches[0][2].split(",")
    # Assert that inputs tensors are unranked
    ptr_cnt = sum(1 for arg in kernel_args if "ptr" in arg)
    assert ptr_cnt == num_tensors

    return kernel_name


def make_profiled_return(iter_in):
    """Generate a profile for each return type"""
    profiled_return = namedtuple(
        "ReturnProfile",
        (
            "idx",  # Index of return type
            "dtype",  # CPP data type of the return type
            "rank",  # Rank of the return type (None for scalars)
        ),
    )
    return profiled_return(*iter_in)


def parse_return_types(return_type_list):

    return_types = []
    for idx, r in enumerate(return_type_list):
        rank = None if r[0] == 0 else r[0]
        dtype = str(r[1])

        return_types.append(make_profiled_return([idx, get_mlir_to_ctype(dtype), rank]))

    return return_types


def get_exec_mode():
    """
    Determines the expected execution target (device or simulator)
    which is required for generating the cpp wrapper and
    configuring the HexagonExecutor.
    Defaults to runnig the kernel on the device.
    """
    RUN_ON_SIM = get_env_var("RUN_ON_SIM", default="0")
    if RUN_ON_SIM == "0":
        return "device"
    elif RUN_ON_SIM == "1":
        return "simulator"
    else:
        raise ValueError(
            "Invalid value for RUN_ON_SIM. Set RUN_ON_SIM to 1 "
            "to run the kernel on the simulator and 0 to run it on the device."
        )
