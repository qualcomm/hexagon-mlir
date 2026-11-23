# ===- utils.py -------------------------------------------------------------===
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause.
# For more license information:
#   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
#
# ===------------------------------------------------------------------------===

import hashlib
import re
from collections import namedtuple
from pathlib import Path
from typing import Any, Literal, NamedTuple

try:
    import torch
except ImportError:
    torch = None  # type: ignore

from . import knobs


class PackedMetadata(NamedTuple):
    """Kernel metadata handed to the launcher on every call.

    Built by HexagonBackend.pack_metadata (compiler.py) and read by name in
    driver.py, so the two sides cannot disagree on positions. Values are taken
    as-is from the compiled metadata; some were stringified by
    ttsharedir_to_obj and are parsed by the launchers.
    """

    num_warps: Any
    num_ctas: Any
    shared: Any
    cluster_dim_x: Any
    cluster_dim_y: Any
    cluster_dim_z: Any
    name: Any
    return_types: Any
    iterations: Any
    scratch: Any
    enableMultiThreading: Any
    enableThreadedDispatch: Any
    enableLWP: Any
    enableHexKL: Any = None
    hexKLMode: Any = None
    num_threads: Any = None
    enableNSPSharding: Any = None
    nspCores: Any = None
    nspCount: Any = None
    nspThreads: Any = None


def split_path(path: str | Path) -> tuple[Path, str, str]:
    p = Path(path)
    return p.parent, p.stem, p.suffix


def to_torch_type(type_id):
    cpp_to_torch_map = {
        0: torch.bool,
        1: torch.float16,
        2: torch.float32,
        3: torch.float64,
        4: torch.int8,
        5: torch.int16,
        6: torch.int32,
        7: torch.int64,
        8: torch.uint8,
        11: torch.bfloat16,
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


def get_shape(shape, actual_strides=None):
    """Converts torch sizes (and optionally actual strides) to C-initializer strings.

    When actual_strides is provided (e.g. from tensor.stride()), those strides
    are used directly — necessary for non-contiguous tensors such as the result
    of reinterpret_tensor / as_strided.  When omitted, contiguous strides are
    computed from the shape (legacy path, safe only for contiguous tensors).
    """
    sizes = [dimension for dimension in shape]
    if actual_strides is not None:
        strides = list(actual_strides)
    else:
        strides = [1]
        for dim_shape in reversed(sizes[1:]):
            strides.insert(0, strides[0] * dim_shape)
    sizes = replace_list_brackets(sizes)
    strides = replace_list_brackets(strides)
    return sizes, strides


def _is_cpu_scalar_tensor(x) -> bool:
    """Return True for 0-d CPU tensors that Triton passes as scalar values."""
    return isinstance(x, torch.Tensor) and x.dim() == 0 and x.device.type == "cpu"


def _is_pointer_arg(sig_dtype) -> bool:
    return isinstance(sig_dtype, str) and sig_dtype.startswith("*")


def _is_scalar_arg(inp, sig_dtype=None) -> bool:
    return _is_cpu_scalar_tensor(inp) and not _is_pointer_arg(sig_dtype)


def _sig_dtype_at(arg_types, index):
    """Look up the Triton signature dtype for a given argument index.

    Returns None if arg_types is not provided or index is out of range.
    """
    if arg_types is not None and index < len(arg_types):
        return arg_types[index]
    return None


def kernel_entry_name(func_name: str, inputs, arg_types=None) -> str:
    """Return the C entry-point symbol for a compiled Triton kernel.

    The hash covers ``func_name`` plus every input's identity:
    - tensors: dtype, rank, shape+strides
    - scalars: their value (scalars are baked as constants into the wrapper)

    A rank-0 CPU tensor can be either a tensor or a scalar depending on the
    Triton signature.  If ``arg_types`` marks it as a pointer (e.g. '*fp32'),
    it is hashed as a tensor.  Otherwise it is hashed as a scalar via
    ``.item()``.  When ``arg_types`` is None, all rank-0 CPU tensors fall
    back to the legacy behavior and are treated as scalars.

    Inputs are sorted (tensors first, then scalars) before hashing so the
    entry name is stable regardless of the caller's argument order.

    ``kernel_entry_name_from_profiles()`` must also produce an identical
    hash — see that function's docstring for how the fields correspond.
    """
    h = hashlib.md5(func_name.encode("utf-8"))
    # 0-d CPU tensors are scalars only when Triton's signature does not mark the
    # argument as a pointer. Keep this in sync with profile_inputs().
    sorted_inputs = sorted(
        enumerate(inputs),
        key=lambda x: (
            0
            if (
                isinstance(x[1], torch.Tensor)
                and not _is_scalar_arg(x[1], _sig_dtype_at(arg_types, x[0]))
            )
            else 1
        ),
    )
    for index, inp in sorted_inputs:
        sig_dtype = _sig_dtype_at(arg_types, index)
        if isinstance(inp, torch.Tensor) and not _is_scalar_arg(inp, sig_dtype):
            # Keep this representation in sync with profile_inputs() so the
            # executor and generated wrapper derive the same entry symbol.
            h.update(str(get_ctype(inp.dtype)).encode("utf-8"))
            h.update(str(inp.dim()).encode("utf-8"))
            h.update(str(get_shape(inp.shape, inp.stride())).encode("utf-8"))
        elif _is_scalar_arg(inp, sig_dtype):
            h.update(str(inp.item()).encode("utf-8"))
        else:
            h.update(str(inp).encode("utf-8"))
    return "kernel_" + h.hexdigest()


def kernel_entry_name_from_profiles(func_name: str, input_profs) -> str:
    """Return an entry-point name from stable input-profile metadata.

    Tensor ``value`` is live runtime data and may be an opaque object by the
    time a profile reaches this function.  The wrapper signature is determined
    by the profile's type and layout information instead.

    IMPORTANT: This function MUST produce the same hash as
    ``kernel_entry_name()`` for the same inputs.  If ``profile_inputs()``
    changes how it populates dtype/rank/shape, or ``kernel_entry_name()``
    changes what it hashes, both functions must be updated together.
    """
    h = hashlib.md5(func_name.encode("utf-8"))
    # Match kernel_entry_name(), which places tensor arguments before scalars.
    for prof in sorted(
        input_profs, key=lambda prof: 0 if prof.input_type == "tensor" else 1
    ):
        if prof.input_type == "tensor":
            h.update(str(prof.dtype).encode("utf-8"))
            h.update(str(prof.rank).encode("utf-8"))
            h.update(str(prof.shape).encode("utf-8"))
        else:
            h.update(str(prof.value).encode("utf-8"))
    return "kernel_" + h.hexdigest()


def compute_kernel_content_hash(
    func_name: str,
    kernel_obj_as_bytes: bytes,
    input_profs,
    output_profs,
    options: dict,
    launch_grid: tuple,
) -> str:
    """Deterministic content hash for a compiled Triton kernel invocation.

    Used to name the build directory / shared object so that repeated calls with an
    identical kernel signature reuse the same artifacts instead of recompiling into a
    fresh timestamped directory every time (see create_hashed_folder in
    hexagon_launcher_base.py). Mirrors the field selection already used by
    kernel_entry_name() (func_name + per-input dtype/shape/stride) plus the compiled
    object bytes, output profiles, resolved options, and launch grid -- everything that
    can change the generated wrapper or compiled .so for an otherwise-identical
    func_name. Excludes each InputProfile's live `.value` (a torch.Tensor for tensor
    inputs, not a stable/deterministic hash input) in favor of its dtype/rank/shape.
    """
    h = hashlib.sha256(func_name.encode("utf-8"))
    h.update(hashlib.sha256(kernel_obj_as_bytes).digest())
    h.update(kernel_entry_name_from_profiles(func_name, input_profs).encode("utf-8"))

    for prof in input_profs:
        h.update(str(prof.dtype).encode("utf-8"))
        h.update(str(prof.rank).encode("utf-8"))
        h.update(str(prof.shape).encode("utf-8"))
        if prof.input_type != "tensor":
            # Scalars are baked as constants into the generated wrapper, so their
            # actual value (not just dtype) must participate in the hash.
            h.update(str(prof.value).encode("utf-8"))

    for prof in output_profs:
        h.update(str(prof.dtype).encode("utf-8"))
        h.update(str(prof.rank).encode("utf-8"))

    h.update(str(sorted(options.items())).encode("utf-8"))
    h.update(str(launch_grid).encode("utf-8"))

    return h.hexdigest()


def get_ctype(dtype):
    """Maps a torch data type to hexagon-clang compatible data types"""
    mapping = {
        torch.double: "float",
        torch.float: "float",
        torch.float64: "float",
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
        torch.bfloat16: "uint16_t",
        torch.float8_e4m3fnuz: "uint8_t",
        torch.float8_e4m3fn: "uint8_t",
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
        "f64": "float",
        "f32": "float",
        "f16": "_Float16",
        "i32": "int32_t",
        "i64": "int64_t",
        "i16": "int16_t",
        "i8": "int8_t",
        # MLIR i1 tensors are byte-addressable at the C ABI boundary.  Do not
        # leave this unmapped (or confuse it with i8): bool is one byte on the
        # Hexagon toolchain and preserves the logical boolean type.
        "i1": "bool",
        "b8": "bool",
    }

    return mapping.get(
        dtype, "unknown_type"
    )  # returns value of key-value pair if found in mapping, else returns "unknown_type"


# Triton signature dtype string -> hexagon-clang C type. Used to type scalar
# kernel arguments so the generated wrapper's ABI matches the compiled kernel.
# Critically, this preserves 64-bit scalars (i64/u64) instead of inferring the
# width from the host Python type (which is always "int" and would silently
# truncate to 32 bits, shifting every subsequent kernel argument).
# Note: Triton uses "fp32"/"fp64" notation; MLIR uses "f32"/"f64". Both are
# present here because Triton signatures can contain either form depending on
# the compiler stage. get_mlir_to_ctype() handles MLIR-only keys separately.
# i1 scalar arguments intentionally use int32_t: the generated device entry
# point follows the target ABI's integer argument extension rule. This is
# different from i1 tensor storage, which is one byte and maps to bool above.
_TRITON_SCALAR_CTYPE = {
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
    "fp16": "_Float16",
    "bf16": "uint16_t",
    "fp32": "float",
    "f32": "float",
    "fp64": "float",
    "f64": "float",
}


def scalar_ctype_from_signature(sig_dtype, value):
    """Resolve the C type for a scalar kernel argument.

    Prefers the Triton signature dtype (e.g. "i64") so 64-bit scalars keep
    their width. Falls back to the host Python type name when no signature
    dtype is available (legacy behavior).
    """
    if isinstance(sig_dtype, str):
        ctype = _TRITON_SCALAR_CTYPE.get(sig_dtype.lstrip("*"))
        if ctype is not None:
            return ctype
    return type(value).__name__


def profile_triton_inputs(inputs, arg_types=None):
    return profile_inputs(inputs, arg_types)


def profile_torch_mlir_inputs(inputs, arg_types=None):
    return profile_inputs(inputs, arg_types)


# Given an list of input tensors, to return the profile inputs by categorizing the inputs
# into various inputs types and deriving related metadata.
def profile_inputs(inputs, arg_types=None, rank_zero_tensors_as_scalars=True):
    """Build input profiles, optionally preserving rank-zero tensor operands."""
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
        sig_dtype = _sig_dtype_at(arg_types, index)
        is_cpu_scalar_tensor = rank_zero_tensors_as_scalars and _is_scalar_arg(
            inp, sig_dtype
        )
        is_py_scalar = isinstance(inp, (int, float))
        if is_cpu_scalar_tensor or is_py_scalar:
            scalar_val = inp.item() if is_cpu_scalar_tensor else inp
            profiled.append(
                Profiled_input(
                    input_type="scalar",
                    idx=scalar_index,
                    input_id=index,
                    value=scalar_val,
                    dtype=scalar_ctype_from_signature(sig_dtype, scalar_val),
                    rank=None,
                    shape=None,
                )
            )
            scalar_index += 1
        elif isinstance(inp, torch.Tensor):
            profiled.append(
                Profiled_input(
                    input_type="tensor",
                    idx=tensor_index,
                    input_id=index,
                    value=inp.data,
                    dtype=get_ctype(inp.dtype),
                    rank=inp.dim(),
                    shape=get_shape(inp.shape, inp.stride()),
                )
            )
            tensor_index += 1
        else:
            raise ValueError(f"Unsupported input type {type(inp).__name__}")
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


def get_exec_mode() -> Literal["simulator", "device"]:
    """
    Determines the expected execution target (device or simulator)
    which is required for generating the cpp wrapper and
    configuring the HexagonExecutor.
    Defaults to runnig the kernel on the device.
    """
    return "simulator" if knobs.hexagon.run_on_sim else "device"
