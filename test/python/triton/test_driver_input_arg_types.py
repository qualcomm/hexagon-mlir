# ===- test_driver_input_arg_types.py --------------------------------------===
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause.
# For more license information:
#   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
#
# ===------------------------------------------------------------------------===
# Device-free regression tests for the launcher __call__ path in driver.py.
#
# Background: `getHexagonLauncherClass(...).__call__` builds `input_arg_types`,
# the per-input Triton signature dtypes that are threaded into the launcher so
# the generated wrapper ABI matches the compiled kernel (notably i64 scalars).
#
# `self.input_type_list` is a *dict* keyed by integer arg index (built from
# `src.signature` in __init__). A previous "cleanup" rewrote the list comp to
# slice it directly -- `self.input_type_list[: len(inputs_with_constants)]` --
# which uses an unhashable `slice` object as a dict key and raises
#   TypeError: unhashable type: 'slice'
# the moment any kernel is launched. That broke collection of every Triton
# kernel test (LE, masked-gather, bitonic sort) because those modules launch a
# kernel at import time. None of the existing wrapper-generator tests exercised
# this launcher path, so CI was the first to catch it. These tests pin the
# launcher __call__ behaviour directly.

from types import SimpleNamespace

import pytest

from triton.backends.qcom_hexagon_backend import driver as driver_mod
from triton.backends.qcom_hexagon_backend.driver import getHexagonLauncherClass
from triton.backends.qcom_hexagon_backend.utils import PackedMetadata


class _CapturingLauncher:
    """Stand-in for TritonHexagonLauncher that records _exec_kernel kwargs."""

    def __init__(self, *args, **kwargs):
        self.captured = None

    def _exec_kernel(
        self,
        kernel_llir,
        iterations,
        func_name,
        inputs,
        return_profs,
        launch_grid,
        **kwargs,
    ):
        self.captured = dict(kwargs)
        self.captured["inputs"] = inputs


def _make_src(arg_names, signature):
    fn = SimpleNamespace(arg_names=arg_names)
    return SimpleNamespace(fn=fn, signature=signature)


def _make_call_args(inputs_with_constants, grid=(40, 1, 1)):
    # __call__ reads: args[0:3]=grid, args[4]=llir, args[5]=pack_metadata,
    # then num_fixed_args=9, inputs = args[9:].
    # pack_metadata is read by name: only name, return_types and iterations
    # matter here; every other field is left as None.
    pack_metadata = PackedMetadata(
        **{field: None for field in PackedMetadata._fields}
    )._replace(
        name="abi_kernel",
        return_types=[],  # no returns -> no make_profiled_return calls
        iterations=1,
    )
    fixed = [
        grid[0],
        grid[1],
        grid[2],
        None,
        "llir_blob",
        pack_metadata,
        None,
        None,
        None,
    ]
    return tuple(fixed) + tuple(inputs_with_constants)


def _build_launcher(monkeypatch, arg_names, signature):
    # Force __init__ to build our capturing launcher instead of the real one.
    monkeypatch.setattr(driver_mod, "TritonHexagonLauncher", _CapturingLauncher)
    cls = getHexagonLauncherClass(device_type="dsp")
    cls._initialized = True
    src = _make_src(arg_names, signature)
    return cls(src, metadata=None)


def test_input_arg_types_filters_constexpr_and_preserves_i64(monkeypatch):
    # Signature: two fp32 pointers, an i64 scalar, and a constexpr block size.
    launcher = _build_launcher(
        monkeypatch,
        arg_names=["x", "y", "n", "BLOCK"],
        signature={"x": "*fp32", "y": "*fp32", "n": "i64", "BLOCK": "constexpr"},
    )

    # input_type_list must be a dict keyed by integer index -- this is what makes
    # the old `[: len(...)]` slice illegal.
    assert isinstance(launcher.input_type_list, dict)

    inputs_with_constants = [object(), object(), 320, 64]  # last is the constexpr
    launcher(*_make_call_args(inputs_with_constants))

    captured = launcher.launcher.captured
    assert captured is not None, "_exec_kernel was not invoked"
    # constexpr dropped; order preserved; i64 not truncated.
    assert captured["input_arg_types"] == ["*fp32", "*fp32", "i64"]
    # input_arg_types must align 1:1 with the (also constexpr-filtered) inputs.
    assert len(captured["input_arg_types"]) == len(captured["inputs"])


def test_input_arg_types_does_not_slice_the_dict(monkeypatch):
    # Regression for `TypeError: unhashable type: 'slice'`. A dict slice would
    # raise here; a correct integer-index comprehension must not.
    launcher = _build_launcher(
        monkeypatch,
        arg_names=["a", "b", "stride", "size"],
        signature={"a": "*fp32", "b": "*fp32", "stride": "i64", "size": "i32"},
    )

    inputs_with_constants = [object(), object(), 2560, 4]
    # Must not raise TypeError.
    launcher(*_make_call_args(inputs_with_constants))

    assert launcher.launcher.captured["input_arg_types"] == [
        "*fp32",
        "*fp32",
        "i64",
        "i32",
    ]


def test_all_constexpr_inputs_yield_empty_arg_types(monkeypatch):
    launcher = _build_launcher(
        monkeypatch,
        arg_names=["BLOCK_M", "BLOCK_N"],
        signature={"BLOCK_M": "constexpr", "BLOCK_N": "constexpr"},
    )

    launcher(*_make_call_args([16, 32]))

    captured = launcher.launcher.captured
    assert captured["input_arg_types"] == []
    assert captured["inputs"] == []
