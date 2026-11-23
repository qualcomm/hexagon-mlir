# ===- test_backend_utils.py -----------------------------------------------===
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause.
# For more license information:
#   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
#
# ===------------------------------------------------------------------------===


import pytest
import torch
from triton.backends.qcom_hexagon_backend.hexagon_options import HexagonOptions
from triton.backends.qcom_hexagon_backend.triton_hexagon_launcher import (
    TritonHexagonWrapperGenerator,
)
from triton.backends.qcom_hexagon_backend.utils import (
    _TRITON_SCALAR_CTYPE,
    get_ctype,
    get_mlir_to_ctype,
)


def test_get_ctype_bfloat16():
    assert get_ctype(torch.bfloat16) == "uint16_t"


# --- f64/fp64 → float (Hexagon NSP has no native f64 arithmetic) ---


def test_get_ctype_float64_maps_to_float():
    assert get_ctype(torch.float64) == "float"


def test_get_ctype_double_maps_to_float():
    # torch.double is an alias for torch.float64
    assert get_ctype(torch.double) == "float"


def test_get_mlir_to_ctype_f64_maps_to_float():
    assert get_mlir_to_ctype("f64") == "float"


def test_triton_scalar_ctype_fp64_maps_to_float():
    # Triton uses "fp64" notation in kernel signatures
    assert _TRITON_SCALAR_CTYPE["fp64"] == "float"


def test_triton_scalar_ctype_f64_maps_to_float():
    # MLIR uses "f64" notation in kernel signatures
    assert _TRITON_SCALAR_CTYPE["f64"] == "float"


# Sanity: other float widths are unchanged
def test_get_ctype_float32_unchanged():
    assert get_ctype(torch.float32) == "float"


def test_get_ctype_float16_unchanged():
    assert get_ctype(torch.float16) == "_Float16"


def _make_wrapper(*, scratch=0, grid=(1, 1, 1), **option_overrides):
    options = HexagonOptions().__dict__.copy()
    options.update(
        enableHexKL=True,
        hexKLMode="micro-daemon",
        scratch=scratch,
        **option_overrides,
    )
    return TritonHexagonWrapperGenerator([], 1, "kernel", [], grid, options)


def test_micro_daemon_without_hvx_scratch_releases_vtcm_and_checks_config_size():
    wrapper = _make_wrapper()

    assert "uint32_t hexkl_micro_hmx_config_size(void);" in (
        wrapper.common_strings.micro_daemon_code_headers
    )
    declarations = wrapper.generate_input_declarations()
    assert "hmx_config_bytes = hexkl_micro_hmx_config_size()" in declarations
    assert "hmx_config_bytes > 16384u" in declarations
    assert "+ 2047) & ~(int64_t)2047" in declarations
    assert wrapper.generate_cleanup().endswith(
        "HAP_compute_res_release(vtcmContextID);\n"
    )


def test_micro_daemon_hvx_scratch_composes_vtcm_abi():
    wrapper = _make_wrapper(scratch=65536)

    signature = wrapper.generate_llvm_function_signature()
    assert "int8_t *, int8_t *, int64_t, int64_t, int64_t" in signature
    declarations = wrapper.generate_input_declarations()
    assert "hvx_start_bytes" in declarations
    assert "int8_t *hvxBase = vtcmBase + hvx_start_bytes;" in declarations


def test_lwp_guard_uses_effective_hvx_worker_count():
    _make_wrapper(enableLWP=True, enableThreadedDispatch=True, num_threads=1)

    with pytest.raises(ValueError, match="effective HVX worker count is 2"):
        _make_wrapper(
            grid=(2, 1, 1),
            enableLWP=True,
            enableThreadedDispatch=True,
            num_threads=2,
        )
