# ===- test_vectorize_sum_of_squares.py ------------------------------------===
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause.
# For more license information:
#   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
#
# ===------------------------------------------------------------------------===


import os
import sys
import re
import glob

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from mlir_test_utils import run_mlir_kernel_test


def env_flag(name, default=True):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() not in ("0", "false", "off", "no")


def extract_perf(kernel_name):
    folders = glob.glob(f"/tmp/{kernel_name}-*")
    if not folders:
        return None
    latest = max(folders, key=os.path.getmtime)
    perf_file = os.path.join(latest, f"lib{kernel_name}_perf.txt")
    if not os.path.exists(perf_file):
        return None
    content = open(perf_file).read()
    match = re.search(r"Perf:(\d+\.?\d*)", content)
    return float(match.group(1)) if match else None


def run_case(
    enable_packed_generic_vectorization,
    enable_loop_splitting=True,
    eager_vectorization=False,
):
    kernel_name = "vectorize_sum_of_squares"
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    enable_vtcm_tiling = env_flag("HEXAGON_ENABLE_VTCM_TILING", False)
    run_mlir_kernel_test(
        kernel_name=kernel_name,
        current_file_dir=current_file_dir,
        options_overrides={
            "enablePackedGenericVectorization": enable_packed_generic_vectorization,
            "enableVTCMTiling": enable_vtcm_tiling,
            "enablePackedGenericLoopSplitting": enable_loop_splitting,
            "enablePackedGenericEagerVectorization": eager_vectorization,
        },
        validate_result=True,
        enable_etm=False,
    )
    perf = extract_perf(kernel_name)
    return perf


def test_with_flag_enabled():
    perf = run_case(enable_packed_generic_vectorization=True)
    if perf is not None:
        print(f"Perf (vectorization enabled): {perf} us")


def test_with_flag_disabled():
    perf = run_case(enable_packed_generic_vectorization=False)
    if perf is not None:
        print(f"Perf (vectorization disabled): {perf} us")


def test_with_no_loop_splitting():
    perf = run_case(
        enable_packed_generic_vectorization=True, enable_loop_splitting=False
    )
    if perf is not None:
        print(f"Perf (no loop splitting): {perf} us")


def test_with_eager_vectorization():
    perf = run_case(enable_packed_generic_vectorization=True, eager_vectorization=True)
    if perf is not None:
        print(f"Perf (eager vectorization): {perf} us")


def test_with_eager_and_no_split():
    perf = run_case(
        enable_packed_generic_vectorization=True,
        enable_loop_splitting=False,
        eager_vectorization=True,
    )
    if perf is not None:
        print(f"Perf (eager + no split): {perf} us")
