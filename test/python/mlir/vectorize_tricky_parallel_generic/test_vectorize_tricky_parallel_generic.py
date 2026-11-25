# ===- test_vectorize_tricky_parallel_generic.py -----------------------------===
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


def extract_perf_from_latest_folder(kernel_name):
    folders = glob.glob(f"/tmp/{kernel_name}-*")
    if not folders:
        return None
    latest_folder = max(folders, key=os.path.getmtime)
    perf_file = os.path.join(latest_folder, f"lib{kernel_name}_perf.txt")
    if not os.path.exists(perf_file):
        return None
    with open(perf_file, "r") as f:
        content = f.read()
    match = re.search(r"Perf:(\d+\.?\d*)", content)
    return float(match.group(1)) if match else None


def env_flag(name, default=True):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() not in ("0", "false", "off", "no")


def run_case(
    enable_vectorization, enable_loop_splitting=True, eager_vectorization=False
):
    kernel_name = "vectorize_tricky_parallel_generic"
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    # Default to VTCM tiling disabled unless explicitly enabled.
    enable_vtcm_tiling = env_flag("HEXAGON_ENABLE_VTCM_TILING", False)

    run_mlir_kernel_test(
        kernel_name=kernel_name,
        current_file_dir=current_file_dir,
        options_overrides={
            "enablePackedGenericVectorization": enable_vectorization,
            "enableVTCMTiling": enable_vtcm_tiling,
            "enablePackedGenericLoopSplitting": enable_loop_splitting,
            "enablePackedGenericEagerVectorization": eager_vectorization,
        },
        validate_result=True,
        enable_etm=False,
    )
    return extract_perf_from_latest_folder(kernel_name)


def test_mlir_suite_enabled():
    print("Running tricky parallel generic with packed-generic vectorization enabled")
    perf = run_case(True)
    if perf is not None:
        print(f"Enabled perf: {perf} us")


def test_mlir_suite_disabled():
    print("Running tricky parallel generic with packed-generic vectorization disabled")
    perf = run_case(False)
    if perf is not None:
        print(f"Disabled perf: {perf} us")


def test_mlir_suite_no_loop_splitting():
    print("Running tricky parallel generic with loop splitting disabled")
    perf = run_case(True, enable_loop_splitting=False)
    if perf is not None:
        print(f"No loop splitting perf: {perf} us")


def test_mlir_suite_eager_vectorization():
    print("Running tricky parallel generic with eager vectorization")
    perf = run_case(True, eager_vectorization=True)
    if perf is not None:
        print(f"Eager vectorization perf: {perf} us")


def test_mlir_suite_eager_and_no_split():
    print(
        "Running tricky parallel generic with eager vectorization + no loop splitting"
    )
    perf = run_case(True, enable_loop_splitting=False, eager_vectorization=True)
    if perf is not None:
        print(f"Eager + no split perf: {perf} us")
