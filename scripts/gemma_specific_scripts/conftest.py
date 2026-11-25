# ===- conftest.py ---------------------------------------------------------===
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause.
# For more license information:
#   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
#
# ===------------------------------------------------------------------------===

import os
import pytest


def pytest_addoption(parser):
    parser.addoption("--file_name", action="store", help="Save linalg IR to the file")
    parser.addoption(
        "--compile-only", action="store_true", help="Exit after compile step"
    )
    parser.addoption("--builder", action="store", help="Enter the builder running")
    parser.addoption(
        "--output-dir",
        type=str,
        help="Specify the output directory for the build artifacts subdirectory",
    )
    parser.addoption(
        "--enablelwp", action="store_true", help="Enable lightweight profiling"
    )
    parser.addoption(
        "--from-llvm",
        action="store_true",
        default=False,
        help="Start from a pre-lowered LLVM IR file instead of linalg MLIR",
    )
    parser.addoption(
        "--llvm-opt-level",
        type=int,
        default=3,
        help="LLVM optimization level for --from-llvm path (0-3, default 3)",
    )
    parser.addoption(
        "--stage",
        type=str,
        default=None,
        choices=["linalg2llvm", "llvm2obj", "llvm2so", "linalg2so"],
        help="Run a single compile stage on the single-layer module "
        "(linalg2llvm | llvm2obj | llvm2so | linalg2so), compile-only, with timing",
    )
    parser.addoption(
        "--from-obj",
        action="store_true",
        default=False,
        help="Start from a pre-compiled .o file; do obj->so link and optionally execute",
    )
    parser.addoption(
        "--from-so",
        action="store_true",
        default=False,
        help="Execute a .so already staged on device (skip all adb push). "
        "Set GEMMA4_DEVICE_DIR to the on-device path and GEMMA4_LOCAL_DIR to the "
        "local run directory where input .raw files and the .so live.",
    )
    parser.addoption(
        "--inputs-dir",
        type=str,
        default=None,
        help="Directory containing pre-generated _mlir_ciface_main_t{i}.raw files "
        "and manifest.json (from generate_inputs.py). When set, skips tokenizer "
        "and weight loading in --from-obj mode.",
    )
    parser.addoption(
        "--linalg", type=str, default=None, help="Path to the linalg .mlir file."
    )
    parser.addoption(
        "--weights",
        type=str,
        default=None,
        help="Path to the .safetensors weights file.",
    )
    parser.addoption(
        "--metadata",
        type=str,
        default=None,
        help="Path to the .weights_metadata.json file.",
    )
    parser.addoption(
        "--tokenizer-dir",
        type=str,
        default=None,
        help="Path to the tokenizer directory (fast_model_dir).",
    )
    parser.addoption(
        "--gemma-options",
        type=str,
        default=None,
        help="JSON file containing top-level Gemma options.",
    )


@pytest.fixture
def file_name(request):
    return request.config.getoption("--file_name")


@pytest.fixture
def compile_only(request):
    return request.config.getoption("--compile-only")


@pytest.fixture
def builder(request):
    return request.config.getoption("--builder")


@pytest.fixture
def base_dir_for_artifacts(request):
    return request.config.getoption("--output-dir")


@pytest.fixture
def enablelwp(request):
    return request.config.getoption("--enablelwp")


@pytest.fixture
def from_llvm(request):
    return request.config.getoption("--from-llvm")


@pytest.fixture
def llvm_opt_level(request):
    return request.config.getoption("--llvm-opt-level")


@pytest.fixture
def stage(request):
    return request.config.getoption("--stage")


@pytest.fixture
def from_obj(request):
    return request.config.getoption("--from-obj")


@pytest.fixture
def from_so(request):
    return request.config.getoption("--from-so")


@pytest.fixture
def inputs_dir(request):
    return request.config.getoption("--inputs-dir")


@pytest.fixture
def linalg_file(request):
    return request.config.getoption("--linalg")


@pytest.fixture
def weights_file(request):
    v = request.config.getoption("--weights")
    if v is None:
        v = os.environ.get("GEMMA4_WEIGHTS")
    return v


@pytest.fixture
def metadata_file(request):
    v = request.config.getoption("--metadata")
    if v is None:
        v = os.environ.get("GEMMA4_METADATA")
    return v


@pytest.fixture
def tokenizer_dir(request):
    v = request.config.getoption("--tokenizer-dir")
    if v is None:
        v = os.environ.get("TOKENIZER_DIR")
    return v


@pytest.fixture
def gemma_options_file(request):
    return request.config.getoption("--gemma-options")
