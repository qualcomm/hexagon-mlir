# ===- lit.cfg.py -----------------------------------------------------------===
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause.
# For more license information:
#   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
#
# ===------------------------------------------------------------------------===

# -*- Python -*-

import os

import lit.formats
import lit.util
from lit.llvm import llvm_config
from lit.llvm.subst import ToolSubst

# Configuration file for the 'lit' test runner

# name: The name of this test suite
config.name = "LINALG-HEXAGON"

config.test_format = lit.formats.ShTest(not llvm_config.use_lit_shell)

# suffixes: A list of file extensions to treat as test files.
config.suffixes = [".mlir", ".mllvm"]

config.substitutions.append(("%PATH%", config.environment["PATH"]))
config.substitutions.append(("%shlibext", config.llvm_shlib_ext))

llvm_config.with_system_environment(["HOME", "INCLUDE", "LIB", "TMP", "TEMP"])

# excludes: A list of directories to exclude from the testsuite. The 'Inputs'
# subdirectories contain auxiliary inputs for various tests in their parent
# directories.
config.excludes = [
    "CMakeLists.txt",
]

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(config.linalg_hexagon_obj_root, "test")
config.triton_tools_dir = os.path.join(config.linalg_hexagon_obj_root, "bin")
config.filecheck_dir = os.path.join(config.triton_obj_root, "bin", "FileCheck")
tool_dirs = [config.triton_tools_dir, config.llvm_tools_dir, config.filecheck_dir]

# Tweak the PATH to include the tools dir.
for d in tool_dirs:
    llvm_config.with_environment("PATH", d, append_path=True)
tools = [
    "linalg-hexagon-translate",
    ToolSubst("%PYTHON", config.python_executable, unresolved="ignore"),
]

llvm_config.add_tool_substitutions(tools, tool_dirs)

# TODO: what's this?
llvm_config.with_environment(
    "PYTHONPATH",
    [
        os.path.join(config.mlir_binary_dir, "python_packages", "triton"),
    ],
    append_path=True,
)

# Pass external env flags to lit test config
config.environment["HEXAGON_ARCH_VERSION"] = os.environ.get("HEXAGON_ARCH_VERSION")
