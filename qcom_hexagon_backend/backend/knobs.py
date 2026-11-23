# ===- knobs.py -------------------------------------------------------------===
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause.
# For more license information:
#   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
#
# ===------------------------------------------------------------------------===

import shutil
from pathlib import Path

from triton import knobs

# ===------------------------------------------------------------------------===
# GLOBALS
# ===------------------------------------------------------------------------===

# Path to the current directory in the binary wheel
_BACKEND_DIR = Path(__file__).resolve().parent

DEFAULT_TRITON_SHARED_OPT = _BACKEND_DIR / "bin" / "triton-shared-opt"

# Installed wheels do not require the original source tree; use the packaged
# runtime assets as the default root/libs dir when they are present.
_PACKAGED_HEXAGON_MLIR_ROOT = _BACKEND_DIR / "hexagon_mlir_root"
DEFAULT_HEXAGON_MLIR_ROOT = (
    _PACKAGED_HEXAGON_MLIR_ROOT if _PACKAGED_HEXAGON_MLIR_ROOT.exists() else None
)

DEFAULT_HEXAGON_MLIR_DUMP_DIR = Path("/tmp")


def _default_runtime_libs_dir() -> Path | None:
    # 1. Prefer the runtime dir next to linalg-hexagon-opt on PATH.
    # 2. Otherwise, fall back to the packaged runtime dir if present.
    linalg_hexagon_opt_path = shutil.which("linalg-hexagon-opt")
    if linalg_hexagon_opt_path is not None:
        return Path(linalg_hexagon_opt_path).resolve().parent / "runtime"
    packaged_runtime_libs_dir = _BACKEND_DIR / "bin" / "runtime"
    if packaged_runtime_libs_dir.exists():
        return packaged_runtime_libs_dir
    return None


DEFAULT_HEXAGON_RUNTIME_LIBS_DIR = _default_runtime_libs_dir()

# ===------------------------------------------------------------------------===
# ENVS
# ===------------------------------------------------------------------------===

env_str = knobs.env_str
env_int = knobs.env_int
env_bool = knobs.env_bool
env_opt_str = knobs.env_opt_str


class env_path(knobs.env_base[Path, Path]):
    def __init__(self, key: str, default: Path | None):
        super().__init__(key)
        self.default = default

    def get(self) -> Path:
        val = knobs.getenv(self.key)
        if val is not None:
            return Path(val)
        if self.default is None:
            raise RuntimeError(f"{self.key} is not set and has no default")
        return self.default


class env_opt_path(knobs.env_base[Path, Path]):
    def get(self) -> Path | None:
        val = knobs.getenv(self.key)
        if val is None:
            return None
        return Path(val)


class env_int_callable_default(knobs.env_base[int, int]):
    """Like `triton.knobs.env_str_callable_default`, but for ints."""

    def __init__(self, key: str, default_factory):
        super().__init__(key)
        self.default_factory = default_factory

    def get(self) -> int:
        val = knobs.getenv(self.key)
        if val is None:
            return self.default_factory()
        return int(val)


# ===------------------------------------------------------------------------===
# KNOBS
# ===------------------------------------------------------------------------===


class hexagon_knobs(knobs.base_knobs):
    tools: env_path = env_path("HEXAGON_TOOLS", None)
    sdk_root: env_path = env_path("HEXAGON_SDK_ROOT", None)
    mlir_root: env_path = env_path("HEXAGON_MLIR_ROOT", DEFAULT_HEXAGON_MLIR_ROOT)
    hexkl_root: env_opt_path = env_opt_path("HEXKL_ROOT")
    runtime_libs_dir: env_path = env_path(
        "HEXAGON_RUNTIME_LIBS_DIR", DEFAULT_HEXAGON_RUNTIME_LIBS_DIR
    )
    arch_version: env_str = env_str("HEXAGON_ARCH_VERSION", "68")
    tools_version: env_str = env_str("HEXAGON_TOOLS_VERSION", "")
    sdk_version: env_opt_str = env_opt_str("HEXAGON_SDK_VERSION")
    llvm_args: env_str = env_str("HEXAGON_LLVM_ARGS", "")
    enable_accuracy: env_bool = env_bool("HEXAGON_ENABLE_ACCURACY", False)
    run_on_sim: env_bool = env_bool("RUN_ON_SIM", False)
    android_host: env_opt_str = env_opt_str("ANDROID_HOST")
    android_serial: env_opt_str = env_opt_str("ANDROID_SERIAL")


class hexagon_mlir_knobs(knobs.base_knobs):
    dump_dir: env_path = env_path(
        "HEXAGON_MLIR_DUMP_DIR", DEFAULT_HEXAGON_MLIR_DUMP_DIR
    )
    logs: env_opt_str = env_opt_str("HEXAGON_MLIR_LOGS")
    # TODO(gleite): Deprecate LOG_LEVEL in the future, in favor of `logs` above.
    logs_deprecated: env_opt_str = env_opt_str("LOG_LEVEL")


class hexagon_profiler_knobs(knobs.base_knobs):
    pyetm: env_opt_path = env_opt_path("PYETM")
    cdsp_path: env_opt_path = env_opt_path("CDSP_PATH")


class triton_shared_knobs(knobs.base_knobs):
    disable_modulo_support: env_bool = env_bool(
        "TRITON_SHARED_DISABLE_MODULO_SUPPORT", True
    )
    opt_path: env_path = env_path("TRITON_SHARED_OPT_PATH", DEFAULT_TRITON_SHARED_OPT)


hexagon = hexagon_knobs()
hexagon_mlir = hexagon_mlir_knobs()
hexagon_profiler = hexagon_profiler_knobs()
triton_shared = triton_shared_knobs()
