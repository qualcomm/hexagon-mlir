# ===- gemma_options.py ----------------------------------------------------===
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause.
# For more license information:
#   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
#
# ===------------------------------------------------------------------------===

"""Top-level Gemma options and phase-specific backend projections."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
from itertools import product
import json
import os
from pathlib import Path
from typing import ClassVar, Iterator


@dataclass(frozen=True)
class GemmaOptions:
    """Options owned by run_gemma.py and shared by every pipeline phase."""

    DEFAULT_VA_CHUNK_BYTES: ClassVar[int] = 256 * 1024 * 1024
    DEFAULT_VA_MIN_SIZE_FOR_FD: ClassVar[int] = 1024 * 1024
    DEFAULT_RIBBON_STACK_BYTES: ClassVar[int] = 16 * 1024 * 1024
    QUICK_COMPILE_LLVM_ARGS: ClassVar[tuple[str, ...]] = (
        "-enable-local-pull-up=false",
        "-hexagon-insert=false",
        "-hexagon-bit=false",
        "-rdf-opt=false",
        "-disable-hexagon-amodeopt",
    )

    enable_vtcm_tiling: bool = False
    enable_hvx_inlining: bool = False
    enable_double_buffering: bool = False
    enable_multi_threading: bool = False
    num_threads: int = 1
    enable_broadcasted_reduction_to_butterfly: bool = False
    outline_linalg_generic: bool = True
    fusion_elementwise_only: bool = True
    enable_convert_to_hexagonmem: bool = False
    enableHexagonmemCopyToDMA: bool = False
    enable_hexkl: bool = False
    relax_fp32_matmul_to_fp16: bool = False
    hexkl_mode: str = "micro"
    quick_compile: bool = False
    # Raw LLVM cl::opt arguments forwarded during object generation.
    # Example: "-enable-local-pull-up=false".
    hexagon_llvm_args: str = ""
    disable_buffer_dealloc: bool = False
    dealloc_after_last_use: bool = True
    dealloc_chunk_bytes: int = 0
    simple_extend_va: bool = True
    va_min_size_for_fd: int = DEFAULT_VA_MIN_SIZE_FOR_FD
    ribbon_stack_bytes: int = DEFAULT_RIBBON_STACK_BYTES
    va_chunk_bytes: int = DEFAULT_VA_CHUNK_BYTES

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        """Add all Gemma option groups to the driver's argument parser."""
        cls.add_optimization_arguments(parser)
        cls.add_debug_arguments(parser)
        cls.add_execution_arguments(parser)

    @classmethod
    def add_optimization_arguments(cls, parser: argparse.ArgumentParser) -> None:
        """Add optimization options to the driver's argument parser."""
        defaults = cls()
        optimization = parser.add_argument_group(
            "Optimization",
            "Code generation knobs to optimize performance.",
        )
        optimization.add_argument(
            "--enable-vtcm-tiling",
            action=argparse.BooleanOptionalAction,
            default=defaults.enable_vtcm_tiling,
            help="Enable VTCM tiling.",
        )
        optimization.add_argument(
            "--enable-hvx-inlining",
            action=argparse.BooleanOptionalAction,
            default=defaults.enable_hvx_inlining,
            help="Enable HVX inlining.",
        )
        optimization.add_argument(
            "--enable-double-buffering",
            action=argparse.BooleanOptionalAction,
            default=defaults.enable_double_buffering,
            help="Enable double buffering.",
        )
        optimization.add_argument(
            "--enable-multi-threading",
            action=argparse.BooleanOptionalAction,
            default=defaults.enable_multi_threading,
            help="Enable linalg-generic multi-threading.",
        )
        optimization.add_argument(
            "--num-threads",
            type=int,
            default=defaults.num_threads,
            help="Thread count used by compilation and execution. [default: 1]",
        )
        optimization.add_argument(
            "--enable-broadcasted-reduction-to-butterfly",
            action=argparse.BooleanOptionalAction,
            default=defaults.enable_broadcasted_reduction_to_butterfly,
            help="Lower broadcasted reductions to butterfly operations.",
        )
        optimization.add_argument(
            "--outline-linalg-generic",
            action=argparse.BooleanOptionalAction,
            default=defaults.outline_linalg_generic,
            help=(
                "Outline linalg.generic operations into noinline functions "
                "to reduce object-generation time."
            ),
        )
        optimization.add_argument(
            "--fusion-elementwise-only",
            action=argparse.BooleanOptionalAction,
            default=defaults.fusion_elementwise_only,
            help="Restrict fusion to elementwise operations.",
        )
        optimization.add_argument(
            "--enable-convert-to-hexagonmem",
            action=argparse.BooleanOptionalAction,
            default=defaults.enable_convert_to_hexagonmem,
            help="Enable conversion to hexagonmem operations.",
        )
        optimization.add_argument(
            "--enable-hexagonmem-copy-to-dma",
            "--enableHexagonmemCopyToDMA",
            dest="enableHexagonmemCopyToDMA",
            action=argparse.BooleanOptionalAction,
            default=defaults.enableHexagonmemCopyToDMA,
            help="Lower hexagonmem.copy operations to DMA operations.",
        )
        optimization.add_argument(
            "--enable-hexkl",
            action=argparse.BooleanOptionalAction,
            default=defaults.enable_hexkl,
            help="Enable HexKL lowering and linking.",
        )
        optimization.add_argument(
            "--relax-fp32-matmul-to-fp16",
            action=argparse.BooleanOptionalAction,
            default=defaults.relax_fp32_matmul_to_fp16,
            help="Relax FP32 matmul inputs to FP16 for HexKL lowering.",
        )
        optimization.add_argument(
            "--hexkl-mode",
            choices=("micro", "macro", "micro-daemon"),
            default=defaults.hexkl_mode,
            help="HexKL runtime mode.",
        )
        optimization.add_argument(
            "--hexagon-llvm-args",
            default=os.environ.get("HEXAGON_LLVM_ARGS", defaults.hexagon_llvm_args),
            help=(
                "Raw LLVM backend arguments for object generation. "
                "Example: --hexagon-llvm-args=-enable-local-pull-up=false"
            ),
        )

    @classmethod
    def add_debug_arguments(cls, parser: argparse.ArgumentParser) -> None:
        """Add debug and workflow options to the driver's argument parser."""
        defaults = cls()
        debug = parser.add_argument_group(
            "Debug",
            "Diagnostics and debugging controls.",
        )
        debug.add_argument(
            "--disable-buffer-dealloc",
            action=argparse.BooleanOptionalAction,
            default=defaults.disable_buffer_dealloc,
            help="Disable buffer deallocation during compilation.",
        )
        debug.add_argument(
            "--dealloc-after-last-use",
            action=argparse.BooleanOptionalAction,
            default=defaults.dealloc_after_last_use,
            help="Use the interprocedural deallocation pass.",
        )
        debug.add_argument(
            "--dealloc-chunk-bytes",
            type=human_bytes,
            default=defaults.dealloc_chunk_bytes,
            metavar="SIZE",
            help="Allocation size threshold for interprocedural deallocation.",
        )
        debug.add_argument(
            "--quick-compile",
            action=argparse.BooleanOptionalAction,
            default=defaults.quick_compile,
            help=(
                "Reduce LLVM object-generation time by disabling selected "
                "backend optimizations.\n"
                "-enable-local-pull-up=false: skip same-basic-block local "
                "instruction pull-up while retaining global scheduling and "
                "packetization.\n"
                "-hexagon-insert=false: skip Hexagon instruction insertion "
                "and cleanup.\n"
                "-hexagon-bit=false: skip Hexagon bit simplification.\n"
                "-rdf-opt=false: skip Hexagon RDF data-flow optimization.\n"
                "-disable-hexagon-amodeopt: skip addressing-mode optimization.\n"
                "These settings can reduce execution performance; use this "
                "for compile-time-focused builds."
            ),
        )
        debug.add_argument(
            "--chop-after-call",
            default="",
            help="Chop after an outlined call.",
        )
        debug.add_argument(
            "--chop-occurrence",
            type=int,
            default=1,
            help="Chop occurrence.",
        )
        debug.add_argument(
            "--dump-ir",
            action="store_true",
            help="Dump MLIR IR.",
        )
        debug.add_argument(
            "--skip-compile",
            action="store_true",
            help="Reuse an existing object.",
        )
        debug.add_argument(
            "--chop-debug",
            action="store_true",
            help="Debug a chopped object.",
        )
        debug.add_argument(
            "--skip-run",
            action="store_true",
            help="Compile only.",
        )
        debug.add_argument(
            "--record-perf",
            action="store_true",
            help="Record run and per-kernel timings to CSV after the run.",
        )
        debug.add_argument(
            "--record-perf-dir",
            type=Path,
            default=None,
            help="Record timings and write both CSVs to this directory, instead "
            "of appending the run summary to the shared CSV.",
        )
        debug.add_argument(
            "--record-perf-comment",
            default="",
            help="Free-text note recorded in the performance-summary comment "
            "column.",
        )

    @classmethod
    def debug_option_strings(cls) -> set[str]:
        """Return every command-line spelling belonging to the Debug group."""
        parser = argparse.ArgumentParser(add_help=False)
        cls.add_debug_arguments(parser)
        return {
            option for action in parser._actions for option in action.option_strings
        }

    @staticmethod
    def add_sweep_arguments(parser: argparse.ArgumentParser) -> None:
        """Add parameter-sweep controls to the driver's argument parser."""
        default_json = GemmaOptions.default_sweep_json_path()
        sweep = parser.add_argument_group(
            "Sweep",
            "Benchmark combinations of optimization flags.",
        )
        sweep.add_argument(
            "--sweep",
            action="store_true",
            help=f"Enable a sweep using {default_json}.",
        )
        sweep.add_argument(
            "--sweep-json",
            type=Path,
            default=None,
            help="Enable a sweep using the specified JSON file.",
        )
        sweep.add_argument(
            "--sweep-csv",
            type=Path,
            default=None,
            help=(
                "Sweep results CSV path. Defaults to "
                "<sweep-output-dir>/sweep_results.csv."
            ),
        )
        sweep.add_argument(
            "--sweep-output-dir",
            type=Path,
            default=None,
            help="Per-run artifacts directory. Defaults to "
            "/tmp/gemma_sweep_<timestamp>.",
        )
        sweep.add_argument(
            "--device-pool",
            action="store_true",
            help="Run sweep combinations concurrently using rented devices. "
            "[default: false]",
        )
        sweep.add_argument(
            "--num-devices",
            type=int,
            default=10,
            help="Number of rented devices. [default: 10]",
        )
        sweep.add_argument(
            "--timeout",
            type=int,
            default=120,
            metavar="MINUTES",
            help="Device rental timeout in minutes. [default: 120]",
        )
        sweep.add_argument(
            "--execution-timeout",
            type=int,
            default=1800,
            metavar="SECONDS",
            help="Maximum execution time per run in seconds. [default: 1800]",
        )
        sweep.add_argument(
            "--codename",
            default="kaanapali",
            help="Device codename passed to drental. [default: kaanapali]",
        )
        sweep.add_argument(
            "--drental",
            default=(
                "docker run --rm "
                "docker-registry.qualcomm.com/hexnn_infra/device-rental:default"
            ),
            help=(
                "Drental command used to acquire devices. [default: docker "
                "run --rm docker-registry.qualcomm.com/hexnn_infra/"
                "device-rental:default]"
            ),
        )

    @staticmethod
    def default_sweep_json_path() -> Path:
        root = os.environ.get("HEXAGON_MLIR_ROOT")
        if root:
            return Path(root) / "scripts/gemma_specific_scripts/sweep_options.json"
        return Path(__file__).resolve().parent / "sweep_options.json"

    @classmethod
    def add_execution_arguments(cls, parser: argparse.ArgumentParser) -> None:
        """Add execution options to the driver's argument parser."""
        defaults = cls()
        execution = parser.add_argument_group(
            "Execution",
            "Settings needed for end-to-end execution.",
        )
        execution.add_argument(
            "--simple-extend-va",
            action=argparse.BooleanOptionalAction,
            default=defaults.simple_extend_va,
            help="Use the simple fd-based VA path.",
        )
        execution.add_argument(
            "--va-min-size-for-fd",
            type=human_bytes,
            default=defaults.va_min_size_for_fd,
            metavar="SIZE",
            help="Minimum buffer size using the fd ABI. [default: 1MB]",
        )
        execution.add_argument(
            "--ribbon-stack",
            type=human_bytes,
            default=defaults.ribbon_stack_bytes,
            metavar="SIZE",
            help="Ribbon thread stack size and host stack limit. " "[default: 16MB]",
        )
        execution.add_argument(
            "--chunk-bytes",
            type=human_bytes,
            default=defaults.va_chunk_bytes,
            metavar="SIZE",
            help="VA and vMem chunk size used during compile and run. "
            "[default: 256MB]",
        )

    @staticmethod
    def add_setup_arguments(parser: argparse.ArgumentParser) -> None:
        """Add model, input, and device arguments used by run_gemma.py."""
        setup = parser.add_argument_group(
            "Setup",
            "Model paths, inputs, and device connection.",
        )
        setup.add_argument(
            "--gemma-root",
            type=Path,
            default=os.environ.get("GEMMA_ROOT"),
            help="Gemma model root; required for standard runs (env: GEMMA_ROOT).",
        )
        setup.add_argument(
            "--out-dir",
            type=Path,
            default=None,
            help="Artifact directory. Defaults to /tmp/gemma_<timestamp>.",
        )
        setup.add_argument(
            "--obj",
            type=Path,
            default=None,
            help="Kernel object path.",
        )
        setup.add_argument(
            "--inputs-dir",
            type=Path,
            default=None,
            help="Pre-generated input directory.",
        )
        setup.add_argument(
            "--linalg",
            type=Path,
            default=None,
            help="Input linalg module.",
        )
        setup.add_argument(
            "--weights",
            type=Path,
            default=None,
            help="Weights file.",
        )
        setup.add_argument(
            "--metadata",
            type=Path,
            default=None,
            help="Weights metadata file.",
        )
        setup.add_argument(
            "--tokenizer-dir",
            type=Path,
            default=None,
            help="Tokenizer directory.",
        )
        setup.add_argument(
            "--android-host",
            default=os.environ.get("ANDROID_HOST"),
            help="ADB host (env: ANDROID_HOST).",
        )
        setup.add_argument(
            "--android-serial",
            default=os.environ.get("ANDROID_SERIAL"),
            help="ADB serial (env: ANDROID_SERIAL).",
        )
        setup.add_argument(
            "--prompts",
            default="dog,cat",
            help="Comma-separated prompts.",
        )
        setup.add_argument(
            "--no-inputs",
            action="store_true",
            help="Use synthetic inputs.",
        )

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "GemmaOptions":
        """Build the shared options from arguments parsed by run_gemma.py."""
        return cls(
            enable_vtcm_tiling=args.enable_vtcm_tiling,
            enable_hvx_inlining=args.enable_hvx_inlining,
            enable_double_buffering=args.enable_double_buffering,
            enable_multi_threading=args.enable_multi_threading,
            num_threads=args.num_threads,
            enable_broadcasted_reduction_to_butterfly=(
                args.enable_broadcasted_reduction_to_butterfly
            ),
            outline_linalg_generic=args.outline_linalg_generic,
            fusion_elementwise_only=args.fusion_elementwise_only,
            enable_convert_to_hexagonmem=args.enable_convert_to_hexagonmem,
            enableHexagonmemCopyToDMA=args.enableHexagonmemCopyToDMA,
            enable_hexkl=args.enable_hexkl,
            relax_fp32_matmul_to_fp16=args.relax_fp32_matmul_to_fp16,
            hexkl_mode=args.hexkl_mode,
            quick_compile=args.quick_compile,
            hexagon_llvm_args=args.hexagon_llvm_args,
            disable_buffer_dealloc=args.disable_buffer_dealloc,
            dealloc_after_last_use=args.dealloc_after_last_use,
            dealloc_chunk_bytes=args.dealloc_chunk_bytes,
            simple_extend_va=args.simple_extend_va,
            va_min_size_for_fd=args.va_min_size_for_fd,
            ribbon_stack_bytes=args.ribbon_stack,
            va_chunk_bytes=args.chunk_bytes,
        )

    def effective_hexagon_llvm_args(self) -> str:
        """Return LLVM arguments after applying the quick-compile bundle."""
        if not self.quick_compile:
            return self.hexagon_llvm_args
        base_args = self.hexagon_llvm_args.strip()
        base_arg_tokens = set(base_args.split())
        return " ".join(
            arg
            for arg in (
                base_args,
                *(
                    arg
                    for arg in self.QUICK_COMPILE_LLVM_ARGS
                    if arg not in base_arg_tokens
                ),
            )
            if arg
        )

    @classmethod
    def load_optimization_sweep(cls, path: Path) -> dict[str, list[object]]:
        """Load and validate a JSON sweep over Optimization flags."""
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"Could not read sweep JSON {path}: {error}") from error

        if not isinstance(payload, dict) or not payload:
            raise ValueError("Sweep JSON must be a non-empty object.")

        optimization_actions = cls._optimization_actions()
        defaults = cls()
        values_by_field: dict[str, list[object]] = {}
        for field, values in payload.items():
            action = optimization_actions.get(field)
            if action is None:
                raise ValueError(
                    f"{field!r} is not an Optimization field supported by "
                    "GemmaOptions."
                )
            if not isinstance(values, list):
                values = [values]
            if not values:
                raise ValueError(
                    f"{field!r} must map to a non-empty JSON list or value."
                )

            expected_type = type(getattr(defaults, field))

            for value in values:
                if expected_type is bool and type(value) is not bool:
                    raise ValueError(f"{field!r} values must be JSON booleans.")
                if expected_type is int and (
                    type(value) is not int or isinstance(value, bool)
                ):
                    raise ValueError(f"{field!r} values must be JSON integers.")
                if expected_type is str and not isinstance(value, str):
                    raise ValueError(f"{field!r} values must be JSON strings.")
                choices = action.choices
                if choices is not None and value not in choices:
                    raise ValueError(
                        f"{field!r} value {value!r} is not one of {choices}."
                    )
            values_by_field[field] = values
        return values_by_field

    @classmethod
    def _optimization_actions(
        cls,
    ) -> dict[str, argparse.Action]:
        """Return argparse actions belonging to the Optimization group."""
        parser = argparse.ArgumentParser(add_help=False)
        cls.add_optimization_arguments(parser)
        return {
            action.dest: action for action in parser._actions if action.dest != "help"
        }

    @classmethod
    def optimization_sweep_combinations(cls, path: Path) -> Iterator[dict[str, object]]:
        """Yield every Cartesian-product combination in JSON key order."""
        values_by_field = cls.load_optimization_sweep(path)
        fields = list(values_by_field)
        for values in product(*(values_by_field[field] for field in fields)):
            yield dict(zip(fields, values))

    def with_optimization_sweep_values(
        self, values: dict[str, object]
    ) -> "GemmaOptions":
        """Return options with validated Optimization values applied."""
        return replace(self, **values)

    @staticmethod
    def optimization_sweep_arguments(combination: dict[str, object]) -> list[str]:
        """Convert a sweep combination into run_gemma.py arguments."""
        arguments: list[str] = []
        for field, value in combination.items():
            flag = "--" + field.replace("_", "-")
            if isinstance(value, bool):
                arguments.append(flag if value else f"--no-{flag[2:]}")
            elif field == "hexagon_llvm_args":
                # LLVM args begin with "-" and must stay attached to avoid
                # argparse treating them as separate options.
                arguments.append(f"{flag}={value}")
            else:
                arguments.extend([flag, str(value)])
        return arguments

    def compiler_options(self) -> dict[str, object]:
        """Return the options consumed by the compiler."""
        return {
            "enableConvertToHexagonmem": _bool_text(self.enable_convert_to_hexagonmem),
            "enableMultiThreading": _bool_text(self.enable_multi_threading),
            "enableBroadcastedReductionToButterfly": _bool_text(
                self.enable_broadcasted_reduction_to_butterfly
            ),
            "outlineLinalgGeneric": _bool_text(self.outline_linalg_generic),
            "disableBufferDealloc": _bool_text(self.disable_buffer_dealloc),
            "deallocAfterLastUse": _bool_text(self.dealloc_after_last_use),
            "deallocChunkBytes": self.dealloc_chunk_bytes,
            "fusionElementwiseOnly": _bool_text(self.fusion_elementwise_only),
            "enableHexagonmemCopyToDMA": _bool_text(self.enableHexagonmemCopyToDMA),
            "num_threads": self.num_threads,
            "enableVTCMTiling": _bool_text(self.enable_vtcm_tiling),
            "enableHVXInlining": _bool_text(self.enable_hvx_inlining),
            "enableDoubleBuffering": _bool_text(self.enable_double_buffering),
            "enableHexKL": _bool_text(self.enable_hexkl),
            "relaxFP32MatmulToFP16": _bool_text(self.relax_fp32_matmul_to_fp16),
            "hexKLMode": self.hexkl_mode,
            "simpleExtendVA": _bool_text(self.simple_extend_va),
            "vaChunkBytes": self.va_chunk_bytes,
            "vMemChunkBytes": self.va_chunk_bytes,
        }

    def link_options(self) -> dict[str, object]:
        """Return options needed when linking the compiled object."""
        return {"enableHexKL": self.enable_hexkl}

    def runtime_options(self) -> dict[str, object]:
        """Return options consumed by wrapper generation and execution."""
        return {
            "enableConvertToHexagonmem": self.enable_convert_to_hexagonmem,
            "enableMultiThreading": self.enable_multi_threading,
            "num_threads": self.num_threads,
            "enableHexKL": self.enable_hexkl,
            "hexKLMode": self.hexkl_mode,
            "simpleExtendVA": self.simple_extend_va,
            # Gemma's simpleExtendVA wrapper has its own bounded file-read
            # path; use the explicit Tensor chunked API only otherwise.
            "chunkedFileLoads": not self.simple_extend_va,
            "vaChunkBytes": self.va_chunk_bytes,
            "vaMinSizeForFd": self.va_min_size_for_fd,
        }

    def write(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=2, sort_keys=True) + "\n")

    @classmethod
    def from_file(cls, path: Path) -> "GemmaOptions":
        return cls(**json.loads(path.read_text()))

    @classmethod
    def from_object(
        cls,
        obj: Path,
        fallback: "GemmaOptions | None" = None,
    ) -> "GemmaOptions":
        """Load compile/run VA values from object sidecars when available."""
        fallback = cls() if fallback is None else fallback
        return replace(
            fallback,
            va_chunk_bytes=_read_int_sidecar(
                obj, "chunk_bytes", fallback.va_chunk_bytes
            ),
            va_min_size_for_fd=_read_int_sidecar(
                obj, "min_bytes", fallback.va_min_size_for_fd
            ),
        )


def human_bytes(text: str) -> int:
    """Parse a size like ``512MB`` / ``1g`` / ``1048576`` into bytes."""
    text = str(text).strip().lower()
    units = {
        "k": 1024,
        "kb": 1024,
        "m": 1024**2,
        "mb": 1024**2,
        "g": 1024**3,
        "gb": 1024**3,
        "b": 1,
        "": 1,
    }
    for suffix in ("kb", "mb", "gb", "k", "m", "g", "b"):
        if text.endswith(suffix):
            return int(float(text[: -len(suffix)]) * units[suffix])
    return int(text)


def _read_int_sidecar(obj: Path, suffix: str, default: int) -> int:
    sidecar = obj.with_suffix("." + suffix)
    if not sidecar.is_file():
        return default
    return int(sidecar.read_text().strip())


def _bool_text(value: bool) -> str:
    return "True" if value else "False"
