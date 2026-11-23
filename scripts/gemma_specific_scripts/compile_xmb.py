#!/usr/bin/env python3

# ===- compile_xmb.py ------------------------------------------------------===
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause.
# For more license information:
#   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
#
# ===------------------------------------------------------------------------===

"""Compile a Gemma linalg module into a Hexagon object file."""

import argparse
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
import os
from pathlib import Path
import re
import subprocess
import sys
import time

_HEXAGON_MLIR_ROOT = os.environ.get("HEXAGON_MLIR_ROOT")
_TRITON_PYTHON_DIR = (
    Path(_HEXAGON_MLIR_ROOT) / "triton" / "python" if _HEXAGON_MLIR_ROOT else None
)
if _TRITON_PYTHON_DIR is not None and _TRITON_PYTHON_DIR.is_dir():
    sys.path.insert(0, str(_TRITON_PYTHON_DIR))

os.environ.setdefault("MLIR_ENABLE_DUMP", "0")

if __package__:
    from .gemma_options import GemmaOptions
else:
    from gemma_options import GemmaOptions

from triton._C.libtriton import ir, qcom_hexagon_backend
from triton.backends.qcom_hexagon_backend.compiler import HexagonOptions


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--linalg",
        type=Path,
        required=True,
        help="Path to the linalg .mlir to compile",
    )
    parser.add_argument(
        "--gemma-options",
        type=Path,
        help="JSON file written by run_gemma.py",
    )
    args, _unknown_args = parser.parse_known_args(argv)
    if not args.linalg.is_file():
        parser.error(f"linalg file not found: {args.linalg}")
    return args


def load_gemma_options(options_path: Path | None):
    if options_path is None:
        return GemmaOptions()

    options = GemmaOptions.from_file(options_path)
    os.environ["HEXAGON_LLVM_ARGS"] = options.effective_hexagon_llvm_args()
    return options


def bytes_tag(byte_count: int) -> str:
    if byte_count % (1024 * 1024) == 0:
        return f"{byte_count // (1024 * 1024)}MB"
    if byte_count % 1024 == 0:
        return f"{byte_count // 1024}KB"
    return f"{byte_count}B"


def find_return_type(linalg_path: Path, func_name: str) -> str | None:
    if not func_name:
        return None

    pattern = re.compile(
        r"func\.func\s+(?:private|public)?\s*@"
        + re.escape(func_name)
        + r"\s*\([^)]*\)\s*->\s*([^\s{]+)"
    )
    with linalg_path.open() as linalg_file:
        for line in linalg_file:
            match = pattern.search(line)
            if match:
                return match.group(1).rstrip(",")
    return None


def require_output_dir() -> Path:
    output_dir = os.environ.get("GEMMA4_OUT_DIR")
    if not output_dir:
        raise RuntimeError("GEMMA4_OUT_DIR needs to be in the environment")

    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_output_object(
    output_dir: Path, chop_after_call: str, object_path: str | None
) -> Path:
    chop_tag = (
        f"_chopped{chop_after_call.split('_')[-1]}" if chop_after_call else "_fullmodel"
    )
    default_path = output_dir / f"gemma_feo_nomt{chop_tag}_vmem.o"
    return Path(object_path) if object_path else default_path


def build_compiler_options(hexagon_options_type, gemma_options) -> dict[str, str]:
    options = hexagon_options_type().__dict__.copy()
    options.update(gemma_options.compiler_options())
    # Keep the option available for the deferred simpleExtendVA compiler PR.
    options["simpleExtendVAMapAllocs"] = False
    return {key: str(value) for key, value in options.items()}


def write_compilation_metadata(
    output_object: Path,
    linalg_path: Path,
    chop_after_call: str,
    gemma_options,
    tag: str,
) -> None:
    output_object.parent.mkdir(parents=True, exist_ok=True)
    output_object.with_suffix(".chunk_bytes").write_text(
        f"{gemma_options.va_chunk_bytes}\n"
    )
    output_object.with_suffix(".min_bytes").write_text(
        f"{gemma_options.va_min_size_for_fd}\n"
    )
    print(
        f"{tag} chunk={bytes_tag(gemma_options.va_chunk_bytes)}  "
        f"chop=@{chop_after_call}  out={output_object.name}",
        flush=True,
    )

    return_type = find_return_type(linalg_path, chop_after_call)
    if return_type:
        output_object.with_suffix(".return_type").write_text(return_type + "\n")
        print(f"{tag} return type: {return_type}", flush=True)
    elif chop_after_call:
        print(
            f"{tag} Warning: return type not found for @{chop_after_call}",
            flush=True,
        )


@contextmanager
def capture_compile_logs(output_dir: Path, dump_ir: bool) -> Iterator[Path | None]:
    stdout_path = output_dir / "compile_stdout.log"
    stderr_path = output_dir / "compile_irdump.mlir" if dump_ir else None
    original_stdout = sys.stdout
    original_stderr = sys.stderr

    with ExitStack() as stack:
        stdout_log = stack.enter_context(open(stdout_path, "w", buffering=1))
        stderr_log = (
            stack.enter_context(open(stderr_path, "w", buffering=1))
            if stderr_path is not None
            else None
        )
        saved_stdout_fd = os.dup(1)
        saved_stderr_fd = os.dup(2) if stderr_log is not None else None
        try:
            os.dup2(stdout_log.fileno(), 1)
            sys.stdout = stdout_log
            if stderr_log is not None:
                os.dup2(stderr_log.fileno(), 2)
                sys.stderr = stderr_log
            yield stderr_path
        finally:
            stdout_log.flush()
            if stderr_log is not None:
                stderr_log.flush()
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            try:
                os.dup2(saved_stdout_fd, 1)
                if saved_stderr_fd is not None:
                    os.dup2(saved_stderr_fd, 2)
            finally:
                os.close(saved_stdout_fd)
                if saved_stderr_fd is not None:
                    os.close(saved_stderr_fd)


def compile_linalg(
    linalg_path: Path,
    options: dict[str, str],
    qcom_hexagon_backend,
    ir,
    tag: str,
) -> tuple[bytes, float, int]:
    print(f"{tag} Parsing {linalg_path.name}...", flush=True)
    context = ir.context()
    qcom_hexagon_backend.load_dialects(context)
    module = qcom_hexagon_backend.parse_mlir_module_from_file(str(linalg_path), context)

    print(f"{tag} Compiling...", flush=True)
    start_time = time.monotonic()
    object_modules = qcom_hexagon_backend.translate_linalg_to_obj(module, options)
    elapsed = time.monotonic() - start_time
    if not object_modules:
        raise RuntimeError("Hexagon backend produced no object modules")
    return object_modules[0], elapsed, len(object_modules)


def write_object_file(output_object: Path, object_code: bytes, tag: str) -> None:
    output_object.write_bytes(object_code)
    size_mb = output_object.stat().st_size / 1024 / 1024
    print(
        f"{tag} Written: {output_object}  ({size_mb:.1f} MB)",
        flush=True,
    )
    print(f"{tag} DONE", flush=True)


def extract_ir_snapshots(stderr_path: Path | None, output_dir: Path) -> None:
    if stderr_path is None:
        return

    extractor = Path(__file__).parent / "extract_snippet.py"
    subprocess.run(
        [
            sys.executable,
            str(extractor),
            str(stderr_path),
            "--output-dir",
            str(output_dir),
        ],
        check=False,
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if _TRITON_PYTHON_DIR is None:
        raise RuntimeError("HEXAGON_MLIR_ROOT needs to be in the environment")
    if not _TRITON_PYTHON_DIR.is_dir():
        raise RuntimeError(f"Triton Python sources not found: {_TRITON_PYTHON_DIR}")

    dump_ir = os.environ["MLIR_ENABLE_DUMP"] != "0"
    gemma_options = load_gemma_options(args.gemma_options)

    output_dir = require_output_dir()
    chop_after_call = os.environ.get("CHOP_AFTER_CALL", "__outlined_generic_1000")
    output_object = resolve_output_object(
        output_dir, chop_after_call, os.environ.get("GEMMA4_OBJ_PATH")
    )
    tag = f"[{bytes_tag(gemma_options.va_chunk_bytes)}]"
    options = build_compiler_options(HexagonOptions, gemma_options)

    print(f"Linalg is {args.linalg}")
    with capture_compile_logs(output_dir, dump_ir) as stderr_path:
        write_compilation_metadata(
            output_object,
            args.linalg,
            chop_after_call,
            gemma_options,
            tag,
        )
        object_code, elapsed, module_count = compile_linalg(
            args.linalg, options, qcom_hexagon_backend, ir, tag
        )
        print(
            f"{tag} Done in {elapsed:.1f}s  ({module_count} module(s))",
            flush=True,
        )
        write_object_file(output_object, object_code, tag)

    extract_ir_snapshots(stderr_path, output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
