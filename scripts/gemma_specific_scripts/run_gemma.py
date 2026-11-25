#!/usr/bin/env python3

# ===- run_gemma.py --------------------------------------------------------===
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause.
# For more license information:
#   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
#
# ===------------------------------------------------------------------------===

"""
Single entry point for building and running a GemmaV4 model on Hexagon.

Given a ``--gemma-root`` this script derives model dependencies, wires up
compile/exec arguments via a shared options file, then:

    1. compiles the linalg .mlir into a kernel .o   (compile_xmb.py)
    2. captures device logcat and runs the kernel via pytest
       (test_gemma_from_obj.py --from-obj)

Typical use when the device is already configured::

    ./run_gemma.py --gemma-root /path/to/gemma4/model/deps

All other args have a usable default that can be edited for debug.
Run ``./run_gemma.py --help`` for the list.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import os
import resource
import subprocess
import sys
import textwrap
from pathlib import Path

from gemma_options import GemmaOptions
from record_perf import record_run
from sweep import resolve_sweep_args, run_parameter_sweep

SCRIPT_DIR = Path(__file__).resolve().parent


class _GemmaHelpFormatter(argparse.RawDescriptionHelpFormatter):
    """Preserve intentional option-help line breaks without losing wrapping."""

    def _split_lines(self, text, width):
        lines = []
        for line in text.splitlines():
            if not line:
                lines.append("")
                continue
            lines.extend(
                textwrap.wrap(
                    line,
                    width,
                    break_long_words=False,
                    break_on_hyphens=False,
                )
            )
        return lines


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="run_gemma.py",
        usage="%(prog)s --gemma-root GEMMA_ROOT [options]",
        description="Compile and run Gemma on Hexagon.",
        formatter_class=_GemmaHelpFormatter,
        epilog=(
            "Standard run:\n"
            "  run_gemma.py --gemma-root <path>\n"
            "\n"
            "--gemma-root is the only required argument for a standard run. "
            "It may also be set with GEMMA_ROOT."
        ),
    )

    GemmaOptions.add_setup_arguments(p)
    GemmaOptions.add_optimization_arguments(p)
    GemmaOptions.add_debug_arguments(p)
    GemmaOptions.add_execution_arguments(p)
    GemmaOptions.add_sweep_arguments(p)

    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = resolve_sweep_args(p, p.parse_args(raw_argv), raw_argv)

    # In --chop-debug mode the linalg is a synthetic stub derived from the .o's
    # .return_type sidecar, so --gemma-root/--linalg are not required.
    if (
        not args.chop_debug
        and args.gemma_root is None
        and (args.linalg is None or args.weights is None)
    ):
        p.error("--gemma-root is required (or pass --linalg/--weights/... explicitly)")

    if not args.skip_run and not args.device_pool:
        if not args.android_host:
            p.error(
                "--android-host is required "
                "(pass it directly or set the ANDROID_HOST environment variable)"
            )
        if not args.android_serial:
            p.error(
                "--android-serial is required "
                "(pass it directly or set the ANDROID_SERIAL environment variable)"
            )
    return args


def resolve_paths(args: argparse.Namespace) -> argparse.Namespace:
    """Fill in every path default relative to --gemma-root."""
    root = args.gemma_root

    if args.out_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.out_dir = Path("/tmp") / f"gemma_{timestamp}"

    if root is not None:
        args.linalg = args.linalg or root / "gemma.linalg"
        args.weights = args.weights or root / "weights.safetensors"
        args.metadata = args.metadata or root / "weights_metadata.json"
        args.tokenizer_dir = args.tokenizer_dir or root / "tokenizer"

    args.obj = args.obj or args.out_dir / "gemma_kernel.o"
    args.inputs_dir = args.inputs_dir or args.out_dir

    # --chop-debug synthesizes a stub linalg from the .o's sidecar (done in
    # run_on_device), and the stub run only needs the return signature, so the
    # weights/metadata/tokenizer paths are not required.
    required = ("linalg", "weights", "metadata", "tokenizer_dir")
    if args.chop_debug:
        required = ()
    missing = [n for n in required if getattr(args, n) is None]
    if missing:
        raise SystemExit(
            "Could not resolve: "
            + ", ".join(missing)
            + " (pass them explicitly or set --gemma-root)"
        )
    return args


def _apply_driver_overrides(options: GemmaOptions) -> GemmaOptions:
    """Apply fixed driver policy after command-line parsing."""
    # Keep driver-specific overrides here so they are visible before the
    # shared options are handed to compile_xmb.py and pytest.
    return options


def run_compile(
    args: argparse.Namespace,
    gemma_options_path: Path,
    gemma_options: GemmaOptions,
) -> None:
    print(f"=== Compiling {args.linalg} -> {args.obj} ===", flush=True)
    env = os.environ.copy()
    env["GEMMA4_OUT_DIR"] = str(args.out_dir)
    env["GEMMA4_OBJ_PATH"] = str(args.obj)
    env["CHOP_AFTER_CALL"] = args.chop_after_call
    env["CHOP_OCCURRENCE"] = str(args.chop_occurrence)
    env["MLIR_ENABLE_DUMP"] = "1" if args.dump_ir else "0"
    # Keep LLVM backend flags in the compile artifact's options and child env.
    # This makes cache keys and reruns reproducible.
    env["HEXAGON_LLVM_ARGS"] = gemma_options.effective_hexagon_llvm_args()

    subprocess.run(
        [
            sys.executable,
            str(SCRIPT_DIR / "compile_xmb.py"),
            "--linalg",
            str(args.linalg),
            "--gemma-options",
            str(gemma_options_path),
        ],
        env=env,
        check=True,
    )


def _read_sidecar(obj: Path, suffix: str) -> str | None:
    """Read a ``<obj-stem>.<suffix>`` sidecar written at compile time."""
    sidecar = obj.with_suffix("." + suffix)
    if sidecar.is_file():
        return sidecar.read_text().strip()
    return None


def _chop_debug_linalg(obj: Path) -> Path:
    """Return a stub linalg for debugging a chopped .o.

    The stub exposes just ``@main`` returning the chop target's return type,
    which is all test_gemma_from_obj.py needs for the return signature. The
    return type is read from the ``.return_type`` sidecar written by
    compile_xmb.py; if absent, an existing ``.linalg_stub.mlir`` is reused.
    """
    stub = obj.with_suffix(".linalg_stub.mlir")
    ret_type = _read_sidecar(obj, "return_type")
    if ret_type:
        stub.write_text(
            f"func.func public @main() -> {ret_type} {{\n"
            f"  %0 = tensor.empty() : {ret_type}\n"
            f"  return %0 : {ret_type}\n"
            f"}}\n"
        )
        print(f"--- Auto-generated chop stub: {ret_type} -> {stub} ---", flush=True)
        return stub
    if stub.is_file():
        print(f"--- Using chop stub linalg: {stub} ---", flush=True)
        return stub
    raise SystemExit(
        f"--chop-debug: no .return_type sidecar next to {obj} and no existing "
        f"{stub}; recompile with --chop-after-call or pass --linalg explicitly."
    )


def _adb_base(args: argparse.Namespace) -> list[str]:
    return ["adb", "-H", args.android_host, "-s", args.android_serial]


def _device_setup(args: argparse.Namespace) -> None:
    """Disable the FastRPC watchdog and enable Hexagon logcat (all levels)."""
    print("--- Device setup ---", flush=True)
    subprocess.run(
        _adb_base(args)
        + ["shell", "setprop", "vendor.fastrpc.process.rpctimeout", "0"],
        check=True,
    )
    # farf goes in /data/local/tmp (vendor partition is read-only after reboot).
    subprocess.run(
        _adb_base(args)
        + [
            "shell",
            "mkdir -p /data/local/tmp && echo 0x1f > "
            "/data/local/tmp/run_main_on_hexagon.farf",
        ],
        check=False,
    )


def _start_logcat_capture(args: argparse.Namespace, logcat_file: Path):
    """Capture logcat continuously without draining the device ring buffer."""
    logcat_file.parent.mkdir(parents=True, exist_ok=True)
    adb = _adb_base(args)
    subprocess.run(
        adb + ["logcat", "-c"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )

    output = logcat_file.open("wb")
    try:
        process = subprocess.Popen(
            adb + ["logcat"], stdout=output, stderr=subprocess.DEVNULL
        )
    except OSError as error:
        output.close()
        print(f"[logcat] unable to start live capture: {error}", flush=True)
        return None
    return process, output


def _stop_logcat_capture(capture) -> None:
    if capture is None:
        return
    process, output = capture
    try:
        if process.poll() is None:
            process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
    finally:
        output.close()
        Path(output.name).touch()


def run_on_device(
    args: argparse.Namespace,
    gemma_options_path: Path,
    gemma_options: GemmaOptions,
) -> None:
    print(f"=== Running {args.obj} on device ===", flush=True)

    # ---- chop-debug: synthesize a stub linalg from the .o's sidecar --------
    if args.chop_debug and args.linalg is None:
        args.linalg = _chop_debug_linalg(args.obj)

    # ---- env consumed by test_gemma_from_obj.py ----------------------------
    env = os.environ.copy()
    env["ANDROID_HOST"] = args.android_host  # validated non-empty in parse_args
    env["ANDROID_SERIAL"] = args.android_serial
    env["GEMMA4_OBJ"] = str(args.obj)
    env["GEMMA4_LINALG"] = str(args.linalg)
    env["GEMMA4_PROMPTS"] = args.prompts
    env["GEMMA4_NO_INPUTS"] = "1" if args.no_inputs else "0"
    env["HEXAGON_RIBBON_STACK_SIZE"] = str(gemma_options.ribbon_stack_bytes)
    # These are absent in --chop-debug mode (the stub run needs no real data).
    if args.weights is not None:
        env["GEMMA4_WEIGHTS"] = str(args.weights)
    if args.metadata is not None:
        env["GEMMA4_METADATA"] = str(args.metadata)
    if args.tokenizer_dir is not None:
        env["TOKENIZER_DIR"] = str(args.tokenizer_dir)
    # Use the active python's bundled libstdc++ (often newer than system).
    _py_lib = Path(sys.executable).resolve().parent.parent / "lib"
    env["LD_LIBRARY_PATH"] = os.pathsep.join(
        [str(_py_lib)]
        + ([env["LD_LIBRARY_PATH"]] if env.get("LD_LIBRARY_PATH") else [])
    )

    # Clear stale device artifacts before pushing model inputs.
    print("--- Cleaning /data/local/tmp on device ---", flush=True)
    subprocess.run(_adb_base(args) + ["shell", "rm -rf /data/local/tmp/*"], check=True)

    # ---- process stack + device setup ---------------------------------------
    resource.setrlimit(
        resource.RLIMIT_STACK,
        (gemma_options.ribbon_stack_bytes, resource.RLIM_INFINITY),
    )
    _device_setup(args)
    # ---- build pytest args (mirrors the old run_from_obj.sh flags) ---------
    pytest_args = [
        "--from-obj",
        "--linalg",
        str(args.linalg),
        "--output-dir",
        str(args.out_dir),
        "--gemma-options",
        str(gemma_options_path),
    ]
    # Optional data paths — omitted in --chop-debug mode (stub run needs none).
    if args.metadata is not None:
        pytest_args += ["--metadata", str(args.metadata)]
    if args.weights is not None:
        pytest_args += ["--weights", str(args.weights)]
    if args.tokenizer_dir is not None:
        pytest_args += ["--tokenizer-dir", str(args.tokenizer_dir)]
    if not args.no_inputs:
        pytest_args += ["--inputs-dir", str(args.inputs_dir)]

    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "-sv",
        "--rootdir",
        str(SCRIPT_DIR),
        "-p",
        "no:cacheprovider",
        str(SCRIPT_DIR / "test_gemma_from_obj.py"),
        *pytest_args,
    ]

    print("--- Running test_gemma_from_obj.py (--from-obj) ---", flush=True)
    logcat_capture = _start_logcat_capture(args, args.out_dir / "logcat.txt")
    try:
        subprocess.run(cmd, env=env, check=True)
    finally:
        _stop_logcat_capture(logcat_capture)


def main(argv=None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(raw_argv)
    args = resolve_paths(args)
    if args.sweep_json is not None:
        return run_parameter_sweep(args, raw_argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    gemma_options = _apply_driver_overrides(GemmaOptions.from_args(args))
    gemma_options_path = args.out_dir / "gemma_options.json"
    gemma_options.write(gemma_options_path)

    if not args.skip_compile:
        run_compile(args, gemma_options_path, gemma_options)
    else:
        print(f"=== Skipping compile, reusing {args.obj} ===", flush=True)

    if not args.skip_run:
        try:
            run_on_device(args, gemma_options_path, gemma_options)
        finally:
            # Recorded even when the run fails: the logcat from a multi-hour
            # run is worth keeping regardless of how it ended. An exception
            # raised here would replace whatever ended the run -- including a
            # SystemExit from the executor -- so recording failures are caught
            # and reported instead of propagating out of the finally block.
            if args.record_perf or args.record_perf_dir is not None:
                try:
                    record_run(
                        out_dir=args.out_dir,
                        csv_dir=args.record_perf_dir,
                        options=gemma_options,
                        repo=SCRIPT_DIR.parent.parent,
                        gemma_root=args.gemma_root,
                        linalg=args.linalg,
                        obj=args.obj,
                        prompts=args.prompts,
                        comment=args.record_perf_comment,
                    )
                except Exception as exc:
                    print(
                        f"[record-perf] recording failed: "
                        f"{type(exc).__name__}: {exc}",
                        flush=True,
                    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
