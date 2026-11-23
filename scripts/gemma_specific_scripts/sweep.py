# ===- sweep.py ------------------------------------------------------------===
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause.
# For more license information:
#   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
#
# ===------------------------------------------------------------------------===

"""Parameter-sweep orchestration for the Gemma driver."""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
import os
import queue
import re
import signal
import shlex
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

from gemma_options import GemmaOptions

SCRIPT_DIR = Path(__file__).resolve().parent

_COMPILE_TIME_RE = re.compile(r"Done in\s+([0-9]+(?:\.[0-9]+)?)s", re.IGNORECASE)
_PERF_RE = re.compile(r"(?im)^\s*Perf:\s*([-+0-9.eE]+)\s*$")
_UNITS_RE = re.compile(r"(?im)^\s*Units:\s*(\S+)\s*$")
_RESULT_RE = re.compile(r"(?im)^\s*Result:\s*(\S+)\s*$")
_DEVICE_ENV_RE = re.compile(
    r"(?:^|\s)(?:export\s+)?(ANDROID_HOST|ANDROID_SERIAL)\s*=\s*([^\s]+)"
)


@dataclass
class _DeviceLease:
    host: str
    serial: str
    expires_at: float


@dataclass
class _DeviceSlot:
    index: int
    lease: _DeviceLease | None


class _DevicePool:
    """Acquire and renew device rentals for concurrent sweep workers."""

    def __init__(self, args: argparse.Namespace, output_dir: Path):
        self.args = args
        self.log_path = output_dir / "device_pool.log"
        self._log_lock = threading.Lock()

    def _log(self, text: str) -> None:
        with self._log_lock:
            with self.log_path.open("a") as log:
                log.write(text.rstrip() + "\n")

    def acquire(self, slot_index: int) -> _DeviceLease | None:
        try:
            command = shlex.split(self.args.drental)
        except ValueError as error:
            self._log(f"[slot {slot_index}] invalid drental command: {error}")
            return None
        if not command:
            self._log(f"[slot {slot_index}] drental command is empty")
            return None

        command.extend(
            [
                "acquire",
                "--device-timeout",
                str(self.args.timeout),
                "-r",
                f"codename={self.args.codename}",
            ]
        )
        self._log(f"[slot {slot_index}] $ {shlex.join(command)}")
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=False,
            )
        except OSError as error:
            self._log(f"[slot {slot_index}] failed to start drental: {error}")
            return None

        output = "\n".join(
            part for part in (completed.stdout, completed.stderr) if part
        )
        self._log(
            f"[slot {slot_index}] drental exited {completed.returncode}\n{output}"
        )
        if completed.returncode != 0:
            return None

        values = {
            name: value.strip("\"'") for name, value in _DEVICE_ENV_RE.findall(output)
        }
        host = values.get("ANDROID_HOST")
        serial = values.get("ANDROID_SERIAL")
        if not host or not serial:
            self._log(
                f"[slot {slot_index}] drental output did not contain "
                "ANDROID_HOST and ANDROID_SERIAL"
            )
            return None

        lease = _DeviceLease(
            host=host,
            serial=serial,
            expires_at=time.monotonic() + self.args.timeout * 60,
        )
        print(
            f"[device-pool] slot {slot_index} acquired {host} {serial} "
            f"for {self.args.timeout} minutes",
            flush=True,
        )
        return lease

    def ensure_current(self, slot: _DeviceSlot) -> _DeviceLease | None:
        if slot.lease is not None and time.monotonic() < slot.lease.expires_at:
            return slot.lease
        if slot.lease is not None:
            print(
                f"[device-pool] slot {slot.index} rental expired; acquiring replacement",
                flush=True,
            )
        slot.lease = self.acquire(slot.index)
        return slot.lease


def resolve_sweep_args(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    raw_argv: list[str],
) -> argparse.Namespace:
    """Resolve sweep mode and reject incompatible explicit options."""
    if args.sweep and args.sweep_json:
        parser.error("use either --sweep or --sweep-json, not both")
    if args.sweep:
        args.sweep_json = GemmaOptions.default_sweep_json_path()

    if args.device_pool and not args.sweep_json:
        parser.error("--device-pool requires --sweep or --sweep-json")
    if args.device_pool and args.num_devices <= 0:
        parser.error("--num-devices must be positive")
    if args.device_pool and args.timeout <= 0:
        parser.error("--timeout must be positive")
    if args.execution_timeout <= 0:
        parser.error("--execution-timeout must be positive")

    debug_arguments = {
        argument.split("=", 1)[0]
        for argument in raw_argv
        if argument.split("=", 1)[0] in GemmaOptions.debug_option_strings()
    }
    if args.sweep_json and debug_arguments:
        parser.error(
            "Debug options are not supported during sweeps: "
            + ", ".join(sorted(debug_arguments))
        )
    return args


def _strip_sweep_arguments(
    argv: list[str], strip_device_connection: bool = False
) -> list[str]:
    """Remove sweep-only arguments before invoking a child run."""
    result: list[str] = []
    index = 0
    flag_arguments = {
        "--sweep",
        "--device-pool",
    }
    value_arguments = {
        "--sweep-json",
        "--sweep-csv",
        "--sweep-output-dir",
        "--num-devices",
        "--timeout",
        "--execution-timeout",
        "--codename",
        "--drental",
    }
    if strip_device_connection:
        value_arguments.update({"--android-host", "--android-serial"})

    while index < len(argv):
        argument = argv[index]
        if argument in flag_arguments or argument in value_arguments:
            index += 1 if argument in flag_arguments else 2
            continue
        if any(argument.startswith(flag + "=") for flag in flag_arguments):
            index += 1
            continue
        if any(argument.startswith(flag + "=") for flag in value_arguments):
            index += 1
            continue
        result.append(argument)
        index += 1
    return result


def _sweep_value_text(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _sweep_run_name(index: int) -> str:
    return f"{index:04d}"


def _cleanup_run_artifacts(run_dir: Path) -> None:
    """Remove large per-execution directories while keeping sweep summaries."""
    for artifact in run_dir.glob("main-*"):
        if artifact.is_dir() and not artifact.is_symlink():
            shutil.rmtree(artifact, ignore_errors=True)
        else:
            artifact.unlink(missing_ok=True)


def _clear_device_logcat(host: str, serial: str, label: str) -> None:
    """Clear the selected device's logcat before a sweep execution."""
    command = ["adb", "-H", host, "-s", serial, "logcat", "-c"]
    try:
        completed = subprocess.run(
            command,
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        print(f"[sweep/{label}] unable to clear logcat: {error}", flush=True)
        return
    if completed.returncode != 0:
        print(
            f"[sweep/{label}] logcat clear exited with " f"{completed.returncode}",
            flush=True,
        )


def _run_sweep_child(
    command: list[str],
    log_path: Path,
    label: str,
    env_overrides: dict[str, str] | None = None,
    timeout_s: int | None = None,
) -> tuple[int | None, float, bool]:
    """Run one isolated sweep phase and save its combined output."""
    start = time.perf_counter()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    if env_overrides:
        env.update(env_overrides)
    with log_path.open("w", buffering=1) as log:
        log.write("$ " + " ".join(command) + "\n\n")
        if env_overrides:
            log.write("Environment overrides: " + repr(env_overrides) + "\n\n")
        try:
            process = subprocess.Popen(
                command,
                cwd=SCRIPT_DIR.parents[2],
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
        except OSError as error:
            log.write(f"{type(error).__name__}: {error}\n")
            print(f"[sweep/{label}] failed to start: {error}", flush=True)
            return None, time.perf_counter() - start, False

        timed_out = False
        try:
            process.communicate(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            timed_out = True
            log.write(
                f"\nTimed out after {timeout_s} seconds; "
                "terminating the child process group.\n"
            )
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                log.write("Process group did not exit; sending SIGKILL.\n")
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.communicate()

    elapsed = time.perf_counter() - start
    if timed_out:
        print(
            f"[sweep/{label}] timed out after {elapsed:.2f}s",
            flush=True,
        )
        return process.returncode, elapsed, True
    print(
        f"[sweep/{label}] exited with {process.returncode} " f"after {elapsed:.2f}s",
        flush=True,
    )
    return process.returncode, elapsed, False


def _compile_backend_time(run_dir: Path) -> float | None:
    path = run_dir / "compile_stdout.log"
    if not path.is_file():
        return None
    matches = _COMPILE_TIME_RE.findall(path.read_text(errors="replace"))
    return float(matches[-1]) if matches else None


def _perf_execution_data(run_dir: Path) -> tuple[float | None, str | None]:
    candidates = list(run_dir.rglob("*_perf.txt"))
    direct = run_dir / "perf.txt"
    if direct.is_file():
        candidates.append(direct)
    texts = [
        path.read_text(errors="replace")
        for path in sorted(candidates, key=lambda path: path.stat().st_mtime_ns)
    ]
    execute_log = run_dir / "execute.log"
    if execute_log.is_file():
        texts.append(execute_log.read_text(errors="replace"))
    text = "\n".join(texts)
    perf_matches = _PERF_RE.findall(text)
    unit_matches = _UNITS_RE.findall(text)
    result_matches = _RESULT_RE.findall(text)
    result = result_matches[-1].lower() if result_matches else None
    if not perf_matches or not unit_matches:
        return None, result

    perf = float(perf_matches[-1])
    units = unit_matches[-1].lower()
    if units in {"us", "usec", "microsecond", "microseconds"}:
        return perf / 1_000_000.0, result
    if units in {"ms", "msec", "millisecond", "milliseconds"}:
        return perf / 1_000.0, result
    if units in {"s", "sec", "second", "seconds"}:
        return perf, result
    return None, result


def _sweep_row(
    combination: dict[str, object],
    compile_time: float | None,
    perf_time: float | None,
    error: str,
) -> dict[str, object]:
    row = {flag: _sweep_value_text(value) for flag, value in combination.items()}
    row.update(
        {
            "compile_time_s": compile_time if not error else None,
            "perf_execution_time_s": perf_time if not error else None,
            "error": error,
        }
    )
    return row


def _run_sweep_combination(
    index: int,
    combination: dict[str, object],
    base_argv: list[str],
    sweep_output_dir: Path,
    execution_timeout: int,
    android_host: str | None = None,
    android_serial: str | None = None,
    device_pool: _DevicePool | None = None,
    device_slots: queue.Queue[_DeviceSlot] | None = None,
) -> dict[str, object]:
    """Run one compile/execute combination and return its CSV row."""
    run_dir = sweep_output_dir / _sweep_run_name(index)
    obj_path = run_dir / "gemma_kernel.o"
    slot = device_slots.get() if device_slots is not None else None
    try:
        common_argv = [
            *base_argv,
            "--out-dir",
            str(run_dir),
            "--obj",
            str(obj_path),
            *GemmaOptions.optimization_sweep_arguments(combination),
        ]
        compile_command = [
            sys.executable,
            str(Path(__file__).resolve().with_name("run_gemma.py")),
            *common_argv,
            "--skip-run",
        ]
        execute_command = [
            sys.executable,
            str(Path(__file__).resolve().with_name("run_gemma.py")),
            *common_argv,
            "--skip-compile",
        ]

        print(f"\n--> Launching {index + 1}...", flush=True)
        compile_returncode, _, compile_timed_out = _run_sweep_child(
            compile_command,
            run_dir / "compile.log",
            f"{index}/compile",
        )
        compile_time = _compile_backend_time(run_dir)
        if compile_timed_out:
            return _sweep_row(combination, compile_time, None, "Timeout")
        if compile_returncode != 0 or not obj_path.is_file() or compile_time is None:
            return _sweep_row(combination, compile_time, None, "Compile")

        env_overrides = None
        device_host = android_host
        device_serial = android_serial
        if device_pool is not None:
            assert slot is not None
            lease = device_pool.ensure_current(slot)
            if lease is None:
                return _sweep_row(combination, compile_time, None, "Acquire")
            device_host = lease.host
            device_serial = lease.serial
            env_overrides = {
                "ANDROID_HOST": lease.host,
                "ANDROID_SERIAL": lease.serial,
            }

        if device_host and device_serial:
            _clear_device_logcat(
                device_host,
                device_serial,
                f"{index}/execute",
            )
        execute_returncode, _, execute_timed_out = _run_sweep_child(
            execute_command,
            run_dir / "execute.log",
            f"{index}/execute",
            env_overrides,
            execution_timeout,
        )
        if execute_timed_out:
            return _sweep_row(combination, compile_time, None, "Timeout")
        perf_time, result_text = _perf_execution_data(run_dir)
        if (
            execute_returncode != 0
            or perf_time is None
            or (result_text and result_text != "pass")
        ):
            return _sweep_row(combination, compile_time, perf_time, "Execute")
        return _sweep_row(combination, compile_time, perf_time, "")
    except Exception as error:
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "sweep_error.log").write_text(f"{type(error).__name__}: {error}\n")
        return _sweep_row(combination, None, None, "Sweep")
    finally:
        _cleanup_run_artifacts(run_dir)
        if device_slots is not None and slot is not None:
            device_slots.put(slot)


def run_parameter_sweep(args: argparse.Namespace, raw_argv: list[str]) -> int:
    """Run every combination from --sweep-json and write one CSV row each."""
    assert args.sweep_json is not None
    try:
        combinations = list(
            GemmaOptions.optimization_sweep_combinations(args.sweep_json)
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error

    sweep_output_dir = args.sweep_output_dir
    if sweep_output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        sweep_output_dir = Path("/tmp") / f"gemma_sweep_{timestamp}"
    sweep_output_dir.mkdir(parents=True, exist_ok=True)

    results_csv = args.sweep_csv or sweep_output_dir / "sweep_results.csv"
    results_csv.parent.mkdir(parents=True, exist_ok=True)
    base_argv = _strip_sweep_arguments(raw_argv, args.device_pool)
    parameter_columns = list(combinations[0])
    fieldnames = [
        *parameter_columns,
        "compile_time_s",
        "perf_execution_time_s",
        "error",
    ]

    with results_csv.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        completed_count = 0
        total_runs = len(combinations)

        if not args.device_pool:
            for index, combination in enumerate(combinations):
                writer.writerow(
                    _run_sweep_combination(
                        index,
                        combination,
                        base_argv,
                        sweep_output_dir,
                        args.execution_timeout,
                        args.android_host,
                        args.android_serial,
                    )
                )
                csv_file.flush()
                completed_count += 1
                print(
                    f"-----> {index + 1} Complete " f"({completed_count}/{total_runs})",
                    flush=True,
                )
        else:
            device_count = min(len(combinations), args.num_devices)
            device_pool = _DevicePool(args, sweep_output_dir)
            device_slots: queue.Queue[_DeviceSlot] = queue.Queue()
            for slot_index in range(device_count):
                device_slots.put(
                    _DeviceSlot(
                        slot_index,
                        device_pool.acquire(slot_index),
                    )
                )

            with ThreadPoolExecutor(max_workers=device_count) as executor:
                futures = {
                    executor.submit(
                        _run_sweep_combination,
                        index,
                        combination,
                        base_argv,
                        sweep_output_dir,
                        args.execution_timeout,
                        args.android_host,
                        args.android_serial,
                        device_pool,
                        device_slots,
                    ): index
                    + 1
                    for index, combination in enumerate(combinations)
                }
                for future in as_completed(futures):
                    writer.writerow(future.result())
                    csv_file.flush()
                    completed_count += 1
                    print(
                        f"-----> {futures[future]} Complete "
                        f"({completed_count}/{total_runs})",
                        flush=True,
                    )

    print(f"\nSweep results: {results_csv}", flush=True)
    return 0
