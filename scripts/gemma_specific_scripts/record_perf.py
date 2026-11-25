# ===- record_perf.py ------------------------------------------------------===
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause.
# For more license information:
#   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
#
# ===------------------------------------------------------------------------===

"""Save one Gemma performance snapshot requested by ``--record-perf`` or
``--record-perf-dir``.

After a run, this module reads the run's ``logcat.txt`` and ``*_perf.txt``.
The compiler-inserted ``[kernel-time]`` logcat records provide one DSP timing
record for every selected function invocation. The wrapper's ``perf.txt``
provides the measured whole-model total.

The wrapper calls the model twice: first as an untimed warm-up, then as the
measured run. This module selects the ``parent=0`` ``[kernel-time]`` record
with the greatest DSP end timestamp (preferring the longest interval on a
tie), then keeps unique timing records within that root interval. Neither CSV
therefore includes warm-up function records, even when logcat repeats or
reorders messages.

It writes two kinds of CSV data:

* One run-summary row for the measured model call in ``<model>_runs.csv``
  (or ``<model>_run_<timestamp>.csv`` with ``--record-perf-dir``): the
  wrapper's overall ``Perf`` time, the DSP root function's elapsed time, the
  inputs and options used, and the Git state.
* One row per selected function invocation in
  ``<model>_kernels_<timestamp>.csv``:
  its inclusive duration, its percentage of the root-function time, and its
  caller relationship. When two selected functions run consecutively on the
  same DSP execution thread, the row also reports the derived gap from the
  prior function's ``end`` timestamp to this function's ``start`` timestamp.
  This ``gluecode_exec_time_us`` value is calculated from logcat timestamps;
  it is not a separate DSP log message.

Without ``--record-perf-dir``:

  <shared-dir>/<model>_runs.csv          append one run-summary row
  <output-dir>/<model>_kernels_<timestamp>.csv
                                           write this run's kernel rows

With ``--record-perf-dir``:

  <csv-dir>/<model>_run_<timestamp>.csv  write one run-summary row
  <csv-dir>/<model>_kernels_<timestamp>.csv
                                           write this run's kernel rows
"""

from __future__ import annotations

import csv
from dataclasses import asdict
from datetime import datetime
import json
import os
from pathlib import Path
import re
import subprocess

# "[kernel-time] <sym> id=<id> parent=<id> thread=<ptr> start=<us>us
# end=<us>us duration=<us>us" -- emitted at every llvm.return, so a parent's
# record follows all of its children.
_KERNEL_TIME_RE = re.compile(
    r"\[kernel-time\]\s+(\S+)\s+id=(\d+)\s+parent=(-?\d+)"
    r"\s+thread=(\S+)\s+start=(\d+)us\s+end=(\d+)us"
    r"\s+duration=(\d+)us"
)
# "[progress] <sym>  dt=0us" -- emitted only at model entry.
_PROGRESS_RE = re.compile(r"\[progress\]\s+(\S+)\s+dt=(\d+)us")
# TestReport::save() output in *_perf.txt.
_PERF_RE = re.compile(r"Perf:\s*([0-9.]+)")
_PERF_UNITS_RE = re.compile(r"Units:\s*(\w+)")

SHARED_SUMMARY_DIR = Path(
    os.environ.get("GEMMA_PERF_SUMMARY_DIR", "record_snapshot_perf")
)
KERNEL_FIELDS = [
    "seq",
    "symbol",
    "kernel_id",
    "parent_id",
    "parent_symbol",
    "depth",
    "duration_us",
    "model_time_pct(x_t/main_t)*100",
    "func_before_glue",
    "gluecode_exec_time_us",
]

RUN_FIELDS = [
    "run_id",
    "timestamp",
    "perf_total_us",
    "main_dt_us",
    "entry_symbol",
    "commit",
    "branch",
    "dirty",
    "patch_path",
    "kernels_csv",
    "options_json",
    "linalg",
    "obj",
    "out_dir",
    "logcat",
    "prompts",
    "comment",
]


class _LogcatRecord:
    """One injected record, with the DSP metadata needed to order it."""

    __slots__ = (
        "symbol",
        "kind",
        "value",
        "kernel_id",
        "parent_id",
        "thread_key",
        "start_us",
        "end_us",
    )

    def __init__(
        self,
        symbol,
        kind,
        value,
        kernel_id=None,
        parent_id=None,
        thread_key=None,
        start_us=None,
        end_us=None,
    ):
        self.symbol = symbol
        self.kind = kind  # "kernel-time" | "progress"
        self.value = value  # duration_us | dt_us
        self.kernel_id = kernel_id
        self.parent_id = parent_id
        self.thread_key = thread_key
        self.start_us = start_us
        self.end_us = end_us


def find_logcat(out_dir: Path) -> Path | None:
    """Locate the logcat dumped by HexagonExecutor under the run's out-dir."""
    candidates = sorted(out_dir.rglob("logcat.txt"), key=lambda p: p.stat().st_mtime_ns)
    return candidates[-1] if candidates else None


def find_perf_txt(out_dir: Path) -> Path | None:
    """Locate the TestReport perf file written by the wrapper."""
    candidates = [p for p in out_dir.rglob("*_perf.txt")]
    direct = out_dir / "perf.txt"
    if direct.is_file():
        candidates.append(direct)
    if not candidates:
        return None
    return sorted(candidates, key=lambda p: p.stat().st_mtime_ns)[-1]


def model_csv_name(gemma_root: Path | None, linalg: Path | None) -> str:
    """Return the model-specific prefix used in summary CSV filenames.

    Packaged models keep runtime inputs in a ``deps`` directory, so use its
    parent name: ``.../<model>/deps`` becomes ``<model>``. A direct model root
    uses its own name; an explicit linalg-only run uses its stem.
    """
    if gemma_root is not None:
        root = gemma_root.parent if gemma_root.name == "deps" else gemma_root
        return root.name
    if linalg is not None:
        return linalg.stem
    return "gemma"


def read_perf_total_us(perf_txt: Path | None) -> float | None:
    """Return the wrapper's measured total in microseconds."""
    if perf_txt is None or not perf_txt.is_file():
        return None
    text = perf_txt.read_text(errors="replace")
    perf = _PERF_RE.search(text)
    if not perf:
        return None
    value = float(perf.group(1))
    units = _PERF_UNITS_RE.search(text)
    unit = units.group(1).lower() if units else "us"
    if unit.startswith("ms"):
        return value * 1_000.0
    if unit in {"s", "sec", "second", "seconds"}:
        return value * 1_000_000.0
    return value


def parse_logcat(logcat: Path) -> tuple[list[_LogcatRecord], list[_LogcatRecord]]:
    """Return (kernel_time_records, progress_records) in log order."""
    kernel_times: list[_LogcatRecord] = []
    progress: list[_LogcatRecord] = []
    with logcat.open(errors="replace") as handle:
        for line in handle:
            kernel = _KERNEL_TIME_RE.search(line)
            if kernel:
                kernel_times.append(
                    _LogcatRecord(
                        kernel.group(1),
                        "kernel-time",
                        int(kernel.group(7)),
                        int(kernel.group(2)),
                        int(kernel.group(3)),
                        kernel.group(4),
                        int(kernel.group(5)),
                        int(kernel.group(6)),
                    )
                )
                continue

            entry = _PROGRESS_RE.search(line)
            if entry:
                progress.append(
                    _LogcatRecord(entry.group(1), "progress", int(entry.group(2)))
                )
    return kernel_times, progress


def marked_entry_symbol(progress: list[_LogcatRecord]) -> str | None:
    """Return the symbol the pass marked with its dt=0us entry record.

    The pass marks one public function per invocation, but which public
    function it picks depends on module walk order, so this is not necessarily
    the outermost frame. It is reported for provenance only; the measured root
    is selected from ``parent=0``. See ``measured_root_record``.
    """
    zeros = [rec for rec in progress if rec.value == 0]
    return zeros[0].symbol if zeros else None


def _depth_of(kernel_id: int, parent_of: dict[int, int]) -> int:
    """Walk the parent chain to a root. Returns -1 if the chain is cyclic."""
    depth = 0
    seen = set()
    current = kernel_id
    while True:
        parent = parent_of.get(current, 0)
        if parent == 0:
            return depth
        if parent in seen:
            return -1
        seen.add(parent)
        depth += 1
        current = parent


def measured_root_record(
    kernel_times: list[_LogcatRecord],
) -> _LogcatRecord | None:
    """Return the measured outermost ``[kernel-time]`` record.

    ``parent=0`` identifies an outermost instrumented frame on its DSP thread:
    the first selected function on a worker thread also has no caller in that
    thread's TLS slot. The controller model call normally has the latest end
    timestamp. If both end in the same microsecond, the model call normally
    ran longer because it waited for the worker work, so it is selected.

    The ``[progress]`` value is always literal zero, so its log line is not a
    usable boundary either: buffered DSP messages can arrive after a root
    return record. The DSP timestamps are the ordering source of truth, so the
    root invocation with the greatest end timestamp is the measured call.
    """
    roots = [rec for rec in kernel_times if rec.parent_id == 0]
    if not roots:
        return None
    return max(roots, key=lambda rec: (rec.end_us, rec.value))


def _timing_event_key(rec: _LogcatRecord) -> tuple[object, ...]:
    """Identify one timing event independently of duplicated logcat lines."""
    return (
        rec.symbol,
        rec.kernel_id,
        rec.parent_id,
        rec.thread_key,
        rec.start_us,
        rec.end_us,
        rec.value,
    )


def _is_outlined_symbol(symbol: str) -> bool:
    return "__outlined_generic_" in symbol or "__merged_outlined_generic_" in symbol


def print_timing_summary(
    kernel_rows: list[dict[str, object]],
    measured_root: _LogcatRecord | None,
    perf_total_us: float | None,
) -> None:
    """Print outlined execution and glue totals relative to the model root."""
    if measured_root is None or measured_root.value == 0:
        return

    root_us = measured_root.value
    if perf_total_us is None:
        perf_text = "unavailable"
    elif perf_total_us.is_integer():
        perf_text = f"{int(perf_total_us)} us"
    else:
        perf_text = f"{perf_total_us} us"
    print("[record-perf] timing summary:", flush=True)
    print(
        f"[record-perf]   root: {measured_root.symbol} " "(outer model-call function)",
        flush=True,
    )
    print(f"[record-perf]   perf_total:             {perf_text}", flush=True)
    print(f"[record-perf]   root total:             {root_us} us", flush=True)

    outlined = [row for row in kernel_rows if _is_outlined_symbol(str(row["symbol"]))]
    if not outlined:
        return

    outlined_us = sum(int(row["duration_us"]) for row in outlined)
    between_outline_glue_us = sum(
        int(row["gluecode_exec_time_us"])
        for row in outlined
        if (
            row["gluecode_exec_time_us"] != ""
            and _is_outlined_symbol(str(row["func_before_glue"]))
        )
    )
    print(
        f"[record-perf]   outlined execution:     {outlined_us} us "
        f"({outlined_us * 100.0 / root_us:.4f}% of root total)",
        flush=True,
    )
    print(
        f"[record-perf]   between-outline glue:   {between_outline_glue_us} us "
        f"({between_outline_glue_us * 100.0 / root_us:.4f}% of root total)",
        flush=True,
    )


def build_kernel_rows(
    kernel_times: list[_LogcatRecord],
    main_dt_us: int | None,
    measured_root: _LogcatRecord | None,
) -> list[dict[str, object]]:
    """One row per kernel invocation in the measured pass.

    Selection is by DSP timestamp: a record is kept when its interval lies
    inside ``measured_root``'s, and duplicate logcat deliveries of one event
    are collapsed. There is deliberately no line-order fallback -- logcat
    replays buffered DSP messages, so "everything after line N" spans an
    unpredictable number of replays and would silently multiply the row count.

    ``model_time_pct(x_t/main_t)*100`` is each inclusive invocation duration
    divided by the measured entry-function duration, multiplied by 100.
    Inclusive nested rows need not add to 100 percent.

    ``gluecode_exec_time_us`` is the interval from the most recently completed
    selected function to this function's entry on the same thread. The row's
    ``symbol`` is therefore the function after the gap, and
    ``func_before_glue`` identifies the function before it. The gap includes
    instrumentation around the two timestamps as well as uninstrumented code.
    It is blank when no earlier selected function completed on that thread.
    """
    if measured_root is None:
        raise RuntimeError(
            "no timing record for the outermost run; cannot tell which "
            "invocation was measured. The log may be truncated (Android log "
            "buffer wrap)."
        )
    seen: set[tuple[object, ...]] = set()
    measured = []
    for rec in kernel_times:
        if rec.start_us < measured_root.start_us or rec.end_us > measured_root.end_us:
            continue
        event_key = _timing_event_key(rec)
        if event_key in seen:
            continue
        seen.add(event_key)
        measured.append(rec)
    # ``seq`` reflects DSP invocation order, not delivery order in logcat.
    measured.sort(
        key=lambda rec: (
            rec.start_us,
            -rec.end_us,
            rec.thread_key,
            rec.kernel_id,
            rec.parent_id,
            rec.symbol,
        )
    )
    symbol_of = {rec.kernel_id: rec.symbol for rec in measured}
    parent_of = {rec.kernel_id: rec.parent_id for rec in measured}
    prior_for: dict[tuple[object, ...], _LogcatRecord] = {}
    timestamped_by_thread: dict[str, list[_LogcatRecord]] = {}
    for rec in measured:
        if (
            rec.thread_key is not None
            and rec.start_us is not None
            and rec.end_us is not None
        ):
            timestamped_by_thread.setdefault(rec.thread_key, []).append(rec)

    # A log line is written at return and threads may interleave those lines.
    # Match using DSP timestamps and the TLS address, rather than CSV/log order.
    for records in timestamped_by_thread.values():
        for rec in records:
            completed = [
                earlier
                for earlier in records
                if earlier is not rec and earlier.end_us <= rec.start_us
            ]
            if completed:
                prior_for[_timing_event_key(rec)] = max(
                    completed, key=lambda item: item.end_us
                )

    rows = []
    for seq, rec in enumerate(measured):
        prior = prior_for.get(_timing_event_key(rec))
        rows.append(
            {
                "seq": seq,
                "symbol": rec.symbol,
                "kernel_id": rec.kernel_id,
                "parent_id": rec.parent_id,
                "parent_symbol": symbol_of.get(rec.parent_id, ""),
                "depth": _depth_of(rec.kernel_id, parent_of),
                "duration_us": rec.value,
                "model_time_pct(x_t/main_t)*100": (
                    rec.value * 100.0 / main_dt_us if main_dt_us else ""
                ),
                "func_before_glue": prior.symbol if prior else "",
                "gluecode_exec_time_us": (rec.start_us - prior.end_us if prior else ""),
            }
        )
    return rows


def git_state(repo: Path, patch_dest: Path) -> dict[str, object]:
    """Capture commit/branch/dirty and write a patch of local changes."""

    def _git(*argv: str) -> str | None:
        result = subprocess.run(
            ["git", "-C", str(repo), *argv],
            capture_output=True,
            text=True,
            check=False,
        )
        return result.stdout if result.returncode == 0 else None

    commit = (_git("rev-parse", "HEAD") or "").strip()
    branch = (_git("rev-parse", "--abbrev-ref", "HEAD") or "").strip()
    # Tracked modifications only; a patch cannot carry untracked files.
    diff = _git("diff", "HEAD") or ""
    patch_path = ""
    if diff.strip():
        patch_dest.parent.mkdir(parents=True, exist_ok=True)
        patch_dest.write_text(diff)
        patch_path = str(patch_dest)
    return {
        "commit": commit,
        "branch": branch,
        "dirty": bool(diff.strip()),
        "patch_path": patch_path,
    }


def write_summary_csv(
    summary_path: Path,
    row: dict[str, object],
    *,
    append: bool,
) -> None:
    """Write one summary row, adding a header for a new shared CSV.

    ``RUN_FIELDS`` is the ordered list of summary-CSV column names. Before
    appending, require the existing header to match it exactly; otherwise a
    row could put values in the wrong columns.
    """
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    needs_header = (
        not append or not summary_path.exists() or (summary_path.stat().st_size == 0)
    )
    if append and not needs_header:
        with summary_path.open(newline="") as handle:
            header = next(csv.reader(handle), [])
        if header != RUN_FIELDS:
            raise RuntimeError(
                f"{summary_path} has a different column layout; not appending. "
                f"Move it aside or migrate it. Found: {','.join(header)}"
            )
    with summary_path.open("a" if append else "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RUN_FIELDS)
        if needs_header:
            writer.writeheader()
        writer.writerow(row)


def record_run(
    out_dir: Path,
    csv_dir: Path | None,
    options,
    repo: Path,
    gemma_root: Path | None,
    linalg: Path | None,
    obj: Path | None,
    prompts: str,
    comment: str = "",
) -> Path | None:
    """Write the run-summary and per-kernel CSVs. Returns the summary path."""
    now = datetime.now()
    run_id = now.strftime("%Y%m%d_%H%M%S")
    model_name = model_csv_name(gemma_root, linalg)

    logcat = find_logcat(out_dir)
    if logcat is None:
        print(
            f"[record-perf] no logcat.txt under {out_dir}; nothing recorded", flush=True
        )
        return None

    kernel_times, progress = parse_logcat(logcat)
    measured_root = measured_root_record(kernel_times)
    main_dt_us = measured_root.value if measured_root is not None else None
    # Names the function ``main_dt_us`` came from. Falls back to the
    # ``[progress]`` marker only for logs with no timing records at all.
    entry_symbol = (
        measured_root.symbol
        if measured_root is not None
        else marked_entry_symbol(progress)
    )

    kernel_rows: list[dict[str, object]] = []
    if kernel_times:
        kernel_rows = build_kernel_rows(kernel_times, main_dt_us, measured_root)
    elif progress:
        # Deltas between consecutive [progress] records are only valid while
        # the log is one sequential stream, so they are not written as
        # per-kernel rows. The column records that the object predates the
        # [kernel-time] instrumentation.
        print(
            "[record-perf] no [kernel-time] records: this object predates the "
            "timing instrumentation. Per-kernel CSV not written; rebuild to "
            "get exact durations.",
            flush=True,
        )
    else:
        print(
            "[record-perf] no injected records in the logcat; is "
            "injectProgressPrints enabled?",
            flush=True,
        )

    output_dir = csv_dir if csv_dir is not None else out_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    kernels_csv = ""
    if kernel_rows:
        kernels_path = output_dir / f"{model_name}_kernels_{run_id}.csv"
        with kernels_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=KERNEL_FIELDS)
            writer.writeheader()
            writer.writerows(kernel_rows)
        kernels_csv = str(kernels_path)

    perf_total_us = read_perf_total_us(find_perf_txt(out_dir))
    state = git_state(repo, out_dir / f"local_changes_{run_id}.patch")

    row = {
        "run_id": run_id,
        "timestamp": now.isoformat(timespec="seconds"),
        "perf_total_us": perf_total_us if perf_total_us is not None else "",
        "main_dt_us": main_dt_us if main_dt_us is not None else "",
        "entry_symbol": entry_symbol or "",
        "kernels_csv": kernels_csv,
        "options_json": json.dumps(asdict(options), sort_keys=True),
        "linalg": str(linalg) if linalg else "",
        "obj": str(obj) if obj else "",
        "out_dir": str(out_dir),
        "logcat": str(logcat),
        "prompts": prompts,
        "comment": comment,
        **state,
    }

    if csv_dir is None:
        summary_path = SHARED_SUMMARY_DIR / f"{model_name}_runs.csv"
        write_summary_csv(summary_path, row, append=True)
    else:
        summary_path = output_dir / f"{model_name}_run_{run_id}.csv"
        write_summary_csv(summary_path, row, append=False)

    print(f"[record-perf] run summary: {summary_path}", flush=True)
    if kernels_csv:
        print(
            f"[record-perf] {len(kernel_rows)} invocations "
            f"({len({row['symbol'] for row in kernel_rows})} distinct): "
            f"{kernels_csv}",
            flush=True,
        )
    if state["patch_path"]:
        print(f"[record-perf] local changes: {state['patch_path']}", flush=True)
    if kernels_csv:
        print_timing_summary(kernel_rows, measured_root, perf_total_us)
    return summary_path
