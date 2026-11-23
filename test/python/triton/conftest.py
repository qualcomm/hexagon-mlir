# ===- conftest.py ----------------------------------------------------------===
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause.
# For more license information:
#   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
#
# ===------------------------------------------------------------------------===

import json
import os
import re
import sys
import time
from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import TextIO

import pytest

# ===------------------------------------------------------------------------===
# GLOBALS
# ===------------------------------------------------------------------------===

# Regex that maches a Pytest node id such as:
#   /path/to/test_file.py::test_function[param0-param1]
RE_NODEID = re.compile(r"^(?P<file>[^:]+)::(?P<function>\w+)(\[(?P<params>.+?)\])?")

# Number of cores for this execution
C: int = int(os.environ.get("NUM_CORES", "1"))

# Number of threads for this execution
T: int = int(os.environ.get("NUM_THREADS", "1"))

# Tests that are prefixed with key are marked with value
PREFIX_MARKERS = {
    "test_block_": "block",
}

# ===------------------------------------------------------------------------===
# LOCAL FUNCTIONS
# ===------------------------------------------------------------------------===


def parse_nodeid(nodeid: str) -> tuple[Path, str, list[str]]:
    match = RE_NODEID.match(nodeid)
    assert match is not None, "Expected nodeid to match regex"

    file = match.group("file")
    function = match.group("function")
    if params := match.group("params"):
        params = params.split("-")
    else:
        params = []

    return Path(file), function, params


def divider(title: str) -> str:
    padding = "-" * (80 - len(title) - 5)
    return f"::: {title} {padding}\n"


# ===------------------------------------------------------------------------===
# PYTEST HOOKS
# ===------------------------------------------------------------------------===


def pytest_addoption(parser):
    """Add custom Pytest options."""
    parser.addoption(
        "-O",
        "--output",
        action="store",
        default=os.environ.get("PYTEST_OUTPUT"),
        type=Path,
    )
    parser.addoption(
        "--exception-summary",
        action="store_true",
        default=False,
    )
    parser.addoption(
        "--skip-passed",
        action="store_true",
        default=False,
        help="Deselect all tests that previously passed.",
    )
    parser.addoption(
        "--skip-failed",
        action="store_true",
        default=False,
        help="Skip tests that previously failed.",
    )
    parser.addoption(
        "--skip-failed-with",
        type=str,
        default=None,
        metavar="EXCEPTIONS",
        help="Skip tests that previously failed with one of the errors in comma-separated list.",
    )
    parser.addoption(
        "--skip-failed-except",
        type=str,
        default=None,
        metavar="EXCEPTIONS",
        help="Skip all previously failed tests except those with one of the errors in comma-separated list.",
    )
    parser.addoption(
        "--skip-first",
        type=int,
        default=0,
        metavar="N",
        help="Skip the first N tests.",
    )
    parser.addoption(
        "--stop-after",
        type=int,
        default=None,
        metavar="N",
        help="Stop the session after N tests have finished.",
    )


def pytest_configure(config):
    """Add target plugin if the flag is set."""
    config.addinivalue_line("markers", "dsp: test is targetted on DSP platform.")
    config.addinivalue_line("markers", "block: test uses triton block pointer.")
    config.addinivalue_line("markers", "hmx: test uses HMX instructions.")
    config.addinivalue_line("markers", "model: tests a full model.")

    # PyTorch-related markers
    # Add them here to just supress pytest's unknown mark warnings
    config.addinivalue_line("markers", "bfloat16: todo")
    config.addinivalue_line("markers", "hw_only: todo")
    config.addinivalue_line("markers", "postflight: todo")
    config.addinivalue_line("markers", "inductor: todo")

    for dtype in ["f16", "f32", "f64", "i1", "i8", "i16", "i32", "i64"]:
        config.addinivalue_line(
            "markers", f"{dtype}: test operates from {dtype} values"
        )
    if outputdir := config.getoption("--output"):
        os.environ["PYTEST_OUTPUT"] = str(outputdir)
        config.pluginmanager.register(TargetPlugin(Path(outputdir)), "target_plugin")
    if config.getoption("--exception-summary"):
        config.pluginmanager.register(ExceptionSummaryPlugin(), "exc_summary_plugin")
    if (stop_after := config.getoption("--stop-after")) is not None:
        config.pluginmanager.register(StopAfterPlugin(stop_after), "stop_after_plugin")
    config.pluginmanager.register(OutcomeCachePlugin(config), "outcome_cache_plugin")
    skip_passed = config.getoption("--skip-passed")
    skip_failed = config.getoption("--skip-failed")
    skip_failed_with = config.getoption("--skip-failed-with")
    skip_failed_except = config.getoption("--skip-failed-except")
    skip_first = config.getoption("--skip-first")
    if (
        skip_passed
        or skip_failed
        or skip_failed_with
        or skip_failed_except
        or skip_first
    ):
        with_exceptions = (
            {e.strip() for e in skip_failed_with.split(",")}
            if skip_failed_with
            else set()
        )
        except_exceptions = (
            {e.strip() for e in skip_failed_except.split(",")}
            if skip_failed_except
            else set()
        )
        config.pluginmanager.register(
            SkipTestsPlugin(
                config,
                skip_passed=skip_passed,
                skip_failed=skip_failed,
                skip_failed_with=with_exceptions,
                skip_failed_except=except_exceptions,
                skip_first=skip_first,
            ),
            "skip_tests_plugin",
        )


def pytest_collection_modifyitems(items):
    for item in items:
        for prefix, mark in PREFIX_MARKERS.items():
            if item.name.startswith(prefix):
                item.add_marker(getattr(pytest.mark, mark))


# ===------------------------------------------------------------------------===
# PYTEST PLUGINS
# ===------------------------------------------------------------------------===


class TargetPlugin:
    """Plugin that saves test logs and session metadata to a target output directory.

    Writes a metadata.<timestamp> JSON file at startup (date, invocation,
    environment), updates it after collection to include the full list of
    collected node ids, and overwrites it at session finish with elapsed_seconds
    added. Each test's log file includes the per-test elapsed time on the line
    immediately after the outcome.
    """

    def __init__(self, directory: Path) -> None:
        self.outputdir = directory
        self.excinfos: dict = {}
        self.start_time: float = time.monotonic()
        self.start_datetime: datetime = datetime.now()
        self.tests: dict[str, str | None] = {}
        directory.mkdir(parents=True, exist_ok=True)
        self._dump_metadata()

    def _dump_metadata(self, elapsed_seconds: float | None = None) -> None:
        path = (
            self.outputdir
            / f"metadata.{self.start_datetime.strftime('%Y-%m-%dT%H-%M-%S')}"
        )
        metadata = {
            "date": self.start_datetime.isoformat(),
            "invocation": sys.argv,
            "tests": self.tests,
            "env": dict(os.environ),
        }
        if elapsed_seconds is not None:
            metadata["elapsed_seconds"] = elapsed_seconds
        with path.open(mode="w") as f:
            json.dump(metadata, f, indent=2)

    def pytest_collection_finish(self, session) -> None:
        self.tests = {item.nodeid: None for item in session.items}
        self._dump_metadata()

    @contextmanager
    def open_log_file(self, function: str, params: list[str]) -> Iterator[TextIO]:
        if params:
            name = f"{function}_{'_'.join(params)}_{C}c_{T}t.log"
        else:
            name = f"{function}_{C}c_{T}t.log"
        path = self.outputdir / name

        with path.open(mode="w") as f:
            yield f

    def write_report(self, report, *, logfile: TextIO) -> None:
        outcome = report.outcome.upper()
        nodeid = report.nodeid

        logfile.write("=" * 80 + "\n")
        logfile.write(f"{outcome}: {nodeid}\n")
        logfile.write(f"  Elapsed: {report.duration:.3f}\n")
        if excinfo := self.excinfos.get(nodeid):
            logfile.write(f"  Exception: {excinfo.type.__name__}\n")
            logfile.write(f"  Message: {excinfo.value}\n")
        if hasattr(report.longrepr, "reprcrash"):
            crash = report.longrepr.reprcrash
            logfile.write(f"  Location: {crash.path}:{crash.lineno}\n")
        if hasattr(report.longrepr, "reprtraceback"):
            entries = report.longrepr.reprtraceback.reprentries
            if entries:
                context = next(
                    (l.lstrip("> ") for l in entries[-1].lines if l.startswith(">")),
                    None,
                )
                if context:
                    logfile.write(f"  Context: {context}\n")
        logfile.write("=" * 80 + "\n\n")
        for name, content in report.sections:
            logfile.write(divider(name))
            logfile.write(content)
            logfile.write("\n")

        if not report.failed:
            return

        if hasattr(report.longrepr, "reprtraceback"):
            logfile.write(divider("TRACEBACK"))
            logfile.write(f"{report.longrepr.reprtraceback}\n\n")

        if hasattr(report.longrepr, "reprcrash"):
            logfile.write(divider("CRASH"))
            logfile.write(f"{report.longrepr.reprcrash}\n\n")

    def pytest_runtest_makereport(self, item, call):
        if call.excinfo:
            self.excinfos[item.nodeid] = call.excinfo

    def pytest_runtest_logreport(self, report):
        if report.when != "call":
            return

        self.tests[report.nodeid] = report.outcome.upper()
        self._dump_metadata()
        testfile, function, params = parse_nodeid(report.nodeid)
        with self.open_log_file(function, params) as f:
            self.write_report(report, logfile=f)

    def pytest_sessionfinish(self, session, exitstatus):
        elapsed = time.monotonic() - self.start_time
        self._dump_metadata(elapsed_seconds=elapsed)

    def pytest_terminal_summary(self, terminalreporter, exitstatus, config):
        terminalreporter.write_sep("=", "Output")
        terminalreporter.write(
            f"Metadata and log files saved in: {self.outputdir.resolve()}\n"
        )


class ExceptionSummaryPlugin:
    """Displays a grouped summary of exception types after the test run."""

    failures: defaultdict

    def __init__(self) -> None:
        self.failures = defaultdict(int)

    def pytest_exception_interact(self, node, call, report):
        if call.excinfo:
            name = call.excinfo.type.__name__
            self.failures[name] += 1

    @pytest.hookimpl(wrapper=True, tryfirst=True)
    def pytest_terminal_summary(self, terminalreporter, exitstatus, config):
        yield
        if not self.failures:
            return
        terminalreporter.write_sep("=", "Exception Summary")
        for name, count in self.failures.items():
            terminalreporter.write_line(f"{name} ({count})")


class OutcomeCachePlugin:
    """Persists test outcomes to the pytest cache: None for passed, dict for failed."""

    CACHE_KEY = "hexagon_mlir/outcomes"

    def __init__(self, config) -> None:
        self.cache = config.cache
        self.outcomes: dict = self.cache.get(self.CACHE_KEY, {})

    def pytest_exception_interact(self, node, call, report):
        if call.excinfo:
            self.outcomes[node.nodeid] = {
                "exc_name": call.excinfo.type.__name__,
                "exc_message": str(call.excinfo.value),
            }
            self.cache.set(self.CACHE_KEY, self.outcomes)

    def pytest_runtest_logreport(self, report):
        if report.when == "call" and report.passed:
            self.outcomes[report.nodeid] = None
            self.cache.set(self.CACHE_KEY, self.outcomes)


class StopAfterPlugin:
    """Stops the test session after N tests have finished."""

    def __init__(self, n: int) -> None:
        self.n = n
        self.count = 0

    def pytest_runtest_logreport(self, report):
        if report.when != "call":
            return
        self.count += 1
        if self.count >= self.n:
            pytest.exit(f"Stopped after {self.n} tests.", returncode=0)


class SkipTestsPlugin:
    """Skips tests based on their previously cached outcome."""

    def __init__(
        self,
        config,
        *,
        skip_passed: bool,
        skip_failed: bool,
        skip_failed_with: set[str],
        skip_failed_except: set[str],
        skip_first: int,
    ) -> None:
        self.skip_passed = skip_passed
        self.skip_failed = skip_failed
        self.skip_failed_with = skip_failed_with
        self.skip_failed_except = skip_failed_except
        self.skip_first = skip_first
        self.count = 0
        self.outcomes: dict = config.cache.get(OutcomeCachePlugin.CACHE_KEY, {})

    def pytest_collection_modifyitems(self, config, items):
        if not self.skip_passed:
            return
        remaining, deselected = [], []
        for item in items:
            outcome = self.outcomes.get(item.nodeid)
            if outcome is None and item.nodeid in self.outcomes:
                deselected.append(item)
            else:
                remaining.append(item)
        if deselected:
            config.hook.pytest_deselected(items=deselected)
            items[:] = remaining

    def pytest_report_collectionfinish(self, config, items):
        lines = []
        if self.skip_passed:
            lines.append("skip-passed: deselecting tests that previously passed")
        if self.skip_failed:
            lines.append("skip-failed: skipping all previously failed tests")
        if self.skip_failed_with:
            lines.append(
                f"skip-failed-with: skipping previously failed tests with {', '.join(sorted(self.skip_failed_with))}"
            )
        if self.skip_failed_except:
            lines.append(
                f"skip-failed-except: skipping all previously failed tests except those with {', '.join(sorted(self.skip_failed_except))}"
            )
        if self.skip_first:
            lines.append(f"skip-first: skipping first {self.skip_first} tests")
        return lines

    def pytest_runtest_setup(self, item):
        outcome = self.outcomes.get(item.nodeid)
        if isinstance(outcome, dict):
            exc_name = outcome["exc_name"]
            if self.skip_failed:
                pytest.skip(f"Skipping: previously failed with {exc_name}")
            if self.skip_failed_with and exc_name in self.skip_failed_with:
                pytest.skip(f"Skipping: previously failed with {exc_name}")
            if self.skip_failed_except and exc_name not in self.skip_failed_except:
                pytest.skip(f"Skipping: previously failed with {exc_name}")
        if self.count < self.skip_first:
            self.count += 1
            pytest.skip(f"Skipping: first {self.skip_first} tests")
