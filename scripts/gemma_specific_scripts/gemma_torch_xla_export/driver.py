#!/usr/bin/env python3

# ===- driver.py -----------------------------------------------------------===
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause.
# For more license information:
#   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
#
# ===------------------------------------------------------------------------===

"""Export a fixed-shape Gemma4 text forward pass to StableHLO and Linalg.
Also supports benchmarking StableHLO launcher execution vs. baseline torch
To see supported options, run:
  python scripts/gemma_specific_scripts/gemma_torch_xla_export/driver.py --help
"""

from __future__ import annotations

import sys
from pathlib import Path

# To resolve our Triton shadowing the Triton namespace used by Torch 2.9
SCRIPT_DIR = Path(__file__).resolve().parent
for _candidate in (SCRIPT_DIR, *SCRIPT_DIR.parents):
    if (_candidate / "triton").is_dir():
        REPO_ROOT = _candidate
        break
else:
    raise RuntimeError(
        f"Could not find repository root containing triton/: {SCRIPT_DIR}"
    )
_cleaned_path = []
for _entry in sys.path:
    _path = Path.cwd() if not _entry else Path(_entry)
    try:
        if _path.resolve() == REPO_ROOT:
            continue
    except OSError:
        pass
    _cleaned_path.append(_entry)
sys.path[:] = [str(SCRIPT_DIR), *_cleaned_path]

import options as export_options  # pylint: disable=wrong-import-position
import pipeline  # pylint: disable=wrong-import-position


def main() -> None:
    pipeline.run(export_options.parse_args())


if __name__ == "__main__":
    main()
