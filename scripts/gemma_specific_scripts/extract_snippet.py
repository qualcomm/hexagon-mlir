#!/usr/bin/env python3

# ===- extract_snippet.py --------------------------------------------------===
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause.
# For more license information:
#   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
#
# ===------------------------------------------------------------------------===

"""Extract IR snapshots at key pipeline stages from an MLIR pass dump."""

import argparse
import re
import sys
from pathlib import Path

DEFAULT_INPUT = None

IR_DUMP_RE = re.compile(r"^// -----// IR Dump ")

# (pass_header_substring, output_filename_stem)
# "before X" stages capture the pass immediately preceding X in the pipeline:
#   before_fusion       = after InlinerPass        (runs just before HexagonFusion)
#   before_vectorization = after DisableI1VectorizationPass (runs just before HexagonVectorization)
STAGES = [
    ("OneShotBufferizePass", "after_oneshot_bufferize"),
    ("LinalgGeneralize", "after_linalg_generalize"),
    ("FindIdenticalLinalgGenerics", "after_identical_linalg_generics"),
    ("OutlineLinalgGenericPass", "after_outlining"),
    ("InlinerPass", "before_fusion"),
    ("HexagonFusion", "after_fusion"),
    ("DisableI1VectorizationPass", "before_vectorization"),
    ("HexagonVectorization", "after_vectorization"),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input", nargs="?", default=None, help="Path to the MLIR pass dump file"
    )
    parser.add_argument(
        "--log-file",
        "-l",
        default=None,
        help="Path to the MLIR pass dump file (alternative to positional arg)",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default=None,
        help="Directory to write outputs (default: same dir as input)",
    )
    args = parser.parse_args()

    raw = args.log_file or args.input
    if not raw:
        parser.error("input file required (pass as positional arg or --log-file)")
    input_path = Path(raw)
    if not input_path.exists():
        print(f"ERROR: input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args.output_dir) if args.output_dir else input_path.parent

    open_files = {}  # suffix -> file handle (opened once, appended across occurrences)
    counts = {}  # suffix -> occurrence count
    written = {}  # suffix -> path

    with open(input_path) as f:
        active_suffix = None
        for line in f:
            if IR_DUMP_RE.match(line):
                active_suffix = None
                for stage_key, suffix in STAGES:
                    if stage_key in line:
                        active_suffix = suffix
                        counts[suffix] = counts.get(suffix, 0) + 1
                        if suffix not in open_files:
                            outpath = output_dir / f"{suffix}.mlir"
                            open_files[suffix] = open(outpath, "w")
                            written[suffix] = outpath
                        else:
                            open_files[suffix].write("\n")
                        open_files[suffix].write(line)
                        break
            elif active_suffix is not None:
                open_files[active_suffix].write(line)

    for fh in open_files.values():
        fh.close()

    if not written:
        print("ERROR: no matching stages found in the dump.", file=sys.stderr)
        sys.exit(1)

    for _, suffix in STAGES:
        if suffix in written:
            n = counts[suffix]
            print(
                f"Wrote {suffix} ({n} occurrence{'s' if n > 1 else ''}) -> {written[suffix]}"
            )

    missing = [suffix for _, suffix in STAGES if suffix not in written]
    if missing:
        print(
            f"WARNING: stages not found in dump: {', '.join(missing)}", file=sys.stderr
        )


if __name__ == "__main__":
    main()
