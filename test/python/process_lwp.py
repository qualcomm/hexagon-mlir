# ===- process_lwp.py -------------------------------------------------------===
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause.
# For more license information:
#   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
#
# ===------------------------------------------------------------------------===

import json
import csv
import sys
from collections import defaultdict
import re
import subprocess
import os
import glob


def process_json(json_filename):
    with open(json_filename, "r") as f:
        data = json.load(f)

    entries = data["entries"]

    # Stack to track open entries and their nesting
    stack = []
    summary = {}

    # Process entries to find valid pairs and nesting
    for entry in entries:
        current_id = entry["id"]
        current_cyc = entry["cyc"]
        # Check if the current loop is closing
        if stack and stack[-1][0] == current_id:
            # Pop the matching start entry from the stack
            start_id, start_cyc, parent_id = stack.pop()
            duration = current_cyc - start_cyc
            # Initialize summary entry if not already present
            if current_id not in summary:
                summary[current_id] = {"pcycles": 0, "iter_count": 0, "parent_id": None}
            summary[current_id]["pcycles"] += duration
            summary[current_id]["iter_count"] += 1
            # Set parent ID if it's not already set
            if parent_id is not None and summary[current_id]["parent_id"] is None:
                summary[current_id]["parent_id"] = parent_id
        else:
            # Check for interleaved loops (same ID already open in stack)
            open_ids = [item[0] for item in stack]
            assert current_id not in open_ids, (
                f"==> Interleaved loop detected! Loop {current_id} is starting again before closing previous instance. "
                f"Currently open loops: {open_ids}"
            )

            # Determine parent ID (top of stack if available)
            parent_id = stack[-1][0] if stack else None
            # Push current loop start info onto the stack
            stack.append((current_id, current_cyc, parent_id))
    return summary


def process_dump_file(dump_file):
    # Parse location file
    id_to_location = defaultdict(list)
    id_to_ops = defaultdict(list)

    with open(dump_file, "r") as loc_file:
        for line in loc_file:
            line = line.strip()
            # Split the line into location and ops parts
            if " | Collected ops:" in line:
                loc_part, ops_part = line.split(" | Collected ops:", 1)

                # Parse location and ID
                loc_match = re.search(
                    r"Location ([\d,\s]+) corresponds to ID (\d+)", loc_part
                )
                if loc_match:
                    locations_str = loc_match.group(1)
                    loop_id = int(loc_match.group(2))
                    locations = [int(loc.strip()) for loc in locations_str.split(",")]
                    id_to_location[loop_id] = locations

                    # Parse ops (can be empty)
                    ops = [op.strip() for op in ops_part.split(",") if op.strip()]
                    id_to_ops[loop_id] = ops
    return id_to_location, id_to_ops


# Write results to CSV
def write_to_csv(summary, id_to_location, id_to_ops):
    with open("/tmp/lwp_output.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["ID", "pcycles", "iter_count", "parent-id", "line_num", "ops"])
        for loop_id, values in summary.items():
            parent_id = summary[loop_id]["parent_id"] or "-"
            location = id_to_location.get(loop_id, [])
            ops = id_to_ops.get(loop_id, [])
            writer.writerow(
                [
                    loop_id,
                    values["pcycles"],
                    values["iter_count"],
                    parent_id,
                    location,
                    ops,
                ]
            )

    print(
        "==> LWP data is processed and CSV file '/tmp/lwp_output.csv' has been created."
    )


def process_mlirbc(mlir_file):
    # Construct the new filename by inserting '-new' before the extension
    base_name, ext = os.path.splitext(mlir_file)
    new_mlir_file = f"{base_name}-noconst{ext}"

    # Run the shell command to filter out lines starting with 'torch_tensor_'
    command = f'egrep -v "^ *torch_tensor_" "{mlir_file}" > "{new_mlir_file}"'
    subprocess.run(command, shell=True, check=True)

    print(f"==> Constants removed in initial linalg IR, saved as: {new_mlir_file}")


def process_lwp_data(json_filename, dump_file, initial_linalg_IR=None):
    if initial_linalg_IR:
        print(f"Using initial linalg file: {initial_linalg_IR}")
        process_mlirbc(initial_linalg_IR)
    else:
        print("No initial linalg file provided.")

    summary = process_json(json_filename)
    id_to_location, id_to_ops = process_dump_file(dump_file)
    write_to_csv(summary, id_to_location, id_to_ops)


if __name__ == "__main__":
    if len(sys.argv) == 3 or len(sys.argv) == 4:
        input_file = sys.argv[1]
        dump_file = sys.argv[2]
        initial_linalg_file = sys.argv[3] if len(sys.argv) == 4 else None
        process_lwp_data(input_file, dump_file, initial_linalg_file)
    else:
        print(
            "==> Invalid number of arguments. Usage: python process_lwp.py <input_file.json> <dump_file.txt> [<initial_linalg.mlir>]"
        )
        print(
            "==> For hexagon-mlir pipeline with lwp enabled, <input_file.json> can be found in /tmp/lwp.json and <dump_file.txt> can be found in /tmp/lwp_infodump.txt."
        )
        print(
            "==> Passing <initial_linalg.mlir> is optional and not valid for triton workflow. It can also be found in /tmp/initial-linalg.mlir, if applicable"
        )
