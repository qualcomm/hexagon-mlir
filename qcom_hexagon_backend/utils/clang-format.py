# ===- clang-format.py ------------------------------------------------------===
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause.
# For more license information:
#   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
#
# ===------------------------------------------------------------------------===

# This script is supposed to be run from the root of the repository.
# After confirming, `.clang-format`exists, it will check which all files
# in ${HEXAGON_MLIR_ROOT}/qcom_hexagon_backend/ need to be formatted.
# Once, the exploration phase is done, all marked files will be formatted
# using clang-format.

# Usage: python3 qcom_hexagon_backend/utils/clang-format.py [--add-to-git]

import os
import subprocess
import logging
import sys

logging.basicConfig(level=logging.INFO)


def check_clang_format():
    try:
        subprocess.run(
            ["clang-format", "--version"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        logging.info("clang-format is installed.")
        return True
    except subprocess.CalledProcessError:
        logging.error("clang-format is not installed.")
        return False


def check_file_needs_formatting(file_path):
    try:
        result = subprocess.run(
            ["clang-format", "--dry-run", "-Werror", file_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if result.returncode != 0:
            logging.info(f"File needs formatting: {file_path}")
            return True
        else:
            return False
    except Exception as e:
        logging.error(f"Error checking file {file_path}: {e}")
        return False


def find_files_to_format(root_dir):
    if not os.path.exists(root_dir):
        logging.error(f"Directory {root_dir} does not exist.")
        return []

    files_to_format = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith((".cpp", ".h", ".c", ".cc")):
                full_path = os.path.join(dirpath, filename)
                if check_file_needs_formatting(full_path):
                    files_to_format.append(full_path)
    return files_to_format


def format_files(files, add_to_git=False):
    for file_path in files:
        try:
            subprocess.run(["clang-format", "-i", file_path], check=True)
            logging.info(f"Formatted file: {file_path}")
            # Optionally, add the formatted file to git
            if add_to_git:
                subprocess.run(["git", "add", file_path], check=True)
        except subprocess.CalledProcessError as e:
            logging.error(f"Error formatting file {file_path}: {e}")


def main():
    if not check_clang_format():
        sys.exit(1)

    hexagon_mlir_root = os.getenv("HEXAGON_MLIR_ROOT")
    if not hexagon_mlir_root:
        logging.error("Environment variable HEXAGON_MLIR_ROOT is not set.")
        sys.exit(1)

    # There might be a --add-to-git flag to add formatted files to git
    if len(sys.argv) > 1 and sys.argv[1] == "--add-to-git":
        logging.info("You have requested to add formatted files to git.")
        add_to_git = True
    else:
        add_to_git = False

    files_to_format = find_files_to_format(hexagon_mlir_root + "/qcom_hexagon_backend")
    if files_to_format:
        logging.info(f"Found {len(files_to_format)} files to format.")
        format_files(files_to_format, add_to_git=add_to_git)
    else:
        logging.info("No files need formatting.")


if __name__ == "__main__":
    main()
