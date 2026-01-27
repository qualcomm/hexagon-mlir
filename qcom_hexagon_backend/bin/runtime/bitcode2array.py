# ===- bitcode2array.py -----------------------------------------------------===
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause.
# For more license information:
#   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
#
# ===------------------------------------------------------------------------===

import argparse
import sys


def bitcode_to_cpp_array(bitcode_file, array_name, output_file):
    try:
        with open(bitcode_file, "rb") as f:
            bitcode = f.read()
    except FileNotFoundError:
        sys.exit(f"Error reading bitcode file: {bitcode_file}")

    hex_array = ", ".join(f"0x{byte:02x}" for byte in bitcode)
    array_length = len(bitcode)

    cpp_array = f"""
    extern const unsigned char hexagon_runtime_module_{array_name}[] = {{
        {hex_array}
    }};
    extern const unsigned int hexagon_runtime_module_{array_name}_len = {array_length};
    """

    try:
        with open(output_file, "w") as f:
            f.write(cpp_array)
    except Exception:
        sys.exit(f"Could not write output_file: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Convert LLVM bitcode to C++ array.")
    parser.add_argument("bitcode_file", help="Path to the LLVM bitcode file.")
    parser.add_argument("array_name", help="Name of the C++ array.")
    parser.add_argument("output_file", help="Path to the output C++ file.")

    args = parser.parse_args()

    bitcode_to_cpp_array(args.bitcode_file, args.array_name, args.output_file)


if __name__ == "__main__":
    main()
