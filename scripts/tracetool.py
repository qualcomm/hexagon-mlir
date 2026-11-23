#!/usr/bin/env python

# ===- tracetool.py --------------------------------------------------------===
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause.
# For more license information:
#   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
#
# ===------------------------------------------------------------------------===

import shlex
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

print(
    f"""
================================================================================
                               DEPRECATION NOTICE
================================================================================

  There is a new way to use this tool, please install it via pip inside a
  conda environment with the following command:

  $ pip install -e ".[dev]"

  After that the tools will be available in your $PATH and can be invoked
  directly with the command:

  $ tracetool {shlex.join(sys.argv[1:])}

================================================================================
""".strip()
)

try:
    from tools.tracetool import main

    sys.exit(main())
except ModuleNotFoundError as e:
    print(f"Error: {e}, please install dependencies with `pip install -e '.[dev]'`.")
    sys.exit(1)
