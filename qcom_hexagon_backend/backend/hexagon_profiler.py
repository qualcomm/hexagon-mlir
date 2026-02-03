# ===- hexagon_profiler.py --------------------------------------------------===
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause.
# For more license information:
#   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
#
# ===------------------------------------------------------------------------===

import subprocess
import time
import os
import json


class HexagonProfiler:
    """Hexagon Profiler"""

    def __init__(self, etm_local_dir, profiling_mode=None, kernel_name=""):
        raise NotImplementedError("HexagonProfiler is not implemented yet.")
