# ===- qaic_options.py ------------------------------------------------------===
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause.
# For more license information:
#   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
#
# ===------------------------------------------------------------------------===


from dataclasses import dataclass
from .compiler import HexagonOptions


@dataclass(frozen=True)
class QAicOptions(HexagonOptions):
    """QAic-specific defaults and extensions to HexagonOptions.

    For fields shared with HexagonOptions, semantics are unchanged; only
    defaults may differ. This class may also define QAic-only flags.
    :seealso: HexagonOptions (field descriptions and behavior).
    """

    enableConvertToHexagonmem: bool = False
    enableHexagonmemCopyToDMA: bool = False
    enableHexKL: bool = False
    enableMultiThreading: bool = False
    enableSCFThreading: bool = False
    enableSplitReduction: bool = False
    enableVTCMTiling: bool = False
