# ===- __init__.py ----------------------------------------------------------===
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause.
# For more license information:
#   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
#
# ===------------------------------------------------------------------------===

import logging
import warnings

from . import knobs

DEFAULT_LOG_LEVEL = logging.WARNING


def get_log_level() -> int:
    """Get default log level from the environment."""
    if level := knobs.hexagon_mlir.logs:
        return getattr(logging, level.upper(), DEFAULT_LOG_LEVEL)

    # TODO(gleite): Deprecate LOG_LEVEL in the future
    if level := knobs.hexagon_mlir.logs_deprecated:
        warnings.warn(
            "Environment varaible LOG_LEVEL is deprecated, use HEXAGON_MLIR_LOGS instead.",
            DeprecationWarning,
        )
        return getattr(logging, level.upper(), DEFAULT_LOG_LEVEL)

    return DEFAULT_LOG_LEVEL


logging.basicConfig(
    level=get_log_level(),
    format="[%(asctime)s] %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
