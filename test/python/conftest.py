# ===- conftest.py ----------------------------------------------------------===
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause.
# For more license information:
#   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
#
# ===------------------------------------------------------------------------===
#
# Shared conftest for the on-device test suites (triton/, mlir/, torch-mlir/).
# pytest loads it for any test collected under test/python, so its hooks run
# once per invocation regardless of which suite is selected.
#
# ===------------------------------------------------------------------------===


def pytest_configure(config):
    """Push the version-locked shared libs (skel + C++ runtime) to the device
    once, before any test runs, so individual tests never re-check or re-push
    them. No-op on simulator runs and best-effort on device (see
    push_common_libs_for_session). Imported lazily so a missing backend package
    can never break collection for suites that don't need the device."""
    try:
        from triton.backends.qcom_hexagon_backend.hexagon_executor import (
            push_common_libs_for_session,
        )
    except Exception as e:  # noqa: BLE001 - optional dependency at collection time
        print(f"Warning: common-lib pre-push unavailable: {e}")
        return
    push_common_libs_for_session()
