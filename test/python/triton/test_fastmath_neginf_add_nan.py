# ===- test_fastmath_neginf_add_nan.py ----------------------------------------===
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause.
# For more license information:
#   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
#
# ===------------------------------------------------------------------------===
#
# Regression tests for removing -ffast-math from compilation flags.
#
# Two failure modes share the same root cause (-ffast-math enables the nnan/ninf
# LLVM attributes, letting the backend assume inputs are never NaN or ±inf):
#
# Issue 1 — add: -inf + (-finfo.max) = nan instead of -inf.
#
#   tmp0 = tl.load(...)   # where_1: may be -inf (masked logits)
#   tmp1 = tl.load(...)   # where:   0.0 or -finfo(float32).max (causal bias)
#   tmp2 = tmp0 + tmp1    # BUG: -inf + (-finfo.max) = nan under -ffast-math
#
# Issue 2 — exp: exp(-inf - finite) = nan instead of 0.
#
#   S[i, j] = -inf for masked positions; exp(S - m[:,None]) = nan under
#   -ffast-math because nnan lets the backend skip the special -inf path.
#
# IEEE 754 requires: -inf + finite = -inf; exp(-inf) = 0.
#
# The fix replaces -ffast-math -fno-finite-math-only with -fno-math-errno,
# preserving performance-safe errno removal while restoring correct inf/NaN
# semantics.
