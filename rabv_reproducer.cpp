// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt

// Standalone reproducer for semgrep RABV false-positive alerts in
// qualcomm/hexagon-mlir PR #111.
//
// Rules triggered:
//   reusable-workflows.semgrep_rules.cxx.rabv.check.array-access-before-reject-bounds-check
//   reusable-workflows.semgrep_rules.cxx.rabv.check.array-access-before-allow-bounds-check
//
// To reproduce:
//   semgrep --config <path-to-qcom-rabv-rules> rabv_reproducer.cpp
//
// CLAIM: both patterns below are SAFE.  The bounds of getShape() are
// guaranteed by the getRank() check that precedes each access.  The rule
// fires because it identifies the LATER value-validity checks (e.g.
// "K <= 0", "sz >= N") as "bounds checks that should precede the access".
// Those checks are semantic/range checks on the extracted values, not array
// bounds checks.  We believe this is a false positive.

#include <cstddef>
#include <cstdint>
#include <vector>

// -----------------------------------------------------------------
// Minimal stand-ins for the MLIR types used in the real code
// (mlir::ShapedType / mlir::MemRefType).
// -----------------------------------------------------------------
struct Shape {
    std::vector<int64_t> dims;
    int64_t operator[](size_t i) const { return dims[i]; }
    size_t  size()               const { return dims.size(); }
};

struct ShapedTy {
    Shape shape;
    Shape getShape() const { return shape; }
    int   getRank()  const { return (int)shape.size(); }
};

struct Status { bool ok; };
static Status failure()      { return {false}; }
static Status success()      { return {true};  }
static Status matchFailure() { return {false}; }

// -----------------------------------------------------------------
// Pattern 1 – mirrors MatmulToHexKLPass.cpp lines 315-349.
//
// The CI flags line 318 for BOTH "reject" and "allow" variants.
// -----------------------------------------------------------------
Status matmul_pattern(ShapedTy A, ShapedTy B, bool macroMode) {

    // ✅ Bounds guaranteed: rank >= 2  ⟹  getShape().size() >= 2
    if (A.getRank() < 2 || B.getRank() < 2)
        return failure();

    // ❌ CI flags this line:
    //   "Array 'A.getShape()' accessed with index '0' before a later
    //    bounds/reject check."
    int64_t M = A.getShape()[0]; // nosemgrep: reusable-workflows.semgrep_rules.cxx.rabv.check.array-access-before-reject-bounds-check,reusable-workflows.semgrep_rules.cxx.rabv.check.array-access-before-allow-bounds-check
    int64_t K = A.getShape()[1];
    int64_t N = B.getShape()[1];

    // The rule appears to treat this VALUE check on K as the
    // "later bounds check" that should precede the access to [0].
    if (K <= 0 || K % 32 != 0)
        return matchFailure();

    if (macroMode) {
        const int64_t MAX_M = 1600;
        // And/or this check on M as the "later reject bounds check":
        if (M > MAX_M)
            return matchFailure();
    }

    (void)N;
    return success();
}

// -----------------------------------------------------------------
// Pattern 2 – mirrors HoistLoopInvariantAllocPass.cpp lines 165-200.
//
// The CI flags line 168 for BOTH "reject" and "allow" variants.
// -----------------------------------------------------------------
Status hoist_pattern(ShapedTy allocTy, const std::vector<int64_t>& sizes) {

    // ✅ Bounds guaranteed: rank == 1  ⟹  getShape().size() == 1
    if (allocTy.getRank() != 1)
        return failure();
    // NFC
    // ❌ CI flags this line:
    //   "Array 'allocTy.getShape()' accessed with index '0' before a later
    //    allow-style bounds check."
    int64_t N = allocTy.getShape()[0]; // nosemgrep: reusable-workflows.semgrep_rules.cxx.rabv.check.array-access-before-reject-bounds-check,reusable-workflows.semgrep_rules.cxx.rabv.check.array-access-before-allow-bounds-check

    // The rule treats this loop comparison (sz >= N) as the
    // "later allow-style bounds check" that should precede the access.
    for (int64_t sz : sizes) {
        if (sz >= N)   // N used as upper bound — this is the "allow" check
            continue;
        // ... process sz ...
    }

    // And this as the "later reject bounds check":
    if (N <= 0)
        return matchFailure();

    return success();
}
