//===- fast_inverse_rmsnorm_pipeline.mlir ---------------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

// Integration test for the lower-libdevice → convert-math-to-llvm →
// hexagon-fast-inverse pipeline interaction.
//
// LowerLibdevice must NOT convert scalar math.sqrt to a QHL library call,
// otherwise FastInverse cannot recognise divf(1.0, sqrt(x)) and the Quake
// rsqrt optimisation does not fire. If LowerScalarMathPattern registers
// math::SqrtOp, math.sqrt becomes an opaque QHL call (alloca + store +
// call + load) that FastInverse cannot see through → arith.divf remains.
//
// The Python driver is used because lower-libdevice is not registered in
// linalg-hexagon-opt (it depends on the Triton tt dialect ecosystem).
// FileCheck directives live here; the Python script prints transformed IR.

// RUN: TORCH_DEVICE_BACKEND_AUTOLOAD=0 %PYTHON %S/fast-inverse-rmsnorm-pipeline-python.py \
// RUN:   | FileCheck %s

// --- Scalar: divf(1.0, sqrt(x)) → Quake_rsqrt(x) ---
// CHECK: success=True
// CHECK-LABEL: func.func @scalar_div_one_by_sqrt
// Quake rsqrt bit-trick: bitcast → shift right → subtract magic constant
// CHECK:         arith.bitcast %{{.*}} : f32 to i32
// CHECK:         arith.shrui %{{.*}}, %c1_i32
// CHECK:         arith.subi %c1597463007_i32
// CHECK:         arith.bitcast %{{.*}} : i32 to f32
// Newton-Raphson refinement (2 iterations):
// CHECK:         arith.mulf
// CHECK:         arith.subf
// CHECK:         arith.mulf
// No division should remain:
// CHECK-NOT:     arith.divf
