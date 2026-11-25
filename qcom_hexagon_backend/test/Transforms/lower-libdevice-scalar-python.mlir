//===- lower-libdevice-scalar-python.mlir ---------------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

// RUN: TORCH_DEVICE_BACKEND_AUTOLOAD=0 %PYTHON %S/lower-libdevice-scalar-python.py | FileCheck %s

// FileCheck directives for lower-libdevice-scalar-python.py.
// The Python driver runs run_lower_libdevice() on each embedded module and
// prints the transformed MLIR to stdout.

// CHECK: success=True

// --- f32 scalar: math.sin → qhmath_hvx_sin_af ---------------------------------
// CHECK:       llvm.func @qhmath_hvx_sin_af(!llvm.ptr, !llvm.ptr, i32) -> i32
// CHECK-LABEL: func.func @scalar_sin_f32
// CHECK-NOT:   math.sin
// CHECK:       memref.alloca() : memref<1xf32>
// CHECK:       memref.alloca() : memref<1xf32>
// CHECK:       memref.store
// CHECK:       llvm.call @qhmath_hvx_sin_af(%{{.*}}, %{{.*}}, %{{.*}}) : (!llvm.ptr, !llvm.ptr, i32) -> i32
// CHECK:       memref.load

// --- f32 scalar: math.cos → qhmath_hvx_cos_af ---------------------------------
// CHECK: success=True
// CHECK:       llvm.func @qhmath_hvx_cos_af(!llvm.ptr, !llvm.ptr, i32) -> i32
// CHECK-LABEL: func.func @scalar_cos_f32
// CHECK-NOT:   math.cos
// CHECK:       memref.alloca() : memref<1xf32>
// CHECK:       memref.alloca() : memref<1xf32>
// CHECK:       memref.store
// CHECK:       llvm.call @qhmath_hvx_cos_af(%{{.*}}, %{{.*}}, %{{.*}}) : (!llvm.ptr, !llvm.ptr, i32) -> i32
// CHECK:       memref.load

// --- bf16 scalar: extf bf16→f32, call _af, truncf f32→bf16 -------------------
// CHECK: success=True
// CHECK-LABEL: func.func @scalar_sin_bf16
// CHECK-NOT:   math.sin
// CHECK:       arith.extf %arg0 : bf16 to f32
// CHECK:       memref.alloca() : memref<1xf32>
// CHECK:       memref.alloca() : memref<1xf32>
// CHECK:       memref.store
// CHECK:       llvm.call @qhmath_hvx_sin_af(%{{.*}}, %{{.*}}, %{{.*}}) : (!llvm.ptr, !llvm.ptr, i32) -> i32
// CHECK:       %[[RES:.*]] = memref.load
// CHECK:       arith.truncf %[[RES]] : f32 to bf16

// --- f16 scalar: math.sin → qhmath_hvx_sin_ahf --------------------------------
// CHECK: success=True
// CHECK:       llvm.func @qhmath_hvx_sin_ahf(!llvm.ptr, !llvm.ptr, i32) -> i32
// CHECK-LABEL: func.func @scalar_sin_f16
// CHECK-NOT:   math.sin
// CHECK:       memref.alloca() : memref<1xf16>
// CHECK:       memref.alloca() : memref<1xf16>
// CHECK:       memref.store
// CHECK:       llvm.call @qhmath_hvx_sin_ahf(%{{.*}}, %{{.*}}, %{{.*}}) : (!llvm.ptr, !llvm.ptr, i32) -> i32
// CHECK:       memref.load

// --- inside linalg.generic: math.sin must NOT be rewritten -------------------
// CHECK: success=True
// CHECK-LABEL: func.func @scalar_sin_inside_linalg_untouched
// CHECK:       linalg.generic
// CHECK:         math.sin

// --- Scalar tt.extern_elementwise: f32 → qhmath_hvx_rsqrt_af -----------------
// CHECK: success=True
// CHECK: llvm.func @qhmath_hvx_rsqrt_af(!llvm.ptr, !llvm.ptr, i32) -> i32
// CHECK-LABEL: func.func @scalar_rsqrt_f32
// CHECK-SAME:    attributes {hexagon.uses_libdevice}
// CHECK:         memref.alloc() : memref<1xf32>
// CHECK:         memref.alloc() : memref<1xf32>
// CHECK:         memref.store %arg0
// CHECK:         llvm.call @qhmath_hvx_rsqrt_af(%{{.*}}, %{{.*}}, %{{.*}}) : (!llvm.ptr, !llvm.ptr, i32) -> i32
// CHECK:         memref.load
// CHECK-NOT:     tt.extern_elementwise

// --- bf16 scalar (extf -> f32 -> call -> truncf) ----------------------------
// CHECK: success=True
// CHECK-LABEL: func.func @scalar_rsqrt_bf16
// CHECK-SAME:    attributes {hexagon.uses_libdevice}
// CHECK:         memref.alloc() : memref<1xf32>
// CHECK:         memref.alloc() : memref<1xf32>
// CHECK:         arith.extf %arg0 : bf16 to f32
// CHECK:         memref.store
// CHECK:         llvm.call @qhmath_hvx_rsqrt_af(%{{.*}}, %{{.*}}, %{{.*}}) : (!llvm.ptr, !llvm.ptr, i32) -> i32
// CHECK:         memref.load
// CHECK:         arith.truncf %{{.*}} : f32 to bf16
// CHECK-NOT:     tt.extern_elementwise

// --- fp8 scalar (extf -> f16 -> call -> truncf) -----------------------------
// CHECK: success=True
// CHECK: llvm.func @qhmath_hvx_rsqrt_ahf(!llvm.ptr, !llvm.ptr, i32) -> i32
// CHECK-LABEL: func.func @scalar_rsqrt_fp8
// CHECK-SAME:    attributes {hexagon.uses_libdevice}
// CHECK:         memref.alloc() : memref<1xf16>
// CHECK:         memref.alloc() : memref<1xf16>
// CHECK:         arith.extf %arg0 : f8E4M3FN to f16
// CHECK:         memref.store
// CHECK:         llvm.call @qhmath_hvx_rsqrt_ahf(%{{.*}}, %{{.*}}, %{{.*}}) : (!llvm.ptr, !llvm.ptr, i32) -> i32
// CHECK:         memref.load
// CHECK:         arith.truncf %{{.*}} : f16 to f8E4M3FN
// CHECK-NOT:     tt.extern_elementwise

// --- f8E5M2 scalar math.sin (extf -> f16 -> call -> truncf) -----------------
// CHECK: success=True
// CHECK-LABEL: func.func @scalar_sin_f8e5m2
// CHECK-NOT:   math.sin
// CHECK:       arith.extf %arg0 : f8E5M2 to f16
// CHECK:       memref.alloca() : memref<1xf16>
// CHECK:       memref.alloca() : memref<1xf16>
// CHECK:       memref.store
// CHECK:       llvm.call @qhmath_hvx_sin_ahf(%{{.*}}, %{{.*}}, %{{.*}}) : (!llvm.ptr, !llvm.ptr, i32) -> i32
// CHECK:       %[[RES:.*]] = memref.load
// CHECK:       arith.truncf %[[RES]] : f16 to f8E5M2

// --- f8E5M2FNUZ scalar math.sin (extf -> f16 -> call -> truncf) -------------
// CHECK: success=True
// CHECK-LABEL: func.func @scalar_sin_f8e5m2fnuz
// CHECK-NOT:   math.sin
// CHECK:       arith.extf %arg0 : f8E5M2FNUZ to f16
// CHECK:       memref.alloca() : memref<1xf16>
// CHECK:       memref.alloca() : memref<1xf16>
// CHECK:       memref.store
// CHECK:       llvm.call @qhmath_hvx_sin_ahf(%{{.*}}, %{{.*}}, %{{.*}}) : (!llvm.ptr, !llvm.ptr, i32) -> i32
// CHECK:       %[[RES:.*]] = memref.load
// CHECK:       arith.truncf %[[RES]] : f16 to f8E5M2FNUZ

// --- f8E5M2 scalar tt.extern_elementwise (extf -> f16 -> call -> truncf) ----
// CHECK: success=True
// CHECK-LABEL: func.func @scalar_rsqrt_f8e5m2
// CHECK-SAME:    attributes {hexagon.uses_libdevice}
// CHECK:         memref.alloc() : memref<1xf16>
// CHECK:         memref.alloc() : memref<1xf16>
// CHECK:         arith.extf %arg0 : f8E5M2 to f16
// CHECK:         memref.store
// CHECK:         llvm.call @qhmath_hvx_rsqrt_ahf(%{{.*}}, %{{.*}}, %{{.*}}) : (!llvm.ptr, !llvm.ptr, i32) -> i32
// CHECK:         memref.load
// CHECK:         arith.truncf %{{.*}} : f16 to f8E5M2
// CHECK-NOT:     tt.extern_elementwise

// --- f8E5M2FNUZ scalar tt.extern_elementwise (extf -> f16 -> call -> truncf)
// CHECK: success=True
// CHECK-LABEL: func.func @scalar_rsqrt_f8e5m2fnuz
// CHECK-SAME:    attributes {hexagon.uses_libdevice}
// CHECK:         memref.alloc() : memref<1xf16>
// CHECK:         memref.alloc() : memref<1xf16>
// CHECK:         arith.extf %arg0 : f8E5M2FNUZ to f16
// CHECK:         memref.store
// CHECK:         llvm.call @qhmath_hvx_rsqrt_ahf(%{{.*}}, %{{.*}}, %{{.*}}) : (!llvm.ptr, !llvm.ptr, i32) -> i32
// CHECK:         memref.load
// CHECK:         arith.truncf %{{.*}} : f16 to f8E5M2FNUZ
// CHECK-NOT:     tt.extern_elementwise

// --- tensor path (regression guard: unchanged from before) ------------------
// CHECK: success=True
// CHECK-LABEL: func.func @tensor_rsqrt
// CHECK-SAME:    attributes {hexagon.uses_libdevice}
// CHECK:         memref.alloc() : memref<4xf32>
// CHECK:         bufferization.to_buffer %arg0 : tensor<4xf32> to memref<4xf32>
// CHECK:         llvm.call @qhmath_hvx_rsqrt_af(%{{.*}}, %{{.*}}, %{{.*}}) : (!llvm.ptr, !llvm.ptr, i32) -> i32
// CHECK:         bufferization.to_tensor %{{.*}} restrict writable : memref<4xf32> to tensor<4xf32>
// CHECK-NOT:     tt.extern_elementwise
