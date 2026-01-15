// RUN: linalg-hexagon-opt %s -split-input-file -pass-pipeline='builtin.module(func.func(hoist-scalar-ops-in-linalg-generic))' | FileCheck %s

// Test 1: Basic scalar hoisting - rsqrt should be hoisted
// CHECK-LABEL: func.func @hoist_scalar_rsqrt
func.func @hoist_scalar_rsqrt(%scalar: tensor<f32>, %vector: tensor<128xf32>, %output: tensor<128xf32>) -> tensor<128xf32> {
  // CHECK: math.rsqrt
  // CHECK: linalg.generic
  // CHECK-SAME: ins(%{{.*}}, %{{.*}}, %{{.*}} : tensor<f32>, tensor<128xf32>, tensor<f32>)
  // CHECK: arith.mulf

  %result = linalg.generic {
    indexing_maps = [affine_map<(d0) -> ()>,
                     affine_map<(d0) -> (d0)>,
                     affine_map<(d0) -> (d0)>],
    iterator_types = ["parallel"]
  } ins(%scalar, %vector : tensor<f32>, tensor<128xf32>)
    outs(%output : tensor<128xf32>) {
  ^bb0(%s: f32, %v: f32, %out: f32):
    %rsqrt = math.rsqrt %s : f32
    %mul = arith.mulf %v, %rsqrt : f32
    linalg.yield %mul : f32
  } -> tensor<128xf32>

  return %result : tensor<128xf32>
}

// -----

// Test 2: Chain of operations - both divf and rsqrt hoisted
// CHECK-LABEL: func.func @hoist_scalar_chain
func.func @hoist_scalar_chain(%scalar: tensor<f32>, %vector: tensor<128xf32>, %output: tensor<128xf32>) -> tensor<128xf32> {
  // CHECK: arith.divf
  // CHECK: math.rsqrt
  // CHECK: linalg.generic
  // CHECK-SAME: ins(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : tensor<f32>, tensor<128xf32>, tensor<f32>, tensor<f32>, tensor<f32>)
  // CHECK: arith.mulf

  %result = linalg.generic {
    indexing_maps = [affine_map<(d0) -> ()>,
                     affine_map<(d0) -> (d0)>,
                     affine_map<(d0) -> (d0)>],
    iterator_types = ["parallel"]
  } ins(%scalar, %vector : tensor<f32>, tensor<128xf32>)
    outs(%output : tensor<128xf32>) {
  ^bb0(%s: f32, %v: f32, %out: f32):
    %cst = arith.constant 2.0 : f32
    %div = arith.divf %s, %cst : f32
    %rsqrt = math.rsqrt %div : f32
    %mul = arith.mulf %v, %rsqrt : f32
    linalg.yield %mul : f32
  } -> tensor<128xf32>

  return %result : tensor<128xf32>
}

// -----

// Test 3: Multiple scalar inputs - addf and rsqrt hoisted
// CHECK-LABEL: func.func @hoist_multiple_scalars
func.func @hoist_multiple_scalars(%scalar1: tensor<f32>, %scalar2: tensor<f32>, %vector: tensor<128xf32>, %output: tensor<128xf32>) -> tensor<128xf32> {
  // CHECK: arith.addf
  // CHECK: math.rsqrt
  // CHECK: linalg.generic
  // CHECK-SAME: ins(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : tensor<f32>, tensor<f32>, tensor<128xf32>, tensor<f32>, tensor<f32>)
  // CHECK: arith.mulf

  %result = linalg.generic {
    indexing_maps = [affine_map<(d0) -> ()>,
                     affine_map<(d0) -> ()>,
                     affine_map<(d0) -> (d0)>,
                     affine_map<(d0) -> (d0)>],
    iterator_types = ["parallel"]
  } ins(%scalar1, %scalar2, %vector : tensor<f32>, tensor<f32>, tensor<128xf32>)
    outs(%output : tensor<128xf32>) {
  ^bb0(%s1: f32, %s2: f32, %v: f32, %out: f32):
    %add = arith.addf %s1, %s2 : f32
    %rsqrt = math.rsqrt %add : f32
    %mul = arith.mulf %v, %rsqrt : f32
    linalg.yield %mul : f32
  } -> tensor<128xf32>

  return %result : tensor<128xf32>
}

// -----

// Test 4: No hoisting - operations use vector elements
// CHECK-LABEL: func.func @no_hoist_loop_variant
func.func @no_hoist_loop_variant(%vector1: tensor<128xf32>, %vector2: tensor<128xf32>, %output: tensor<128xf32>) -> tensor<128xf32> {
  // CHECK: linalg.generic
  // CHECK: arith.addf
  // CHECK: math.rsqrt

  %result = linalg.generic {
    indexing_maps = [affine_map<(d0) -> (d0)>,
                     affine_map<(d0) -> (d0)>,
                     affine_map<(d0) -> (d0)>],
    iterator_types = ["parallel"]
  } ins(%vector1, %vector2 : tensor<128xf32>, tensor<128xf32>)
    outs(%output : tensor<128xf32>) {
  ^bb0(%v1: f32, %v2: f32, %out: f32):
    %add = arith.addf %v1, %v2 : f32
    %rsqrt = math.rsqrt %add : f32
    linalg.yield %rsqrt : f32
  } -> tensor<128xf32>

  return %result : tensor<128xf32>
}

// -----

// Test 5: Type-changing operations - rsqrt and truncf hoisted
// CHECK-LABEL: func.func @hoist_with_truncf
func.func @hoist_with_truncf(%scalar: tensor<f32>, %vector: tensor<128xf16>, %output: tensor<128xf16>) -> tensor<128xf16> {
  // CHECK: math.rsqrt
  // CHECK: arith.truncf
  // CHECK: linalg.generic
  // CHECK-SAME: ins(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : tensor<f32>, tensor<128xf16>, tensor<f32>, tensor<f16>)
  // CHECK: arith.mulf

  %result = linalg.generic {
    indexing_maps = [affine_map<(d0) -> ()>,
                     affine_map<(d0) -> (d0)>,
                     affine_map<(d0) -> (d0)>],
    iterator_types = ["parallel"]
  } ins(%scalar, %vector : tensor<f32>, tensor<128xf16>)
    outs(%output : tensor<128xf16>) {
  ^bb0(%s: f32, %v: f16, %out: f16):
    %rsqrt = math.rsqrt %s : f32
    %trunc = arith.truncf %rsqrt : f32 to f16
    %mul = arith.mulf %v, %trunc : f16
    linalg.yield %mul : f16
  } -> tensor<128xf16>

  return %result : tensor<128xf16>
}

// -----

// Test 6: Mixed operations - only rsqrt hoisted
// CHECK-LABEL: func.func @mixed_operations
func.func @mixed_operations(%scalar: tensor<f32>, %vector: tensor<128xf32>, %output: tensor<128xf32>) -> tensor<128xf32> {
  // CHECK: math.rsqrt
  // CHECK: linalg.generic
  // CHECK-SAME: ins(%{{.*}}, %{{.*}}, %{{.*}} : tensor<f32>, tensor<128xf32>, tensor<f32>)
  // CHECK: arith.addf
  // CHECK: arith.mulf

  %result = linalg.generic {
    indexing_maps = [affine_map<(d0) -> ()>,
                     affine_map<(d0) -> (d0)>,
                     affine_map<(d0) -> (d0)>],
    iterator_types = ["parallel"]
  } ins(%scalar, %vector : tensor<f32>, tensor<128xf32>)
    outs(%output : tensor<128xf32>) {
  ^bb0(%s: f32, %v: f32, %out: f32):
    %rsqrt = math.rsqrt %s : f32
    %add = arith.addf %v, %rsqrt : f32
    %mul = arith.mulf %add, %rsqrt : f32
    linalg.yield %mul : f32
  } -> tensor<128xf32>

  return %result : tensor<128xf32>
}

// -----

// Test 7: Integer operations - muli hoisted
// CHECK-LABEL: func.func @hoist_integer_ops
func.func @hoist_integer_ops(%scalar: tensor<i32>, %vector: tensor<128xi32>, %output: tensor<128xi32>) -> tensor<128xi32> {
  // CHECK: arith.constant 2
  // CHECK: arith.muli
  // CHECK: linalg.generic
  // CHECK-SAME: ins(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : tensor<i32>, tensor<128xi32>, tensor<i32>, tensor<i32>)
  // CHECK: arith.addi

  %result = linalg.generic {
    indexing_maps = [affine_map<(d0) -> ()>,
                     affine_map<(d0) -> (d0)>,
                     affine_map<(d0) -> (d0)>],
    iterator_types = ["parallel"]
  } ins(%scalar, %vector : tensor<i32>, tensor<128xi32>)
    outs(%output : tensor<128xi32>) {
  ^bb0(%s: i32, %v: i32, %out: i32):
    %cst = arith.constant 2 : i32
    %mul_scalar = arith.muli %s, %cst : i32
    %add = arith.addi %v, %mul_scalar : i32
    linalg.yield %add : i32
  } -> tensor<128xi32>

  return %result : tensor<128xi32>
}