// RUN: linalg-hexagon-opt %s -split-input-file -pass-pipeline='builtin.module(small-exponent-to-multiply)' | FileCheck %s

// Simple scalar expansion: x^3 -> x*x*x
func.func @scalar_cube (%x:f32) -> f32 {
    %c3 = arith.constant 3 : i64
    %res = math.fpowi %x, %c3 : f32, i64
    return %res : f32
}

// CHECK-LABEL:  @scalar_cube
// CHECK: [[RESULT:%.+]] = arith.mulf [[X:%.*]], [[X]] : f32
// CHECK: [[RESULT2:%.+]] = arith.mulf [[X]], [[RESULT]] : f32
// CHECK: return [[RESULT2]] : f32

// -----
// Scalar negative exponent: x^-2 -> 1/(x*x)
func.func @scalar_square_neg (%x:f32) -> f32 {
    %cm2 = arith.constant -2 : i64
    %res = math.fpowi %x, %cm2 : f32, i64
    return %res : f32
}

// CHECK-LABEL:  @scalar_square_neg
// CHECK: [[ONE:%.+]] = arith.constant 1.000000e+00 : f32
// CHECK: [[MUL:%.+]] = arith.mulf [[X:%.*]], [[X]] : f32
// CHECK: [[RESULT:%.+]] = arith.divf [[ONE]], [[MUL]] : f32
// CHECK: return [[RESULT]] : f32

// -----
// Vector-splat exponent: x^4 -> x*x*x*x
func.func @vector_quad (%x: vector<4xf32>) -> vector<4xf32> {
    %c4 = arith.constant dense<4> : vector<4xi64>
    %res = math.fpowi %x, %c4 : vector<4xf32>, vector<4xi64>
    return %res : vector<4xf32>
}

// CHECK-LABEL:  @vector_quad
// CHECK: [[RESULT1:%.+]] = arith.mulf [[X:%.*]], [[X]] : vector<4xf32>
// CHECK: [[RESULT2:%.+]] = arith.mulf [[RESULT1]], [[RESULT1]] : vector<4xf32>
// CHECK: return [[RESULT2]] : vector<4xf32>

// -----
// Vector negative exponent x^-3 -> 1/(x*x*x)
func.func @vector_cube_neg (%x: vector<8xf32>) -> vector<8xf32> {
    %cm3 = arith.constant dense<-3> : vector<8xi64>
    %res = math.fpowi %x, %cm3 : vector<8xf32>, vector<8xi64>
    return %res : vector<8xf32>
}

// CHECK-LABEL:  @vector_cube_neg
// CHECK: [[ONE:%.+]] = arith.constant dense<1.000000e+00> : vector<8xf32>
// CHECK: [[MUL1:%.+]] = arith.mulf [[X:%.*]], [[X]] : vector<8xf32>
// CHECK: [[MUL2:%.+]] = arith.mulf [[X]], [[MUL1]] : vector<8xf32>
// CHECK: [[RESULT:%.+]] = arith.divf [[ONE]], [[MUL2]] : vector<8xf32>
// CHECK: return [[RESULT]] : vector<8xf32>

// -----
// Exponent too large: x^10 should not be changed
func.func @scalar_large_exp (%x:f32) -> f32 {
    %c10 = arith.constant 10 : i64
    %res = math.fpowi %x, %c10 : f32, i64
    return %res : f32
}

// CHECK-LABEL:  @scalar_large_exp
// CHECK: [[RESULT:%.+]] = math.fpowi [[X:%.*]], [[C10:%.*]] : f32, i64
// CHECK: return [[RESULT]] : f32
