// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(hexagon-fast-inverse)' | FileCheck %s

// This test converts the fp32 vector division with denominator as broadcast operation into
// a scalar div, and then broadcast the result followed by multiplication with numerator.
// CHECK-LABEL:   func.func @test_scalar_division_f32(
// CHECK-SAME:                                        %[[VAL_0:.*]]: vector<128xf32>,
// CHECK-SAME:                                        %[[VAL_1:.*]]: memref<f32>,
// CHECK-SAME:                                        %[[VAL_2:.*]]: memref<128xf32>) {
// CHECK:           %[[VAL_3:.*]] = arith.constant 1 : i32
// CHECK:           %[[VAL_4:.*]] = arith.constant 5.000000e-01 : f32
// CHECK:           %[[VAL_5:.*]] = arith.constant 1.500000e+00 : f32
// CHECK:           %[[VAL_6:.*]] = arith.constant 1597463007 : i32
// CHECK:           %[[VAL_7:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_8:.*]] = memref.load %[[VAL_1]][] : memref<f32>
// CHECK:           %[[VAL_9:.*]] = arith.mulf %[[VAL_8]], %[[VAL_8]] : f32
// CHECK:           %[[VAL_10:.*]] = arith.bitcast %[[VAL_9]] : f32 to i32
// CHECK:           %[[VAL_11:.*]] = arith.shrui %[[VAL_10]], %[[VAL_3]] : i32
// CHECK:           %[[VAL_12:.*]] = arith.subi %[[VAL_6]], %[[VAL_11]] : i32
// CHECK:           %[[VAL_13:.*]] = arith.bitcast %[[VAL_12]] : i32 to f32
// CHECK:           %[[VAL_14:.*]] = arith.mulf %[[VAL_13]], %[[VAL_13]] : f32
// CHECK:           %[[VAL_15:.*]] = arith.mulf %[[VAL_9]], %[[VAL_14]] : f32
// CHECK:           %[[VAL_16:.*]] = arith.mulf %[[VAL_15]], %[[VAL_4]] : f32
// CHECK:           %[[VAL_17:.*]] = arith.subf %[[VAL_5]], %[[VAL_16]] : f32
// CHECK:           %[[VAL_18:.*]] = arith.mulf %[[VAL_13]], %[[VAL_17]] : f32
// CHECK:           %[[VAL_19:.*]] = arith.mulf %[[VAL_18]], %[[VAL_18]] : f32
// CHECK:           %[[VAL_20:.*]] = arith.mulf %[[VAL_9]], %[[VAL_19]] : f32
// CHECK:           %[[VAL_21:.*]] = arith.mulf %[[VAL_20]], %[[VAL_4]] : f32
// CHECK:           %[[VAL_22:.*]] = arith.subf %[[VAL_5]], %[[VAL_21]] : f32
// CHECK:           %[[VAL_23:.*]] = arith.mulf %[[VAL_18]], %[[VAL_22]] : f32
// CHECK:           %[[VAL_24:.*]] = vector.broadcast %[[VAL_23]] : f32 to vector<128xf32>
// CHECK:           %[[VAL_25:.*]] = arith.mulf %[[VAL_0]], %[[VAL_24]] : vector<128xf32>
// CHECK:           vector.transfer_write %[[VAL_25]], %[[VAL_2]][%[[VAL_7]]] {in_bounds = [true]} : vector<128xf32>, memref<128xf32>
// CHECK:           return
// CHECK:         }
func.func @test_scalar_division_f32(%arg0: vector<128xf32>, %arg1: memref<f32>, %arg2: memref<128xf32>) {
  %c0 = arith.constant 0 : index
  %scalar = memref.load %arg1[] : memref<f32>
  // Broadcast the scalar to a vector
  %denominator = vector.broadcast %scalar : f32 to vector<128xf32>
  %div = arith.divf %arg0, %denominator : vector<128xf32>
  vector.transfer_write %div, %arg2[%c0] : vector<128xf32>, memref<128xf32>
  return
}

// This test converts the fp16 vector division with denominator as broadcast operation into
// a scalar div, and then broadcast the result followed by multiplication with numerator
// CHECK-LABEL:   func.func @test_scalar_division_f16(
// CHECK-SAME:                                        %[[VAL_0:.*]]: vector<128xf16>,
// CHECK-SAME:                                        %[[VAL_1:.*]]: memref<f16>,
// CHECK-SAME:                                        %[[VAL_2:.*]]: memref<128xf16>) {
// CHECK:           %[[VAL_3:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_4:.*]] = memref.load %[[VAL_1]][] : memref<f16>
// CHECK:           %[[VAL_5:.*]] = vector.broadcast %[[VAL_4]] : f16 to vector<128xf16>
// CHECK:           %[[VAL_6:.*]] = arith.divf %[[VAL_0]], %[[VAL_5]] : vector<128xf16>
// CHECK:           vector.transfer_write %[[VAL_6]], %[[VAL_2]]{{\[}}%[[VAL_3]]] {in_bounds = [true]} : vector<128xf16>, memref<128xf16>
// CHECK:           return
// CHECK:         }
func.func @test_scalar_division_f16(%arg0: vector<128xf16>, %arg1: memref<f16>, %arg2: memref<128xf16>) {
  %c0 = arith.constant 0 : index
  %scalar = memref.load %arg1[] : memref<f16>
  // Broadcast the scalar to a vector
  %denominator = vector.broadcast %scalar : f16 to vector<128xf16>
  %div = arith.divf %arg0, %denominator : vector<128xf16>
  vector.transfer_write %div, %arg2[%c0] : vector<128xf16>, memref<128xf16>
  return
}

// This test converts the fp32 vector division into numerator * reciprocal
// where reciprocal is computed using fast inverse logic.
// CHECK-LABEL:   func.func @test_fast_inverse_division_f32(
// CHECK:         %[[VAL_3:.*]] = arith.constant dense<1> : vector<256xi32>
// CHECK:         %[[VAL_4:.*]] = arith.constant dense<5.000000e-01> : vector<256xf32>
// CHECK:         %[[VAL_5:.*]] = arith.constant dense<1.500000e+00> : vector<256xf32>
// CHECK:         %[[VAL_6:.*]] = arith.constant dense<1597463007> : vector<256xi32>
// CHECK:         %[[VAL_7:.*]] = arith.constant 0 : index
// CHECK:         %[[VAL_8:.*]] = arith.mulf %[[VAL_1]], %[[VAL_1]] : vector<256xf32>
// CHECK:         %[[VAL_9:.*]] = arith.bitcast %[[VAL_8]] : vector<256xf32> to vector<256xi32>
// CHECK:         %[[VAL_10:.*]] = arith.shrui %[[VAL_9]], %[[VAL_3]] : vector<256xi32>
// CHECK:         %[[VAL_11:.*]] = arith.subi %[[VAL_6]], %[[VAL_10]] : vector<256xi32>
// CHECK:         %[[VAL_12:.*]] = arith.bitcast %[[VAL_11]] : vector<256xi32> to vector<256xf32>
// CHECK:         %[[VAL_13:.*]] = arith.mulf %[[VAL_12]], %[[VAL_12]] : vector<256xf32>
// CHECK:         %[[VAL_14:.*]] = arith.mulf %[[VAL_8]], %[[VAL_13]] : vector<256xf32>
// CHECK:         %[[VAL_15:.*]] = arith.mulf %[[VAL_14]], %[[VAL_4]] : vector<256xf32>
// CHECK:         %[[VAL_16:.*]] = arith.subf %[[VAL_5]], %[[VAL_15]] : vector<256xf32>
// CHECK:         %[[VAL_17:.*]] = arith.mulf %[[VAL_12]], %[[VAL_16]] : vector<256xf32>
// CHECK:         %[[VAL_18:.*]] = arith.mulf %[[VAL_17]], %[[VAL_17]] : vector<256xf32>
// CHECK:         %[[VAL_19:.*]] = arith.mulf %[[VAL_8]], %[[VAL_18]] : vector<256xf32>
// CHECK:         %[[VAL_20:.*]] = arith.mulf %[[VAL_19]], %[[VAL_4]] : vector<256xf32>
// CHECK:         %[[VAL_21:.*]] = arith.subf %[[VAL_5]], %[[VAL_20]] : vector<256xf32>
// CHECK:         %[[VAL_22:.*]] = arith.mulf %[[VAL_17]], %[[VAL_21]] : vector<256xf32>
// CHECK:         %[[VAL_23:.*]] = arith.mulf %[[VAL_0]], %[[VAL_22]] : vector<256xf32>
// CHECK:         vector.transfer_write %[[VAL_23]], %[[VAL_2]][%[[VAL_7]]] {in_bounds = [true]} : vector<256xf32>, memref<256xf32>
// CHECK:         return    
// CHECK:       }
func.func @test_fast_inverse_division_f32(%arg0: vector<256xf32>, %arg1: vector<256xf32>, %arg2: memref<256xf32>) {
  %c0 = arith.constant 0 : index
  %div = arith.divf %arg0, %arg1 : vector<256xf32>
  vector.transfer_write %div, %arg2[%c0] : vector<256xf32>, memref<256xf32>
  return
}


// This test ensures that the fp32 division of a constant 1.0f by sqrt(a)
// is transformed directly into the code that implements
// Quake III's Fast Inverse Square Root algorithm
// CHECK-LABEL: func.func @test_one_div_sqrt_f32(
// CHECK-SAME:  %[[A:.*]]: f32)
// For computing the first approximation
// CHECK: %[[I0:.*]] = arith.bitcast %[[A]] : f32 to i32
// CHECK: %[[I1:.*]] = arith.shrui %[[I0]], %c1_i32 : i32
// CHECK: %[[I2:.*]] = arith.subi %c1597463007_i32, %[[I1]] : i32
// CHECK: %[[Y0:.*]] = arith.bitcast %[[I2]] : i32 to f32
// For Newton's first iteration
// CHECK: arith.mulf
// CHECK: arith.subf
// CHECK: arith.mulf
// For Newton's second iteration
// CHECK: arith.mulf
// CHECK: arith.subf
// CHECK: arith.mulf
// Important: ensuring that there is no divf produced
// CHECK-NOT: arith.divf
// Important: ensuring that there is no llvm.intr.sqrt produced
// CHECK-NOT: llvm.intr.sqrt
func.func @test_one_div_sqrt_f32(%arg0: f32) -> f32 {
  %cst = arith.constant 1.000000e+00 : f32
  %1 = llvm.intr.sqrt(%arg0) {fastmathFlags = #llvm.fastmath<fast>} : (f32) -> f32
  %2 = arith.divf %cst, %1 fastmath<fast> : f32
  return %2 : f32
}


// This test ensures that the fp32 division of a variable x by sqrt(a)
// is transformed into x multiplied by the code that implements
// Quake III's Fast Inverse Square Root algorithm
// CHECK-LABEL: func.func @test_var_div_sqrt_f32(
// CHECK-SAME:  %[[X:.*]]: f32,
// CHECK-SAME:  %[[A:.*]]: f32)
// For computing the first approximation of rsqrt(a)
// CHECK: %[[I0:.*]] = arith.bitcast %[[A]] : f32 to i32
// CHECK: %[[I1:.*]] = arith.shrui %[[I0]], %c1_i32 : i32
// CHECK: %[[I2:.*]] = arith.subi %c1597463007_i32, %[[I1]] : i32
// CHECK: %[[Y0:.*]] = arith.bitcast %[[I2]] : i32 to f32
// For Newton's first iteration
// CHECK: arith.mulf
// CHECK: arith.subf
// CHECK: arith.mulf
// For Newton's second iteration
// CHECK: arith.mulf
// CHECK: arith.subf
// CHECK: arith.mulf
// Finally, x is multiplied by rsqrt(a)
// CHECK: arith.mulf %[[X]],
// Important: ensuring that there is no divf produced
// CHECK-NOT: arith.divf
// Important: ensuring that there is no llvm.intr.sqrt produced
// CHECK-NOT: llvm.intr.sqrt
func.func @test_var_div_sqrt_f32(%arg0: f32, %arg1: f32) -> f32 {
  // %arg0 = x
  // %arg1 = a
  %1 = llvm.intr.sqrt(%arg1) {fastmathFlags = #llvm.fastmath<fast>} : (f32) -> f32
  %2 = arith.divf %arg0, %1 fastmath<fast> : f32
  return %2 : f32
}
