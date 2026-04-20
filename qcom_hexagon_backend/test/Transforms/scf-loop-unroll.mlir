// RUN: linalg-hexagon-opt %s -scf-loop-unroll='unroll-factor=4' -cse -canonicalize | FileCheck %s

// Test basic loop unrolling with constant bounds and step
// Loop: for i = 0 to 128 step 32 (trip count = 4, evenly divisible by unroll factor 4)
func.func @simple_loop_unroll(%arg0: memref<128xf32>) {
  %c0 = arith.constant 0 : index
  %c32 = arith.constant 32 : index
  %c128 = arith.constant 128 : index
  %cst = arith.constant 0.000000e+00 : f32

  scf.for %i = %c0 to %c128 step %c32 {
    %0 = vector.transfer_read %arg0[%i], %cst {in_bounds = [true]} : memref<128xf32>, vector<32xf32>
    %1 = arith.addf %0, %0 : vector<32xf32>
    vector.transfer_write %1, %arg0[%i] {in_bounds = [true]} : vector<32xf32>, memref<128xf32>
  }
  return
}

// CHECK-LABEL: func.func @simple_loop_unroll
// CHECK-DAG: %[[C96:.*]] = arith.constant 96 : index
// CHECK-DAG: %[[C64:.*]] = arith.constant 64 : index
// CHECK-DAG: %[[C32:.*]] = arith.constant 32 : index
// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[CST:.*]] = arith.constant 0.000000e+00 : f32
// CHECK: %[[V0:.*]] = vector.transfer_read %{{.*}}[%[[C0]]], %[[CST]]
// CHECK: %[[A0:.*]] = arith.addf %[[V0]], %[[V0]]
// CHECK: vector.transfer_write %[[A0]], %{{.*}}[%[[C0]]]
// CHECK: %[[V1:.*]] = vector.transfer_read %{{.*}}[%[[C32]]], %[[CST]]
// CHECK: %[[A1:.*]] = arith.addf %[[V1]], %[[V1]]
// CHECK: vector.transfer_write %[[A1]], %{{.*}}[%[[C32]]]
// CHECK: %[[V2:.*]] = vector.transfer_read %{{.*}}[%[[C64]]], %[[CST]]
// CHECK: %[[A2:.*]] = arith.addf %[[V2]], %[[V2]]
// CHECK: vector.transfer_write %[[A2]], %{{.*}}[%[[C64]]]
// CHECK: %[[V3:.*]] = vector.transfer_read %{{.*}}[%[[C96]]], %[[CST]]
// CHECK: %[[A3:.*]] = arith.addf %[[V3]], %[[V3]]
// CHECK: vector.transfer_write %[[A3]], %{{.*}}[%[[C96]]]

// -----

// Test loop with trip count divisible by unroll factor
// Loop: for i = 0 to 100 step 32 (trip count = 4, evenly divisible by unroll factor 4)
func.func @divisible_loop_100(%arg0: memref<100xf32>) {
  %c0 = arith.constant 0 : index
  %c32 = arith.constant 32 : index
  %c100 = arith.constant 100 : index
  %cst = arith.constant 0.000000e+00 : f32

  scf.for %i = %c0 to %c100 step %c32 {
    %0 = vector.transfer_read %arg0[%i], %cst {in_bounds = [true]} : memref<100xf32>, vector<32xf32>
    %1 = arith.addf %0, %0 : vector<32xf32>
    vector.transfer_write %1, %arg0[%i] {in_bounds = [true]} : vector<32xf32>, memref<100xf32>
  }
  return
}

// CHECK-LABEL: func.func @divisible_loop_100
// CHECK-DAG: %[[C96:.*]] = arith.constant 96 : index
// CHECK-DAG: %[[C64:.*]] = arith.constant 64 : index
// CHECK-DAG: %[[C32:.*]] = arith.constant 32 : index
// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
// CHECK: vector.transfer_read %{{.*}}[%[[C0]]]
// CHECK: vector.transfer_read %{{.*}}[%[[C32]]]
// CHECK: vector.transfer_read %{{.*}}[%[[C64]]]
// CHECK: vector.transfer_read %{{.*}}[%[[C96]]]

// -----

// Test loop with iter_args should NOT be unrolled
func.func @loop_with_iter_args(%arg0: memref<128xf32>) -> f32 {
  %c0 = arith.constant 0 : index
  %c32 = arith.constant 32 : index
  %c128 = arith.constant 128 : index
  %cst = arith.constant 0.000000e+00 : f32

  %result = scf.for %i = %c0 to %c128 step %c32 iter_args(%acc = %cst) -> (f32) {
    %0 = vector.transfer_read %arg0[%i], %cst {in_bounds = [true]} : memref<128xf32>, vector<32xf32>
    %1 = vector.reduction <add>, %0 : vector<32xf32> into f32
    %2 = arith.addf %acc, %1 : f32
    scf.yield %2 : f32
  }
  return %result : f32
}

// CHECK-LABEL: func.func @loop_with_iter_args
// CHECK: %[[RESULT:.*]] = scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%{{.*}} = %{{.*}}) -> (f32)
// CHECK-NOT: arith.addi
// CHECK: return %[[RESULT]]

// -----

// Test nested loops - only innermost should be unrolled
func.func @nested_loops(%arg0: memref<128x128xf32>) {
  %c0 = arith.constant 0 : index
  %c32 = arith.constant 32 : index
  %c128 = arith.constant 128 : index
  %cst = arith.constant 0.000000e+00 : f32

  scf.for %i = %c0 to %c128 step %c32 {
    scf.for %j = %c0 to %c128 step %c32 {
      %0 = memref.load %arg0[%i, %j] : memref<128x128xf32>
      %1 = arith.addf %0, %0 : f32
      memref.store %1, %arg0[%i, %j] : memref<128x128xf32>
    }
  }
  return
}

// CHECK-LABEL: func.func @nested_loops
// CHECK: scf.for %[[I:.*]] = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK:   memref.load %{{.*}}[%[[I]], %{{.*}}]
// CHECK:   memref.load %{{.*}}[%[[I]], %{{.*}}]
// CHECK:   memref.load %{{.*}}[%[[I]], %{{.*}}]
// CHECK:   memref.load %{{.*}}[%[[I]], %{{.*}}]
// CHECK: }

// -----

// Test loop with step > 1 that divides evenly
// Loop: for i = 0 to 64 step 16 (trip count = 4, evenly divisible)
func.func @loop_step_16(%arg0: memref<64xf32>) {
  %c0 = arith.constant 0 : index
  %c16 = arith.constant 16 : index
  %c64 = arith.constant 64 : index
  %cst = arith.constant 1.000000e+00 : f32

  scf.for %i = %c0 to %c64 step %c16 {
    %0 = vector.transfer_read %arg0[%i], %cst {in_bounds = [true]} : memref<64xf32>, vector<16xf32>
    %1 = arith.mulf %0, %0 : vector<16xf32>
    vector.transfer_write %1, %arg0[%i] {in_bounds = [true]} : vector<16xf32>, memref<64xf32>
  }
  return
}

// CHECK-LABEL: func.func @loop_step_16
// CHECK-DAG: %[[C48:.*]] = arith.constant 48 : index
// CHECK-DAG: %[[C32:.*]] = arith.constant 32 : index
// CHECK-DAG: %[[C16:.*]] = arith.constant 16 : index
// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
// CHECK: vector.transfer_read %{{.*}}[%[[C0]]]
// CHECK: vector.transfer_read %{{.*}}[%[[C16]]]
// CHECK: vector.transfer_read %{{.*}}[%[[C32]]]
// CHECK: vector.transfer_read %{{.*}}[%[[C48]]]

// -----

// Test softmax-like pattern: reduction followed by normalization
func.func @softmax_pattern(%arg0: memref<128xf32>, %arg1: memref<128xf32>) {
  %c0 = arith.constant 0 : index
  %c32 = arith.constant 32 : index
  %c128 = arith.constant 128 : index
  %cst = arith.constant 0.000000e+00 : f32
  %alloc = memref.alloc() : memref<f32>

  // Reduction loop
  memref.store %cst, %alloc[] : memref<f32>
  scf.for %i = %c0 to %c128 step %c32 {
    %0 = vector.transfer_read %arg0[%i], %cst {in_bounds = [true]} : memref<128xf32>, vector<32xf32>
    %1 = memref.load %alloc[] : memref<f32>
    %2 = vector.reduction <add>, %0, %1 : vector<32xf32> into f32
    memref.store %2, %alloc[] : memref<f32>
  }

  %sum = memref.load %alloc[] : memref<f32>

  // Normalization loop
  scf.for %i = %c0 to %c128 step %c32 {
    %0 = vector.transfer_read %arg0[%i], %cst {in_bounds = [true]} : memref<128xf32>, vector<32xf32>
    %1 = vector.broadcast %sum : f32 to vector<32xf32>
    %2 = arith.divf %0, %1 : vector<32xf32>
    vector.transfer_write %2, %arg1[%i] {in_bounds = [true]} : vector<32xf32>, memref<128xf32>
  }

  memref.dealloc %alloc : memref<f32>
  return
}

// CHECK-LABEL: func.func @softmax_pattern
// CHECK: vector.transfer_read
// CHECK: vector.reduction <add>
// CHECK: vector.transfer_read
// CHECK: vector.reduction <add>
// CHECK: vector.transfer_read
// CHECK: vector.reduction <add>
// CHECK: vector.transfer_read
// CHECK: vector.reduction <add>
// CHECK: %[[SUM:.*]] = memref.load
// CHECK: %[[BCAST:.*]] = vector.broadcast %[[SUM]]
// CHECK: arith.divf %{{.*}}, %[[BCAST]]
// CHECK: arith.divf %{{.*}}, %[[BCAST]]
// CHECK: arith.divf %{{.*}}, %[[BCAST]]
// CHECK: arith.divf %{{.*}}, %[[BCAST]]
