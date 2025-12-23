// RUN: linalg-hexagon-opt %s  -pass-pipeline='builtin.module(func.func(expand-bool-vec))' \
// RUN:   | FileCheck %s -check-prefixes=CHECK,SEL_ON
//
// RUN: linalg-hexagon-opt %s -linalg-to-llvm | linalg-hexagon-translate | FileCheck %s -check-prefixes=CHECK,LLVM

// CHECK-LABEL: @ui1_to_fp32
//
// SEL_ON: %[[Cond:.+]] = vector.load %{{.*}} : memref<?xi1>, vector<5xi1>
// SEL_ON-DAG: %[[ZEROS:.+]] = arith.constant dense<0.000000e+00> : vector<5xf32>
// SEL_ON-DAG: %[[ONES:.+]] = arith.constant dense<1.000000e+00> : vector<5xf32>
// SEL_ON: %[[RES:.+]] = arith.select %[[Cond]], %[[ONES]], %[[ZEROS]] : vector<5xi1>, vector<5xf32>
// SEL_ON: vector.store %[[RES]], %{{.*}} : memref<?xf32>, vector<5xf32>
//
// LLVM: %[[Cond:.+]] = load <5 x i1>, ptr %{{.*}}
// LLVM: %[[RES:.+]] = select <5 x i1> %[[Cond]], <5 x float> splat (float 1.000000e+00), <5 x float> zeroinitializer
// LLVM: store <5 x float> %[[RES]], ptr %{{.*}}
//
func.func @ui1_to_fp32(%loadmem : memref<?xi1>, %storemem : memref<?xf32>) {
  %c_0 = arith.constant 0 : index
  %val = vector.load %loadmem[%c_0] : memref<?xi1>, vector<5xi1>
  %r = arith.uitofp %val : vector<5xi1> to vector<5xf32>
  vector.store %r , %storemem[%c_0] : memref<?xf32>, vector<5xf32>
  return
}

