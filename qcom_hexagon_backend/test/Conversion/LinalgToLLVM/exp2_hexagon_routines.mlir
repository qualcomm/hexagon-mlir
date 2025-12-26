// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(llvm-intr-to-hexagon-routines)' | FileCheck %s -check-prefixes=CHECK

// CHECK-LABEL: llvm.func @_hexagon_runtime_pow__vf(vector<32xi32>, vector<32xi32>) -> vector<32xi32> attributes {sym_visibility = "private"}
// CHECK-LABEL: llvm.func @_hexagon_runtime_pow__vhf(vector<32xi32>, vector<32xi32>) -> vector<32xi32> attributes {sym_visibility = "private"}

llvm.func @foo(%arg0: vector<64xf16>, %arg1: vector<64xf32>) -> vector<64xf16> {

 // CHECK: [[CONST1:%.+]] = llvm.mlir.constant(dense<2.000000e+00> : vector<32xf32>) : vector<32xf32>
 // CHECK-NEXT: [[UNDEF:%.+]] = llvm.mlir.undef : vector<64xf32>
 // CHECK-NEXT: [[CONST2:%.+]] = llvm.mlir.constant(dense<2.000000e+00> : vector<64xf16>) : vector<64xf16>
 // CHECK-NEXT: [[CONSTCAST1:%.+]] = llvm.bitcast [[CONST2]] : vector<64xf16> to vector<32xi32>
 // CHECK-NEXT: [[BITCAST:%.+]] = llvm.bitcast %arg0 : vector<64xf16> to vector<32xi32>
 // CHECK-NEXT: [[INTRINCALL:%.+]] = llvm.call @_hexagon_runtime_pow__vhf([[CONSTCAST1]], [[BITCAST]]) : (vector<32xi32>, vector<32xi32>) -> vector<32xi32>
 // CHECK-NEXT: [[EXP:%.+]] = llvm.bitcast [[INTRINCALL]] : vector<32xi32> to vector<64xf16>

 %47 = llvm.intr.exp2(%arg0)  {fastmathFlags = #llvm.fastmath<fast>} : (vector<64xf16>) -> vector<64xf16>

 // CHECK: [[EXTRACTV:%.+]] = llvm.intr.vector.extract %arg1[0] : vector<32xf32> from vector<64xf32>
 // CHECK-NEXT: [[CONSTCAST2:%.+]] = llvm.bitcast [[CONST1]] : vector<32xf32> to vector<32xi32>
 // CHECK-NEXT: [[BITCAST1:%.+]] = llvm.bitcast [[EXTRACTV]] : vector<32xf32> to vector<32xi32>
 // CHECK-NEXT: [[EXP21:%.+]] = llvm.call @_hexagon_runtime_pow__vf([[CONSTCAST2]], [[BITCAST1]]) : (vector<32xi32>, vector<32xi32>) -> vector<32xi32>
 // CHECK-NEXT: [[BITCAST2:%.+]] = llvm.bitcast [[EXP21]] : vector<32xi32> to vector<32xf32>
 // CHECK-NEXT: [[INSERTV:%.+]] = llvm.intr.vector.insert [[BITCAST2]], [[UNDEF]][0] : vector<32xf32> into vector<64xf32>
 // CHECK-NEXT: [[EXTRACTV2:%.+]] = llvm.intr.vector.extract %arg1[32] : vector<32xf32> from vector<64xf32>
 // CHECK-NEXT: [[CONSTCAST3:%.+]] = llvm.bitcast [[CONST1]] : vector<32xf32> to vector<32xi32>
 // CHECK-NEXT: [[BITCAST3:%.+]] = llvm.bitcast [[EXTRACTV2]] : vector<32xf32> to vector<32xi32>
 // CHECK-NEXT: [[EXP22:%.+]] = llvm.call @_hexagon_runtime_pow__vf([[CONSTCAST3]], [[BITCAST3]]) : (vector<32xi32>, vector<32xi32>) -> vector<32xi32>
 // CHECK-NEXT: [[BITCAST4:%.+]] = llvm.bitcast [[EXP22]] : vector<32xi32> to vector<32xf32>
 // CHECK-NEXT: [[INSERTV2:%.+]] = llvm.intr.vector.insert [[BITCAST4]], [[INSERTV]][32] : vector<32xf32> into vector<64xf32>

 %48 = llvm.intr.exp2(%arg1)  {fastmathFlags = #llvm.fastmath<fast>} : (vector<64xf32>) -> vector<64xf32>

 // CHECK: [[TRUNC:%.+]] = llvm.fptrunc [[INSERTV2]] : vector<64xf32> to vector<64xf16>

  %49 = llvm.fptrunc %48 : vector<64xf32> to vector<64xf16>

 // CHECK: [[SUB:%.+]] = llvm.fsub [[EXP]], [[TRUNC]] {fastmathFlags = #llvm.fastmath<fast>} : vector<64xf16>
  %50 = llvm.fsub %47, %49  {fastmathFlags = #llvm.fastmath<fast>} : vector<64xf16>
  llvm.return %50 : vector<64xf16>
}
