// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(llvm-intr-to-hexagon-routines)' | FileCheck %s

// CHECK-LABEL: llvm.func @_hexagon_runtime_tan__vf(vector<32xi32>)
// CHECK-LABEL: llvm.func @_hexagon_runtime_cos__vf(vector<32xi32>)
// CHECK-LABEL: llvm.func @_hexagon_runtime_sin__vf(vector<32xi32>)

// CHECK-LABEL: llvm.func @trig_hexagon_routine

llvm.func @trig_hexagon_routine(%arg0: vector<64xf32>) -> vector<64xf32> {
  // CHECK: %[[ARG0:.*]]: vector<64xf32>
  // CHECK: %[[UNDEF:.*]] = llvm.mlir.undef : vector<64xf32>

  
  // CHECK: %[[LowSin:.*]] = llvm.intr.vector.extract %[[ARG0]][0] : vector<32xf32> from vector<64xf32>
  // CHECK: %[[LowSinBC:.*]] = llvm.bitcast %[[LowSin]] : vector<32xf32> to vector<32xi32>
  // CHECK: %[[LowSinCall:.*]] = llvm.call @_hexagon_runtime_sin__vf(%[[LowSinBC]]) : (vector<32xi32>) -> vector<32xi32>
  // CHECK: %[[LowSinRes:.*]] = llvm.bitcast %[[LowSinCall]] : vector<32xi32> to vector<32xf32>
  // CHECK: %[[InsertLowSin:.*]] = llvm.intr.vector.insert %[[LowSinRes]], %[[UNDEF]][0] : vector<32xf32> into vector<64xf32>

  // CHECK: %[[HighSin:.*]] = llvm.intr.vector.extract %[[ARG0]][32] : vector<32xf32> from vector<64xf32>
  // CHECK: %[[HighSinBC:.*]] = llvm.bitcast %[[HighSin]] : vector<32xf32> to vector<32xi32>
  // CHECK: %[[HighSinCall:.*]] = llvm.call @_hexagon_runtime_sin__vf(%[[HighSinBC]]) : (vector<32xi32>) -> vector<32xi32>
  // CHECK: %[[HighSinRes:.*]] = llvm.bitcast %[[HighSinCall]] : vector<32xi32> to vector<32xf32>
  // CHECK: %[[InsertHighSin:.*]] = llvm.intr.vector.insert %[[HighSinRes]], %[[InsertLowSin]][32] : vector<32xf32> into vector<64xf32>
  
  // CHECK: %[[LowCos:.*]] = llvm.intr.vector.extract %[[ARG0]][0] : vector<32xf32> from vector<64xf32>
  // CHECK: %[[LowCosBC:.*]] = llvm.bitcast %[[LowCos]] : vector<32xf32> to vector<32xi32>
  // CHECK: %[[LowCosCall:.*]] = llvm.call @_hexagon_runtime_cos__vf(%[[LowCosBC]]) : (vector<32xi32>) -> vector<32xi32>
  // CHECK: %[[LowCosRes:.*]] = llvm.bitcast %[[LowCosCall]] : vector<32xi32> to vector<32xf32>
  // CHECK: %[[InsertLowCos:.*]] = llvm.intr.vector.insert %[[LowCosRes]], %[[UNDEF]][0] : vector<32xf32> into vector<64xf32>

  // CHECK: %[[HighCos:.*]] = llvm.intr.vector.extract %[[ARG0]][32] : vector<32xf32> from vector<64xf32>
  // CHECK: %[[HighCosBC:.*]] = llvm.bitcast %[[HighCos]] : vector<32xf32> to vector<32xi32>
  // CHECK: %[[HighCosCall:.*]] = llvm.call @_hexagon_runtime_cos__vf(%[[HighCosBC]]) : (vector<32xi32>) -> vector<32xi32>
  // CHECK: %[[HighCosRes:.*]] = llvm.bitcast %[[HighCosCall]] : vector<32xi32> to vector<32xf32>
  // CHECK: %[[InsertHighCos:.*]] = llvm.intr.vector.insert %[[HighCosRes]], %[[InsertLowCos]][32] : vector<32xf32> into vector<64xf32>

  // CHECK: %[[LowTan:.*]] = llvm.intr.vector.extract %[[ARG0]][0] : vector<32xf32> from vector<64xf32>
  // CHECK: %[[LowTanBC:.*]] = llvm.bitcast %[[LowTan]] : vector<32xf32> to vector<32xi32>
  // CHECK: %[[LowTanCall:.*]] = llvm.call @_hexagon_runtime_tan__vf(%[[LowTanBC]]) : (vector<32xi32>) -> vector<32xi32>
  // CHECK: %[[LowTanRes:.*]] = llvm.bitcast %[[LowTanCall]] : vector<32xi32> to vector<32xf32>
  // CHECK: %[[InsertLowTan:.*]] = llvm.intr.vector.insert %[[LowTanRes]], %[[UNDEF]][0] : vector<32xf32> into vector<64xf32>

  // CHECK: %[[HighTan:.*]] = llvm.intr.vector.extract %[[ARG0]][32] : vector<32xf32> from vector<64xf32>
  // CHECK: %[[HighTanBC:.*]] = llvm.bitcast %[[HighTan]] : vector<32xf32> to vector<32xi32>
  // CHECK: %[[HighTanCall:.*]] = llvm.call @_hexagon_runtime_tan__vf(%[[HighTanBC]]) : (vector<32xi32>) -> vector<32xi32>
  // CHECK: %[[HighTanRes:.*]] = llvm.bitcast %[[HighTanCall]] : vector<32xi32> to vector<32xf32>
  // CHECK: %[[InsertHighTan:.*]] = llvm.intr.vector.insert %[[HighTanRes]], %[[InsertLowTan]][32] : vector<32xf32> into vector<64xf32>

  %res_sin = llvm.intr.sin(%arg0) {fastmathFlags = #llvm.fastmath<fast>} : (vector<64xf32>) -> vector<64xf32>
  %res_cos = llvm.intr.cos(%arg0) {fastmathFlags = #llvm.fastmath<fast>} : (vector<64xf32>) -> vector<64xf32>
  %res_tan = llvm.intr.tan(%arg0) {fastmathFlags = #llvm.fastmath<fast>} : (vector<64xf32>) -> vector<64xf32>

  %res_add = llvm.fadd %res_sin, %res_cos : vector<64xf32>
  %res_final = llvm.fadd %res_add, %res_tan : vector<64xf32>

  llvm.return %res_final : vector<64xf32>
}
