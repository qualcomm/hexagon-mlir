// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(llvm-intr-to-hexagon-routines)' -split-input-file | FileCheck %s

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

// -----

// CHECK-LABEL: llvm.func @_hexagon_runtime_tanh__vhf(vector<32xi32>)
// CHECK-LABEL: llvm.func @tanh_hexagon_routine

llvm.func @tanh_hexagon_routine(%arg0:vector<64xf16>) -> vector<64xf16> {
  // CHECK: %[[ARG0:.*]]: vector<64xf16>
  // CHECK: %[[BC1:.*]] = llvm.bitcast %[[ARG0]] : vector<64xf16> to vector<32xi32>
  // CHECK: %[[CALL:.*]] = llvm.call @_hexagon_runtime_tanh__vhf(%[[BC1]]) : (vector<32xi32>) -> vector<32xi32>
  // CHECK: %[[RES:.*]] = llvm.bitcast %[[CALL]] : vector<32xi32> to vector<64xf16>
  // CHECK: llvm.return %[[RES]] : vector<64xf16>
  %res = llvm.intr.tanh(%arg0) {fastmathFlags = #llvm.fastmath<fast>} : (vector<64xf16>) -> vector<64xf16>
  llvm.return %res : vector<64xf16>
}

// -----

// CHECK-LABEL: llvm.func @_hexagon_runtime_asin__vf(vector<32xi32>)
// CHECK-LABEL: llvm.func @asin_hexagon_routine

llvm.func @asin_hexagon_routine(%arg0: vector<64xf32>) -> vector<64xf32> {
  // CHECK: %[[ARG0:.*]]: vector<64xf32>
  // CHECK: %[[UNDEF:.*]] = llvm.mlir.undef : vector<64xf32>
  // CHECK: %[[LowASin:.*]] = llvm.intr.vector.extract %[[ARG0]][0] : vector<32xf32> from vector<64xf32>
  // CHECK: %[[LowASinBC:.*]] = llvm.bitcast %[[LowASin]] : vector<32xf32> to vector<32xi32>
  // CHECK: %[[LowASinCall:.*]] = llvm.call @_hexagon_runtime_asin__vf(%[[LowASinBC]]) : (vector<32xi32>) -> vector<32xi32>
  // CHECK: %[[LowASinRes:.*]] = llvm.bitcast %[[LowASinCall]] : vector<32xi32> to vector<32xf32>
  // CHECK: %[[InsertLowASin:.*]] = llvm.intr.vector.insert %[[LowASinRes]], %[[UNDEF]][0] : vector<32xf32> into vector<64xf32>
  // CHECK: %[[HighASin:.*]] = llvm.intr.vector.extract %[[ARG0]][32] : vector<32xf32> from vector<64xf32>
  // CHECK: %[[HighASinBC:.*]] = llvm.bitcast %[[HighASin]] : vector<32xf32> to vector<32xi32>
  // CHECK: %[[HighASinCall:.*]] = llvm.call @_hexagon_runtime_asin__vf(%[[HighASinBC]]) : (vector<32xi32>) -> vector<32xi32>
  // CHECK: %[[HighASinRes:.*]] = llvm.bitcast %[[HighASinCall]] : vector<32xi32> to vector<32xf32>
  // CHECK: %[[InsertHighASin:.*]] = llvm.intr.vector.insert %[[HighASinRes]], %[[InsertLowASin]][32] : vector<32xf32> into vector<64xf32>
  // CHECK: llvm.return %[[InsertHighASin]] : vector<64xf32>
  %res = llvm.intr.asin(%arg0) {fastmathFlags = #llvm.fastmath<fast>} : (vector<64xf32>) -> vector<64xf32>
  llvm.return %res : vector<64xf32>
}

// -----
// CHECK-LABEL: llvm.func @_hexagon_runtime_acos__vf(vector<32xi32>)
// CHECK-LABEL: llvm.func @acos_hexagon_routine 

llvm.func @acos_hexagon_routine(%arg0: vector<64xf32>) -> vector<64xf32> {
  // CHECK: %[[ARG0:.*]]: vector<64xf32>
  // CHECK: %[[UNDEF:.*]] = llvm.mlir.undef : vector<64xf32>
  // CHECK: %[[LowACos:.*]] = llvm.intr.vector.extract %[[ARG0]][0] : vector<32xf32> from vector<64xf32>
  // CHECK: %[[LowACosBC:.*]] = llvm.bitcast %[[LowACos]] : vector<32xf32> to vector<32xi32>
  // CHECK: %[[LowACosCall:.*]] = llvm.call @_hexagon_runtime_acos__vf(%[[LowACosBC]]) : (vector<32xi32>) -> vector<32xi32>
  // CHECK: %[[LowACosRes:.*]] = llvm.bitcast %[[LowACosCall]] : vector<32xi32> to vector<32xf32>
  // CHECK: %[[InsertLowACos:.*]] = llvm.intr.vector.insert %[[LowACosRes]], %[[UNDEF]][0] : vector<32xf32> into vector<64xf32>
  // CHECK: %[[HighACos:.*]] = llvm.intr.vector.extract %[[ARG0]][32] : vector<32xf32> from vector<64xf32>
  // CHECK: %[[HighACosBC:.*]] = llvm.bitcast %[[HighACos]] : vector<32xf32> to vector<32xi32>
  // CHECK: %[[HighACosCall:.*]] = llvm.call @_hexagon_runtime_acos__vf(%[[HighACosBC]]) : (vector<32xi32>) -> vector<32xi32>
  // CHECK: %[[HighACosRes:.*]] = llvm.bitcast %[[HighACosCall]] : vector<32xi32> to vector<32xf32>
  // CHECK: %[[InsertHighACos:.*]] = llvm.intr.vector.insert %[[HighACosRes]], %[[InsertLowACos]][32] : vector<32xf32> into vector<64xf32>
  // CHECK: llvm.return %[[InsertHighACos]] : vector<64xf32>
  %res = llvm.intr.acos(%arg0) {fastmathFlags = #llvm.fastmath<fast>} : (vector<64xf32>) -> vector<64xf32>
  llvm.return %res : vector<64xf32>
}

// -----
// CHECK-LABEL: llvm.func @_hexagon_runtime_atan__vf(vector<32xi32>)
// CHECK-LABEL: llvm.func @atan_hexagon_routine

llvm.func @atan_hexagon_routine(%arg0: vector<64xf32>) -> vector<64xf32> {
  // CHECK: %[[ARG0:.*]]: vector<64xf32>
  // CHECK: %[[UNDEF:.*]] = llvm.mlir.undef : vector<64xf32>
  // CHECK: %[[LowATan:.*]] = llvm.intr.vector.extract %[[ARG0]][0] : vector<32xf32> from vector<64xf32>
  // CHECK: %[[LowATanBC:.*]] = llvm.bitcast %[[LowATan]] : vector<32xf32> to vector<32xi32>
  // CHECK: %[[LowATanCall:.*]] = llvm.call @_hexagon_runtime_atan__vf(%[[LowATanBC]]) : (vector<32xi32>) -> vector<32xi32>
  // CHECK: %[[LowATanRes:.*]] = llvm.bitcast %[[LowATanCall]] : vector<32xi32> to vector<32xf32>
  // CHECK: %[[InsertLowATan:.*]] = llvm.intr.vector.insert %[[LowATanRes]], %[[UNDEF]][0] : vector<32xf32> into vector<64xf32>
  // CHECK: %[[HighATan:.*]] = llvm.intr.vector.extract %[[ARG0]][32] : vector<32xf32> from vector<64xf32>
  // CHECK: %[[HighATanBC:.*]] = llvm.bitcast %[[HighATan]] : vector<32xf32> to vector<32xi32>
  // CHECK: %[[HighATanCall:.*]] = llvm.call @_hexagon_runtime_atan__vf(%[[HighATanBC]]) : (vector<32xi32>) -> vector<32xi32>
  // CHECK: %[[HighATanRes:.*]] = llvm.bitcast %[[HighATanCall]] : vector<32xi32> to vector<32xf32>
  // CHECK: %[[InsertHighATan:.*]] = llvm.intr.vector.insert %[[HighATanRes]], %[[InsertLowATan]][32] : vector<32xf32> into vector<64xf32>
  // CHECK: llvm.return %[[InsertHighATan]] : vector<64xf32>
  %res = llvm.intr.atan(%arg0) {fastmathFlags = #llvm.fastmath<fast>} : (vector<64xf32>) -> vector<64xf32>
  llvm.return %res : vector<64xf32>
}

// -----
// CHECK-LABEL: llvm.func @_hexagon_runtime_floor__vf(vector<32xi32>)
// CHECK-LABEL: llvm.func @floor_hexagon_routine

llvm.func @floor_hexagon_routine(%arg0: vector<64xf32>) -> vector<64xf32> {
  // CHECK: %[[ARG0:.*]]: vector<64xf32>
  // CHECK: %[[UNDEF:.*]] = llvm.mlir.undef : vector<64xf32>
  // CHECK: %[[LowFloor:.*]] = llvm.intr.vector.extract %[[ARG0]][0] : vector<32xf32> from vector<64xf32>
  // CHECK: %[[LowFloorBC:.*]] = llvm.bitcast %[[LowFloor]] : vector<32xf32> to vector<32xi32>
  // CHECK: %[[LowFloorCall:.*]] = llvm.call @_hexagon_runtime_floor__vf(%[[LowFloorBC]]) : (vector<32xi32>) -> vector<32xi32
  // CHECK: %[[LowFloorRes:.*]] = llvm.bitcast %[[LowFloorCall]] : vector<32xi32> to vector<32xf32>
  // CHECK: %[[InsertLowFloor:.*]] = llvm.intr.vector.insert %[[LowFloorRes]], %[[UNDEF]][0] : vector<32xf32> into vector<64xf32>
  // CHECK: %[[HighFloor:.*]] = llvm.intr.vector.extract %[[ARG0]][32] : vector<32xf32> from vector<64xf32> 
  // CHECK: %[[HighFloorBC:.*]] = llvm.bitcast %[[HighFloor]] : vector<32xf32> to vector<32xi32>
  // CHECK: %[[HighFloorCall:.*]] = llvm.call @_hexagon_runtime_floor__vf(%[[HighFloorBC]]) : (vector<32xi32>) -> vector<32xi32>
  // CHECK: %[[HighFloorRes:.*]] = llvm.bitcast %[[HighFloorCall]] : vector<32xi32> to vector<32xf32>
  // CHECK: %[[InsertHighFloor:.*]] = llvm.intr.vector.insert %[[HighFloorRes]], %[[InsertLowFloor]][32] : vector<32xf32> into vector<64xf32>
  // CHECK: llvm.return %[[InsertHighFloor]] : vector<64xf32>
  %res = llvm.intr.floor(%arg0) : (vector<64xf32>) -> vector<64xf32>
  llvm.return %res : vector<64xf32>
}

// -----
// CHECK-LABEL: llvm.func @_hexagon_runtime_floor__vhf(vector<32xi32>)
// CHECK-LABEL: llvm.func @floor_f16_hexagon_routine
llvm.func @floor_f16_hexagon_routine(%arg0: vector<64xf16>) -> vector<64xf16> {
  // CHECK: %[[BC0:.*]] = llvm.bitcast %[[ARG0:.*]] : vector<64xf16> to vector<32xi32>
  // CHECK: %[[CALL:.*]] = llvm.call @_hexagon_runtime_floor__vhf(%[[BC0]]) : (vector<32xi32>) -> vector<32xi32>
  // CHECK: %[[RES:.*]] = llvm.bitcast %[[CALL]] : vector<32xi32> to vector<64xf16>
  // CHECK: llvm.return %[[RES]] : vector<64xf16>
  %res = llvm.intr.floor(%arg0) : (vector<64xf16>) -> vector<64xf16>
  llvm.return %res : vector<64xf16>
}

// -----
// CHECK-LABEL: llvm.func @_hexagon_runtime_ceil__vf(vector<32xi32>)
// CHECK-LABEL: llvm.func @ceil_hexagon_routine

llvm.func @ceil_hexagon_routine(%arg0: vector<64xf32>) -> vector<64xf32> {
  // CHECK: %[[ARG0:.*]]: vector<64xf32>
  // CHECK: %[[UNDEF:.*]] = llvm.mlir.undef : vector<64xf32>
  // CHECK: %[[LowCeil:.*]] = llvm.intr.vector.extract %[[ARG0]][0] : vector<32xf32> from vector<64xf32>
  // CHECK: %[[LowCeilBC:.*]] = llvm.bitcast %[[LowCeil]] : vector<32xf32> to vector<32xi32>
  // CHECK: %[[LowCeilCall:.*]] = llvm.call @_hexagon_runtime_ceil__vf(%[[LowCeilBC]]) : (vector<32xi32>) -> vector<32xi32>
  // CHECK: %[[LowCeilRes:.*]] = llvm.bitcast %[[LowCeilCall]] : vector<32xi32> to vector<32xf32>
  // CHECK: %[[InsertLowCeil:.*]] = llvm.intr.vector.insert %[[LowCeilRes]], %[[UNDEF]][0] : vector<32xf32> into vector<64xf32>
  // CHECK: %[[HighCeil:.*]] = llvm.intr.vector.extract %[[ARG0]][32] : vector<32xf32> from vector<64xf32>
  // CHECK: %[[HighCeilBC:.*]] = llvm.bitcast %[[HighCeil]] : vector<32xf32> to vector<32xi32>
  // CHECK: %[[HighCeilCall:.*]] = llvm.call @_hexagon_runtime_ceil__vf(%[[HighCeilBC]]) : (vector<32xi32>) -> vector<32xi32>
  // CHECK: %[[HighCeilRes:.*]] = llvm.bitcast %[[HighCeilCall]] : vector<32xi32> to vector<32xf32>
  // CHECK: %[[InsertHighCeil:.*]] = llvm.intr.vector.insert %[[HighCeilRes]], %[[InsertLowCeil]][32] : vector<32xf32> into vector<64xf32>
  // CHECK: llvm.return %[[InsertHighCeil]] : vector<64xf32>
  %res = llvm.intr.ceil(%arg0) : (vector<64xf32>) -> vector<64xf32>
  llvm.return %res : vector<64xf32>
}

// -----
// CHECK-LABEL: llvm.func @_hexagon_runtime_ceil__vhf(vector<32xi32>)
// CHECK-LABEL: llvm.func @ceil_f16_hexagon_routine

llvm.func @ceil_f16_hexagon_routine(%arg0: vector<64xf16>) -> vector<64xf16> {
  // CHECK: %[[BC0:.*]] = llvm.bitcast %[[ARG0:.*]] : vector<64xf16> to vector<32xi32>
  // CHECK: %[[CALL:.*]] = llvm.call @_hexagon_runtime_ceil__vhf(%[[BC0]]) : (vector<32xi32>) -> vector<32xi32>
  // CHECK: %[[RES:.*]] = llvm.bitcast %[[CALL]] : vector<32xi32> to vector<64xf16>
  // CHECK: llvm.return %[[RES]] : vector<64xf16>
  %res = llvm.intr.ceil(%arg0) : (vector<64xf16>) -> vector<64xf16>
  llvm.return %res : vector<64xf16>
}
