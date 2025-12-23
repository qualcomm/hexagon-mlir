
// REQUIRES: do-not-run-because-flaky-test-that-needs-being-investigated

// RUN: linalg-hexagon-opt -split-input-file %s -linalg-to-llvm | FileCheck %s

#map = affine_map<(d0) -> (d0)>
module attributes {llvm.target_triple = "hexagon"} {
  func.func @foo(%arg0: memref<*xf32>, %argi : memref<*xi32>, %arg1: memref<*xf32>) {
    %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [0], sizes: [1024], strides: [1] : memref<*xf32> to memref<1024xf32, strided<[1]>>
    %0 = bufferization.to_tensor %reinterpret_cast restrict : memref<1024xf32, strided<[1]>> to tensor<1024xf32>

    %reinterpret_casti = memref.reinterpret_cast %argi to offset: [0], sizes: [1024], strides: [1] : memref<*xi32> to memref<1024xi32, strided<[1]>>
    %ti = bufferization.to_tensor %reinterpret_casti restrict : memref<1024xi32, strided<[1]>> to tensor<1024xi32>

    %1 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%0, %ti : tensor<1024xf32>, tensor<1024xi32>) outs(%0 : tensor<1024xf32>) {
    ^bb0(%in: f32, %ini : i32, %out: f32):
      %2 = arith.truncf %in : f32 to f16
      %3 = math.exp %2 : f16
      %4 = arith.mulf %3, %3 : f16
      %5 = arith.addf %3, %4 : f16

      %xi = arith.trunci %ini : i32 to i16
      %yi = arith.addi %xi, %xi : i16
      %zf = arith.sitofp %yi : i16 to f16 
      %w = arith.addf %5, %zf : f16

      %6 = arith.extf %w : f16 to f32
      %7 = arith.mulf %6, %in : f32

      linalg.yield %7 : f32
    } -> tensor<1024xf32>
    %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [0], sizes: [1024], strides: [1] : memref<*xf32> to memref<1024xf32, strided<[1]>>
    bufferization.materialize_in_destination %1 in writable %reinterpret_cast_0 : (tensor<1024xf32>, memref<1024xf32, strided<[1]>>) -> ()
    return
  }
}

// CHECK: [[TRUNCF16:%.+]] = llvm.fptrunc {{.*}} : vector<64xf32> to vector<64xf16>
// CHECK-NEXT: [[BITCAST1:%.+]] = llvm.bitcast [[TRUNCF16]] : vector<64xf16> to vector<32xi32>
// CHECK-NEXT: [[EXP:%.+]] = llvm.call @_hexagon_runtime_exp__vhf([[BITCAST1]]) : (vector<32xi32>) -> vector<32xi32>
// CHECK-NEXT: [[BITCAST2:%.+]] = llvm.bitcast [[EXP]] : vector<32xi32> to vector<64xf16>
// CHECK-NEXT: [[MUL:%.+]] = llvm.fmul [[BITCAST2]], [[BITCAST2]]  {fastmathFlags = #llvm.fastmath<fast>} : vector<64xf16>
// CHECK-NEXT: [[FADD:%.+]] = llvm.fadd [[BITCAST2]], [[MUL]]  {fastmathFlags = #llvm.fastmath<fast>} : vector<64xf16>
// CHECK-NEXT: [[TRUNC16:%.+]] = llvm.trunc {{.*}} : vector<64xi32> to vector<64xi16>
// CHECK-NEXT: [[ADD:%.+]] = llvm.add [[TRUNC16]], [[TRUNC16]] : vector<64xi16>
// CHECK-NEXT: [[SITOFP:%.+]] = llvm.sitofp [[ADD]] : vector<64xi16> to vector<64xf16>
// CHECK-NEXT: [[FADD2:%.+]] = llvm.fadd [[FADD]], [[SITOFP]]  {fastmathFlags = #llvm.fastmath<fast>} : vector<64xf16>
// CHECK-NEXT: [[EXTEND:%.+]] = llvm.fpext [[FADD2]] {fastmath = #arith.fastmath<fast>} : vector<64xf16> to vector<64xf32>
// CHECK-NEXT: [[RES:%.+]] = llvm.fmul [[EXTEND]], {{.*}} {fastmathFlags = #llvm.fastmath<fast>} : vector<64xf32>

// -----

#map = affine_map<(d0) -> (d0)>
module attributes {llvm.target_triple = "hexagon"} {
  func.func @foo2(%arg0: tensor<1024xf32>) -> tensor<1024xf32> {
    %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%arg0: tensor<1024xf32>) outs(%arg0 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %2 = arith.fptosi %in : f32 to i8
        %3 = arith.muli %2, %2 : i8
        %4 = arith.sitofp %3 : i8 to f32
        %5 = math.exp %4 : f32
        linalg.yield %5 : f32
      } -> tensor<1024xf32>
    return %1 :tensor<1024xf32>
  }
}

// CHECK: [[FPTOSI:%.+]] = llvm.fptosi {{.*}} : vector<128xf32> to vector<128xi8>
// CHECK-NEXT: [[MUL:%.+]] = llvm.mul [[FPTOSI]], [[FPTOSI]] : vector<128xi8>
// CHECK-NEXT: [[SITOFP:%.+]] = llvm.sitofp [[MUL]] : vector<128xi8> to vector<128xf32>
// CHECK-NEXT: [[VEXTRACT1:%.+]] = llvm.intr.vector.extract [[SITOFP]][0] : vector<32xf32> from vector<128xf32>
// CHECK-NEXT: [[BITCAST1:%.+]] = llvm.bitcast [[VEXTRACT1]] : vector<32xf32> to vector<32xi32>
// CHECK-NEXT: [[EXP1:%.+]] = llvm.call @_hexagon_runtime_exp__vf([[BITCAST1]]) : (vector<32xi32>) -> vector<32xi32>
// CHECK-NEXT: [[BITCAST2:%.+]] = llvm.bitcast [[EXP1]] : vector<32xi32> to vector<32xf32>
// CHECK-NEXT: [[VINSERT1:%.+]] = llvm.intr.vector.insert [[BITCAST2]], [[UNDEF:%.+]][0] : vector<32xf32> into vector<128xf32>
// CHECK-NEXT: [[VEXTRACT2:%.+]] = llvm.intr.vector.extract [[SITOFP]][32] : vector<32xf32> from vector<128xf32>
// CHECK-NEXT: [[BITCAST3:%.+]] = llvm.bitcast [[VEXTRACT2]] : vector<32xf32> to vector<32xi32>
// CHECK-NEXT: [[EXP2:%.+]] = llvm.call @_hexagon_runtime_exp__vf([[BITCAST3]]) : (vector<32xi32>) -> vector<32xi32>
// CHECK-NEXT: [[BITCAST4:%.+]] = llvm.bitcast [[EXP2]] : vector<32xi32> to vector<32xf32>
// CHECK-NEXT: [[VINSERT2:%.+]] = llvm.intr.vector.insert [[BITCAST4]], [[VINSERT1]][32] : vector<32xf32> into vector<128xf32>

// -----

#map = affine_map<(d0) -> (d0)>
module attributes {llvm.target_triple = "hexagon"} {
  func.func @test_exp2_vector_widening(%arg0: tensor<1024xf32>) -> tensor<1024xf32> {
    %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%arg0: tensor<1024xf32>) outs(%arg0 : tensor<1024xf32>) {
      ^bb0(%in: f32, %out: f32):
        %2 = arith.fptosi %in : f32 to i8
        %3 = arith.muli %2, %2 : i8
        %4 = arith.sitofp %3 : i8 to f32
        %5 = math.exp2 %4 : f32
        linalg.yield %5 : f32
      } -> tensor<1024xf32>
    return %1 :tensor<1024xf32>
  }
}

// CHECK: [[FPTOSI:%.+]] = llvm.fptosi {{.*}} : vector<128xf32> to vector<128xi8>
// CHECK-NEXT: [[MUL:%.+]] = llvm.mul [[FPTOSI]], [[FPTOSI]] : vector<128xi8>
// CHECK-NEXT: [[SITOFP:%.+]] = llvm.sitofp [[MUL]] : vector<128xi8> to vector<128xf32>
// CHECK-NEXT: [[VEXTRACT1:%.+]] = llvm.intr.vector.extract [[SITOFP]][0] : vector<32xf32> from vector<128xf32>
// CHECK-NEXT: [[CONSTCAST1:%.+]] = llvm.bitcast [[CONST1:%.+]] : vector<32xf32> to vector<32xi32>
// CHECK-NEXT: [[BITCAST1:%.+]] = llvm.bitcast [[VEXTRACT1]] : vector<32xf32> to vector<32xi32>
// CHECK-NEXT: [[EXP1:%.+]] = llvm.call @_hexagon_runtime_pow__vf([[CONSTCAST1]], [[BITCAST1]]) : (vector<32xi32>, vector<32xi32>) -> vector<32xi32>
// CHECK-NEXT: [[BITCAST2:%.+]] = llvm.bitcast [[EXP1]] : vector<32xi32> to vector<32xf32>
// CHECK-NEXT: [[VINSERT1:%.+]] = llvm.intr.vector.insert [[BITCAST2]], [[UNDEF:%.+]][0] : vector<32xf32> into vector<128xf32>
// CHECK-NEXT: [[VEXTRACT2:%.+]] = llvm.intr.vector.extract [[SITOFP]][32] : vector<32xf32> from vector<128xf32>
// CHECK-NEXT: [[CONSTCAST2:%.+]] = llvm.bitcast [[CONST1]] : vector<32xf32> to vector<32xi32>
// CHECK-NEXT: [[BITCAST3:%.+]] = llvm.bitcast [[VEXTRACT2]] : vector<32xf32> to vector<32xi32>
// CHECK-NEXT: [[EXP2:%.+]] = llvm.call @_hexagon_runtime_pow__vf([[CONSTCAST2]], [[BITCAST3]]) : (vector<32xi32>, vector<32xi32>) -> vector<32xi32>
// CHECK-NEXT: [[BITCAST4:%.+]] = llvm.bitcast [[EXP2]] : vector<32xi32> to vector<32xf32>
// CHECK-NEXT: [[VINSERT2:%.+]] = llvm.intr.vector.insert [[BITCAST4]], [[VINSERT1]][32] : vector<32xf32> into vector<128xf32>
