// RUN: linalg-hexagon-opt %s -linalg-to-llvm="enable-lwp=true" | FileCheck %s
// -check-prefixes=CHECK

module {
  func.func @minimal_loop() -> f32 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %zero = arith.constant 0.0 : f32

    %result = scf.for %i = %c0 to %c4 step %c1 iter_args(%val = %zero) -> (f32) {
      %next = arith.addf %val, %val : f32
      scf.yield %next : f32
    }

    return %result : f32
  }
}

// CHECK: llvm.mlir.global internal constant @handler_name("lwp_handler\00") {addr_space = 0 : i32}
// CHECK: llvm.func @llvm.hexagon.instrprof.custom(!llvm.ptr, i32)
// CHECK: %[[MARKER:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK: %[[ADDR:.*]] = llvm.getelementptr {{.*}} : (!llvm.ptr) -> !llvm.ptr, !llvm.array<12 x i8>
// CHECK: llvm.call @llvm.hexagon.instrprof.custom(%[[ADDR]], %[[MARKER]]) : (!llvm.ptr, i32) -> ()
// CHECK: llvm.call @llvm.hexagon.instrprof.custom(%[[ADDR]], %[[MARKER]]) : (!llvm.ptr, i32) -> ()
