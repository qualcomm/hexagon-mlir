// RUN: linalg-hexagon-opt %s -hexagonmem-to-llvm | FileCheck %s

// CHECK-LABEL:   llvm.func @hexagon_runtime_free_2d(!llvm.ptr)
// CHECK:         llvm.func @hexagon_runtime_copy(!llvm.ptr, !llvm.ptr, i32, i1, i1)
// CHECK:         llvm.func @hexagon_runtime_alloc_2d(i32, i32, i64, i1) -> !llvm.ptr

// CHECK-LABEL:   func.func @add
// CHECK:           %[[VAL_3:.*]] = builtin.unrealized_conversion_cast %[[VAL_2:.*]] : !crouton.crouton<64xi8> to !llvm.ptr
// CHECK:           %[[VAL_4:.*]] = builtin.unrealized_conversion_cast %[[VAL_1:.*]] : !crouton.crouton<64xi8> to !llvm.ptr
// CHECK:           %[[VAL_5:.*]] = builtin.unrealized_conversion_cast %[[VAL_0:.*]] : !crouton.crouton<64xi8> to !llvm.ptr
// CHECK:           %[[VAL_6:.*]] = llvm.mlir.constant(2048 : index) : i64
// CHECK:           %[[VAL_7:.*]] = llvm.mlir.constant(true) : i1
// CHECK:           %[[VAL_8:.*]] = llvm.mlir.constant(64 : i32) : i32
// CHECK:           %[[VAL_9:.*]] = llvm.mlir.constant(2048 : i32) : i32
// CHECK:           %[[VAL_10:.*]] = llvm.call @hexagon_runtime_alloc_2d(%[[VAL_8]], %[[VAL_9]], %[[VAL_6]], %[[VAL_7]]) : (i32, i32, i64, i1) -> !llvm.ptr
// CHECK:           %[[VAL_11:.*]] = llvm.mlir.constant(2048 : index) : i64
// CHECK:           %[[VAL_12:.*]] = llvm.mlir.constant(true) : i1
// CHECK:           %[[VAL_13:.*]] = llvm.mlir.constant(64 : i32) : i32
// CHECK:           %[[VAL_14:.*]] = llvm.mlir.constant(2048 : i32) : i32
// CHECK:           %[[VAL_15:.*]] = llvm.call @hexagon_runtime_alloc_2d(%[[VAL_13]], %[[VAL_14]], %[[VAL_11]], %[[VAL_12]]) : (i32, i32, i64, i1) -> !llvm.ptr
// CHECK:           %[[VAL_16:.*]] = llvm.mlir.constant(2048 : index) : i64
// CHECK:           %[[VAL_17:.*]] = llvm.mlir.constant(true) : i1
// CHECK:           %[[VAL_18:.*]] = llvm.mlir.constant(64 : i32) : i32
// CHECK:           %[[VAL_19:.*]] = llvm.mlir.constant(2048 : i32) : i32
// CHECK:           %[[VAL_20:.*]] = llvm.call @hexagon_runtime_alloc_2d(%[[VAL_18]], %[[VAL_19]], %[[VAL_16]], %[[VAL_17]]) : (i32, i32, i64, i1) -> !llvm.ptr
// CHECK:           %[[VAL_21:.*]] = llvm.mlir.constant(131072 : i32) : i32
// CHECK:           %[[VAL_22:.*]] = llvm.mlir.constant(false) : i1
// CHECK:           %[[VAL_23:.*]] = llvm.mlir.constant(true) : i1
// CHECK:           llvm.call @hexagon_runtime_copy(%[[VAL_10]], %[[VAL_5]], %[[VAL_21]], %[[VAL_23]], %[[VAL_22]]) : (!llvm.ptr, !llvm.ptr, i32, i1, i1) -> ()
// CHECK:           %[[VAL_24:.*]] = llvm.mlir.constant(131072 : i32) : i32
// CHECK:           %[[VAL_25:.*]] = llvm.mlir.constant(false) : i1
// CHECK:           %[[VAL_26:.*]] = llvm.mlir.constant(true) : i1
// CHECK:           llvm.call @hexagon_runtime_copy(%[[VAL_15]], %[[VAL_4]], %[[VAL_24]], %[[VAL_26]], %[[VAL_25]]) : (!llvm.ptr, !llvm.ptr, i32, i1, i1) -> ()
// CHECK:           %[[VAL_27:.*]] = llvm.mlir.constant(131072 : i32) : i32
// CHECK:           %[[VAL_28:.*]] = llvm.mlir.constant(true) : i1
// CHECK:           %[[VAL_29:.*]] = llvm.mlir.constant(false) : i1
// CHECK:           llvm.call @hexagon_runtime_copy(%[[VAL_3]], %[[VAL_20]], %[[VAL_27]], %[[VAL_29]], %[[VAL_28]]) : (!llvm.ptr, !llvm.ptr, i32, i1, i1) -> ()
// CHECK:           llvm.call @hexagon_runtime_free_2d(%[[VAL_10]]) : (!llvm.ptr) -> ()
// CHECK:           llvm.call @hexagon_runtime_free_2d(%[[VAL_15]]) : (!llvm.ptr) -> ()
// CHECK:           llvm.call @hexagon_runtime_free_2d(%[[VAL_20]]) : (!llvm.ptr) -> ()

func.func @add(%arg0: !crouton.crouton<64xi8>, %arg1: !crouton.crouton<64xi8>, %arg2: !crouton.crouton<64xi8>) {
  %0 = hexagonmem.alloc() {alignment = 2048 : i64} : !crouton.crouton<64xi8, vtcm>
  %1 = hexagonmem.alloc() {alignment = 2048 : i64} : !crouton.crouton<64xi8, vtcm>
  %2 = hexagonmem.alloc() {alignment = 2048 : i64} : !crouton.crouton<64xi8, vtcm>
  hexagonmem.copy %arg0, %0 : !crouton.crouton<64xi8> to !crouton.crouton<64xi8, vtcm>
  hexagonmem.copy %arg1, %1 : !crouton.crouton<64xi8> to !crouton.crouton<64xi8, vtcm>
  // This is not yet supported
  // linalg.add ins(%0, %1 : !crouton.crouton<64xi8, vtcm>, !crouton.crouton<64xi8, vtcm>) outs(%2 : !crouton.crouton<64xi8, vtcm>)
  hexagonmem.copy %2, %arg2 : !crouton.crouton<64xi8, vtcm> to !crouton.crouton<64xi8>
  hexagonmem.dealloc %0 : !crouton.crouton<64xi8, vtcm>
  hexagonmem.dealloc %1 : !crouton.crouton<64xi8, vtcm>
  hexagonmem.dealloc %2 : !crouton.crouton<64xi8, vtcm>
  return
}
