// RUN: linalg-hexagon-opt %s -hexagonmem-to-llvm | FileCheck %s

// CHECK-LABEL:         llvm.func @hexagon_runtime_alloc_1d(i32, i64, i1) -> !llvm.ptr
// CHECK-LABEL: @func
// CHECK-DAG: %[[ALIGNMENT1:.+]] = llvm.mlir.constant(128 : index) : i64
// CHECK-DAG: %[[IS_VTCM1:.+]] = llvm.mlir.constant(true) : i1
// CHECK-DAG: %[[C16_1:.+]] = llvm.mlir.constant(16 : index) : i64
// CHECK-DAG: %[[C4_1:.+]] = llvm.mlir.constant(4 : index) : i64
// CHECK-DAG: %[[C1_1:.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-DAG: %[[C64_1:.+]] = llvm.mlir.constant(64 : index) : i64
// CHECK-DAG: %[[ZERO_PTR1:.+]] = llvm.mlir.zero : !llvm.ptr
// CHECK: {{.*}} = llvm.getelementptr %[[ZERO_PTR1]][%[[C64_1]]] : (!llvm.ptr, i64) -> !llvm.ptr, i8
// CHECK: {{.*}} = llvm.ptrtoint {{.*}} : !llvm.ptr to i64
// CHECK: %[[NUM_BYTES1:.+]] = llvm.trunc {{.*}} : i64 to i32
// CHECK: %[[VTCM_PTR1:.+]] = llvm.call @hexagon_runtime_alloc_1d(%[[NUM_BYTES1]], %[[ALIGNMENT1]], %[[IS_VTCM1]]) : (i32, i64, i1) -> !llvm.ptr
//
// CHECK: {{.*}} = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK: {{.*}} = llvm.insertvalue %[[VTCM_PTR1:.+]], {{.*}}[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK: {{.*}} = llvm.insertvalue %[[VTCM_PTR1:.+]], {{.*}}[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK: %[[OFFSET1:.+]] = llvm.mlir.constant(0 : index) : i64
// CHECK: {{.*}} = llvm.insertvalue %[[OFFSET1]], {{.*}}[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK: {{.*}} = llvm.insertvalue %[[C16_1]], {{.*}}[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK: {{.*}} = llvm.insertvalue %[[C4_1]], {{.*}}[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK: {{.*}} = llvm.insertvalue %[[C4_1]], {{.*}}[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK: %[[MEMREF:.+]] = llvm.insertvalue %[[C1_1]], {{.*}}[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>

// Check that the second allocation also allocates 64 bytes
// This is because `memref<62xi8, #map>` after normalization becomes `memref<16x4xi8>`
// CHECK-DAG: %[[ALIGNMENT2:.+]] = llvm.mlir.constant(128 : index) : i64
// CHECK-DAG: %[[IS_VTCM2:.+]] = llvm.mlir.constant(true) : i1
// CHECK-DAG: %[[C16_2:.+]] = llvm.mlir.constant(16 : index) : i64
// CHECK-DAG: %[[C4_2:.+]] = llvm.mlir.constant(4 : index) : i64
// CHECK-DAG: %[[C1_2:.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-DAG: %[[C64_2:.+]] = llvm.mlir.constant(64 : index) : i64
// CHECK-DAG: %[[ZERO_PTR2:.+]] = llvm.mlir.zero : !llvm.ptr
// CHECK: {{.*}} = llvm.getelementptr %[[ZERO_PTR2]][%[[C64_2]]] : (!llvm.ptr, i64) -> !llvm.ptr, i8
// CHECK: {{.*}} = llvm.ptrtoint {{.*}} : !llvm.ptr to i64
// CHECK: %[[NUM_BYTES2:.+]] = llvm.trunc {{.*}} : i64 to i32
// CHECK: %[[VTCM_PTR2:.+]] = llvm.call @hexagon_runtime_alloc_1d(%[[NUM_BYTES2]], %[[ALIGNMENT2]], %[[IS_VTCM2]]) : (i32, i64, i1) -> !llvm.ptr

#map = affine_map<(i) -> (i floordiv 4, i mod 4)>
func.func @func() {
  %0 = hexagonmem.alloc() : memref<64xi8, #map, 1>
  %1 = hexagonmem.alloc() : memref<62xi8, #map, 1>
  return
}
