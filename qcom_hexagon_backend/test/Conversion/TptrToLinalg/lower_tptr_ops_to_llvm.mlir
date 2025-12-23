// RUN: linalg-hexagon-opt  %s -pass-pipeline='builtin.module(func.func(lower-tptr),convert-to-llvm,cse,canonicalize)' | FileCheck %s

module {
func.func @ptradd(%base : memref<1xi32>, %offset : i32) -> memref<1xi32> {
  %0 = tptr.from_memref %base : memref<1xi32> to <#tptr.default_memory_space> 
  %1 = tptr.ptradd %0 %offset : <#tptr.default_memory_space>, i32 to <#tptr.default_memory_space>
  %2 = tptr.to_memref %1 : <#tptr.default_memory_space> to memref<1xi32>
  return %2 : memref<1xi32>  
}
}

// CHECK-LABEL:   llvm.func @ptradd
// CHECK-DAG:       %[[RES:.*]] = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-DAG:       %[[ZERO:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK-DAG:       %[[ONE:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-DAG:       %[[PTR_AS_INT:.*]] = llvm.ptrtoint {{.*}} : !llvm.ptr to i64
// CHECK-DAG:       %[[SEXT_INDEX:.*]] = llvm.sext {{.*}} : i32 to i64
// CHECK:           %[[OFFSET:.*]] = llvm.add %[[PTR_AS_INT]], %[[SEXT_INDEX]] : i64
// CHECK:           %[[ADJUSTED_PTR:.*]] = llvm.inttoptr %[[OFFSET]] : i64 to !llvm.ptr
// CHECK:           %[[INSERT0:.*]] = llvm.insertvalue %[[ADJUSTED_PTR]], %[[RES]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:      %[[INSERT1:.*]] = llvm.insertvalue %[[ADJUSTED_PTR]], %[[INSERT0]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:      %[[INSERT2:.*]] = llvm.insertvalue %[[ZERO]], %[[INSERT1]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:      %[[INSERT3:.*]] = llvm.insertvalue %[[ONE]], %[[INSERT2]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:      %[[FINAL_STRUCT:.*]] = llvm.insertvalue %[[ONE]], %[[INSERT3]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK:           llvm.return %[[FINAL_STRUCT]] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
