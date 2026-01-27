// RUN: linalg-hexagon-opt %s -pass-pipeline='builtin.module(func.func(lower-tptr),convert-to-llvm,cse,canonicalize)' | FileCheck %s

module {
func.func @tptr_support(%base : memref<3x4xi32>, %offset : index) -> memref<3x4xi32> {
  %0 = tptr.from_memref %base : memref<3x4xi32> to <#tptr.default_memory_space> 
  %1 = tptr.ptradd %0 %offset : <#tptr.default_memory_space>, index to <#tptr.default_memory_space>
  %2 = tptr.ptradd %1 %offset : <#tptr.default_memory_space>, index to <#tptr.default_memory_space>
  %3 = tptr.to_memref %2 : <#tptr.default_memory_space> to memref<3x4xi32>
  return %3 : memref<3x4xi32>  
}
}

// CHECK-LABEL:   llvm.func @tptr_support
// CHECK-DAG:     %[[UNDEF_STRUCT:.*]] = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-DAG:     %[[ZERO:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK-DAG:     %[[THREE:.*]] = llvm.mlir.constant(3 : i64) : i64
// CHECK-DAG:     %[[FOUR:.*]] = llvm.mlir.constant(4 : i64) : i64
// CHECK-DAG:     %[[ONE:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-DAG:     %[[PTR_AS_INT:.*]] = llvm.ptrtoint {{.*}} : !llvm.ptr to i64
// CHECK:         %[[OFFSET1:.*]] = llvm.add %[[PTR_AS_INT]], {{.*}} : i64
// CHECK:         %[[OFFSET2:.*]] = llvm.add %[[OFFSET1]], {{.*}} : i64
// CHECK:         %[[ADJUSTED_PTR:.*]] = llvm.inttoptr %[[OFFSET2]] : i64 to !llvm.ptr
// CHECK-NEXT:    %[[INSERT0:.*]] = llvm.insertvalue %[[ADJUSTED_PTR]], %[[UNDEF_STRUCT]][0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[INSERT1:.*]] = llvm.insertvalue %[[ADJUSTED_PTR]], %[[INSERT0]][1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[INSERT2:.*]] = llvm.insertvalue %[[ZERO]], %[[INSERT1]][2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[INSERT3:.*]] = llvm.insertvalue %[[THREE]], %[[INSERT2]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[INSERT4:.*]] = llvm.insertvalue %[[FOUR]], %[[INSERT3]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[INSERT5:.*]] = llvm.insertvalue %[[FOUR]], %[[INSERT4]][3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT:    %[[FINAL_STRUCT:.*]] = llvm.insertvalue %[[ONE]], %[[INSERT5]][4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK:         llvm.return %[[FINAL_STRUCT]] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>

