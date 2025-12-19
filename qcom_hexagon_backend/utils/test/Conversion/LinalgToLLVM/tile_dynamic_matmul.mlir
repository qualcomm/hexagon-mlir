// RUN: linalg-hexagon-opt -split-input-file %s -linalg-to-llvm="enable-split-reduction=false" | FileCheck %s

func.func @mixed_matmul(
  %arg0: tensor<?x1024xf32>, %arg1: tensor<?x1024xf32>, %arg2: tensor<?x1024xf32>)
    -> tensor<?x1024xf32> {
  %0 = linalg.matmul  ins(%arg0, %arg1: tensor<?x1024xf32>, tensor<?x1024xf32>)
                     outs(%arg2: tensor<?x1024xf32>)
    -> tensor<?x1024xf32>
  return %0 : tensor<?x1024xf32>
}

// CHECK-LABEL: @mixed_matmul(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i64, %arg7: !llvm.ptr, %arg8: !llvm.ptr, %arg9: i64, %arg10: i64, %arg11: i64, %arg12: i64, %arg13: i64, %arg14: !llvm.ptr, %arg15: !llvm.ptr, %arg16: i64, %arg17: i64, %arg18: i64, %arg19: i64, %arg20: i64) -> !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> attributes {llvm.emit_c_interface} {
// CHECK: %{{[0-9]+}} = llvm.fmul %{{[0-9]+}}, %{{[0-9]+}}  {fastmathFlags = #llvm.fastmath<fast>} : vector<32xf32>

// -----

func.func @dynamic_matmul(
  %arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>, %arg2: tensor<?x?xf32>)
    -> tensor<?x?xf32> {
  %0 = linalg.matmul  ins(%arg0, %arg1: tensor<?x?xf32>, tensor<?x?xf32>)
                     outs(%arg2: tensor<?x?xf32>)
    -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// CHECK-LABEL: @dynamic_matmul
// CHECK: %{{[0-9]+}} = llvm.mlir.constant(32 : index) : i64
// CHECK: llvm.cond_br %{{[0-9]+}}, ^bb{{[0-9]+}}, ^bb{{[0-9]+}}
// CHECK: llvm.cond_br %{{[0-9]+}}, ^bb{{[0-9]+}}, ^bb{{[0-9]+}}
// CHECK: llvm.cond_br %{{[0-9]+}}, ^bb{{[0-9]+}}, ^bb{{[0-9]+}}
// CHECK: llvm.cond_br %{{[0-9]+}}, ^bb{{[0-9]+}}, ^bb{{[0-9]+}}
