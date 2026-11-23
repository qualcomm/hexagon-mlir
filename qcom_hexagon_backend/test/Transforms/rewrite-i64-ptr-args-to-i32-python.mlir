//===- rewrite-i64-ptr-args-to-i32-python.mlir ----------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

// RUN: TORCH_DEVICE_BACKEND_AUTOLOAD=0 %PYTHON %S/rewrite-i64-ptr-args-to-i32-python.py | FileCheck %s

// Please Note: The FileCheck directives live in this file, but the input IR is
// in rewrite-i64-ptr-args-to-i32-python.py in the same folder, which prints
// the rewritten module to stdout for FileCheck to consume.
//
// This is done because the rewrite-i64-ptr-args-to-i32 pass operates on Triton
// (tt) dialect IR and is only exposed through the qcom_hexagon_backend Python
// binding, not via linalg-hexagon-opt. Registering it into linalg-hexagon-opt
// would force that tool to link/register the Triton dialect ecosystem
// (TritonGPU and related deps), which we intentionally keep out of the Hexagon
// opt tool. Driving the test through %PYTHON reuses the already-built Triton
// Python module and avoids adding those dependencies.

// CHECK: changed=True
// CHECK-LABEL: tt.func @kernel(%arg0: !tt.ptr<i32>{{.*}}, %arg1: i32{{.*}})
// CHECK: tt.load %{{.*}} : tensor<4x!tt.ptr<i32>>
// CHECK: arith.extsi %{{.*}} : tensor<4xi32> to tensor<4xi64>
// CHECK-LABEL: tt.func @triton_poi_fused__to_copy_unsqueeze_1(%{{.*}}: !tt.ptr<i32>{{.*}}, %{{.*}}: !tt.ptr<f32>
// CHECK: %[[PTRS:.+]] = tt.splat %{{.*}} : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>>
// CHECK: %[[LOAD:.+]] = tt.load %{{.*}} : tensor<4x!tt.ptr<i32>>
// CHECK: %[[EXT:.+]] = arith.extsi %[[LOAD]] : tensor<4xi32> to tensor<4xi64>
// CHECK: %{{.*}} = arith.sitofp %[[EXT]] : tensor<4xi64> to tensor<4xf32>
// triton_poi_fused_add_6: both in_ptr0 and out_ptr0 rewritten to ptr<i32>.
// The load value is extsi'd to i64 (uniform with all other loads), then
// trunci'd back to i32 at the store to match the narrowed destination pointer.
// CHECK-LABEL: tt.func @triton_poi_fused_add_6(%{{.*}}: !tt.ptr<i32>{{.*}}, %{{.*}}: !tt.ptr<i32>
// CHECK: %[[LOAD6:.+]] = tt.load %{{.*}} : tensor<4x!tt.ptr<i32>>
// CHECK: %[[EXT6:.+]] = arith.extsi %[[LOAD6]] : tensor<4xi32> to tensor<4xi64>
// CHECK: %[[TRUNC6:.+]] = arith.trunci %[[EXT6]] : tensor<4xi64> to tensor<4xi32>
// CHECK: tt.store %{{.*}}, %[[TRUNC6]], %{{.*}} : tensor<4x!tt.ptr<i32>>
// triton_poi_fused_addindex_7: both ptrs rewritten to ptr<i32>, with i64
// arithmetic between load and store. The load result is extsi'd to i64 for the
// addi, then the i64 sum is trunci'd back to i32 at the store to match the
// narrowed destination pointer.
// CHECK-LABEL: tt.func @triton_poi_fused_addindex_7(%{{.*}}: !tt.ptr<i32>{{.*}}, %{{.*}}: !tt.ptr<i32>
// CHECK: %[[LOAD7:.+]] = tt.load %{{.*}} : tensor<4x!tt.ptr<i32>>
// CHECK: %[[EXT7:.+]] = arith.extsi %[[LOAD7]] : tensor<4xi32> to tensor<4xi64>
// CHECK: %[[SUM7:.+]] = arith.addi %[[EXT7]], %{{.*}} : tensor<4xi64>
// CHECK: %[[TRUNC7:.+]] = arith.trunci %[[SUM7]] : tensor<4xi64> to tensor<4xi32>
// CHECK: tt.store %{{.*}}, %[[TRUNC7]], %{{.*}} : tensor<4x!tt.ptr<i32>>
// triton_poi_fused_inplace_8: a single arg used as BOTH load source and store
// destination (in-place / in_out_ptr0). The shared pointer is rewritten to
// ptr<i32> once; the load narrows to i32, i64 arithmetic stays i64, and the
// store trunci's back to i32 through the same pointer.
// CHECK-LABEL: tt.func @triton_poi_fused_inplace_8(%{{.*}}: !tt.ptr<i32>
// CHECK: %[[PTR8:.+]] = tt.addptr %{{.*}} : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
// CHECK: %[[LOAD8:.+]] = tt.load %[[PTR8]]{{.*}} : tensor<4x!tt.ptr<i32>>
// CHECK: %[[EXT8:.+]] = arith.extsi %[[LOAD8]] : tensor<4xi32> to tensor<4xi64>
// CHECK: %[[SUM8:.+]] = arith.addi %[[EXT8]], %{{.*}} : tensor<4xi64>
// CHECK: %[[TRUNC8:.+]] = arith.trunci %[[SUM8]] : tensor<4xi64> to tensor<4xi32>
// CHECK: tt.store %[[PTR8]], %[[TRUNC8]], %{{.*}} : tensor<4x!tt.ptr<i32>>
// CHECK-NOT: !tt.ptr<i64>
