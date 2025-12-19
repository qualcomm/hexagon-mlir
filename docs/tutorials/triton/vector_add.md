# 🚀 Triton Kernel Lowering to Hexagon: Vector Addition Tutorial

> **Overview**: This tutorial demonstrates the compilation pipeline from Python Triton kernel to Hexagon object code through a vector addition example.

## Table of Contents

- [🔄 Compilation Pipeline Overview](#-compilation-pipeline-overview)
  - [🔧 Compilation Stages](#-compilation-stages)
- [🧮 Vector Add Triton Kernel](#-vector-add-triton-kernel)
  - [📝 Python Implementation](#-python-implementation)
  - [⚙️ Kernel Configuration](#️-kernel-configuration)
- [🏃 Compilation and Execution](#-compilation-and-execution)
  - [🔨 Build Commands](#-build-commands)
  - [✅ Expected Output](#-expected-output)
- [🔍 Deep Dive into Lowering](#-deep-dive-into-lowering)
  - [Triton IR](#triton-ir)
  - [Triton-IR to Linalg](#triton-ir-to-linalg)
  - [Punting Input Buffers](#punting-input-buffers)
  - [Tiled, Multi-threaded, Vectorized Code](#tiled-multi-threaded-vectorized-code)
  - [LLVM-IR](#llvm-ir)
- [🔄 Creating Shared Object](#-creating-shared-object)

---

## 🔄 Compilation Pipeline Overview

The Python Triton kernel undergoes multi-stage compilation process.

### 🔧 Compilation Stages

| Stage | Tool/Library | Description |
|-------|-------------|-------------|
| **1** | [triton-compiler](https://github.com/triton-lang/triton) | `Triton Kernel → Triton-IR` |
| **2** | [triton-shared](https://github.com/microsoft/triton-shared) | `Triton IR → Linalg IR` |
| **3** | `linalg-hexagon-opt`/libcall | `Linalg IR → MLIR-LLVM` (ML lowering and opt) |
| **4** | `linalg-hexagon-translate` | `MLIR-LLVM IR → Object Code` |
| **5** | Linker | Create `libadd_kernel.so` |
| **6** | Runtime | Load and execute on device |

> **💡 Key Insight**: Stage 3 and 4 is the heart of our compiler, leveraging HMX and HVX engines, TCM (Tightly Coupled Memory), DMA transfers, multi-threading support, and custom ML optimizations.

---

## 🧮 Vector Add Triton Kernel

This example demonstrates element-wise addition of two tensors and can be found in `test/python/triton/test_vec_add.py`.

### 📝 Python Implementation

```python
import torch
import triton
import triton.language as tl
from triton.backends.qcom_hexagon_backend.driver import HexagonDriver

triton.runtime.driver.set_active(HexagonDriver())
BLOCK_SIZE = 131072

@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offsets)
    y = tl.load(y_ptr + offsets)
    output = x + y
    tl.store(output_ptr + offsets, output)

def test_vec_add():
    x = torch.rand(BLOCK_SIZE)
    y = torch.rand(BLOCK_SIZE)
    output = torch.empty_like(x)
    
    add_kernel[(1,)](x, y, output,
                     BLOCK_SIZE=BLOCK_SIZE,
                     enableMultiThreading=True,
                     enableVTCMTiling=True,
                     enableConvertToHexagonmem=True,
                     enableHexagonmemCopyToDMA=True)
```

### ⚙️ Kernel Configuration

The kernel is configured with several optimization flags:

- **`enableMultiThreading=True`**: Enables parallel execution across multiple HVX threads.
- **`enableVTCMTiling=True`**: Utilizes Tightly Coupled Memory for efficient data access.
- **`enableConvertToHexagonmem=True`**: Optimizes memory layout for Hexagon architecture.
- **`enableHexagonmemCopyToDMA=True`**: Enables DMA transfers for improved memory bandwidth.

---

## 🏃 Compilation and Execution

### 🔨 Build Commands

```bash
# Basic execution
TRITON_ALWAYS_COMPILE=1 pytest -sv test/python/triton/test_vec_add.py

# Enable MLIR IR and LLVM-IR dumping to see intermediate representations
TRITON_ALWAYS_COMPILE=1 MLIR_ENABLE_DUMP=1 LLVM_IR_ENABLE_DUMP=1 pytest -sv test/python/triton/test_vec_add.py
```

### ✅ Expected Output

If setup and device acquisition work correctly, you should see:

```bash
Test_Info: {
        Name:add_kernel
        Result:Pass
        Perf: xxxx
        Units:us
}
==> Ran successfully, collecting results
PASSED                                  
```

---

## 🔍 Deep Dive into Lowering

This section explores the intermediate representations at each compilation stage.

### Triton IR

**Stage 1**: After initial Triton front-end processing

```mlir
module {
  tt.func public @add_kernel(%x_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32} ,
                             %y_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32} ,
                              %output_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32} )) attributes {noinline = false} {
    %offsets = tt.make_range {end = 131072 : i32, start = 0 : i32} : tensor<131072xi32>
    %x = tt.splat %x_ptr : !tt.ptr<f32> -> tensor<131072x!tt.ptr<f32>> 
    %x_0 = tt.addptr %x, %offsets : tensor<131072x!tt.ptr<f32>>, tensor<131072xi32> 
    %x_1 = tt.load %x_0 : tensor<131072x!tt.ptr<f32>> loc(#loc15)
    %y = tt.splat %y_ptr : !tt.ptr<f32> -> tensor<131072x!tt.ptr<f32>> 
    %y_2 = tt.addptr %y, %offsets : tensor<131072x!tt.ptr<f32>>, tensor<131072xi32>
    %y_3 = tt.load %y_2 : tensor<131072x!tt.ptr<f32>>
    %output = arith.addf %x_1, %y_3 : tensor<131072xf32>
    %0 = tt.splat %output_ptr : !tt.ptr<f32> -> tensor<131072x!tt.ptr<f32>>
    %1 = tt.addptr %0, %offsets : tensor<131072x!tt.ptr<f32>>, tensor<131072xi32> 
    tt.store %1, %output : tensor<131072x!tt.ptr<f32>> 
     tt.return
  }
} 
```

### Triton-IR to Linalg

**Stage 2**: After triton-to-linalg lowering using triton-shared

```mlir
map = affine_map<(d0) -> (d0)>
module {
  func.func @add_kernel(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, 
                        %arg1: memref<*xf32> {tt.divisibility = 16 : i32}, 
                        %arg2: memref<*xf32> {tt.divisibility = 16 : i32},
                        %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32) {
    %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [0], sizes: [131072], strides: [1] 
             : memref<*xf32> to memref<131072xf32, strided<[1]>>
    %alloc = memref.alloc() : memref<131072xf32>
    memref.copy %reinterpret_cast, %alloc : memref<131072xf32, strided<[1]>> to memref<131072xf32>
    %0 = bufferization.to_tensor %alloc restrict writable : memref<131072xf32> to tensor<131072xf32>
    %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [0], sizes: [131072], strides: [1]
             : memref<*xf32> to memref<131072xf32, strided<[1]>>
    %alloc_1 = memref.alloc() : memref<131072xf32>
    memref.copy %reinterpret_cast_0, %alloc_1 : memref<131072xf32, strided<[1]>> to memref<131072xf32>
    %1 = bufferization.to_tensor %alloc_1 restrict writable : memref<131072xf32> to tensor<131072xf32>

    %2 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]}
           ins(%0, %1 : tensor<131072xf32>, tensor<131072xf32>) outs(%0 : tensor<131072xf32>) {
    ^bb0(%in: f32, %in_3: f32, %out: f32):
      %3 = arith.addf %in, %in_3 : f32
      linalg.yield %3 : f32
    } -> tensor<131072xf32>

    %reinterpret_cast_2 = memref.reinterpret_cast %arg2 to offset: [0], sizes: [131072], strides: [1]
            : memref<*xf32> to memref<131072xf32, strided<[1]>>
    bufferization.materialize_in_destination %2 in writable %reinterpret_cast_2
            : (tensor<131072xf32>, memref<131072xf32, strided<[1]>>) -> ()
    return
  }
}
```

### Punting Input Buffers


> **🎯 Optimization Goal**: Remove redundant memory operations that aren't needed in our execution context.

```mlir
#map = affine_map<(d0) -> (d0)>
module attributes {llvm.target_triple = "hexagon"} {
  func.func @add_kernel(%arg0: memref<*xf32> {tt.divisibility = 16 : i32},
                        %arg1: memref<*xf32> {tt.divisibility = 16 : i32},
                        %arg2: memref<*xf32> {tt.divisibility = 16 : i32},
                        %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32) {
   
    %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [0], sizes: [131072], strides: [1]
           : memref<*xf32> to memref<131072xf32, strided<[1]>>
    %0 = bufferization.to_tensor %reinterpret_cast restrict
           : memref<131072xf32, strided<[1]>> to tensor<131072xf32>
    %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [0], sizes: [131072], strides: [1]
           : memref<*xf32> to memref<131072xf32, strided<[1]>>
    %1 = bufferization.to_tensor %reinterpret_cast_0 restrict : memref<131072xf32, strided<[1]>> to tensor<131072xf32>
   
    %2 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]}
           ins(%0, %1 : tensor<131072xf32>, tensor<131072xf32>) outs(%0 : tensor<131072xf32>) {
    ^bb0(%in: f32, %in_2: f32, %out: f32):
      %3 = arith.addf %in, %in_2 : f32
      linalg.yield %3 : f32
    } -> tensor<131072xf32>
    %reinterpret_cast_1 = memref.reinterpret_cast %arg2 to offset: [0], sizes: [131072], strides: [1] 
           : memref<*xf32> to memref<131072xf32, strided<[1]>>
    bufferization.materialize_in_destination %2 in writable %reinterpret_cast_1 
          : (tensor<131072xf32>, memref<131072xf32, strided<[1]>>) -> ()
    return
  }
}
```

### Tiled, Multi-threaded, Vectorized Code

**Optimizations**: The IR shown below is after some optimization passes have run, and those for this example   multi-threading and vectorization.

> **🔧 Analysis Tip**: Use `MLIR_ENABLE_DUMP=1` to analyze IR after each pass.

#### 🎯 Key Optimizations Observed:

- **🧵 Multi-threading**: Uses `async-execute` (lowers to LLVM coroutines).
  - Breaks 131,072 elements into **4 HVX threads** × 32,768 elements each.
- **🔲 Tiling**: In this example the tensors fit entirely in TCM, so are brought in full before computation.
- **📡 DMA Transfers**: Efficient DDR ↔ TCM data movement.
- **⚡ Vectorization**: Main computation vectorized for HVX (`vector<32xf32>`).

```mlir
  ...
  memref.dma_start %reinterpret_cast_3[%c0], %2[%c0], %c131072, %alloc_4[%c0] : memref<131072xf32, strided<[1]>>, memref<131072xf32, 1>, memref<1xi32>
  memref.dma_wait %alloc_4[%c0], %c131072 : memref<1xi32>
  memref.dealloc %alloc_4 : memref<1xi32>
  %c0_5 = arith.constant 0 : index
  %c131072_6 = arith.constant 131072 : index
  %c32768_7 = arith.constant 32768 : index
  %3 = arith.divui %c131072_6, %c32768_7 : index
  %4 = async.create_group %3 : !async.group
  scf.for %arg9 = %c0_5 to %c131072_6 step %c32768_7 {
    %token = async.execute {
      %subview = memref.subview %0[%arg9] [32768] [1]
                   : memref<131072xf32, 1> to memref<32768xf32, strided<[1], offset: ?>, 1>
      %subview_10 = memref.subview %2[%arg9] [32768] [1]
                   : memref<131072xf32, 1> to memref<32768xf32, strided<[1], offset: ?>, 1>
      %subview_11 = memref.subview %1[%arg9] [32768] [1]
                   : memref<131072xf32, 1> to memref<32768xf32, strided<[1], offset: ?>, 1>

      scf.for %arg10 = %c0 to %c32768 step %c32 {
        %subview_12 = memref.subview %subview[%arg10] [32] [1]
                        : memref<32768xf32, strided<[1], offset: ?>, 1> to memref<32xf32, strided<[1], offset: ?>, 1>
        %subview_13 = memref.subview %subview_10[%arg10] [32] [1]
                        : memref<32768xf32, strided<[1], offset: ?>, 1> to memref<32xf32, strided<[1], offset: ?>, 1>
        %subview_14 = memref.subview %subview_11[%arg10] [32] [1]
                        : memref<32768xf32, strided<[1], offset: ?>, 1> to memref<32xf32, strided<[1], offset: ?>, 1>

        %6 = vector.transfer_read %subview_12[%c0], %cst {in_bounds = [true]} 
                        : memref<32xf32, strided<[1], offset: ?>, 1>, vector<32xf32>
        %7 = vector.transfer_read %subview_13[%c0], %cst {in_bounds = [true]}
                        : memref<32xf32, strided<[1], offset: ?>, 1>, vector<32xf32>
        %8 = arith.addf %6, %7 fastmath<fast> : vector<32xf32>
        vector.transfer_write %8, %subview_14[%c0] {in_bounds = [true]}
                        : vector<32xf32>, memref<32xf32, strided<[1], offset: ?>, 1>
      }
      async.yield
    }
    %5 = async.add_to_group %token, %4 : !async.token
  }
  ```

### LLVM-IR

**Final Stage**: The main computational kernel lowered to LLVM-IR. Each thread executes this optimized code with data loaded from TCM for HVX vector operations.

```mlir
  ; Function Attrs: mustprogress nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #3

define internal fastcc void @async_execute_fn.resume(ptr noundef nonnull align 8 captures(none) dereferenceable(40) %0) #1 {
.preheader.from.AfterCoroSuspend:
  %.reload.addr9 = getelementptr inbounds nuw i8, ptr %0, i32 24
  %.reload.addr6 = getelementptr inbounds nuw i8, ptr %0, i32 20
  %.reload.addr3 = getelementptr inbounds nuw i8, ptr %0, i32 8
  %.reload.addr = getelementptr inbounds nuw i8, ptr %0, i32 16
  br label %.preheader

.preheader:     ; preds = %.preheader, %.preheader.from.AfterCoroSuspend
  %1 = phi i64 [ 0, %.preheader.from.AfterCoroSuspend ], [ %10, %.preheader ]
  %.reload10 = load ptr, ptr %.reload.addr9, align 8
  %.reload7 = load ptr, ptr %.reload.addr6, align 4
  %.reload4 = load i64, ptr %.reload.addr3, align 8
  %.reload = load ptr, ptr %.reload.addr, align 8
  %2 = add i64 %.reload4, %1
  %3 = trunc i64 %2 to i32
  %4 = getelementptr float, ptr %.reload, i32 %3
  %5 = load <32 x float>, ptr %4, align 4
  %6 = getelementptr float, ptr %.reload7, i32 %3
  %7 = load <32 x float>, ptr %6, align 4
  %8 = fadd fast <32 x float> %7, %5
  %9 = getelementptr float, ptr %.reload10, i32 %3
  store <32 x float> %8, ptr %9, align 4
  %10 = add nuw nsw i64 %1, 32
  %11 = icmp samesign ult i64 %1, 32736
  br i1 %11, label %.preheader, label %CoroEnd

CoroEnd:                                          ; preds = %.preheader
  %.reload.addr12 = getelementptr inbounds nuw i8, ptr %0, i32 28
  %.reload13 = load ptr, ptr %.reload.addr12, align 4
  tail call void @mlirAsyncRuntimeEmplaceToken(ptr %.reload13)
  tail call void @free(ptr nonnull %0)
  ret void
}
```

---
## 🔄 Creating Shared Object

Once the kernel has been lowered and compiled to an objcet file (`add_kernel.o` in this partcular example), the scripts create a shared-object. You will see something like these printed on screen.

```bash
==> Folder 'add_kernel-...date...' created successfully inside of /tmp (as default location)
==> kernel obj saved in: /tmp/add_kernel-2025-10-08_05-16-17/add_kernel.o
==> Wrapper generator correctly instantiated
wrapper_path =  /tmp/add_kernel-2025-10-08_05-16-17/add_kernel_wrapper.cpp
kernel_code =  /tmp/add_kernel-2025-10-08_05-16-17/add_kernel.o
==> Compiling shared object file ...
hexagon-clang++ -O3 -mv73 -mhvx -I/../hexagon-mlir/test/utils/dsp/include -I/../Hexagon_SDK/6.0.0.2/incs -I/../Hexagon_SDK/6.0.0.2/incs/stddef -I/../qurt -I/../rtos/qurt/...L/../hexagon-mlir/triton/build/cmake.linux-x86_64-cpython-3.11/third_party/qcom_hexagon_backend/bin/runtime -...tmp/add_kernel-2025-10-08_05-16-17/add_kernel_wrapper.cpp /tmp/add_kernel-2025-10-08_05-16-17/add_kernel.o -lqhmath_hvx -lhexagon_mlir_async_runtime  -shared -fPIC -G0  -o /tmp/add_kernel-2025-10-08_05-16-17/libadd_kernel.so
==> Shared object generated:  /tmp/add_kernel-2025-10-08_05-16-17/libadd_kernel.so
==> Running the kernel on device ...
```
The `add_kernel_wrapper.cpp` is an auto-generated code to connect main to the compiled kernel.
```C
#define FARF_ALWAYS 1
#include <HAP_farf.h>
...

extern "C" void add_kernel(int64_t, MemRefDescriptor<float, 1> *, 
                           int64_t, MemRefDescriptor<float, 1> *, 
                          int64_t, MemRefDescriptor<float, 1> *, int, int, int, int, int, int);

int main() {
  Tensor<float, 1> T0({{131072}, {1}, 128}, MemType::HEAP);
  MemRefDescriptor<float, 1> *dt0 = T0.toMemRefDesc();
  ...
  T0.load_from_file("/vendor/bin/add_kernel_t0.raw");
  ...
  uint64_t avg_time_us = benchmark_time_us(50, [&]() {
      add_kernel(1, dt0, 1, dt1, 1, dt2, 1, 1, 1, 0, 0, 0);
  });
  TestReport tr("add_kernel", avg_time_us, "us", Result::Pass);
  tr.save();
  ...
  return 0;
}
```
All the above IR and files can be found from debug prints and inspected.


> **🎉 Congratulations!** You've successfully walked through the complete compilation pipeline from high-level Python Triton code to optimized Hexagon assembly, witnessing the sophisticated optimizations that make efficient ML computation possible on Hexagon architecture.
