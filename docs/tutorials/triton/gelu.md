# 🚀 GELU Triton Kernel

> **Overview**: This tutorial shows the compilation of GELU (Gaussian Error Linear Units) Triton Kernel. We will skip
some of the illustrations that are already emphasized in [vector-addition](vector_add.md) example.

## Table of Contents

- [🔄 Compilation Pipeline Overview](#-compilation-pipeline-overview)
  - [🔧 Compilation Stages](#-compilation-stages)
- [🧮 GELU Triton Kernel](#-gelu-triton-kernel)
  - [📝 Python Implementation](#-python-implementation)
  - [⚙️ Kernel Configuration](#️-kernel-configuration)
- [🏃 Compilation and Execution](#-compilation-and-execution)
  - [🔨 Build Commands](#-build-commands)
  - [✅ Expected Output](#-expected-output)
- [🔍 Deep Dive into Lowering](#-deep-dive-into-lowering)
  - [Triton IR](#triton-ir)
  - [Triton-IR to Linalg](#triton-ir-to-linalg)
  - [Fusion](#fusion)
  - [Tiled, Multi-threaded, Vectorized LLVM-IR](#tiled-multi-threaded-vectorized-llvm-ir)

---

## 🔄 Compilation Pipeline Overview

The Python Triton kernel undergoes a multi-stage compilation process.

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

## 🧮 GELU Triton Kernel

### 📝 Python Implementation

```python
import pytest

import torch
import triton
import triton.language as tl
from triton.backends.qcom_hexagon_backend.driver import HexagonDriver
from .. import parameterize_func_name

triton.runtime.driver.set_active(HexagonDriver())

# 1e-2 is the minimum ATOL where the fp16 test succeeds
ATOL_FP32 = 1e-5
ATOL_FP16 = 1e-2

@triton.jit
def tanh(x):
    result = tl.exp((-2 * x).to(tl.float32)).to(x.dtype)
    result = (1 - result) / (1 + result)
    return result

@pytest.mark.parametrize("num_elements", [16384])
@pytest.mark.parametrize("num_threads", [1])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("enable_mt", [True])
def test_gelu(num_elements, num_threads, dtype, enable_mt):
    print(f"Running with num-elements: {num_elements}, num-triton-thread: {num_threads}, "
          f"dtype: {dtype}, enable-multi-threading: {enable_mt}")
    print('-' * 100)

    @triton.jit
    @parameterize_func_name(dtype)
    def gelu_kernel(
        x_ptr,
        output_ptr,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offsets = (pid * BLOCK_SIZE) + tl.arange(0, BLOCK_SIZE)
        x = tl.load(x_ptr + offsets)
        output = 0.5 * x * (1 + tanh(0.797885 * x + 0.035677 * x * x * x))
        tl.store(output_ptr + offsets, output)

    x = torch.rand(num_elements, dtype=dtype)
    output = torch.empty_like(x, dtype=dtype)

    block_size = num_elements // num_threads
    true_if_single_threaded = num_threads == 1
    grid = (num_threads,)
    gelu_kernel[grid](x, output,
                      BLOCK_SIZE=block_size,
                      enableMultiThreading=true_if_single_threaded and enable_mt,
                      enableVTCMTiling=True,
                      enableConvertToHexagonmem=True,
                      enableHexagonmemCopyToDMA=True)

    reference = torch.nn.functional.gelu(x, approximate="tanh")
    assert(torch.allclose(output, reference, atol=ATOL_FP16 if dtype == torch.float16 else ATOL_FP32))

    reference = torch.nn.functional.gelu(x, approximate="tanh") 
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
# Enable MLIR IR and LLVM-IR dumping to see intermediate representations
TRITON_ALWAYS_COMPILE=1 MLIR_ENABLE_DUMP=1 LLVM_IR_ENABLE_DUMP=1 pytest -sv test/python/triton/test_vec_add.py
```

### ✅ Expected Output

If setup and device acquisition work correctly, you should see:

```bash
Test_Info: {
        Name:gelu_kernel
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
  tt.func public @gelu_kernel(%x_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                              %output_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}) 
                              attributes {noinline = false} {
    %result = arith.constant dense<-2.000000e+00> : tensor<16384xf32>
    %cst = arith.constant dense<1.000000e+00> : tensor<16384xf32>  
    %output = arith.constant dense<3.567700e-02> : tensor<16384xf32>  
    %output_0 = arith.constant dense<7.978850e-01> : tensor<16384xf32>  
    %output_1 = arith.constant dense<5.000000e-01> : tensor<16384xf32>  
    %c16384_i32 = arith.constant 16384 : i32  
    %pid = tt.get_program_id x : i32  
    %offsets = arith.muli %pid, %c16384_i32 : i32  
    %offsets_2 = tt.make_range {end = 16384 : i32, start = 0 : i32} : tensor<16384xi32>  
    %offsets_3 = tt.splat %offsets : i32 -> tensor<16384xi32>  
    %offsets_4 = arith.addi %offsets_3, %offsets_2 : tensor<16384xi32>  
    %x = tt.splat %x_ptr : !tt.ptr<f32> -> tensor<16384x!tt.ptr<f32>>  
    %x_5 = tt.addptr %x, %offsets_4 : tensor<16384x!tt.ptr<f32>>, tensor<16384xi32>  
    %x_6 = tt.load %x_5 : tensor<16384x!tt.ptr<f32>>  
    %output_7 = arith.mulf %x_6, %output_1 : tensor<16384xf32>  
    %output_8 = arith.mulf %x_6, %output_0 : tensor<16384xf32>  
    %output_9 = arith.mulf %x_6, %output : tensor<16384xf32>  
    %output_10 = arith.mulf %output_9, %x_6 : tensor<16384xf32>  
    %output_11 = arith.mulf %output_10, %x_6 : tensor<16384xf32>  
    %output_12 = arith.addf %output_8, %output_11 : tensor<16384xf32>  
    %result_13 = arith.mulf %output_12, %result : tensor<16384xf32>  
    %result_14 = math.exp %result_13 : tensor<16384xf32>  
    %result_15 = arith.subf %cst, %result_14 : tensor<16384xf32>  
    %result_16 = arith.addf %result_14, %cst : tensor<16384xf32>  
    %result_17 = arith.divf %result_15, %result_16 : tensor<16384xf32>  
    %output_18 = arith.addf %result_17, %cst : tensor<16384xf32>  
    %output_19 = arith.mulf %output_7, %output_18 : tensor<16384xf32>  
    %0 = tt.splat %output_ptr : !tt.ptr<f32> -> tensor<16384x!tt.ptr<f32>>  
    %1 = tt.addptr %0, %offsets_4 : tensor<16384x!tt.ptr<f32>>, tensor<16384xi32>  
    tt.store %1, %output_19 : tensor<16384x!tt.ptr<f32>>  
    tt.return  
  }  
}  
```

### Triton-IR to Linalg

**Stage 2**: After triton-to-linalg lowering using triton-shared

```mlir
#map = affine_map<(d0) -> (d0)>
module {
  func.func @gelu_kernel(%arg0: memref<*xf32> {tt.divisibility = 16 : i32},
                         %arg1: memref<*xf32> {tt.divisibility = 16 : i32},
                         %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32) {
    %cst = arith.constant -2.000000e+00 : f32
    %cst_0 = arith.constant 1.000000e+00 : f32
    %cst_1 = arith.constant 3.567700e-02 : f32
    %cst_2 = arith.constant 7.978850e-01 : f32
    %cst_3 = arith.constant 5.000000e-01 : f32
    %c16384_i32 = arith.constant 16384 : i32
    %0 = tensor.empty() : tensor<16384xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<16384xf32>) -> tensor<16384xf32>
    %2 = linalg.fill ins(%cst_0 : f32) outs(%0 : tensor<16384xf32>) -> tensor<16384xf32>
    %3 = linalg.fill ins(%cst_1 : f32) outs(%0 : tensor<16384xf32>) -> tensor<16384xf32>
    %4 = linalg.fill ins(%cst_2 : f32) outs(%0 : tensor<16384xf32>) -> tensor<16384xf32>
    %5 = linalg.fill ins(%cst_3 : f32) outs(%0 : tensor<16384xf32>) -> tensor<16384xf32>
    %6 = arith.muli %arg5, %c16384_i32 : i32
    %7 = arith.index_cast %6 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%7], sizes: [16384], strides: [1] : memref<*xf32> to memref<16384xf32, strided<[1], offset: ?>>
    %alloc = memref.alloc() : memref<16384xf32>
    memref.copy %reinterpret_cast, %alloc : memref<16384xf32, strided<[1], offset: ?>> to memref<16384xf32>
    %8 = bufferization.to_tensor %alloc restrict writable : memref<16384xf32> to tensor<16384xf32>
    %9 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%8, %5 : tensor<16384xf32>, tensor<16384xf32>) outs(%8 : tensor<16384xf32>) {
    ^bb0(%in: f32, %in_5: f32, %out: f32):
      %22 = arith.mulf %in, %in_5 : f32
      linalg.yield %22 : f32
    } -> tensor<16384xf32>
    %10 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%8, %4 : tensor<16384xf32>, tensor<16384xf32>) outs(%8 : tensor<16384xf32>) {
    ^bb0(%in: f32, %in_5: f32, %out: f32):
      %22 = arith.mulf %in, %in_5 : f32
      linalg.yield %22 : f32
    } -> tensor<16384xf32>
    %11 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%8, %3 : tensor<16384xf32>, tensor<16384xf32>) outs(%8 : tensor<16384xf32>) {
    ^bb0(%in: f32, %in_5: f32, %out: f32):
      %22 = arith.mulf %in, %in_5 : f32
      linalg.yield %22 : f32
    } -> tensor<16384xf32>
    %12 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%11, %8 : tensor<16384xf32>, tensor<16384xf32>) outs(%11 : tensor<16384xf32>) {
    ^bb0(%in: f32, %in_5: f32, %out: f32):
      %22 = arith.mulf %in, %in_5 : f32
      linalg.yield %22 : f32
    } -> tensor<16384xf32>
    %13 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%12, %8 : tensor<16384xf32>, tensor<16384xf32>) outs(%12 : tensor<16384xf32>) {
    ^bb0(%in: f32, %in_5: f32, %out: f32):
      %22 = arith.mulf %in, %in_5 : f32
      linalg.yield %22 : f32
    } -> tensor<16384xf32>
    %14 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%10, %13 : tensor<16384xf32>, tensor<16384xf32>) outs(%10 : tensor<16384xf32>) {
    ^bb0(%in: f32, %in_5: f32, %out: f32):
      %22 = arith.addf %in, %in_5 : f32
      linalg.yield %22 : f32
    } -> tensor<16384xf32>
    %15 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%14, %1 : tensor<16384xf32>, tensor<16384xf32>) outs(%14 : tensor<16384xf32>) {
    ^bb0(%in: f32, %in_5: f32, %out: f32):
      %22 = arith.mulf %in, %in_5 : f32
      linalg.yield %22 : f32
    } -> tensor<16384xf32>
    %16 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%15 : tensor<16384xf32>) outs(%15 : tensor<16384xf32>) {
    ^bb0(%in: f32, %out: f32):
      %22 = math.exp %in : f32
      linalg.yield %22 : f32
    } -> tensor<16384xf32>
    %17 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%2, %16 : tensor<16384xf32>, tensor<16384xf32>) outs(%2 : tensor<16384xf32>) {
    ^bb0(%in: f32, %in_5: f32, %out: f32):
      %22 = arith.subf %in, %in_5 : f32
      linalg.yield %22 : f32
    } -> tensor<16384xf32>
        %18 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%16, %2 : tensor<16384xf32>, tensor<16384xf32>) outs(%16 : tensor<16384xf32>) {
    ^bb0(%in: f32, %in_5: f32, %out: f32):
      %22 = arith.addf %in, %in_5 : f32
      linalg.yield %22 : f32
    } -> tensor<16384xf32>
    %19 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%17, %18 : tensor<16384xf32>, tensor<16384xf32>) outs(%17 : tensor<16384xf32>) {
    ^bb0(%in: f32, %in_5: f32, %out: f32):
      %22 = arith.divf %in, %in_5 : f32
      linalg.yield %22 : f32
    } -> tensor<16384xf32>
    %20 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%19, %2 : tensor<16384xf32>, tensor<16384xf32>) outs(%19 : tensor<16384xf32>) {
    ^bb0(%in: f32, %in_5: f32, %out: f32):
      %22 = arith.addf %in, %in_5 : f32
      linalg.yield %22 : f32
    } -> tensor<16384xf32>
    %21 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%9, %20 : tensor<16384xf32>, tensor<16384xf32>) outs(%9 : tensor<16384xf32>) {
    ^bb0(%in: f32, %in_5: f32, %out: f32):
      %22 = arith.mulf %in, %in_5 : f32
      linalg.yield %22 : f32
    } -> tensor<16384xf32>
    %reinterpret_cast_4 = memref.reinterpret_cast %arg1 to offset: [%7], sizes: [16384], strides: [1] : memref<*xf32> to memref<16384xf32, strided<[1], offset: ?>>
    bufferization.materialize_in_destination %21 in writable %reinterpret_cast_4 : (tensor<16384xf32>, memref<16384xf32, strided<[1], offset: ?>>) -> ()
    return
  }
}
```

### Fusion

> **🎯 Optimization Goal**: The input IR is series of linalg-generics but we can fuse them into a single op. 

```mlir
func.func @gelu_kernel(%arg0: memref<*xf32> {tt.divisibility = 16 : i32}, 
                       %arg1: memref<*xf32> {tt.divisibility = 16 : i32},
                       %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32) {
  %cst = arith.constant -2.000000e+00 : f32
  %cst_0 = arith.constant 1.000000e+00 : f32
  %cst_1 = arith.constant 3.567700e-02 : f32
  %cst_2 = arith.constant 7.978850e-01 : f32
  %cst_3 = arith.constant 5.000000e-01 : f32
  %c16384_i32 = arith.constant 16384 : i32
  %0 = tensor.empty() : tensor<16384xf32>
  %1 = arith.muli %arg5, %c16384_i32 : i32
  %2 = arith.index_cast %1 : i32 to index
  %reinterpret_cast = memref.reinterpret_cast %arg1 to offset: [%2], sizes: [16384], strides: [1] : memref<*xf32> to memref<16384xf32, strided<[1], offset: ?>>
  %reinterpret_cast_4 = memref.reinterpret_cast %arg0 to offset: [%2], sizes: [16384], strides: [1] : memref<*xf32> to memref<16384xf32, strided<[1], offset: ?>>
  %3 = bufferization.to_tensor %reinterpret_cast_4 restrict : memref<16384xf32, strided<[1], offset: ?>> to tensor<16384xf32>
  %4 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%3 : tensor<16384xf32>) outs(%0 : tensor<16384xf32>) {
  ^bb0(%in: f32, %out: f32):
    %5 = arith.mulf %in, %cst_1 : f32
    %6 = arith.mulf %5, %in : f32
    %7 = arith.mulf %6, %in : f32
    %8 = arith.mulf %in, %cst_2 : f32
    %9 = arith.addf %8, %7 : f32
    %10 = arith.mulf %9, %cst : f32
    %11 = math.exp %10 : f32
    %12 = arith.addf %11, %cst_0 : f32
    %13 = arith.subf %cst_0, %11 : f32
    %14 = arith.divf %13, %12 : f32
    %15 = arith.addf %14, %cst_0 : f32
    %16 = arith.mulf %in, %cst_3 : f32
    %17 = arith.mulf %16, %15 : f32
    linalg.yield %17 : f32
  } -> tensor<16384xf32>
  bufferization.materialize_in_destination %4 in restrict writable %reinterpret_cast : (tensor<16384xf32>, memref<16384xf32, strided<[1], offset: ?>>) -> ()
  return
}
```

### Tiled, Multi-threaded, Vectorized LLVM-IR

We skip the things we already talked about in the vector-add tutorial, and focus instead on LLVM-IR
level calls, which may be interesting.


#### 🎯 Key Optimizations Observed:

- **🧵 Multi-threading**: Uses `async-execute` (lowers to LLVM coroutines).
  - Breaks 16384 elements into **4 HVX threads** × 4096 elements each.
- **🔲 Tiling**: In this example the tensors fit entirely in TCM, so are brought in full before computation.
- **📡 DMA Transfers**: Efficient DDR ↔ TCM data movement.
- **⚡ Vectorization**: Main computation vectorized for HVX .

Below we only show snippets that highlight DMA calls, multi-threading calls, and custom math library function calls.
```mlir
  ...
  %25 = llvm.call @hexagon_runtime_alloc_1d(%24, %4, %13) : (i32, i64, i1) -> !llvm.ptr
  %26 = llvm.getelementptr %6[1] : (!llvm.ptr) -> !llvm.ptr, i32
  %27 = llvm.ptrtoint %26 : !llvm.ptr to i64
  ...
  %44 = llvm.call @mlirAsyncRuntimeCreateGroup(%12) : (i64) -> !llvm.ptr
  ...
  %59 = llvm.call @hexagon_runtime_dma_start(%25, %1, %57, %2, %30, %2, %2, %58) : (!llvm.ptr, i32, !llvm.ptr, i32, i32, i32, i32, !llvm.ptr) -> i32
  ...
  .. = llvm.intr.coro.begin %17, %24 : (!llvm.token, !llvm.ptr) -> !llvm.ptr
  .. = llvm.intr.coro.save %25 : (!llvm.ptr) -> !llvm.token
  ...
  %48 = llvm.call @_hexagon_runtime_exp__vf(%47) : (vector<32xi32>) -> vector<32xi32>.
  ...
   ^bb4:  // pred: ^bb2
    llvm.call @mlirAsyncRuntimeEmplaceToken(%16) : (!llvm.ptr) -> ()
    llvm.br ^bb5
  ^bb5:  // 2 preds: ^bb0, ^bb4
    %64 = llvm.intr.coro.free %17, %25 : (!llvm.token, !llvm.ptr) -> !llvm.ptr
  ...
    llvm.br ^bb1(%9 : i64)
    %31 = llvm.call @hexagon_runtime_dma_start(%23, %2, %25, %1, %30, %2, %2, %28) : (!llvm.ptr, i32, !llvm.ptr, i32, i32, i32, i32, !llvm.ptr) -> i32
    llvm.store %31, %28 : i32, !llvm.ptr
    %32 = llvm.load %28 : !llvm.ptr -> i32
   ...
  ```
All the above IR and files can be found from debug prints and inspected.


> **🎉 Congratulations!** You've successfully walked through the complete compilation pipeline from high-level Python Triton code to optimized Hexagon assembly, witnessing the sophisticated optimizations that make efficient ML computation possible on Hexagon architecture.
