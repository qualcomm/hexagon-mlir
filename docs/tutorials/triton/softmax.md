# 🚀 Softmax Triton Kernel

> **Overview**: This tutorial shows the compilation of Softmax Triton Kernel. We will skip
some of the steps that are already emphasized in previous examples e.g. [vector-addition](vector_add.md).

## Table of Contents

- [🔄 Compilation Pipeline Overview](#-compilation-pipeline-overview)
  - [🔧 Compilation Stages](#-compilation-stages)
- [🧮 Softmax Triton Kernel](#-softmax-triton-kernel)
- [🏃 Compilation and Execution](#-compilation-and-execution)
  - [🔨 Build Commands](#-build-commands)
  - [✅ Expected Output](#-expected-output)
- [🔍 Deep Dive into Lowering](#-deep-dive-into-lowering)
  - [Lower IR](#lower-ir)
  - [Lowered to LLVM](#lowered-to-llvm)

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
| **5** | Linker | Create `libsoftmax_kernel.so` |
| **6** | Runtime | Load and execute on device |

---

## 🧮 Softmax Triton Kernel

This example is originally from open-source [Triton Tutorial](https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html).
We want to bring to attention some changes we made to the kernel for our target and compiler:
- Triton uses e.g. `tl.program_id(0)`. We dont need it currently for single NPU. Instead in the example below the compiler does the HVX multi-threading over each row of the softmax. We enable by setting `enableSCFThreading`. We are working on auto-detection by the compiler but currently the user needs to specify it with this option that the loop over the core of the kernel is parallel.
- We have two style of multi-threading. The one discussed in the the vector-addition example auto-detects parallelism over fused linalg-generics. In kernels where there is no over-arching `for` loop, or one that is not parallel, the `enableMultithreading` option is the right one to use. It is much more generic (works on fused linalg generic) and is totally compiler auto-detection and validation.

```python
import pytest
import torch
import triton
import triton.language as tl
from triton.backends.qcom_hexagon_backend.driver import HexagonDriver
from .. import parameterize_func_name

triton.runtime.driver.set_active(HexagonDriver())

NUM_PROGRAMS = 1
NUM_ROWS = 4 
NUM_COLS = 65536 
BLOCK_SIZE = triton.next_power_of_2(NUM_COLS)
ATOL_FP32 = 1e-5
ATOL_FP16 = 1e-3

@triton.jit
def _max(a,b):
    return tl.maximum(a, b)

@pytest.mark.parametrize("dtype", [torch.float16])
def test_softmax(dtype):

    @triton.jit
    @parameterize_func_name(dtype)
    def softmax_kernel(
        x_ptr,
        output_ptr,
        NUM_ROWS: tl.constexpr,
        NUM_COLS: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):

        n_elements = NUM_ROWS * NUM_COLS
        col_offsets = tl.arange(0, BLOCK_SIZE)
        for start_n in range(0, n_elements, NUM_COLS):
            offsets = start_n + col_offsets
            x = tl.load(x_ptr + offsets, mask=col_offsets < NUM_COLS, other=-float("inf"))
            z = x - tl.reduce(x, 0,_max)
            num = tl.exp(z.to(tl.float32)).to(x.dtype)
            denom = tl.sum(num, axis=0)
            y = num / denom
            tl.store(output_ptr + offsets, y, mask=col_offsets < NUM_COLS)

    x = torch.rand(NUM_ROWS, NUM_COLS, dtype=dtype)
    output = torch.empty_like(x, dtype=dtype)

    grid = (NUM_PROGRAMS,)
    softmax_kernel[grid](x, output,
                         NUM_ROWS=NUM_ROWS,
                         NUM_COLS=NUM_COLS,
                         BLOCK_SIZE=BLOCK_SIZE,
                         enableSCFThreading=True,
                         enableMultiThreading=False,
                         enableVTCMTiling=False,
                         enableConvertToHexagonmem=False,
                         enableHexagonmemCopyToDMA=False)

    reference = torch.nn.functional.softmax(x, dim=1, dtype=dtype)
    assert torch.allclose(
        output, reference, atol=ATOL_FP16 if dtype == torch.float16 else ATOL_FP32
    )
```    

---

## 🏃 Compilation and Execution

### 🔨 Build Commands

```bash
# Enable MLIR IR and LLVM-IR dumping to see intermediate representations
TRITON_ALWAYS_COMPILE=1 MLIR_ENABLE_DUMP=1 LLVM_IR_ENABLE_DUMP=1 pytest -sv test/python/triton/test_softmax_scf_multithread.py
```

### ✅ Expected Output

If setup and device acquisition work correctly, you should see:

```bash
Test_Info: {
        Name:softmax_kernel
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

### Lower IR

After some initial lowering at point where we have multi-threaded the softmax kernel - 

```mlir
// -----// IR Dump After FormAsyncThreads (form-async-threads) //----- //
func.func @softmax_kernel(%arg0: memref<*xf16> {tt.divisibility = 16 : i32}, %arg1: memref<*xf16> {tt.divisibility = 16 : i32},
                         %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32) {
  %cst = arith.constant 0.000000e+00 : f16
  %c65536 = arith.constant 65536 : index
  %c0 = arith.constant 0 : index
  %c64 = arith.constant 64 : index
  %cst_0 = arith.constant 0xFC00 : f16
  %cst_1 = arith.constant 0.000000e+00 : f32
  %c0_2 = arith.constant 0 : index
  %c262144 = arith.constant 262144 : index
  %c65536_3 = arith.constant 65536 : index
  %0 = arith.divui %c262144, %c65536_3 : index
  %1 = async.create_group %0 : !async.group
  scf.for %arg8 = %c0_2 to %c262144 step %c65536_3 {
    %token = async.execute {
      %reinterpret_cast = memref.reinterpret_cast %arg1 to offset: [%arg8], sizes: [65536], strides: [1]
       : memref<*xf16> to memref<65536xf16, strided<[1], offset: ?>>
      %reinterpret_cast_4 = memref.reinterpret_cast %arg0 to offset: [%arg8], sizes: [65536], strides: [1]
       : memref<*xf16> to memref<65536xf16, strided<[1], offset: ?>>
      %alloc = memref.alloc() {alignment = 64 : i64} : memref<65536xf16>
      %alloc_5 = memref.alloc() {alignment = 64 : i64} : memref<f16>
      memref.store %cst_0, %alloc_5[] : memref<f16>
      scf.for %arg9 = %c0 to %c65536 step %c64 {
        %subview = memref.subview %reinterpret_cast_4[%arg9] [64] [1]
         : memref<65536xf16, strided<[1], offset: ?>> to memref<64xf16, strided<[1], offset: ?>>
        %5 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]}
         : memref<64xf16, strided<[1], offset: ?>>, vector<64xf16>
        %6 = memref.load %alloc_5[] : memref<f16>
        %7 = vector.reduction <maxnumf>, %5, %6 fastmath<fast> : vector<64xf16> into f16
        memref.store %7, %alloc_5[] : memref<f16>
      }
      %3 = memref.load %alloc_5[] : memref<f16>
      scf.for %arg9 = %c0 to %c65536 step %c64 {
        %subview = memref.subview %reinterpret_cast_4[%arg9] [64] [1]
         : memref<65536xf16, strided<[1], offset: ?>> to memref<64xf16, strided<[1], offset: ?>>
        %subview_7 = memref.subview %alloc[%arg9] [64] [1]
         : memref<65536xf16> to memref<64xf16, strided<[1], offset: ?>>
        %5 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]}
         : memref<64xf16, strided<[1], offset: ?>>, vector<64xf16>
        %6 = vector.broadcast %3 : f16 to vector<64xf16>
        %7 = arith.subf %5, %6 fastmath<fast> : vector<64xf16>
        %8 = arith.extf %7 fastmath<fast> : vector<64xf16> to vector<64xf32>
        %9 = math.exp %8 fastmath<fast> : vector<64xf32>
        %10 = arith.truncf %9 fastmath<fast> : vector<64xf32> to vector<64xf16>
        vector.transfer_write %10, %subview_7[%c0] {in_bounds = [true]}
          : vector<64xf16>, memref<64xf16, strided<[1], offset: ?>>
      }
      %alloc_6 = memref.alloc() {alignment = 64 : i64} : memref<f32>
      memref.store %cst_1, %alloc_6[] : memref<f32>
      scf.for %arg9 = %c0 to %c65536 step %c64 {
        %subview = memref.subview %alloc[%arg9] [64] [1] : memref<65536xf16> to memref<64xf16, strided<[1], offset: ?>>
        %5 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]}
         : memref<64xf16, strided<[1], offset: ?>>, vector<64xf16>
        %6 = memref.load %alloc_6[] : memref<f32>
        %7 = arith.extf %5 fastmath<fast> : vector<64xf16> to vector<64xf32>
        %8 = vector.reduction <add>, %7, %6 fastmath<fast> : vector<64xf32> into f32
        memref.store %8, %alloc_6[] : memref<f32>
      }
      %4 = memref.load %alloc_6[] : memref<f32>
      scf.for %arg9 = %c0 to %c65536 step %c64 {
        %subview = memref.subview %alloc[%arg9] [64] [1]
         : memref<65536xf16> to memref<64xf16, strided<[1], offset: ?>>
        %subview_7 = memref.subview %reinterpret_cast[%arg9] [64] [1]
         : memref<65536xf16, strided<[1], offset: ?>> to memref<64xf16, strided<[1], offset: ?>>
        %5 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]}
         : memref<64xf16, strided<[1], offset: ?>>, vector<64xf16>
        %6 = arith.extf %5 fastmath<fast> : vector<64xf16> to vector<64xf32>
        %7 = vector.broadcast %4 : f32 to vector<64xf32>
        %8 = arith.divf %6, %7 fastmath<fast> : vector<64xf32>
        %9 = arith.truncf %8 fastmath<fast> : vector<64xf32> to vector<64xf16>
        vector.transfer_write %9, %subview_7[%c0] {in_bounds = [true]}
         : vector<64xf16>, memref<64xf16, strided<[1], offset: ?>>
      }
      memref.dealloc %alloc : memref<65536xf16>
      memref.dealloc %alloc_5 : memref<f16>
      memref.dealloc %alloc_6 : memref<f32>
      async.yield
    }
    %2 = async.add_to_group %token, %1 : !async.token
  }
  async.await_all %1
  return
}
```

### Lowered to LLVM

Below we just show the core part of the softmax kernel when lowered to MLIR LLVM-IR. We have vector code for HVX and for math we are calling custom ` @_hexagon_runtime_exp__vf` implementations.

```mlir
    ...
    %69 = llvm.fsub %66, %68 {fastmathFlags = #llvm.fastmath<fast>} : vector<64xf16>
    %70 = llvm.fpext %69 {fastmath = #arith.fastmath<fast>} : vector<64xf16> to vector<64xf32>
    %71 = llvm.intr.vector.extract %70[0] : vector<32xf32> from vector<64xf32>
    %72 = llvm.bitcast %71 : vector<32xf32> to vector<32xi32>
    %73 = llvm.call @_hexagon_runtime_exp__vf(%72) : (vector<32xi32>) -> vector<32xi32>
    %74 = llvm.bitcast %73 : vector<32xi32> to vector<32xf32>
    %75 = llvm.intr.vector.insert %74, %0[0] : vector<32xf32> into vector<64xf32>
    %76 = llvm.intr.vector.extract %70[32] : vector<32xf32> from vector<64xf32>
    %77 = llvm.bitcast %76 : vector<32xf32> to vector<32xi32>
    %78 = llvm.call @_hexagon_runtime_exp__vf(%77) : (vector<32xi32>) -> vector<32xi32>
    %79 = llvm.bitcast %78 : vector<32xi32> to vector<32xf32>
    %80 = llvm.intr.vector.insert %79, %75[32] : vector<32xf32> into vector<64xf32>
    %81 = llvm.fptrunc %80 {fastmath = #arith.fastmath<fast>} : vector<64xf32> to vector<64xf16>
    %82 = llvm.getelementptr %42[%62] : (!llvm.ptr, i64) -> !llvm.ptr, f16
    llvm.store %81, %82 {alignment = 2 : i64} : vector<64xf16>, !llvm.ptr
    ...
```
All the above IR and files can be found from debug prints and inspected.


> **🎉 Congratulations!** You've successfully completed the tutorial.
