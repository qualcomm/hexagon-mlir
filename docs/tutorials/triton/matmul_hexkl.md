# 🚀 Matmul HexKL Triton Kernel

> **Overview**: This tutorial demonstrates compilation of the Matmul Triton Kernel using the Hexagon Kernel Library (HexKL).
It assumes that HexKL is already set up. Please note that only a limited set of HexKL features are currently supported, and its use is experimental. The code shown below can be found in our repository at : `test/python/triton/test_matmul_hexkl.py`.

We will skip some of the steps that are already emphasized in previous examples e.g. [vector-addition](vector_add.md).

## Table of Contents

- [🔄 Compilation Pipeline Overview](#-compilation-pipeline-overview)
  - [🔧 Compilation Stages](#-compilation-stages)
- [🧮 Matmul HexKL Triton Kernel](#-matmul-hexkl-triton-kernel)
  - [📝 Python Implementation](#-python-implementation)
- [🏃 Compilation and Execution](#-compilation-and-execution)
  - [🔨 Build Commands](#-build-commands)
  - [✅ Expected Output](#-expected-output)
- [🔍 Deep Dive into Lowering](#-deep-dive-into-lowering)
  - [Triton IR](#triton-ir)
  - [Triton-IR to Linalg](#triton-ir-to-linalg)
  - [Linalg Bufferized IR](#linalg-bufferized-ir)
  - [HexKL Dialect IR](#hexkl-dialect-ir)
  - [HexKL Runtime IR](#hexkl-runtime-ir)
  - [LLVM IR](#llvm-ir)
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
| **5** | Linker | Create `libmatmul_hexkl.so` |
| **6** | Runtime | Load and execute on device |

---

## 🧮 Matmul HexKL Triton Kernel

This example demonstrates matrix multiplication on FP16 inputs and producing an FP32 output. The `enable_hexkl` compiler flag is set to direct the lowering pipeline to use HexKL library calls for matmul, bypassing the default HVX-based tiling and vectorization workflow.

### 📝 Python Implementation

```python
import torch
import triton
import triton.language as tl
from triton.backends.qcom_hexagon_backend.driver import HexagonDriver

triton.runtime.driver.set_active(HexagonDriver())

N_ROWS = 1024
N_COLUMNS = 512
N_INNER = 64

@triton.jit
def matmul_hexkl(
    A,
    B,
    C,
    N_ROWS: tl.constexpr,
    N_COLUMNS: tl.constexpr,
    N_INNER: tl.constexpr,
):
    A_block_ptr = tl.make_block_ptr(
        base=A,
        shape=(N_ROWS, N_INNER),
        strides=(N_INNER, 1),
        offsets=(0, 0),
        block_shape=(N_ROWS, N_INNER),
        order=(1, 0),
    )
    B_block_ptr = tl.make_block_ptr(
        base=B,
        shape=(N_INNER, N_COLUMNS),
        strides=(N_COLUMNS, 1),
        offsets=(0, 0),
        block_shape=(N_INNER, N_COLUMNS),
        order=(1, 0),
    )
    C_block_ptr = tl.make_block_ptr(
        base=C,
        shape=(N_ROWS, N_COLUMNS),
        strides=(N_COLUMNS, 1),
        offsets=(0, 0),
        block_shape=(N_ROWS, N_COLUMNS),
        order=(1, 0),
    )
    q = tl.load(A_block_ptr)
    k_t = tl.load(B_block_ptr)
    qk = tl.dot(q, k_t, out_dtype=tl.float32)
    tl.store(C_block_ptr, qk.to(C.type.element_ty))


def test_matmul_hexkl():
    mat_A = torch.rand((N_ROWS, N_INNER), dtype=torch.float16)
    mat_B = torch.rand((N_INNER, N_COLUMNS), dtype=torch.float16)
    mat_C = torch.zeros((N_ROWS, N_COLUMNS), dtype=torch.float32)

    matmul_hexkl[(1,)](mat_A, mat_B, mat_C, N_ROWS, N_COLUMNS, N_INNER,
                       enableHexKL=True,
                       iterations=1)

    # Perform the matrix multiplication
    reference = torch.matmul(mat_A, mat_B).to(torch.float32)
    assert torch.allclose(mat_C, reference, rtol=1e-02)
```
---

## 🏃 Compilation and Execution

### 🔨 Build Commands

```bash
# Enable MLIR IR and LLVM-IR dumping to see intermediate representations in the log file.
TRITON_ALWAYS_COMPILE=1 MLIR_ENABLE_DUMP=1 LLVM_IR_ENABLE_DUMP=1 pytest -sv test/python/triton/test_matmul_hexkl.py |& tee matmul_hexkl.log
```

### ✅ Expected Output

If the test passes, the following prompt would confirm that the kernel executed correctly on the device and the results match the Torch reference, indicating no accuracy issues.

```bash
Test_Info: {
        Name:matmul_hexkl
        Result:Pass
        Perf: xxxx
        Units:us
}
==> Ran successfully, collecting results
PASSED                                  
```

---

## 🔍 Deep Dive into Lowering

This section explores the IR at different compilation stages.

### Triton IR

**Stage 1**: In the initial Triton IR, `tl.dot` which is the triton language construct used to express matrix multiplication is
lowered to `tt.dot` operating on FP16 input tensors and producing an FP32 output tensor.
 
```mlir
module {
  tt.func public @matmul_hexkl(%A: !tt.ptr<f16> {tt.divisibility = 16 : i32} , %B: !tt.ptr<f16> {tt.divisibility = 16 : i32} , %C: !tt.ptr<f32> {tt.divisibility = 16 : i32} ) attributes {noinline = false} {
    %A_block_ptr = arith.constant 1024 : i64 
    %A_block_ptr_0 = arith.constant 64 : i64 
    %A_block_ptr_1 = arith.constant 64 : i64 
    %A_block_ptr_2 = arith.constant 1 : i64 
    %A_block_ptr_3 = arith.constant 0 : i32 
    %A_block_ptr_4 = arith.constant 0 : i32 
    %A_block_ptr_5 = tt.make_tensor_ptr %A, [%A_block_ptr, %A_block_ptr_0], [%A_block_ptr_1, %A_block_ptr_2], [%A_block_ptr_3, %A_block_ptr_4] {order = array<i32: 1, 0>} : <tensor<1024x64xf16>> 
    %B_block_ptr = arith.constant 64 : i64 
    %B_block_ptr_6 = arith.constant 512 : i64 
    %B_block_ptr_7 = arith.constant 512 : i64 
    %B_block_ptr_8 = arith.constant 1 : i64 
    %B_block_ptr_9 = arith.constant 0 : i32 
    %B_block_ptr_10 = arith.constant 0 : i32 
    %B_block_ptr_11 = tt.make_tensor_ptr %B, [%B_block_ptr, %B_block_ptr_6], [%B_block_ptr_7, %B_block_ptr_8], [%B_block_ptr_9, %B_block_ptr_10] {order = array<i32: 1, 0>} : <tensor<64x512xf16>> 
    %C_block_ptr = arith.constant 1024 : i64 
    %C_block_ptr_12 = arith.constant 512 : i64 
    %C_block_ptr_13 = arith.constant 512 : i64 
    %C_block_ptr_14 = arith.constant 1 : i64 
    %C_block_ptr_15 = arith.constant 0 : i32 
    %C_block_ptr_16 = arith.constant 0 : i32 
    %C_block_ptr_17 = tt.make_tensor_ptr %C, [%C_block_ptr, %C_block_ptr_12], [%C_block_ptr_13, %C_block_ptr_14], [%C_block_ptr_15, %C_block_ptr_16] {order = array<i32: 1, 0>} : <tensor<1024x512xf32>> 
    %q = tt.load %A_block_ptr_5 : !tt.ptr<tensor<1024x64xf16>> 
    %k_t = tt.load %B_block_ptr_11 : !tt.ptr<tensor<64x512xf16>> 
    %qk = arith.constant 0.000000e+00 : f32 
    %qk_18 = arith.constant dense<0.000000e+00> : tensor<1024x512xf32> 
    %qk_19 = tt.dot %q, %k_t, %qk_18 : tensor<1024x64xf16> * tensor<64x512xf16> -> tensor<1024x512xf32> 
    tt.store %C_block_ptr_17, %qk_19 : !tt.ptr<tensor<1024x512xf32>> 
    tt.return 
  } 
} 
```

### Triton-IR to Linalg

**Stage 2**: After triton-to-linalg lowering using triton-shared, and some initial hexagon-mlir passes, we have a single `linalg.matmul` that's lowered from the `tt.dot` op.

```mlir
module attributes {llvm.target_triple = "hexagon"} {
  func.func @matmul_hexkl(%arg0: memref<*xf16> {tt.divisibility = 16 : i32}, %arg1: memref<*xf16> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32) {
    %cst = arith.constant 0.000000e+00 : f32
    %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [0], sizes: [1024, 64], strides: [64, 1] : memref<*xf16> to memref<1024x64xf16, strided<[64, 1]>>
    %0 = bufferization.to_tensor %reinterpret_cast restrict : memref<1024x64xf16, strided<[64, 1]>> to tensor<1024x64xf16>
    %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [0], sizes: [64, 512], strides: [512, 1] : memref<*xf16> to memref<64x512xf16, strided<[512, 1]>>
    %1 = bufferization.to_tensor %reinterpret_cast_0 restrict : memref<64x512xf16, strided<[512, 1]>> to tensor<64x512xf16>
    %2 = tensor.empty() : tensor<1024x512xf32>
    %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<1024x512xf32>) -> tensor<1024x512xf32>
    %4 = linalg.matmul ins(%0, %1 : tensor<1024x64xf16>, tensor<64x512xf16>) outs(%3 : tensor<1024x512xf32>) -> tensor<1024x512xf32>
    %reinterpret_cast_1 = memref.reinterpret_cast %arg2 to offset: [0], sizes: [1024, 512], strides: [512, 1] : memref<*xf32> to memref<1024x512xf32, strided<[512, 1]>>
    bufferization.materialize_in_destination %4 in writable %reinterpret_cast_1 : (tensor<1024x512xf32>, memref<1024x512xf32, strided<[512, 1]>>) -> ()
    return
  }
}
```

### Linalg Bufferized IR

Since the `enabelHexKL` flag was enabled, we skip various passes in the HVX flow such as generalization, tiling and vectorization. Hence, after One shot bufferization, `linalg.matmul` operates on memref instead of tensors.
```mlir
module attributes {llvm.target_triple = "hexagon"} {
  func.func @matmul_hexkl(%arg0: memref<*xf16> {tt.divisibility = 16 : i32}, %arg1: memref<*xf16> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32) {
    %cst = arith.constant 0.000000e+00 : f32
    %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [0], sizes: [1024, 512], strides: [512, 1] : memref<*xf32> to memref<1024x512xf32, strided<[512, 1]>>
    %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [0], sizes: [1024, 64], strides: [64, 1] : memref<*xf16> to memref<1024x64xf16, strided<[64, 1]>>
    %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [0], sizes: [64, 512], strides: [512, 1] : memref<*xf16> to memref<64x512xf16, strided<[512, 1]>>
    linalg.fill ins(%cst : f32) outs(%reinterpret_cast : memref<1024x512xf32, strided<[512, 1]>>)
    linalg.matmul ins(%reinterpret_cast_0, %reinterpret_cast_1 : memref<1024x64xf16, strided<[64, 1]>>, memref<64x512xf16, strided<[512, 1]>>) outs(%reinterpret_cast : memref<1024x512xf32, strided<[512, 1]>>)
    return
  }
}
```

### HexKL Dialect IR
After some cleanup passes, `linalg.matmul` is lowered to lowered to the HexKL dialect. In the current implementation, the IR looks like:

```mlir
func.func @matmul_hexkl(%arg0: memref<*xf16> {tt.divisibility = 16 : i32}, %arg1: memref<*xf16> {tt.divisibility = 16 : i32}, %arg2: memref<*xf32> {tt.divisibility = 16 : i32}, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32) {
  %cst = arith.constant 0.000000e+00 : f32
  %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [0], sizes: [1024, 512], strides: [512, 1] : memref<*xf32> to memref<1024x512xf32, strided<[512, 1]>>
  %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [0], sizes: [1024, 64], strides: [64, 1] : memref<*xf16> to memref<1024x64xf16, strided<[64, 1]>>
  %reinterpret_cast_1 = memref.reinterpret_cast %arg1 to offset: [0], sizes: [64, 512], strides: [512, 1] : memref<*xf16> to memref<64x512xf16, strided<[512, 1]>>
  linalg.fill ins(%cst : f32) outs(%reinterpret_cast : memref<1024x512xf32, strided<[512, 1]>>)
  hexkl.matmul ins(%reinterpret_cast_0, %reinterpret_cast_1 : memref<1024x64xf16, strided<[64, 1]>>, memref<64x512xf16, strided<[512, 1]>>) outs(%reinterpret_cast : memref<1024x512xf32, strided<[512, 1]>>)
  return
}
```

### HexKL Runtime IR
Finally, while lowering the HexKL dialect to LLVM dialect, the `hexkl.matmul` is lowered to the HexKL runtime function call. The inputs to the hexkl runtime calls are Memref descriptors corresponding to the inputs and output.

```mlir
  llvm.func @hexkl_matmul_f16f16_f32(i64, i64, i64, !llvm.ptr, !llvm.ptr, !llvm.ptr)
  llvm.func @matmul_hexkl(%arg0: i64, %arg1: !llvm.ptr, %arg2: i64, %arg3: !llvm.ptr, %arg4: i64, %arg5: !llvm.ptr, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32) {
    %0 = llvm.mlir.poison : !llvm.struct<(i64, ptr)>
    %1 = llvm.insertvalue %arg4, %0[0] : !llvm.struct<(i64, ptr)> 
    %2 = llvm.insertvalue %arg5, %1[1] : !llvm.struct<(i64, ptr)> 
    %3 = builtin.unrealized_conversion_cast %2 : !llvm.struct<(i64, ptr)> to memref<*xf32>
    %4 = llvm.mlir.poison : !llvm.struct<(i64, ptr)>
    %5 = llvm.insertvalue %arg2, %4[0] : !llvm.struct<(i64, ptr)> 
    %6 = llvm.insertvalue %arg3, %5[1] : !llvm.struct<(i64, ptr)> 
    %7 = builtin.unrealized_conversion_cast %6 : !llvm.struct<(i64, ptr)> to memref<*xf16>
    %8 = llvm.mlir.poison : !llvm.struct<(i64, ptr)>
    %9 = llvm.insertvalue %arg0, %8[0] : !llvm.struct<(i64, ptr)> 
    %10 = llvm.insertvalue %arg1, %9[1] : !llvm.struct<(i64, ptr)> 
    %11 = builtin.unrealized_conversion_cast %10 : !llvm.struct<(i64, ptr)> to memref<*xf16>
    %c512 = arith.constant 512 : index
    %c1 = arith.constant 1 : index
    %c1024 = arith.constant 1024 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f32
    %reinterpret_cast = memref.reinterpret_cast %3 to offset: [0], sizes: [1024, 512], strides: [512, 1] : memref<*xf32> to memref<1024x512xf32, strided<[512, 1]>>
    %12 = builtin.unrealized_conversion_cast %reinterpret_cast : memref<1024x512xf32, strided<[512, 1]>> to !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %reinterpret_cast_0 = memref.reinterpret_cast %11 to offset: [0], sizes: [1024, 64], strides: [64, 1] : memref<*xf16> to memref<1024x64xf16, strided<[64, 1]>>
    %13 = builtin.unrealized_conversion_cast %reinterpret_cast_0 : memref<1024x64xf16, strided<[64, 1]>> to !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %reinterpret_cast_1 = memref.reinterpret_cast %7 to offset: [0], sizes: [64, 512], strides: [512, 1] : memref<*xf16> to memref<64x512xf16, strided<[512, 1]>>
    %14 = builtin.unrealized_conversion_cast %reinterpret_cast_1 : memref<64x512xf16, strided<[512, 1]>> to !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    cf.br ^bb1(%c0 : index)
  ^bb1(%15: index):  // 2 preds: ^bb0, ^bb5
    %16 = arith.cmpi slt, %15, %c1024 : index
    cf.cond_br %16, ^bb2, ^bb6
  ^bb2:  // pred: ^bb1
    cf.br ^bb3(%c0 : index)
  ^bb3(%17: index):  // 2 preds: ^bb2, ^bb4
    %18 = arith.cmpi slt, %17, %c512 : index
    cf.cond_br %18, ^bb4, ^bb5
  ^bb4:  // pred: ^bb3
    memref.store %cst, %reinterpret_cast[%15, %17] : memref<1024x512xf32, strided<[512, 1]>>
    %19 = arith.addi %17, %c1 : index
    cf.br ^bb3(%19 : index)
  ^bb5:  // pred: ^bb3
    %20 = arith.addi %15, %c1 : index
    cf.br ^bb1(%20 : index)
  ^bb6:  // pred: ^bb1
    %21 = llvm.extractvalue %12[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %22 = llvm.extractvalue %13[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %23 = llvm.extractvalue %14[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %24 = llvm.extractvalue %13[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %25 = llvm.extractvalue %13[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %26 = llvm.extractvalue %12[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    llvm.call @hexkl_matmul_f16f16_f32(%24, %26, %25, %21, %22, %23) : (i64, i64, i64, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    llvm.return
  }
``` 

### LLVM IR

**Stage 3**: After all MLIR lowering passes, the MLIR in LLVM dialect gets lowered to LLVM-IR. Here we can see the matmul op is lowered to a single HexKL runtime function call.

```mlir
define void @matmul_hexkl(i64 %0, ptr readonly captures(none) %1, i64 %2, ptr readonly captures(none) %3, i64 %4, ptr readonly captures(none) %5, i32 %6, i32 %7, i32 %8, i32 %9, i32 %10, i32 %11) local_unnamed_addr #0 {
  %13 = getelementptr i8, ptr %5, i32 4
  %14 = load ptr, ptr %13, align 8
  %15 = getelementptr i8, ptr %1, i32 4
  %16 = load ptr, ptr %15, align 8
  %17 = getelementptr i8, ptr %3, i32 4
  %18 = load ptr, ptr %17, align 8
  tail call void @llvm.memset.p0.i32(ptr noundef nonnull align 4 dereferenceable(2097152) %14, i8 0, i32 2097152, i1 false)
  tail call void @hexkl_matmul_f16f16_f32(i64 1024, i64 512, i64 64, ptr nonnull %14, ptr %16, ptr %18)
  ret void
}
```

All the above IR and files can be found from debug prints and inspected.

> **🎉 Congratulations!** You've successfully completed the Matmul HexKL tutorial, witnessing the different lowering IR stages from Triton to Hexagon runtime APIs.
