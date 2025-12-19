# 🚀 Flash Attention Triton Kernel

> **Overview**: This tutorial is on Flash Attention Triton Kernel. The kernel is originally from [Triton Flash Attention v2](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html#sphx-glr-getting-started-tutorials-06-fused-attention-py). We have modified below to cover only the forward path. The code shown below can be found in our repository at : `test/python/triton/test_flash_attention.py`.

We will skip some of the steps that are already emphasized in previous examples e.g. [vector-addition](vector_add.md). Also, we do not get into special handling of matmuls via HMX, DMA transfers, layout transformation etc. But we show general lowering of Flash Attention.

## Table of Contents

- [🔄 Compilation Pipeline Overview](#-compilation-pipeline-overview)
  - [🔧 Compilation Stages](#-compilation-stages)
- [🧮 Flash Attention Triton Kernel](#-flash-attention-triton-kernel)
  - [📝 Python Implementation](#-python-implementation)
- [🏃 Compilation and Execution](#-compilation-and-execution)
  - [🔨 Build Commands](#-build-commands)
  - [✅ Expected Output](#-expected-output)
- [🔍 Deep Dive into Lowering](#-deep-dive-into-lowering)
  - [Linalg Initial IR](#linalg-initial-ir)
  - [Linalg Fused IR](#linalg-fused-ir)
  - [Multi-threading IR](#multi-threading-ir)

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
| **5** | Linker | Create `libflash_attention_kernel.so` |
| **6** | Runtime | Load and execute on device |

---

## 🧮 Flash Attention Triton Kernel

### 📝 Python Implementation

The Triton kernel below is from [Triton Flash Attention v2](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html#sphx-glr-getting-started-tutorials-06-fused-attention-py). It is modified to cover only the forward path.

```python
import pytest
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from triton.backends.qcom_hexagon_backend.driver import HexagonDriver

triton.runtime.driver.set_active(HexagonDriver())

Z, H, N_CTX, D_HEAD = 1, 1, 1024, 64
BLOCK_N = 64
BLOCK_DMODEL = 64
STAGE = 1
NUM_THREADS = 4

assert N_CTX % NUM_THREADS == 0
BLOCK_M = N_CTX // NUM_THREADS

stride_0 = H * N_CTX * D_HEAD
stride_1 = N_CTX * D_HEAD
stride_2 = D_HEAD
stride_3 = 1
SCALE = 0.5

@triton.jit
def _attn_fwd_inner(
    acc, l_i, m_i, q, K_block_ptr, V_block_ptr, qk_scale,
    BLOCK_N: tl.constexpr, N_CTX: tl.constexpr,
):
    # range of values handled by this stage
    lo, hi = 0, N_CTX
    K_block_ptr = tl.advance(K_block_ptr, (lo, 0))
    V_block_ptr = tl.advance(V_block_ptr, (lo, 0))
    # loop over k, v and update accumulator
    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- compute qk ----
        k = tl.load(K_block_ptr)
        qk = tl.dot(
            q, tl.trans(k)
        )
        m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
        qk = qk * qk_scale - m_ij[:, None]
        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)
        # -- update m_i and l_i
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        # -- update output accumulator --
        acc = acc * alpha[:, None]
        # update acc
        v = tl.load(V_block_ptr)
        # p = p.to(tl.float32)
        acc = tl.dot(p, v, acc)
        # update m_i and l_i
        m_i = m_ij
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
        K_block_ptr = tl.advance(K_block_ptr, (BLOCK_N, 0))
    return acc, l_i, m_i


@triton.jit
def attention_fwd_kernel(
    pid1,  ## Set to 0
    Q,
    K,
    V,
    sm_scale: tl.constexpr, #
    Out,  #
    H,  #
    N_CTX: tl.constexpr,  #
    BLOCK_M: tl.constexpr,  #
    BLOCK_DMODEL: tl.constexpr,  #
    BLOCK_N: tl.constexpr,  #
    STAGE: tl.constexpr,  #
    stride_0: tl.constexpr, #
    stride_1: tl.constexpr, #
    stride_2: tl.constexpr, #
    stride_3: tl.constexpr, #
):

    pid0 = tl.program_id(0)
    start_m = pid0
    off_hz = pid1
    off_z = off_hz // H
    off_h = off_hz % H
    qvk_offset = off_z.to(tl.int32) * stride_0 + off_h.to(tl.int32) * stride_1
    # block pointers
    Q_block_ptr = tl.make_block_ptr(
        base=Q + qvk_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_2, stride_3),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        base=V + qvk_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_2, stride_3),
        offsets=(0, 0),
        block_shape=(BLOCK_N, BLOCK_DMODEL),
        order=(1, 0),
    )
    K_block_ptr = tl.make_block_ptr(
        base=K + qvk_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_2, stride_3),
        offsets=(0, 0),
        block_shape=(BLOCK_N, BLOCK_DMODEL),
        order=(1, 0),
    )
    O_block_ptr = tl.make_block_ptr(
        base=Out + qvk_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_2, stride_3),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0),
    )
    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - 99999999999999.0  # float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    # load scales
    qk_scale = sm_scale
    qk_scale *= 1.44269504  # 1/log(2)
    # load q: it will stay in SRAM throughout
    q = tl.load(Q_block_ptr)
    # stage 1: off-band
    # For causal = True, STAGE = 3 and _attn_fwd_inner gets 1 as its STAGE
    # For causal = False, STAGE = 1, and _attn_fwd_inner gets 3 as its STAGE
    acc, l_i, m_i = _attn_fwd_inner(
        acc,
        l_i,
        m_i,
        q,
        K_block_ptr,
        V_block_ptr,  #
        qk_scale,  #
        BLOCK_N,  #
        N_CTX,  #
    )
    # epilogue
    acc = acc / l_i[:, None]
    tl.store(O_block_ptr, acc.to(Out.type.element_ty))


def test_flash_attention():
    query = torch.rand(Z, H, N_CTX, D_HEAD)
    key = torch.rand(Z, H, N_CTX, D_HEAD)
    value = torch.rand(Z, H, N_CTX, D_HEAD)

    matrix_stride_0 = H * N_CTX * D_HEAD
    matrix_stride_1 = N_CTX * D_HEAD
    matrix_stride_2 = D_HEAD
    matrix_stride_3 = 1

    output = torch.zeros_like(query)

    grid = (NUM_THREADS,)
    attention_fwd_kernel[grid](
        pid1=0,
        Q=query,
        K=key,
        V=value,
        sm_scale=SCALE,
        Out=output,
        H=H,  #
        N_CTX=N_CTX,  #
        BLOCK_M=BLOCK_M,  #
        BLOCK_DMODEL=BLOCK_DMODEL,  #
        BLOCK_N=BLOCK_N,  #
        STAGE=STAGE,
        stride_0=stride_0,
        stride_1=stride_1,
        stride_2=stride_2,
        stride_3=stride_3,
        enableVTCMTiling=False,
        enableConvertToHexagonmem=False
    )

    reference = F.scaled_dot_product_attention(query, key, value, scale=SCALE)
    assert torch.allclose(output, reference)
```
---

## 🏃 Compilation and Execution

### 🔨 Build Commands

```bash
# Enable MLIR IR and LLVM-IR dumping to see intermediate representations
TRITON_ALWAYS_COMPILE=1 MLIR_ENABLE_DUMP=1 LLVM_IR_ENABLE_DUMP=1 pytest -sv test/python/triton/test_flash_attention.py
```

### ✅ Expected Output

If setup and device acquisition work correctly, you should see:

```bash
Test_Info: {
        Name:flash_attention_kernel
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

### Linalg Initial IR

After initial Triton front-end processing, and some initial hexagon-mlir passes, this is what the IR at Linalg level looks like. It is series unfused linalg-generic t this stage:
```mlir
#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
#map2 = affine_map<(d0, d1) -> (d0, 0)>
module attributes {llvm.target_triple = "hexagon"} {
  func.func @attention_fwd_kernel(
    %arg0: i32 {tt.divisibility = 16 : i32},
    %arg1: memref<*xf32> {tt.divisibility = 16 : i32},
    %arg2: memref<*xf32> {tt.divisibility = 16 : i32},
    %arg3: memref<*xf32> {tt.divisibility = 16 : i32},
    %arg4: memref<*xf32> {tt.divisibility = 16 : i32},
    %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32) {
    %cst = arith.constant 0.72134751 : f32
    %c0_i64 = arith.constant 0 : i64
    %c64_i64 = arith.constant 64 : i64
    %c64 = arith.constant 64 : index
    %c0_i32 = arith.constant 0 : i32
    %cst_0 = arith.constant 1.000000e+00 : f32
    %cst_1 = arith.constant -1.000000e+14 : f32
    %c1024_i32 = arith.constant 1024 : i32
    %c64_i32 = arith.constant 64 : i32
    %cst_2 = arith.constant 0.000000e+00 : f32
    %c256_i32 = arith.constant 256 : i32
    %c65536_i32 = arith.constant 65536 : i32
    %cst_3 = arith.constant 0xFF800000 : f32
    %0 = tensor.empty() : tensor<256x64xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<256x64xf32>) -> tensor<256x64xf32>
    %2 = tensor.empty() : tensor<256xf32>
    %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<256xf32>) -> tensor<256xf32>
    %4 = linalg.fill ins(%cst_0 : f32) outs(%2 : tensor<256xf32>) -> tensor<256xf32>
    %5 = linalg.fill ins(%cst_1 : f32) outs(%2 : tensor<256xf32>) -> tensor<256xf32>
    %6 = linalg.fill ins(%cst_2 : f32) outs(%0 : tensor<256x64xf32>) -> tensor<256x64xf32>
    %7 = arith.muli %arg0, %c65536_i32 : i32
    %8 = arith.index_cast %7 : i32 to index
    %9 = arith.muli %arg8, %c256_i32 : i32
    %10 = arith.index_cast %9 : i32 to index
    %11 = arith.muli %10, %c64 : index
    %12 = arith.addi %8, %11 : index

    %reinterpret_cast = memref.reinterpret_cast %arg1 to offset: [%12], sizes: [256, 64], strides: [64, 1]
     : memref<*xf32> to memref<256x64xf32, strided<[64, 1], offset: ?>>
    %13 = bufferization.to_tensor %reinterpret_cast restrict
     : memref<256x64xf32, strided<[64, 1], offset: ?>> to tensor<256x64xf32>

    %14:5 = scf.for %arg11 = %c0_i32 to %c1024_i32 step %c64_i32
              iter_args(%arg12 = %6, %arg13 = %4, %arg14 = %5, %arg15 = %c0_i64, %arg16 = %c0_i64)
             -> (tensor<256x64xf32>, tensor<256xf32>, tensor<256xf32>, i64, i64)  : i32 {                
      %17 = arith.index_cast %arg15 : i64 to index
      %18 = arith.muli %17, %c64 : index
      %19 = arith.addi %8, %18 : index
      %reinterpret_cast_5 = memref.reinterpret_cast %arg2 to offset: [%19], sizes: [64, 64], strides: [64, 1]
       : memref<*xf32> to memref<64x64xf32, strided<[64, 1], offset: ?>>
      %20 = bufferization.to_tensor %reinterpret_cast_5 restrict
       : memref<64x64xf32, strided<[64, 1], offset: ?>> to tensor<64x64xf32>
      %21 = tensor.empty() : tensor<64x64xf32>

      %transposed = linalg.transpose ins(%20 : tensor<64x64xf32>) outs(%21 : tensor<64x64xf32>) permutation = [1, 0] 
      %22 = linalg.matmul
              ins(%13, %transposed : tensor<256x64xf32>, tensor<64x64xf32>)
              outs(%6 : tensor<256x64xf32>) -> tensor<256x64xf32>

      %23 = tensor.empty() : tensor<64x256xf32>
      %transposed_6 = linalg.transpose
              ins(%22 : tensor<256x64xf32>) outs(%23 : tensor<64x256xf32>) permutation = [1, 0] 
      %24 = linalg.fill ins(%cst_3 : f32) outs(%2 : tensor<256xf32>) -> tensor<256xf32>

      %reduced = linalg.reduce ins(%transposed_6 : tensor<64x256xf32>) outs(%24 : tensor<256xf32>) dimensions = [0] 
        (%in: f32, %init: f32) {
          %46 = arith.maxnumf %in, %init : f32
          linalg.yield %46 : f32
      }

      %25 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]}
               ins(%reduced, %3 : tensor<256xf32>, tensor<256xf32>) outs(%reduced : tensor<256xf32>) {
      ^bb0(%in: f32, %in_12: f32, %out: f32):
        %46 = arith.mulf %in, %in_12 : f32
        linalg.yield %46 : f32
      } -> tensor<256xf32>

      %26 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]}
               ins(%arg14, %25 : tensor<256xf32>, tensor<256xf32>) outs(%arg14 : tensor<256xf32>) {
      ^bb0(%in: f32, %in_12: f32, %out: f32):
        %46 = arith.maxnumf %in, %in_12 : f32
        linalg.yield %46 : f32
      } -> tensor<256xf32>

      %27 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]}
               ins(%22, %1 : tensor<256x64xf32>, tensor<256x64xf32>) outs(%22 : tensor<256x64xf32>) {
      ^bb0(%in: f32, %in_12: f32, %out: f32):
        %46 = arith.mulf %in, %in_12 : f32
        linalg.yield %46 : f32
      } -> tensor<256x64xf32>

      %expanded_7 = tensor.expand_shape %26 [[0, 1]] output_shape [256, 1] : tensor<256xf32> into tensor<256x1xf32>

      %28 = linalg.generic {indexing_maps = [#map2, #map1], iterator_types = ["parallel", "parallel"]}
              ins(%expanded_7 : tensor<256x1xf32>) outs(%0 : tensor<256x64xf32>) attrs =  {broadcastDims = array<i64: 1>} {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      } -> tensor<256x64xf32>

      %29 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]}
              ins(%27, %28 : tensor<256x64xf32>, tensor<256x64xf32>) outs(%27 : tensor<256x64xf32>) {
      ^bb0(%in: f32, %in_12: f32, %out: f32):
        %46 = arith.subf %in, %in_12 : f32
        linalg.yield %46 : f32
      } -> tensor<256x64xf32>

      %30 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]}
               ins(%29 : tensor<256x64xf32>) outs(%29 : tensor<256x64xf32>) {
      ^bb0(%in: f32, %out: f32):
        %46 = math.exp2 %in : f32
        linalg.yield %46 : f32
      } -> tensor<256x64xf32>

      %transposed_8 = linalg.transpose ins(%30 : tensor<256x64xf32>) outs(%23 : tensor<64x256xf32>) permutation = [1, 0] 

      %31 = linalg.fill ins(%cst_2 : f32) outs(%2 : tensor<256xf32>) -> tensor<256xf32>
      %reduced_9 = linalg.reduce
            ins(%transposed_8 : tensor<64x256xf32>) outs(%31 : tensor<256xf32>) dimensions = [0] 
        (%in: f32, %init: f32) {
          %46 = arith.addf %in, %init : f32
          linalg.yield %46 : f32
      }

      %32 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]}
             ins(%arg14, %26 : tensor<256xf32>, tensor<256xf32>) outs(%arg14 : tensor<256xf32>) {
      ^bb0(%in: f32, %in_12: f32, %out: f32):
        %46 = arith.subf %in, %in_12 : f32
        linalg.yield %46 : f32
      } -> tensor<256xf32>

      %33 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]}
             ins(%32 : tensor<256xf32>) outs(%32 : tensor<256xf32>) {
      ^bb0(%in: f32, %out: f32):
        %46 = math.exp2 %in : f32
        linalg.yield %46 : f32
      } -> tensor<256xf32>

      %34 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]}
             ins(%arg13, %33 : tensor<256xf32>, tensor<256xf32>) outs(%arg13 : tensor<256xf32>) {
      ^bb0(%in: f32, %in_12: f32, %out: f32):
        %46 = arith.mulf %in, %in_12 : f32
        linalg.yield %46 : f32
      } -> tensor<256xf32>

      %35 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]}
              ins(%34, %reduced_9 : tensor<256xf32>, tensor<256xf32>) outs(%34 : tensor<256xf32>) {
      ^bb0(%in: f32, %in_12: f32, %out: f32):
        %46 = arith.addf %in, %in_12 : f32
        linalg.yield %46 : f32
      } -> tensor<256xf32>

      %expanded_10 = tensor.expand_shape %33 [[0, 1]] output_shape [256, 1] : tensor<256xf32> into tensor<256x1xf32>

      %36 = linalg.generic {indexing_maps = [#map2, #map1], iterator_types = ["parallel", "parallel"]}
             ins(%expanded_10 : tensor<256x1xf32>) outs(%0 : tensor<256x64xf32>) attrs =  {broadcastDims = array<i64: 1>} {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      } -> tensor<256x64xf32>

      %37 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]}
              ins(%arg12, %36 : tensor<256x64xf32>, tensor<256x64xf32>) outs(%arg12 : tensor<256x64xf32>) {
      ^bb0(%in: f32, %in_12: f32, %out: f32):
        %46 = arith.mulf %in, %in_12 : f32
        linalg.yield %46 : f32
      } -> tensor<256x64xf32>
      %38 = arith.index_cast %arg16 : i64 to index
      %39 = arith.muli %38, %c64 : index
      %40 = arith.addi %8, %39 : index
      %reinterpret_cast_11 = memref.reinterpret_cast %arg3 to offset: [%40], sizes: [64, 64], strides: [64, 1]
         : memref<*xf32> to memref<64x64xf32, strided<[64, 1], offset: ?>>
      %41 = bufferization.to_tensor %reinterpret_cast_11 restrict
         : memref<64x64xf32, strided<[64, 1], offset: ?>> to tensor<64x64xf32>

      %42 = linalg.matmul
          ins(%30, %41 : tensor<256x64xf32>, tensor<64x64xf32>) outs(%6 : tensor<256x64xf32>) -> tensor<256x64xf32>
          
      %43 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]}
              ins(%37, %42 : tensor<256x64xf32>, tensor<256x64xf32>) outs(%37 : tensor<256x64xf32>) {
      ^bb0(%in: f32, %in_12: f32, %out: f32):
        %46 = arith.addf %in, %in_12 : f32
        linalg.yield %46 : f32
      } -> tensor<256x64xf32>
      %44 = arith.addi %arg16, %c64_i64 : i64
      %45 = arith.addi %arg15, %c64_i64 : i64
      scf.yield %43, %35, %26, %45, %44 : tensor<256x64xf32>, tensor<256xf32>, tensor<256xf32>, i64, i64
    }
    %expanded = tensor.expand_shape %14#1 [[0, 1]] output_shape [256, 1] : tensor<256xf32> into tensor<256x1xf32>

    %15 = linalg.generic {indexing_maps = [#map2, #map1], iterator_types = ["parallel", "parallel"]}
     ins(%expanded : tensor<256x1xf32>) outs(%0 : tensor<256x64xf32>) attrs =  {broadcastDims = array<i64: 1>} {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<256x64xf32>

    %16 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]}
             ins(%14#0, %15 : tensor<256x64xf32>, tensor<256x64xf32>) outs(%14#0 : tensor<256x64xf32>) {
    ^bb0(%in: f32, %in_5: f32, %out: f32):
      %17 = arith.divf %in, %in_5 : f32
      linalg.yield %17 : f32
    } -> tensor<256x64xf32>

    %reinterpret_cast_4 = memref.reinterpret_cast %arg4 to offset: [%12], sizes: [256, 64], strides: [64, 1]
     : memref<*xf32> to memref<256x64xf32, strided<[64, 1], offset: ?>>
    bufferization.materialize_in_destination %16 in writable %reinterpret_cast_4
     : (tensor<256x64xf32>, memref<256x64xf32, strided<[64, 1], offset: ?>>) -> ()
    return
  }
}
```

### Linalg Fused IR

After linalg level operator fusion, the IR looks bit like:
```mlir
// -----// IR Dump After HexagonFusion (hexagon-fusion) //----- //
func.func @attention_fwd_kernel(
              %arg0: i32 {tt.divisibility = 16 : i32},
              %arg1: memref<*xf32> {tt.divisibility = 16 : i32},
              %arg2: memref<*xf32> {tt.divisibility = 16 : i32},
              %arg3: memref<*xf32> {tt.divisibility = 16 : i32},
              %arg4: memref<*xf32> {tt.divisibility = 16 : i32},
               %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32) {
  %cst = arith.constant 0.72134751 : f32
  %c0_i64 = arith.constant 0 : i64
  %c64_i64 = arith.constant 64 : i64
  %c64 = arith.constant 64 : index
  %c0_i32 = arith.constant 0 : i32
  %cst_0 = arith.constant 1.000000e+00 : f32
  %cst_1 = arith.constant -1.000000e+14 : f32
  %c1024_i32 = arith.constant 1024 : i32
  %c64_i32 = arith.constant 64 : i32
  %cst_2 = arith.constant 0.000000e+00 : f32
  %c256_i32 = arith.constant 256 : i32
  %c65536_i32 = arith.constant 65536 : i32
  %cst_3 = arith.constant 0xFF800000 : f32
  %0 = tensor.empty() : tensor<256x64xf32>
  %1 = tensor.empty() : tensor<256xf32>
  %2 = linalg.fill ins(%cst_0 : f32) outs(%1 : tensor<256xf32>) -> tensor<256xf32>
  %3 = linalg.fill ins(%cst_1 : f32) outs(%1 : tensor<256xf32>) -> tensor<256xf32>
  %4 = linalg.fill ins(%cst_2 : f32) outs(%0 : tensor<256x64xf32>) -> tensor<256x64xf32>
  %5 = arith.muli %arg0, %c65536_i32 : i32
  %6 = arith.index_cast %5 : i32 to index
  %7 = arith.muli %arg8, %c256_i32 : i32
  %8 = arith.index_cast %7 : i32 to index
  %9 = arith.muli %8, %c64 : index
  %10 = arith.addi %6, %9 : index
  %reinterpret_cast = memref.reinterpret_cast %arg4 to offset: [%10], sizes: [256, 64], strides: [64, 1]
   : memref<*xf32> to memref<256x64xf32, strided<[64, 1], offset: ?>>
  %reinterpret_cast_4 = memref.reinterpret_cast %arg1 to offset: [%10], sizes: [256, 64], strides: [64, 1]
   : memref<*xf32> to memref<256x64xf32, strided<[64, 1], offset: ?>>

  %11 = bufferization.to_tensor %reinterpret_cast_4 restrict
   : memref<256x64xf32, strided<[64, 1], offset: ?>> to tensor<256x64xf32>

  %12:5 = scf.for %arg11 = %c0_i32 to %c1024_i32 step %c64_i32
              iter_args(%arg12 = %4, %arg13 = %2, %arg14 = %3, %arg15 = %c0_i64, %arg16 = %c0_i64)
              -> (tensor<256x64xf32>, tensor<256xf32>, tensor<256xf32>, i64, i64)  : i32 {
    %14 = arith.index_cast %arg15 : i64 to index
    %15 = arith.muli %14, %c64 : index
    %16 = arith.addi %6, %15 : index

    %reinterpret_cast_5 = memref.reinterpret_cast %arg2 to offset: [%16], sizes: [64, 64], strides: [64, 1]
     : memref<*xf32> to memref<64x64xf32, strided<[64, 1], offset: ?>>
    %17 = bufferization.to_tensor %reinterpret_cast_5 restrict
     : memref<64x64xf32, strided<[64, 1], offset: ?>> to tensor<64x64xf32>
    %18 = linalg.generic
     {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1)>,
                       affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d2)>],
                       iterator_types = ["parallel", "reduction", "parallel"]}
                       ins(%11, %17 : tensor<256x64xf32>, tensor<64x64xf32>) outs(%4 : tensor<256x64xf32>) {
    ^bb0(%in: f32, %in_9: f32, %out: f32):
      %34 = arith.mulf %in, %in_9 : f32
      %35 = arith.addf %out, %34 : f32
      linalg.yield %35 : f32
    } -> tensor<256x64xf32>

    %19 = linalg.fill ins(%cst_3 : f32) outs(%1 : tensor<256xf32>) -> tensor<256xf32>
    %20 = linalg.generic 
    {indexing_maps = [affine_map<(d0, d1) -> (d1, d0)>, affine_map<(d0, d1) -> (d1)>],
     iterator_types = ["reduction", "parallel"]} ins(%18 : tensor<256x64xf32>) outs(%19 : tensor<256xf32>) {
    ^bb0(%in: f32, %out: f32):
      %34 = arith.maxnumf %in, %out : f32
      linalg.yield %34 : f32
    } -> tensor<256xf32>

    %21 = linalg.generic
     {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]}
       ins(%arg14, %20 : tensor<256xf32>, tensor<256xf32>) outs(%1 : tensor<256xf32>) {
    ^bb0(%in: f32, %in_9: f32, %out: f32):
      %34 = arith.mulf %in_9, %cst : f32
      %35 = arith.maxnumf %in, %34 : f32
      linalg.yield %35 : f32
    } -> tensor<256xf32>

    %expanded_6 = tensor.expand_shape %21 [[0, 1]] output_shape [256, 1] : tensor<256xf32> into tensor<256x1xf32>
    %22 = linalg.generic
     {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (d0, d1)>],
     iterator_types = ["parallel", "parallel"]}
      ins(%18, %expanded_6 : tensor<256x64xf32>, tensor<256x1xf32>) outs(%0 : tensor<256x64xf32>) {
    ^bb0(%in: f32, %in_9: f32, %out: f32):
      %34 = arith.mulf %in, %cst : f32
      %35 = arith.subf %34, %in_9 : f32
      %36 = math.exp2 %35 : f32
      linalg.yield %36 : f32
    } -> tensor<256x64xf32>

    %23 = linalg.fill ins(%cst_2 : f32) outs(%1 : tensor<256xf32>) -> tensor<256xf32>
    %24 = linalg.generic
     {indexing_maps = [affine_map<(d0, d1) -> (d1, d0)>, affine_map<(d0, d1) -> (d1)>],
      iterator_types = ["reduction", "parallel"]}
       ins(%22 : tensor<256x64xf32>) outs(%23 : tensor<256xf32>) {
    ^bb0(%in: f32, %out: f32):
      %34 = arith.addf %in, %out : f32
      linalg.yield %34 : f32
    } -> tensor<256xf32>

    %25:2 = linalg.generic
     {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>,
                       affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]}
       ins(%arg13, %arg14, %21, %24 : tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>)
       uts(%1, %1 : tensor<256xf32>, tensor<256xf32>) {
    ^bb0(%in: f32, %in_9: f32, %in_10: f32, %in_11: f32, %out: f32, %out_12: f32):
      %34 = arith.subf %in_9, %in_10 : f32
      %35 = math.exp2 %34 : f32
      %36 = arith.mulf %in, %35 : f32
      %37 = arith.addf %36, %in_11 : f32
      linalg.yield %35, %37 : f32, f32
    } -> (tensor<256xf32>, tensor<256xf32>)

    %expanded_7 = tensor.expand_shape %25#0 [[0, 1]] output_shape [256, 1] : tensor<256xf32> into tensor<256x1xf32>
    %26 = arith.index_cast %arg16 : i64 to index
    %27 = arith.muli %26, %c64 : index
    %28 = arith.addi %6, %27 : index
    %reinterpret_cast_8 = memref.reinterpret_cast %arg3 to offset: [%28], sizes: [64, 64], strides: [64, 1]
     : memref<*xf32> to memref<64x64xf32, strided<[64, 1], offset: ?>>

    %29 = bufferization.to_tensor %reinterpret_cast_8 restrict
     : memref<64x64xf32, strided<[64, 1], offset: ?>> to tensor<64x64xf32>

    %30 = linalg.generic
     {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d1, d2)>,
                       affine_map<(d0, d1, d2) -> (d0, d2)>],
      iterator_types = ["parallel", "reduction", "parallel"]}
       ins(%22, %29 : tensor<256x64xf32>, tensor<64x64xf32>) outs(%4 : tensor<256x64xf32>) {
    ^bb0(%in: f32, %in_9: f32, %out: f32):
      %34 = arith.mulf %in, %in_9 : f32
      %35 = arith.addf %out, %34 : f32
      linalg.yield %35 : f32
    } -> tensor<256x64xf32>

    %31 = linalg.generic
     {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, 0)>,
                      affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
                       iterator_types = ["parallel", "parallel"]}
       ins(%arg12, %expanded_7, %30 : tensor<256x64xf32>, tensor<256x1xf32>, tensor<256x64xf32>) outs(%0 : tensor<256x64xf32>) {
    ^bb0(%in: f32, %in_9: f32, %in_10: f32, %out: f32):
      %34 = arith.mulf %in, %in_9 : f32
      %35 = arith.addf %34, %in_10 : f32
      linalg.yield %35 : f32
    } -> tensor<256x64xf32>
    %32 = arith.addi %arg16, %c64_i64 : i64
    %33 = arith.addi %arg15, %c64_i64 : i64
    scf.yield %31, %25#1, %21, %33, %32 : tensor<256x64xf32>, tensor<256xf32>, tensor<256xf32>, i64, i64
  }

  %expanded = tensor.expand_shape %12#1 [[0, 1]] output_shape [256, 1] : tensor<256xf32> into tensor<256x1xf32>
  %13 = linalg.generic
   {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]} ins(%12#0, %expanded
     : tensor<256x64xf32>, tensor<256x1xf32>) outs(%0 : tensor<256x64xf32>) {
  ^bb0(%in: f32, %in_5: f32, %out: f32):
    %14 = arith.divf %in, %in_5 : f32
    linalg.yield %14 : f32
  } -> tensor<256x64xf32>
  bufferization.materialize_in_destination %13 in restrict writable %reinterpret_cast
   : (tensor<256x64xf32>, memref<256x64xf32, strided<[64, 1], offset: ?>>) -> ()
  return
}
```
### Multi-threading IR
Below is snap-shot of IR after HVX multi-threading and vectorization:

```mlir
    scf.for %arg18 = %c0_37 to %c256_38 step %c64_39 {
      %token = async.execute {
        %subview = memref.subview %reinterpret_cast_9[%arg18, 0] [64, 64] [1, 1] : memref<256x64xf32, strided<[64, 1], offset: ?>> to memref<64x64xf32, strided<[64, 1], offset: ?>>
        %subview_61 = memref.subview %alloc_11[%arg18, 0] [64, 64] [1, 1] : memref<256x64xf32> to memref<64x64xf32, strided<[64, 1], offset: ?>>
        scf.for %arg19 = %c0 to %c64 step %c1 {
          scf.for %arg20 = %c0 to %c64 step %c1 {
            scf.for %arg21 = %c0 to %c64 step %c32 {
              %subview_62 = memref.subview %subview[%arg19, %arg20] [1, 1] [1, 1]
                   : memref<64x64xf32, strided<[64, 1], offset: ?>> to memref<f32, strided<[], offset: ?>>
              %subview_63 = memref.subview %reinterpret_cast_36[%arg21, %arg20] [32, 1] [1, 1] 
                   : memref<64x64xf32, strided<[64, 1], offset: ?>> to memref<32xf32, strided<[64], offset: ?>>
              %subview_64 = memref.subview %subview_61[%arg19, %arg21] [1, 32] [1, 1]
                   : memref<64x64xf32, strided<[64, 1], offset: ?>> to memref<32xf32, strided<[1], offset: ?>>
              %26 = vector.transfer_read %subview_62[], %cst : memref<f32, strided<[], offset: ?>>, vector<f32>
              %27 = vector.broadcast %26 : vector<f32> to vector<32xf32>
              %28 = vector.transfer_read %subview_63[%c0], %cst {in_bounds = [true]} 
                  : memref<32xf32, strided<[64], offset: ?>>, vector<32xf32>
              %29 = vector.transfer_read %subview_64[%c0], %cst {in_bounds = [true]}
                  : memref<32xf32, strided<[1], offset: ?>>, vector<32xf32>
              %30 = arith.mulf %27, %28 fastmath<fast> : vector<32xf32>
              %31 = arith.addf %29, %30 fastmath<fast> : vector<32xf32>
              vector.transfer_write %31, %subview_64[%c0] {in_bounds = [true]}
                   : vector<32xf32>, memref<32xf32, strided<[1], offset: ?>>
            }
          }
        }
        async.yield
      }
      %25 = async.add_to_group %token, %17 : !async.token
    }
    async.await_all %17
    ...
          scf.for %arg18 = %c0 to %c256 step %c1 {
        scf.for %arg19 = %c0 to %c64 step %c32 {
          %subview = memref.subview %alloc_11[%arg18, %arg19] [1, 32] [1, 1] : memref<256x64xf32> to memref<32xf32, strided<[1], offset: ?>>
          %subview_65 = memref.subview %expand_shape_43[%arg18, 0] [1, 1] [1, 1] : memref<256x1xf32> to memref<f32, strided<[], offset: ?>>
          %subview_66 = memref.subview %alloc[%arg18, %arg19] [1, 32] [1, 1] : memref<256x64xf32> to memref<32xf32, strided<[1], offset: ?>>
          %27 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xf32, strided<[1], offset: ?>>, vector<32xf32>
          %28 = vector.transfer_read %subview_65[], %cst : memref<f32, strided<[], offset: ?>>, vector<f32>
          %29 = vector.broadcast %28 : vector<f32> to vector<32xf32>
          %30 = arith.mulf %27, %cst_0 fastmath<fast> : vector<32xf32>
          %31 = arith.subf %30, %29 fastmath<fast> : vector<32xf32>
          %32 = math.exp2 %31 fastmath<fast> : vector<32xf32>
          vector.transfer_write %32, %subview_66[%c0] {in_bounds = [true]} : vector<32xf32>, memref<32xf32, strided<[1], offset: ?>>
        }
      }
```    

All the above IR and files can be found from debug prints and inspected.

> **🎉 Congratulations!** You've successfully completed the Flash Attention tutorial, witnessing how memory-efficient attention mechanisms can be compiled and optimized for Hexagon architecture with sophisticated tiling, vectorization, and online softmax computation.
