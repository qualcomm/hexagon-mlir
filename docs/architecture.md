# Hexagon-MLIR Architecture

This document describes the high-level architecture of the Hexagon-MLIR compiler stack.

## Overview

Hexagon-MLIR is an MLIR-based AI compiler stack that compiles **Triton kernels** and **PyTorch models** to run on Qualcomm Hexagon NPUs. The system bridges the gap between high-level ML frameworks and low-level Hexagon hardware through a series of MLIR dialect lowerings and hardware-specific optimizations.

```
 ┌─────────────────────────────────────────────────────────────────────┐
 │                        User Entry Points                           │
 │                                                                    │
 │   ┌──────────────────────┐         ┌──────────────────────────┐    │
 │   │   Triton Kernels     │         │   PyTorch Models         │    │
 │   │   (Python)           │         │   (torch.nn)             │    │
 │   └─────────┬────────────┘         └────────────┬─────────────┘    │
 └─────────────┼───────────────────────────────────┼──────────────────┘
               │                                   │
               ▼                                   ▼
 ┌─────────────────────────┐         ┌──────────────────────────┐
 │  Triton Frontend (TTIR) │         │  Torch-MLIR Frontend     │
 │  ┌───────────────────┐  │         │  (torch-mlir)            │
 │  │ make_ttir()       │  │         └────────────┬─────────────┘
 │  │ - Inliner         │  │                      │
 │  │ - Canonicalizer   │  │                      │
 │  │ - CSE             │  │                      │
 │  │ - Loop Unroll     │  │                      │
 │  └────────┬──────────┘  │                      │
 └───────────┼─────────────┘                      │
             │                                    │
             ▼                                    │
 ┌─────────────────────────┐                      │
 │  Triton-Shared           │                      │
 │  (triton-shared-opt)    │                      │
 │  --triton-to-linalg-    │                      │
 │    experimental         │                      │
 └───────────┬─────────────┘                      │
             │                                    │
             ▼                                    ▼
 ┌─────────────────────────────────────────────────────────────────────┐
 │                          Linalg IR                                  │
 │              (Common intermediate representation)                   │
 └──────────────────────────────┬──────────────────────────────────────┘
                                │
                                ▼
 ┌─────────────────────────────────────────────────────────────────────┐
 │                                                                     │
 │              Hexagon-MLIR Backend (qcom_hexagon_backend)            │
 │                                                                     │
 │  ┌───────────────────────────────────────────────────────────────┐  │
 │  │                   Custom MLIR Dialects                        │  │
 │  │                                                               │  │
 │  │  ┌──────────┐ ┌───────────┐ ┌─────┐ ┌──────────┐ ┌───────┐  │  │
 │  │  │ HexKL    │ │HexagonMem │ │ TTX │ │HexagonTPtr│ │TmTensor│  │  │
 │  │  │(Matrix   │ │(Memory    │ │(Scan│ │(Typed    │ │(Atten- │  │  │
 │  │  │ Multiply)│ │ Mgmt)     │ │Ops) │ │ Pointer) │ │tion)   │  │  │
 │  │  └──────────┘ └───────────┘ └─────┘ └──────────┘ └───────┘  │  │
 │  └───────────────────────────────────────────────────────────────┘  │
 │                                                                     │
 │  ┌───────────────────────────────────────────────────────────────┐  │
 │  │              Optimization Transforms                          │  │
 │  │                                                               │  │
 │  │  ┌─────────────────────┐  ┌───────────────────────────────┐  │  │
 │  │  │ Tiling & Scheduling │  │ Memory Optimization           │  │  │
 │  │  │ - HexagonTiling     │  │ - VTCMTiling (DDR->VTCM)     │  │  │
 │  │  │ - ConvTiling        │  │ - MemoryOffsets               │  │  │
 │  │  │ - AffineTiling      │  │ - ConvertToHexagonmem         │  │  │
 │  │  │ - FormVirtualThreads│  │ - DoubleBuffering (S1/S2)     │  │  │
 │  │  │ - ScheduleMatmulHVX │  │ - CopyCanonicalization        │  │  │
 │  │  └─────────────────────┘  │ - HexmemCpyToDMA             │  │  │
 │  │                           └───────────────────────────────┘  │  │
 │  │  ┌─────────────────────┐  ┌───────────────────────────────┐  │  │
 │  │  │ Compute Transforms  │  │ Vectorization & Lowering      │  │  │
 │  │  │ - HexagonFusion     │  │ - HexagonVectorization        │  │  │
 │  │  │ - LinalgGeneralize  │  │ - HexagonVectorLowering       │  │  │
 │  │  │ - MatmulToHexKL     │  │ - AffineVectorize             │  │  │
 │  │  │ - MatmulToConv      │  │ - ExpandBoolVec               │  │  │
 │  │  │ - DecomposeHexKL    │  │ - HexagonRoutines (HVX)       │  │  │
 │  │  │ - HoistScalarOps    │  │ - ExpandMathOps               │  │  │
 │  │  │ - FastInverse       │  │ - SmallExponentToMultiply     │  │  │
 │  │  └─────────────────────┘  └───────────────────────────────┘  │  │
 │  │  ┌─────────────────────┐  ┌───────────────────────────────┐  │  │
 │  │  │ Multi-threading     │  │ Dialect Lowering              │  │  │
 │  │  │ - FormSCFThreads    │  │ - LowerTTX                    │  │  │
 │  │  │ - FormAsyncThreads  │  │ - LowerTPtr                   │  │  │
 │  │  │ - FormVirtualThreads│  │ - LowerTmTensor               │  │  │
 │  │  │ - HexagonSlicing    │  │ - LowerLibdevice              │  │  │
 │  │  └─────────────────────┘  └───────────────────────────────┘  │  │
 │  └───────────────────────────────────────────────────────────────┘  │
 │                                                                     │
 │  ┌───────────────────────────────────────────────────────────────┐  │
 │  │               Conversion Passes (to LLVM)                     │  │
 │  │                                                               │  │
 │  │  ┌─────────────┐  ┌──────────────┐  ┌────────────────────┐  │  │
 │  │  │LinalgToLLVM │  │AffineToLLVM  │  │HexagonMemToLLVM    │  │  │
 │  │  │(main pipe-  │  │(loop tiling, │  │(alloc/copy/dealloc │  │  │
 │  │  │ line with   │  │ vectorize,   │  │ -> runtime calls)  │  │  │
 │  │  │ 40+ options)│  │ tile memory) │  └────────────────────┘  │  │
 │  │  └─────────────┘  └──────────────┘  ┌────────────────────┐  │  │
 │  │  ┌─────────────┐  ┌──────────────┐  │DMAToLLVM           │  │  │
 │  │  │HexKLToLLVM  │  │LowerConstants│  │(memref.dma ->      │  │  │
 │  │  │(matmul ->   │  │Separately    │  │ runtime calls)     │  │  │
 │  │  │ runtime API)│  └──────────────┘  └────────────────────┘  │  │
 │  │  └─────────────┘                                             │  │
 │  └───────────────────────────────────────────────────────────────┘  │
 │                                                                     │
 │  ┌───────────────────────────────────────────────────────────────┐  │
 │  │                Target Translation                             │  │
 │  │                                                               │  │
 │  │  ┌──────────────────────────┐  ┌──────────────────────────┐  │  │
 │  │  │ Linalg -> MLIR LLVM      │  │ MLIR LLVM -> LLVM IR     │  │  │
 │  │  │ (translateLinalgTo       │  │ (translateHexagonMlirLlvm │  │  │
 │  │  │  LLVMMLIR)               │  │  ToLLVMIR)               │  │  │
 │  │  └──────────────────────────┘  └─────────────┬────────────┘  │  │
 │  │                                               │               │  │
 │  │                                               ▼               │  │
 │  │                                 ┌──────────────────────────┐  │  │
 │  │                                 │ LLVM IR -> Object Code   │  │  │
 │  │                                 │ (Hexagon Target .o)      │  │  │
 │  │                                 └──────────────────────────┘  │  │
 │  └───────────────────────────────────────────────────────────────┘  │
 └──────────────────────────────────┬──────────────────────────────────┘
                                    │
                                    ▼
 ┌─────────────────────────────────────────────────────────────────────┐
 │                    Runtime & Execution Layer                        │
 │                                                                     │
 │  ┌───────────────────────┐   ┌───────────────────────────────────┐  │
 │  │  Wrapper Generator    │   │  Hexagon Runtime Libraries        │  │
 │  │  - TritonHexagon      │   │                                   │  │
 │  │    WrapperGenerator   │   │  ┌──────────┐  ┌──────────────┐  │  │
 │  │  - TorchMlirHexagon   │   │  │ HexagonAPI│  │ HexKL API    │  │  │
 │  │    WrapperGenerator   │   │  │ (Buffer  │  │ (Matrix      │  │  │
 │  │                       │   │  │  Mgmt)   │  │  Library)    │  │  │
 │  │  Generates C++ launch │   │  └──────────┘  └──────────────┘  │  │
 │  │  code for kernels     │   │  ┌──────────┐  ┌──────────────┐  │  │
 │  └───────────────────────┘   │  │ UserDMA  │  │ VTCMPool     │  │  │
 │                              │  │ (DMA     │  │ (TCM Memory  │  │  │
 │  ┌───────────────────────┐   │  │ Engine)  │  │  Allocator)  │  │  │
 │  │  Hexagon Executor     │   │  └──────────┘  └──────────────┘  │  │
 │  │  - Compile .o -> .so  │   │  ┌──────────┐  ┌──────────────┐  │  │
 │  │  - Push to device     │   │  │ HVX      │  │ Async        │  │  │
 │  │  - Execute on NPU     │   │  │Intrinsics│  │ Runtime      │  │  │
 │  │  - Collect results    │   │  │ (Vector  │  │ (Thread Pool │  │  │
 │  └───────────────────────┘   │  │  Ops)    │  │  + Multi-    │  │  │
 │                              │  └──────────┘  │  threading)  │  │  │
 │  ┌───────────────────────┐   │               └──────────────┘  │  │
 │  │  Launchers            │   └───────────────────────────────────┘  │
 │  │  - TritonHexagon-     │                                          │
 │  │    Launcher           │                                          │
 │  │  - TorchMLIRHexagon-  │                                          │
 │  │    Launcher           │                                          │
 │  │  - MLIRHexagonLauncher│                                          │
 │  └───────────────────────┘                                          │
 └─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
                       ┌──────────────────────┐
                       │  Hexagon NPU (HTP)   │
                       │  - HVX Vector Units  │
                       │  - HMX Matrix Units  │
                       │  - TCM (Tightly      │
                       │    Coupled Memory)   │
                       │  - DDR Memory        │
                       └──────────────────────┘
```

## Repository Structure

```
hexagon-mlir/
├── qcom_hexagon_backend/          # Core compiler backend
│   ├── include/hexagon/           # C++ headers
│   │   ├── Dialect/               # MLIR dialect definitions (.td + .h)
│   │   │   ├── HexKL/             #   Matrix multiplication (HMX hardware)
│   │   │   ├── HexagonMem/        #   Memory management (VTCM/DDR)
│   │   │   ├── HexagonTPtr/       #   Typed pointer operations
│   │   │   ├── TTX/               #   Triton eXtension ops (scan, cumsum)
│   │   │   └── TmTensor/          #   Tensor-model ops (attention)
│   │   ├── Conversion/            # Dialect-to-LLVM conversion passes
│   │   │   ├── AffineToLLVM/      #   Affine loops -> LLVM + tiling/vectorization
│   │   │   ├── DMAToLLVM/         #   DMA ops -> runtime function calls
│   │   │   ├── HexKLToLLVM/       #   HexKL matmul -> runtime API calls
│   │   │   ├── HexagonMemToLLVM/  #   Memory ops -> runtime alloc/dealloc calls
│   │   │   └── LinalgToLLVM/      #   Main pipeline (40+ configurable options)
│   │   ├── Transforms/            # Optimization passes
│   │   └── Target/                # MLIR -> LLVM IR -> Object code translation
│   ├── lib/                       # C++ implementations of dialects/passes
│   ├── bin/                       # Executables and runtime
│   │   ├── linalg-hexagon-opt.cpp #   mlir-opt with Hexagon passes registered
│   │   ├── linalg-hexagon-translate.cpp  # MLIR -> Hexagon object translation
│   │   └── runtime/               #   Hexagon on-device runtime libraries
│   │       ├── src/               #     Buffer management, C API
│   │       ├── include/           #     Runtime headers
│   │       ├── UserDMA/           #     DMA engine implementation
│   │       ├── multithreading/    #     Thread pool + async runtime
│   │       └── test/              #     Runtime unit tests
│   ├── backend/                   # Python backend integration
│   │   ├── compiler.py            #   HexagonBackend (Triton BaseBackend)
│   │   ├── driver.py              #   HexagonDriver (Triton DriverBase)
│   │   ├── triton_hexagon_launcher.py    # Triton kernel execution
│   │   ├── torch_mlir_hexagon_launcher.py # PyTorch model execution
│   │   ├── mlir_launcher.py       #   Direct MLIR testing launcher
│   │   ├── hexagon_executor.py    #   .o -> .so linking + device execution
│   │   └── hexagon_extern/        #   libdevice function mappings
│   ├── python/                    # Pybind11 bindings (C++ <-> Python)
│   ├── test/                      # LIT tests for passes and dialects
│   └── utils/                     # Build utilities
├── test/                          # End-to-end integration tests
│   └── python/
│       ├── triton/                #   Triton kernel tests
│       ├── torch-mlir/            #   PyTorch model tests
│       └── mlir/                  #   MLIR-level tests
├── scripts/                       # Build scripts
│   ├── build_hexagon_mlir.sh      #   Build the Hexagon MLIR backend
│   ├── build_triton.sh            #   Build Triton with Hexagon backend
│   └── set_local_env.sh           #   Environment variable setup
├── ci/                            # CI pipeline configuration
├── third_party_software/          # Patches for Triton and triton-shared
└── docs/                          # User guide, tutorials, developer guide
```

## Component Details

### 1. Custom MLIR Dialects

Five custom dialects model Hexagon-specific hardware concepts:

| Dialect | Purpose | Key Operations |
|---------|---------|----------------|
| **HexKL** | Hexagon Kernel Library for matrix multiplication using HMX | `matmul`, `micro_hmx_mm_f16`, `micro_hmx_setup_acc_read_f16`, layout transforms |
| **HexagonMem** | Memory management across DDR and VTCM address spaces | `alloc`, `dealloc`, `copy`, `convert_layout` |
| **TTX** | Triton eXtension operations not in standard MLIR | `scan`, `scan.return`, `cumsum` |
| **HexagonTPtr** | Typed pointer arithmetic for Triton pointer semantics | `type_offset`, `from_memref`, `to_memref`, `ptradd` |
| **TmTensor** | Tensor-model operations for neural network patterns | `attention` (Scaled Dot Product Attention) |

### 2. Compilation Pipeline

The compilation flows through two main paths that converge on Linalg IR:

**Triton Path:**
```
Triton Python -> TTIR -> triton-shared (Linalg) -> LinalgToLLVM pipeline -> Object Code
```

**Torch-MLIR Path:**
```
PyTorch Model -> torch-mlir (Linalg) -> LinalgToLLVM pipeline -> Object Code
```

The **LinalgToLLVM** pass is the central pipeline with 40+ configurable options controlling:
- **Fusion**: Operator fusion with multi-use and recompute support
- **Tiling**: VTCM tiling (DDR -> TCM), convolution tiling, HVX-optimized tiling
- **Vectorization**: HVX vector lowering (128-byte SIMD)
- **Multi-threading**: SCF threading, async threads, virtual thread formation
- **Memory**: Double buffering, buffer punting, DMA optimization
- **Matrix acceleration**: HexKL for HMX-based matrix multiply
- **Bufferization**: Tensor -> MemRef conversion

### 3. Target Translation

Two-phase lowering from MLIR to machine code:

1. **Linalg -> MLIR LLVM Dialect** (`translateLinalgToLLVMMLIR`): Runs the full optimization pipeline
2. **MLIR LLVM -> LLVM IR** (`translateHexagonMlirLlvmToLLVMIR`): Standard MLIR-to-LLVM translation with Hexagon target configuration
3. **LLVM IR -> Object Code** (`llvm_module_to_obj_string`): LLVM backend compilation targeting Hexagon ISA

### 4. Runtime System

The on-device runtime provides:

- **HexagonAPI / HexagonCAPI**: Buffer management and kernel invocation interface
- **HexKL API**: Hexagon Kernel Library for HMX matrix operations
- **UserDMA**: Direct Memory Access engine for efficient DDR <-> VTCM transfers
- **VTCMPool**: Tightly Coupled Memory allocator for low-latency scratch space
- **AsyncRuntime**: Thread pool with work-stealing for multi-threaded execution
- **HVX Intrinsics**: Vectorized Hexagon routines for common math operations

### 5. Execution Launchers

Three launchers handle different input workflows:

| Launcher | Input | Use Case |
|----------|-------|----------|
| `TritonHexagonLauncher` | Triton kernel `.o` bytes | Production Triton kernel execution with SPMD grid support |
| `TorchMLIRHexagonLauncher` | MLIR bytecode file | PyTorch model execution via torch-mlir |
| `MLIRHexagonLauncher` | MLIR file + C++ stub | Unit testing and debugging small MLIR examples |

All launchers follow the same steps:
1. Compile MLIR -> Object code (`.o`)
2. Generate C++ wrapper code for kernel launch
3. Link into shared library (`.so`) via `HexagonExecutor`
4. Push to device and execute
5. Collect output tensors

### 6. CLI Tools

| Tool | Description |
|------|-------------|
| `linalg-hexagon-opt` | `mlir-opt` variant with all Hexagon dialects and passes registered. Used for running individual passes on MLIR files. |
| `linalg-hexagon-translate` | Translates MLIR modules to Hexagon object code. |
| `hexagon-mlir-lsp-server` | Language Server Protocol support for IDE integration. |

## Key Hexagon Hardware Concepts

- **HVX (Hexagon Vector eXtensions)**: 128-byte SIMD vector units; the vectorization passes target these
- **HMX (Hexagon Matrix eXtensions)**: Hardware matrix multiply units; HexKL dialect provides the programming model
- **VTCM (Vector Tightly Coupled Memory)**: Low-latency on-chip scratchpad; VTCMTiling moves data from DDR to VTCM for compute
- **DMA**: Hardware DMA engine for async DDR <-> VTCM transfers; double buffering overlaps compute and data movement
- **Multi-threading**: NPU supports hardware threads; the compiler forms parallel loops mapped to a thread pool

## Data Flow Summary

```
Input (Triton/PyTorch)
       │
       ▼
  Linalg on Tensors (high-level)
       │
       ├── Fusion & Generalization
       ├── VTCM Tiling (DDR → VTCM tiles)
       ├── HVX Tiling (for vectorization)
       ├── Bufferization (Tensor → MemRef)
       ├── Double Buffering (DMA overlap)
       ├── Vectorization (HVX 128-byte)
       ├── [Optional] MatMul → HexKL (HMX)
       ├── [Optional] Multi-threading
       │
       ▼
  MLIR LLVM Dialect
       │
       ▼
  LLVM IR (Hexagon Target)
       │
       ▼
  Object Code (.o)
       │
       ▼
  Shared Library (.so)
       │
       ▼
  Execute on Hexagon NPU
```
