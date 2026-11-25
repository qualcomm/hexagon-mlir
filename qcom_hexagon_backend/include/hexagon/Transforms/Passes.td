//===- Passes.td - Linalg lowering and optimization passes -*- tablegen -*-===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

#ifndef HEXAGON_TRANSFORMS_PASSES
#define HEXAGON_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"


def CopyCanonicalization :
    Pass<"copy-canonicalization", "func::FuncOp"> {
  let summary = "Eliminate intermediate buffer allocations in copy chains";
  let description = [{
    Canonicalizes memory copy operations by eliminating intermediate
    allocations that serve as passthroughs in write-once, read-many patterns.
  }];
  let constructor =
       "hexagon::createCopyCanonicalizationPass()";
}

def HexagonHoistLoopInvariantAlloc :
    Pass<"hexagon-hoist-loop-invariant-alloc", "func::FuncOp"> {
  let summary = "Hoist loop-invariant memref.alloc/alloca out of scf.for loops";
  let description = [{
    Hoists static-shape memref.alloc and memref.alloca ops out of an enclosing
    scf.for when the allocation does not escape its parent block. The
    allocation may sit directly in the for-body or inside an scf.if nested in
    the for. Complements upstream bufferization::BufferLoopHoisting by handling
    the if-nested case and stack allocations.
  }];
  let constructor = "hexagon::createHexagonHoistLoopInvariantAllocPass()";
}

def HexagonMaterializeStridedLoads :
    Pass<"hexagon-materialize-strided-loads", "func::FuncOp"> {
  let summary = "Materialize strided memref.reinterpret_cast feeding "
                "vector.transfer_read/transfer_write into a contiguous buffer";
  let description = [{
    For a 1-D static memref.reinterpret_cast with stride > 1 that feeds a
    vector.transfer_read and/or vector.transfer_write via a memref.subview,
    allocates a contiguous HVX-aligned buffer and rewires the subviews against
    it so the vector ops use unit-stride loads/stores instead of scalar
    gathers/scatters. A read use gets a one-shot copy-in (gather) from the
    strided source; a write use gets a one-shot drain copy-back (scatter) after
    the writers; a read-modify-write use gets both. A pure-write use skips the
    copy-in.
  }];
  let constructor = "hexagon::createHexagonMaterializeStridedLoadsPass()";
}

def ConvertLayout : Pass<"convert-layout", "func::FuncOp"> {
  let summary = "Convert hexagonmem::ConvertLayoutOp to linalg::GenericOp";
  let constructor = "hexagon::createConvertLayoutPass()";
}

def ConvertToHexagonmem :
     InterfacePass<"convert-to-hexagonmem", "mlir::FunctionOpInterface"> {
  let summary = "Lower memref ops to hexagonmem";
  let description = [{
    Converts vtcm allocs and copies, `memref.alloc memref<4x4096, 1>`
    to equivalent hexagonmem opps e.g. `hexagonmem.alloc .. `.
  }];
  let constructor = "hexagon::createConvertToHexagonmemPass()";
}

def ConvertZeroSizeMemref :
    Pass<"convert-zero-size-memref", "func::FuncOp"> {
    let summary = "Convert zero size memref to unit size.";
    let constructor = "hexagon::createConvertZeroSizeMemrefPass()";
}

def DecomposeTensorConcat : Pass<"triton-decompose-tensor-concat", "func::FuncOp"> {
  let summary = "Convert tensor.concat to other tensor ops";
  let constructor = "hexagon::createDecomposeTensorConcatPass()";
}

def EraseUnusedLinalgOperands :
     Pass<"erase-unused-linalg-operands", "mlir::ModuleOp"> {
   let summary = "Erase unused linalg operand and remove dead code.";
   let constructor = "hexagon::createEraseUnusedLinalgOperands()";
}

def ExpandBoolVec :
    InterfacePass<"expand-bool-vec", "mlir::FunctionOpInterface"> {
  let summary = "replace ui1_to_fp with select %cond %vec_1s %vec_0s";
  let constructor = "hexagon::createExpandBoolVecPass()";
}

def ExpandMathOps : Pass<"expand-math-ops", "func::FuncOp"> {
  let summary = "Expand complex math ops to simple math ops";
  let constructor = "hexagon::createExpandMathOpsPass()";
  let options = [
    Option<"expandUneOnVectors", "expand-une-on-vectors", "bool", "true",
           "Expand arith.cmpf une on vectors to integer bit ops; "
           "HVX vectors only support ordered float compares">,
  ];
}

def FastExpApprox : Pass<"fast-exp-approx", "func::FuncOp"> {
  let summary = "Replace math.exp / math.exp2 with vectorizable polynomial approximation";
  let description = [{
    Rewrites math.exp/math.exp2 on f32 to exp(x) = 2^k * 2^f via a minimax
    Horner polynomial for 2^f and an IEEE-754 bit-trick for 2^k. No clamps are
    inserted; callers must ensure inputs are in roughly [-87, 88].

    The degree option selects the polynomial used for 2^f on [-1, 0].
    Higher degrees reduce approximation error at the cost of more arith ops:
      degree 1: max_err ~3e-2   degree 4: max_err ~2e-6
      degree 2: max_err ~2e-3   degree 5: max_err ~1e-7
      degree 3: max_err ~6e-5   degree 6+: below f32 precision (~1e-7)
    Default is 3, matching the existing Qualcomm HVX softmax reference.
  }];
  let constructor = "hexagon::createFastExpApproxPass()";
  let options = [
    Option<"degree", "degree", "unsigned", "3",
           "Degree of the 2^f minimax polynomial (1..10)">,
  ];
}

def FastInverse :
    Pass<"fast-inverse", "mlir::ModuleOp"> {
  let summary = "Pass to convert element-wise division to multiplication by reciprocal.";
  let constructor = "hexagon::createFastInversePass()";
}

def BroadcastedReductionToButterfly :
    Pass<"broadcasted-reduction-to-butterfly", "mlir::ModuleOp"> {
  let summary = "Replace vector.reduction + (scalar arith chain) + vector.broadcast with a butterfly reduction.";
  let description = [{
    Rewrites `vector.reduction` → (optional single-use scalar arith/math chain)
    → `vector.broadcast` into an in-place butterfly: log2(N) shuffle+combine
    steps leave every lane holding the reduced value, and the scalar chain is
    vectorized lane-wise at native HVX width.
  }];
  let constructor = "hexagon::createBroadcastedReductionToButterflyPass()";
  let dependentDialects = ["arith::ArithDialect", "vector::VectorDialect"];
}

def HexagonAddFastMath :
    Pass<"hexagon-add-fastmath", "mlir::ModuleOp"> {
  let summary = "Add fast math flag to selected ops.";
  let constructor = "hexagon::createHexagonAddFastMathPass()";
}

def HexagonLLVMEnableHexagonRoutines :
    Pass<"llvm-intr-to-hexagon-routines", "mlir::ModuleOp"> {
  let summary = "Replace LLVM intrinsics with vectorized Hexagon routines.";
  let constructor = "hexagon::createHexagonLLVMEnableHexagonRoutinesPass()";
}

def HexagonLWPPass : Pass<"hexagon-lwp-instrumentation", "func::FuncOp"> {
  let summary = "Pass to instrument and profile with hexagon instrinsic";

  let options = [
    Option<"disableLWPLoop", "loop", "bool",
           /*default=*/"false",
           "Turn on or off loop level instrumentation">,
    Option<"LWPloopDepth", "depth", "int",
           /*default=*/"1",
           "Depth level instrumentation">,
   ];

  let constructor = "hexagon::createHexagonLWPPass()";
}

def HexagonPuntBuffer :
    InterfacePass<"hexagon-punt-buffer", "mlir::FunctionOpInterface"> {
  let summary = "erase copies and forward buffer directly to user";
  let constructor =
       "hexagon::createHexagonPuntBufferPass()";
}

def HexagonDoubleBufferGenericS1 :
    InterfacePass<"hexagon-double-buffer-generic-s1",
                  "mlir::FunctionOpInterface"> {
  let summary = "implement double buffering - Stage 1";

  let description = [{
    Implements sotfware pipelining using  double-buffering.
    It takes a tiled IR that originated from a fused
    linalg.generic and rewrites it as a software-pipelined
    loop using double-buffering mechanism.
    This is implemented in two stages:

      S1 : (this pass)
         This pass must run after one-shot-bufferization.
         The double buffers are introduced into the IR but
         data transfers are kept as `memref.copy` because
         de-allocator has not run yet.
      S2 : Please see HexagonDoubleBufferS2 pass.
  }];
  let constructor =
       "hexagon::createHexagonDoubleBufferGenericS1Pass()";
}

def HexagonDoubleBufferGenericS2 :
    InterfacePass<"hexagon-double-buffer-generic-s2",
                  "mlir::FunctionOpInterface"> {
  let summary = "implement double buffering - Stage 2";

  let description = [{
    Implements sotfware pipelining using  double-buffering.
    It has two stages.
      S1 : See HexagonDoubleBufferGenericS1 for Stage 1.

      S2 : (this pass)
         This stage runs after loop-hoisting and deallocation
         passes have run. In this stage  memref.dma_start
         and memref.dma_wait are introduced into the IR.
  }];
  let constructor =
       "hexagon::createHexagonDoubleBufferGenericS2Pass()";
}

def HexagonLowerTmTensor :
    InterfacePass<"hexagon-lower-tm-tensor", "mlir::FunctionOpInterface"> {
  let summary = "Lower lower tm-tensor ops to mlir dialect ops";
  let constructor = "hexagon::createHexagonLowerTmTensorPass()";
}

def GuardedOptimizeAllocationLiveness :
    Pass<"guarded-optimize-allocation-liveness", "func::FuncOp"> {
  let summary = "OptimizeAllocationLivenessPass, skipped on functions "
                "tagged hexagon.uses_libdevice";
  let constructor =
      "hexagon::createGuardedOptimizeAllocationLivenessPass()";
}

def HexagonReplaceWithLibraryCalls :
    Pass<"hexagon-replace-with-library-calls", "mlir::ModuleOp"> {
  let summary = "Replaces linalg ops with library_call attr set with "
                "library calls.";
  let constructor =
       "hexagon::createHexagonReplaceWithLibraryCallsPass()";
}

def HexagonRVO :
    InterfacePass<"hexagon-rvo", "mlir::FunctionOpInterface"> {
  let summary = "return value optimization";
  let description = [{
    This pass rewrites the graph so that instead of creating results
    in a temporary buffer and then copying to output buffer, instead
    the materialization of result occurs directly into output buffer;
  }];

  let constructor =
       "hexagon::createHexagonRVOPass()";
}

def HexagonVectorLowering :
    Pass<"hexagon-vector-lowering", "mlir::ModuleOp"> {
  let summary = "Pass to simplify vector multi-reduction and contraction.";
  let constructor = "hexagon::createHexagonVectorLoweringPass()";
}

def LinalgGeneralize :
    InterfacePass<"linalg-generalize", "mlir::FunctionOpInterface"> {
  let summary = "Pass to lower linalg named op to generic.";

  let constructor =
       "hexagon::createLinalgGeneralizePass()";
}

def LowerLibdevice :
    Pass<"lower-libdevice", "mlir::ModuleOp"> {
    let summary = "Convert tt.extern_elementwise libdevice ops to func.func";
	let constructor = "hexagon::createLowerLibdevicePass()";
}

def LowerTritonPrint :
    Pass<"lower-triton-print", "mlir::ModuleOp"> {
    let summary = "Lower tt.print (tl.device_print) to runtime print calls";
    let description = [{
      Lowers `triton::PrintOp` to calls into the hexagon runtime print shims
      (`hexagon_print_scalar_*` / `hexagon_print_tensor_*`). Tensor operands are
      bufferized and passed as a raw data pointer plus element count; the shim
      handles element iteration and formatting. The emitted symbol carries a
      `_dsp` suffix. The `hex` and `isSigned` print attributes are forwarded to
      the shim.
    }];
	let constructor = "hexagon::createLowerTritonPrintPass()";
    let options = [
      Option<"device_type", "device-type", "std::string",
             /*default=*/"\"hexagon\"",
             "Target device type for symbol selection (unused, reserved)">
    ];
}

def LegalizeFP8CastsToLLVMCall :
  Pass<"legalize-fp8-casts-to-llvm-call", "mlir::ModuleOp"> {
  let summary = "Lower fp8<->f16 cast fragments to explicit LLVM runtime calls";
  let description = [{
    Rewrites fp8<->f16 unrealized-cast lowering fragments into explicit
    LLVM runtime calls so the IR remains legal during LLVM lowering.

    This pass matches the pre-arith-to-llvm vector patterns on 128 lanes.
    For example, before:
    ```mlir
    %fp8 = builtin.unrealized_conversion_cast %in
      : vector<128xi8> to vector<128xf8E4M3FNUZ>
    %f16 = arith.extf %fp8 : vector<128xf8E4M3FNUZ> to vector<128xf16>
    ```
    after:
    ```mlir
    %f16 = llvm.call @convert_fp8_fp16_fn(%in)
      : (vector<128xi8>) -> vector<128xf16>
    ```

    And before:
    ```mlir
    %fp8 = arith.truncf %in : vector<128xf16> to vector<128xf8E4M3FNUZ>
    %i8 = builtin.unrealized_conversion_cast %fp8
      : vector<128xf8E4M3FNUZ> to vector<128xi8>
    ```
    after:
    ```mlir
    %i8 = llvm.call @convert_fp16_fp8_fn(%in)
      : (vector<128xf16>) -> vector<128xi8>
    ```

    The pass inserts the needed LLVM function declarations
    (`convert_fp8_fp16_fn{,uz}` and `convert_fp16_fp8_fn{,uz}`) in-module.
  }];
  let constructor = "hexagon::createLegalizeFP8CastsToLLVMCallPass()";
}

def LegalizeLateFP8CastsToLLVMCall :
  Pass<"legalize-late-fp8-casts-to-llvm-call", "mlir::ModuleOp"> {
  let summary = "Lower late fp8 cast fragments after arith-to-llvm";
  let description = [{
    Rewrites fp8 cast fragments that survive until after arith-to-llvm into
    explicit LLVM runtime calls.

    This pass handles both vector<128x...> and scalar forms.
    For example, before (vector path):
    ```mlir
    %fp8 = builtin.unrealized_conversion_cast %in
      : vector<128xi8> to vector<128xf8E4M3FNUZ>
    %f16 = arith.extf %fp8 : vector<128xf8E4M3FNUZ> to vector<128xf16>
    ```
    after:
    ```mlir
    %in_i32 = llvm.bitcast %in : vector<128xi8> to vector<32xi32>
    %out_i32 = llvm.call @convert_fp8_fp16_fnuz(%in_i32)
      : (vector<32xi32>) -> vector<64xi32>
    %f16 = llvm.bitcast %out_i32 : vector<64xi32> to vector<128xf16>
    ```

    And before (vector path):
    ```mlir
    %fp8 = arith.truncf %in : vector<128xf16> to vector<128xf8E4M3FNUZ>
    %i8 = builtin.unrealized_conversion_cast %fp8
      : vector<128xf8E4M3FNUZ> to vector<128xi8>
    ```
    after:
    ```mlir
    %in_i32 = llvm.bitcast %in : vector<128xf16> to vector<64xi32>
    %out_i32 = llvm.call @convert_fp16_fp8_fnuz(%in_i32)
      : (vector<64xi32>) -> vector<32xi32>
    %i8 = llvm.bitcast %out_i32 : vector<32xi32> to vector<128xi8>
    ```

    For f32 inputs it first normalizes to f16, and for scalar conversions it
    uses splat/extract lowering around the same vector runtime ABI.
    Supported fp8 element types are `f8E4M3FN` and `f8E4M3FNUZ`.
  }];
  let constructor = "hexagon::createLegalizeLateFP8CastsToLLVMCallPass()";
}

def LowerTPtr :
    InterfacePass<"lower-tptr", "mlir::FunctionOpInterface"> {
  let summary = "Lower tptr ops to mlir dialect ops";
  let constructor = "hexagon::createLowerTPtrPass()";
}

def FinalizeMemRefAndCorePtrToLLVMConversion :
    Pass<"finalize-memref-and-core-ptr-to-llvm-conversion",
         "mlir::ModuleOp"> {
  let summary = "Convert ptr and memref dialect ops to LLVM using a custom "
                "LLVMTypeConverter";
    let description = [{
      This pass is the final lowering step for pointer and memory operations. It
      converts all remaining `ptr` dialect ops (ptr.to_ptr, ptr.from_ptr,
      ptr.ptr_add, ptr.type_offset) and `memref` dialect ops (alloc, dealloc,
      load, store, global, etc.) into their LLVM dialect equivalents. It also
      handles memrefs whose element type is a ptr (e.g. memref<4x!ptr.ptr<...>>),
      which require both dialects to be lowered in the same conversion.

      When `enable-collapse-address-space` is true (the default), the pass
      additionally rewrites all non-zero LLVM address spaces (e.g. !llvm.ptr<1>
      for VTCM) down to address space 0, since the Hexagon LLVM backend does not
      distinguish address spaces. This must happen in the same pass because
      memref-to-LLVM conversion can emit ops like `llvm.call @free` whose operand
      carries the source memref's address space, which would be illegal against
      the space-0 declaration of @free if not immediately collapsed.

      Replaces the previously separate LowerCorePtr, FinalizeMemRefToLLVM, and
      CollapseAddressSpace passes that had fragile ordering dependencies.
    }];
  let constructor =
      "mlir::hexagon::createFinalizeMemRefAndCorePtrToLLVMConversionPass()";

  let dependentDialects = ["mlir::LLVM::LLVMDialect",
                           "mlir::ptr::PtrDialect"];

  let options = [
    Option<"enableCollapseAddressSpace", "enable-collapse-address-space",
           "bool", /*default=*/"true",
           "replace llvm.ptr<1> with llvm.ptr after ptr/memref-to-llvm conversion">,
  ];
}

def LowerTTX :
    InterfacePass<"lower-ttx", "mlir::FunctionOpInterface"> {
  let summary = "Lower ttx ops to mlir dialect ops";
  let constructor = "hexagon::createLowerTTXPass()";
}

def MatmulToConv : Pass<"matmul-to-conv", "func::FuncOp"> {
  let summary = "Convert linalg.matmul to linalg.conv2d ops";

  let constructor = "hexagon::createMatmulToConvPass()";
}

def RelaxFP32MatmulToFP16 : Pass<"relax-fp32-matmul-to-fp16", "func::FuncOp"> {
  let summary = "Relax fp32 matmul inputs to fp16 for HexKL lowering";
  let description = [{
    Rewrites fp32 linalg.matmul ops to use fp16 input tensors while preserving
    fp32 accumulation and output. The pass also inserts fp32-to-fp16 casts on
    the inputs before the matmul.
  }];

  let constructor = "hexagon::createRelaxFP32MatmulToFP16Pass()";
}

def MatmulToHexKL : InterfacePass<"matmul-to-hexkl", "mlir::FunctionOpInterface"> {
  let summary = "Convert linalg.matmul to hexkl.matmul";

  let description = [{
    Converts linalg.matmul operations with FP16 inputs to hexkl.matmul.

    Modes:
    - macro: High-level API with stricter requirements
      * Outputs f16 only (must upcast to f32 later if needed)
      * Strict dimension limits (M≤1600, N≤5120, K≤76030)
      * Memory allocation handled by macro API
      * Optimized DMA transfers and tiling strategy

    - micro(default): Low-level API with more flexibility
      * Can output f16 or f32 directly
      * Fine grained control over tiling and memory allocation
  }];

  let constructor = "mlir::hexagon::createMatmulToHexKLPass()";

  let options = [
    Option<"mode", "mode", "std::string",
           /*default=*/"\"micro\"",
           "HexKL API mode: 'micro' (default) or 'macro'">
  ];
}

def DecomposeHexKLMatmul :
    InterfacePass<"decompose-hexkl-matmul", "mlir::FunctionOpInterface"> {
  let summary = "Decompose hexkl.matmul into hexkl micro HMX operations";
  let description = [{
    Decomposes hexkl::MatmulOp into a sequence of hexkl micro HMX operations
    that perform tiled matrix multiplication. This pass implements the same
    tiling strategy as hexkl_matmul_f16f16_f32 in the runtime library.

    The decomposition includes:
    - VTCM allocation for tile buffers
    - Triple-nested loops for row/column/inner tiling
    - Activation and weight tile loading and layout conversion
    - HMX matrix multiplication with accumulation
    - Result readback and format conversion (f16 to f32)
  }];
  let constructor = "hexagon::createDecomposeHexKLMatmulPass()";
}

def DecomposeHexKLMatmulToDaemon :
    InterfacePass<"decompose-hexkl-matmul-to-daemon", "mlir::FunctionOpInterface"> {
  let summary = "Decompose hexkl.matmul into worker layout ops + HMX micro-daemon "
                "dispatch/wait for the dedicated-HMX-thread model";
  let description = [{
    Like DecomposeHexKLMatmul, but targets the mobile/DSP dedicated-HMX-thread
    (daemon) model.  Instead of emitting the inline accumulator loop
    (setup_acc_read / acc_clear / mm / acc_read) that every worker runs itself,
    this pass keeps only the worker-side layout ops (copy_submatrix, rm_to_ah,
    rm_to_wh, ah_to_rm, copy_f16_to_f32) and replaces the accumulator work with
    hexkl.micro_hmx_dispatch (one K-block matmul job) plus a per-output-tile
    hexkl.micro_hmx_wait_idle.  The accumulator-touching ops run on the daemon
    thread that owns HMX.

    Requires the enclosing func.func to carry hmx.thread_id_arg,
    hexkl_hmx.vtcm_base_arg and hmx.queue_arg (run InjectHMXThreadIdArgPass,
    InjectHMXVtcmBaseArgPass and InjectHMXQueueArgPass first).  The VTCM base is
    the un-sliced global base; this pass derives each worker's slice as
    `threadId * vtcmPerInstanceBytes` and globalizes every tile offset.

    Weight tiles use per-K-tile buffers so back-to-back dispatches in a single
    output tile's K-loop do not race on a shared weight buffer while the daemon
    is still reading an earlier job.
  }];
  let constructor = "hexagon::createDecomposeHexKLMatmulToDaemonPass()";
}

def ConvTiling : Pass<"conv-tiling", "func::FuncOp"> {
  let summary = "Tile linalg.conv2d ops";
  let description = [{
    Tiles Conv2D operations along specified dimensions.

    Assumes NHWC layout for input/output tensors:
    - Dimension 0: Batch (N)
    - Dimension 1: Height (H)
    - Dimension 2: Width (W)
    - Dimension 3: Channels (C)

    Example: --conv-tile-sizes="1:8,3:32"
    This tiles dimension 1 (height) with size 8 and dimension 3 (channels) with size 32.

    Constraints:
    - Height tile size (dim 1) must be a multiple of 8
    - Channel tile size (dim 3) must be a multiple of 32
  }];
  let options = [
    Option<"convTileSizes", "conv-tile-sizes", "std::string",
           /*default=*/"",
           "Tile sizes for conv2d dimensions in format: dim:size,dim:size,... ">
  ];
  let constructor = "hexagon::createConvTilingPass()";
}

def InsertScratchArg : Pass<"insert-scratch-arg", "func::FuncOp"> {
  let summary = "Inject per-instance VTCM scratch argument";
  let description = [{
    Adds a function argument typed as memref<Nxi8, 1> {hexagon.scratch} before
    the trailing Triton program-info arg pack (num_programs_x/y/z, pid_x/y/z),
    where N is the per-instance VTCM budget in bytes (the `scratch` option).

    The static size enables MemoryOffsetsPass to validate at compile time that
    all VTCM allocs fit within the per-instance budget.

    The wrapper allocates scratch * actual_threads bytes total and passes each
    active thread a non-overlapping slice of size `scratch` bytes. With grid
    oversubscription, a thread reuses its slice for multiple program IDs
    sequentially.

    Idempotent: if a hexagon.scratch arg already exists, no arg is inserted.
    Skips functions that do not have the trailing 6 i32 program-info ABI pack.
  }];
  let options = [
    Option<"scratch", "scratch", "int64_t",
           /*default=*/"0",
           "Per-instance VTCM scratch size in bytes (must be > 0)">
  ];
  let constructor = "hexagon::createInsertScratchArgPass()";
}

def MemoryOffsets : Pass<"memory-offsets", "func::FuncOp"> {
  let summary = "Assigns offsets to VTCM buffers.";
  let description = [{
    Replaces VTCM memref.alloc/hexagonmem.alloc ops with memref.view ops
    into a single VTCM buffer. Allocs with non-overlapping lifetimes share
    an offset.

    If a function argument is tagged "hexagon.scratch" (must be statically
    sized memref<Nxi8, 1>), that argument is the buffer and the pass errors
    if allocs don't fit. Otherwise a hexagonmem.alloc sized to the packed
    total is created at entry, capped by bufferSize.

    Run optimize-allocation-liveness before this pass.
  }];
  let options = [
    Option<"bufferSize", "buffer-size", "int64_t",
           /*default=*/"1048576",
           "Fallback-mode VTCM budget cap in bytes">,
    Option<"alignment", "alignment", "int64_t",
           /*default=*/"2048",
           "Byte alignment for the base VTCM buffer">,
  ];
  let constructor = "hexagon::createMemoryOffsetsPass()";
}

def RemoveMLProgramPass :
    Pass<"remove-ml-program", "mlir::ModuleOp"> {
    let summary = "Remove ml_program operations from the IR";
	let constructor = "hexagon::removeMLProgramPass()";
}

def ReduceContractionRankPass :
  Pass<"reduce-contraction-rank", "mlir::func::FuncOp"> {
  let summary = "Reduce rank of named contraction ops with unit dims";
  let description = [{
    Converts batch_matmul with batch=1 to matmul, and matvec with unit dims to dot.
  }];
  let constructor = "hexagon::createReduceContractionRankPass()";
}


def FoldCastsIntoMatmul : Pass<"fold-casts-into-matmul", "func::FuncOp"> {
  let summary = "Fold FP16 to FP32 casts into linalg.matmul operations";
  let description = [{
    This pass identifies linalg.matmul operations with FP32 inputs that come
    from FP16→FP32 extension casts and folds the casts into the matmul,
    resulting in a matmul with FP16 inputs and FP32 output.

    This enables efficient mixed-precision computation where the matmul can
    be performed in FP16 with FP32 accumulation, avoiding unnecessary type
    conversions.
  }];
  let constructor = "mlir::hexagon::createFoldCastsIntoMatmulPass()";
  let dependentDialects = ["linalg::LinalgDialect", "arith::ArithDialect",
                           "tensor::TensorDialect"];
}

def VectorAccumulatorForReduction
    : Pass<"hexagon-vector-accumulator-reduction", "func::FuncOp"> {
  let summary = "Replace per-tile scalar accumulator in tiled reductions with "
                "a vector accumulator and a single final horizontal reduction";
  let description = [{
    After HexagonTilingPass, reduction linalg.generics are tiled into nested
    scf.for loops where the inner body performs a horizontal reduction at each
    tile step (e.g. 16 reductions for a 512-wide reduction with tile size 32).
    This pass replaces the per-tile horizontal reduction with an elementwise
    accumulation into a vector register, followed by a single horizontal
    reduction after the inner loop, eliminating ~15x the serialising reductions.
  }];
  let constructor = "mlir::hexagon::createVectorAccumulatorForReductionPass()";
  let dependentDialects = ["linalg::LinalgDialect", "scf::SCFDialect",
                           "tensor::TensorDialect", "arith::ArithDialect"];
}

def HoistScalarOps : Pass<"hoist-scalar-ops-in-linalg-generic", "func::FuncOp"> {
  let summary = "Hoist scalar arith/math ops out of linalg.generic regions";
  let description = [{
    This pass identifies loop-invariant scalar arithmetic and math operations
    within linalg.generic regions and hoists them outside the operation.
    The hoisted scalar values are then passed back into the generic operation
    as additional inputs.
  }];
  let constructor = "mlir::hexagon::createHoistScalarOpsPass()";
}

def OutlineLinalgGeneric : Pass<"outline-linalg-generic", "mlir::ModuleOp"> {
  let summary = "Outline every linalg.generic into its own noinline func.func";
  let description = [{
    Outlines every `linalg.generic` op into its own private `func.func` and
    replaces the original op with a `func.call`. This helps reduce llvm obj
    and so sizes.
  }];
  let constructor = "hexagon::createOutlineLinalgGenericPass()";
  let dependentDialects = ["func::FuncDialect", "linalg::LinalgDialect",
                           "LLVM::LLVMDialect"];
}

def HexagonDeallocAfterLastUse :
  Pass<"hexagon-dealloc-after-last-use", "mlir::ModuleOp"> {
  let summary = "Place memref and hexagonmem deallocs after the last use";
  let description = [{
    Standalone deallocation placer for buffers whose lifetime crosses outlined
    function calls or uses VTCM/DMA-specific operations.

    Static memref.alloc buffers at or above chunkBytes, VTCM memref.alloc
    buffers, and static hexagonmem.alloc buffers are tracked. The pass
    forward-traces every alias through view ops (memref.subview,
    expand_shape, collapse_shape, cast, and reinterpret_cast) and
    interprocedurally through calls that pass an argument straight to a return.
    The last program point that touches any alias is located and the matching
    memref.dealloc or hexagonmem.dealloc is inserted or moved immediately
    after it.

    A memref.dma_start source or destination remains live through the first
    matching memref.dma_wait on the same tag. If no completion can be proven,
    the pass leaves the existing lifetime unchanged.

    Aggressive: when the buffer's fate past a call cannot be proven a
    pass-through, it is assumed dead after that call. The only hard exclusion
    is a buffer that escapes via the entry function's own return, which is
    never deallocated.

    Intended to run after ConvertToHexagonmem and DMA insertion when the
    standard buffer-deallocation pipeline is disabled.
  }];
  let constructor = "hexagon::createHexagonDeallocAfterLastUsePass()";
  let dependentDialects = ["func::FuncDialect", "memref::MemRefDialect",
                            "hexagonmem::HexagonMemDialect"];
  let options = [
    Option<"chunkBytes", "chunk-bytes", "int64_t",
           /*default=*/"0",
           "Byte threshold at or above which non-VTCM memref.alloc roots are "
           "tracked (0 = disabled); VTCM roots are always tracked.">
  ];
}


def FoldMulFByZeroPass : Pass<"fold-mulf-by-zero"> {
  let summary = "Add pass to fold x * 0 => 0 when fast math is enabled.";
  let description = [{
    Peephole optimization to fold x * 0 => 0 when appropriate fast math flags
    are enabled.
  }];
  let constructor = "mlir::hexagon::createFoldMulFByZeroPass()";
}

def FoldResourceTranspose : Pass<"fold-resource-transpose"> {
  let summary = "Fold transpose of f16 dense resource constants";
  let description = [{
    Pass to fold linalg.transpose ops acting on dense_resource<f16> constants
    into a new pre-transposed constant. Used to avoid runtime transpose overhead
    for static weights.
  }];
  let constructor = "mlir::hexagon::createFoldResourceTransposePass()";
  let dependentDialects = ["linalg::LinalgDialect", "arith::ArithDialect",
                           "tensor::TensorDialect"];
}

def ScheduleMatmulForHVX :
    InterfacePass<"schedule-matmul-for-hvx", "mlir::FunctionOpInterface"> {
  let summary = "Pass to schedule matmul and its variants";

  let constructor =
       "hexagon::createScheduleMatmulForHVXPass()";
}

def PreprocessTiledConv2D :
  InterfacePass<"preprocess-tiled-conv2d", "mlir::FunctionOpInterface"> {
  let summary = "Preprocess tiled linalg.conv2d for hmx.conv2d conversion";
  let constructor = "hexagon::createPreprocessTiledConv2DPass()";
}

def SeedLayoutConversions :
  InterfacePass<"seed-layout-conversions", "mlir::FunctionOpInterface"> {
  let summary = "wrap conv2d ops with packs and conversion cast";
  let description = [{
    Wrap hmx applicable conv2d ops with pack/unpack ops.
    To ensure it is valid IR after pack insertions, this pass also inserts
    conversion casts around the conv2d.
  For example, given :
  ```mlir
     %0 = linalg.conv_2d_nhwc_fhwc
                    ins(%in1, %in2 : tensor<1x15x7x59xf16>,  tensor<59x1x1x59xf16>)
                     outs(%result: tensor<1x15x7x59xf16>) -> tensor<1x15x7x59xf16>
  after running the pass:

  %cst = arith.constant dense<[2, 32, 1, 1, 2, 16, 2]> : tensor<7xindex>
  %cst_0 = arith.constant 0.000000e+00 : f16

  // pack_1 of activation
  %0 = tensor.empty() : tensor<1x2x2x2x8x4x32xf16>
  %pack = linalg.pack %arg0 padding_value(%cst_0 : f16)
            inner_dims_pos = [1, 2, 3] inner_tiles = [8, 4, 32]
            into %0 : tensor<1x15x7x59xf16> -> tensor<1x2x2x2x8x4x32xf16>

  // pack_2 of activation
  %1 = tensor.empty() : tensor<1x2x2x2x8x2x32x2xf16>
  %pack_1 = linalg.pack %pack
            inner_dims_pos = [5] inner_tiles = [2]
            into %1 : tensor<1x2x2x2x8x4x32xf16> -> tensor<1x2x2x2x8x2x32x2xf16>

  // wrap into conversion_cast
  %2 = builtin.unrealized_conversion_cast %pack_1
           : tensor<1x2x2x2x8x2x32x2xf16> to tensor<1x15x7x59xf16>

  // pad, reshape and transpose filter
  %padded = tensor.pad %arg1 low[0, 0, 0, 0] high[5, 0, 0, 5] {
  ^bb0(%arg3: index, %arg4: index, %arg5: index, %arg6: index):
    tensor.yield %cst_0 : f16
  } : tensor<59x1x1x59xf16> to tensor<64x1x1x64xf16>
  %reshape = tensor.reshape %padded(%cst)
           : (tensor<64x1x1x64xf16>, tensor<7xindex>) -> tensor<2x32x1x1x2x16x2xf16>
  %3 = tensor.empty() : tensor<2x2x1x1x16x32x2xf16>
  %transposed = linalg.transpose
           ins(%reshape : tensor<2x32x1x1x2x16x2xf16>)
           outs(%3 : tensor<2x2x1x1x16x32x2xf16>) permutation = [0, 4, 2, 3, 5, 1, 6]

  %4 = builtin.unrealized_conversion_cast %transposed : tensor<2x2x1x1x16x32x2xf16> to tensor<59x1x1x59xf16>
  %5 = tensor.empty() : tensor<1x15x7x59xf16>
  %6 = linalg.conv_2d_nhwc_fhwc ins(%2, %4 : tensor<1x15x7x59xf16>, tensor<59x1x1x59xf16>) outs(%5 : tensor<1x15x7x59xf16>) -> tensor<1x15x7x59xf16>
  %7 = builtin.unrealized_conversion_cast %6 : tensor<1x15x7x59xf16> to tensor<1x2x2x2x8x2x32x2xf16>

  // unpack
  %unpack = linalg.unpack %7 inner_dims_pos = [5] inner_tiles = [2] into %0 : tensor<1x2x2x2x8x2x32x2xf16> -> tensor<1x2x2x2x8x4x32xf16>
  %unpack_2 = linalg.unpack %unpack inner_dims_pos = [1, 2, 3] inner_tiles = [8, 4, 32] into %5 : tensor<1x2x2x2x8x4x32xf16> -> tensor<1x15x7x59xf16>
  }];

  let constructor =
       "hexagon::createSeedLayoutConversionsPass()";

}

def LowerHexKLMatmulToMacro : InterfacePass<"lower-hexkl-matmul-to-macro", "mlir::FunctionOpInterface"> {
  let summary = "Lower hexkl.matmul to optimized hexkl_macro_mm_f16 call";
  let description = [{
    This pass lowers hexkl::MatmulOp to a single call to the optimized
    hexkl_macro_mm_f16 library function, handling layout transformations
    for activations (RM→AH) and output (AH→RM). Assumes weights are
    already in WH layout (pre-transformed by PreprocessWeightsForHMX pass).
  }];
  let constructor = "mlir::hexagon::createLowerHexKLMatmulToMacroPass()";
}

def PreprocessWeightsForHMX :   InterfacePass<"preprocess-weights-hmx", "mlir::FunctionOpInterface"> {
  let summary = "Transform constant weights to HMX WH layout at compile time";
  let description = [{
    This pass identifies constant weight tensors used as RHS in matmul operations
    and transforms them from row-major to WH (HMX weight) layout by calling
    hexkl_macro_rm_to_wh_f16_inplace at compile time.
  }];
  let constructor = "mlir::hexagon::createPreprocessWeightsForHMXPass()";
}

def ForceHVXCrouton :
    InterfacePass<"force-hvx-crouton", "mlir::FunctionOpInterface"> {
  let summary = "Force HVX Crouton mode by annotating functions";
  let constructor = "hexagon::createForceHVXCroutonPass()";
}

def SmallExponentToMultiply :
    Pass<"small-exponent-to-multiply", "mlir::ModuleOp"> {
    let summary = "Convert pow with small integer exponent to multiplications.";
    let constructor = "hexagon::createSmallExponentToMultiplyPass()";
    let options = [
      Option<"maxExponent", "max-exponent", "int",
             /*default=*/"4",
             "Maximum exponent to convert to multiplications.">,
      Option<"allowNegative", "allow-negative", "bool",
             /*default=*/"true",
             "Allow conversion of negative exponents.">,
    ];
}

def StaticAllocToAlloca :
    Pass<"static-alloc-to-alloca", "func::FuncOp"> {
  let summary = "Replace small static memref.alloc ops with memref.alloca";
  let description = [{
    Converts static-shaped memref.alloc operations in address space 0 to
    memref.alloca, erasing the corresponding memref.dealloc ops. Eliminates
    heap allocation for small temporaries such as scalar reduction accumulators
    and DMA wait handles.

    Allocs are accepted greedily in ascending effective-size order (where
    effective size accounts for alignment padding). The acceptance gate is a
    liveness-aware peak check: after tentatively adding each candidate, the
    peak simultaneous live bytes across all accepted allocs is computed using
    MLIR liveness analysis. If the peak would exceed maxTotalAllocaSizeBytes
    the candidate is rejected. All generated allocas are hoisted to the
    function entry block regardless of where the original alloc appeared.

    Using peak live bytes rather than a running sum means that non-overlapping
    allocs (e.g. sequential DMA wait handles in a loop) do not consume budget
    from each other, while simultaneously-live allocs (e.g. multiple index
    compute buffers) are correctly counted together. This prevents stack
    overflow on targets with a limited stack without requiring a separate
    per-alloc size cap.
  }];
  let options = [
    Option<"maxTotalAllocaSizeBytes", "max-total-alloca-size-bytes", "int64_t",
           /*default=*/"2048",
           "Peak simultaneous live alloca bytes allowed per function. Computed "
           "via liveness analysis so non-overlapping allocs share budget. "
           "Conservative fraction of the stack limit for different NSP.">
  ];
  let constructor = "hexagon::createStaticAllocToAllocaPass()";
}

def FoldPackUnpackConstants :
    Pass<"fold-pack-unpack-constants"> {
  let summary = "Fold pack/unpack/transpose/reshape/pad operations on constant tensors";
  let description = [{
    This pass performs constant folding for tensor operations:
    1. Evaluates pack/unpack operations on constant tensors at compile time
    2. Evaluates transpose operations on constant tensors at compile time
    3. Evaluates expand_shape/collapse_shape operations on constant tensors
    4. Evaluates reshape operations on constant tensors at compile time
    5. Evaluates pad operations on constant tensors at compile time
    6. Replaces the operations with the computed constant result

    This eliminates runtime overhead for transforming constant weights
    and other constant tensors.

    Example:
      %cst = arith.constant dense<...> : tensor<768xf16>
      %packed = linalg.pack %cst inner_dims_pos = [0] inner_tiles = [32]
                into %dest : tensor<768xf16> -> tensor<24x32xf16>

    Becomes:
      %packed = arith.constant dense<...> : tensor<24x32xf16>
  }];
  let constructor = "hexagon::createFoldPackUnpackConstantsPass()";
}

def EliminateRedundantUnpackPack :
    Pass<"eliminate-redundant-unpack-pack"> {
  let summary = "Eliminate redundant unpack/pack pairs";
  let description = [{
    This pass eliminates redundant unpack/pack pairs when:
    1. An unpack operation's result is consumed by a pack operation
    2. Both operations have the same inner_dims_pos and inner_tiles
    3. Both operations have the same outer_dims_perm
    4. The tensor type of unpack's source matches pack's destination type

    Pattern:
      %unpacked = linalg.unpack %packed inner_dims_pos = [...] inner_tiles = [...]
      %repacked = linalg.pack %unpacked padding_value(%pad) inner_dims_pos = [...]
                  inner_tiles = [...]
    
    The tensor type check (unpack's source type == pack's dest type) is a
    NECESSARY but NOT SUFFICIENT condition for correctness. It ensures structural
    compatibility but does NOT guarantee that the padding values added by pack
    match the values discarded by unpack.

    IMPORTANT LIMITATION:
    This pass currently does NOT verify that the padding value used by pack
    matches the values being discarded by unpack. This verification requires
    data flow analysis to prove that the padded values discarded by unpack
    are the same as the pack's padding value.

    Users of this pass must ensure this property holds in their use case, or
    accept that the optimization may be unsound in general. Future improvements
    will add data flow analysis to automatically verify this property.
  }];
  let constructor = "hexagon::createEliminateRedundantUnpackPackPass()";
}

def DivToMulOptimization : Pass<"div-to-mul-optimization", "func::FuncOp"> {
  let summary = "Optimize division by broadcasted values to multiplication";
  let description = [{
    This pass optimizes division operations in linalg.generic when the divisor
    is previously broadcasted (via linalg.fill or linalg.generic broadcast),
    by converting them to multiplications with the reciprocal.

    Transforms:
    - x / fill(y) -> x * fill(1/y)
    - x / broadcast(y) -> x * broadcast(1/y)

    This optimization is beneficial as multiplication is typically faster than
    division on hardware accelerators.
  }];
  let constructor = "hexagon::createDivToMulOptimizationPass()";
}

def SCFLoopUnroll : Pass<"scf-loop-unroll", "func::FuncOp"> {
  let summary = "Unroll SCF for loops with constant bounds";
  let description = [{
    This pass unrolls the innermost scf.for loops which satisfy
    checks and constraints. This is complementary to LLVM level
    loop unrolling. The pass creates a new loop with a larger
    step size (original_step * unroll_factor)
    and clones the loop body unroll_factor times.
  }];
  let options = [
    Option<"unrollFactor", "unroll-factor", "int64_t",
           /*default=*/"2",
           "Factor by which to unroll loops">
  ];
  let constructor = "hexagon::createSCFLoopUnrollPass()";
}

def HexagonUnrollAndJam : Pass<"hexagon-unroll-and-jam", "func::FuncOp"> {
  let summary = "Unroll-and-jam outer reduction loops to expose ILP";
  let description = [{
    This pass unroll-and-jams the outermost qualifying scf.for loop in each
    loop nest. Unlike scf-loop-unroll (which targets innermost loops and simply
    clones the body), unroll-and-jam scales the outer loop step by the factor
    and *interleaves* the factor copies of the body. When the body contains a
    reduction, each unrolled iteration becomes an independent reduction chain,
    so the in-order VLIW scheduler can overlap their latencies for more ILP.

    The mechanical jam is delegated to upstream `mlir::loopUnrollJamByFactor`.
    Because that utility synthesizes its scaled step and shifted induction
    variable as `index`-typed constants, this pass first normalizes an
    integer-typed loop induction variable to `index` (re-casting to the
    original type inside the body), so triton-shared's i32 row loops can be
    jammed without hitting a type mismatch.

    Loops are selected by a single, self-contained heuristic
    (`chooseUnrollJamFactor`): the loop must have no results/iter_args, a
    constant trip count divisible by the factor and at least `min-trip-count`,
    inner-loop bounds invariant in the outer loop, a body containing a
    reduction, a body small enough that the jammed copies stay within
    `max-body-ops`, and a reduction row length within `max-reduction-row-length`
    (the win from jamming fades as the per-row elementwise stream grows).
  }];
  let options = [
    Option<"unrollJamFactor", "unroll-jam-factor", "int64_t",
           /*default=*/"2",
           "Factor by which to unroll-and-jam qualifying loops">,
    Option<"minTripCount", "min-trip-count", "int64_t",
           /*default=*/"4",
           "Minimum constant trip count for a loop to be jammed">,
    Option<"maxBodyOps", "max-body-ops", "int64_t",
           /*default=*/"512",
           "Skip loops whose body op count times the factor exceeds this "
           "(register-pressure guard)">,
    Option<"maxReductionRowLength", "max-reduction-row-length", "int64_t",
           /*default=*/"1024",
           "Skip loops whose reduction row length exceeds this (past this the "
           "elementwise stream dominates and jamming stops paying off)">
  ];
  let constructor = "hexagon::createHexagonUnrollAndJamPass()";
}

def LLVMAlignmentPropagation :
    Pass<"llvm-alignment-propagation", "mlir::ModuleOp"> {
  let summary = "Propagate alignment from memref operations through LLVM conversion chain";
  let description = [{
    Propagates alignment information from memref.reinterpret_cast operations
    through the LLVM conversion chain to llvm.load and llvm.store operations.
  }];
  let options = [
    Option<"setMemrefAllocAlignment", "set-memref-alloc-alignment", "bool",
           /*default=*/"false",
           "Set alignment to 128 for all memref.alloc operations">,
  ];
  let constructor = "hexagon::createLLVMAlignmentPropagationPass()";
}

def SetAliasScopes : Pass<"set-alias-scopes", "func::FuncOp"> {
  let summary = "Emit alias-scope metadata on LLVM-dialect memory ops";
  let description = [{
    For each llvm.load/llvm.store, traces the pointer operand backward through
    the LLVM conversion chain to find the root buffer (a function argument or
    alloca). Creates one alias scope per distinct root and stamps alias_scopes /
    noalias_scopes on every memory op, enabling LLVM's alias analysis to prove
    non-aliasing between different buffers.
  }];
  let constructor = "hexagon::createSetAliasScopesPass()";
}

def SpecializeTranscendentals : Pass<"specialize-transcendentals", "func::FuncOp"> {
  let summary = "Identify transcendentals e.g. tanh within expanded IR";
  let description = [{
    This pass identifies transcendental math functions in their expanded
    form in the IR, and replaces them with the math form so that later
    passes can replace them easily with custom library implementation.

    For instance, tanh(x) = (e^x - e^-x)/(e^x + e^-x), or more often
                            (1 - exp(-2*x)) / (1 + exp(-2*x)).
  }];
  let constructor = "hexagon::createSpecializeTranscendentalsPass()";
}

def InjectHMXThreadIdArg : Pass<"inject-hmx-thread-id-arg", "func::FuncOp"> {
  let summary = "Inject a trailing i32 thread-ID argument into every func.func";
  let description = [{
    Adds a trailing i32 thread-ID function argument (annotated with the
    hmx.thread_id_arg function attribute) to every func.func.  This argument
    is subsequently read by DecomposeHexKLMatmulToDaemonPass to route per-thread
    HMX operations.

    The pass is idempotent: re-running on an already-annotated function is a
    no-op.

    Must run before DecomposeHexKLMatmulToDaemonPass.
  }];
  let constructor = "mlir::hexagon::createInjectHMXThreadIdArgPass()";
}

def InjectHMXQueueArg : Pass<"inject-hmx-queue-arg", "func::FuncOp"> {
  let summary = "Inject a trailing !llvm.ptr HMX-queue argument into every func.func";
  let description = [{
    Adds a trailing !llvm.ptr argument (annotated with the hmx.queue_arg
    function attribute) to every func.func.  In the HMX-daemon model the
    wrapper allocates one HMXQueue per kernel invocation and forwards the
    pointer through this argument; DecomposeHexKLMatmulToDaemonPass uses it to
    emit HMX dispatch / wait_idle calls.

    The pass is idempotent: re-running on an already-annotated function is a
    no-op.

    Pass-order convention: runs AFTER InjectHMXThreadIdArg, so the resulting
    layout is `(..., %tid: i32, %queue: !llvm.ptr)`.
  }];
  let constructor = "mlir::hexagon::createInjectHMXQueueArgPass()";
}

def InjectHMXVtcmBaseArg : Pass<"inject-hmx-vtcm-base-arg", "func::FuncOp"> {
  let summary = "Inject a trailing memref<?xi8, 1> VTCM-base argument into "
                "every func.func";
  let description = [{
    Adds a trailing memref<?xi8, 1> argument (annotated with the
    hexkl_hmx.vtcm_base_arg function attribute) to every func.func.  In the HMX
    micro-daemon model the wrapper acquires one shared VTCM region and forwards
    its un-sliced base through this argument; DecomposeHexKLMatmulToDaemonPass
    derives each worker's slice from it as `threadId * vtcmPerInstanceBytes`.

    The element type and address space (i8, space 1) mirror the alloc type used
    by DecomposeHexKLMatmulPass so both paths agree on the VTCM memref type.

    The pass is idempotent: re-running on an already-annotated function is a
    no-op.

    Pass-order convention: runs AFTER InjectHMXThreadIdArg and BEFORE
    InjectHMXQueueArg, so the resulting layout is
    `(..., %tid: i32, %vtcm_base: memref<?xi8, 1>, %queue: !llvm.ptr)`.
  }];
  let constructor = "mlir::hexagon::createInjectHMXVtcmBaseArgPass()";
}

#endif
