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

def CollapseAddressSpace :
    Pass<"collapse-address-space"> {
  let summary = "Reduce all address spaces to DDR";
  let description = [{
    This is needed for those address spaces (namely TCM)
    that did not get converted to hexagonmem allocs.
  }];

  let constructor = "hexagon::createCollapseAddressSpacePass()";
}

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
}

def FastInverse :
    Pass<"fast-inverse", "mlir::ModuleOp"> {
  let summary = "Pass to convert element-wise division to multiplication by reciprocal.";
  let constructor = "hexagon::createFastInversePass()";
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

def LowerTPtr :
    InterfacePass<"lower-tptr", "mlir::FunctionOpInterface"> {
  let summary = "Lower tptr ops to mlir dialect ops";
  let constructor = "hexagon::createLowerTPtrPass()";
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

    The wrapper allocates scratch * prod(grid) bytes total and passes each
    instance a non-overlapping slice of size `scratch` bytes.

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
    Replaces memref.alloc/hexagonmem.alloc operations in VTCM address space
    (space 1) with memref.view ops into a single VTCM buffer, enabling memory
    reuse through static offset assignment. Dealloc ops are removed only for
    allocations that this pass replaces.

    Two modes of operation:
    1. External VTCM mode: if a function argument is marked with the
       "hexagon.scratch" attribute, that argument is used as the VTCM
       buffer. The argument must be static-sized (memref<Nxi8, 1>); the
       pass validates at compile time that all kernel allocations fit
       within N bytes and fails if they do not.
    2. Fallback mode: if no such argument is present, a single hexagonmem.alloc
       sized to the exact required bytes is created at function entry, subject
       to the configured bufferSize budget cap (default 1 MB).

    In both modes, all eligible memref.alloc/hexagonmem.alloc ops in address
    space 1 are replaced with memref.view ops at statically computed aligned
    offsets.

    Restrictions (see MemoryOffsetsPass.cpp for details):
    - Only static, unconditional allocations are handled. Dynamic-sized allocs
      and allocs inside loops are left untouched.
    - All allocs are laid out simultaneously (no liveness analysis); allocs
      with non-overlapping lifetimes do not share memory.
  }];
  let options = [
    Option<"bufferSize", "buffer-size", "int64_t",
           /*default=*/"1048576",
           "Fallback-mode VTCM budget cap in bytes">,
    Option<"alignment", "alignment", "int64_t",
           /*default=*/"2048",
           "Byte alignment for buffer offsets in VTCM">,
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

#endif
