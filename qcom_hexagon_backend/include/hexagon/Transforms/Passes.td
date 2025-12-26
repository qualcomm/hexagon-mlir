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

def MatmulToHexKL :
    InterfacePass<"matmul-to-hexkl", "mlir::FunctionOpInterface"> {
  let summary = "Convert linalg.matmul to hexkl ops";
  let constructor = "hexagon::createMatmulToHexKLPass()";
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
  let options = [
    Option<"convTilingFactor", "conv-tile-size", "int",
           /*default=*/"-1",
           "override tile size. Only intended for debugging.">,
    Option<"convTileHeightDim", "conv-tile-height-dim", "bool",
           /*default=*/"false",
           "Only tile height dimension. Only intended for debugging">,
    Option<"convTileOCDim", "conv-tile-oc-dim", "bool",
           /*default=*/"false",
           "Only tile OC dim. Only intended for debugging.">

  ];
  let constructor = "hexagon::createConvTilingPass()";
}

def MemoryOffsets : Pass<"memory-offsets", "func::FuncOp"> {
  let summary = "Assigns offsets to buffers.";
  let description = [{
    Assigns offsets to buffers within a function to support memory
    reuse and layout optimization. The basic strategy is to replace multiple
    hexagonmem.alloc operations with views into a single statically allocated
    buffer, enabling memory reuse through offset assignment.
  }];
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

#endif
