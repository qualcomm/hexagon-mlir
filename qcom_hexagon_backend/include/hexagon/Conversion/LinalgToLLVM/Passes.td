//===- Passes.td - Linalg to LLVM conversion passes  -------*- tablegen -*-===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

#ifndef HEXAGON_LINALG_TO_LLVM_CONVERSION_PASSES
#define HEXAGON_LINALG_TO_LLVM_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

// -- The main linalg-to-llvm pass-pipeline --
def LinalgToLLVM : Pass<"linalg-to-llvm", "mlir::ModuleOp"> {
  let summary = "Convert Linalg to LLVM";
  let constructor = "hexagon::createLinalgToLLVMPass()";
  let options = [
    Option<"puntBuffer", "punt-buffer", "bool",
           /*default=*/"true",
           "Punt input kernel buffer pointer directly to user instead"
           " of making local copy first.">,
    Option<"returnValueOptimization", "rvo", "bool",
           /*default=*/"true",
           "Rewrites the IR so that it is easier for one-shot bufferizer and"
           " eliminate empty-tensor to fold away unnecessary copying from"
           " local buffer to output buffer">,
    Option<"fusion", "fusion", "bool",
           /*default=*/"true",
           "Attempt to fuse linalg generic ops">,
    Option<"fusionAllowRecompute", "far", "bool",
           /*default=*/"false",
           "Allow recomputing values when fusing linalg generic ops">,
    Option<"fusionDoMultiUse", "fdmu", "bool",
           /*default=*/"true",
           "Do multi-use fusion">,
    Option<"useInterchangeVector", "use-interchange-vector", "bool",
           /*default=*/"false",
           "Use interchange vector in the HexagonTiling pass"
           " if default inner loop is not conducive for vectorization.">,
    Option<"enableDoubleBuffering", "enable-double-buffering", "bool",
          /*default=*/"false",
          "Enable double buffering optimization">,
    Option<"enableSCFThreading", "enable-scf-threading", "bool",
          /*default=*/"false",
          "Enable scf-level multi-threading">,
    Option<"enableMultiThreading", "enable-multi-threading", "bool",
          /*default=*/"false",
          "Enable asnyc multi-threading">,
    Option<"addFastMath", "add-fastmath", "bool",
           /*default=*/"true",
           "Add Fast Math Flag to HexagonAddFastMath Pass.">,
    Option<"splitTilingRange", "split-tiling-range", "bool",
          /*default=*/"true",
          "Configure HexagonTiling pass to do op splitting, "
          "instead of padding, when dealing with loops "
          "which are not  perfectly tileable.">,
    Option<"enableSplitReduction", "enable-split-reduction", "bool",
           /*default=*/"true",
           "Configure HexagonTiling pass to enable split reduction, "
           " for reduction loops. ">,
    Option<"enableSlicing", "enable-slicing", "bool",
           /*default=*/"false",
           "Flag to enable/disable operator slicing pass">,

    Option<"slicingFactor", "slicing-factor", "int",
           /*default=*/"1",
           "Operator slicing factor.">,
    Option<"expandBoolVec", "expand-bool-vec", "bool",
           /*default=*/"true",
           "replace ui1_to_fp with select %cond %vec_1s %vec_0s">,
    Option<"enableHexagonRoutines", "enable-hexagon-routines", "bool",
          /*default=*/"true",
          "Replaces LLVM intrisics with vectorized Hexagon routines.">,
    Option<"enableVTCMTiling", "enable-vtcm-tiling", "bool",
          /*default=*/"true",
          "enable lowering linalg-on-tensors (DDR) to tiled-compute on VTCM">,
    Option<"tileSizes", "tile-sizes-override", "std::string",
           /*default=*/"",
           "override tile size.">,
    Option<"enableConvertToHexagonmem", "enable-convert-to-hexagonmem", "bool",
          /*default=*/"true",
          "enable conversion from memref to hexagonmem ops">,
    Option<"enableHexagonmemCopyToDMA", "enable-hexagonmem-copy-to-dma", "bool",
          /*default=*/"false",
          "enable conversion from hexagonmem copy to dma ops">,
    Option<"enableHexKL", "enable-hexkl", "bool",
          /*default=*/"false",
          "Enable linalg matmul to hexkl dialect lowering">,
    Option<"enableCollapseAddressSpace", "enable-collapse-address-space", "bool",
          /*default=*/"true",
          "replace llvm.ptr<1> with llvm.ptr at last stage of convert-to-llvm">,
    Option<"lowerConstantsInSeparateSharedObjects", "lower-constants-in-separate-shared-objects", "bool",
          /*default=*/"false",
          "Lowers the constant tensors separately in their own MLIR modules, which then get compiled independently into their own .o, and then independently into various shared objects (.so)">,
    Option<"enableBufferization", "enable-bufferization", "bool",
          /*default=*/"true",
          "enable bufferization passes; set to false for manual buffer management">,
    Option<"enableConvTiling", "enable-conv-tiling", "bool",
          /*default=*/"false",
          "Tile the Convolution Op">,
    Option<"convTilingFactor", "conv-tile-size", "int",
          /*default=*/"-1",
          "override tile size. Only intended for debugging.">,
    Option<"convTileHeightDim", "conv-tile-height-dim", "bool",
          /*default=*/"false",
          "Only tile height dimension. Only intended for debugging">,
    Option<"convTileOCDim", "conv-tile-oc-dim", "bool",
           /*default=*/"false",
           "Only tile OC dim. Only intended for debugging.">,
    Option<"enableLWP", "enable-lwp", "bool",
           /*default=*/"false",
           "Flag to enable/disable lwp instrumentation">,
    Option<"disableLWPLoop", "loop", "bool",
           /*default=*/"false",
           "Turn on or off loop level instrumentation">,
    Option<"LWPloopDepth", "depth", "int",
           /*default=*/"1",
           "Depth level instrumentation">,
    Option<"extendPackUpperFrontier", "upper", "bool",
           /*default=*/"true",
           "Look for opportunities to extend linalg.pack frontiers.">,
    Option<"extendPackLowerFrontier", "lower", "bool",
           /*default=*/"true",
           "Look for opportunities to extend linalg.unpack frontiers.">,
    Option<"extendPackParallelsOnly", "parallels-only", "bool",
           /*default=*/"true",
           "Allow un-staging linalg ops that resolve to reduction "
           "generics even when it is possibly to pack those. This impacts "
           "both upper and lower frontier">,
    Option<"enableVectorization", "enable-vectorization", "bool",
          /*default=*/"true",
          "Enable vectorization for HVX">
   ];
}

// --- Supporting passes ---

def FormAsyncThreads : Pass<"form-async-threads", "func::FuncOp"> {
  let summary = "Rewrite scf::forall as async threads";
  let constructor = "hexagon::createFormAsyncThreadsPass()";
}
def FormSCFThreads : Pass<"form-scf-threads", "func::FuncOp"> {
  let summary = "Multi-thread SCF loop";
  let description = [{
    This pass turns scf::for into parallel loops for multi-threading.
    It runs early on in the pipeline and looks for scf:for that is
    over-arching over most of the kernel. The pass will do some
    checks but the onus is currently mostly on the user to guarantee
    parallel conversion correctness.
  }];
  let constructor = "hexagon::createFormSCFThreadsPass()";
}

def FormVirtualThreads : Pass<"form-virtual-threads", "func::FuncOp"> {
  let summary = "Rewrite generics as scf::forall for thread formation"
                 "after further lowering.";
  let constructor = "hexagon::createFormVirtualThreadsPass()";
  let options = [
       Option<"preferredNumThreads", "preferred-num-threads", "int64_t",
              /*default=*/"4",
              "Preferred number of virtual threads to create.">,
       Option<"blockSize", "block-size", "int64_t",
              /*default=*/"0",
              "Block size for testing (user-override)">,
  ];
}

def HexmemCpyToDMA :
    InterfacePass<"hexmem-cpy-to-dma", "mlir::FunctionOpInterface"> {
  let summary = "Pass to rewrite some hexagonmem.copy ops to memref.dma ops";

  let constructor =
       "hexagon::createHexmemCpyToDMAPass()";
}

def HexagonFusion :
    InterfacePass<"hexagon-fusion", "mlir::FunctionOpInterface"> {
  let summary = "Pass to fuse linalg generic ops";
  let options = [
    Option<"fusionAllowRecompute", "far", "bool",
           /*default=*/"false",
           "Allow recomputing values when fusing linalg generic ops">,
    Option<"fusionDoMultiUse", "fdmu", "bool",
           /*default=*/"true",
           "Do multi-use fusion">
   ];
  let constructor =
       "hexagon::createHexagonFusionPass()";
}

def HexagonSlicing :
    Pass<"hexagon-slicing", "mlir::ModuleOp"> {
  let summary = "Pass to slice ops into K parts, "
                " where K is identified by slicingFactor";
  let options = [
    Option<"slicingFactor", "slicing-factor", "int",
           /*default=*/"1",
           "Operator slicing factor.">
  ];
  let constructor = "hexagon::createHexagonSlicingPass()";
}

def HexagonTiling :
    Pass<"hexagon-tiling", "mlir::ModuleOp"> {
  let summary = "Pass to tile tensor linalg ops. Presently, "
                " just for vectorization.";

  let options = [
    Option<"useInterchangeVector", "use-interchange-vector", "bool",
           /*default=*/"false",
           "Use interchange vector if default inner loop "
           "is not conducive for vectorization.">,
    Option<"splitTilingRange", "split-tiling-range", "bool",
           /*default=*/"true",
           "Split the not perfectly tileable loop into perfectly tileable "
           " and the remainder parts. If set to false, padding is added "
           "to make it a perfectly tileable shape">,
    Option<"enableSplitReduction", "enable-split-reduction", "bool",
           /*default=*/"false",
           "Splits reduction loop to expose more opportunities for "
           "vectorization, ">
   ];

  let constructor = "hexagon::createHexagonTilingPass()";
}

def HexagonVectorization :
    Pass<"hexagon-vectorization", "mlir::ModuleOp"> {
  let summary = "Pass to vectorize tensor linalg ops.";
  let constructor = "hexagon::createHexagonVectorizationPass()";
}

def LowerConstantsSeparately : Pass<"lower-constants-separately", "mlir::ModuleOp"> {
  let summary = "Lowers constants separately from the code, possibly in multiple modules if above a certain threshold"; 
  let constructor = "hexagon::createLowerConstantsSeparatelyPass()";
}

def RewriteUBPoisonToZero :
    Pass<"rewrite-ub-poison-to-zero", "mlir::ModuleOp"> {
  let summary = "Pass to rewrite ub poison values to zero.";
  let constructor = "hexagon::createRewriteUBPoisonToZeroPass()";
}

def VTCMTiling :
     InterfacePass<"vtcm-tiling", "mlir::FunctionOpInterface"> {
  let summary = "Lower linalg-on-tensors (DDR) to tiled-compute on VTCM";
  let options = [
    Option<"tileSizes", "tile-size-override", "std::string",
           /*default=*/"",
           "override tile size. Only intended for debugging.">
  ];
  let constructor = "hexagon::createVTCMTilingPass()";
}

#endif
