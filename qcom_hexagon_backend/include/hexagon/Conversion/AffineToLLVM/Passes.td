#ifndef HEXAGON_AFFINE_TO_LLVM_CONVERSION_PASSES
#define HEXAGON_AFFINE_TO_LLVM_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def AffineToLLVM : Pass<"affine-to-llvm", "mlir::ModuleOp"> {
  let summary = "Convert Affine to LLVM";
  let constructor = "hexagon::createAffineToLLVMPass()";
  let options =
      [Option<"dummyOption", "dummy-option", "bool",
              /*default=*/"false", "just a placeholder option for later.">];
}

def AffineTiling : Pass<"affine-tiling", "func::FuncOp"> {
  let summary = "Tile affine loops using heuristic for tile sizes";
  let constructor = "hexagon::createAffineTilingPass()";
  let options = [
    Option<"cacheSizeInKiB", "cache-size", "uint64_t", /*default=*/"4096",
           "Set size of cache to tile for in KiB (default: 4096)">,
    Option<"separate", "separate", "bool", /*default=*/"false",
           "Separate full and partial tiles (default: false)">
  ];
}

def AffinePipelineFusion : Pass<"affine-pipeline-fusion", "func::FuncOp"> {
  let summary = "Perform pipeline fusion on affine loops";
  let constructor = "hexagon::createAffinePipelineFusionPass()";
  let options = [];
}

def AffineVectorize : Pass<"affine-vectorize", "mlir::func::FuncOp"> {
  let summary = "vectorize affine loops";
  let constructor = "hexagon::createAffineVectorizePass()";
}

def AffineTileMemory : Pass<"affine-tile-memory", "mlir::func::FuncOp"> {
  let summary = "Optimized tiled loops to use local memrefs for large memrefs";
  let constructor = "hexagon::createAffineTileMemoryPass()";
  let options = [
    Option<"cacheSizeInKiB", "cache-size", "uint64_t", /*default=*/"4096",
           "Set size of cache to tile for in KiB (default: 4096)">,
    Option<"cacheMemorySpace", "cache-space", "unsigned", /*default=*/"0",
           "Set memory space of cached memrefs">
  ];
}

#endif
