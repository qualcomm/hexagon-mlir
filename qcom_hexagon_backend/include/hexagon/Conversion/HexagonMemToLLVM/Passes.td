#ifndef HEXAGON_CONVERSION_HEXAGONMEMTOLLVM_PASSES_TD
#define HEXAGON_CONVERSION_HEXAGONMEMTOLLVM_PASSES_TD

include "mlir/Pass/PassBase.td"

def HexagonMemToLLVM : Pass<"hexagonmem-to-llvm", "mlir::ModuleOp"> {
  let summary = "Convert HexagonMem dialect to LLVM";
  let description = [{
    HexagonMem ops (like alloc, dealloc, copy, etc.) work on memref types.
    This pass lowers the ops to LLVM.
    For example, `%0 = hexagonmem.alloc() : memref<64xi8, 1>`
    can be lowered to:

    ```mlir
        %3 = llvm.mlir.constant(64 : i32) : i32
        %4 = llvm.mlir.constant(true) : i1
        %5 = llvm.call @hexagon_runtime_alloc_1d(%3, %4) : (i32, i1) -> !llvm.ptr
        ...
    ```

    All hexagonmem dialect ops are expected to be lowered after this pass.
    }];
  let constructor = "mlir::hexagonmem::createHexagonMemToLLVMPass()";
}

#endif // HEXAGON_CONVERSION_HEXAGONMEMTOLLVM_PASSES_TD
