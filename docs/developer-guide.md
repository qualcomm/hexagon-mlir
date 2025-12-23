# Hexagon-MLIR Developer Guide

## Table of Contents

- [Introduction](#introduction)
- [Contributions](#contributions-guide)
- [Testing](#testing)
- [Debugging](#debugging)
- [Limitations](#limitations)  
- [Additional Resources](#additional-resources)  

## Introduction

This developer guide covers debugging and optimization topics not covered in our other documentations: [user guide](user-guide.md), [tutorials](tutorials/README.md), and [FAQ](faq.md). In this guide’s context, developers are the users of
 the Hexagon-MLIR compiler who want to debug at the IR level and improve performance for their Triton kernels or PyTorch models.

Prerequisites: You have built the Hexagon-MLIR compiler and successfully run the basic vector addition tutorial.

## Debugging

You can enable several IR dumps to trace the compilation lowering process:

```bash
TRITON_ALWAYS_COMPILE=1 MLIR_ENABLE_DUMP=1 LLVM_IR_ENABLE_DUMP=1 \
TRITON_ENABLE_LLVM_DEBUG=1 pytest -sv test/python/triton/test_vec_add.py
```

### Assembly and Object Code Dump 
We provide ways to dump the final assembly code along with the object binary code:

- `HEXAGON_ASM_DUMP`: Enable assembly code dump. By default, outputs to a random filepath.
- `HEXAGON_ASM_DUMP_FILE`: Control the output location for assembly dumps.
- `HEXAGON_ASM_TO_OBJ`: Enable object binary code dump.

Note: The filepath environment variable should be set without a file extension suffix, as the system adds the appropriate extension based on the output type.

For additional debugging options, see [Triton documentation](https://github.com/triton-lang/triton).

### Developer Standalone Tool

One of the executables built is `linalg-hexagon-opt` (similar to `mlir-opt` and includes Hexagon-MLIR passes).

```bash
linalg-hexagon-opt --help | grep hexagon
USAGE: linalg-hexagon-opt [options] <input file>
      ...
      --hexagon-add-fastmath                                 -   Add fast math flag to selected ops.
      --hexagon-fusion                                       -   Pass to fuse linalg generic ops
      --hexagon-lwp-instrumentation                          -   Pass to instrument and profile with hexagon instrinsic
      --hexagon-punt-buffer                                  -   erase copies and forward buffer directly to user
      ...
      --hexagonmem-to-llvm                                   -   Convert HexagonMem dialect to LLVM
      ...
```

You can take a valid MLIR IR and lower it to LLVM IR, and see all the passes that get triggered:

```bash
linalg-hexagon-opt test.mlir -linalg-to-llvm --mlir-print-ir-after-all
```

## Limitations

The following Triton language features are currently **not supported**:

- **Data Types**: 
  - Scalar/boolean return values
  - Complex number operations
  
- **Operations**:
  - Shape manipulation operations
  - `tl.interleave()` function
  - `tl.join()` function
  - Dynamic shape operations

- **Memory**:
  - Some advanced memory access patterns
  - Certain atomic operations

- **Control Flow**:
  - Complex nested control structures
  - Some loop constructs

## Additional Resources

### MLIR Resources
- [MLIR Documentation](https://mlir.llvm.org/) - MLIR project documentation
- [MLIR Dialects](https://mlir.llvm.org/docs/Dialects/) - Available MLIR dialects

### Triton Resources
- [OpenAI Triton](https://github.com/openai/triton) - Main Triton repository
- [Triton Language Reference](https://triton-lang.org/main/python-api/triton.language.html) - Language documentation
- [Triton Tutorials](https://triton-lang.org/main/getting-started/tutorials/index.html) - Learning resources
- [Triton-Shared Middle Layer](https://github.com/microsoft/triton-shared) - Triton to Linalg toolchain
