# MLIR Test Options Guide

This guide explains how to pass options to kernels compiled and executed via `test_mlir.py` using the new command-line argument system. This system allows developers to experiment quickly with input.mlir and wrapper.cpp files that are not currently located in the repo. For example the triton flow generates wrappers and mlir files that can be later debugged with this flow. This system is created for prototyping whereas in order to contribute mlir tests to the hexagon-mlir repo, one would need to create a new directory under test/python/mlir directory with input.mlir, wrapper.cpp and pytest file.

## Overview

The `test_mlir.py` script now supports passing Hexagon backend options through command-line arguments. These options correspond to the boolean flags defined in `qcom_hexagon_backend/include/hexagon/Conversion/LinalgToLLVM/Passes.td` and implemented in `qcom_hexagon_backend/backend/compiler.py` (HexagonOptions class).

## Two Ways to Pass Options

### 1. Explicit Flags (Commonly Used Options)

Five commonly used options have dedicated command-line flags:

- `--enable-multi-threading`: Enable async multi-threading (default: False)
- `--enable-vtcm-tiling`: Enable VTCM tiling (default: True)
- `--enable-convert-to-hexagonmem`: Enable conversion to hexagonmem ops (default: True)
- `--enable-hexagonmem-copy-to-dma`: Enable DMA conversion (default: False)
- `--enable-bufferization`: Enable bufferization passes (default: True)

### 2. Generic Option Flag (All Other Options)

For any other HexagonOptions field, use the `--option` flag with `key=value` format:

```bash
--option <key>=<value>
```

You can use `--option` multiple times to set multiple options.

## Usage Examples

### Basic Usage

```bash
python test_mlir.py \
    --mlir_file_name input.mlir \
    --cpp_wrapper_path wrapper.cpp
```

### Enable DMA Conversion

```bash
python test_mlir.py \
    --mlir_file_name input.mlir \
    --cpp_wrapper_path wrapper.cpp \
    --enable-hexagonmem-copy-to-dma
```

### Disable Bufferization (for manual buffer management)

```bash
python test_mlir.py \
    --mlir_file_name input.mlir \
    --cpp_wrapper_path wrapper.cpp \
    --option enableBufferization=false
```

### Enable Multi-Threading

```bash
python test_mlir.py \
    --mlir_file_name input.mlir \
    --cpp_wrapper_path wrapper.cpp \
    --enable-multi-threading
```

### Use Generic Options for Less Common Flags

```bash
python test_mlir.py \
    --mlir_file_name input.mlir \
    --cpp_wrapper_path wrapper.cpp \
    --option fusion=false \
    --option enableLWP=true \
    --option tileSizes="64,64"
```

### Combine Explicit and Generic Options

```bash
python test_mlir.py \
    --mlir_file_name input.mlir \
    --cpp_wrapper_path wrapper.cpp \
    --enable-multi-threading \
    --enable-hexagonmem-copy-to-dma \
    --option enableLWP=true \
    --option slicingFactor=2
```

### Validate Results

```bash
python test_mlir.py \
    --mlir_file_name input.mlir \
    --cpp_wrapper_path wrapper.cpp \
    --validate_result \
    --enable-hexagonmem-copy-to-dma
```

## Available Options

All options from `HexagonOptions` class can be set. Here are the most commonly used ones:

### Boolean Options (use true/false)

- `enableMultiThreading`: Enable async multi-threading
- `enableSCFThreading`: Enable SCF-level multi-threading
- `enableVTCMTiling`: Enable VTCM tiling
- `enableConvertToHexagonmem`: Convert memref to hexagonmem ops
- `enableHexagonmemCopyToDMA`: Convert hexagonmem copy to DMA ops
- `enableBufferization`: Enable bufferization passes
- `enableCollapseAddressSpace`: Collapse address spaces
- `enableHexKL`: Use HexKL for matmul/convolutions
- `enableSeedLayoutConversions`: Seed layout conversions around conv2d
- `enableLWP`: Enable lightweight profiling
- `fusion`: Enable linalg generic op fusion
- `fusionAllowRecompute`: Allow recomputing values during fusion
- `fusionDoMultiUse`: Enable multi-use fusion
- `useInterchangeVector`: Use interchange vector in tiling
- `addFastMath`: Add fast math flags
- `splitTilingRange`: Split non-perfectly tileable loops
- `enableSplitReduction`: Enable split reduction
- `enableSlicing`: Enable operator slicing
- `expandBoolVec`: Expand boolean vectors
- `enableHexagonRoutines`: Use vectorized Hexagon routines
- `lowerConstantsInSeparateSharedObjects`: Lower constants separately
- `disableLWPLoop`: Disable loop-level LWP instrumentation
- `extendPackUpperFrontier`: Extend linalg.pack frontiers
- `extendPackLowerFrontier`: Extend linalg.unpack frontiers
- `extendPackParallelsOnly`: Only extend pack for parallel ops

### Integer Options

- `slicingFactor`: Operator slicing factor (default: 1)
- `LWPloopDepth`: LWP instrumentation depth (default: 1)
- `num_threads`: Number of threads (default: 4)
- `iterations`: Benchmarking iteration count (default: 10)

### String Options

- `tileSizes`: Override tile sizes (e.g., "64,64")

## Option Value Types

The `--option` flag automatically converts values:

- **Boolean**: `true` or `false` (case-insensitive)
- **Integer**: Any numeric string (e.g., `4`, `128`)
- **String**: Any other value (e.g., `"64,64"`)

## Error Handling

The script validates options:

1. **Invalid format**: If `--option` doesn't follow `key=value` format, you'll get an error
2. **Unknown option**: If the key doesn't exist in `HexagonOptions`, you'll get an error with the invalid key name

## Help

To see all available options and examples:

```bash
python test_mlir.py --help
```

## Mapping to Passes.td

The options correspond to the `Option<>` definitions in
`qcom_hexagon_backend/include/hexagon/Conversion/LinalgToLLVM/Passes.td`:

| Passes.td Option | HexagonOptions Field | CLI Flag |
|-----------------|---------------------|----------|
| `enableHexagonmemCopyToDMA` | `enableHexagonmemCopyToDMA` | `--enable-hexagonmem-copy-to-dma` |
| `enableBufferization` | `enableBufferization` | `--enable-bufferization` |
| `enableMultiThreading` | `enableMultiThreading` | `--enable-multi-threading` |
| `enableVTCMTiling` | `enableVTCMTiling` | `--enable-vtcm-tiling` |
| `enableConvertToHexagonmem` | `enableConvertToHexagonmem` | `--enable-convert-to-hexagonmem` |
| `fusion` | `fusion` | `--option fusion=true/false` |
| `enableLWP` | `enableLWP` | `--option enableLWP=true/false` |
| ... | ... | ... |

## Notes

- Options set via command-line override the defaults from `HexagonOptions`
- The explicit flags (`--enable-*`) are preferred for commonly used options
- Use `--option` for less common options or when scripting
- All boolean options default to their values in `HexagonOptions` class
