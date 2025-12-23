# Hexagon-MLIR FAQ

## Table of Contents

- [General Questions](#general-questions)
- [Installation & Setup](#installation--setup)
- [Development & Usage](#development--usage)
- [Performance & Optimization](#performance--optimization)
- [Troubleshooting](#troubleshooting)
- [Advanced Topics](#advanced-topics)

---

## General Questions

### What is Hexagon-MLIR?

Hexagon-MLIR is an end-to-end compiler toolchain for compiling and executing AI kernels and models on Qualcomm Hexagon Neural Processing Units (NPUs).

### How does your Triton support relate to OpenAI Triton?

Hexagon-MLIR extends the Triton ecosystem by providing a new backend that compiles Triton kernels to Qualcomm Hexagon hardware. You write kernels using the same Triton Python API, but they execute on Hexagon NPU instead of GPUs.

### What hardware does this support?

Tested Platforms:
- **v73**: Hexagon DSP v73 with HVX support
- **v75**: Hexagon DSP v75 with HVX support
- **v79**: Hexagon DSP v79 with HVX support

You do need access to a Qualcomm device with Hexagon NPU hardware to run the compiled kernels.

### I dont have access to Qualcomm device, what can I still do?

As part of the tools built, we provide `linalg-hexagon-opt` executable that you can use to go through our pass-pipeline lowering, and dump the intermediate IRs.
```bash
find . -name linalg-hexagon-opt
path-to/linalg-hexagon-opt --help | grep hexagon
```

### What's the difference between this and other ML frameworks?

Unlike frameworks that target CPUs or GPUs, Hexagon-MLIR specifically targets Qualcomm's specialized NPU hardware, leveraging:
- **HVX units** for vector processing
- **HMX units** for matrix operations
- **TCM (Tightly Coupled Memory)** for low-latency memory access
- **DMA engines** for efficient data movement
- **Multi-threading** for Hexagon architecture

---

## Installation & Setup

Please refer to [user-guide](user-guide.md)

---

## Development & Usage

### How do I write my first Triton kernel?
 Please read the [tutorials][tutorials/README.md]

### How do I debug compilation issues?

Enable debug output to see intermediate representations:

```bash
# Enable MLIR IR dumping
TRITON_ALWAYS_COMPILE=1 MLIR_ENABLE_DUMP=1 pytest -sv your_test.py
```
### How do I debug linking and loading issues?

This may change over time but presently look at python scripts in `hexagon-mlir/qcom_hexagon_backend/backend` if you need to debug print more than what is presently.

### How does multi-threading work?

The compiler tries to automatically parallelize work across multiple HVX threads.
There is a initial version but there is more to come.
- Large tensors are divided into chunks
- Each thread processes its chunk independently

### What is VTCM and when should I use it?

VTCM (Vector Tightly Coupled Memory) is a high-speed on-chip memory:
- **Advantages**: Faster access than DDR memory
- **Limitations**: Limited size (typically a few MB)
- **Best for**: Frequently accessed data, intermediate results

### How do DMA transfers work?

DMA (Direct Memory Access) enables efficient data movement:
- Transfers data between DDR and TCM without CPU involvement
- Overlaps computation with data movement
- Automatically inserted by the compiler for large data transfers
- See the tutorials for more details

### What vector sizes work best?

The Hexagon HVX units are optimized for:
- **32-element vectors** for 32-bit data (vector<32xf32>)
- **64-element vectors** for 16-bit data (vector<64xf16>)
- **128-element vectors** for 8-bit data (vector<128xi8>)

---

## Troubleshooting

### "TRITON_SHARED_OPT_PATH is not set" error

Set the environment variable:
Use `find . -name triton-shared-opt` to locate path (below is just an example), and export it.
```bash
export TRITON_SHARED_OPT_PATH=$TRITON_ROOT/build/cmake.linux-x86_64-cpython-3.11/third_party/triton_shared/tools/triton-shared-opt/triton-shared-opt
```

### "linalg-hexagon-opt not found" error

Ensure the binary is in your PATH:
```bash
export PATH=$TRITON_ROOT/build/cmake.linux-x86_64-cpython-3.11/third_party/qcom_hexagon_backend/bin/:$PATH
```

### Device connection issues

Check your device setup:
```bash
# Verify device variables are set
echo $ANDROID_HOST
echo $ANDROID_SERIAL

# Test device connectivity
adb devices
```
For more options please refere to [Android Debug Bridge documentation](https://developer.android.com/tools/adb).


### Runtime execution fails

Check these common issues:
1. **Device permissions**: Ensure you have access to Hexagon hardware.
2. **Hangs**: Ensure you are not enabling Triton multi-threading (NUM_THREADS) and hexagon-mlir multi-threading `enableMultiThreading` at the same time.
3. **Tensor sizes**: Very large tensors may exceed hardware limits.

---

## Advanced Topics

### How does the compilation pipeline work?

The compilation follows these stages:

1. **Python Triton → Triton-IR**: Frontend parsing and initial lowering
2. **Triton-IR → Linalg-IR**: Using triton-shared for dialect conversion
3. **Linalg-IR optimization**: Hexagon-specific passes (fusion, tiling, vectorization)
4. **MLIR-LLVM → Object Code**: Final code generation for Hexagon target
5. **Linking**: Create shared library with runtime wrapper
6. **Execution**: Load and run on device

### What is Hexagon Kernel Library (HexKL)?
HexKL is Qualcomm's Hexagon Kernel Library (currently available on request) providing providing both host CPU-side and NPU-side kernels optimized for AI/ML workloads such as matrix multiplication in various configurations. Details on setup are available in the [user-guide](user-guide.md) and please refer to [matmul-hexkl-tutorial](tutorials/triton/matmul_hexkl.md) for a tutorial demonstrating it's usage within hexagon-mlir.

### How do I profile kernel performance?

Use the built-in profiling:
```python
# Profiling is automatically enabled
# Results are printed as:
# Test_Info: {
#     Name: kernel_name
#     Result: Pass/Fail
#     Perf: xxxx
#     Units: us
# }
```

For detailed profiling, enable LWP (Lightweight Profiling):
```python
kernel[(1,)](args..., enableLWP=True)
```

### Can I use this with PyTorch models?

Yes, through Torch-MLIR integration.

### How do I contribute to the project?

We will update on this in the future.

### What's the roadmap for new features?

Planned improvements include:
- Support for more Triton language features
- Additional optimization passes
- Better debugging tools
- Extended hardware support

---

