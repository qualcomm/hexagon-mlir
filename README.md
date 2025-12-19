# Hexagon-MLIR

Hexagon-MLIR is a compiler toolchain for compiling and executing AI kernels and models on Qualcomm Hexagon Neural Processing Units (NPUs).
It provides support for compiling `Triton` and `PyTorch` graphs.

## Table of Contents

- [Features](#features)
- [Documentation](#documentation)
- [License](#license)
- [Support](#support)

## Features
- **Triton Kernel Compilation & Execution**: Compile Triton kernels and execute on Hexagon NPU targets
- **PyTorch Model Compilation & Execution**: Compile PyTorch models and execute on Hexagon NPU targets
- **Performance Optimization**: Leverage Hexagon-specific features for maximum performance, including:
  - **Multi-threading**: Hexagon-optimized parallel execution of operations for improved performance
  - **Vector Processing**: Optimized code generation targeting Hexagon Vector eXtensions (HVX) units
  - **TCM Utilization**: Leverage Tightly Coupled Memory (TCM) for reduced memory latency
  - **DMA Optimization**: Efficient DMA transfers between DDR and TCM memory spaces
  - **Matrix Processing (experimental)**: Leverage Qualcomm's Hexagon Kernel Library for matrix multiplication
- **IR Inspection**: Inspect and analyze IR lowering passes, helping you understand how your code is optimized

## Documentation
- 📖 [User Guide](docs/user-guide.md) - For instructions on how to download, setup our compiler and start running Triton kernels or PyTorch models on Hexagon NPUs
- 🎓 [Tutorials](docs/tutorials/README.md) - A set of tutorials on Triton kernels and PyTorch models
- ❓ [FAQ](docs/faq.md) - Frequently asked questions
- 🏗️ [Developer Guide](docs/developer-guide.md) - Insights on how to develop, debug and profile using our compiler toolchain for Triton kernels and PyTorch models

## License

Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.

---

<div align="center">
  <sub>Built with ❤️ by the  Hexagon-MLIR Team</sub>
</div>
