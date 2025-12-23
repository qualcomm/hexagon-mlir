# Hexagon-MLIR User Guide

## Table of Contents

- [Introduction](#introduction)
- [Prerequisites](#prerequisites)
  - [System Requirements](#system-requirements)
  - [Required Permissions](#required-permissions)
- [Setup](#setup)
  - [System Packages](#system-packages)
  - [Python Setup](#python-setup)
  - [Set Hexagon architecture](#set-hexagon-architecture)
- [Building Hexagon-MLIR Compiler](#building-hexagon-mlir-compiler)
  - [Clone Repository](#clone-repository)
  - [Software Dependencies](#software-dependencies)
    - [Install Python Packages](#install-python-packages)
    - [Install Hexagon SDK and Tools](#install-hexagon-sdk-and-tools)
    - [Build Verified LLVM](#build-verified-llvm)
    - [Setup Host Toolchain LLVM](#setup-host-toolchain-llvm)
    - [Build Hexagon-MLIR](#build-hexagon-mlir)
    - [Set Environment Variables for Testing](#set-environment-variables-for-testing)
- [Verify the Setup](#verify-the-setup)
  - [Step 1: Verify Build Tools](#step-1-verify-build-tools)
  - [Step 2: Run LIT Tests](#step-2-run-lit-tests)
  - [Step 3: Run Triton Test](#step-3-run-triton-test)
  - [Step 4: Run PyTorch Test](#step-4-run-pytorch-test)
- [Additional Resources](#additional-resources)
  - [MLIR Resources](#mlir-resources)
  - [Triton Resources](#triton-resources)

## Introduction
This user guide provides instructions for setting up and installing Qualcomm Hexagon-MLIR compiler,
 and executing end-to-end tests on Qualcomm Hexagon NPUs.

## Prerequisites
### System Requirements
- **Operating System**: Ubuntu 22.04 (recommended)
- **Python Version**: Python 3.11 (recommended)
- **Hardware**: Qualcomm Hexagon NPU (tested architectures - v73, v75, v79)
- **Device Access**: Access to Qualcomm Hexagon NPU enabled device

### Required Permissions
- Root/sudo access for system package installation
- Device access permissions for Hexagon NPU hardware
- Network access for downloading dependencies

## Setup

### System Packages
The following system packages are required:
```bash
# Essential build tools
sudo add-apt-repository ppa:ubuntu-toolchain-r/test
sudo apt-get update
sudo apt-get upgrade
sudo apt-get dist-upgrade

sudo apt-get install -y build-essential cmake ninja-build git wget curl
sudo apt-get install -y python3-dev python3-pip python3-venv
sudo apt-get install -y clang-format

# C++ compiler setup
sudo apt-get install -y g++-9 libstdc++-12-dev
```

### Python Setup
Set up a dedicated Python 3.11 environment (e.g. using [conda](https://anaconda.org/anaconda/conda)):
```bash
mkdir -p ./miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ./miniconda3/miniconda.sh
bash ./miniconda3/miniconda.sh -b -u -p ./miniconda3
rm -rf ./miniconda3/miniconda.sh
./miniconda3/bin/conda init bash
conda create --name main python=3.11
conda activate main
```

### Set Hexagon architecture
Set the Hexagon architecture version according to your target device.
For example, if you are targeting v75,
```
export HEXAGON_ARCH_VERSION=75
```

## Building Hexagon-MLIR Compiler
###  Clone Repository
```bash
# Clone the main repository with submodules
git clone https://github.com/qualcomm/hexagon-mlir.git --recurse-submodules
cd hexagon-mlir
export HEXAGON_MLIR_ROOT=$PWD
export TRITON_ROOT=$HEXAGON_MLIR_ROOT/triton
```

### Software Dependencies
- **Python Packages**: Listed in buildbot/requirements.txt
- **Hexagon SDK**: Version 6.4.0.0
- **Hexagon Tools**: Version 21.0
- **LLVM/MLIR**: Custom build for Hexagon target
- **Host Toolchain**: clang+llvm-13.0.1
- **(Optional) Hexagon Kernel Library**: Version 1.0

Set up steps follow the above order:

### Install Python Packages

The command below instructs pip (Python's package manager) to install dependencies e.g.:
- dependencies that are needed to build, test, and run the Hexagon‑MLIR.
- exact **PyTorch** stack and the **Torch‑MLIR** bridge, ensuring model conversion to MLIR works reproducibly.

```bash
pip install -r buildbot/requirements.txt
```


### Install Hexagon SDK and Tools
Skip the download/install steps if you already have access to:
- Hexagon SDK 6.4.0.0
- Hexagon Tools 21.0
- (Optional) Hexagon Kernel Library 1.0

#### QPM For Internal (to Qualcomm) Developers 
1. Join [hexagonsdk.users](https://lists.qualcomm.com/ListManager?match=eq&field=default&query=hexagonsdk.users)
2. Download and install QPM from [an internal source](https://qpm.qualcomm.com/#/main/tools/details/QPM3)

#### QPM For External (to Qualcomm) Developers
Download and install QPM from [here](https://softwarecenter.qualcomm.com/catalog/item/QPM3).
<!-- TODO: Add an installation option (once we find one) that doesn't require QPM credentials -->

#### Download and Install via QPM
Install Hexagon SDK 6.4.0.0:
```bash
# Login and activate license
qpm-cli --login
qpm-cli --license-activate hexagonsdk6.x
# Set SDK_INSTALL_PATH to your desired install location
qpm-cli --extract hexagonsdk6.x --version 6.4.0.0 --path $SDK_INSTALL_PATH
```
Install Hexagon Tools 21.0:
```bash
qpm-cli --license-activate Hexagon21.0
# Set TOOLS_INSTALL_PATH to your desired install location
qpm-cli --extract Hexagon21.0 --path $TOOLS_INSTALL_PATH
```

#### Configure Hexagon Dependencies
```bash
# If skipping the download steps, see above for how SDK_INSTALL_PATH and TOOLS_INSTALL_PATH must be set
# Run SDK setup script (setting HEXAGON_SDK_ROOT)
source $SDK_INSTALL_PATH/setup_sdk_env.source
# Set tools path
export HEXAGON_TOOLS=$TOOLS_INSTALL_PATH/Tools
# Set HEXKL_ROOT to the HexKL(Hexagon kernel library) installation path
export HEXKL_ROOT=<HEXKL_INSTALL_PATH>/sdkl_addon
```

### Build Verified LLVM
```bash
# Use the provided build script, which builds:
#   - For a Hexagon backend
#   - From an LLVM SHA we've verified for hexagon-mlir
#   - Ensuring required LLVM projects are built (e.g. lld)
source buildbot/build_llvm.sh -s <path_to_llvm_to_be_built>
```

### Setup Host Toolchain LLVM
Download and install host toolchain:
```bash
wget https://github.com/llvm/llvm-project/releases/download/llvmorg-13.0.1/clang+llvm-13.0.1-x86_64-linux-gnu-ubuntu-18.04.tar.xz
tar -xf clang+llvm-13.0.1-x86_64-linux-gnu-ubuntu-18.04.tar.xz
export HOST_TOOLCHAIN=$(pwd)/clang+llvm-13.0.1-x86_64-linux-gnu-ubuntu-18.04
export PATH=$HOST_TOOLCHAIN/bin:$PATH
```

###  Build Hexagon-MLIR
```bash
# Build the Hexagon-MLIR backend
cd $TRITON_ROOT
TRITON_HOME=$HEXAGON_MLIR_ROOT/.. \
TRITON_PLUGIN_DIRS="$HEXAGON_MLIR_ROOT/triton_shared;$HEXAGON_MLIR_ROOT/qcom_hexagon_backend" \
TRITON_BUILD_WITH_CLANG_LLD=1 \
TRITON_BUILD_WITH_CCACHE=true \
LLVM_INCLUDE_DIRS=$LLVM_INSTALL_DIR/include \
LLVM_LIBRARY_DIR=$LLVM_INSTALL_DIR/lib \
LLVM_SYSPATH=$LLVM_INSTALL_DIR \
pip install -e . --no-build-isolation --verbose
cd $HEXAGON_MLIR_ROOT
# This builds:
# - Triton with Hexagon backend
# - linalg-hexagon-opt, linalg-hexagon-translate, triton-shared-opt
# - Python bindings
```

### Set Environment Variables for Testing
```bash
export TRITON_SHARED_OPT_PATH=$TRITON_ROOT/build/cmake.linux-x86_64-cpython-3.11/third_party/triton_shared/tools/triton-shared-opt/triton-shared-opt
export PATH=$TRITON_ROOT/build/cmake.linux-x86_64-cpython-3.11/third_party/qcom_hexagon_backend/bin/:$TRITON_ROOT/build/cmake.linux-x86_64-cpython-3.11/third_party/triton_shared/tools/triton-shared-opt:$PATH
export PYTHONPATH=$TRITON_ROOT/python:$HEXAGON_MLIR_ROOT:$PYTHONPATH
```

## Verify the Setup
### Step 1: Verify Build Tools
Check that the build was successful by locating key tools:
```bash
# Find the main developer tool
find . -name linalg-hexagon-opt
# Expected output: ./triton/build/cmake.linux-x86_64-cpython-3.11/third_party/qcom_hexagon_backend/bin/linalg-hexagon-opt

# Verify tool is accessible
which linalg-hexagon-opt
```

### Step 2: Run LIT Tests
```bash
lit triton/build/cmake.linux-x86_64-cpython-3.11/third_party/qcom_hexagon_backend/test/
```

### Step 3: Run Triton Test
If you have access to a Qualcomm NPU device, please set the variables ANDROID_HOST and ANDROID_SERIAL to the host name and hexagon device number.
 You can then run an end-to-end compilation and execution flow using the following command:
```bash
# Compile and execute a simple Triton kernel
pytest -sv test/python/triton/test_vec_add.py
```

### Step 4: Run PyTorch Test
There are more details under docs/tutorials/torch-mlir, but treat the following as a cheatsheet to run an end-to-end PyTorch test:
```bash
# Compile and execute PyTorch Softmax test
pytest -sv test/python/torch-mlir/test_softmax_torch.py
```

## Additional Resources

### MLIR Resources
- [MLIR Documentation](https://mlir.llvm.org/) - MLIR project documentation
- [MLIR Dialects](https://mlir.llvm.org/docs/Dialects/) - Available MLIR dialects

### Triton Resources
- [OpenAI Triton](https://github.com/openai/triton) - Main Triton repository
- [Triton Language Reference](https://triton-lang.org/main/python-api/triton.language.html) - Language documentation
- [Triton Tutorials](https://triton-lang.org/main/getting-started/tutorials/index.html) - Learning resources
- [Triton-Shared Middle Layer](https://github.com/microsoft/triton-shared) - Triton to Linalg toolchain


