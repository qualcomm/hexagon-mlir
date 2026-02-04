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
    - [Script-based Build](#script-based-build)
    - [Overview of the Manual Setup](#overview-of-the-manual-setup)
        - [Required Environment Variables](#required-environment-variables)
        - [Installing Required Components](#installing-required-components)
            - [triton submodule](#triton-submodule)
            - [triton_shared submodule](#triton_shared-submodule)
            - [Hexagon SDK](#hexagon-sdk)
            - [Hexagon Tools](#hexagon-tools)
            - [Hexagon Kernel Library (HexKL)](#hexagon-kernel-library-hexkl)
            - [Download the clang toolchain](#download-the-clang-toolchain)
        - [Building LLVM for Triton](#building-llvm-for-triton)
        - [Creating a Python Environment](#creating-a-python-environment)
        - [Building Triton Locally](#building-triton-locally)
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
If you don't have access to a Qualcomm NPU device you can still build and look at the lowering, and see the generated LLVM-IR along with Hexagon assembly.

## Building Hexagon-MLIR Compiler
###  Clone Repository
```bash
# Clone the main repository with submodules
git clone https://github.com/qualcomm/hexagon-mlir.git --recurse-submodules
cd hexagon-mlir
export HEXAGON_MLIR_ROOT=$PWD
export TRITON_ROOT=$HEXAGON_MLIR_ROOT/triton
```

### Script-based Build

We provide a script `HEXAGON_MLIR_ROOT/scripts/build_hexagon_mlir.sh` that automates much of the setup process of building a complete `Hexagon-MLIR + Triton + Torch-MLIR` environment locally, including: 

* Submodule setup for `triton` and `triton_shared`.
* Downloading and extracting Hexagon SDK, Hexagon Tools, and Hexagon Kernel Library (HexKL).
* Downloading and building the required LLVM version for Triton.
* Creating a Python virtual environment and installing required Python packages.
* Building Triton with Hexagon backend support and running tests.

To run the script, navigate to the root of the `hexagon-mlir` repository and execute:

```bash
bash ./scripts/build_hexagon_mlir.sh
```
The script builds hexagon-mlir and runs the LIT tests for you to ensure the build succeeded.

However, if you prefer to set up the environment manually, follow the steps below.

### Overview of the Manual Setup

Local development requires downloading several components: 

* [Hexagon SDK 6.4.0.2](https://softwarecenter.qualcomm.com/catalog/item/Hexagon_SDK)
* [Hexagon Tools 19.0.02](https://softwarecenter.qualcomm.com/catalog/item/Hexagon_open_access)
* [Hexagon Kernel Library (HexKL 1.0.0)](https://softwarecenter.qualcomm.com/catalog/item/Hexagon_KL)
* LLVM (Triton-compatible)
* Python Virtual Environment
* [triton](https://github.com/triton-lang/triton)
* [triton-shared](https://github.com/microsoft/triton-shared)

#### Required Environment Variables

Export the following environment variables 

```bash
export HEXAGON_SDK_ROOT=/path/to/Hexagon_SDK/6.4.0.2
export HEXAGON_TOOLS=/path/to/Tools
export HEXKL_ROOT=/path/to/hexkl_addon
export LLVM_PROJECT_BUILD_dir=/path/to/llvm/build
export CONDA_ENV=/path/to/your/python/env
```

#### Installing Required Components

##### triton submodule
Set up the triton submodule and apply Qualcomm specific patches. 

```bash
REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "${REPO_ROOT}"
git submodule add --force https://github.com/triton-lang/triton.git triton
cd triton
git checkout e44bd1c83c1c3e8deac7c4f02683cfb3cc395c8b
git apply "${REPO_ROOT}/third_party_software/patches/triton/third_party_triton.patch"
```

##### triton_shared submodule
Set up the triton_shared submodule and apply Qualcomm specific patches.

```bash
REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "${REPO_ROOT}"
git submodule add --force https://github.com/microsoft/triton-shared triton_shared
cd triton_shared
git checkout 2b728ad97bc02af821a0805b09075838911d4c19
git apply "${REPO_ROOT}/third_party_software/patches/triton_shared/max_with_nan_propagation.patch"
git apply "${REPO_ROOT}/third_party_software/patches/triton_shared/tt_shared_split_dim.patch"
```

##### Hexagon SDK

```bash 
wget https://softwarecenter.qualcomm.com/api/download/software/sdks/Hexagon_SDK/Linux/Debian/6.4.0.2/Hexagon_SDK_lnx.zip
unzip -q Hexagon_SDK_lnx.zip 
export HEXAGON_SDK_ROOT=/path/to/Hexagon_SDK/6.4.0.2
```

##### Hexagon Tools

```bash
wget https://softwarecenter.qualcomm.com/api/download/software/tools/Hexagon_open_access/Linux/Debian/19.0.02/Hexagon_open_access.Core.19.0.02.Linux-Any.tar.gz
tar -xzf Hexagon_open_access.Core.19.0.02.Linux-Any.tar.gz
export HEXAGON_TOOLS=/path/to/Tools
```

##### Hexagon Kernel Library (HexKL)

* Download the HexKL package.
* Extract the outer zip.
* Extract the inner zip (e.g., hexkl-1.0.0-beta1-6.4.0.0.zip).
* Locate the `hexkl_addon` directory. 

```bash
wget https://softwarecenter.qualcomm.com/api/download/software/tools/Hexagon_KL/Linux/1.0.0/Hexagon_KL.Core.1.0.0.Linux-Any.zip
unzip Hexagon_KL.Core.1.0.0.Linux-Any.zip 
unzip hexkl-1.0.0-beta1-6.4.0.0.zip
export HEXKL_ROOT=/path/to/hexkl_addon
```

##### Download the clang toolchain

```bash
wget https://github.com/llvm/llvm-project/releases/download/llvmorg-13.0.1/clang+llvm-13.0.1-x86_64-linux-gnu-ubuntu-18.04.tar.xz
tar -xf clang+llvm-13.0.1-x86_64-linux-gnu-ubuntu-18.04.tar.xz --strip-components=1
export HOST_TOOLCHAIN=/path/to/clang+llvm-13.0.1-x86_64-linux-gnu-ubuntu-18.04
export PATH="${HOST_TOOLCHAIN}/bin:${PATH}"
export CC="${HOST_TOOLCHAIN}/bin/clang"
export CXX="${HOST_TOOLCHAIN}/bin/clang++"
```

#### Building LLVM for Triton 

Triton requires a specific LLVM revision; and you can find the expected hash in `triton/cmake/llvm-hash.txt`

##### Build Steps
```bash
git clone https://github.com/llvm/llvm-project.git
cd llvm-project
git checkout <hash-from-triton>

mkdir build && cd build
cmake -G "Ninja" ../llvm \
    -DLLVM_ENABLE_PROJECTS="llvm;mlir;lld" \
    -DCMAKE_C_COMPILER="${CC}" \
    -DCMAKE_CXX_COMPILER="${CXX}" \
    -DCMAKE_ASM_COMPILER="${CC}" \
    -DLLVM_BUILD_EXAMPLES=ON \
    -DLLVM_INSTALL_UTILS=ON \
    -DLLVM_TARGETS_TO_BUILD="AMDGPU;NVPTX;X86;Hexagon" \
    -DCMAKE_BUILD_TYPE="RelWithDebInfo" \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DLLVM_ENABLE_RTTI=ON \
    -DLLVM_CCACHE_BUILD:BOOL=ON \
    -DLLVM_ENABLE_EH=ON \
    -DLLVM_BUILD_EXAMPLES:BOOL=OFF \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=1 \
    -DLLVM_DEFAULT_TARGET_TRIPLE=x86_64-unknown-linux-gnu \
    -DCMAKE_INSTALL_PREFIX="${LLVM_PROJECT_BUILD_DIR}/install"
```

Set:
`export LLVM_PROJECT_BUILD_DIR=/path/to/llvm-project/build`

#### Creating a Python Environment

Example using `venv`:

```bash
python3 -m venv mlir-env
source mlir-env/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r ci/requirements.txt
```

Then set:

`export CONDA_ENV=/path/to/mlir-env`

(Letting developers choose between a miniconda or a Python virtual env, the variable name is still CONDA_ENV for consistency.)

#### Building Triton Locally

Now, your variables might be already set for `HEXAGON_SDK_ROOT`, `HEXAGON_TOOLS`, `HEXKL_ROOT`, `LLVM_PROJECT_BUILD_DIR`, and `CONDA_ENV`.
There are some `triton` specific variables that need to be set in order to build Triton with Hexagon backend support:

```bash
export TRITON_ROOT=$HEXAGON_MLIR_ROOT/triton
export TRITON_SHARED_OPT_PATH=$TRITON_ROOT/build/cmake.linux-x86_64-cpython-${PYTHON_VERSION}/third_party/triton_shared/tools/triton-shared-opt/triton-shared-opt
export TRITON_HOME=$HEXAGON_MLIR_ROOT
export TRITON_PLUGIN_DIRS="$HEXAGON_MLIR_ROOT/triton_shared;$HEXAGON_MLIR_ROOT/qcom_hexagon_backend"
export PATH=$TRITON_ROOT/build/cmake.linux-x86_64-cpython-${PYTHON_VERSION}/third_party/qcom_hexagon_backend/bin/:$TRITON_ROOT/build/cmake.linux-x86_64-cpython-${PYTHON_VERSION}/third_party/triton_shared/tools/triton-shared-opt:$PATH
export PYTHONPATH=$TRITON_ROOT/python:$PYTHONPATH
```
*Note*: Replace `${PYTHON_VERSION}` with your actual Python version, e.g., `3.11`. `HEXAGON_MLIR_ROOT/scripts/set_local_env.sh` is a good reference for setting these variables.

Once your environment is valid:

```bash
./scripts/build_triton.sh
```

This script:

* Sources the environment
* Activates your Python environment
* Builds Triton using your local LLVM

## Verify the Setup
### Step 1: Verify Build Tools
Check that the build was successful by locating key tools:
```bash
# Find the main developer tool
find . -name linalg-hexagon-opt
# Expected output: ./triton/build/cmake.linux-x86_64-cpython-${PYTHON_VERSION}/third_party/qcom_hexagon_backend/bin/linalg-hexagon-opt

# Verify tool is accessible
which linalg-hexagon-opt
```

### Step 2: Run LIT Tests
```bash
lit triton/build/cmake.linux-x86_64-cpython-${PYTHON_VERSION}/third_party/qcom_hexagon_backend/test/
```

*NOTE*: Steps 3 and 4 are only applicable if you have access to a Qualcomm Hexagon NPU device.
If you do not have access to such a device, you can skip these steps; however, please contact us at [hexagon-mlir.support@qti.qualcomm.com](mailto:hexagon-mlir.support@qti.qualcomm.com) for assistance in obtaining access to a test device.

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
