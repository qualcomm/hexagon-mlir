This document explains how to set up a **local development environment** for working on the Hexagon-MLIR + Triton integration.

# Overview 

Local development requires downloading several components: 

* [Hexagon SDK 6.4.0.2](https://softwarecenter.qualcomm.com/catalog/item/Hexagon_SDK)
* [Hexagon Tools 19.0.02](https://softwarecenter.qualcomm.com/catalog/item/Hexagon_open_access)
* [Hexagon Kernel Library (HexKL 1.0.0)](https://softwarecenter.qualcomm.com/catalog/item/Hexagon_KL)
* LLVM (Triton-compatible)
* Python Virtual Environment
* [triton](https://github.com/triton-lang/triton)
* [triton-shared](https://github.com/microsoft/triton-shared)

# Required Environment Variables

Export the following environment variables 

```bash
export HEXAGON_SDK_ROOT=/path/to/Hexagon_SDK/6.4.0.2
export HEXAGON_TOOLS=/path/to/Tools
export HEXKL_ROOT=/path/to/hexkl_addon
export LLVM_PROJECT_BUILD_dir=/path/to/llvm/build
export CONDA_ENV=/path/to/your/python/env
```

These variables are validated by `scripts/check_local_env.sh`

# Installing Required Components

## triton submodule
Set up the triton submodule and apply Qualcomm specific patches. 

```bash
REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "${REPO_ROOT}"
git submodule add --force https://github.com/triton-lang/triton.git triton
cd triton
git checkout e44bd1c83c1c3e8deac7c4f02683cfb3cc395c8b
git apply "${REPO_ROOT}/third_party_software/patches/triton/third_party_triton.patch"
```

## triton_shared submodule
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

## Hexagon SDK

```bash 
wget https://softwarecenter.qualcomm.com/api/download/software/sdks/Hexagon_SDK/Linux/Debian/6.4.0.2/Hexagon_SDK_lnx.zip
unzip -q Hexagon_SDK_lnx.zip 
export HEXAGON_SDK_ROOT=/path/to/Hexagon_SDK/6.4.0.2
```

## Hexagon Tools

```bash
wget https://softwarecenter.qualcomm.com/api/download/software/tools/Hexagon_open_access/Linux/Debian/19.0.02/Hexagon_open_access.Core.19.0.02.Linux-Any.tar.gz
tar -xzf Hexagon_open_access.Core.19.0.02.Linux-Any.tar.gz
export HEXAGON_TOOLS=/path/to/Tools
```

## Hexagon Kernel Library (HexKL)

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

## Download the clang toolchain

```bash
wget https://github.com/llvm/llvm-project/releases/download/llvmorg-13.0.1/clang+llvm-13.0.1-x86_64-linux-gnu-ubuntu-18.04.tar.xz
tar -xf clang+llvm-13.0.1-x86_64-linux-gnu-ubuntu-18.04.tar.xz --strip-components=1
export HOST_TOOLCHAIN=/path/to/clang+llvm-13.0.1-x86_64-linux-gnu-ubuntu-18.04
export PATH="${HOST_TOOLCHAIN}/bin:${PATH}"
export CC="${HOST_TOOLCHAIN}/bin/clang"
export CXX="${HOST_TOOLCHAIN}/bin/clang++"
```

## Building LLVM for Triton 

Triton requires a specific LLVM revision; and you can find the expected hash in `triton/cmake/llvm-hash.txt`

### Build Steps

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

## Creating a Python Environment

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

## Verifying your setup

```bash
./scripts/check_local_env.sh
```

This script verifies:

* All required environment variables
* All required directories
* LLVM build completeness (`mlir-opt` exists)
* triton and triton_shared submodule presence
* Python interpreter inside your environment

## Building Triton Locally

Once your environment is valid:

```bash
./scripts/build_local_triton.sh
```

This script:

* Sources the environment
* Activates your Python environment
* Builds Triton using your local LLVM