#!/bin/bash

# By default we use upstream LLVM.
# Usage:
#   To build triton with a prebuilt version of upstream LLVM:
#     ./local_build.sh
#       or
#     ./local_build.sh --upstream-llvm
#   To build triton with a prebuilt version of downstream LLVM:
#     ./local_build.sh --downstream-llvm
#   To build triton with a user built version of LLVM:
#     ./local_build.sh -p <path_to_llvm_install_dir>

set -x
export HEXAGON_MLIR_ROOT=$PWD
export TRITON_ROOT=$HEXAGON_MLIR_ROOT/triton

exit_with_usage() {

cat << EOF
  Usage: ./local_build.sh
         ./local_build.sh [--downstream-llvm | --upstream-llvm]
         ./local_build.sh [-p <filepath>]

  Options:
        --downstream-llvm : Use prebuilt downstream LLVM
        --upstream-llvm   : Use prebuilt upstream LLVM
        -p <path> : Path to custom LLVM install directory

  Error: Issue with the arguments passed
EOF

  if [[ "${BASH_SOURCE[0]}" != "${0}" ]]; then
    set +x
    return 1  # Script is sourced
  else
    exit 1  # Script is executed
  fi
}

use_upstream_llvm=true
use_downstream_llvm=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --upstream-llvm)
            use_downstream_llvm=false
            use_upstream_llvm=true
            shift 1
            ;;
        --downstream-llvm)
            use_downstream_llvm=true
            use_upstream_llvm=false
            shift 1
            ;;
        -p)
            if [[ -z "$2" ]]; then
                echo "ERROR: No path provided for -p option."
                exit_with_usage
                return 1
            fi
            LLVM_INSTALL_DIR=$2
            use_downstream_llvm=false
            use_upstream_llvm=false
            shift 2
            ;;
        *)
            shift 1
            exit_with_usage
            return 1
            ;;
    esac
done 

# Check if the LLVM install directory path is present
if [ -z "$1" ]; then
    # Default prebuilt LLVM
    if [ "$use_downstream_llvm" == true ]; then
        # Default prebuilt downstream LLVM
        LLVM_SHA=$(cat $HEXAGON_MLIR_ROOT/llvm-hash-downstream.txt)
        LLVM_INSTALL_DIR=/prj/qct/llvm/devops/build_tools/triton/llvm-project-downstream/$LLVM_SHA
    elif [ "$use_upstream_llvm" == true ]; then
        # Default prebuilt upstream LLVM
        LLVM_SHA=$(cat $TRITON_ROOT/cmake/llvm-hash.txt)
        LLVM_INSTALL_DIR=/prj/qct/llvm/devops/build_tools/triton/llvm-project-upstream/$LLVM_SHA
    fi
else
    exit_with_usage
fi

echo "Using LLVM from path: $LLVM_INSTALL_DIR"
if [ ! -d "$LLVM_INSTALL_DIR" ]; then
    echo "ERROR: Cannot find LLVM install directory at $LLVM_INSTALL_DIR"
    exit 1
fi

source $HEXAGON_MLIR_ROOT/local_env.sh

# Setup the host toolchain
export HOST_TOOLCHAIN=/pkg/qct/software/llvm/build_tools/clang+llvm-13.0.1-x86_64-linux-gnu-ubuntu-18.04
export PATH=$HOST_TOOLCHAIN/bin:$PATH
# Unsetting LD_LIBRARY_PATH for consistency across setups (although this is not necessary)
export LD_LIBRARY_PATH=

# Build Triton using upstream LLVM
cd $TRITON_ROOT
echo Building triton
# Note: if getting compilations errors after pulling the most recent changes, temporary turn off TRITON_BUILD_WITH_CCACHE to make sure everything is completely rebuilt
TRITON_BUILD_WITH_CLANG_LLD=1 \
    TRITON_BUILD_WITH_CCACHE=true \
    LLVM_INCLUDE_DIRS=$LLVM_INSTALL_DIR/include \
    LLVM_LIBRARY_DIR=$LLVM_INSTALL_DIR/lib \
    LLVM_SYSPATH=$LLVM_INSTALL_DIR \
    pip3 install -e . --no-build-isolation --verbose
if [ $? -ne 0 ]; then
    echo Building hexagon-mlir failed
else
    echo hexagon-mlir successfully built
fi

cd $HEXAGON_MLIR_ROOT

set +x
