#!/usr/bin/env bash
set -euo pipefail

echo "Configuring git for submodules"

# Add submodules if missing
if [ ! -d "triton" ]; then
  git submodule add --force git@github.qualcomm.com:ai-tools/triton.git triton
fi

if [ ! -d "triton_shared" ]; then
  git submodule add --force https://github.com/microsoft/triton-shared triton_shared
fi

# Initialize and fetch
git submodule update --init --recursive

# Checkout specific commits
(
  cd triton
  git fetch --all
  git checkout 66daa234f28cb316704688130401a7ec6cbdda78
)

(
  cd triton_shared
  git fetch --all
  git checkout 2b728ad97bc02af821a0805b09075838911d4c19
)

echo "Submodules triton and triton_shared initialized and pinned."
