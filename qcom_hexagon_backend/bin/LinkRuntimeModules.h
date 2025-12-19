//===- LinkRuntimeModules.h -----------------------------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// Declaration of functions to link runtime modules with the kernel.
//
//===----------------------------------------------------------------------===//

#ifndef HEXAGON_BIN_LINK_RUNTIME_MODULES_H
#define HEXAGON_BIN_LINK_RUNTIME_MODULES_H

#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>

namespace mlir::Hexagon::Translate {

void linkRuntimeModules(llvm::LLVMContext &ctx,
                        std::unique_ptr<llvm::Module> &module);

} // namespace mlir::Hexagon::Translate

#endif
