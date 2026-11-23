//===-- triton_qcom_hexagon_backend_api.h ---------------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

#pragma once

#include "mlir/IR/MLIRContext.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/IR/Module.h"
#include <pybind11/buffer_info.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <string>

namespace hexagon_backend {
std::vector<std::pair<int, mlir::Type>> getReturnList(mlir::ModuleOp module_op,
                                                      const std::string &fName);
std::string extractSingleFuncName(mlir::ModuleOp &module_op);
// Named-attribute readback for decompose-pass-internal outputs, surfaced to
// the wrapper generator.
int64_t getFuncI64Attr(mlir::ModuleOp module_op, const std::string &fName,
                       const std::string &attrName, int64_t defaultValue);
std::string getFuncStrAttr(mlir::ModuleOp module_op, const std::string &fName,
                           const std::string &attrName,
                           const std::string &defaultValue);
void loadDialects(mlir::MLIRContext &context);

mlir::ModuleOp parseMlirFromFile(const std::string &path,
                                 mlir::MLIRContext &context);
mlir::ModuleOp parseMlirFromString(const std::string &src,
                                   mlir::MLIRContext &context);

std::vector<std::vector<char>> translateLinalgToObj(
    mlir::ModuleOp &linalg_module,
    const std::unordered_map<std::string, std::string> &options_map);

std::string translateLinalgToLLVMIR(
    mlir::ModuleOp &linalg_module,
    const std::unordered_map<std::string, std::string> &options_map);

bool reorderFuncArgsAndCallsTensorFirst(mlir::ModuleOp &module_op,
                                        const std::string &fname);
bool rewriteI64PtrArgsToI32(mlir::ModuleOp &module_op);
void eliminateSafeBoundaryChecks(mlir::ModuleOp &module_op);
std::unique_ptr<mlir::Pass> createEliminateSafeBoundaryChecksPass();
void rewriteMaskedAccessToInboundsFastPath(mlir::ModuleOp &module_op);
std::unique_ptr<mlir::Pass> createRewriteMaskedAccessToInboundsFastPathPass();
// Runs only the lower-libdevice pass; returns true on success. Used by
// the python-driven lit regression for scalar tt.extern_elementwise lowering.
bool runLowerLibdevice(mlir::ModuleOp &module_op);
} // namespace hexagon_backend
