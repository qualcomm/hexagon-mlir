//===- DMATExternalFnNames.h - DMA external function names ------*- C++ -*-===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

#ifndef HEXAGON_CONVERSION_DMATOLLVM_DMAEXTERNALFNNAMES_H
#define HEXAGON_CONVERSION_DMATOLLVM_DMAEXTERNALFNNAMES_H

#include <mlir/IR/Types.h>
#include <string>

namespace mlir {
namespace hexagon {

std::string getDMAStartFnName();
std::string getDMA2DStartFnName();
std::string getDMAWaitFnName();

} // namespace hexagon
} // namespace mlir

#endif // HEXAGON_CONVERSION_DMATOLLVM_DMAEXTERNALFNNAMES_H
