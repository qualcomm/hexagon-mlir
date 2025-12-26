//===- OptionsParsing.h - helper functions to parse options ---------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

#ifndef HEXAGON_TRANSFORMS_OPTIONSPARSING_H
#define HEXAGON_TRANSFORMS_OPTIONSPARSING_H
#include "mlir/IR/Operation.h"
#include <optional>
#include <string>

namespace mlir::hexagon {

std::optional<SmallVector<int64_t>> parseTileSizes(std::string tileSizesStr);

} // namespace mlir::hexagon

#endif // HEXAGON_TRANSFORMS_OPTIONSPARSING_H
