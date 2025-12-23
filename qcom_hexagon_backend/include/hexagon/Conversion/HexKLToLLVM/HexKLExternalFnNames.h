//===- HexKLExternalFnNames.h - Get runtime names for hexkl ops -----------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
#ifndef HEXAGON_CONVERSION_HEXKLTOLLVM_HEXKLEXTERNALFNNAMES_H
#define HEXAGON_CONVERSION_HEXKLTOLLVM_HEXKLEXTERNALFNNAMES_H

#include <string>

namespace mlir {
namespace hexkl {

std::string getMatmulMicroFnName();
std::string getHmxConfigSizeFnName();
std::string getHmxSetupAccReadF16FnName();
std::string getHmxAccClearF16FnName();
std::string getHmxAccReadF16FnName();
std::string getHmxCopySubmatrixToF16FnName();
std::string getHmxRmToAhF16FnName();
std::string getHmxRmToWhF16FnName();
std::string getHmxMmF16FnName();
std::string getHmxAhToRmF16FnName();
std::string getHmxCopyF16ToF32SubmatrixFnName();

} // namespace hexkl
} // namespace mlir

#endif // HEXAGON_CONVERSION_HEXKLTOLLVM_HEXKLEXTERNALFNNAMES_H
