//===- HexagonMemExternalFnNames.h - Get runtime names for hexagonmem ops -===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

#ifndef HEXAGON_CONVERSION_HEXAGONMEMTOLLVM_HEXAGONMEMEXTERNALFNNAMES_H
#define HEXAGON_CONVERSION_HEXAGONMEMTOLLVM_HEXAGONMEMEXTERNALFNNAMES_H

#include <mlir/IR/Types.h>
#include <string>

namespace mlir {
namespace hexagonmem {

std::string getAllocFnName(bool isCroutonType,
                           const std::string &deviceType = "hexagon");
std::string getDeallocFnName(bool isCroutonType,
                             const std::string &deviceType = "hexagon");
std::string getCopyFnName(const std::string &deviceType = "hexagon");
std::string getMemrefToCroutonFnName(const std::string &deviceType = "hexagon");
std::string getCroutonToMemrefFnName(const std::string &deviceType = "hexagon");

} // namespace hexagonmem
} // namespace mlir

#endif // HEXAGON_CONVERSION_HEXAGONMEMTOLLVM_HEXAGONMEMEXTERNALFNNAMES_H
