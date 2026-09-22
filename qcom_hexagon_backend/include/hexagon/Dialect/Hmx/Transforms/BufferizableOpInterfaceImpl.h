//===- BufferizableOpInterfaceImpl.h - Impl. of BufferizableOpInterface ---===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
#ifndef HEXAGON_DIALECT_HMX_TRANSFORMS_BUFFERIZABLEOPINTERFACEIMPL_H
#define HEXAGON_DIALECT_HMX_TRANSFORMS_BUFFERIZABLEOPINTERFACEIMPL_H

#include "mlir/IR/DialectRegistry.h"

namespace mlir {
namespace hmx {

void registerBufferizableOpInterfaceExternalModels(DialectRegistry &registry);

} // namespace hmx
} // namespace mlir

#endif // HEXAGON_DIALECT_HMX_TRANSFORMS_BUFFERIZABLEOPINTERFACEIMPL_H
