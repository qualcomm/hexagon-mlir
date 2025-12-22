//===- BufferizableOpInterfaceImpl.h - Impl. of BufferizableOpInterface  --===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

#ifndef HEXAGON_DIALECT_HEXKL_BUFFERIZABLEOPINTERFACEIMPL_H
#define HEXAGON_DIALECT_HEXKL_BUFFERIZABLEOPINTERFACEIMPL_H

namespace mlir {
class DialectRegistry;

namespace hexkl {
void registerBufferizableOpInterfaceExternalModels(DialectRegistry &registry);
} // namespace hexkl
} // namespace mlir

#endif // HEXAGON_DIALECT_HEXKL_BUFFERIZABLEOPINTERFACEIMPL_H
