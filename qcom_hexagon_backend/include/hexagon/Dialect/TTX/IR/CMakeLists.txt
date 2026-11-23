# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause.
# For more license information:
#   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
#

set(LLVM_TARGET_DEFINITIONS TTXDialect.td)
mlir_tablegen(TTXDialect.h.inc -gen-dialect-decls -dialect=ttx)
mlir_tablegen(TTXDialect.cpp.inc -gen-dialect-defs -dialect=ttx)

set(LLVM_TARGET_DEFINITIONS TTXOps.td)
mlir_tablegen(TTXOps.h.inc -gen-op-decls)
mlir_tablegen(TTXOps.cpp.inc -gen-op-defs)

set(LLVM_TARGET_DEFINITIONS TTXTypes.td)
mlir_tablegen(TTXTypes.h.inc -gen-typedef-decls)
mlir_tablegen(TTXTypes.cpp.inc -gen-typedef-defs)
mlir_tablegen(TTXEnums.h.inc -gen-enum-decls)
mlir_tablegen(TTXEnums.cpp.inc -gen-enum-defs)
mlir_tablegen(TTXAttrs.h.inc -gen-attrdef-decls)
mlir_tablegen(TTXAttrs.cpp.inc -gen-attrdef-defs)

add_public_tablegen_target(TTXTableGen)
