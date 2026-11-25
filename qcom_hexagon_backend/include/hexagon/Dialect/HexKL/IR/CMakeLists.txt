# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause.
# For more license information:
#   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
#

set(LLVM_TARGET_DEFINITIONS HexKLDialect.td)
mlir_tablegen(HexKLDialect.h.inc -gen-dialect-decls -dialect=hexkl)
mlir_tablegen(HexKLDialect.cpp.inc -gen-dialect-defs -dialect=hexkl)

set(LLVM_TARGET_DEFINITIONS HexKLOps.td)
mlir_tablegen(HexKLOps.h.inc -gen-op-decls)
mlir_tablegen(HexKLOps.cpp.inc -gen-op-defs)

add_public_tablegen_target(HexKLTableGen)
