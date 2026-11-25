# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause.
# For more license information:
#   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
#

set(LLVM_TARGET_DEFINITIONS CroutonDialect.td)
mlir_tablegen(CroutonDialect.h.inc -gen-dialect-decls -dialect=crouton)
mlir_tablegen(CroutonDialect.cpp.inc -gen-dialect-defs -dialect=crouton)

set(LLVM_TARGET_DEFINITIONS CroutonTypes.td)
mlir_tablegen(CroutonTypes.h.inc -gen-typedef-decls)
mlir_tablegen(CroutonTypes.cpp.inc -gen-typedef-defs)

add_public_tablegen_target(CroutonTableGen)
