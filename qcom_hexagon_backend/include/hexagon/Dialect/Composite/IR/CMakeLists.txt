# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause.
# For more license information:
#   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
#

set(LLVM_TARGET_DEFINITIONS CompositeDialect.td)
mlir_tablegen(CompositeDialect.h.inc -gen-dialect-decls -dialect=composite)
mlir_tablegen(CompositeDialect.cpp.inc -gen-dialect-defs -dialect=composite)

set(LLVM_TARGET_DEFINITIONS CompositeOps.td)
mlir_tablegen(CompositeOps.h.inc -gen-op-decls)
mlir_tablegen(CompositeOps.cpp.inc -gen-op-defs)

add_public_tablegen_target(CompositeTableGen)
