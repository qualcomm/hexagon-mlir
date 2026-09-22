set(LLVM_TARGET_DEFINITIONS HvxDialect.td)
mlir_tablegen(HvxDialect.h.inc -gen-dialect-decls -dialect=hvx)
mlir_tablegen(HvxDialect.cpp.inc -gen-dialect-defs -dialect=hvx)

set(LLVM_TARGET_DEFINITIONS HvxOps.td)
mlir_tablegen(HvxOps.h.inc -gen-op-decls)
mlir_tablegen(HvxOps.cpp.inc -gen-op-defs)

add_public_tablegen_target(HvxTableGen)
