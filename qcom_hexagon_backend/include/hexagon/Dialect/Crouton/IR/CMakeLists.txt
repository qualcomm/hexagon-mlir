set(LLVM_TARGET_DEFINITIONS CroutonDialect.td)
mlir_tablegen(CroutonDialect.h.inc -gen-dialect-decls -dialect=crouton)
mlir_tablegen(CroutonDialect.cpp.inc -gen-dialect-defs -dialect=crouton)

set(LLVM_TARGET_DEFINITIONS CroutonTypes.td)
mlir_tablegen(CroutonTypes.h.inc -gen-typedef-decls)
mlir_tablegen(CroutonTypes.cpp.inc -gen-typedef-defs)

add_public_tablegen_target(CroutonTableGen)
