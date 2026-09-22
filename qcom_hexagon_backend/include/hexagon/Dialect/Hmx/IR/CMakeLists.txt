set(LLVM_TARGET_DEFINITIONS HmxDialect.td)
mlir_tablegen(HmxDialect.h.inc -gen-dialect-decls -dialect=hmx)
mlir_tablegen(HmxDialect.cpp.inc -gen-dialect-defs -dialect=hmx)

set(LLVM_TARGET_DEFINITIONS HmxOps.td)
mlir_tablegen(HmxOps.h.inc -gen-op-decls)
mlir_tablegen(HmxOps.cpp.inc -gen-op-defs)

set(LLVM_TARGET_DEFINITIONS HmxAttrs.td)
mlir_tablegen(HmxAttrs.h.inc -gen-attrdef-decls -attrdefs-dialect=hmx)
mlir_tablegen(HmxAttrs.cpp.inc -gen-attrdef-defs -attrdefs-dialect=hmx)

add_public_tablegen_target(HmxTableGen)
