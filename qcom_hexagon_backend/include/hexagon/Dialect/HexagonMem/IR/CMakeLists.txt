set(LLVM_TARGET_DEFINITIONS HexagonMemDialect.td)
mlir_tablegen(HexagonMemDialect.h.inc -gen-dialect-decls -dialect=hexagonmem)
mlir_tablegen(HexagonMemDialect.cpp.inc -gen-dialect-defs -dialect=hexagonmem)

set(LLVM_TARGET_DEFINITIONS HexagonMemOps.td)
mlir_tablegen(HexagonMemOps.h.inc -gen-op-decls)
mlir_tablegen(HexagonMemOps.cpp.inc -gen-op-defs)

set(LLVM_TARGET_DEFINITIONS HexagonMemTypes.td)
mlir_tablegen(HexagonMemTypes.h.inc -gen-typedef-decls)
mlir_tablegen(HexagonMemTypes.cpp.inc -gen-typedef-defs)
mlir_tablegen(HexagonMemEnums.h.inc -gen-enum-decls)
mlir_tablegen(HexagonMemEnums.cpp.inc -gen-enum-defs)
mlir_tablegen(HexagonMemAttrs.h.inc -gen-attrdef-decls)
mlir_tablegen(HexagonMemAttrs.cpp.inc -gen-attrdef-defs)

add_public_tablegen_target(HexagonMemTableGen)
