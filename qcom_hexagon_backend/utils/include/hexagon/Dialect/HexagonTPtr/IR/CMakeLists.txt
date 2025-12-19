set(LLVM_TARGET_DEFINITIONS HexagonTPtrDialect.td)
mlir_tablegen(HexagonTPtrDialect.h.inc -gen-dialect-decls -dialect=tptr)
mlir_tablegen(HexagonTPtrDialect.cpp.inc -gen-dialect-defs -dialect=tptr)

set(LLVM_TARGET_DEFINITIONS HexagonTPtrOps.td)
mlir_tablegen(HexagonTPtrOps.h.inc -gen-op-decls)
mlir_tablegen(HexagonTPtrOps.cpp.inc -gen-op-defs)

set(LLVM_TARGET_DEFINITIONS HexagonTPtrTypes.td)
mlir_tablegen(HexagonTPtrTypes.h.inc -gen-typedef-decls -typedefs-dialect=tptr)
mlir_tablegen(HexagonTPtrTypes.cpp.inc -gen-typedef-defs -typedefs-dialect=tptr)

set(LLVM_TARGET_DEFINITIONS HexagonTPtrAttributes.td)
mlir_tablegen(HexagonTPtrAttributes.h.inc -gen-attrdef-decls -attrdefs-dialect=tptr)
mlir_tablegen(HexagonTPtrAttributes.cpp.inc -gen-attrdef-defs -attrdefs-dialect=tptr)

add_public_tablegen_target(HexagonTPtrTableGen)
