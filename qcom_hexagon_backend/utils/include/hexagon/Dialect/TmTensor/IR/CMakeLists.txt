set(LLVM_TARGET_DEFINITIONS TmTensorDialect.td)
mlir_tablegen(TmTensorDialect.h.inc -gen-dialect-decls -dialect=tm_tensor)
mlir_tablegen(TmTensorDialect.cpp.inc -gen-dialect-defs -dialect=tm_tensor)

set(LLVM_TARGET_DEFINITIONS TmTensorOps.td)
mlir_tablegen(TmTensorOps.h.inc -gen-op-decls)
mlir_tablegen(TmTensorOps.cpp.inc -gen-op-defs)

set(LLVM_TARGET_DEFINITIONS TmTensorTypes.td)
mlir_tablegen(TmTensorTypes.h.inc -gen-typedef-decls)
mlir_tablegen(TmTensorTypes.cpp.inc -gen-typedef-defs)
mlir_tablegen(TmTensorEnums.h.inc -gen-enum-decls)
mlir_tablegen(TmTensorEnums.cpp.inc -gen-enum-defs)
mlir_tablegen(TmTensorAttrs.h.inc -gen-attrdef-decls)
mlir_tablegen(TmTensorAttrs.cpp.inc -gen-attrdef-defs)

add_public_tablegen_target(TmTensorTableGen)
