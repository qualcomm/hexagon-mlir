set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls --name HmxToLLVM)
add_public_tablegen_target(HmxToLLVMConversionPassIncGen)
