module {
  llvm.func @fdiv_sqrt_pattern_scalar_f16(%arg0: f16) -> f16 {
    %sqrt_f16 = llvm.intr.sqrt(%arg0) {fastmathFlags = #llvm.fastmath<fast>} : (f16) -> f16
    %one_f16 = llvm.mlir.constant(1.0 : f16) : f16
    %rsqrt_f16 = llvm.fdiv %one_f16, %sqrt_f16 {fastmathFlags = #llvm.fastmath<fast>} : f16

    llvm.return %rsqrt_f16 : f16
  }

  llvm.func @fdiv_sqrt_pattern_scalar_f32(%arg1: f32) -> f32 {
    %sqrt_f32 = llvm.intr.sqrt(%arg1) {fastmathFlags = #llvm.fastmath<fast>} : (f32) -> f32
    %one_f32 = llvm.mlir.constant(1.0 : f32) : f32
    %rsqrt_f32 = llvm.fdiv %one_f32, %sqrt_f32 {fastmathFlags = #llvm.fastmath<fast>} : f32
    
    llvm.return %rsqrt_f32 : f32
  }
}
