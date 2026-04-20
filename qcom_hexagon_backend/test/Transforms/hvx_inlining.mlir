// RUN: linalg-hexagon-translate %s --enable-hvx-inlining=false | FileCheck %s --check-prefix=CHECK-NO-INLINE
// RUN: linalg-hexagon-translate %s --enable-hvx-inlining=true | FileCheck %s --check-prefix=CHECK-INLINE

// When inlining is disabled, we should see calls to _hexagon_runtime_* functions
// CHECK-NO-INLINE: call <32 x i32> @_hexagon_runtime_exp__vf
// CHECK-NO-INLINE: call <32 x i32> @_hexagon_runtime_sin__vf
// CHECK-NO-INLINE: call <32 x i32> @_hexagon_runtime_cos__vf
// CHECK-NO-INLINE: call <32 x i32> @_hexagon_runtime_tan__vf
// CHECK-NO-INLINE: call <32 x i32> @_hexagon_runtime_tanh__vf
// CHECK-NO-INLINE: call <32 x i32> @_hexagon_runtime_acos__vf
// CHECK-NO-INLINE: call <32 x i32> @_hexagon_runtime_asin__vf
// CHECK-NO-INLINE: call <32 x i32> @_hexagon_runtime_atan__vf
// CHECK-NO-INLINE: call <32 x i32> @_hexagon_runtime_sqrt__vf
// CHECK-NO-INLINE: call <32 x i32> @_hexagon_runtime_rsqrt__vf
// CHECK-NO-INLINE: call <32 x i32> @_hexagon_runtime_exp__vhf
// CHECK-NO-INLINE: call <32 x i32> @_hexagon_runtime_tanh__vhf
// CHECK-NO-INLINE: call <32 x i32> @_hexagon_runtime_acos__vhf
// CHECK-NO-INLINE: call <32 x i32> @_hexagon_runtime_asin__vhf
// CHECK-NO-INLINE: call <32 x i32> @_hexagon_runtime_atan__vhf
// CHECK-NO-INLINE: call <32 x i32> @_hexagon_runtime_sqrt__vhf
// CHECK-NO-INLINE: call <32 x i32> @_hexagon_runtime_rsqrt__vhf
// CHECK-NO-INLINE: call <32 x i32> @_hexagon_runtime_floor__vhf
// CHECK-NO-INLINE: call <32 x i32> @_hexagon_runtime_ceil__vhf
// CHECK-NO-INLINE: call <32 x i32> @_hexagon_runtime_floor__vsf
// CHECK-NO-INLINE: call <32 x i32> @_hexagon_runtime_ceil__vsf
// CHECK-NO-INLINE: call <32 x i32> @_hexagon_runtime_pow__vf
// CHECK-NO-INLINE: call <32 x i32> @_hexagon_runtime_pow__vhf

// When inlining is enabled, the runtime functions should be inlined and we shouldn't see the calls
// CHECK-INLINE-NOT: call {{.*}} @_hexagon_runtime_exp__vf
// CHECK-INLINE-NOT: call {{.*}} @_hexagon_runtime_sin__vf
// CHECK-INLINE-NOT: call {{.*}} @_hexagon_runtime_cos__vf
// CHECK-INLINE-NOT: call {{.*}} @_hexagon_runtime_tan__vf
// CHECK-INLINE-NOT: call {{.*}} @_hexagon_runtime_tanh__vf
// CHECK-INLINE-NOT: call {{.*}} @_hexagon_runtime_acos__vf
// CHECK-INLINE-NOT: call {{.*}} @_hexagon_runtime_asin__vf
// CHECK-INLINE-NOT: call {{.*}} @_hexagon_runtime_atan__vf
// CHECK-INLINE-NOT: call {{.*}} @_hexagon_runtime_sqrt__vf
// CHECK-INLINE-NOT: call {{.*}} @_hexagon_runtime_rsqrt__vf
// CHECK-INLINE-NOT: call {{.*}} @_hexagon_runtime_exp__vhf
// CHECK-INLINE-NOT: call {{.*}} @_hexagon_runtime_tanh__vhf
// CHECK-INLINE-NOT: call {{.*}} @_hexagon_runtime_acos__vhf
// CHECK-INLINE-NOT: call {{.*}} @_hexagon_runtime_asin__vhf
// CHECK-INLINE-NOT: call {{.*}} @_hexagon_runtime_atan__vhf
// CHECK-INLINE-NOT: call {{.*}} @_hexagon_runtime_sqrt__vhf
// CHECK-INLINE-NOT: call {{.*}} @_hexagon_runtime_rsqrt__vhf
// CHECK-INLINE-NOT: call {{.*}} @_hexagon_runtime_floor__vhf
// CHECK-INLINE-NOT: call {{.*}} @_hexagon_runtime_ceil__vhf
// CHECK-INLINE-NOT: call {{.*}} @_hexagon_runtime_floor__vsf
// CHECK-INLINE-NOT: call {{.*}} @_hexagon_runtime_ceil__vsf
// CHECK-INLINE-NOT: call {{.*}} @_hexagon_runtime_pow__vf
// CHECK-INLINE-NOT: call {{.*}} @_hexagon_runtime_pow__vhf

module attributes {llvm.target_triple = "hexagon"} {
  // Test function that calls all HVX runtime functions
  llvm.func @test_hvx_runtime_functions(%arg0: vector<32xi32>, %arg1: vector<32xi32>) -> vector<32xi32> {
    // Single argument functions - vf suffix
    %0 = llvm.call @_hexagon_runtime_exp__vf(%arg0) : (vector<32xi32>) -> vector<32xi32>
    %1 = llvm.call @_hexagon_runtime_sin__vf(%0) : (vector<32xi32>) -> vector<32xi32>
    %2 = llvm.call @_hexagon_runtime_cos__vf(%1) : (vector<32xi32>) -> vector<32xi32>
    %3 = llvm.call @_hexagon_runtime_tan__vf(%2) : (vector<32xi32>) -> vector<32xi32>
    %4 = llvm.call @_hexagon_runtime_tanh__vf(%3) : (vector<32xi32>) -> vector<32xi32>
    %5 = llvm.call @_hexagon_runtime_acos__vf(%4) : (vector<32xi32>) -> vector<32xi32>
    %6 = llvm.call @_hexagon_runtime_asin__vf(%5) : (vector<32xi32>) -> vector<32xi32>
    %7 = llvm.call @_hexagon_runtime_atan__vf(%6) : (vector<32xi32>) -> vector<32xi32>
    %8 = llvm.call @_hexagon_runtime_sqrt__vf(%7) : (vector<32xi32>) -> vector<32xi32>
    %9 = llvm.call @_hexagon_runtime_rsqrt__vf(%8) : (vector<32xi32>) -> vector<32xi32>
    
    // Single argument functions - vhf suffix
    %10 = llvm.call @_hexagon_runtime_exp__vhf(%9) : (vector<32xi32>) -> vector<32xi32>
    %11 = llvm.call @_hexagon_runtime_tanh__vhf(%10) : (vector<32xi32>) -> vector<32xi32>
    %12 = llvm.call @_hexagon_runtime_acos__vhf(%11) : (vector<32xi32>) -> vector<32xi32>
    %13 = llvm.call @_hexagon_runtime_asin__vhf(%12) : (vector<32xi32>) -> vector<32xi32>
    %14 = llvm.call @_hexagon_runtime_atan__vhf(%13) : (vector<32xi32>) -> vector<32xi32>
    %15 = llvm.call @_hexagon_runtime_sqrt__vhf(%14) : (vector<32xi32>) -> vector<32xi32>
    %16 = llvm.call @_hexagon_runtime_rsqrt__vhf(%15) : (vector<32xi32>) -> vector<32xi32>
    %17 = llvm.call @_hexagon_runtime_floor__vhf(%16) : (vector<32xi32>) -> vector<32xi32>
    %18 = llvm.call @_hexagon_runtime_ceil__vhf(%17) : (vector<32xi32>) -> vector<32xi32>
    
    // Single argument functions - vsf suffix
    %19 = llvm.call @_hexagon_runtime_floor__vsf(%18) : (vector<32xi32>) -> vector<32xi32>
    %20 = llvm.call @_hexagon_runtime_ceil__vsf(%19) : (vector<32xi32>) -> vector<32xi32>
    
    // Double argument functions - vf suffix
    %21 = llvm.call @_hexagon_runtime_pow__vf(%20, %arg1) : (vector<32xi32>, vector<32xi32>) -> vector<32xi32>
    
    // Double argument functions - vhf suffix
    %22 = llvm.call @_hexagon_runtime_pow__vhf(%21, %arg1) : (vector<32xi32>, vector<32xi32>) -> vector<32xi32>
    
    llvm.return %22 : vector<32xi32>
  }
  
  // Function declarations for all HVX runtime functions
  llvm.func @_hexagon_runtime_exp__vf(vector<32xi32>) -> vector<32xi32> attributes {sym_visibility = "private"}
  llvm.func @_hexagon_runtime_exp__vhf(vector<32xi32>) -> vector<32xi32> attributes {sym_visibility = "private"}
  llvm.func @_hexagon_runtime_pow__vf(vector<32xi32>, vector<32xi32>) -> vector<32xi32> attributes {sym_visibility = "private"}
  llvm.func @_hexagon_runtime_pow__vhf(vector<32xi32>, vector<32xi32>) -> vector<32xi32> attributes {sym_visibility = "private"}
  llvm.func @_hexagon_runtime_sin__vf(vector<32xi32>) -> vector<32xi32> attributes {sym_visibility = "private"}
  llvm.func @_hexagon_runtime_cos__vf(vector<32xi32>) -> vector<32xi32> attributes {sym_visibility = "private"}
  llvm.func @_hexagon_runtime_tan__vf(vector<32xi32>) -> vector<32xi32> attributes {sym_visibility = "private"}
  llvm.func @_hexagon_runtime_tanh__vf(vector<32xi32>) -> vector<32xi32> attributes {sym_visibility = "private"}
  llvm.func @_hexagon_runtime_tanh__vhf(vector<32xi32>) -> vector<32xi32> attributes {sym_visibility = "private"}
  llvm.func @_hexagon_runtime_acos__vf(vector<32xi32>) -> vector<32xi32> attributes {sym_visibility = "private"}
  llvm.func @_hexagon_runtime_acos__vhf(vector<32xi32>) -> vector<32xi32> attributes {sym_visibility = "private"}
  llvm.func @_hexagon_runtime_asin__vf(vector<32xi32>) -> vector<32xi32> attributes {sym_visibility = "private"}
  llvm.func @_hexagon_runtime_asin__vhf(vector<32xi32>) -> vector<32xi32> attributes {sym_visibility = "private"}
  llvm.func @_hexagon_runtime_atan__vf(vector<32xi32>) -> vector<32xi32> attributes {sym_visibility = "private"}
  llvm.func @_hexagon_runtime_atan__vhf(vector<32xi32>) -> vector<32xi32> attributes {sym_visibility = "private"}
  llvm.func @_hexagon_runtime_sqrt__vf(vector<32xi32>) -> vector<32xi32> attributes {sym_visibility = "private"}
  llvm.func @_hexagon_runtime_sqrt__vhf(vector<32xi32>) -> vector<32xi32> attributes {sym_visibility = "private"}
  llvm.func @_hexagon_runtime_rsqrt__vf(vector<32xi32>) -> vector<32xi32> attributes {sym_visibility = "private"}
  llvm.func @_hexagon_runtime_rsqrt__vhf(vector<32xi32>) -> vector<32xi32> attributes {sym_visibility = "private"}
  llvm.func @_hexagon_runtime_floor__vhf(vector<32xi32>) -> vector<32xi32> attributes {sym_visibility = "private"}
  llvm.func @_hexagon_runtime_floor__vsf(vector<32xi32>) -> vector<32xi32> attributes {sym_visibility = "private"}
  llvm.func @_hexagon_runtime_ceil__vhf(vector<32xi32>) -> vector<32xi32> attributes {sym_visibility = "private"}
  llvm.func @_hexagon_runtime_ceil__vsf(vector<32xi32>) -> vector<32xi32> attributes {sym_visibility = "private"}
}