#!/usr/bin/env python3

# ===- lower-libdevice-scalar-python.py -------------------------------------===
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause.
# For more license information:
#   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
#
# ===------------------------------------------------------------------------===
#
# Regression test for commit 5906d30d: LowerScalarMathPattern in
# LowerLibdevicePass handles bare scalar math.* ops produced when
# TritonReorderBroadcast sinks tt.splat past a math op on a 1-element tensor.
#
# Without the fix, scalar math.sin/cos/etc lower to llvm.intr.sin → sinf@plt
# which is unmapped in the DSP JIT and triggers a ~14 s DSP watchdog timeout
# (Error code: 500).
#
# This driver is modeled on rewrite-i64-ptr-args-to-i32-python.py.
# FileCheck directives live in lower-libdevice-scalar-python.mlir.

from triton._C.libtriton import ir, qcom_hexagon_backend


def run(mlir_str):
    ctx = ir.context()
    qcom_hexagon_backend.load_dialects(ctx)
    mod = qcom_hexagon_backend.parse_mlir_module_from_str(mlir_str, ctx)
    ok = qcom_hexagon_backend.run_lower_libdevice(mod)
    print(f"success={ok}")
    print(mod)


def main():
    # f32 scalar math.sin → qhmath_hvx_sin_af
    run(
        r"""
module {
  func.func @scalar_sin_f32(%arg0: f32) -> f32 {
    %0 = math.sin %arg0 : f32
    return %0 : f32
  }
}
"""
    )

    # f32 scalar math.cos → qhmath_hvx_cos_af
    run(
        r"""
module {
  func.func @scalar_cos_f32(%arg0: f32) -> f32 {
    %0 = math.cos %arg0 : f32
    return %0 : f32
  }
}
"""
    )

    # bf16 scalar: upconvert bf16→f32, call _af, downconvert f32→bf16
    run(
        r"""
module {
  func.func @scalar_sin_bf16(%arg0: bf16) -> bf16 {
    %0 = math.sin %arg0 : bf16
    return %0 : bf16
  }
}
"""
    )

    # f16 scalar math.sin → qhmath_hvx_sin_ahf
    run(
        r"""
module {
  func.func @scalar_sin_f16(%arg0: f16) -> f16 {
    %0 = math.sin %arg0 : f16
    return %0 : f16
  }
}
"""
    )

    # scalar math.sin inside linalg.generic — must NOT be rewritten
    run(
        r"""
module {
  func.func @scalar_sin_inside_linalg_untouched(%in: tensor<4xf32>, %out: tensor<4xf32>) -> tensor<4xf32> {
    %r = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]
    } ins(%in : tensor<4xf32>) outs(%out : tensor<4xf32>) {
    ^bb0(%a: f32, %b: f32):
      %s = math.sin %a : f32
      linalg.yield %s : f32
    } -> tensor<4xf32>
    return %r : tensor<4xf32>
  }
}
"""
    )

    # Scalar tt.extern_elementwise: f32 → qhmath_hvx_rsqrt_af
    run(
        r"""
module attributes {llvm.target_triple = "hexagon"} {
  func.func @scalar_rsqrt_f32(%arg0: f32) -> f32 {
    %0 = tt.extern_elementwise %arg0 {libname = "", libpath = "", pure = true, symbol = "qhmath_hvx_rsqrt_af"} : (f32) -> f32
    return %0 : f32
  }
}
"""
    )

    run(
        r"""
module attributes {llvm.target_triple = "hexagon"} {
  func.func @scalar_rsqrt_bf16(%arg0: bf16) -> bf16 {
    %0 = tt.extern_elementwise %arg0 {libname = "", libpath = "", pure = true, symbol = "qhmath_hvx_rsqrt_af"} : (bf16) -> bf16
    return %0 : bf16
  }
}
"""
    )

    run(
        r"""
module attributes {llvm.target_triple = "hexagon"} {
  func.func @scalar_rsqrt_fp8(%arg0: f8E4M3FN) -> f8E4M3FN {
    %0 = tt.extern_elementwise %arg0 {libname = "", libpath = "", pure = true, symbol = "qhmath_hvx_rsqrt_ahf"} : (f8E4M3FN) -> f8E4M3FN
    return %0 : f8E4M3FN
  }
}
"""
    )

    # f8E5M2 scalar math.sin → extf f8E5M2→f16, call _ahf, truncf f16→f8E5M2
    run(
        r"""
module {
  func.func @scalar_sin_f8e5m2(%arg0: f8E5M2) -> f8E5M2 {
    %0 = math.sin %arg0 : f8E5M2
    return %0 : f8E5M2
  }
}
"""
    )

    # f8E5M2FNUZ scalar math.sin → extf f8E5M2FNUZ→f16, call _ahf, truncf f16→f8E5M2FNUZ
    run(
        r"""
module {
  func.func @scalar_sin_f8e5m2fnuz(%arg0: f8E5M2FNUZ) -> f8E5M2FNUZ {
    %0 = math.sin %arg0 : f8E5M2FNUZ
    return %0 : f8E5M2FNUZ
  }
}
"""
    )

    # f8E5M2 scalar tt.extern_elementwise → extf f8E5M2→f16, call _ahf, truncf f16→f8E5M2
    run(
        r"""
module attributes {llvm.target_triple = "hexagon"} {
  func.func @scalar_rsqrt_f8e5m2(%arg0: f8E5M2) -> f8E5M2 {
    %0 = tt.extern_elementwise %arg0 {libname = "", libpath = "", pure = true, symbol = "qhmath_hvx_rsqrt_ahf"} : (f8E5M2) -> f8E5M2
    return %0 : f8E5M2
  }
}
"""
    )

    # f8E5M2FNUZ scalar tt.extern_elementwise → extf f8E5M2FNUZ→f16, call _ahf, truncf f16→f8E5M2FNUZ
    run(
        r"""
module attributes {llvm.target_triple = "hexagon"} {
  func.func @scalar_rsqrt_f8e5m2fnuz(%arg0: f8E5M2FNUZ) -> f8E5M2FNUZ {
    %0 = tt.extern_elementwise %arg0 {libname = "", libpath = "", pure = true, symbol = "qhmath_hvx_rsqrt_ahf"} : (f8E5M2FNUZ) -> f8E5M2FNUZ
    return %0 : f8E5M2FNUZ
  }
}
"""
    )

    # Tensor path must remain unchanged (regression guard for the pre-existing
    # ranked-tensor lowering).
    run(
        r"""
module attributes {llvm.target_triple = "hexagon"} {
  func.func @tensor_rsqrt(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    %0 = tt.extern_elementwise %arg0 {libname = "", libpath = "", pure = true, symbol = "qhmath_hvx_rsqrt_af"} : (tensor<4xf32>) -> tensor<4xf32>
    return %0 : tensor<4xf32>
  }
}
"""
    )


if __name__ == "__main__":
    main()
