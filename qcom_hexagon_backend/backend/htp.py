# ===- htp.py ---------------------------------------------------------------===
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause.
# For more license information:
#   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
#
# ===------------------------------------------------------------------------===

import triton
from triton.backends.compiler import GPUTarget

PLAINTEXT_ARTIFACTS = {"ttir", "ttsharedir", "llir"}
BINARY_ARTIFACTS = {"o", "so"}


# For .so generation for hexagon-nn-v3 custom ops, must set optional argument file_ext="so"
def triton_htp_compile(kernel, signature, constants={}, file_ext="o"):
    src = triton.compiler.ASTSource(
        fn=kernel, signature=signature, constexprs=constants
    )
    compiled_kernel_artifacts = triton.compile(
        src=src,
        target=GPUTarget("hexagon", 0, 0),
        options={
            "htp_kernel_gen": True,
            "target_artifact": file_ext,
            "enableConvertToHexagonmem": False,
            "enableVTCMTiling": False,
        },
    )
    compiled_kernel = compiled_kernel_artifacts.asm[file_ext]
    valid_extensions = PLAINTEXT_ARTIFACTS | BINARY_ARTIFACTS
    if file_ext in valid_extensions:
        # Ensure consistent return types: decode plaintext, return binary as-is
        if file_ext in PLAINTEXT_ARTIFACTS and isinstance(compiled_kernel, bytes):
            return compiled_kernel.decode("utf-8")
        return compiled_kernel
    else:
        str_valid_extensions = ", ".join(sorted(valid_extensions))
        raise ValueError(
            f'Unsupported file extension "{file_ext}" in custom op workflow. '
            f"Valid extensions are: {str_valid_extensions}"
        )
