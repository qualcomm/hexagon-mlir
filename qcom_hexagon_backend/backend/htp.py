# ===- htp.py ---------------------------------------------------------------===
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause.
# For more license information:
#   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
#
# ===------------------------------------------------------------------------===

from triton.backends.compiler import GPUTarget

import triton

PLAINTEXT_ARTIFACTS = {"ttir", "ttsharedir", "llir"}
BINARY_ARTIFACTS = {"o", "so"}


def _build_htp_attrs(kernel, signature):
    """Build tt.divisibility attrs for HTP AOT compilation.

    If specialize_ptr_alignment is True, all pointer args get tt.divisibility=128
    (assumed 128-byte aligned). Integer args present in constants get divisibility
    derived from their value.
    """
    attrs = {}
    for arg_name, type_str in signature.items():
        if arg_name not in kernel.arg_names:
            continue
        idx = kernel.arg_names.index(arg_name)
        path = (idx,)
        if isinstance(type_str, str) and type_str.startswith("*"):
            attrs[path] = [["tt.divisibility", 128]]
    return attrs


# For .so generation for hexagon-nn-v3 custom ops, must set optional argument file_ext="so"
def triton_htp_compile(
    kernel, signature, constants=None, file_ext="o", extra_options=None
):
    if constants is None:
        constants = {}
    if extra_options is None:
        extra_options = {}

    options = {
        "htp_kernel_gen": True,
        "target_artifact": file_ext,
        "enableConvertToHexagonmem": False,
        "enableVTCMTiling": False,
        "enableSplitReduceGeneric": True,
        "enableHVXInlining": True,
        "enableSCFLoopUnroll": False,
        "enableUnrollAndJam": True,
        "specializeKernelToArgAlignment": True,
        "enableFastExpApprox": True,
        "enableBroadcastedReductionToButterfly": True,
        "enableSetAliasScopes": True,
    }

    # Allow callers to override or extend Triton compile options
    options.update(extra_options)

    specialize = options.get("specializeKernelToArgAlignment", False)
    attrs = {}
    if specialize:
        attrs = _build_htp_attrs(kernel, signature)
    src = triton.compiler.ASTSource(
        fn=kernel, signature=signature, constexprs=constants, attrs=attrs
    )

    compiled_kernel_artifacts = triton.compile(
        src=src,
        target=GPUTarget("hexagon", 0, 0),
        options=options,
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
