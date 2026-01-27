//===- HexKLExternalFnNames.cpp - HexKL external function names  ----------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

#include "hexagon/Conversion/HexKLToLLVM/HexKLExternalFnNames.h"

using namespace mlir;

std::string hexkl::getMatmulMicroFnName() {
  static const std::string matmulMicroFnName = "hexkl_matmul_f16f16_f32";
  return matmulMicroFnName;
}

std::string hexkl::getHmxConfigSizeFnName() {
  static const std::string fnName = "hexkl_micro_hmx_config_size";
  return fnName;
}

std::string hexkl::getHmxSetupAccReadF16FnName() {
  static const std::string fnName = "hexkl_micro_hmx_setup_acc_read_f16";
  return fnName;
}

std::string hexkl::getHmxAccClearF16FnName() {
  static const std::string fnName = "hexkl_micro_hmx_acc_clear_f16";
  return fnName;
}

std::string hexkl::getHmxAccReadF16FnName() {
  static const std::string fnName = "hexkl_micro_hmx_acc_read_f16";
  return fnName;
}

std::string hexkl::getHmxCopySubmatrixToF16FnName() {
  static const std::string fnName = "hexkl_micro_hmx_copy_submatrix_to_f16";
  return fnName;
}

std::string hexkl::getHmxRmToAhF16FnName() {
  static const std::string fnName = "hexkl_micro_hmx_rm_to_ah_f16";
  return fnName;
}

std::string hexkl::getHmxRmToWhF16FnName() {
  static const std::string fnName = "hexkl_micro_hmx_rm_to_wh_f16";
  return fnName;
}

std::string hexkl::getHmxMmF16FnName() {
  static const std::string fnName = "hexkl_micro_hmx_mm_f16";
  return fnName;
}

std::string hexkl::getHmxAhToRmF16FnName() {
  static const std::string fnName = "hexkl_micro_hmx_ah_to_rm_f16";
  return fnName;
}

std::string hexkl::getHmxCopyF16ToF32SubmatrixFnName() {
  static const std::string fnName = "hexkl_micro_hmx_copy_f16_to_f32_submatrix";
  return fnName;
}
