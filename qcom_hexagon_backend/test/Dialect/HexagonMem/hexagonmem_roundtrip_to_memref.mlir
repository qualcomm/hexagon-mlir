//===- hexagonmem_roundtrip_to_memref.mlir --------------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

// RUN: linalg-hexagon-opt %s | FileCheck %s

// CHECK:   func.func @crouton_to_memref(%[[VAL0:.+]]: !crouton.crouton<2x4x2xf16, vtcm>) -> memref<2x4x2x1024xf16, 1> {
// CHECK:     %[[VAL1:.+]] = hexagonmem.crouton_to_memref(%[[VAL0]] : <2x4x2xf16, vtcm>) : memref<2x4x2x1024xf16, 1>
// CHECK:     return %[[VAL1:.+]] : memref<2x4x2x1024xf16, 1>
// CHECK:   }

func.func @crouton_to_memref(%arg0: !crouton.crouton<2x4x2xf16, vtcm>) -> memref<2x4x2x1024xf16, 1> {
  %out = hexagonmem.crouton_to_memref (%arg0 : !crouton.crouton<2x4x2xf16, vtcm>) : memref<2x4x2x1024xf16, 1>
  return %out : memref<2x4x2x1024xf16, 1>
}

