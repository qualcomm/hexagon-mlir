//===- HmxDialect.td - HMX Dialect -------------------------*- tablegen -*-===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
// Describes the hmx dialect: the matrix engine's innermost operations.
//
// Scope (docs/hmx/hmx-system-design.md, ADR-001): LEAF granularity. The
// tile loop, tiling, VTCM budgeting and pipelining stay in MLIR; this dialect
// exposes only the innermost operations, which the device runtime implements
// with the HMX builtins (`llvm.hexagon.M8.*`). Those builtins are only lowered
// by the SDK's clang, never by our own LLVM, which is why the leaves ship as a
// prebuilt static library and MLIR can only reach them through `llvm.call`.
//
// The ops are memref based and therefore post-bufferization: operands are typed
// views (a crouton `memref<16x32x2xf16, 1>` and the runtime-acquired VTCM
// region) rather than raw addresses, so layout and memory space stay visible in
// the IR. Turning them into the address-only runtime ABI is HmxToLLVM's job.
//===----------------------------------------------------------------------===//

#ifndef HEXAGON_DIALECT_HMX_IR_HMX_DIALECT_TD
#define HEXAGON_DIALECT_HMX_IR_HMX_DIALECT_TD

include "mlir/IR/OpBase.td"

def Hmx_Dialect : Dialect {
  let name = "hmx";

  let cppNamespace = "::mlir::hmx";

  let summary = "HMX matrix-engine operations";

  let description = [{
    The `hmx` dialect models the Hexagon v79+ matrix engine (HMX).

    It replaces the `hexkl.micro_hmx_*` path (ADR-001) because that path cannot
    work: it has no environment or bias setup (the three mandatory prerequisites
    are missing), it reads the accumulator with the split form that zeroes
    negative values on v79 silicon, and it is implemented by a library the
    community SDK does not ship.

    The dialect is deliberately NOT named `hexkl` and does not reuse its op
    names, so the zero-investment `hexkl` path stays inert and untouched.
  }];

  let dependentDialects = [
    "mlir::memref::MemRefDialect",
  ];

  // The dialect defines attributes (`#hmx.crouton`); without this the generated
  // dialect has no attribute parsing/printing hook and every `#hmx.` attribute
  // fails to parse.
  let useDefaultAttributePrinterParser = 1;

  let extraClassDeclaration = [{
    void registerOperations();
    void registerAttributes();
  }];
}

#endif // HEXAGON_DIALECT_HMX_IR_HMX_DIALECT_TD
