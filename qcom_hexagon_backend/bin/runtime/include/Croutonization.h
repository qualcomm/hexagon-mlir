//===- Croutonization.h - routines for croutonization   -------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// Contains routines for croutonization and decroutonization of buffers.
//
//===----------------------------------------------------------------------===//

#ifndef HEXAGON_BIN_RUNTIME_INC_CROUTONIZATION_H
#define HEXAGON_BIN_RUNTIME_INC_CROUTONIZATION_H

#include <cstdint>

extern "C" void _hexagon_runtime_load_crouton_b(uint8_t *buf, uint32_t depth,
                                                uint32_t width, uint32_t height,
                                                uint8_t **crouton_table,
                                                uint32_t crouton_count,
                                                int layout);
extern "C" void _hexagon_runtime_load_crouton_h(uint16_t *buf, uint32_t depth,
                                                uint32_t width, uint32_t height,
                                                uint8_t **crouton_table,
                                                uint32_t crouton_count,
                                                int layout);
extern "C" void
_hexagon_runtime_store_crouton_b(uint8_t *buf, uint32_t depth, uint32_t width,
                                 uint32_t height, uint8_t **crouton_table,
                                 uint32_t crouton_count, int layout);
extern "C" void
_hexagon_runtime_store_crouton_h(uint16_t *buf, uint32_t depth, uint32_t width,
                                 uint32_t height, uint8_t **crouton_table,
                                 uint32_t crouton_count, int layout);

#endif // HEXAGON_BIN_RUNTIME_INC_CROUTONIZATION_H
