/* ========================================================================== */
/* Contains routines for croutonization and decroutonization of buffers.      */
/* ========================================================================== */

#ifndef CROUTONIZATION_H
#define CROUTONIZATION_H

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

#endif // CROUTONIZATION_H
