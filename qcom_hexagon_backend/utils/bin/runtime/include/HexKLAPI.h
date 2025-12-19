#ifndef HEXKLAPI_H_
#define HEXKLAPI_H_

#include "HAP_power.h"
#include <memory>
#include <stdint.h>

extern "C" {
int hexkl_matmul_f16f16_f32(int64_t n_row, int64_t n_col, int64_t n_inner,
                            float *restrict outM,
                            const _Float16 *restrict inAct,
                            const _Float16 *restrict inW);
}
#endif // HEXKLAPI_H_
