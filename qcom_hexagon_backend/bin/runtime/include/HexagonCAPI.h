//===- HexagonCAPI.h - hexagon alloc-free runtime calls -------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// / The source pointer is a pointer to the base of memref.
//
//===----------------------------------------------------------------------===//

#ifndef HEXAGON_BIN_RUNTIME_INCLUDE_HEXAGONCAPI_H_
#define HEXAGON_BIN_RUNTIME_INCLUDE_HEXAGONCAPI_H_

#include "HexagonAPI.h"
extern "C" {
void *hexagon_runtime_alloc_1d(size_t bytes, uint64_t alignment, bool isVtcm);
void hexagon_runtime_free_1d(void *ptr);
void *hexagon_runtime_alloc_2d(size_t numBlocks, size_t blockSize,
                               uint64_t alignment, bool isVtcm);
void hexagon_runtime_free_2d(void *ptr);
void hexagon_runtime_copy(void *dst, void *src, size_t nbytes, bool isDVtcm,
                          bool isSrcVtcm);

/// The source pointer is a pointer to the base of memref
void *hexagon_runtime_build_crouton(void *source, size_t nbytes);
/// The source pointer is a pointer to crouton table
void *hexagon_runtime_get_contiguous_memref(void *source);

/// Allocate (once, per process) a resident VTCM buffer for the weight whose
/// compile-time image starts at `src` and is `bytes` long. The first call
/// allocates the buffer and copies the weight in; later calls return the same
/// address without copying. Resident buffers are not freed by the per-launch
/// deallocation, so a weight loaded from a `memref.global` is packed into VTCM
/// exactly once instead of once per launch.
void *hexagon_runtime_weight_resident(uint64_t src, size_t bytes);

/// Allocate (once, per process) a resident, uninitialised VTCM workspace buffer
/// for `key`. The first call allocates it; later calls return the same address.
/// Nothing is copied: the kernel refills the buffer itself on every launch.
/// Resident buffers are not freed by the per-launch deallocation, so a
/// workspace buffer is allocated exactly once per process instead of once per
/// launch.
void *hexagon_runtime_workspace_resident(uint64_t key, size_t bytes);

/// Power the HMX engine up and acquire it, without allocating anything else.
/// Emitted once at the entry of every function that issues HMX leaves; see the
/// device-side definition for why it cannot be skipped.
void hexagon_runtime_hmx_ensure(void);

/// Unlock the HMX unit from the current thread. Emitted before every return of a
/// function that issues HMX leaves, paired with the ensure at its entry; see the
/// device-side definition. Only unlocks when this thread holds the lock
/// (thread_local flag), so a call without a matching ensure is a no-op.
void hexagon_runtime_hmx_unlock(void);

#ifdef HEXMLIR_RUNTIME_BRINGUP_PROBE
/// Format the recorded per-step bring-up timings into `buf` as "BRINGUP ..."
/// lines and return the number of bytes written (0 when the bring-up never ran
/// in this process). Defined in HexagonAPI.cpp; probe build only. See
/// exp/hmx/bringup_probe.
int hexagon_runtime_bringup_report(char *buf, int cap);
#endif
}
#endif // HEXAGON_BIN_RUNTIME_INCLUDE_HEXAGONCAPI_H_
