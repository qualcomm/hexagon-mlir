//===- HMXMicroDaemon.h - HexKL micro-op HMX queue and daemon -------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// HMX micro-op daemon library (mobile/DSP path).
//
// A Hexagon NSP core ships with one HMX engine and one 32x32 FP16 accumulator.
// In this model (mobile/DSP HMX daemon):
// one dedicated thread owns HMX and drains a lock-free queue while the SPMD
// workers submit matmul jobs and never touch HMX.
//
// The lifecycle splits the daemon into acquire-once / serve-per-iteration /
// release-once phases so a benchmark loop pays the HMX acquire+lock exactly
// once instead of on every iteration. The daemon is a PERSISTENT pool thread,
// so it holds the thread-bound HMX lock across every serve() in between:
// clang-format off
//   wrapper:  acquire VTCM -> vtcmBase
//             config_offset = totalVtcmBytes - hexkl_micro_hmx_config_size()
//             hmx_micro_queue_init(storage, vtcmBase, config_offset)
//             | spawn (actual_threads + 1) threads
//             daemon thread: hmx_micro_daemon_setup(q)      // acquire+lock HMX
//             for each iteration:
//             |   wrapper:       hmx_micro_daemon_rearm(q)  // clear exit_signal
//             |   daemon thread: hmx_micro_daemon_serve(q)  // drain until done
//             |   workers:       layout -> hmx_micro_dispatch(q, wid, ...)
//             |                  ... -> hmx_micro_wait_idle(q, wid)
//             |   last worker:   hmx_micro_daemon_exit(q)   // ends this serve()
//             daemon thread: hmx_micro_daemon_teardown(q)   // unlock+release HMX
//             | join
//             v
//   wrapper:  release VTCM
// clang-format on
//
//===----------------------------------------------------------------------===//

#ifndef HEXAGONBACKEND_BIN_RUNTIME_HMXMICRODAEMON_H
#define HEXAGONBACKEND_BIN_RUNTIME_HMXMICRODAEMON_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
#include <atomic>
#define HMX_ATOMIC_UINT32 std::atomic<uint32_t>
#else
#include <stdatomic.h>
#define HMX_ATOMIC_UINT32 _Atomic uint32_t
#endif

#ifdef __cplusplus
extern "C" {
#endif

// Maximum worker lanes. The daemon probes only [0, num_lanes); callers must use
// worker IDs in that range and set num_lanes no greater than this limit.
#define HMX_MICRO_QUEUE_MAX_LANES 8

// Per-lane power-of-two ring depth. The lowering pass keeps its VTCM ring at
// least one slot deeper, so dispatch backpressure prevents WAR reuse.
#define HMX_MICRO_QUEUE_LANE_DEPTH 4

// Job accumulator mode. It travels with each job because the compiler knows the
// element type; the daemon lazily reprograms HMX when the mode changes.
typedef enum {
  HMX_MICRO_JOB_KIND_F16 = 0,
  HMX_MICRO_JOB_KIND_U8I8 = 1,
} HMXMicroJobKind;

// Independently accessed fields occupy separate cache lines. The implementation
// asserts this layout to prevent false sharing on 32-bit Hexagon.
#define HMX_MICRO_CACHE_LINE 64
#define HMX_MICRO_LINE_ALIGNED __attribute__((aligned(HMX_MICRO_CACHE_LINE)))

// One K=32 matmul job. Offsets are global VTCM byte offsets, so workers and the
// daemon use the same base without per-lane conversion.
typedef struct HMXMicroJob {
  // Accumulator control. `clear_acc` starts a new partial sum (first K-block of
  // a tile); `write_out` commits it via acc_read (last K-block of a tile).
  bool clear_acc;
  bool write_out;
  // Selects the f16 or u8i8 accumulator-mode runtime functions.
  uint8_t job_kind;

  // AH-laid-out activation block  -> hexkl_micro_hmx_mm_{f16,u8i8} arg1.
  uint32_t act_ah_offset;
  // WH-laid-out weight block      -> hexkl_micro_hmx_mm_{f16,u8i8} arg2.
  uint32_t wgt_wh_offset;
  // acc-read destination; consumed only when `write_out` is set.
  uint32_t out_offset;
} HMXMicroJob;

// A slot is valid when tail <= slot < head. The worker publishes its payload by
// release-storing head; the daemon acquire-loads head. Each slot has one line.
typedef struct HMX_MICRO_LINE_ALIGNED HMXMicroJobSlot {
  HMXMicroJob job;
} HMXMicroJobSlot;

// One per worker thread. Workers write `head`; the daemon writes `tail`. Each
// gets its own line so the two counters never false-share.
typedef struct HMXMicroLane {
  HMX_MICRO_LINE_ALIGNED HMX_ATOMIC_UINT32 head; // worker writes; daemon reads
  HMX_MICRO_LINE_ALIGNED HMX_ATOMIC_UINT32 tail; // daemon writes; worker reads
  HMX_MICRO_LINE_ALIGNED HMXMicroJobSlot slots[HMX_MICRO_QUEUE_LANE_DEPTH];
} HMXMicroLane;

typedef struct HMXMicroQueue {
  // Written before threads start and read-only thereafter.
  //   vtcm_base     : per-core HAP least address; base for all job offsets and
  //                   for the daemon's setup_acc_read / mm / acc_read calls.
  //   config_offset : byte offset of the HMX config block (carved off the VTCM
  //                   tail, matching the single-threaded lowering).
  //   num_lanes     : active worker count; the daemon probes only lanes
  //                   [0, num_lanes), skipping the unused tail of the array.
  void *vtcm_base;
  uint32_t config_offset;
  uint32_t num_lanes;

  // Lifecycle handshake (hot): a worker signals exit by storing 1; the daemon
  // returns once the signal is set AND every lane is empty.
  HMX_MICRO_LINE_ALIGNED HMX_ATOMIC_UINT32 exit_signal;

  HMX_MICRO_LINE_ALIGNED HMXMicroLane lanes[HMX_MICRO_QUEUE_MAX_LANES];
} HMXMicroQueue;

#undef HMX_ATOMIC_UINT32

//===----------------------------------------------------------------------===//
// Setup APIs (called by the wrapper, never by the kernel)
//===----------------------------------------------------------------------===//

// Queue storage size. The wrapper allocates a 64-byte-aligned cached buffer;
// VTCM remains available for worker slices and the HMX config.
size_t hmx_micro_queue_size_bytes(void);

// Zeroes every counter, slot, and the exit signal, stores {vtcm_base,
// config_offset, num_lanes}, then issues a release fence. Returns `mem`
// reinterpreted as `HMXMicroQueue *`. Does NOT touch HMX: setup_acc_read
// programs the HMX unit and is therefore deferred to the daemon thread (see
// hmx_micro_daemon_serve), because the HMX unit is locked to whichever thread
// runs it. `vtcm_base` MUST be the HAP least address so `vtcm_base +
// global_offset` resolves any tile. `num_lanes` is the active worker count
// (<= HMX_MICRO_QUEUE_MAX_LANES); the daemon probes only lanes [0, num_lanes).
// The accumulator mode is not a parameter: each job carries its own kind, so
// the wrapper does not need to know the kernel's element type.
HMXMicroQueue *hmx_micro_queue_init(void *mem, void *vtcm_base,
                                    uint32_t config_offset, uint32_t num_lanes);

//===----------------------------------------------------------------------===//
// Worker-side APIs (every pool thread except the daemon, inside the kernel)
//===----------------------------------------------------------------------===//

// Submit one K-block job onto lane `worker_id`. Spin-waits when the lane is
// full so a worker can never over-run the daemon by more than
// HMX_MICRO_QUEUE_LANE_DEPTH jobs. Non-blocking once the slot is published --
// the mm runs asynchronously on the daemon thread. `job_kind` selects the f16
// or u8i8 accumulator-mode runtime functions (HMXMicroJobKind); it comes from
// the compiler, which knows the matmul's element type.
void hmx_micro_dispatch(HMXMicroQueue *q, int worker_id, uint32_t act_ah_offset,
                        uint32_t wgt_wh_offset, uint32_t out_offset,
                        bool clear_acc, bool write_out, uint32_t job_kind);

// Block until lane `worker_id` is fully drained by the daemon. The acquire load
// on `tail` pairs with the daemon's release store, so the acc_read output the
// daemon wrote is visible to this worker when the call returns (before it runs
// ah_to_rm / store). The spin is unbounded and does not yield, so a genuine
// stall presents as a hang rather than as a silently passing run.
void hmx_micro_wait_idle(HMXMicroQueue *q, int worker_id);

//===----------------------------------------------------------------------===//
// Daemon-side APIs (the one dedicated pool thread only, except rearm)
//===----------------------------------------------------------------------===//

// Acquire and lock the HMX unit to THIS thread (power up +
// HAP_compute_res_acquire + HAP_compute_res_hmx_lock). Runs ONCE before the
// first serve(). Does NOT program setup_acc_read: which variant to run depends
// on the job kind, which is only known once a job arrives, so serve() programs
// it lazily. The HMX lock is thread-bound and the daemon is a persistent pool
// thread, so the lock taken here stays held across every serve() until teardown
// releases it on this same thread. Stashes the compute-resource context id in a
// TU-static (one daemon per core), which teardown consumes.
void hmx_micro_daemon_setup(HMXMicroQueue *q);

// Drain one iteration's worth of jobs: round-robin over all lanes issuing
// acc_clear / mm / acc_read (f16 or u8i8 per the job's own kind) for each
// published job, returning once the exit signal is set AND every lane is empty.
void hmx_micro_daemon_serve(HMXMicroQueue *q);

// Unlock and release the HMX unit on THIS thread. Runs ONCE after the last
// serve(); pairs with hmx_micro_daemon_setup and must run on the same daemon
// thread that acquired the lock.
void hmx_micro_daemon_teardown(HMXMicroQueue *q);

// Clear the exit signal so the next serve() does not terminate immediately.
// Called by the wrapper BEFORE each iteration's barrier (may run off the daemon
// thread); the barrier's release/acquire publishes the cleared signal to the
// daemon. Replaces a full per-iteration queue re-init: after a drained
// iteration every lane already has head == tail, so only exit_signal needs
// resetting.
void hmx_micro_daemon_rearm(HMXMicroQueue *q);

// Set the exit signal. Called by the last worker that finishes its tile loop.
// The daemon may still drain in-flight jobs after this returns -- the
// all-lanes-empty check inside hmx_micro_daemon_serve is what terminates the
// current iteration's loop. Does NOT release HMX; release happens in teardown.
void hmx_micro_daemon_exit(HMXMicroQueue *q);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // HEXAGONBACKEND_BIN_RUNTIME_HMXMICRODAEMON_H
