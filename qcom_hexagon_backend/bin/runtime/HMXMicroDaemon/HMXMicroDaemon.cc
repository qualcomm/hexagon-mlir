//===- HMXMicroDaemon.cc - HexKL micro-op HMX queue and daemon ------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// Worker / daemon protocol for the queue defined in HMXMicroDaemon.h.
//
// The daemon runs in three phases on its dedicated thread: setup() acquires +
// locks HMX and runs setup_acc_read ONCE; serve() drains jobs (below) and is
// re-entered per iteration while the lock stays held; teardown() unlocks +
// releases HMX ONCE. rearm() clears exit_signal between iterations.
//
//   workers                              daemon serve() (lock already held)
//   ---------------------                ------------------------------------
//   write job into slots[head]           round-robin lanes[0..num_lanes-1]
//   head++           (RELEASE)           if head != tail (ACQUIRE on head):
//                                            [acc_clear] mm_{f16,u8i8}
//                                            [acc_read]
//                                            tail++           (RELEASE)
//                                            stay on the same lane until a
//                                            job with write_out=1 is consumed
//                                         return when exit_signal && all empty
//
// head is the sole publication signal (worker RELEASE-stores head after writing
// the job; daemon ACQUIRE-loads it before reading the payload), and tail is the
// sole slot-release signal (daemon RELEASE-stores tail after reading; worker
// ACQUIRE-loads it in the backpressure loop before reusing a slot).
//
// `wait_idle` for lane `worker_id` is a tail == head spin (ACQUIRE), so the
// worker only resumes once the daemon has written the acc_read output. Both
// worker-side spins are unbounded and do not yield, so a genuine stall presents
// as a hang rather than as a silently passing run.
//
// The queue protocol (backpressure, wait_idle, partialInFlight lane-sticking)
// otherwise follows HMXDaemon.cc; the per-job body, the HMX acquire/release
// (hoisted to setup/teardown), and the dropped per-slot valid differ.
//
//===----------------------------------------------------------------------===//

#include "HMXMicroDaemon.h"

#include <atomic>
#include <string.h>

using std::atomic_load_explicit;
using std::atomic_store_explicit;
using std::memory_order_acquire;
using std::memory_order_relaxed;
using std::memory_order_release;

#include "HAP_compute_res.h"
#include "HMXPower.h"

#ifdef __GNUC__
#define restrict __restrict__
#endif
#include "hexkl_micro.h"
#ifdef __GNUC__
#undef restrict
#endif

// The launcher 2048-aligns config_offset rather than using the documented
// HEXKL_HMX_CONFIG_ALIGNMENT, because acc_read_int32 writes a8
// activation-layout tiles inside the config region (at config_offset + 4/6/8/10
// KB) and those need HEXKL_HMX_ACTIVATION_ALIGNMENT. Assert the stricter value
// still satisfies the documented one, so a future bump of
// HEXKL_HMX_CONFIG_ALIGNMENT past 2048 is caught here instead of silently
// violating the config precondition.
static_assert(2048U % HEXKL_HMX_CONFIG_ALIGNMENT == 0,
              "backend/triton_hexagon_launcher.py 2048-aligns config_offset; "
              "that must remain a multiple of HEXKL_HMX_CONFIG_ALIGNMENT");
static_assert(2048U % HEXKL_HMX_ACTIVATION_ALIGNMENT == 0,
              "config_offset alignment must also satisfy the activation "
              "alignment acc_read_int32's interior a8 writes require");

// Enforce the declared cache-line layout; the launcher aligns each queue.
static_assert(offsetof(HMXMicroQueue, exit_signal) % HMX_MICRO_CACHE_LINE == 0,
              "exit_signal must start on its own cache line");
static_assert(offsetof(HMXMicroQueue, lanes) % HMX_MICRO_CACHE_LINE == 0,
              "the lane array must start on a cache line boundary");
static_assert(
    sizeof(HMXMicroLane) % HMX_MICRO_CACHE_LINE == 0,
    "lane stride must be a whole number of cache lines, else lane N's "
    "fields drift off-line as N grows");
static_assert(offsetof(HMXMicroLane, tail) % HMX_MICRO_CACHE_LINE == 0,
              "head and tail must not share a cache line");
static_assert(offsetof(HMXMicroLane, slots) % HMX_MICRO_CACHE_LINE == 0,
              "the slot array must start on a cache line boundary");
static_assert(sizeof(HMXMicroJobSlot) == HMX_MICRO_CACHE_LINE,
              "a job slot must occupy exactly one cache line so no two "
              "adjacent slots can share one");

namespace {

static_assert((HMX_MICRO_QUEUE_LANE_DEPTH & (HMX_MICRO_QUEUE_LANE_DEPTH - 1)) ==
                  0,
              "HMX_MICRO_QUEUE_LANE_DEPTH must be a power of two");
constexpr uint32_t kSlotMask = HMX_MICRO_QUEUE_LANE_DEPTH - 1;

// Outcome of a single round-robin probe of a lane. Drives the daemon's
// tile-boundary scheduling: the loop must stay on the same lane while it is
// midway through a tile (PartialKSlice) and may advance otherwise.
enum class HMXJobOutcome {
  Empty,         // No job available right now.
  PartialKSlice, // Consumed a job with write_out=false; tile still in flight.
  TileEnd,       // Consumed a job with write_out=true; tile completed.
};

// Daemon-thread-local accumulator mode. Program it lazily because the job
// determines the mode; preserve it across serve() calls.
constexpr int kProgrammedNone = -1;
int programmedAccMode = kProgrammedNone;

// Program setup_acc_read for `kind` if it is not already the active mode. Must
// run on the daemon thread with the HMX lock held.
inline void ensureAccModeProgrammed(HMXMicroQueue *q, uint8_t kind) {
  if (programmedAccMode == static_cast<int>(kind))
    return;

  auto *vtcmBase = static_cast<uint8_t *>(q->vtcm_base);
  if (kind == HMX_MICRO_JOB_KIND_U8I8) {
    hexkl_micro_hmx_setup_acc_read_int32(vtcmBase, q->config_offset);
  } else {
    hexkl_micro_hmx_setup_acc_read_f16(vtcmBase, q->config_offset);
  }
  programmedAccMode = static_cast<int>(kind);
}

// Pulls one job from `lane` and runs it on HMX via the HexKL accumulator micro-
// ops, then publishes completion. `q` supplies the global VTCM base and the
// config-block offset that acc_read needs, and the job's own `job_kind` selects
// which accumulator-mode runtime functions it runs through (f16 or u8i8 -- see
// HMXMicroJobKind).
inline HMXJobOutcome consumeOneJob(HMXMicroQueue *q, HMXMicroLane &lane) {
  uint32_t tail = atomic_load_explicit(&lane.tail, memory_order_acquire);
  // ACQUIRE on head is the publication point: it pairs with the worker's
  // RELEASE store on head, so once head != tail the job payload written before
  // it is visible below. No separate per-slot valid load is needed.
  uint32_t head = atomic_load_explicit(&lane.head, memory_order_acquire);
  if (head == tail) {
    return HMXJobOutcome::Empty;
  }

  HMXMicroJobSlot &slot = lane.slots[tail & kSlotMask];
  HMXMicroJob job = slot.job;

  // Program the job's accumulator mode before its K=32 matmul.
  ensureAccModeProgrammed(q, job.job_kind);

  auto *vtcmBase = static_cast<uint8_t *>(q->vtcm_base);
  if (job.job_kind == HMX_MICRO_JOB_KIND_U8I8) {
    if (job.clear_acc) {
      hexkl_micro_hmx_acc_clear_int32();
    }
    hexkl_micro_hmx_mm_u8i8(vtcmBase, job.act_ah_offset, job.wgt_wh_offset);
    if (job.write_out) {
      hexkl_micro_hmx_acc_read_int32(vtcmBase, q->config_offset,
                                     job.out_offset);
    }
  } else {
    if (job.clear_acc) {
      hexkl_micro_hmx_acc_clear_f16();
    }
    hexkl_micro_hmx_mm_f16(vtcmBase, job.act_ah_offset, job.wgt_wh_offset);
    if (job.write_out) {
      hexkl_micro_hmx_acc_read_f16(vtcmBase, q->config_offset, job.out_offset);
    }
  }

  // RELEASE on tail is the slot-release point: it pairs with the worker's
  // ACQUIRE load of tail in the backpressure loop, so the worker only reuses
  // this slot after the payload read above has completed (no WAR race).
  atomic_store_explicit(&lane.tail, tail + 1, memory_order_release);
  HMXJobOutcome outcome =
      job.write_out ? HMXJobOutcome::TileEnd : HMXJobOutcome::PartialKSlice;
  return outcome;
}

inline bool laneIsEmpty(const HMXMicroLane &lane) {
  uint32_t head = atomic_load_explicit(&lane.head, memory_order_acquire);
  uint32_t tail = atomic_load_explicit(&lane.tail, memory_order_acquire);
  return head == tail;
}

// Advance the round-robin lane index, wrapping at the active lane count.
// num_lanes tracks the actual worker count (not necessarily a power of two),
// so wrap with a compare-and-reset -- cheaper than a modulo and exact for any
// num_lanes >= 1.
inline int nextLane(int laneIdx, uint32_t numLanes) {
  int next = laneIdx + 1;
  return (next == static_cast<int>(numLanes)) ? 0 : next;
}

// Power on, acquire, and lock the HMX unit to the CALLING thread. Mirrors the
// HexagonRuntime sequence (HexagonAPI::initialize_and_acquire_hmx) but is
// inlined here so the daemon does not depend on the runtime. Returns the HAP
// compute-resource context id (0 on failure).
uint32_t acquireAndLockHmx() {
  // The power vote is keyed by a client pointer that HAP retains for the life
  // of the vote. A function-local static gives a stable address with no
  // per-call leak; only one daemon runs at a time (one HMX per core).
  static int powerContext;
  if (hexagonPowerUpHmx(&powerContext) != AEE_SUCCESS)
    return 0;

  compute_res_attr_t computeRes;
  HAP_compute_res_attr_init(&computeRes);
  HAP_compute_res_attr_set_hmx_param(&computeRes, 1);
  uint32_t contextID =
      HAP_compute_res_acquire(&computeRes, 100000); // wait up to 100 ms
  if (contextID == 0)
    return 0;

  // Locks the HMX unit to the current thread and prepares it to issue HMX
  // instructions. Every acc_clear / mm / acc_read must run on this thread.
  HAP_compute_res_hmx_lock(contextID);
  return contextID;
}

// Unlock and release the HMX unit. Must run on the same thread that acquired
// it.
void releaseHmx(uint32_t contextID) {
  if (contextID) {
    HAP_compute_res_hmx_unlock(contextID);
    HAP_compute_res_release(contextID);
  }
}

// HMX context ID shared by setup and teardown on the daemon thread.
uint32_t daemonHmxContextID = 0;

} // namespace

extern "C" {

size_t hmx_micro_queue_size_bytes(void) { return sizeof(HMXMicroQueue); }

HMXMicroQueue *hmx_micro_queue_init(void *mem, void *vtcm_base,
                                    uint32_t config_offset,
                                    uint32_t num_lanes) {
  HMXMicroQueue *q = static_cast<HMXMicroQueue *>(mem);
  // memset zero-initialises every atomic (counters, exit signal); the release
  // fence at the bottom publishes that view to all threads.
  memset(q, 0, sizeof(HMXMicroQueue));
  q->vtcm_base = vtcm_base;
  q->config_offset = config_offset;
  q->num_lanes = num_lanes;
  atomic_thread_fence(memory_order_release);
  return q;
}

void hmx_micro_dispatch(HMXMicroQueue *q, int worker_id, uint32_t act_ah_offset,
                        uint32_t wgt_wh_offset, uint32_t out_offset,
                        bool clear_acc, bool write_out, uint32_t job_kind) {
  HMXMicroLane &lane = q->lanes[worker_id];

  // Backpressure: spin until the lane has a free slot. The daemon advances tail
  // with RELEASE, so we acquire to observe those updates.
  uint32_t head = atomic_load_explicit(&lane.head, memory_order_acquire);
  while (true) {
    uint32_t tail = atomic_load_explicit(&lane.tail, memory_order_acquire);
    if (head - tail < HMX_MICRO_QUEUE_LANE_DEPTH)
      break;
  }

  HMXMicroJobSlot &slot = lane.slots[head & kSlotMask];
  // The backpressure ACQUIRE on tail above ensures the daemon has already read
  // (and released via tail) any previous occupant of this slot, so overwriting
  // the payload here races nothing.
  slot.job.clear_acc = clear_acc;
  slot.job.write_out = write_out;
  slot.job.job_kind = static_cast<uint8_t>(job_kind);
  slot.job.act_ah_offset = act_ah_offset;
  slot.job.wgt_wh_offset = wgt_wh_offset;
  slot.job.out_offset = out_offset;

  // Publication: the RELEASE store on head makes the payload written above
  // visible to the daemon's ACQUIRE load of head. head is the sole publication
  // signal (no per-slot valid flag).
  atomic_store_explicit(&lane.head, head + 1, memory_order_release);
}

void hmx_micro_wait_idle(HMXMicroQueue *q, int worker_id) {
  HMXMicroLane &lane = q->lanes[worker_id];
  // Spin until the daemon has caught up. The ACQUIRE on `tail` pairs with the
  // daemon's RELEASE store and guarantees the acc_read output is visible before
  // the worker returns from this call.
  while (true) {
    uint32_t head = atomic_load_explicit(&lane.head, memory_order_acquire);
    uint32_t tail = atomic_load_explicit(&lane.tail, memory_order_acquire);
    if (head == tail)
      return;
  }
}

void hmx_micro_daemon_setup(HMXMicroQueue *) {
  // Take HMX ownership. Job processing lazily programs the accumulator mode.
  // Do not lock an HVX context here. Limit the pool so workers plus the daemon
  // fit the target's HVX context count.
  daemonHmxContextID = acquireAndLockHmx();
  programmedAccMode = kProgrammedNone;
}

void hmx_micro_daemon_serve(HMXMicroQueue *q) {
  int laneIdx = 0;

  // Stay on a lane until its in-progress tile reaches write_out.
  bool partialInFlight = false;

  while (true) {
    // Exit only after every lane is drained.
    bool exitRequested =
        atomic_load_explicit(&q->exit_signal, memory_order_acquire) != 0;
    if (exitRequested) {
      bool allEmpty = true;
      for (uint32_t i = 0; i < q->num_lanes; ++i) {
        if (!laneIsEmpty(q->lanes[i])) {
          allEmpty = false;
          break;
        }
      }
      if (allEmpty) {
        break;
      }
    }

    HMXMicroLane &lane = q->lanes[laneIdx];
    HMXJobOutcome outcome = consumeOneJob(q, lane);

    switch (outcome) {
    case HMXJobOutcome::PartialKSlice:
      // Keep this lane while its accumulator holds a partial tile.
      partialInFlight = true;
      break;
    case HMXJobOutcome::TileEnd:
      // Tile complete; resume round-robin scheduling.
      partialInFlight = false;
      laneIdx = nextLane(laneIdx, q->num_lanes);
      break;
    case HMXJobOutcome::Empty:
      // Do not let another lane clear an in-progress accumulator.
      if (!partialInFlight) {
        laneIdx = nextLane(laneIdx, q->num_lanes);
      }
      break;
    }
  }
}

void hmx_micro_daemon_teardown(HMXMicroQueue *) {
  // Relinquish HMX on this same thread; pairs with the acquire in setup.
  releaseHmx(daemonHmxContextID);
}

void hmx_micro_daemon_rearm(HMXMicroQueue *q) {
  // Release-publish the next serve's cleared exit signal; drained lanes need no
  // reset.
  atomic_store_explicit(&q->exit_signal, 0u, memory_order_release);
}

void hmx_micro_daemon_exit(HMXMicroQueue *q) {
  atomic_store_explicit(&q->exit_signal, 1u, memory_order_release);
}

} // extern "C"
