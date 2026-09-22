//===- HexagonAPI.cpp -                                           ---------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// DataSpace: static allocations for Hexagon
//
//===----------------------------------------------------------------------===//

#include <atomic>
#include <cstdlib>
#include <cstring>
#include <mutex>

#ifdef HEXMLIR_RUNTIME_BRINGUP_PROBE
#include <cstdio>
#include "HAP_perf.h"
#endif

#include "HexagonAPI.h"
#include "HexagonBufferAlias.h"

// HAP_set_dcvs_v3_protected_bus_corners lives here; it is not pulled in by
// HAP_power.h.
#include "HAP_dcvs.h"

namespace {
// Process-wide bring-up lock. One qurt thread per grid program enters
// HexagonAPI::Global() concurrently (tm.exec in the launcher wrapper), so the
// power-up + HAP_compute_res_acquire + hmx_lock sequence below must run exactly
// once: a second HMX acquire squeezes the VTCM pool below its 1 MB CHECK and
// aborts the DSP (see docs/hmx/hmx-system-design.md 10.12 for the same
// failure through a different entry point). The mutex is only taken on the
// first call; afterwards the acquire-load below is the whole fast path, which
// also keeps the reentrant Global() calls from HexagonBuffer ctor/dtor off the
// lock (they run under the BufferManager mutex, so locking here would
// self-deadlock).
std::mutex gBringupMutex;
// Published only after the constructor fully returns, so a thread that observes
// a non-null pointer never sees a half-built instance.
std::atomic<HexagonAPI *> gInstance{nullptr};

// Per-thread HMX-lock state. The HMX unit lock is owned by one thread at a
// time (HAP_compute_res.h: only the thread holding a valid lock may execute
// HMX), so every thread that runs HMX leaves must hold the lock itself, on the
// single shared context acquired once above. thread_local is used instead of a
// mutex-guarded set of qurt_thread_get_id() values: it needs no qurt headers,
// no extra lock on the fast path, and it is already proven in this exact
// device binary (VTCMPool.cpp's fmt helpers use `static thread_local`, built
// by the same hexagon-clang bitcode rule). Pairing invariant: ensure locks
// only when this flag is false, unlock only unlocks when it is true, and each
// HMX segment (acc_clear .. acc_store) pairs one of each, so the flag always
// reflects whether this thread currently holds the lock. The constructor does
// not leave the flag set: see initialize_and_acquire_hmx.
thread_local bool tHoldsHmxLock = false;
} // namespace

#ifdef HEXMLIR_RUNTIME_BRINGUP_PROBE
namespace {
// One-shot bring-up recorder. The lazy singleton runs AcquireResources() exactly
// once per process, so these counters are written once and read by the probe
// wrapper through hexagon_runtime_bringup_report(). Steps are timed against the
// 19.2 MHz qtimer (HAP_perf_get_time_us).
struct BringupStep {
  const char *name;
  unsigned long long us;
};
constexpr unsigned kMaxBringupSteps = 16;
BringupStep gBringupSteps[kMaxBringupSteps];
unsigned gBringupStepCount = 0;
unsigned long long gBringupStart = 0;
unsigned long long gBringupPrev = 0;
unsigned long long gBringupTotal = 0;

unsigned long long bringupNowUs() {
  return (unsigned long long)HAP_perf_get_time_us();
}
} // namespace

void BringupProbeBegin() {
  gBringupStart = bringupNowUs();
  gBringupPrev = gBringupStart;
  gBringupStepCount = 0;
}

void BringupProbeStep(const char *name) {
  unsigned long long now = bringupNowUs();
  unsigned long long us = now - gBringupPrev;
  gBringupPrev = now;
  if (gBringupStepCount < kMaxBringupSteps)
    gBringupSteps[gBringupStepCount++] = BringupStep{name, us};
  FARF(ALWAYS, "BRINGUP step=%s us=%llu", name, us);
}

void BringupProbeEnd() {
  gBringupTotal = bringupNowUs() - gBringupStart;
  FARF(ALWAYS, "BRINGUP dsp_total=%llu", gBringupTotal);
}

// Format the recording into `buf` as "BRINGUP ..." lines; returns the byte
// count written (0 when the bring-up never ran in this process).
extern "C" int hexagon_runtime_bringup_report(char *buf, int cap) {
  if (buf == nullptr || cap <= 0)
    return 0;
  int off = 0;
  for (unsigned i = 0; i < gBringupStepCount; ++i) {
    int n = snprintf(buf + off, (size_t)(cap - off), "BRINGUP step=%s us=%llu\n",
                     gBringupSteps[i].name, gBringupSteps[i].us);
    if (n <= 0 || n >= cap - off)
      break;
    off += n;
  }
  int n = snprintf(buf + off, (size_t)(cap - off), "BRINGUP dsp_total=%llu\n",
                   gBringupTotal);
  if (n > 0 && n < cap - off)
    off += n;
  return off;
}
#endif // HEXMLIR_RUNTIME_BRINGUP_PROBE

HexagonAPI *HexagonAPI::Global() {
  HexagonAPI *inst = gInstance.load(std::memory_order_acquire);
  if (inst == nullptr) {
    std::lock_guard<std::mutex> lock(gBringupMutex);
    inst = gInstance.load(std::memory_order_relaxed);
    if (inst == nullptr) {
      inst = new HexagonAPI();
      gInstance.store(inst, std::memory_order_release);
    }
  }
  return inst;
}
// DataSpace: static allocations for Hexagon
void *HexagonAPI::Alloc(size_t nbytes, uint64_t alignment, bool isVtcm) {
  // Allocate a single, contiguous memory region.
  void *base_ptr = bufferManager->AllocateHexagonBuffer(
      nbytes, static_cast<size_t>(alignment), isVtcm);
  return base_ptr;
}

// Vote the DSP to its highest power/clock state, once per process, before any
// kernel runs. This mirrors the handwritten ggml-hexagon HTP bring-up
// (llama.cpp/ggml/src/ggml-hexagon/htp/main.c: "Set client class" / "DCVS
// setup" / "Power on HMX and set HMX clock"). Without these votes MMPM keeps
// the DSP at its default corners, which costs DDR/bus bandwidth and lets the
// selected clock drift between shapes. min == max == MAX pins the corners, so
// DCVS can no longer clock down and the DSP cannot enter sleep. That is the
// intended throughput-over-power trade for a short-lived inference session.
//
// Every step here is best-effort: a failure only logs, because the HMX power
// request in the caller is the part the runtime cannot proceed without.
//
static void votePowerAndClock(void *power_context) {
  // Compute client class: an unknown class does not get the DCVS votes applied.
  {
    HAP_power_request_t request;
    memset(&request, 0, sizeof(request));
    request.type = HAP_power_set_apptype;
    request.apptype = HAP_POWER_COMPUTE_CLIENT_CLASS;
    int retVal = HAP_power_set(power_context, &request);
    if (retVal != AEE_SUCCESS)
      FARF(ERROR, "HAP_power_set_apptype(COMPUTE) failed with value %d", retVal);
  }
  BringupProbeStep("hmx_power_apptype");

  // Pin core and bus to the maximum corner and disable sleep.
  {
    HAP_power_request_t request;
    memset(&request, 0, sizeof(request));
    request.type = HAP_power_set_DCVS_v3;
    request.dcvs_v3.set_dcvs_enable = TRUE;
    request.dcvs_v3.dcvs_enable = FALSE;
    request.dcvs_v3.set_core_params = TRUE;
    request.dcvs_v3.core_params.min_corner = HAP_DCVS_VCORNER_MAX;
    request.dcvs_v3.core_params.max_corner = HAP_DCVS_VCORNER_MAX;
    request.dcvs_v3.core_params.target_corner = HAP_DCVS_VCORNER_MAX;
    request.dcvs_v3.set_bus_params = TRUE;
    request.dcvs_v3.bus_params.min_corner = HAP_DCVS_VCORNER_MAX;
    request.dcvs_v3.bus_params.max_corner = HAP_DCVS_VCORNER_MAX;
    request.dcvs_v3.bus_params.target_corner = HAP_DCVS_VCORNER_MAX;
    request.dcvs_v3.set_sleep_disable = TRUE;
    request.dcvs_v3.sleep_disable = TRUE;

    // On v79+ bus corner requests above TURBO_PLUS are capped unless this
    // override is applied. The handwritten path guards the call with
    // `#if __HEXAGON_ARCH__ >= 79`, but this runtime's bitcode is compiled at
    // -mv75 (QHL workaround in bin/runtime/CMakeLists.txt), so the macro reads
    // 75 on the v79 device and that guard would compile the call out. The SDK
    // documents AEE_EVERSIONNOTSUPPORT as safe to ignore where the override
    // does not exist, so call it unconditionally and accept that code.
    int protRet = HAP_set_dcvs_v3_protected_bus_corners(&request, 1);
    if (protRet != AEE_SUCCESS && protRet != AEE_EVERSIONNOTSUPPORT)
      FARF(ERROR, "HAP_set_dcvs_v3_protected_bus_corners failed with value %d",
           protRet);

    int retVal = HAP_power_set(power_context, &request);
    if (retVal != AEE_SUCCESS)
      FARF(ERROR, "HAP_power_set_DCVS_v3(max corners) failed with value %d",
           retVal);
  }
  BringupProbeStep("hmx_power_dcvs");

  // HVX must be powered on before any HVX kernel runs.
  {
    HAP_power_request_t request;
    memset(&request, 0, sizeof(request));
    request.type = HAP_power_set_HVX;
    request.hvx.power_up = TRUE;
    int retVal = HAP_power_set(power_context, &request);
    if (retVal != AEE_SUCCESS)
      FARF(ERROR, "HAP_power_set_HVX(power_up) failed with value %d", retVal);
  }
  BringupProbeStep("hmx_power_hvx");
}

uint32_t HexagonAPI::initialize_and_acquire_hmx() {
  int *power_context = (int *)malloc(sizeof(int));

  votePowerAndClock((void *)power_context);

  // Power on HMX. From v75 on, HMX_v2 also votes the HMX clock to the maximum
  // corner (set_clock), so the engine does not sit at its default low clock;
  // below v75 only the plain power-up request exists.
  HAP_power_request_t request;
  memset(&request, 0, sizeof(HAP_power_request_t));
#if __HVX_ARCH__ >= 75
  request.type = HAP_power_set_HMX_v2;
  request.hmx_v2.set_power = TRUE;
  request.hmx_v2.power_up = TRUE;
  request.hmx_v2.set_clock = TRUE;
  request.hmx_v2.target_corner = HAP_DCVS_EXP_VCORNER_MAX;
  request.hmx_v2.min_corner = HAP_DCVS_EXP_VCORNER_MAX;
  request.hmx_v2.max_corner = HAP_DCVS_EXP_VCORNER_MAX;
  request.hmx_v2.perf_mode = HAP_CLK_PERF_HIGH;
#else
  request.type = HAP_power_set_HMX;
  request.hmx.power_up = TRUE;
#endif

  int retVal = HAP_power_set((void *)power_context, &request);
  if (retVal != AEE_SUCCESS) {
    FARF(ERROR,
         "HMX power request (type %d) failed with the return value %d",
         (int)request.type, retVal);
    return retVal;
  }
  BringupProbeStep("hmx_power_hmx");

  // Request HMX using HAP compute resource manager
  compute_res_attr_t compute_res;
  uint32_t contextID = 0;

  HAP_compute_res_attr_init(&compute_res);

  // Request the HMX resource
  HAP_compute_res_attr_set_hmx_param(&compute_res, 1);
  contextID = HAP_compute_res_acquire(&compute_res, 100000); // wait till 100ms

  if (contextID == 0) {
    return AEE_ERESOURCENOTFOUND;
  }
  BringupProbeStep("hmx_acquire");

  // Lock once so the bring-up sequence stays exactly (power-up, acquire, lock),
  // then release it immediately. The constructing thread is NOT the thread that
  // executes HMX once the tile loop is an `scf.forall`: it only dispatches the
  // async segments. A lock left held here would never be released by that thread
  // and the first segment's EnsureHmxLockForThisThread on a worker thread would
  // block forever. Bring-up still owns the acquire; every HMX segment now locks
  // for the thread that actually runs it.
  HAP_compute_res_hmx_lock(contextID);
  HAP_compute_res_hmx_unlock(contextID);
  tHoldsHmxLock = false;
  BringupProbeStep("hmx_lock");

  return contextID;
}

void HexagonAPI::EnsureHmxLockForThisThread() {
  if (tHoldsHmxLock)
    return;
  if (HAP_compute_res_hmx_lock(hmx_context_id) == 0) {
    tHoldsHmxLock = true;
  } else {
    FARF(ERROR, "HMX lock failed for this thread (context %u)", hmx_context_id);
  }
}

void HexagonAPI::ReleaseHmxLockForThisThread() {
  if (!tHoldsHmxLock)
    return;
  int ret = HAP_compute_res_hmx_unlock(hmx_context_id);
  // Clear unconditionally: after the unlock call this thread no longer holds
  // a usable lock (on failure the unit is still not ours to use), so the next
  // kernel must re-lock instead of assuming it holds the unit.
  tHoldsHmxLock = false;
  if (ret != 0) {
    FARF(ERROR, "HMX unlock failed for this thread (context %u, ret %d)",
         hmx_context_id, ret);
  }
}

void *HexagonAPI::Alloc(size_t nallocs, size_t nbytes, uint64_t alignment,
                        bool isVtcm) {
  // Allocate the region(s) needed for Hexagon's indirect-tensor format.
  void *base_ptr = bufferManager->AllocateHexagonBuffer(
      nallocs, nbytes, static_cast<size_t>(alignment), isVtcm);
  return base_ptr;
}

void *HexagonAPI::WeightResident(uint64_t key, size_t nbytes,
                                 const void *src) {
  return runtimeVtcm->Resident(key, nbytes, src);
}

void *HexagonAPI::WorkspaceResident(uint64_t key, size_t nbytes) {
  // No source: nothing is copied in, the pinning of the storage is the whole
  // point. The kernel refills the buffer on every launch.
  return runtimeVtcm->Resident(key, nbytes, nullptr);
}

/// Takes a `ptr` to the base of the memref and returns a pointer to the
/// crouton table
void *HexagonAPI::CreateBufferAlias(void *ptr, size_t nbytes) {
  return bufferManager->CreateBufferAlias(ptr, nbytes);
}

/// Takes the pointer to crouton table that was created as an alias and returns
/// the base pointer to the memref
void *HexagonAPI::GetOrigBufferFromAlias(void *aliasPtr) {
  return bufferManager->GetOrigBufferFromAlias(aliasPtr);
}

void HexagonAPI::Free(void *ptr) {
  if (bufferManager) {
    bufferManager->FreeHexagonBuffer(ptr);
  } else {
    // Either AcquireResources was never called, or ReleaseResources was called.
    // Since this can occur in the normal course of shutdown, log a message and
    // continue.
    CHECK((false), "Free called outside a session");
  }
}

// TODO: Add support to handle buffer aliases on copying
void HexagonAPI::Copy(void *dst, void *src, size_t nbytes) {
  if (bufferManager) {
    bufferManager->Copy(dst, src, nbytes);
  } else {
    // Either AcquireResources was never called, or ReleaseResources was called.
    // Since this can occur in the normal course of shutdown, log a message and
    // continue.
    CHECK((false), "Free called outside a session");
  }
}
