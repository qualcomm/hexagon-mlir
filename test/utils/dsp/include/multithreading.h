//===- multithreading.h ---------------------------------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// Persistent-thread ThreadManager for Hexagon DSP.
//
// Threads are created ONCE in the constructor and reused across exec() calls
// via QuRT semaphore-based work dispatch. This eliminates qurt_thread_create /
// stack-allocation overhead from the benchmark hot path, allowing accurate
// measurement of compute-only time.
//
//===----------------------------------------------------------------------===//

#ifndef MULTITHREADING_H
#define MULTITHREADING_H

#include <atomic>
#include <qurt.h>
#include <stdint.h>
#include <stdlib.h>
#include <vector>

extern void *memalign(size_t, size_t);

// T -> Closure class
template <typename T> class ThreadManager {
public:
  explicit ThreadManager(int num_threads);
  ~ThreadManager();

  int thread_cnt() const { return static_cast<int>(workers_.size()); }

  // Execute f for all closures using persistent threads.
  void exec(void (*f)(void *), T *closures);

  // Execute f serially (no threads).
  void exec_serial(void (*f)(void *), T *closures);

private:
  // Per-worker state.
  struct Worker {
    void (*func)(void *);   // work function (set before signaling start)
    void *closure;          // argument (set before signaling start)
    std::atomic<bool> stop; // set to true to terminate worker loop
    qurt_sem_t start;       // main → worker: "work is ready"
    qurt_sem_t done;        // worker → main: "work is complete"
    void *stack;            // stack memory owned by this worker
    qurt_thread_t tid;      // QuRT thread handle
  };

  static void worker_entry(void *arg);

  std::vector<Worker *> workers_;
  const unsigned int stack_size_ = 1024 * 1024;
  const unsigned short priority_ = 100;
};

template <typename T> void ThreadManager<T>::worker_entry(void *arg) {
  Worker *w = static_cast<Worker *>(arg);
  while (true) {
    qurt_sem_down(&w->start); // wait for main to post work
    if (w->stop)
      break;
    w->func(w->closure);   // execute assigned work item
    qurt_sem_up(&w->done); // notify main that work is done
  }
}

template <typename T> ThreadManager<T>::ThreadManager(int num_threads) {
  workers_.reserve(num_threads);
  for (int i = 0; i < num_threads; ++i) {
    Worker *w = new Worker();
    w->func = nullptr;
    w->closure = nullptr;
    w->stop = false;
    qurt_sem_init_val(&w->start, 0);
    qurt_sem_init_val(&w->done, 0);
    w->stack = memalign(128, stack_size_);

    qurt_thread_attr_t attr;
    qurt_thread_attr_init(&attr);
    qurt_thread_attr_set_stack_addr(&attr, w->stack);
    qurt_thread_attr_set_stack_size(&attr, stack_size_);
    qurt_thread_attr_set_priority(&attr, priority_);
    qurt_thread_create(&w->tid, &attr, worker_entry, w);

    workers_.push_back(w);
  }
}

template <typename T> ThreadManager<T>::~ThreadManager() {
  // Signal all workers to exit.
  for (Worker *w : workers_) {
    w->stop = true;
    qurt_sem_up(&w->start);
  }
  // Join and free resources.
  for (Worker *w : workers_) {
    int ret = 0;
    qurt_thread_join(w->tid, &ret);
    free(w->stack);
    qurt_sem_destroy(&w->start);
    qurt_sem_destroy(&w->done);
    delete w;
  }
}

template <typename T>
void ThreadManager<T>::exec(void (*f)(void *), T *closures) {
  if (thread_cnt() == 1) {
    f(&closures[0]);
    return;
  }
  // Post work to all persistent workers.
  for (int i = 0; i < thread_cnt(); ++i) {
    workers_[i]->func = f;
    workers_[i]->closure = &closures[i];
    qurt_sem_up(&workers_[i]->start); // wake worker i
  }
  // Wait for all workers to finish.
  for (int i = 0; i < thread_cnt(); ++i) {
    qurt_sem_down(&workers_[i]->done);
  }
}

template <typename T>
void ThreadManager<T>::exec_serial(void (*f)(void *), T *closures) {
  for (int i = 0; i < thread_cnt(); ++i) {
    f(&closures[i]);
  }
}

#endif // MULTITHREADING_H
