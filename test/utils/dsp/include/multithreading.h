//===- multithreading.h ---------------------------------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

#ifndef MULTITHREADING_H
#define MULTITHREADING_H

#include <stdint.h>
#include <stdlib.h>
#include <vector>

extern void *memalign(size_t, size_t);

struct spawned_thread {
  void (*f)(void *);
  void *closure;
  void *stack;
  qurt_thread_t tid;
};

// T -> Closure class
template <typename T> class ThreadManager {
public:
  ThreadManager(int num_threads) : thread_pool(num_threads) {}
  int thread_cnt() const { return thread_pool.size(); }
  // Execute the function f for all closures in the collection.
  void exec(void (*f)(void *), T *closures);
  // Execute the function f serially (does not use qurt threads)
  void exec_serial(void (*f)(void *), T *closures);

  // Create qurt threads to execute function f with arguments provided as
  // the closure. Also (no of closures) == (size of thread_pool)
  void spawn_threads(void (*f)(void *), T *closures);
  // Wait for all the specified thread to finish.
  void join_threads();

private:
  spawned_thread *spawn_thread(void (*f)(void *), void *closure);
  // Wait for the specified thread to finish.
  void join_thread(spawned_thread *thread_arg);

  std::vector<spawned_thread *> thread_pool;
  const unsigned int stack_size = 1024 * 1024;
  const unsigned short priority = 100;
};

static void spawn_thread_helper(void *arg) {
  spawned_thread *t = (spawned_thread *)arg;
  t->f(t->closure);
}

// TODO: Keep the thread pool alive.
template <typename T>
void ThreadManager<T>::exec(void (*f)(void *), T *closures) {
  if (thread_cnt() == 1) {
    return f(&closures[0]);
  }
  spawn_threads(f, closures);
  join_threads();
}

template <typename T>
void ThreadManager<T>::exec_serial(void (*f)(void *), T *closures) {
  for (int i = 0; i < thread_cnt(); i++) {
    f(&closures[i]);
  }
}

template <typename T>
spawned_thread *ThreadManager<T>::spawn_thread(void (*f)(void *),
                                               void *closure) {
  spawned_thread *t = (spawned_thread *)malloc(sizeof(spawned_thread));
  t->f = f;
  t->closure = closure;
  t->stack = memalign(128, ThreadManager<T>::stack_size);
  qurt_thread_attr_t thread_attr;
  qurt_thread_attr_init(&thread_attr);
  qurt_thread_attr_set_stack_addr(&thread_attr, t->stack);
  qurt_thread_attr_set_stack_size(&thread_attr, ThreadManager<T>::stack_size);
  qurt_thread_attr_set_priority(&thread_attr, ThreadManager<T>::priority);
  qurt_thread_create(&t->tid, &thread_attr, spawn_thread_helper, t);
  return (spawned_thread *)t;
}

template <typename T>
void ThreadManager<T>::spawn_threads(void (*f)(void *), T *closures) {
  for (int i = 0; i < thread_pool.size(); i++) {
    thread_pool[i] = spawn_thread(f, &closures[i]);
  }
}

template <typename T>
void ThreadManager<T>::join_thread(struct spawned_thread *thread_arg) {
  spawned_thread *t = (spawned_thread *)thread_arg;
  int ret = 0;
  qurt_thread_join(t->tid, &ret);
  free(t->stack);
  free(t);
}

template <typename T> void ThreadManager<T>::join_threads() {
  for (int i = 0; i < thread_pool.size(); i++) {
    join_thread(thread_pool[i]);
  }
}

#endif // MULTITHREADING_H