//===- HexagonThreadPool.cpp - Hexagon thread pool             ------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

#include "HexagonThreadPool.h"

HexagonThreadPool::HexagonThreadPool(size_t numThreads)
    : stop(false), activeTasks(0) {
  for (size_t i = 0; i < numThreads; ++i) {
    workers.emplace_back(&HexagonThreadPool::workerThread, this);
  }
}

HexagonThreadPool::~HexagonThreadPool() {
  {
    std::unique_lock<std::mutex> lock(queueMutex);
    stop = true;
  }
  condition.notify_all();
  for (std::thread &worker : workers) {
    worker.join();
  }
}

void HexagonThreadPool::enqueueTask(std::function<void()> task) {
  {
    std::unique_lock<std::mutex> lock(queueMutex);
    tasks.push([this, task] {
      task();
      {
        std::lock_guard<std::mutex> lock(allTasksDoneMutex);
        if (--activeTasks == 0) {
          allTasksDone.notify_all();
        }
      }
    });
    ++activeTasks;
  }
  condition.notify_one();
}

void HexagonThreadPool::wait() {
  std::unique_lock<std::mutex> lock(allTasksDoneMutex);
  allTasksDone.wait(lock, [this] { return activeTasks.load() == 0; });
}

size_t HexagonThreadPool::getMaxConcurrency() const { return workers.size(); }

void HexagonThreadPool::workerThread() {
  while (true) {
    std::function<void()> task;
    {
      std::unique_lock<std::mutex> lock(queueMutex);
      condition.wait(lock, [this] { return stop || !tasks.empty(); });
      if (stop && tasks.empty()) {
        return;
      }
      task = std::move(tasks.front());
      tasks.pop();
    }
    task();
  }
}
