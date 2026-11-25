//===- HelloWorld.h - Hello World example class ---------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

#ifndef QCOM_HEXAGON_BACKEND_HELLO_WORLD_H
#define QCOM_HEXAGON_BACKEND_HELLO_WORLD_H

#include <iostream>
#include <string>

class HelloWorld {
public:
  HelloWorld() : message_("Hello, World!") {}
  explicit HelloWorld(std::string message) : message_(std::move(message)) {}

  void greet() const { std::cout << message_ << "\n"; }

  const std::string &getMessage() const { return message_; }

private:
  std::string message_;
};

#endif // QCOM_HEXAGON_BACKEND_HELLO_WORLD_H
