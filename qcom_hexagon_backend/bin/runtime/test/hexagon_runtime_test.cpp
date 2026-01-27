//===- hexagon_runtime_test.cpp - implementation file          ------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// Function to split a string by spaces and return a vector of words
//
//===----------------------------------------------------------------------===//

#include <gtest/gtest.h>
#include <iostream>
#include <vector>

// Function to split a string by spaces and return a vector of words
std::vector<std::string> splitStringBySpace(const std::string &str) {
  std::vector<std::string> words;
  std::stringstream ss(str);
  std::string word;

  while (ss >> word) {
    words.push_back(word);
  }

  return words;
}

int main(int argc, char *argv[]) {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " --gtest-args=\"<args>\""
              << std::endl;
    return 1;
  }

  // Join all command line arguments into a single string
  std::string joinedArgs;
  for (int i = 1; i < argc; ++i) {
    if (i > 1) {
      joinedArgs += " ";
    }
    joinedArgs += argv[i];
  }

  std::string prefix = "--gtest-args=";
  size_t pos = joinedArgs.find(prefix);
  if (pos == std::string::npos) {
    std::cerr << "Error: Argument must contain --gtest-args=" << std::endl;
    return 1;
  }

  std::string str = joinedArgs.substr(pos + prefix.size());
  // Remove the surrounding quotes if present
  if (str.front() == '"' && str.back() == '"') {
    str = str.substr(1, str.size() - 2);
  }

  std::vector<std::string> words = splitStringBySpace(str);

  // Output the words to verify
  for (const auto &w : words) {
    std::cout << w << std::endl;
  }
  std::vector<const char *> gtestArgs;
  gtestArgs.push_back(argv[0]); // Program name
  for (const auto &word : words) {
    gtestArgs.push_back(word.c_str());
  }

  // Initialize gtest
  int gtestArgc = gtestArgs.size();
  ::testing::InitGoogleTest(&gtestArgc, const_cast<char **>(gtestArgs.data()));

  // Redirect gtest output from stdout to file
  // TODO: We could instead use a json output if needed for CI later by
  // passing the gtest-arg '--gtest_output="json:path_to_output_file"'
  freopen("gtest_output.txt", "w", stdout);

  // Run all tests
  int gtest_rcode = RUN_ALL_TESTS();

  fclose(stdout);

  return gtest_rcode;
}
