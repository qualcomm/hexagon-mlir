//===- test_report.h ------------------------------------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

#ifndef TEST_REPORT_H
#define TEST_REPORT_H
#include <string>
#include <vector>

using namespace std;
enum Result : int {
  Pass = 0,
  Fail = 1,
  Unknown = 2,
};

string get_result_string(Result r) {
  switch (r) {
  case Result::Pass:
    return "Pass";
  case Result::Fail:
    return "Fail";
  case Result::Unknown:
    return "Unknown";
  default:
    return "Bad Result Value";
  }
}

class TestReport {
private:
  string name;
  double perf;
  string units;
  Result result;
  string save_path;

  void stringify_report(FILE *stream) {
    fprintf(stream, "Test_Info: {\n");
    fprintf(stream, "\tName:%s\n", name.c_str());
    fprintf(stream, "\tResult:%s\n", get_result_string(result).c_str());
    fprintf(stream, "\tPerf:%f\n", perf);
    fprintf(stream, "\tUnits:%s\n", units.c_str());
    fprintf(stream, "}\n");
  }

public:
  TestReport(string n, double p, string u, Result r = Result::Unknown,
             string path = "")
      : name(n), perf(p), units(u), result(r) {
    if (path.empty()) {
      save_path = "perf.txt";
    }
  }
  void print() { stringify_report(stdout); }
  int save() {
    FILE *fp = fopen(save_path.c_str(), "w");
    if (!fp) {
      fprintf(stderr, "Error opening file: %s\n", save_path.c_str());
      return 1;
    }
    stringify_report(fp);
    fclose(fp);
    return 0;
  }
};

void error(string m, uint64_t avg_time = 0) {
  TestReport tr(m, avg_time, "microseconds", Result::Fail);
  tr.print();
  printf("Failed\n");
  exit(1);
};

#endif
