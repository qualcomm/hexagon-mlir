//===- OptionsParsing.h - Parse options from MLIR Passes ------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

#include "hexagon/Transforms/OptionsParsing.h"
#include "hexagon/Common/Common.h"
#include "llvm/Support/Debug.h"
#include <charconv>
#include <sstream>
#include <system_error>

#define DEBUG_TYPE "options-parsing"
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define DBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir {
namespace hexagon {

std::optional<SmallVector<int64_t>> parseTileSizes(std::string tileSizesStr) {

  if (tileSizesStr.empty()) {
    DBG("-> No tile size string provided.");
    return std::nullopt;
  }

  SmallVector<int64_t> tile_sizes;
  std::stringstream tss(tileSizesStr);
  std::string tile_size;
  while (std::getline(tss, tile_size, ',')) {
    tile_sizes.push_back(std::stoi(tile_size));
  }

  assert(!tile_sizes.empty() &&
         "invalid tile size format. Form is : Int,Int,.. \n");
  DBG("-> User provided Tile sizes: {" << toString(tile_sizes) << "}");
  return tile_sizes;
}

int safe_stoi(const std::string &str) {
  int result = 0;
  // std::from_chars(start, end, result)
  auto [ptr, ec] = std::from_chars(str.data(), str.data() + str.size(), result);

  if (ec == std::errc()) {
    return result; // Success
  } else if (ec == std::errc::invalid_argument) {
    // Handle invalid input (e.g., return default or error code)
    DBG("-> Invalid format: non-numeric value in tile sizes string. Expected "
        "format: <dim>:<size>,<dim>:<size>,...");
    return -1;
  } else if (ec == std::errc::result_out_of_range) {
    // Handle overflow
    DBG("-> Invalid format: numeric value out of range");
    return -1;
  }
  return 0;
}

std::optional<DenseMap<int64_t, int64_t>>
parseConvTileSizes(std::string tileSizesStr) {

  DenseMap<int64_t, int64_t> tileSizesForDims;
  if (tileSizesStr.empty()) {
    DBG("-> No tile size string provided.");
    return tileSizesForDims;
  }

  std::stringstream tss(tileSizesStr);
  std::string tile_dim, tile_size;
  while (std::getline(tss, tile_dim, ':') &&
         std::getline(tss, tile_size, ',')) {
    // Trim whitespace
    tile_dim.erase(0, tile_dim.find_first_not_of(" \t\n\r"));
    tile_dim.erase(tile_dim.find_last_not_of(" \t\n\r") + 1);
    tile_size.erase(0, tile_size.find_first_not_of(" \t\n\r"));
    tile_size.erase(tile_size.find_last_not_of(" \t\n\r") + 1);

    // Parse integers
    int64_t dim = safe_stoi(tile_dim);
    int64_t size = safe_stoi(tile_size);

    // Validate positive values
    if (dim < 0 || size <= 0) {
      DBG("-> Invalid format: dimension must be non-negative and size must "
          "be positive");
      return std::nullopt;
    }

    tileSizesForDims.insert(std::make_pair(dim, size));
  }

  DBG("-> Successfully parsed tile sizes: ");
  for (const auto &pair : tileSizesForDims) {
    DBG("Dimension " << pair.first << " -> " << pair.second);
  }

  return tileSizesForDims;
}
} // namespace hexagon
} // namespace mlir
