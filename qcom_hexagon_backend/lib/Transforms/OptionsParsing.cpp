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
#include <sstream>

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
} // namespace hexagon
} // namespace mlir
