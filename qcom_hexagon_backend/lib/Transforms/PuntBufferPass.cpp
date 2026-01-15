//===- PuntBufferPass.cpp - implementation of punt-buffer pass ---------======//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// This file implements removing of copying of kernel input arg buffer content
// to locally created buffers. Instead forwards it to user where valid.
//
//===----------------------------------------------------------------------===//
//
// e.g.
// ```
//   func.func @foo(%arg0: memref<*xf16>, ...) {
//      ...
//      %reinterpret_cast = memref.reinterpret_cast %arg0 to ...]
//                          : memref<*xElType> to memref<ShapeTyxElType>
//      %alloc = memref.alloc() : memref<ShapeTyxElType>
//      memref.copy %reinterpret_cast, %alloc : memref<...> to memref<...>
//      %... = bufferization.to_tensor %alloc restrict writable : memref<...>
//      ...
// ```
// is replaced by
// ```
//      %reinterpret_cast = memref.reinterpret_cast %arg0 to ...]
//                          : memref<*xElType> to memref<ShapeTyxElType>
//      %... = bufferization.to_tensor %reinterpret_cast restrict : memref<...>
// ```
//
// Things can quickly get complicated when interacting with memref views and
// loops. e.g.
// ```
//   func.func @foo(%arg0: memref<*xf16>, ...) {
//      ...
//      %reinterpret_cast = memref.reinterpret_cast %arg0 to ...]
//                          : memref<*xf16> to memref<ShapeTyxElType>
//      ... = scf.for %i = %start to %stop step %step
//                iter_args(..., %travelling_memref = %reinterpret_cast)
//                      -> (..., memref<...>)  : i32 {
//            ...
//            %alloc = memref.alloc() : memref<...>
//            memref.copy %travelling_memref, %alloc :  memref<...> to
//            memref<...>
//            ... =  bufferization.to_tensor %alloc restrict writable :
//            memref<...>
//            ...
//            %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to ...
//           scf.yield ..., %reinterpret_cast_0 : ...
//        }
// ```
// We are able to handle such complicated cases using buffer alias analysis.

#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "mlir/Dialect/Bufferization/Transforms/BufferViewFlowAnalysis.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/Support/Debug.h"
#include <algorithm>
#include <numeric>
#include <vector>

#include "hexagon/Conversion/LinalgToLLVM/Common.h"
#include "hexagon/Transforms/Transforms.h"

#define DEBUG_TYPE "hexagon-punt-buffer"

#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define DBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
using namespace hexagon;

#define GEN_PASS_DEF_HEXAGONPUNTBUFFER
#include "hexagon/Transforms/Passes.h.inc"

namespace {

using PuntCandidates =
    std::vector<std::pair<memref::CopyOp, bufferization::ToTensorOp>>;

struct HexagonPuntBufferPass
    : public ::impl::HexagonPuntBufferBase<HexagonPuntBufferPass> {

  void runOnOperation() override;
};

// Use alias-analysis framework to work out if a particular memref value is
// a view of the kernel input memref arg. The check below is not most efficient
// but when reverse resolve works this can be reworked to make it efficient.
static bool isaViewOfFuncInputBuffer(Value value, FunctionOpInterface funcOp) {
  BufferViewFlowAnalysis analysis(funcOp);
  for (const auto &arg : funcOp.getArguments()) {
    if (isa<BaseMemRefType>(arg.getType())) {
      auto viewSet = analysis.resolve(arg);
      for (auto view : viewSet) {
        if (value == view)
          return true;
      }
    }
  }
  return false;
}

// Populate punting candidates iff copy involves `source` that is a view
// of kernel input arg (checked separately), and `target` is an alloc
// with no further use.
bool populatePuntCandidate(memref::CopyOp copyOp, PuntCandidates &candidates) {
  // Check we are copying to an allocation.
  auto alloc = copyOp.getTarget().getDefiningOp<memref::AllocOp>();
  if (!alloc)
    return false;

  auto uses = copyOp.getTarget().getUses();
  if (std::distance(uses.begin(), uses.end()) != 2)
    return false;

  auto firstUse = uses.begin()->getOwner();
  auto secondUse = std::next(uses.begin())->getOwner();
  if (!firstUse || !secondUse || (firstUse != copyOp && secondUse != copyOp))
    return false;

  auto otherUse = firstUse == copyOp ? secondUse : firstUse;
  auto toTensor = llvm::dyn_cast<bufferization::ToTensorOp>(otherUse);
  if (!toTensor)
    return false;
  candidates.emplace_back(copyOp, toTensor);
  DBG("___target is a valid single use alloc");
  return true;
}

void HexagonPuntBufferPass::runOnOperation() {
  auto funcOp = getOperation();
  IRRewriter rewriter(&getContext());
  PuntCandidates candidates;

  funcOp.walk([&](memref::CopyOp copyOp) {
    DBG("punt buffer candidate: " << copyOp);

    if (!isaViewOfFuncInputBuffer(copyOp.getSource(), funcOp)) {
      DBG("___src is not a view of memref func arg");
      return WalkResult::advance();
    }

    populatePuntCandidate(copyOp, candidates);
    return WalkResult::advance();
  });

  // Punt the buffer use and erase the copy.
  for (auto &candidate : candidates) {
    auto copyOp = candidate.first;
    DBG("punted : " << copyOp);
    // Do the replacement.
    rewriter.replaceAllUsesWith(copyOp.getTarget(), copyOp.getSource());

    // We don't need the copy anymore.
    rewriter.eraseOp(copyOp);

    // Make sure to_tensor is no longer `writable`.
    candidate.second->removeAttr("writable");

    // Don't erase alloc as cleaning passes will do that.
  }
}
} // namespace

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
hexagon::createHexagonPuntBufferPass() {
  return std::make_unique<HexagonPuntBufferPass>();
}
