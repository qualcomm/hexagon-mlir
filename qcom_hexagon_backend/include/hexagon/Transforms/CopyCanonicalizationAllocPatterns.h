//===- CopyCanonicalizationPatterns.h - copy-canon populate fns -----------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//

#ifndef HEXAGON_TRANSFORMS_COPYCANONICALIZATIONPATTERNS_H
#define HEXAGON_TRANSFORMS_COPYCANONICALIZATIONPATTERNS_H

namespace mlir {
class DominanceInfo;
class RewritePatternSet;

namespace hexagon {

/// Populate `patterns` with the alloc-anchored copy-canonicalization
/// rewrites: `RedirectAllocWritesToDestinationPattern` and
/// `EliminateAllocInIfWithCopyPattern`.  `domInfo` (typically from
/// `getAnalysis<DominanceInfo>()`) must outlive the populated set.
void populateCopyCanonicalizationAllocPatterns(RewritePatternSet &patterns,
                                               DominanceInfo &domInfo);

} // namespace hexagon
} // namespace mlir

#endif // HEXAGON_TRANSFORMS_COPYCANONICALIZATIONPATTERNS_H
