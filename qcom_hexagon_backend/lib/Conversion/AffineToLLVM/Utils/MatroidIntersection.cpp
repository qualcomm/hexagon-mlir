//===------------ MatroidIntersection.cpp - Matroid Intersection ----------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// This file implements matroid intersection routines.
//===----------------------------------------------------------------------===//

/*
 * Matroid intersection is computed with an augmenting path algorithm which uses
 * a circuit-finding oracle to construct the exchange graph. I found these video
 * lectures helpful: <https://www.youtube.com/watch?v=ftEgEYjJEak>.
 *
 * Some notes on the linear-partition intersection algorithm:
 * 1. The partition matroid has very simple structure: there is always one back-
 * edge from a node outside the current selection; it points to the element that
 * currently occupies that node's group: in order to select a node, either it is
 * a sink (it is in an unoccupied group), or you must kick out the current group
 * occupant.
 * 2. For the prefix-only case, it is sufficient to choose augmenting paths that
 * only use elements in the current prefix (groups) and end in the next group to
 * be added. First, observe that we never terminate prematurely: given a maximal
 * independent set, all prefixes are also independent sets, so each prefix has a
 * solution. Next, observe that there always exists some augmenting path from an
 * assignment (except if it is of maximum size, in which case we are done.) Take
 * the symmetric difference of the current assignment and the (i+1)-th prefix of
 * a maximum-size independent set (let this be D_i); by the basis exchange lemma
 * we know some augmenting path exists that is a subset of D_i, and observe that
 * the only sink (element that is independent in the partition matroid) must lie
 * in the (i+1)-th group, since every previous group is occupied, and D_i cannot
 * have any elements outside the prefix, and thus the only unoccupied group is i
 * + 1. Thus the augmenting path is also a valid path in the restricted exchange
 * graph, and so we are guaranteed forward progress.
 */

#include "hexagon/Conversion/AffineToLLVM/Utils/MatroidIntersection.h"
#include "hexagon/Conversion/AffineToLLVM/Utils/BareissLU.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/ADT/SparseBitVector.h"
#include <deque>

using namespace mlir;
using namespace mlir::presburger;
using namespace mlir::hexagon;
using namespace llvm;

namespace mlir {
namespace hexagon {

constexpr unsigned EMPTY = unsigned(-1);
constexpr unsigned START = unsigned(-1);

SmallVector<unsigned> matroidIntersectionLP(IntMatrix &&mat,
                                            ArrayRef<unsigned> groups,
                                            bool prefixOnly) {
  BareissLU lu(std::forward<IntMatrix>(mat));
  assert(groups.size() == mat.getNumRows());
  if (groups.size() == 0) {
    return lu.I;
  }
  unsigned numGroups = *std::max_element(groups.begin(), groups.end()) + 1;
  unsigned N = mat.getNumRows();
  // Occupancy of each group.
  SmallVector<unsigned> occ(numGroups, EMPTY);
  // Parent of each OUT row in current augmenting path.
  SmallVector<unsigned> parent(N, START);
  // IN node for each OUT node visited.
  SmallVector<unsigned> viaIn(N, START);
  // Visited for BFS.
  SmallBitVector visited(N);
  // End index of current group. Unused if not prefix.
  unsigned curEnd = 0;
  // Queue.
  std::deque<unsigned> Q;
  // Found sink.
  bool foundSink;
  unsigned sink;
  // Fundamental circuit.
  llvm::SmallVector<unsigned> circuit;
  // Augmenting path.
  llvm::SmallVector<unsigned> path;
  // Circuit cache.
  // Bit (N + 1) * i + 1 + j is set if j is in i's circuit.
  // Bit (N + 1) * i is set if the circuit is computed.
  llvm::SparseBitVector circuitCache;
  // Main loop.
  for (unsigned t = 0; t < numGroups; t++) {
    // Fast scan for independent element in new group.
    bool updated = false;
    if (prefixOnly) {
      for (; curEnd < N && groups[curEnd] == t; curEnd++) {
        if (!updated && !lu.checkDependence(curEnd, nullptr)) {
          lu.update({curEnd});
          occ[t] = curEnd;
          updated = true;
        }
      }
      assert(curEnd == N || groups[curEnd] > t);
    } else {
      for (unsigned i = 0; i < N; i++) {
        if (occ[groups[i]] == EMPTY && !lu.checkDependence(i, nullptr)) {
          lu.update({i});
          occ[groups[i]] = i;
          updated = true;
          break;
        }
      }
    }
    if (updated) {
      continue;
    }
    // BFS for augmenting path.
    visited.reset();
    circuitCache.clear();
    std::fill(parent.begin(), parent.end(), START);
    Q.clear();
    unsigned end = prefixOnly ? curEnd : N;
    // Initialize Q with sources: independent rows.
    {
      auto jit = lu.I.begin(), jend = lu.I.end();
      for (unsigned i = 0; i < end; i++) {
        if (jit != jend && i == *jit) {
          ++jit;
          continue;
        }
        if (!lu.checkDependence(i, nullptr)) {
          visited.set(i);
          parent[i] = START;
          Q.push_back(i);
        }
      }
    }
    // BFS loop.
    foundSink = false;
    while (!Q.empty()) {
      unsigned j = Q.front();
      Q.pop_front();
      unsigned g = groups[j];
      // If prefix only, must augment along a path that ends in next group.
      bool isSink = prefixOnly ? g == t : occ[g] == EMPTY;
      if (isSink) {
        sink = j;
        foundSink = true;
        break;
      }
      unsigned i = occ[g];
      assert(i != EMPTY); // If prefix, all groups should be occupied.
      auto jit = lu.I.begin(), jend = lu.I.end();
      for (unsigned jp = 0; jp < end; jp++) {
        if (jit != jend && jp == *jit) {
          ++jit;
          continue;
        }
        if (visited[jp]) {
          continue;
        }
        unsigned cacheStart = (N + 1) * jp;
        if (!circuitCache.test(cacheStart)) {
          circuitCache.set(cacheStart);
          if (lu.checkDependence(jp, &circuit)) {
            for (unsigned el : circuit) {
              circuitCache.set(cacheStart + 1 + el);
            }
          }
        }
        if (circuitCache.test(cacheStart + 1 + i)) {
          visited.set(jp);
          parent[jp] = j;
          viaIn[jp] = i;
          Q.push_back(jp);
        }
      }
    }
    if (!foundSink) {
      break;
    }
    // Augment along path
    unsigned v = sink;
    path.clear();
    while (v != START) {
      unsigned j = v;
      if (parent[j] != START) {
        unsigned i = viaIn[j];
        path.push_back(i);
        unsigned g = groups[i];
        assert(occ[g] == i);
        occ[g] = EMPTY;
      }
      path.push_back(j);
      occ[groups[j]] = j;
      v = parent[j];
    }
    std::sort(path.begin(), path.end());
    lu.update(path);
    assert(std::is_sorted(lu.I.begin(), lu.I.end()));
  }
  return lu.I;
}

} // namespace hexagon
} // namespace mlir
