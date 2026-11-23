//===- LLVMAlignmentPropagationPass.cpp - Propagate alignment to LLVM ----===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// This pass propagates alignment information from triton kernel arguments
// through reinterpret cast and extract_strided_metadata,
// through the LLVM conversion chain to llvm.load and llvm.store
// operations. It handles the following chain:
//   memref.reinterpret_cast {alignment = N}
//   -> memref.extract_strided_metadata (extracts base_buffer)
//   -> memref.reinterpret_cast (loses alignment)
//   -> builtin.unrealized_conversion_cast
//   -> llvm.extractvalue
//   -> llvm.getelementptr
//   -> llvm.load/store (should have alignment = N)
//
// Block-argument handling
// -----------------------
// For loop induction variables (block arguments) we run an *optimistic*
// propagator inside the fixed point. For each block with arguments we use
// DominanceInfo to classify predecessors as forward edges (the predecessor
// is not dominated by the block) or back edges (the predecessor is
// dominated by the block). Each block arg gets a BlockArgSeed that lists
// its incoming operands. On every fixed-point iteration we recompute the
// seed's alignment as the GCD of operand contributions (constants,
// existing valueDivisibility entries, and the IV-pattern shortcut
// `arg + const` for back-edge operands). If every operand contributes,
// the seed is *resolved*; otherwise it is unresolved and any value whose
// provenance touches it is skipped at apply time so we never publish an
// unsound alignment attribute.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/SCCIterator.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/Debug.h"

#include "hexagon/Common/Common.h"
#include "hexagon/Conversion/LinalgToLLVM/Common.h"
#include "hexagon/Conversion/LinalgToLLVM/LinalgToLLVM.h"
#define GEN_PASS_DEF_LLVMALIGNMENTPROPAGATION
#include "hexagon/Dialect/HexagonMem/IR/HexagonMemDialect.h"
#include "hexagon/Transforms/Transforms.h"

#include <numeric>

#define DEBUG_TYPE "llvm-alignment-propagation"

#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define DBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
using namespace hexagon;

// Intra-module call graph: one node per FuncOp, virtual super-root → kernels.
struct CallNode {
  llvm::StringRef name;
  func::FuncOp funcOp;                      // null for the super-root
  llvm::SmallVector<CallNode *, 4> callees; // caller→callee edges
  llvm::SmallVector<CallNode *, 4> callers; // reverse edges
};

struct IntraModuleCallGraph {
  CallNode superRoot; // virtual; name = ""
  std::vector<std::unique_ptr<CallNode>> nodes;
  llvm::DenseMap<llvm::StringRef, CallNode *> nodesByName;
};

namespace llvm {
template <> struct GraphTraits<CallNode *> {
  using NodeRef = CallNode *;
  using ChildIteratorType = SmallVector<CallNode *, 4>::iterator;
  static NodeRef getEntryNode(NodeRef n) { return n; }
  static ChildIteratorType child_begin(NodeRef n) { return n->callees.begin(); }
  static ChildIteratorType child_end(NodeRef n) { return n->callees.end(); }
};
} // namespace llvm

// GCD accumulator across all processed callers per (callee, argIndex).
struct CalleeArgState {
  uint64_t divisibility = 0;
  bool unknown = false; // true if any caller had no info
};
using CalleeArgAccum =
    llvm::DenseMap<StringRef, llvm::SmallDenseMap<unsigned, CalleeArgState>>;

namespace {

//===----------------------------------------------------------------------===//
// Provenance-tracking divisibility map
//===----------------------------------------------------------------------===//

/// Set of block-argument seeds whose optimistic divisibility a value
/// transitively depends on. A non-empty set means "this entry's
/// correctness is conditioned on the listed seeds being resolved at apply
/// time". Two-element small-pointer-set covers virtually every realistic
/// case (one IV per loop, occasionally two for nested loops).
using SeedSet = llvm::SmallPtrSet<BlockArgument, 2>;

struct ValueDivisibilityInfo {
  uint64_t divisibility = 0;
  SeedSet seeds;
};

using DivMap = llvm::DenseMap<Value, ValueDivisibilityInfo>;

/// Insert or merge `seeds` into `dst`. Returns true if `dst` grew.
static bool unionSeeds(SeedSet &dst, const SeedSet &src) {
  bool grew = false;
  for (BlockArgument s : src)
    grew |= dst.insert(s).second;
  return grew;
}

/// Insert into `map[val]` unless an entry already exists. Returns true if
/// the map was modified.
static bool setDivisibility(DivMap &map, Value val, uint64_t divisibility,
                            const SeedSet &seeds, llvm::StringRef dbgMsg) {
  auto [it, inserted] = map.try_emplace(val);
  if (inserted) {
    it->second.divisibility = divisibility;
    it->second.seeds = seeds;
    LLVM_DEBUG(DBGS() << dbgMsg << divisibility << " (seeds=" << seeds.size()
                      << ")\n");
    return true;
  }
  // Already present. Keep the existing divisibility (the first writer wins
  // for compatibility with the previous behavior) but make sure the
  // seed set captures every dependency that has produced this entry, so
  // that apply-time filtering is sound.
  return unionSeeds(it->second.seeds, seeds);
}

/// Look up `source` in `map` and propagate to `dest` (divisibility + seeds).
static bool lookupAndPropagate(DivMap &map, Value source, Value dest,
                               llvm::StringRef dbgMsg) {
  auto it = map.find(source);
  if (it == map.end())
    return false;
  return setDivisibility(map, dest, it->second.divisibility, it->second.seeds,
                         dbgMsg);
}

//===----------------------------------------------------------------------===//
// Block-argument optimistic seeding
//===----------------------------------------------------------------------===//

/// One seed per block argument that can carry a useful alignment. The
/// seed lists every operand reaching the block argument so we can
/// re-evaluate it on every fixed-point iteration.
struct BlockArgSeed {
  BlockArgument arg;
  llvm::SmallVector<Value, 2> forwardOperands;
  llvm::SmallVector<Value, 2> backOperands;
  uint64_t divisibility = 0;
  // True iff every forward and back operand contributed to the divisibility
  // on the most recent recompute. Apply phase trusts a value that
  // depends on this seed only when `resolved` is true.
  bool resolved = false;
};

using SeedMap = llvm::DenseMap<BlockArgument, BlockArgSeed>;

/// Collect the predecessor operand reaching `arg` along edge `pred -> block`.
/// Currently only `cf.br` is supported. Returns std::nullopt if the
/// predecessor is not a `cf.br` (e.g., a `cf.cond_br`'s edge into a
/// block-with-arguments — exceedingly rare in practice).
static std::optional<Value> getDestOperandForArg(Block *pred, Block *block,
                                                 unsigned argIndex) {
  auto branchOp = dyn_cast<cf::BranchOp>(pred->getTerminator());
  if (!branchOp || branchOp.getDest() != block)
    return std::nullopt;
  auto destOperands = branchOp.getDestOperands();
  if (argIndex >= destOperands.size())
    return std::nullopt;
  return destOperands[argIndex];
}

/// Build BlockArgSeed entries for every block with arguments. Predecessors
/// are classified as forward / back edges via DominanceInfo.
static void initializeBlockArgSeeds(func::FuncOp funcOp, DominanceInfo &domInfo,
                                    SeedMap &seeds) {
  for (Block &block : funcOp.getBody()) {
    if (block.getNumArguments() == 0)
      continue;
    for (unsigned i = 0; i < block.getNumArguments(); ++i) {
      BlockArgument arg = block.getArgument(i);
      // Only index- or integer-typed arguments carry useful divisibility
      // information (e.g. loop induction variables). Track both: after the
      // arith->LLVM lowering a loop counter is frequently a plain `i32`
      // block argument rather than `index`. Skipping integer IVs left their
      // downstream `muli`/offset values with divisibility 0 ("unknown"),
      // which `std::gcd(base, 0) == base` then treated as *no tightening* --
      // wrongly preserving a full 128-byte alignment on per-row sub-HVX
      // accesses whose real stride is only 64 bytes. Filtering out non-scalar
      // types (e.g. memref descriptors) is still important.
      if (!arg.getType().isIndex() && !arg.getType().isSignlessInteger())
        continue;
      BlockArgSeed seed;
      seed.arg = arg;
      for (Block *pred : block.getPredecessors()) {
        auto operand = getDestOperandForArg(pred, &block, i);
        if (!operand)
          continue;
        // Forward edge: predecessor is *not* dominated by the destination.
        bool isForward = !domInfo.dominates(&block, pred);
        if (isForward)
          seed.forwardOperands.push_back(*operand);
        else
          seed.backOperands.push_back(*operand);
      }
      if (seed.forwardOperands.empty() && seed.backOperands.empty())
        continue;
      seeds.try_emplace(arg, std::move(seed));
    }
  }
  DBG("Initialized " << seeds.size() << " block-arg seeds");
}

/// Try to extract a constant integer value from a Value
static std::optional<int64_t> getConstantIntValue(Value val) {
  if (auto constOp = val.getDefiningOp<arith::ConstantOp>()) {
    if (auto intAttr = llvm::dyn_cast<IntegerAttr>(constOp.getValue())) {
      return intAttr.getInt();
    }
  }
  return std::nullopt;
}

/// Attempt to extract `(seed.arg, const)` from the canonical IV increment
/// pattern `arg + const` (or `const + arg`). Returns the constant on match.
static std::optional<int64_t> getIVConstShortcut(Value v, BlockArgument arg) {
  auto addOp = v.getDefiningOp<arith::AddIOp>();
  if (!addOp)
    return std::nullopt;
  Value lhs = addOp.getLhs();
  Value rhs = addOp.getRhs();
  if (lhs == arg) {
    if (auto c = getConstantIntValue(rhs))
      return *c;
  } else if (rhs == arg) {
    if (auto c = getConstantIntValue(lhs))
      return *c;
  }
  return std::nullopt;
}

/// Recompute one seed's alignment + resolved status from the current
/// `valueDivisibility`. Returns true if anything changed.
static bool recomputeSeed(BlockArgSeed &seed, const DivMap &map) {
  uint64_t newAlign = 0;
  bool allResolved = true;
  SeedSet derivedSeeds;
  derivedSeeds.insert(seed.arg);

  auto contribute = [&](uint64_t a) {
    newAlign = (newAlign == 0) ? a : std::gcd(newAlign, a);
  };

  auto resolveOperand = [&](Value v, bool allowIVShortcut) {
    if (auto c = getConstantIntValue(v)) {
      int64_t val = *c;
      if (val == 0)
        return;
      contribute(static_cast<uint64_t>(std::abs(val)));
      return;
    }
    auto it = map.find(v);
    if (it != map.end() && it->second.divisibility > 0) {
      contribute(it->second.divisibility);
      unionSeeds(derivedSeeds, it->second.seeds);
      return;
    }
    if (allowIVShortcut) {
      if (auto c = getIVConstShortcut(v, seed.arg)) {
        contribute(static_cast<uint64_t>(std::abs(*c)));
        return;
      }
    }
    allResolved = false;
  };

  for (Value v : seed.forwardOperands)
    resolveOperand(v, /*allowIVShortcut=*/false);
  for (Value v : seed.backOperands)
    resolveOperand(v, /*allowIVShortcut=*/true);

  bool changed = false;
  if (newAlign != seed.divisibility) {
    seed.divisibility = newAlign;
    changed = true;
  }
  if (allResolved != seed.resolved) {
    seed.resolved = allResolved;
    changed = true;
  }
  return changed;
}

/// Run a single pass of seed recomputation, write the new alignments into
/// `valueDivisibility`. Returns true if anything changed.
static bool propagateThroughBlockArgs(SeedMap &seeds,
                                      DivMap &valueDivisibility) {
  bool anyChange = false;
  for (auto &kv : seeds) {
    BlockArgSeed &seed = kv.second;
    if (recomputeSeed(seed, valueDivisibility))
      anyChange = true;
    if (seed.divisibility == 0)
      continue;
    SeedSet provenance;
    provenance.insert(seed.arg);
    auto [it, inserted] = valueDivisibility.try_emplace(seed.arg);
    if (inserted) {
      it->second.divisibility = seed.divisibility;
      it->second.seeds = provenance;
      anyChange = true;
    } else {
      // The block arg's tracked alignment is the (optimistic) seed value.
      // Keep it in sync as the seed evolves across iterations.
      if (it->second.divisibility != seed.divisibility) {
        it->second.divisibility = seed.divisibility;
        anyChange = true;
      }
      anyChange |= unionSeeds(it->second.seeds, provenance);
    }
  }
  return anyChange;
}

//===----------------------------------------------------------------------===//
// Element-size helper
//===----------------------------------------------------------------------===//

static std::optional<uint64_t> getElementSizeInBytes(Type elementType) {
  constexpr uint64_t bitsPerByte = 8;
  constexpr uint64_t hexagonPtrSize = 4;
  if (auto intType = llvm::dyn_cast<IntegerType>(elementType)) {
    return intType.getWidth() / bitsPerByte;
  } else if (auto floatType = llvm::dyn_cast<FloatType>(elementType)) {
    return floatType.getWidth() / bitsPerByte;
  } else if (auto ptrType =
                 llvm::dyn_cast<LLVM::LLVMPointerType>(elementType)) {
    return hexagonPtrSize;
  }
  return std::nullopt;
}

//===----------------------------------------------------------------------===//
// Per-op propagators
//===----------------------------------------------------------------------===//

static void seedConstOpDivisibility(func::FuncOp funcOp, DivMap &map) {
  funcOp.walk([&](arith::ConstantOp constOp) {
    if (auto intAttr = llvm::dyn_cast<IntegerAttr>(constOp.getValue())) {
      int64_t value = intAttr.getInt();
      SeedSet noSeeds;
      if (value >= 0)
        setDivisibility(map, constOp.getResult(), static_cast<uint64_t>(value),
                        noSeeds, "Found constant with divisibility = ");
    }
  });
}

static void seedMemRefAllocDivisibility(func::FuncOp funcOp, DivMap &map) {
  funcOp.walk([&](memref::AllocOp allocOp) {
    if (auto alignAttr = allocOp->getAttrOfType<IntegerAttr>("alignment")) {
      SeedSet noSeeds;
      setDivisibility(map, allocOp.getResult(), alignAttr.getInt(), noSeeds,
                      "Found memref.alloc with alignment = ");
    }
  });
}

static void seedHexagonMemAllocDivisibility(func::FuncOp funcOp, DivMap &map) {
  funcOp.walk([&](hexagonmem::AllocOp allocOp) {
    SeedSet noSeeds;
    setDivisibility(map, allocOp.getBuffer(), allocOp.getAlignment(), noSeeds,
                    "Found hexagonmem.alloc with alignment = ");
  });
}

static void propagateThroughMulI(func::FuncOp funcOp, DivMap &map) {
  funcOp.walk([&](arith::MulIOp mulOp) {
    Value lhs = mulOp.getLhs();
    Value rhs = mulOp.getRhs();
    Value result = mulOp.getResult();

    if (map.find(result) != map.end())
      return;

    auto lhsIt = map.find(lhs);
    auto rhsIt = map.find(rhs);

    uint64_t lhsMult = 0, rhsMult = 0;
    SeedSet seeds;
    if (lhsIt != map.end()) {
      lhsMult = lhsIt->second.divisibility;
      unionSeeds(seeds, lhsIt->second.seeds);
    }
    if (rhsIt != map.end()) {
      rhsMult = rhsIt->second.divisibility;
      unionSeeds(seeds, rhsIt->second.seeds);
    }

    if (lhsMult == 0 && rhsMult == 0)
      return;

    uint64_t resultMult = lhsMult * rhsMult;
    setDivisibility(map, result, resultMult, seeds,
                    "Propagated multiplicity through arith.muli: ");
  });
}

static void propagateThroughIndexCast(func::FuncOp funcOp, DivMap &map) {
  funcOp.walk([&](arith::IndexCastOp castOp) {
    lookupAndPropagate(map, castOp.getOperand(), castOp.getResult(),
                       "Propagated multiplicity through arith.index_cast: ");
  });
}

static void propagateThroughAddI(func::FuncOp funcOp, DivMap &map) {
  funcOp.walk([&](arith::AddIOp addOp) {
    Value lhs = addOp.getLhs();
    Value rhs = addOp.getRhs();
    Value result = addOp.getResult();

    if (map.find(result) != map.end())
      return;

    auto lhsIt = map.find(lhs);
    auto rhsIt = map.find(rhs);
    auto lhsConst = getConstantIntValue(lhs);
    auto rhsConst = getConstantIntValue(rhs);

    uint64_t lhsDiv = 0, rhsDiv = 0;
    bool lhsKnown = false, rhsKnown = false;
    SeedSet seeds;
    if (lhsIt != map.end()) {
      lhsDiv = lhsIt->second.divisibility;
      lhsKnown = true;
      unionSeeds(seeds, lhsIt->second.seeds);
    } else if (lhsConst.has_value()) {
      lhsDiv = static_cast<uint64_t>(std::abs(*lhsConst));
      lhsKnown = true;
    }
    if (rhsIt != map.end()) {
      rhsDiv = rhsIt->second.divisibility;
      rhsKnown = true;
      unionSeeds(seeds, rhsIt->second.seeds);
    } else if (rhsConst.has_value()) {
      rhsDiv = static_cast<uint64_t>(std::abs(*rhsConst));
      rhsKnown = true;
    }

    if (lhsKnown && rhsKnown) {
      setDivisibility(map, result, std::gcd(lhsDiv, rhsDiv), seeds,
                      "Propagated divisibility through arith.addi: ");
    }
  });
}

static void propagateThroughExtractStridedMetadata(func::FuncOp funcOp,
                                                   DivMap &map) {
  funcOp.walk([&](memref::ExtractStridedMetadataOp extractOp) {
    Value source = extractOp.getSource();
    lookupAndPropagate(map, source, extractOp.getBaseBuffer(),
                       "Propagated alignment through "
                       "extract_strided_metadata to base_buffer: ");
    lookupAndPropagate(map, source, extractOp.getOffset(),
                       "Propagated alignment through "
                       "extract_strided_metadata to offset: ");
  });
}

/// Per-op information stashed during propagation; written to the IR in
/// the apply phase.
struct ReinterpretCastAttr {
  uint64_t alignment;
  SeedSet seeds;
};
using ReinterpretCastAttrMap =
    llvm::DenseMap<memref::ReinterpretCastOp, ReinterpretCastAttr>;

/// Propagate through memref.reinterpret_cast.
///
/// Two alignments are tracked:
///   * `resultAlignment` (element-0): GCD(baseAlignment, |offset|*elementSize).
///     Goes into `valueDivisibility[castOp.getResult()]` so the data pointer
///     extracted via the unrealized_conversion_cast → extractvalue → GEP
///     chain stays element-0-precise.
///   * `attrAlignment`: a stride-aware lower bound, kept in a side map and
///     written as the `alignment` attribute on the op only in the apply
///     phase. Per-rank stride contributes |stride[i] - 1| * elementSize so
///     unit stride (the contiguous case) is a no-op.
static void propagateThroughReinterpretCast(func::FuncOp funcOp, DivMap &map,
                                            ReinterpretCastAttrMap &rcAttrs) {
  funcOp.walk([&](memref::ReinterpretCastOp castOp) {
    Value source = castOp.getSource();
    auto it = map.find(source);
    if (it == map.end())
      return;

    uint64_t baseAlignment = it->second.divisibility;
    SeedSet seeds = it->second.seeds;
    uint64_t resultAlignment = baseAlignment;

    auto resultType = llvm::cast<MemRefType>(castOp.getResult().getType());
    auto elementSizeOpt = getElementSizeInBytes(resultType.getElementType());
    if (!elementSizeOpt.has_value())
      return;
    uint64_t elementSize = elementSizeOpt.value();

    auto offsets = castOp.getMixedOffsets();
    if (offsets.size() == 1) {
      auto offset = offsets[0];
      if (auto offsetAttr = llvm::dyn_cast_if_present<Attribute>(offset)) {
        int64_t offsetValue = llvm::cast<IntegerAttr>(offsetAttr).getInt();
        uint64_t offsetBytes =
            static_cast<uint64_t>(std::abs(offsetValue)) * elementSize;
        resultAlignment = std::gcd(baseAlignment, offsetBytes);
      } else if (auto offsetValue = llvm::dyn_cast_if_present<Value>(offset)) {
        auto argIt = map.find(offsetValue);
        if (argIt != map.end()) {
          uint64_t offsetBytes = argIt->second.divisibility * elementSize;
          // Safety net: a divisibility of 0 means "offset unknown". Since
          // std::gcd(base, 0) == base, feeding it directly would leave the
          // base alignment *untightened* and wrongly claim the base's
          // (large) alignment for a dynamically-offset subview. An unknown
          // runtime offset can land on any element boundary, so the only
          // sound lower bound is the element alignment.
          if (offsetBytes == 0)
            resultAlignment = std::gcd(baseAlignment, elementSize);
          else
            resultAlignment = std::gcd(baseAlignment, offsetBytes);
          unionSeeds(seeds, argIt->second.seeds);
        } else {
          return;
        }
      }
    }

    // Fold per-rank strides into a separate value used only for the
    // documented attribute. Unit stride contributes 0 (no tightening).
    // Stride 0 is broadcast and is skipped. Dynamic strides contribute
    // elementSize (loosest non-trivial bound).
    uint64_t attrAlignment = resultAlignment;
    for (auto stride : castOp.getMixedStrides()) {
      if (auto strideAttr = llvm::dyn_cast_if_present<Attribute>(stride)) {
        int64_t strideValue = llvm::cast<IntegerAttr>(strideAttr).getInt();
        if (strideValue == 0)
          continue;
        uint64_t diffBytes =
            static_cast<uint64_t>(std::abs(strideValue - int64_t{1})) *
            elementSize;
        if (diffBytes != 0)
          attrAlignment = std::gcd(attrAlignment, diffBytes);
      } else if (llvm::dyn_cast_if_present<Value>(stride)) {
        attrAlignment = std::gcd(attrAlignment, elementSize);
      }
    }

    // Update `valueDivisibility` immediately (element-0 alignment +
    // provenance). This drives the rest of the propagation; the attribute write
    // is deferred.
    setDivisibility(map, castOp.getResult(), resultAlignment, seeds,
                    "reinterpret_cast result element-0 alignment = ");
    rcAttrs[castOp] = {attrAlignment, seeds};
  });
}

static void propagateThroughUnrealizedConversionCast(func::FuncOp funcOp,
                                                     DivMap &map) {
  funcOp.walk([&](UnrealizedConversionCastOp castOp) {
    for (Value input : castOp.getInputs())
      for (Value output : castOp.getOutputs())
        lookupAndPropagate(map, input, output,
                           "Propagated alignment through "
                           "unrealized_conversion_cast: ");
  });
}

static void propagateThroughExtractValue(func::FuncOp funcOp, DivMap &map) {
  funcOp.walk([&](LLVM::ExtractValueOp extractOp) {
    Value container = extractOp.getContainer();
    auto it = map.find(container);
    if (it == map.end())
      return;
    bool isMemrefStruct = false;
    if (auto castOp = container.getDefiningOp<UnrealizedConversionCastOp>()) {
      if (castOp.getInputs().size() == 1 &&
          llvm::isa<MemRefType>(castOp.getInputs()[0].getType()))
        isMemrefStruct = true;
    }
    if (!isMemrefStruct)
      return;
    setDivisibility(map, extractOp.getResult(), it->second.divisibility,
                    it->second.seeds,
                    "Propagated alignment through llvm.extractvalue: ");
  });
}

static void propagateThroughGEP(func::FuncOp funcOp, DivMap &map) {
  funcOp.walk([&](LLVM::GEPOp gepOp) {
    Value base = gepOp.getBase();
    auto baseIt = map.find(base);
    if (baseIt == map.end())
      return;
    Type elementType = gepOp.getElemType();
    auto elementSizeOpt = getElementSizeInBytes(elementType);
    if (!elementSizeOpt.has_value())
      return;
    uint64_t elementSize = elementSizeOpt.value();
    auto indices = gepOp.getIndices();
    if (indices.size() != 1)
      return;
    auto idxValue = llvm::dyn_cast_if_present<Value>(indices[0]);
    if (!idxValue)
      return;
    auto idxIt = map.find(idxValue);
    if (idxIt == map.end())
      return;
    uint64_t baseAlignment = baseIt->second.divisibility;
    uint64_t indexBytes = idxIt->second.divisibility * elementSize;
    uint64_t resultAlignment = std::gcd(baseAlignment, indexBytes);
    SeedSet seeds = baseIt->second.seeds;
    unionSeeds(seeds, idxIt->second.seeds);
    setDivisibility(map, gepOp.getResult(), resultAlignment, seeds,
                    "GEP result alignment = ");
  });
}

//===----------------------------------------------------------------------===//
// Apply phase
//===----------------------------------------------------------------------===//

static bool isTainted(const SeedSet &seeds,
                      const llvm::SmallPtrSetImpl<BlockArgument> &unresolved) {
  for (BlockArgument s : seeds)
    if (unresolved.contains(s))
      return true;
  return false;
}

template <typename MemOp, typename GetDataTypeFn>
static void
setMemOpAlignments(func::FuncOp funcOp, DivMap &map, GetDataTypeFn getDataType,
                   llvm::StringRef opName,
                   const llvm::SmallPtrSetImpl<BlockArgument> &unresolved) {
  funcOp.walk(
      [&](MemOp memOp) {
        uint64_t alignment = 0;
        auto it = map.find(memOp.getAddr());
        if (it != map.end()) {
          if (isTainted(it->second.seeds, unresolved)) {
            DBG("Skipping "
                << opName
                << " — provenance touches an unresolved block-arg seed");
          } else {
            alignment = std::gcd(static_cast<uint64_t>(kLargeAlignment),
                                 it->second.divisibility);
          }
        }
        if (alignment == 0) {
          if (auto nat = getElementSizeInBytes(getDataType(memOp))) {
            alignment = *nat;
            DBG("Using natural alignment = " << alignment << " for " << opName
                                             << " (no tracked alignment)");
          }
        }
        if (alignment == 0)
          return;
        uint64_t currentAlignment = 0;
        if (auto alignAttr = memOp.getAlignmentAttr())
          currentAlignment = alignAttr.getValue().getZExtValue();
        if (alignment > currentAlignment) {
          OpBuilder builder(memOp);
          memOp.setAlignmentAttr(builder.getI64IntegerAttr(alignment));
          DBG("Set alignment = " << alignment << " on " << opName << " (was "
                                 << currentAlignment << ")");
        }
      });
}

static void
setLoadOpAlignments(func::FuncOp funcOp, DivMap &map,
                    const llvm::SmallPtrSetImpl<BlockArgument> &unresolved) {
  setMemOpAlignments<LLVM::LoadOp>(
      funcOp, map, [](LLVM::LoadOp op) { return op.getResult().getType(); },
      "llvm.load", unresolved);
}

static void
setStoreOpAlignments(func::FuncOp funcOp, DivMap &map,
                     const llvm::SmallPtrSetImpl<BlockArgument> &unresolved) {
  setMemOpAlignments<LLVM::StoreOp>(
      funcOp, map, [](LLVM::StoreOp op) { return op.getValue().getType(); },
      "llvm.store", unresolved);
}

/// Write the deferred reinterpret_cast alignment attributes, filtering by
/// provenance.
static void applyReinterpretCastAttributes(
    const ReinterpretCastAttrMap &rcAttrs,
    const llvm::SmallPtrSetImpl<BlockArgument> &unresolved) {
  for (auto &kv : rcAttrs) {
    memref::ReinterpretCastOp castOp = kv.first;
    const ReinterpretCastAttr &info = kv.second;
    if (castOp->hasAttr("alignment"))
      continue;
    if (isTainted(info.seeds, unresolved)) {
      DBG("Skipping reinterpret_cast attribute — tainted seeds");
      continue;
    }
    OpBuilder builder(castOp);
    castOp->setAttr("alignment", builder.getIntegerAttr(builder.getI64Type(),
                                                        info.alignment));
    DBG("Wrote alignment = " << info.alignment
                             << " on reinterpret_cast (apply phase)");
  }
}

//===----------------------------------------------------------------------===//
// Module-level memref.alloc alignment hint
//===----------------------------------------------------------------------===//

static constexpr uint64_t kMemRefAllocAlignmentMinBufferSizeBytes = 1024;

static void setMemRefAllocAlignment(ModuleOp moduleOp, uint64_t alignment) {
  moduleOp.walk([&](memref::AllocOp allocOp) {
    auto memrefType = llvm::dyn_cast<MemRefType>(allocOp.getType());
    if (!memrefType || !memrefType.hasStaticShape())
      return;
    auto elementSizeOpt = getElementSizeInBytes(memrefType.getElementType());
    if (!elementSizeOpt.has_value())
      return;
    uint64_t totalBytes =
        static_cast<uint64_t>(memrefType.getNumElements()) * *elementSizeOpt;
    if (totalBytes < kMemRefAllocAlignmentMinBufferSizeBytes)
      return;
    OpBuilder builder(allocOp);
    allocOp->setAttr("alignment",
                     builder.getIntegerAttr(builder.getI64Type(), alignment));
  });
}

//===----------------------------------------------------------------------===//
// Driver
//===----------------------------------------------------------------------===//

void propagateLLVMAlignment(
    func::FuncOp funcOp, DivMap &valueDivisibility,
    llvm::SmallPtrSetImpl<BlockArgument> &unresolvedSeeds) {
  // Initialize from function arguments.
  for (auto arg : funcOp.getArguments()) {
    if (auto divisibilityAttr = funcOp.getArgAttrOfType<IntegerAttr>(
            arg.getArgNumber(), "tt.divisibility")) {
      uint64_t argDiv = divisibilityAttr.getInt();
      if (argDiv > 0) {
        ValueDivisibilityInfo info;
        info.divisibility = argDiv;
        valueDivisibility[arg] = std::move(info);
        DBG("Found argument " << arg.getArgNumber()
                              << " with divisibility = " << argDiv);
      }
    }
  }

  seedConstOpDivisibility(funcOp, valueDivisibility);
  seedMemRefAllocDivisibility(funcOp, valueDivisibility);
  seedHexagonMemAllocDivisibility(funcOp, valueDivisibility);

  // Initialize block-arg seeds.
  DominanceInfo domInfo(funcOp);
  SeedMap seeds;
  initializeBlockArgSeeds(funcOp, domInfo, seeds);

  // Side map: deferred reinterpret_cast attribute writes.
  ReinterpretCastAttrMap rcAttrs;

  // Fixed-point iteration. Continue until both `valueDivisibility` size and
  // every seed's (alignment, resolved) stop changing.
  bool changed = true;
  int iteration = 0;
  while (changed) {
    changed = false;
    size_t oldMapSize = valueDivisibility.size();
    DBG("=== Iteration " << iteration << " ===");

    bool seedChanged = propagateThroughBlockArgs(seeds, valueDivisibility);
    propagateThroughMulI(funcOp, valueDivisibility);
    propagateThroughIndexCast(funcOp, valueDivisibility);
    propagateThroughAddI(funcOp, valueDivisibility);
    propagateThroughExtractStridedMetadata(funcOp, valueDivisibility);
    propagateThroughReinterpretCast(funcOp, valueDivisibility, rcAttrs);
    propagateThroughUnrealizedConversionCast(funcOp, valueDivisibility);
    propagateThroughExtractValue(funcOp, valueDivisibility);
    propagateThroughGEP(funcOp, valueDivisibility);

    if (valueDivisibility.size() > oldMapSize || seedChanged)
      changed = true;
    iteration++;
    if (iteration > 64) {
      DBG("Bailing out of fixed point after " << iteration << " iterations");
      break;
    }
  }
  DBG("Completed after " << iteration << " iterations with "
                         << valueDivisibility.size() << " aligned values");

  // Build the unresolved-seed set used for apply-time filtering.
  unresolvedSeeds.clear();
  for (auto &kv : seeds) {
    if (!kv.second.resolved)
      unresolvedSeeds.insert(kv.first);
  }
  DBG(unresolvedSeeds.size() << " unresolved block-arg seeds at apply time");

  applyReinterpretCastAttributes(rcAttrs, unresolvedSeeds);
  setLoadOpAlignments(funcOp, valueDivisibility, unresolvedSeeds);
  setStoreOpAlignments(funcOp, valueDivisibility, unresolvedSeeds);
}

struct LLVMAlignmentPropagationPass
    : public mlir::hexagon::impl::LLVMAlignmentPropagationBase<
          LLVMAlignmentPropagationPass> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LLVMAlignmentPropagationPass)

  explicit LLVMAlignmentPropagationPass(
      const LLVMAlignmentPropagationOptions &options)
      : LLVMAlignmentPropagationBase(options) {}

  DivMap valueDivisibility;
  llvm::SmallPtrSet<BlockArgument, 4> lastUnresolvedSeeds;

  void runOnOperation() override {
    auto moduleOp = getOperation();

    if (setMemrefAllocAlignment) {
      DBG("Setting alignment=" << nativeVectorWidthInBytes
                               << " on all memref.alloc operations");
      setMemRefAllocAlignment(moduleOp, nativeVectorWidthInBytes);
    }

    DBG("===== Starting Module-Level Alignment Propagation =====");

    // Build call graph and compute nodes reachable from kernel roots.
    IntraModuleCallGraph cg = buildCallGraph(moduleOp);
    llvm::SmallPtrSet<CallNode *, 16> reachableSet;
    for (CallNode *n :
         llvm::ReversePostOrderTraversal<CallNode *>(&cg.superRoot))
      reachableSet.insert(n);

    // Skip cross-function propagation if the call graph has cycles.
    bool hasCycles = false;
    for (auto it = llvm::scc_begin(&cg.superRoot);
         it != llvm::scc_end(&cg.superRoot); ++it) {
      if (it.hasCycle()) {
        hasCycles = true;
        break;
      }
    }

    if (hasCycles) {
      DBG("Cycle detected in call graph — skipping cross-function alignment "
          "propagation; running each function independently");
      for (auto funcOp : moduleOp.getOps<func::FuncOp>()) {
        if (!shouldProcess(funcOp))
          continue;
        valueDivisibility.clear();
        lastUnresolvedSeeds.clear();
        propagateLLVMAlignment(funcOp, valueDivisibility, lastUnresolvedSeeds);
      }
      DBG("===== Module-Level Alignment Propagation Complete =====");
      return;
    }

    // RPO = callers before callees; collect GCD of caller contributions first.
    CalleeArgAccum calleeArgAccum;
    for (CallNode *node :
         llvm::ReversePostOrderTraversal<CallNode *>(&cg.superRoot)) {
      if (node == &cg.superRoot)
        continue;

      func::FuncOp funcOp = node->funcOp;

      if (!hasTTDivisibilityAttribute(funcOp)) {
        // Inject tt.divisibility from callers only when every in-module caller
        // is itself reachable; an unreachable caller could pass any alignment.
        bool allCallersReachable = true;
        for (CallNode *caller : node->callers) {
          if (caller != &cg.superRoot && !reachableSet.count(caller)) {
            allCallersReachable = false;
            break;
          }
        }
        if (allCallersReachable) {
          auto it = calleeArgAccum.find(node->name);
          if (it != calleeArgAccum.end())
            applyCalleeArgAccum(funcOp, it->second);
        } else {
          DBG("Callee " << node->name
                        << " has unprocessed callers — skipping injection");
        }
      }
      if (!shouldProcess(funcOp))
        continue;

      DBG("Processing function: " << node->name);
      valueDivisibility.clear();
      lastUnresolvedSeeds.clear();
      propagateLLVMAlignment(funcOp, valueDivisibility, lastUnresolvedSeeds);

      // Record this function's outgoing call contributions.
      recordOutgoingCallDivisibilities(funcOp, calleeArgAccum);
    }

    DBG("===== Module-Level Alignment Propagation Complete =====");
  }

private:
  IntraModuleCallGraph buildCallGraph(ModuleOp moduleOp) {
    IntraModuleCallGraph cg;
    for (auto funcOp : moduleOp.getOps<func::FuncOp>()) {
      auto node = std::make_unique<CallNode>();
      node->name = funcOp.getName();
      node->funcOp = funcOp;
      cg.nodesByName[node->name] = node.get();
      cg.nodes.push_back(std::move(node));
    }
    for (auto funcOp : moduleOp.getOps<func::FuncOp>()) {
      CallNode *callerNode = cg.nodesByName.lookup(funcOp.getName());
      if (!callerNode)
        continue;
      funcOp.walk([&](func::CallOp callOp) {
        CallNode *calleeNode = cg.nodesByName.lookup(callOp.getCallee());
        if (!calleeNode)
          return;
        if (!llvm::is_contained(callerNode->callees, calleeNode)) {
          callerNode->callees.push_back(calleeNode);
          calleeNode->callers.push_back(callerNode);
        }
      });
    }
    for (auto &nodePtr : cg.nodes) {
      if (shouldProcess(nodePtr->funcOp))
        cg.superRoot.callees.push_back(nodePtr.get());
    }
    return cg;
  }

  void recordOutgoingCallDivisibilities(func::FuncOp funcOp,
                                        CalleeArgAccum &accum) {
    funcOp.walk([&](func::CallOp callOp) {
      StringRef callee = callOp.getCallee();
      DBG("Recording outgoing call to: " << callee);
      for (unsigned i = 0; i < callOp.getNumOperands(); ++i) {
        Value operand = callOp.getOperand(i);
        auto &state = accum[callee][i];
        auto it = valueDivisibility.find(operand);
        if (it == valueDivisibility.end() ||
            isTainted(it->second.seeds, lastUnresolvedSeeds)) {
          state.unknown = true;
        } else if (!state.unknown) {
          uint64_t d = it->second.divisibility;
          state.divisibility = std::gcd(state.divisibility, d);
          DBG("  arg " << i << " divisibility " << d
                       << " (running GCD = " << state.divisibility << ")");
        }
      }
    });
  }

  void applyCalleeArgAccum(
      func::FuncOp funcOp,
      const llvm::SmallDenseMap<unsigned, CalleeArgState> &argStates) {
    OpBuilder builder(funcOp.getContext());
    for (auto &[argIdx, state] : argStates) {
      if (state.unknown)
        continue;
      assert(argIdx < funcOp.getNumArguments() &&
             "callee arg index out of bounds — call and funcOp signature "
             "mismatch");
      funcOp.setArgAttr(argIdx, "tt.divisibility",
                        builder.getIntegerAttr(
                            builder.getI32Type(),
                            std::gcd(state.divisibility, kLargeAlignment)));
      DBG("Set tt.divisibility="
          << std::gcd(state.divisibility, kLargeAlignment) << " on "
          << funcOp.getName() << " arg " << argIdx);
    }
  }

  // -----------------------------------------------------------------------
  // Predicates

  bool hasTTDivisibilityAttribute(func::FuncOp funcOp) {
    for (unsigned i = 0; i < funcOp.getNumArguments(); ++i) {
      if (funcOp.getArgAttrOfType<IntegerAttr>(i, "tt.divisibility"))
        return true;
    }
    return false;
  }

  bool shouldProcess(func::FuncOp funcOp) {
    return hasTTDivisibilityAttribute(funcOp);
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
hexagon::createLLVMAlignmentPropagationPass(
    const LLVMAlignmentPropagationOptions &options) {
  return std::make_unique<LLVMAlignmentPropagationPass>(options);
}
