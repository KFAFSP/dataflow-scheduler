//===-- LoopTiling.h --------------------------------------------*- c++ -*-===//
//
// Part of the Dataflow Scheduler project.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//
//
// This analysis evaluates loop nests in a program for tiling opportunities.
// It identifies perfectly nested loops and assigns metadata (divisibility and
// minimum value) to each loop in the nest.
//
// The analysis provides two interfaces:
// 1. LoopTilingInfo: A utility class that can be instantiated directly
// 2. LoopTilingAnalysis: An MLIR analysis wrapper for pass manager integration
//
// Example usage as a utility:
//   mlir::ModuleOp module = ...;
//   LoopTilingInfo tiling_info(module);
//   if (auto* metadata = tiling_info.getMetadataForLoop(loop_op)) {
//     int64_t divis = metadata->divisibility;
//   }
//
// Example usage with MLIR pass manager:
//   void runOnOperation() override {
//     auto& analysis = getAnalysis<LoopTilingAnalysis>();
//     if (auto* metadata = analysis.getMetadataForLoop(loop_op)) {
//       // Use metadata for tiling decisions
//     }
//   }
//
// IMPORTANT: A typical usage scenario is as follows:
// 1. Use getAnalysis<LoopTilingAnalysis>() to share the analysis among multiple
// back-to-back passes
// 2. Mark loops as invalidated after transformation via invalidateLoop()
// 3. Skip invalidated entries when checking for candidates
// 4. Preserve the analysis to allow multiple back-to-back passes to run without
// recomputation This allows tiling and strip-mining passes to run in any order
// efficiently.
//
// Note: after tiling, new inner loops are introduced. To consider such
// loops for further tiling, the analysis should be invalidated in its entirety
// and recomputed.
//
//===----------------------------------------------------------------------===//

#ifndef DATAFLOW_SCHEDULER_DIALECT_KTDF_ANALYSIS_LOOPTILINGANALYSIS_H_
#define DATAFLOW_SCHEDULER_DIALECT_KTDF_ANALYSIS_LOOPTILINGANALYSIS_H_

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/SmallVector.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/Operation.h>
#include <mlir/Pass/AnalysisManager.h>

namespace mlir::ktdf {

// Forward declaration
class LoopTilingAnalysis;

/// Metadata associated with a loop for tiling analysis.
/// Contains information about divisibility constraints and minimum iteration
/// counts that can guide tiling decisions.
struct LoopTilingMetadata {
  /// Optimization strategy for this loop.
  enum class Strategy {
    kNone,    // Not a candidate for optimization
    kTiling,  // Candidate for multi-dimensional tiling (part of parallel loop
              // nest depth > 1)
    kStripMining  // Candidate for strip mining (outermost loop, nest depth ==
                  // 1)
  };

  /// Divisibility factor for tiling. Indicates that the loop bound is
  /// divisible by this value, which can be used as a tile size.
  /// Default: 1 (no specific divisibility constraint)
  int64_t divisibility = 1;

  /// Minimum value for the loop iteration count. Useful for determining
  /// whether tiling is beneficial.
  /// Default: 1 (at least one iteration)
  int64_t min_value = 1;

  /// Optimization strategy for this loop (tiling, strip mining, or none).
  Strategy strategy = Strategy::kNone;

  /// Flag indicating whether this metadata has been invalidated.
  /// When true, the metadata may be stale and should be recomputed.
  bool is_invalidated = false;

  /// Returns true if this loop is a candidate for tiling.
  bool isTilingCandidate() const { return strategy == Strategy::kTiling; }

  /// Returns true if this loop is a candidate for strip mining.
  bool isStripMiningCandidate() const {
    return strategy == Strategy::kStripMining;
  }

  // TODO: Implement smarter heuristics for divisibility and min_value:

  void dump() const;
};

/// Information about a perfectly nested loop group.
/// Contains the loops in the nest (from outermost to innermost) and
/// metadata for each loop.
struct PerfectNestingInfo {
  /// Loops in this nest, ordered from outermost to innermost.
  /// Limited to a maximum depth (default: 4 loops).
  SmallVector<scf::ForOp, 4> loops;

  /// Returns the depth of this loop nest (number of loops).
  size_t depth() const { return loops.size(); }

  /// Returns the outermost loop in the nest.
  scf::ForOp getOutermost() const { return loops.front(); }

  /// Returns the innermost loop in the nest.
  scf::ForOp getInnermost() const { return loops.back(); }

  void dump() const;
};

/// Core utility class for loop tiling analysis.
/// This class can be instantiated directly in transformations without
/// requiring the MLIR pass manager infrastructure.
///
/// The analysis identifies perfectly nested loops and assigns metadata
/// to each loop that can guide tiling decisions. Perfectly nested loops
/// are defined as loops where each loop body contains only the next inner
/// loop (and its terminator), with no intervening operations.
///
/// Example:
///   scf.for %i = 0 to 100 step 1 {
///     scf.for %j = 0 to 100 step 1 {
///       scf.for %k = 0 to 100 step 1 {
///         // body
///       }
///     }
///   }
/// This forms a single perfectly nested loop group of depth 3.
///
/// Non-example (intervening code breaks perfect nesting):
///   scf.for %i = 0 to 100 step 1 {
///     %x = arith.addi %i, %c1
///     scf.for %j = 0 to 100 step 1 {
///       // body
///     }
///   }
/// This forms two separate groups: {%i} and {%j}.
class LoopTilingInfo {
 public:
  /// Maximum depth of loop nests to analyze.
  /// If a perfectly nested loop group is deeper than this, only the first (ie
  /// outermost) kMaxLoopNestDepth loops are included in the analysis.
  static constexpr size_t kMaxLoopNestDepth = 4;

  /// Constructs the analysis for the given root operation (typically a module).
  /// Analyzes all scf.for loops within the operation and identifies perfectly
  /// nested loop groups.
  explicit LoopTilingInfo(Operation* root_op);

  /// Returns metadata for the given loop operation, or nullptr if the loop
  /// is not part of any analyzed nest.
  const LoopTilingMetadata* getMetadataForLoop(Operation* loop) const;

  /// Returns the nest information containing the given loop, or nullptr if
  /// the loop is not part of any analyzed nest.
  const PerfectNestingInfo* getNestContainingLoop(Operation* loop) const;

  /// Returns all identified loop nests.
  const SmallVector<PerfectNestingInfo>& getAllNests() const {
    return loop_nests_;
  }

  /// Marks the metadata for the given loop as invalidated.
  /// This is useful when a transformation modifies a loop and the metadata
  /// needs to be recomputed.
  void invalidateLoop(Operation* loop);

  /// Updates the metadata for the given loop.
  /// If the loop is not currently tracked, this method has no effect.
  void updateMetadata(Operation* loop, const LoopTilingMetadata& metadata);

  /// Dumps the analysis results to stderr for debugging.
  void dump() const;

 private:
  /// Computes all perfectly nested loop groups in the given operation.
  void computeLoopNests(Operation* root_op);

  /// Assigns initial metadata to all loops in the given nest.
  /// Currently assigns divisibility=1 and min_value=1 to all loops.
  void assignMetadata(PerfectNestingInfo& nest);

  /// All identified loop nests.
  SmallVector<PerfectNestingInfo> loop_nests_;

  /// Fast lookup from loop operation to its metadata.
  DenseMap<Operation*, LoopTilingMetadata> loop_to_metadata_;

  /// Fast lookup from loop operation to the index of its containing nest.
  DenseMap<Operation*, size_t> loop_to_nest_index_;
};

/// MLIR analysis wrapper for LoopTilingInfo.
/// This class adheres to MLIR's analysis interface and can be used with
/// the pass manager infrastructure for automatic caching and invalidation.
///
/// Example usage in a pass:
///   struct MyPass : public PassWrapper<MyPass, OperationPass<ModuleOp>> {
///     void runOnOperation() override {
///       auto& analysis = getAnalysis<LoopTilingAnalysis>();
///       getOperation()->walk([&](scf::ForOp loop) {
///         if (auto* metadata = analysis.getMetadataForLoop(loop)) {
///           // Use metadata for optimization decisions
///         }
///       });
///     }
///   };
class LoopTilingAnalysis {
 public:
  /// Constructs the analysis for the given operation.
  /// The AnalysisManager parameter is required by MLIR's analysis interface
  /// but is not currently used by this analysis (no dependencies).
  LoopTilingAnalysis(Operation* op, AnalysisManager& am) : info_(op) {}

  /// Checks if this analysis has been invalidated.
  /// Returns true if the analysis needs to be recomputed.
  bool isInvalidated(const AnalysisManager::PreservedAnalyses& pa) const {
    return !pa.isPreserved<LoopTilingAnalysis>();
  }

  /// Returns metadata for the given loop operation.
  /// Delegates to the underlying LoopTilingInfo instance.
  const LoopTilingMetadata* getMetadataForLoop(Operation* loop) const {
    return info_.getMetadataForLoop(loop);
  }

  /// Returns the nest information containing the given loop.
  /// Delegates to the underlying LoopTilingInfo instance.
  const PerfectNestingInfo* getNestContainingLoop(Operation* loop) const {
    return info_.getNestContainingLoop(loop);
  }

  /// Returns all identified loop nests.
  /// Delegates to the underlying LoopTilingInfo instance.
  const SmallVector<PerfectNestingInfo>& getAllNests() const {
    return info_.getAllNests();
  }

  /// Marks the metadata for the given loop as invalidated.
  void invalidateLoop(Operation* loop) { info_.invalidateLoop(loop); }

  /// Updates the metadata for the given loop.
  void updateMetadata(Operation* loop, const LoopTilingMetadata& metadata) {
    info_.updateMetadata(loop, metadata);
  }

  /// Dumps the analysis results for debugging.
  void dump() const { info_.dump(); }

 private:
  LoopTilingInfo info_;
};

}  // namespace mlir::ktdf

#endif  // DATAFLOW_SCHEDULER_DIALECT_KTDF_ANALYSIS_LOOPTILINGANALYSIS_H_

// Made with Bob
