//===----------------------------------------------------------------------===//
//
// Copyright 2026 The KTIR Scheduler Authors.
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
// Stage Grouping Analysis
//
// This analysis determines which stages in a pipeline can be coarsened together
// based on Loop distribution legality (data dependencies).
// Scalar dependencies are loop-fission preventing if their corresponding memory
// buffers cannot be expanded to turn them into non-scalar dependencies.
//
//===----------------------------------------------------------------------===//

#ifndef KTIR_SCHEDULER_TRANSFORM_STAGEGROUPING_HPP
#define KTIR_SCHEDULER_TRANSFORM_STAGEGROUPING_HPP

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/Operation.h"
#include "src/Dialects/KTDF/KTDFOps.hpp"

namespace ktdf {

/// Represents a group of stages that should be coarsened together
struct StageGroup {
  /// Add a stage to this group
  void addStage(mlir::ktdf::StageOp stage) { stages_.push_back(stage); }

  /// Check if this is a single-stage group
  bool isSingleStage() const { return stages_.size() == 1; }

  /// Get the first stage in the group
  mlir::ktdf::StageOp getFirstStage() const {
    return stages_.empty() ? nullptr : stages_.front();
  }

  /// Get the last stage in the group
  mlir::ktdf::StageOp getLastStage() const {
    return stages_.empty() ? nullptr : stages_.back();
  }

  /// Get all stages in the group
  const llvm::SmallVector<mlir::ktdf::StageOp>& getStages() const {
    return stages_;
  }

 private:
  /// Stages in this group (in program order)
  llvm::SmallVector<mlir::ktdf::StageOp> stages_;
};

/// Stage Grouping Analysis
///
/// Analyzes a pipeline to determine which stages can be coarsened together.
/// This is based on loop distribution legality rules and resource sharing.
class StageGroupingAnalysis {
 public:
  /// Construct analysis for a given pipeline
  explicit StageGroupingAnalysis(mlir::ktdf::PipelineOp pipeline);
  explicit StageGroupingAnalysis(llvm::ArrayRef<mlir::ktdf::StageOp> stages);

  /// Get all stage groups
  const llvm::SmallVector<StageGroup>& getGroups() const { return groups_; }

  /// Get the group that contains a given stage
  const StageGroup* getGroupForStage(mlir::ktdf::StageOp stage) const;

  /// Check if two stages are in the same group
  bool areInSameGroup(mlir::ktdf::StageOp stage1,
                      mlir::ktdf::StageOp stage2) const;

  /// Print the grouping results to the given output stream
  void print(llvm::raw_ostream& os) const;

  /// Print the grouping results for debugging (to llvm::outs())
  void dump() const;

 private:
  void init(llvm::ArrayRef<mlir::ktdf::StageOp> stages);

  /// Collect all stages from the pipeline
  void collectStages(llvm::SmallVector<mlir::ktdf::StageOp>& stages,
                     mlir::ktdf::PipelineOp pipeline);

  /// Determine if two stages should be grouped together
  /// Returns true if stages share SFP registers or FIFO types
  bool shouldGroupTogether(mlir::ktdf::StageOp stage1,
                           mlir::ktdf::StageOp stage2);

  llvm::SmallVector<StageGroup> groups_;
  llvm::DenseMap<mlir::Operation*, unsigned> stage_to_group_;
};

}  // namespace ktdf

#endif  // KTIR_SCHEDULER_TRANSFORM_STAGEGROUPING_HPP
