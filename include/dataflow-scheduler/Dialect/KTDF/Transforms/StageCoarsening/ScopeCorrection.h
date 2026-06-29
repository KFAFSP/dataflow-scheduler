//===-- ScopeCorrection.h ---------------------------------------*- c++ -*-===//
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
// Scope Correction for Private Results
//
// This helper class analyzes and fixes the scoping of SSA values from an outer
// pipeline's ktdf.private that are only used in inner pipeline stages. Such
// values are moved to the inner pipeline's private node.
//
//===----------------------------------------------------------------------===//

#ifndef DATAFLOW_SCHEDULER_DIALECT_KTDF_TRANSFORMS_STAGECOARSENING_SCOPECORRECTION_H_
#define DATAFLOW_SCHEDULER_DIALECT_KTDF_TRANSFORMS_STAGECOARSENING_SCOPECORRECTION_H_

#include "dataflow-scheduler/Dialect/KTDF/KTDF.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/Builders.h"

namespace mlir::ktdf {

/// Helper class to encapsulate private result scope correction logic
class ScopeCorrection {
 public:
  ScopeCorrection(OpBuilder& builder, Operation* outer_pipeline)
      : builder_(builder),
        outer_pipeline_(cast<ktdf::PipelineOp>(outer_pipeline)) {}

  /// Perform the scope correction analysis and transformation
  void run();

 private:
  /// Information about results to move to an inner pipeline
  struct ScopeCorrectionInfo {
    ktdf::PipelineOp target_pipeline;
    SmallVector<unsigned> result_indices;
    SmallVector<Operation*> ops_to_move;
    SmallVector<Value> old_results;

    /// Build result types from operations to move
    SmallVector<Type> getMovingResultTypes() const {
      SmallVector<Type> result_types;
      for (Operation* op : ops_to_move) {
        for (size_t i = 0; i < op->getNumResults(); i++) {
          result_types.push_back(op->getResult(i).getType());
        }
      }
      return result_types;
    }
  };

  /// Find all inner pipelines
  SmallVector<ktdf::PipelineOp> findInnerPipelines();

  /// Analyze which private results are only used in specific inner pipelines,
  /// and updates an internal mapping.
  void analyzePrivateResultUsage();

  /// Move specified private results from outer private to inner pipeline
  void movePrivateResultsToInnerPipeline(ktdf::PipelineOp inner_pipeline,
                                         ScopeCorrectionInfo& sc_info);

  /// Helper to recreate a PrivateOp with new or different results
  ///
  /// @param old_private The original private op to recreate (if null, creates a
  /// new empty private)
  /// @param new_result_types The result types for the new private op
  /// @param ops_to_prepend Operations to clone at the beginning of the body
  /// @param ops_to_exclude Operations from old body to skip (only used if
  /// old_private is non-null)
  ktdf::PrivateOp recreatePrivateOp(ktdf::PrivateOp old_private,
                                    ArrayRef<Type> new_result_types,
                                    ArrayRef<Operation*> ops_to_prepend = {},
                                    ArrayRef<Operation*> ops_to_exclude = {});

  /// Extend an existing inner private node with moved allocations
  void extendInnerPrivate(ktdf::PrivateOp inner_private,
                          ScopeCorrectionInfo& sc_info);

  /// Create a new inner private node with moved allocations
  void createInnerPrivate(ktdf::PipelineOp inner_pipeline,
                          ScopeCorrectionInfo& sc_info);

  /// Update the outer private to remove moved results
  void updateOuterPrivate(ArrayRef<unsigned> removed_indices);

  OpBuilder& builder_;
  ktdf::PipelineOp outer_pipeline_;
  ktdf::PrivateOp outer_private_;
  DenseMap<ktdf::PipelineOp, ScopeCorrectionInfo> pipeline_to_sc_info_;
};

}  // namespace mlir::ktdf

#endif  // DATAFLOW_SCHEDULER_DIALECT_KTDF_TRANSFORMS_STAGECOARSENING_SCOPECORRECTION_H_

// Made with Bob
