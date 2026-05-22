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
// Scope Correction for Private Results
//
// This helper class analyzes and fixes the scoping of SSA values from an outer
// pipeline's ktdf.private that are only used in inner pipeline stages. Such
// values are moved to the inner pipeline's private node.
//===----------------------------------------------------------------------===//

#ifndef KTIR_SCHEDULER_TRANSFORM_SCOPECORRECTION_HPP
#define KTIR_SCHEDULER_TRANSFORM_SCOPECORRECTION_HPP

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/Builders.h"
#include "src/Dialects/KTDF/KTDFOps.hpp"

namespace ktdf {

/// Helper class to encapsulate private result scope correction logic
class ScopeCorrection {
 public:
  ScopeCorrection(mlir::OpBuilder& builder, mlir::Operation* outer_pipeline)
      : builder_(builder),
        outer_pipeline_(mlir::cast<mlir::ktdf::PipelineOp>(outer_pipeline)) {}

  /// Perform the scope correction analysis and transformation
  void run();

 private:
  /// Information about results to move to an inner pipeline
  struct ScopeCorrectionInfo {
    mlir::ktdf::PipelineOp target_pipeline;
    llvm::SmallVector<unsigned> result_indices;
    llvm::SmallVector<mlir::Operation*> ops_to_move;
    llvm::SmallVector<mlir::Value> old_results;

    /// Build result types from operations to move
    llvm::SmallVector<mlir::Type> getMovingResultTypes() const {
      llvm::SmallVector<mlir::Type> result_types;
      for (mlir::Operation* op : ops_to_move) {
        for (int i = 0; i < op->getNumResults(); i++) {
          result_types.push_back(op->getResult(i).getType());
        }
      }
      return result_types;
    }
  };

  /// Find the outer pipeline's private operation
  mlir::ktdf::PrivateOp findOuterPrivate();

  /// Find all inner pipelines
  llvm::SmallVector<mlir::ktdf::PipelineOp> findInnerPipelines();

  /// Analyze which private results are only used in specific inner pipelines,
  /// and updates an internal mapping.
  void analyzePrivateResultUsage();

  /// Move specified private results from outer private to inner pipeline
  void movePrivateResultsToInnerPipeline(mlir::ktdf::PipelineOp inner_pipeline,
                                         ScopeCorrectionInfo& sc_info);

  /// Helper to recreate a PrivateOp with new or different results
  ///
  /// @param old_private The original private op to recreate (if null, creates a
  /// new empty private)
  /// @param new_result_types The result types for the new private op
  /// @param ops_to_prepend Operations to clone at the beginning of the body
  /// @param ops_to_exclude Operations from old body to skip (only used if
  /// old_private is non-null)
  mlir::ktdf::PrivateOp recreatePrivateOp(
      mlir::ktdf::PrivateOp old_private,
      llvm::ArrayRef<mlir::Type> new_result_types,
      llvm::ArrayRef<mlir::Operation*> ops_to_prepend = {},
      llvm::ArrayRef<mlir::Operation*> ops_to_exclude = {});

  /// Extend an existing inner private node with moved allocations
  void extendInnerPrivate(mlir::ktdf::PrivateOp inner_private,
                          ScopeCorrectionInfo& sc_info);

  /// Create a new inner private node with moved allocations
  void createInnerPrivate(mlir::ktdf::PipelineOp inner_pipeline,
                          ScopeCorrectionInfo& sc_info);

  /// Update the outer private to remove moved results
  void updateOuterPrivate(llvm::ArrayRef<unsigned> removed_indices);

  mlir::OpBuilder& builder_;
  mlir::ktdf::PipelineOp outer_pipeline_;
  mlir::ktdf::PrivateOp outer_private_;
  llvm::DenseMap<mlir::ktdf::PipelineOp, ScopeCorrectionInfo>
      pipeline_to_sc_info_;
};

}  // namespace ktdf

#endif  // KTIR_SCHEDULER_TRANSFORM_SCOPECORRECTION_HPP

// Made with Bob
