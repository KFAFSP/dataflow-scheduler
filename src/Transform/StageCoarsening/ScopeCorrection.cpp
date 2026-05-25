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

#include "ScopeCorrection.hpp"

#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/Debug.h"
#include "mlir/IR/IRMapping.h"
#include "src/Utils/OperationUtils.hpp"

#define DEBUG_TYPE "stage-coarsening"

namespace ktdf {

void ScopeCorrection::run() {
  outer_private_ = findOuterPrivate();
  if (!outer_private_) {
    LLVM_DEBUG(llvm::dbgs() << "  No outer private node found, skipping.\n");
    return;
  }

  auto inner_pipelines = findInnerPipelines();
  if (inner_pipelines.empty()) {
    LLVM_DEBUG(llvm::dbgs() << "  No inner pipelines found, skipping.\n");
    return;
  }

  analyzePrivateResultUsage();
  if (pipeline_to_sc_info_.empty()) {
    LLVM_DEBUG(llvm::dbgs() << "  No results need to be moved.\n");
    return;
  }

  // Move results to their respective inner pipelines
  for (auto& [inner_pipeline, sc_info] : pipeline_to_sc_info_) {
    movePrivateResultsToInnerPipeline(inner_pipeline, sc_info);
  }
}

mlir::ktdf::PrivateOp ScopeCorrection::findOuterPrivate() {
  for (auto priv_op : outer_pipeline_.getOps<mlir::ktdf::PrivateOp>()) {
    if (priv_op->getParentOp() == outer_pipeline_.getOperation()) {
      return priv_op;
    }
  }
  return nullptr;
}

llvm::SmallVector<mlir::ktdf::PipelineOp>
ScopeCorrection::findInnerPipelines() {
  llvm::SmallVector<mlir::ktdf::PipelineOp> inner_pipelines;
  outer_pipeline_.walk([&](mlir::ktdf::PipelineOp inner_pipeline) {
    if (inner_pipeline != outer_pipeline_) {
      inner_pipelines.push_back(inner_pipeline);
    }
    return mlir::WalkResult::advance();
  });
  return inner_pipelines;
}

void ScopeCorrection::analyzePrivateResultUsage() {
  auto yield_op = mlir::cast<mlir::ktdf::PrivateYieldOp>(
      outer_private_.getBody().front().getTerminator());

  mlir::SmallPtrSet<mlir::Operation*, 16> visited_moved_ops;
  for (auto [idx, result] : llvm::enumerate(outer_private_.getResults())) {
    // Skip tokens as they are handled by the pipeline tree materialization
    if (mlir::isa<mlir::ktdf::TokenType>(result.getType())) continue;

    // Check all uses of this result
    bool only_used_in_inner = true;
    mlir::ktdf::PipelineOp target_inner_pipeline = nullptr;

    for (mlir::OpOperand& use : result.getUses()) {
      // Find which pipeline (if any) contains this use
      mlir::Operation* user = use.getOwner();
      mlir::ktdf::PipelineOp containing_pipeline =
          user->getParentOfType<mlir::ktdf::PipelineOp>();
      assert(containing_pipeline &&
             "found a private result with uses outside of a pipeline");
      // If use is not in an inner pipeline, we can't move this result
      if (containing_pipeline == outer_pipeline_) {
        only_used_in_inner = false;
        break;
      }

      // Check if all uses are in the same inner pipeline
      if (!target_inner_pipeline) {
        target_inner_pipeline = containing_pipeline;
      } else if (target_inner_pipeline != containing_pipeline) {
        // Used in multiple inner pipelines, can't move
        only_used_in_inner = false;
        break;
      }
    }

    if (!only_used_in_inner || !target_inner_pipeline) continue;
    LLVM_DEBUG(llvm::dbgs()
               << "  Result #" << idx << " only used in inner pipeline\n");

    auto& sc_info = pipeline_to_sc_info_[target_inner_pipeline];
    sc_info.target_pipeline = target_inner_pipeline;
    sc_info.result_indices.push_back(idx);
    sc_info.old_results.push_back(result);

    // Find the operation that produces this yielded value.
    // Avoid adding duplicate ops when there are ops with multiple results being
    // moved (eg. fifo.allocate).
    mlir::Value yielded_val = yield_op.getOperand(idx);
    mlir::Operation* def_op = yielded_val.getDefiningOp();
    if (def_op && visited_moved_ops.insert(def_op).second) {
      sc_info.ops_to_move.push_back(def_op);
    }
  }
}

void ScopeCorrection::movePrivateResultsToInnerPipeline(
    mlir::ktdf::PipelineOp inner_pipeline, ScopeCorrectionInfo& sc_info) {
  LLVM_DEBUG(llvm::dbgs() << "  Moving " << sc_info.result_indices.size()
                          << " result(s) to inner pipeline\n");

  // Find or create the inner pipeline's private operation
  mlir::ktdf::PrivateOp inner_private = nullptr;
  for (mlir::Operation& op : inner_pipeline.getBody().front()) {
    if (auto priv = mlir::dyn_cast<mlir::ktdf::PrivateOp>(&op)) {
      inner_private = priv;
      break;
    }
  }

  if (inner_private) {
    // Inner pipeline already has a private node - extend it
    extendInnerPrivate(inner_private, sc_info);
  } else {
    // Create a new private node for the inner pipeline
    createInnerPrivate(inner_pipeline, sc_info);
  }

  // Update the outer private to remove the moved results
  updateOuterPrivate(sc_info.result_indices);
}

mlir::ktdf::PrivateOp ScopeCorrection::recreatePrivateOp(
    mlir::ktdf::PrivateOp old_private,
    llvm::ArrayRef<mlir::Type> new_result_types,
    llvm::ArrayRef<mlir::Operation*> ops_to_prepend,
    llvm::ArrayRef<mlir::Operation*> ops_to_exclude) {
  // Create new private op with new result types
  builder_.setInsertionPoint(old_private);
  auto new_private = mlir::ktdf::PrivateOp::create(
      builder_, old_private.getLoc(), new_result_types);

  // Build the body
  mlir::Block& new_body = new_private.getBody().front();
  mlir::IRMapping value_map;

  // Clone operations to prepend (e.g., moved operations)
  builder_.setInsertionPointToStart(&new_body);
  for (mlir::Operation* op : ops_to_prepend) {
    builder_.clone(*op, value_map);
  }

  // Build exclusion set for O(1) lookup
  llvm::DenseSet<mlir::Operation*> exclude_set(ops_to_exclude.begin(),
                                               ops_to_exclude.end());

  // Handle existing body operations (if old_private is non-null)
  if (old_private) {
    mlir::Block& old_body = old_private.getBody().front();
    // Clone operations from old body (excluding terminator and excluded ops)
    for (mlir::Operation& op : old_body.without_terminator()) {
      if (!exclude_set.contains(&op)) {
        builder_.clone(op, value_map);
      } else {
        LLVM_DEBUG(llvm::dbgs()
                   << "      Skipping excluded operation: " << op << "\n");
      }
    }
  }

  // Build yield operands: prepended ops' results + old yield operands (if
  // old_private exists)
  llvm::SmallVector<mlir::Value> new_yield_operands;

  // Add results from prepended operations
  for (mlir::Operation* op : ops_to_prepend) {
    for (size_t i = 0; i < op->getNumResults(); i++) {
      new_yield_operands.push_back(value_map.lookup(op->getResult(i)));
    }
  }

  // Add mapped operands from old yield (if old_private exists)
  if (old_private) {
    mlir::Block& old_body = old_private.getBody().front();
    auto old_yield =
        mlir::cast<mlir::ktdf::PrivateYieldOp>(old_body.getTerminator());

    for (auto [idx, operand] : llvm::enumerate(old_yield.getOperands())) {
      // Skip operands produced by excluded operations
      if (mlir::Operation* def_op = operand.getDefiningOp()) {
        if (exclude_set.contains(def_op)) {
          continue;
        }
      }
      new_yield_operands.push_back(value_map.lookupOrDefault(operand));
    }
  }

  // Create new yield
  builder_.setInsertionPointToEnd(&new_body);
  mlir::ktdf::PrivateYieldOp::create(builder_, old_private.getLoc(),
                                     new_yield_operands);

  return new_private;
}

void ScopeCorrection::extendInnerPrivate(mlir::ktdf::PrivateOp inner_private,
                                         ScopeCorrectionInfo& sc_info) {
  LLVM_DEBUG(llvm::dbgs() << "    Extending existing inner private\n");

  // Get the old yield to preserve existing operands
  auto old_yield = mlir::cast<mlir::ktdf::PrivateYieldOp>(
      inner_private.getBody().front().getTerminator());

  // Build new result types: moved results first, then existing results
  llvm::SmallVector<mlir::Type> new_result_types =
      sc_info.getMovingResultTypes();
  for (mlir::Value val : old_yield.getOperands()) {
    new_result_types.push_back(val.getType());
  }

  // Recreate the private op with moved operations prepended
  auto new_inner_private =
      recreatePrivateOp(inner_private, new_result_types,
                        sc_info.ops_to_move,  // ops to prepend
                        {});                  // no ops to exclude

  // Map moved results to new results (at the beginning)
  for (unsigned i = 0; i < sc_info.old_results.size(); ++i) {
    sc_info.old_results[i].replaceAllUsesWith(new_inner_private.getResult(i));
  }

  // Replace uses of old existing results with new results
  const unsigned moved_count = sc_info.old_results.size();
  for (unsigned i = 0; i < inner_private.getNumResults(); ++i) {
    inner_private.getResult(i).replaceAllUsesWith(
        new_inner_private.getResult(moved_count + i));
  }

  inner_private.erase();
}

void ScopeCorrection::createInnerPrivate(mlir::ktdf::PipelineOp inner_pipeline,
                                         ScopeCorrectionInfo& sc_info) {
  LLVM_DEBUG(llvm::dbgs() << "    Creating new inner private\n");

  // Determine result types from operations to move
  llvm::SmallVector<mlir::Type> result_types = sc_info.getMovingResultTypes();

  // Use helper to create the private op with moved operations (no old_private
  // to preserve)
  builder_.setInsertionPointToStart(&inner_pipeline.getBody().front());
  auto new_private = recreatePrivateOp(nullptr, result_types,
                                       sc_info.ops_to_move,  // ops to prepend
                                       {});  // no ops to exclude

  // Replace all uses of old results with new results
  for (unsigned i = 0; i < sc_info.old_results.size(); ++i) {
    sc_info.old_results[i].replaceAllUsesWith(new_private.getResult(i));
  }
}

void ScopeCorrection::updateOuterPrivate(
    llvm::ArrayRef<unsigned> removed_indices) {
  LLVM_DEBUG(llvm::dbgs() << "    Updating outer private\n");

  // Build set of indices to remove for O(1) lookup
  llvm::DenseSet<unsigned> to_remove(removed_indices.begin(),
                                     removed_indices.end());

  // Get the yield operation
  auto old_yield = mlir::cast<mlir::ktdf::PrivateYieldOp>(
      outer_private_.getBody().front().getTerminator());

  // Identify operations that define removed results (to be excluded from new
  // body)
  llvm::SmallVector<mlir::Operation*> ops_moved;
  for (unsigned idx : removed_indices) {
    mlir::Value yielded_val = old_yield.getOperand(idx);
    if (mlir::Operation* def_op = yielded_val.getDefiningOp()) {
      ops_moved.push_back(def_op);
    }
  }

  // Build new yield operands and result types (excluding removed ones)
  llvm::SmallVector<mlir::Type> new_result_types;

  for (auto [idx, operand] : llvm::enumerate(old_yield.getOperands())) {
    if (!to_remove.contains(idx)) {
      new_result_types.push_back(operand.getType());
    }
  }

  // Use helper to recreate the private op, excluding removed operations
  auto new_outer_private = recreatePrivateOp(outer_private_, new_result_types,
                                             {},          // no ops to prepend
                                             ops_moved);  // ops to exclude

  // Map old results to new results (skipping removed ones)
  unsigned new_idx = 0;
  for (unsigned old_idx = 0; old_idx < outer_private_.getNumResults();
       ++old_idx) {
    if (!to_remove.contains(old_idx)) {
      outer_private_.getResult(old_idx).replaceAllUsesWith(
          new_outer_private.getResult(new_idx));
      new_idx++;
    }
  }

  outer_private_.erase();
}

}  // namespace ktdf

// Made with Bob
