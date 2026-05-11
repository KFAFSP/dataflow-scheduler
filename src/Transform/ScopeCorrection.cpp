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

#include "src/Transform/ScopeCorrection.hpp"

#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/Debug.h"
#include "mlir/IR/IRMapping.h"

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
  if (pipeline_to_move_info_.empty()) {
    LLVM_DEBUG(llvm::dbgs() << "  No results need to be moved.\n");
    return;
  }

  // Move results to their respective inner pipelines
  for (auto& [inner_pipeline, move_info] : pipeline_to_move_info_) {
    movePrivateResultsToInnerPipeline(inner_pipeline, move_info);
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

  for (auto [idx, result] : llvm::enumerate(outer_private_.getResults())) {
    // Skip tokens - they're for stage dependencies
    if (mlir::isa<mlir::ktdf::TokenType>(result.getType())) {
      continue;
    }

    // Check all uses of this result
    bool only_used_in_inner = true;
    mlir::ktdf::PipelineOp target_inner_pipeline = nullptr;

    for (mlir::OpOperand& use : result.getUses()) {
      mlir::Operation* user = use.getOwner();

      // Find which pipeline (if any) contains this use
      mlir::ktdf::PipelineOp containing_pipeline =
          user->getParentOfType<mlir::ktdf::PipelineOp>();

      // If use is not in an inner pipeline, we can't move this result
      if (!containing_pipeline || containing_pipeline == outer_pipeline_) {
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

    if (only_used_in_inner && target_inner_pipeline) {
      LLVM_DEBUG(llvm::dbgs()
                 << "  Result #" << idx << " only used in inner pipeline\n");

      auto& move_info = pipeline_to_move_info_[target_inner_pipeline];
      move_info.target_pipeline = target_inner_pipeline;
      move_info.result_indices.push_back(idx);
      move_info.old_results.push_back(result);

      // Find the operation that produces this yielded value
      mlir::Value yielded_val = yield_op.getOperand(idx);
      if (mlir::Operation* def_op = yielded_val.getDefiningOp()) {
        move_info.ops_to_move.push_back(def_op);
      }
    }
  }
}

void ScopeCorrection::movePrivateResultsToInnerPipeline(
    mlir::ktdf::PipelineOp inner_pipeline, ScopeCorrectionInfo& move_info) {
  LLVM_DEBUG(llvm::dbgs() << "  Moving " << move_info.result_indices.size()
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
    extendInnerPrivate(inner_private, move_info);
  } else {
    // Create a new private node for the inner pipeline
    createInnerPrivate(inner_pipeline, move_info);
  }

  // Update the outer private to remove the moved results
  updateOuterPrivate(move_info.result_indices);
}

mlir::ktdf::PrivateOp ScopeCorrection::recreatePrivateOp(
    mlir::ktdf::PrivateOp old_private,
    llvm::ArrayRef<mlir::Type> new_result_types,
    llvm::function_ref<void(mlir::OpBuilder&, mlir::IRMapping&, mlir::Block&)>
        body_builder,
    llvm::function_ref<llvm::SmallVector<mlir::Value>(mlir::IRMapping&)>
        yield_builder) {
  // Create new private op with new result types
  builder_.setInsertionPoint(old_private);
  auto new_private = mlir::ktdf::PrivateOp::create(
      builder_, old_private.getLoc(), new_result_types);

  // Build the body
  mlir::Block& new_body = new_private.getBody().front();
  mlir::Block& old_body = old_private.getBody().front();
  mlir::IRMapping value_map;

  // Let the caller populate the body
  body_builder(builder_, value_map, new_body);

  // Create new yield with mapped operands
  builder_.setInsertionPointToEnd(&new_body);
  llvm::SmallVector<mlir::Value> new_yield_operands = yield_builder(value_map);
  mlir::ktdf::PrivateYieldOp::create(builder_, old_private.getLoc(),
                                     new_yield_operands);

  return new_private;
}

void ScopeCorrection::extendInnerPrivate(mlir::ktdf::PrivateOp inner_private,
                                         ScopeCorrectionInfo& move_info) {
  LLVM_DEBUG(llvm::dbgs() << "    Extending existing inner private\n");

  // Get the old yield to preserve existing operands
  auto old_yield = mlir::cast<mlir::ktdf::PrivateYieldOp>(
      inner_private.getBody().front().getTerminator());
  llvm::SmallVector<mlir::Value> existing_yield_operands(
      old_yield.getOperands().begin(), old_yield.getOperands().end());

  // Build new result types: moved results first, then existing results
  llvm::SmallVector<mlir::Type> new_result_types;
  for (mlir::Operation* op : move_info.ops_to_move) {
    new_result_types.push_back(op->getResult(0).getType());
  }
  for (mlir::Value val : existing_yield_operands) {
    new_result_types.push_back(val.getType());
  }

  // Use helper to recreate the private op
  auto new_inner_private = recreatePrivateOp(
      inner_private, new_result_types,
      // Body builder: clone moved ops, then move existing ops
      [&](mlir::OpBuilder& b, mlir::IRMapping& value_map,
          mlir::Block& new_body) {
        // Clone moved operations at the beginning
        b.setInsertionPointToStart(&new_body);
        for (mlir::Operation* op : move_info.ops_to_move) {
          b.clone(*op, value_map);
        }
        // Move existing operations (excluding terminator)
        mlir::Block& old_body = inner_private.getBody().front();
        new_body.getOperations().splice(
            new_body.end(), old_body.getOperations(), old_body.begin(),
            std::prev(old_body.end()));
      },
      // Yield builder: moved results first, then existing results
      [&](mlir::IRMapping& value_map) {
        llvm::SmallVector<mlir::Value> yield_operands;
        for (mlir::Operation* op : move_info.ops_to_move) {
          yield_operands.push_back(value_map.lookup(op->getResult(0)));
        }
        for (mlir::Value val : existing_yield_operands) {
          yield_operands.push_back(value_map.lookupOrDefault(val));
        }
        return yield_operands;
      });

  // Map moved results to new results (at the beginning)
  for (unsigned i = 0; i < move_info.old_results.size(); ++i) {
    move_info.old_results[i].replaceAllUsesWith(new_inner_private.getResult(i));
  }

  // Replace uses of old existing results with new results (shifted by moved
  // count)
  unsigned moved_count = move_info.old_results.size();
  for (unsigned i = 0; i < inner_private.getNumResults(); ++i) {
    inner_private.getResult(i).replaceAllUsesWith(
        new_inner_private.getResult(moved_count + i));
  }

  inner_private.erase();
}

void ScopeCorrection::createInnerPrivate(mlir::ktdf::PipelineOp inner_pipeline,
                                         ScopeCorrectionInfo& move_info) {
  LLVM_DEBUG(llvm::dbgs() << "    Creating new inner private\n");

  // Determine result types from operations to move
  llvm::SmallVector<mlir::Type> result_types;
  for (mlir::Operation* op : move_info.ops_to_move) {
    result_types.push_back(op->getResult(0).getType());
  }

  // Create a temporary private op to use with the helper
  builder_.setInsertionPointToStart(&inner_pipeline.getBody().front());
  auto temp_private =
      mlir::ktdf::PrivateOp::create(builder_, inner_pipeline.getLoc(), {});

  // Use helper to create the private op with proper body
  auto new_private = recreatePrivateOp(
      temp_private, result_types,
      // Body builder: clone moved operations
      [&](mlir::OpBuilder& b, mlir::IRMapping& value_map,
          mlir::Block& new_body) {
        b.setInsertionPointToStart(&new_body);
        for (mlir::Operation* op : move_info.ops_to_move) {
          b.clone(*op, value_map);
        }
      },
      // Yield builder: return cloned operation results
      [&](mlir::IRMapping& value_map) {
        llvm::SmallVector<mlir::Value> yield_operands;
        for (mlir::Operation* op : move_info.ops_to_move) {
          yield_operands.push_back(value_map.lookup(op->getResult(0)));
        }
        return yield_operands;
      });

  // Replace all uses of old results with new results
  for (unsigned i = 0; i < move_info.old_results.size(); ++i) {
    move_info.old_results[i].replaceAllUsesWith(new_private.getResult(i));
  }

  temp_private.erase();
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
  llvm::DenseSet<mlir::Operation*> ops_to_exclude;
  for (unsigned idx : removed_indices) {
    mlir::Value yielded_val = old_yield.getOperand(idx);
    if (mlir::Operation* def_op = yielded_val.getDefiningOp()) {
      ops_to_exclude.insert(def_op);
    }
  }

  // Build new yield operands and result types (excluding removed ones)
  llvm::SmallVector<mlir::Value> new_yield_operands;
  llvm::SmallVector<mlir::Type> new_result_types;

  for (auto [idx, operand] : llvm::enumerate(old_yield.getOperands())) {
    if (!to_remove.contains(idx)) {
      new_yield_operands.push_back(operand);
      new_result_types.push_back(operand.getType());
    }
  }

  // Use helper to recreate the private op
  auto new_outer_private = recreatePrivateOp(
      outer_private_, new_result_types,
      // Body builder: clone operations excluding removed ones
      [&](mlir::OpBuilder& b, mlir::IRMapping& value_map,
          mlir::Block& new_body) {
        b.setInsertionPointToStart(&new_body);
        mlir::Block& old_body = outer_private_.getBody().front();
        for (mlir::Operation& op : old_body.without_terminator()) {
          // Skip operations that produce removed results
          if (ops_to_exclude.contains(&op)) {
            LLVM_DEBUG(llvm::dbgs()
                       << "      Skipping stale operation: " << op << "\n");
            continue;
          }
          b.clone(op, value_map);
        }
      },
      // Yield builder: return mapped operands
      [&](mlir::IRMapping& value_map) {
        llvm::SmallVector<mlir::Value> mapped_operands;
        for (mlir::Value operand : new_yield_operands) {
          mapped_operands.push_back(value_map.lookupOrDefault(operand));
        }
        return mapped_operands;
      });

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
