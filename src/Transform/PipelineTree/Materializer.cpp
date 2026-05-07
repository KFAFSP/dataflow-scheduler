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
// Pipeline Tree Materializer Implementation
//===----------------------------------------------------------------------===//

#include "Materializer.hpp"

#include "llvm/Support/Debug.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "src/Dialects/KTDF/KTDFOps.hpp"

#define DEBUG_TYPE "pipeline-tree-materializer"

namespace scheduler {

mlir::Operation* PipelineTreeMaterializer::materialize(
    const PipelineTreeNode& root_node) {
  // Clear any previous materialization state
  node_to_materialized_op_.clear();

  // Materialize the root and all its children
  materializeTreeNode(root_node);

  // Return the materialized operation for the root
  return getMaterializedOp(root_node);
}

mlir::Operation* PipelineTreeMaterializer::getMaterializedOp(
    const PipelineTreeNode& node) const {
  auto it = node_to_materialized_op_.find(&node);
  return it != node_to_materialized_op_.end() ? it->second : nullptr;
}

void PipelineTreeMaterializer::materializeTreeNode(
    const PipelineTreeNode& node) {
  // Handle different node types
  if (node.isLoopNode()) {
    const auto& loop_node = static_cast<const LoopNode&>(node);
    materializeLoopNode(loop_node);
  } else if (node.isStageNode()) {
    const auto& stage_node = static_cast<const StageNode&>(node);
    materializeStageNode(stage_node);
  } else if (node.isPipelineNode()) {
    const auto& pipeline_node = static_cast<const PipelineNode&>(node);
    materializePipelineNode(pipeline_node);
  } else if (node.isPrivateNode()) {
    const auto& private_node = static_cast<const PrivateNode&>(node);
    materializePrivateNode(private_node);
  } else {
    llvm_unreachable("unimplemented node type");
  }
}

void PipelineTreeMaterializer::materializeChildren(
    const PipelineTreeNode& parent_node) {
  OperationTreeNode* child = parent_node.getFirstChild();
  while (child) {
    materializeTreeNode(static_cast<const PipelineTreeNode&>(*child));
    child = child->getNextSibling();
  }
}

void PipelineTreeMaterializer::cloneStageBody(mlir::ktdf::StageOp orig_stage) {
  mlir::Block* orig_body = &orig_stage.getBody().front();
  for (mlir::Operation& op : *orig_body) {
    if (op.hasTrait<mlir::OpTrait::IsTerminator>()) continue;
    // Check if this is a DataTransferOp and we have a handler
    if (auto transfer_op = mlir::dyn_cast<mlir::ktdf::DataTransferOp>(&op)) {
      if (buffer_handler_) {
        mlir::Operation* new_op = buffer_handler_->handleDataTransfer(
            transfer_op, builder_, value_map_);
        if (new_op) {
          // Handler created a replacement operation, skip default cloning
          continue;
        }
      }
    }

    // Default: clone the operation
    builder_.clone(op, value_map_);
  }
}

mlir::scf::ForOp PipelineTreeMaterializer::materializeLoopNode(
    const LoopNode& loop_node) {
  // Get the template operation to copy loop bounds from
  mlir::Operation* template_op = loop_node.getOperationOrTemplate();

  assert(template_op && "Loop node must have either operation or template");
  auto template_for = mlir::cast<mlir::scf::ForOp>(template_op);

  // Map the loop bounds through the value_map
  mlir::Value lower_bound =
      value_map_.lookupOrDefault(template_for.getLowerBound());
  mlir::Value upper_bound =
      value_map_.lookupOrDefault(template_for.getUpperBound());
  mlir::Value step = value_map_.lookupOrDefault(template_for.getStep());

  // Create the new loop with empty body
  auto new_loop = mlir::scf::ForOp::create(builder_, template_for.getLoc(),
                                           lower_bound, upper_bound, step);

  // Copy attributes from template loop to new loop
  for (mlir::NamedAttribute attr : template_for->getAttrs()) {
    new_loop->setAttr(attr.getName(), attr.getValue());
  }

  // Map the induction variable
  value_map_.map(template_for.getInductionVar(), new_loop.getInductionVar());

  // Track the materialized operation
  node_to_materialized_op_[&loop_node] = new_loop.getOperation();

  // Set insertion point inside the loop body and materialize children
  builder_.setInsertionPointToStart(new_loop.getBody());
  materializeChildren(loop_node);
  builder_.setInsertionPointAfter(new_loop);

  LLVM_DEBUG(llvm::dbgs() << "    Created loop with IV mapped\n");
  return new_loop;
}

mlir::ktdf::StageOp PipelineTreeMaterializer::materializeStageNode(
    const StageNode& stage_node) {
  // Get the original stage operation (or template for unmaterialized stages)
  mlir::Operation* orig_stage_op = stage_node.getOperationOrTemplate();

  mlir::Location loc = builder_.getUnknownLoc();
  mlir::ktdf::StageOp orig_stage;

  if (orig_stage_op) {
    orig_stage = mlir::cast<mlir::ktdf::StageOp>(orig_stage_op);
    loc = orig_stage.getLoc();
  }

  // For now, create a simple stage without dependencies
  // TODO: Handle token dependencies properly
  auto new_stage =
      mlir::ktdf::StageOp::create(builder_, loc,
                                  /*depends_in=*/mlir::ValueRange{},
                                  /*depends_out=*/mlir::ValueRange{});

  // Track the materialized operation
  node_to_materialized_op_[&stage_node] = new_stage.getOperation();

  // Set insertion point inside the stage body
  mlir::Block* stage_body = &new_stage.getBody().front();
  builder_.setInsertionPointToStart(stage_body);

  // Check if this stage has children (loops) or if we need to clone the body
  OperationTreeNode* first_child = stage_node.getFirstChild();

  if (first_child) {
    // Stage has children (loops), materialize them first
    LLVM_DEBUG(llvm::dbgs() << "    Stage has children, materializing them\n");
    materializeChildren(stage_node);

    // Now clone the stage body into the innermost loop
    if (orig_stage) {
      LLVM_DEBUG(llvm::dbgs()
                 << "    Cloning stage body into innermost loop\n");

      // Find the innermost loop node
      const PipelineTreeNode* first_loop =
          static_cast<const PipelineTreeNode*>(first_child);
      assert(first_loop->isLoopNode() && "First child should be a loop");
      const PipelineTreeNode* innermost_loop_node =
          PipelineTreeNode::findInnermostLoop(first_loop);
      assert(innermost_loop_node);

      // Get the materialized operation for the innermost loop
      mlir::Operation* innermost_op =
          node_to_materialized_op_[innermost_loop_node];
      assert(innermost_op && "Innermost loop should have been materialized");

      // Set insertion point to the innermost loop body and clone stage body
      auto loop_op = mlir::cast<mlir::scf::ForOp>(innermost_op);
      builder_.setInsertionPointToStart(loop_op.getBody());
      cloneStageBody(orig_stage);
    }
  } else if (orig_stage) {
    // Stage has no children, clone the original stage body directly
    LLVM_DEBUG(llvm::dbgs() << "    Cloning original stage body\n");
    cloneStageBody(orig_stage);
  } else {
    llvm_unreachable("Unmaterialized stage with no children and no template");
  }

  // Restore insertion point after the stage
  builder_.setInsertionPointAfter(new_stage);

  LLVM_DEBUG(llvm::dbgs() << "    Created stage " << stage_node.getStageId()
                          << "\n");
  return new_stage;
}

mlir::ktdf::PipelineOp PipelineTreeMaterializer::materializePipelineNode(
    const PipelineNode& pipeline_node) {
  // Get the original or template pipeline operation
  mlir::Operation* template_op = pipeline_node.getOperationOrTemplate();

  assert(template_op && "Pipeline node must have either operation or template");
  auto template_pipeline = mlir::cast<mlir::ktdf::PipelineOp>(template_op);

  // Create a new pipeline operation
  auto new_pipeline =
      mlir::ktdf::PipelineOp::create(builder_, template_pipeline.getLoc());

  // Track the materialized operation
  node_to_materialized_op_[&pipeline_node] = new_pipeline.getOperation();

  // Set insertion point inside the pipeline body and materialize children
  mlir::Block* pipeline_body = &new_pipeline.getBody().front();
  builder_.setInsertionPointToStart(pipeline_body);
  materializeChildren(pipeline_node);
  builder_.setInsertionPointAfter(new_pipeline);

  LLVM_DEBUG(llvm::dbgs() << "    Created pipeline\n");
  return new_pipeline;
}

void PipelineTreeMaterializer::materializePrivateNode(
    const PrivateNode& private_node) {
  // FIXME: For now, we clone the entire private clause into the parent
  // pipeline. This makes all SSA values from the private clause accessible to
  // all stages. In the future, we should properly handle buffer privatization
  // for nested pipelines by moving relevant allocations to the nested
  // pipeline's private clause.
  mlir::Operation* orig_private_op = private_node.getOperation();
  assert(orig_private_op && "Private node must have an operation");

  LLVM_DEBUG(llvm::dbgs() << "    Materializing private clause\n");

  auto orig_private = mlir::cast<mlir::ktdf::PrivateOp>(orig_private_op);
  mlir::Block& orig_body = orig_private.getBody().front();

  // Save the current insertion point (inside the new pipeline body)
  auto saved_insertion_point = builder_.saveInsertionPoint();

  // Create a temporary block to hold operations before splicing into private op
  mlir::Block tmp_block;
  builder_.setInsertionPointToStart(&tmp_block);

  for (mlir::Operation& op : orig_body.without_terminator()) {
    if (auto alloc_op = mlir::dyn_cast<mlir::memref::AllocOp>(&op)) {
      if (buffer_handler_) {
        mlir::Operation* new_op =
            buffer_handler_->handleAllocOp(alloc_op, builder_, value_map_);
        if (new_op) {
          // Handler created a replacement operation
          continue;
        }
      }
    }

    // Default: clone the operation
    builder_.clone(op, value_map_);
  }

  // Get the yield operation and determine result types from mapped operands
  auto orig_yield =
      mlir::cast<mlir::ktdf::PrivateYieldOp>(orig_body.getTerminator());

  llvm::SmallVector<mlir::Type> result_types;
  llvm::SmallVector<mlir::Value> new_yield_operands;
  for (mlir::Value yielded_val : orig_yield.getOperands()) {
    mlir::Value mapped_val = value_map_.lookupOrDefault(
        yielded_val);  // this is wrong...the value_map doesn't hold the
                       // replaced allocations.. should handleAllocOp update it?
    new_yield_operands.push_back(mapped_val);
    result_types.push_back(mapped_val.getType());
  }

  // Restore insertion point to create the private operation in the right place
  builder_.restoreInsertionPoint(saved_insertion_point);

  // Now create the private operation with correct result types
  auto new_private = mlir::ktdf::PrivateOp::create(
      builder_, orig_private.getLoc(), result_types);

  // Get the empty block that was created with the private op
  mlir::Block& private_body = new_private.getBody().front();

  // Splice all operations from tmp_block into the private_body
  private_body.getOperations().splice(private_body.end(),
                                      tmp_block.getOperations());

  // Create yield in the private body
  builder_.setInsertionPointToEnd(&private_body);
  mlir::ktdf::PrivateYieldOp::create(builder_, orig_yield.getLoc(),
                                     new_yield_operands);

  // Map original private results to new private results
  for (auto [orig_result, new_result] :
       llvm::zip(orig_private.getResults(), new_private.getResults())) {
    value_map_.map(orig_result, new_result);
  }

  // Restore insertion point
  builder_.setInsertionPointAfter(new_private);

  // Track the materialized operation
  node_to_materialized_op_[&private_node] = new_private.getOperation();
}

}  // namespace scheduler

// Made with Bob
