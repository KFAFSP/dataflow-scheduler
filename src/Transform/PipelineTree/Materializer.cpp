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
    PipelineTreeNode& root_node, mlir::OpBuilder& builder,
    mlir::IRMapping& value_map) {
  // Clear any previous materialization state
  node_to_materialized_op_.clear();

  // Materialize the root and all its children
  materializeTreeNode(root_node, builder, value_map);

  // Return the materialized operation for the root
  return getMaterializedOp(root_node);
}

mlir::Operation* PipelineTreeMaterializer::getMaterializedOp(
    PipelineTreeNode& node) const {
  auto it = node_to_materialized_op_.find(&node);
  return it != node_to_materialized_op_.end() ? it->second : nullptr;
}

void PipelineTreeMaterializer::materializeTreeNode(PipelineTreeNode& node,
                                                   mlir::OpBuilder& builder,
                                                   mlir::IRMapping& value_map) {
  // Handle different node types
  if (node.isLoopNode()) {
    auto& loop_node = static_cast<LoopNode&>(node);
    materializeLoopNode(loop_node, builder, value_map);
  } else if (node.isStageNode()) {
    auto& stage_node = static_cast<StageNode&>(node);
    materializeStageNode(stage_node, builder, value_map);
  } else if (node.isPipelineNode()) {
    auto& pipeline_node = static_cast<PipelineNode&>(node);
    materializePipelineNode(pipeline_node, builder, value_map);
  } else if (node.isPrivateNode()) {
    // FIXME: For now, we clone the entire private clause into the parent
    // pipeline. This makes all SSA values from the private clause accessible to
    // all stages. In the future, we should properly handle buffer privatization
    // for nested pipelines by moving relevant allocations to the nested
    // pipeline's private clause.
    auto& private_node = static_cast<PrivateNode&>(node);
    mlir::Operation* orig_private_op = private_node.getOperation();
    assert(orig_private_op && "Private node must have an operation");

    LLVM_DEBUG(llvm::dbgs()
               << "    Cloning private clause into parent pipeline\n");
    mlir::Operation* cloned_private =
        builder.clone(*orig_private_op, value_map);

    // Track the materialized operation
    node_to_materialized_op_[&private_node] = cloned_private;
  } else {
    llvm_unreachable("unimplemented node type");
  }
}

void PipelineTreeMaterializer::materializeChildren(
    PipelineTreeNode& parent_node, mlir::OpBuilder& builder,
    mlir::IRMapping& value_map) {
  OperationTreeNode* child = parent_node.getFirstChild();
  while (child) {
    materializeTreeNode(static_cast<PipelineTreeNode&>(*child), builder,
                        value_map);
    child = child->getNextSibling();
  }
}

void PipelineTreeMaterializer::cloneStageBody(mlir::ktdf::StageOp orig_stage,
                                              mlir::OpBuilder& builder,
                                              mlir::IRMapping& value_map) {
  mlir::Block* orig_body = &orig_stage.getBody().front();
  for (mlir::Operation& op : orig_body->without_terminator()) {
    builder.clone(op, value_map);
  }
}

mlir::scf::ForOp PipelineTreeMaterializer::materializeLoopNode(
    LoopNode& loop_node, mlir::OpBuilder& builder, mlir::IRMapping& value_map) {
  // Get the template operation to copy loop bounds from
  mlir::Operation* template_op = loop_node.getOperationOrTemplate();

  assert(template_op && "Loop node must have either operation or template");
  auto template_for = mlir::cast<mlir::scf::ForOp>(template_op);

  // Map the loop bounds through the value_map
  mlir::Value lower_bound =
      value_map.lookupOrDefault(template_for.getLowerBound());
  mlir::Value upper_bound =
      value_map.lookupOrDefault(template_for.getUpperBound());
  mlir::Value step = value_map.lookupOrDefault(template_for.getStep());

  // Create the new loop with empty body
  auto new_loop = mlir::scf::ForOp::create(builder, template_for.getLoc(),
                                           lower_bound, upper_bound, step);

  // Copy attributes from template loop to new loop
  for (mlir::NamedAttribute attr : template_for->getAttrs()) {
    new_loop->setAttr(attr.getName(), attr.getValue());
  }

  // Map the induction variable
  value_map.map(template_for.getInductionVar(), new_loop.getInductionVar());

  // Track the materialized operation
  node_to_materialized_op_[&loop_node] = new_loop.getOperation();

  // Set insertion point inside the loop body and materialize children
  builder.setInsertionPointToStart(new_loop.getBody());
  materializeChildren(loop_node, builder, value_map);
  builder.setInsertionPointAfter(new_loop);

  LLVM_DEBUG(llvm::dbgs() << "    Created loop with IV mapped\n");
  return new_loop;
}

mlir::ktdf::StageOp PipelineTreeMaterializer::materializeStageNode(
    StageNode& stage_node, mlir::OpBuilder& builder,
    mlir::IRMapping& value_map) {
  // Get the original stage operation (or template for unmaterialized stages)
  mlir::Operation* orig_stage_op = stage_node.getOperationOrTemplate();

  mlir::Location loc = builder.getUnknownLoc();
  mlir::ktdf::StageOp orig_stage;

  if (orig_stage_op) {
    orig_stage = mlir::cast<mlir::ktdf::StageOp>(orig_stage_op);
    loc = orig_stage.getLoc();
  }

  // For now, create a simple stage without dependencies
  // TODO: Handle token dependencies properly
  auto new_stage =
      mlir::ktdf::StageOp::create(builder, loc,
                                  /*depends_in=*/mlir::ValueRange{},
                                  /*depends_out=*/mlir::ValueRange{});

  // Track the materialized operation
  node_to_materialized_op_[&stage_node] = new_stage.getOperation();

  // Set insertion point inside the stage body
  mlir::Block* stage_body = &new_stage.getBody().front();
  builder.setInsertionPointToStart(stage_body);

  // Check if this stage has children (loops) or if we need to clone the body
  OperationTreeNode* first_child = stage_node.getFirstChild();

  if (first_child) {
    // Stage has children (loops), materialize them first
    LLVM_DEBUG(llvm::dbgs() << "    Stage has children, materializing them\n");
    materializeChildren(stage_node, builder, value_map);

    // Now clone the stage body into the innermost loop
    if (orig_stage) {
      LLVM_DEBUG(llvm::dbgs()
                 << "    Cloning stage body into innermost loop\n");

      // Find the innermost loop node
      PipelineTreeNode* first_loop =
          static_cast<PipelineTreeNode*>(first_child);
      assert(first_loop->isLoopNode() && "First child should be a loop");
      PipelineTreeNode* innermost_loop_node =
          PipelineTreeNode::findInnermostLoop(first_loop);
      assert(innermost_loop_node);

      // Get the materialized operation for the innermost loop
      mlir::Operation* innermost_op =
          node_to_materialized_op_[innermost_loop_node];
      assert(innermost_op && "Innermost loop should have been materialized");

      // Set insertion point to the innermost loop body and clone stage body
      auto loop_op = mlir::cast<mlir::scf::ForOp>(innermost_op);
      builder.setInsertionPointToStart(loop_op.getBody());
      cloneStageBody(orig_stage, builder, value_map);
    }
  } else if (orig_stage) {
    // Stage has no children, clone the original stage body directly
    LLVM_DEBUG(llvm::dbgs() << "    Cloning original stage body\n");
    cloneStageBody(orig_stage, builder, value_map);
  } else {
    llvm_unreachable("Unmaterialized stage with no children and no template");
  }

  // Restore insertion point after the stage
  builder.setInsertionPointAfter(new_stage);

  LLVM_DEBUG(llvm::dbgs() << "    Created stage " << stage_node.getStageId()
                          << "\n");
  return new_stage;
}

mlir::ktdf::PipelineOp PipelineTreeMaterializer::materializePipelineNode(
    PipelineNode& pipeline_node, mlir::OpBuilder& builder,
    mlir::IRMapping& value_map) {
  // Get the original or template pipeline operation
  mlir::Operation* template_op = pipeline_node.getOperationOrTemplate();

  assert(template_op && "Pipeline node must have either operation or template");
  auto template_pipeline = mlir::cast<mlir::ktdf::PipelineOp>(template_op);

  // Create a new pipeline operation
  auto new_pipeline =
      mlir::ktdf::PipelineOp::create(builder, template_pipeline.getLoc());

  // Track the materialized operation
  node_to_materialized_op_[&pipeline_node] = new_pipeline.getOperation();

  // Set insertion point inside the pipeline body and materialize children
  mlir::Block* pipeline_body = &new_pipeline.getBody().front();
  builder.setInsertionPointToStart(pipeline_body);
  materializeChildren(pipeline_node, builder, value_map);
  builder.setInsertionPointAfter(new_pipeline);

  LLVM_DEBUG(llvm::dbgs() << "    Created pipeline\n");
  return new_pipeline;
}

}  // namespace scheduler

// Made with Bob
