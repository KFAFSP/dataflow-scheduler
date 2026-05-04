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
// Pipeline Tree Materializer
//
// This file defines a materializer for generating MLIR IR from a PipelineTree.
// The materializer traverses the tree and creates corresponding MLIR operations
// (loops, pipelines, stages) while maintaining value mappings for SSA values.
//===----------------------------------------------------------------------===//

#ifndef KTIR_SCHEDULER_TRANSFORM_PIPELINETREE_MATERIALIZER_HPP
#define KTIR_SCHEDULER_TRANSFORM_PIPELINETREE_MATERIALIZER_HPP

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "llvm/ADT/DenseMap.h"
#include "src/Analysis/PipelineTree.hpp"
#include "src/Dialects/KTDF/KTDFOps.hpp"

namespace scheduler {

/// Materializer class for generating MLIR IR from a PipelineTree
class PipelineTreeMaterializer {
 public:
  PipelineTreeMaterializer() = default;

  /// Materialize the entire tree starting from the given node
  /// Returns the root operation created
  mlir::Operation* materialize(PipelineTreeNode& root_node,
                               mlir::OpBuilder& builder,
                               mlir::IRMapping& value_map);

  /// Get the materialized operation for a given tree node
  /// Returns nullptr if the node hasn't been materialized yet
  mlir::Operation* getMaterializedOp(PipelineTreeNode& node) const;

 private:
  /// Materialize a loop node by creating an scf.for operation
  mlir::scf::ForOp materializeLoopNode(LoopNode& loop_node,
                                       mlir::OpBuilder& builder,
                                       mlir::IRMapping& value_map);

  /// Materialize a stage node by creating a ktdf.stage operation
  mlir::ktdf::StageOp materializeStageNode(StageNode& stage_node,
                                           mlir::OpBuilder& builder,
                                           mlir::IRMapping& value_map);

  /// Materialize a pipeline node by creating a ktdf.pipeline operation
  mlir::ktdf::PipelineOp materializePipelineNode(PipelineNode& pipeline_node,
                                                 mlir::OpBuilder& builder,
                                                 mlir::IRMapping& value_map);

  /// Recursively materialize a tree node and its children
  void materializeTreeNode(PipelineTreeNode& node,
                          mlir::OpBuilder& builder,
                          mlir::IRMapping& value_map);

  /// Helper to materialize all children of a node
  void materializeChildren(PipelineTreeNode& parent_node,
                          mlir::OpBuilder& builder,
                          mlir::IRMapping& value_map);

  /// Helper to clone stage body operations
  void cloneStageBody(mlir::ktdf::StageOp orig_stage,
                     mlir::OpBuilder& builder,
                     mlir::IRMapping& value_map);

  /// Tracking map from tree nodes to their materialized operations
  llvm::DenseMap<PipelineTreeNode*, mlir::Operation*> node_to_materialized_op_;
};

}  // namespace scheduler

#endif  // KTIR_SCHEDULER_TRANSFORM_PIPELINETREE_MATERIALIZER_HPP

// Made with Bob
