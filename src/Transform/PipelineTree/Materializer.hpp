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
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/OpDefinition.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "src/Analysis/PipelineTree.hpp"
#include "src/Dialects/KTDF/KTDFOps.hpp"

namespace scheduler {

//===----------------------------------------------------------------------===//
// Buffer Handler Interface
//===----------------------------------------------------------------------===//

/// Handler interface for customizing buffer allocation and data transfer
/// operations during materialization. This allows transformations to modify
/// how buffers are allocated (e.g., expanding dimensions) and how data
/// transfers access those buffers (e.g., adjusting indices).
class BufferHandler {
 public:
  virtual ~BufferHandler() = default;

  /// Called when materializing a memref.alloc operation.
  /// The handler can return a custom allocation operation or nullptr to use
  /// default cloning behavior.
  ///
  /// \param orig_alloc The original allocation operation from the template
  /// \param builder The builder positioned where the new alloc should be created
  /// \param value_map The current value mapping for SSA values
  /// \return A new allocation operation, or nullptr to use default behavior
  virtual mlir::Operation* handleAllocOp(
      mlir::memref::AllocOp orig_alloc,
      mlir::OpBuilder& builder,
      mlir::IRMapping& value_map) = 0;

  /// Called when materializing a data transfer operation.
  /// The handler can return a custom transfer operation or nullptr to use
  /// default cloning behavior.
  ///
  /// \param orig_transfer The original data transfer operation
  /// \param builder The builder positioned where the new transfer should be created
  /// \param value_map The current value mapping for SSA values
  /// \return A new data transfer operation, or nullptr to use default behavior
  virtual mlir::Operation* handleDataTransfer(
      mlir::ktdf::DataTransferOp orig_transfer,
      mlir::OpBuilder& builder,
      mlir::IRMapping& value_map) = 0;
};

//===----------------------------------------------------------------------===//
// Pipeline Tree Materializer
//===----------------------------------------------------------------------===//

/// Materializer class for generating MLIR IR from a PipelineTree
class PipelineTreeMaterializer {
 public:
  /// Constructor
  /// \param builder The MLIR builder for creating operations
  /// \param value_map The value mapping for SSA values
  /// \param buffer_handler Optional handler for customizing buffer operations
  PipelineTreeMaterializer(mlir::OpBuilder& builder,
                          mlir::IRMapping& value_map,
                          BufferHandler* buffer_handler = nullptr)
      : builder_(builder), value_map_(value_map), buffer_handler_(buffer_handler) {}

  /// Materialize the entire tree starting from the given node
  /// Returns the root operation created
  mlir::Operation* materialize(const PipelineTreeNode& root_node);

  /// Get the materialized operation for a given tree node
  /// Returns nullptr if the node hasn't been materialized yet
  mlir::Operation* getMaterializedOp(const PipelineTreeNode& node) const;

 private:
  /// Materialize a loop node by creating an scf.for operation
  mlir::scf::ForOp materializeLoopNode(const LoopNode& loop_node);

  /// Materialize a stage node by creating a ktdf.stage operation
  mlir::ktdf::StageOp materializeStageNode(const StageNode& stage_node);

  /// Materialize a pipeline node by creating a ktdf.pipeline operation
  mlir::ktdf::PipelineOp materializePipelineNode(const PipelineNode& pipeline_node);

  /// Materialize a private node by cloning its operations
  void materializePrivateNode(const PrivateNode& private_node);

  /// Recursively materialize a tree node and its children
  void materializeTreeNode(const PipelineTreeNode& node);

  /// Helper to materialize all children of a node
  void materializeChildren(const PipelineTreeNode& parent_node);

  /// Helper to clone stage body operations
  void cloneStageBody(mlir::ktdf::StageOp orig_stage);

  /// MLIR builder for creating operations
  mlir::OpBuilder& builder_;
  
  /// Value mapping for SSA values
  mlir::IRMapping& value_map_;
  
  /// Optional handler for customizing buffer operations
  BufferHandler* buffer_handler_;
  
  /// Tracking map from tree nodes to their materialized operations
  llvm::DenseMap<const PipelineTreeNode*, mlir::Operation*> node_to_materialized_op_;
};

}  // namespace scheduler

#endif  // KTIR_SCHEDULER_TRANSFORM_PIPELINETREE_MATERIALIZER_HPP

// Made with Bob
