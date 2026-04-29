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
// Pipeline Tree Abstraction
//
// This file defines an internal tree abstraction for representing the
// hierarchical structure of pipelines, stages, and loops. This abstraction
// facilitates transformations like loop sinking and stage coarsening.
//
// The tree represents:
// - Root nodes (top level node)
// - Loop nodes (scf.for operations)
// - Pipeline nodes (ktdf.pipeline operations)
// - Stage nodes (ktdf.stage operations)
// - Private nodes (ktdf.private operations)
//
// Parent-child relationships represent the structural nesting in the IR.
// Stage dependencies (token-based synchronization) are tracked separately.
//===----------------------------------------------------------------------===//

#ifndef KTIR_SCHEDULER_TRANSFORM_PIPELINETREE_HPP
#define KTIR_SCHEDULER_TRANSFORM_PIPELINETREE_HPP

#include "OperationTree.hpp"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"
#include <string>

using llvm::raw_ostream;
using llvm::StringRef;

// Forward declarations
class PipelineTree;
class PipelineTreeNode;
class RootNode;
class LoopNode;
class PipelineNode;
class StageNode;
class PrivateNode;

/// Pipeline-specific tree node base class
/// Inherits from OperationTreeNode to reuse walk algorithms and tree structure
class PipelineTreeNode : public OperationTreeNode {
  friend class PipelineTree;

 public:
  PipelineTreeNode(Operation *op)
      : OperationTreeNode(op), materialized_(op != nullptr), template_op_(nullptr) {}
  
  /// Constructor for unmaterialized nodes with a template operation
  PipelineTreeNode(Operation *op, Operation *templateOp)
      : OperationTreeNode(op), materialized_(op != nullptr), template_op_(templateOp) {}

  // Pipeline-specific type queries
  virtual bool isRootNode() const { return false; }
  virtual bool isLoopNode() const { return false; }
  virtual bool isPipelineNode() const { return false; }
  virtual bool isStageNode() const { return false; }
  virtual bool isPrivateNode() const { return false; }

  // Implement base class pure virtuals
  std::string getNodeType() const override { return "pipeline-tree"; }
  
  /// Check if this node has been materialized (has an associated IR operation)
  bool isMaterialized() const { return materialized_; }

  /// Get the template operation (used for creating new operations when
  /// materializing)
  Operation *getTemplateOp() const { return template_op_; }

  /// Set the template operation
  void setTemplateOp(Operation *templateOp) {
    assert(!operation_op_ && "node already has an operation");
    template_op_ = templateOp;
  }

  /// Mark this node as materialized with the given operation
  void materialize(Operation *op) {
    assert(op && "expect a valid operation for materialization");
    operation_op_ = op;
    materialized_ = true;
  }
  
  // Pipeline-specific print with showDependencies option
  virtual void print(raw_ostream &OS, bool showDependencies) const {
    // Default implementation delegates to base class
    OperationTreeNode::print(OS);
  }

 protected:
  bool materialized_;      // Whether this node has an associated IR operation
  Operation *template_op_; // Template operation to use when materializing this node
};

/// Root node representing the top-level container
class RootNode : public PipelineTreeNode {
 public:
  RootNode() : PipelineTreeNode(nullptr) {}

  bool isRootNode() const final override { return true; }
  std::string getNodeName() const override { return "root"; }
};

/// Loop node representing scf.for operations
class LoopNode : public PipelineTreeNode {
 public:
  LoopNode(Operation *op)
      : PipelineTreeNode(op) {}

  bool isLoopNode() const final override { return true; }
  std::string getNodeName() const override {
    return "loop";
  }

  mlir::scf::ForOp getForOp() const {
    return mlir::cast<mlir::scf::ForOp>(getOperation());
  }
};

/// Pipeline node representing ktdf.pipeline operations
class PipelineNode : public PipelineTreeNode {
 public:
  PipelineNode(Operation *op) : PipelineTreeNode(op) {}

  bool isPipelineNode() const final override { return true; }
  std::string getNodeName() const override { return "pipeline"; }

  /// Get the private node (if any) associated with this pipeline
  PrivateNode *getPrivateNode() const;

  /// Get all stage nodes that are direct children of this pipeline
  llvm::SmallVector<StageNode *> getStages() const;
};

/// Stage node representing ktdf.stage operations
class StageNode : public PipelineTreeNode {
 public:
  StageNode(Operation *op, int stageId)
      : PipelineTreeNode(op), stage_id_(stageId) {}

  bool isStageNode() const final override { return true; }
  std::string getNodeName() const override {
    return "stage " + std::to_string(stage_id_);
  }

  int getStageId() const { return stage_id_; }

  /// Add a dependency to another stage (token-based synchronization)
  void addDependency(StageNode *target) { dependencies_.push_back(target); }

  const llvm::SmallVector<StageNode *> &getDependencies() const {
    return dependencies_;
  }

  void print(raw_ostream &OS, bool showDependencies) const override;

 private:
  int stage_id_;
  llvm::SmallVector<StageNode *> dependencies_;  // Stages this stage depends on
};

/// Private node representing ktdf.private operations
class PrivateNode : public PipelineTreeNode {
 public:
  PrivateNode(Operation *op) : PipelineTreeNode(op) {}

  bool isPrivateNode() const final override { return true; }
  std::string getNodeName() const override { return "private"; }

  /// Track expanded buffer allocations (memref.alloc operations)
  void addExpandedBuffer(Operation *allocOp) {
    expanded_buffers_.push_back(allocOp);
  }

  const llvm::SmallVector<mlir::Operation *> &getExpandedBuffers() const {
    return expanded_buffers_;
  }

  void print(raw_ostream &OS, bool showDependencies) const override;

 private:
  llvm::SmallVector<mlir::Operation *> expanded_buffers_;  // memref allocation operations
};

/// Pipeline tree representing the complete hierarchical structure
/// Inherits from OperationTree to reuse tree management functionality
class PipelineTree : public OperationTree {
 public:
  PipelineTree() {}

  const RootNode *getRoot() const {
    return static_cast<const RootNode *>(OperationTree::getRoot());
  }
  RootNode *getRoot() {
    return static_cast<RootNode *>(OperationTree::getRoot());
  }

  /// Build the tree from an MLIR operation (typically a function)
  void compute(Operation &unit);

  void recompute(Operation &unit) {
    clear();
    compute(unit);
  }

  /// Pipeline-specific print method that supports showing dependencies
  void print(raw_ostream &OS, bool showDependencies) const {
    OperationTreeNode *n = getRoot()->getFirstChild();
    while (n) {
      static_cast<PipelineTreeNode*>(n)->print(OS, showDependencies);
      n = n->getNextSibling();
    }
  }

  /// Get the tree node for a given operation
  PipelineTreeNode *getNodeForOp(Operation *op) const {
    auto it = opToNode_.find(op);
    return it != opToNode_.end() ? it->second : nullptr;
  }

 protected:
  /// Filter function to decide whether an operation should be included
  bool isOperationSelected(const mlir::Operation &op) const override;

 private:
  /// Map from operations to their corresponding tree nodes
  llvm::DenseMap<mlir::Operation*, PipelineTreeNode*> opToNode_;
};


#endif  // KTIR_SCHEDULER_TRANSFORM_PIPELINETREE_HPP

