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

#include <string>

#include "OperationTree.hpp"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LLVM.h"

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
      : OperationTreeNode(op),
        materialized_(op != nullptr),
        template_op_(nullptr) {}

  /// Constructor for unmaterialized nodes with a template operation
  PipelineTreeNode(Operation *op, Operation *template_op)
      : OperationTreeNode(op),
        materialized_(op != nullptr),
        template_op_(template_op) {}

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
  void setTemplateOp(Operation *template_op) {
    assert(!operation_op_ && "node already has an operation");
    template_op_ = template_op;
  }

  /// Get the operation if materialized, otherwise get the template operation
  /// This is a convenience method for code that needs to access either the
  /// materialized operation or the template operation for unmaterialized nodes
  Operation *getOperationOrTemplate() const {
    return isMaterialized() ? getOperation() : getTemplateOp();
  }

  virtual void print(raw_ostream &os) const override {
    printNodeHeader(os);
    os << "\n";
    printChildren(os);
  }

  /// Find the outermost loop visited as you traverse up the tree until the stop
  /// function returns true. Returns nullptr if no loop was visited during the
  /// traversal.
  static const PipelineTreeNode *findOutermostLoop(
      const PipelineTreeNode *start_node,
      std::function<bool(const PipelineTreeNode *)> stop);

  /// Find the innermost loop in a loop nest by walking down the first child
  /// chain. Stops when hitting a non-loop node or when there are no more
  /// children. Returns nullptr if the input node is nullptr or not a loop.
  /// Otherwise returns \p start_node if no other inner loop is found.
  static const PipelineTreeNode *findInnermostLoop(
      const PipelineTreeNode *start_node);

 protected:
  /// Helper method to print common node information (indentation, name,
  /// materialization status, location)
  void printNodeHeader(raw_ostream &os) const;

  /// Helper method to print all children
  void printChildren(raw_ostream &os) const;

  bool materialized_;  // Whether this node has an associated IR operation
  Operation
      *template_op_;  // Template operation to use when materializing this node
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
  LoopNode(Operation *op) : PipelineTreeNode(op) {}

  bool isLoopNode() const final override { return true; }
  std::string getNodeName() const override { return "loop"; }

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
  StageNode(Operation *op, int stage_id)
      : PipelineTreeNode(op), stage_id_(stage_id) {}

  bool isStageNode() const final override { return true; }
  std::string getNodeName() const override {
    return "stage " + std::to_string(stage_id_);
  }

  int getStageId() const { return stage_id_; }

  /// Add a dependency to another stage (token-based synchronization)
  /// This node would be the source of the edge and \p target would be the sink.
  void addDependency(StageNode *target) { dependencies_.push_back(target); }

  /// Get other stages that are dependent on this stage node.
  const llvm::SmallVector<StageNode *> &getDependencies() const {
    return dependencies_;
  }

  /// Replace a dependency with another stage node
  void replaceDependency(StageNode *old_target, StageNode *new_target) {
    for (auto &dep : dependencies_) {
      if (dep == old_target) {
        dep = new_target;
        return;
      }
    }
  }

  /// Mark a dependency for removal by nullifying it
  void nullifyDependency(StageNode *target) {
    for (auto &dep : dependencies_) {
      if (dep == target) {
        dep = nullptr;
        return;
      }
    }
  }

  /// Remove all nullified dependencies
  void removeNullifiedDependencies() {
    dependencies_.erase(
        std::remove(dependencies_.begin(), dependencies_.end(), nullptr),
        dependencies_.end());
  }

  void print(raw_ostream &os) const override;

 private:
  int stage_id_;
  llvm::SmallVector<StageNode *>
      dependencies_;  // Stages that are dependent on this stage
};

/// Private node representing ktdf.private operations
/// Note: we do not represent resource allocations such as memref.alloc or
/// fifo.allocate ops as first-class nodes in this tree, because manipulating
/// them would require tracking SSA chains which would be awkward and difficult
/// during materialization. Therefore by design, this tree is only meant for
/// capturing structural relationships, not def-use relations. Furthermore,
/// stage dependencies are captured by the stages so we do not represent tokens
/// in this tree. As a result prvaite nodes do not currently have children in
/// this representation. The reason we have private nodes at all is for
/// legalization (to disallow anything other than private or stage as immediate
/// children of pipeline nodes)
class PrivateNode : public PipelineTreeNode {
 public:
  PrivateNode(Operation *op) : PipelineTreeNode(op) {}

  bool isPrivateNode() const final override { return true; }
  std::string getNodeName() const override { return "private"; }
  void print(raw_ostream &os) const override;
};

/// Pipeline tree representing the complete hierarchical structure
/// Inherits from OperationTree to reuse tree management functionality
class PipelineTree : public OperationTree {
 public:
  PipelineTree() {}

  /// Destructor - cleans up all allocated nodes
  ~PipelineTree() { deleteAllNodes(); }

  const RootNode *getRoot() const {
    return static_cast<const RootNode *>(OperationTree::getRoot());
  }
  RootNode *getRoot() {
    return static_cast<RootNode *>(OperationTree::getRoot());
  }

  /// Factory methods for creating nodes (manages allocation)
  LoopNode *createLoopNode(mlir::Operation *op) {
    LoopNode *node = new LoopNode(op);
    if (op) opToNode_[op] = node;
    return node;
  }

  PipelineNode *createPipelineNode(mlir::Operation *op) {
    PipelineNode *node = new PipelineNode(op);
    if (op) opToNode_[op] = node;
    return node;
  }

  StageNode *createStageNode(mlir::Operation *op, int stage_id) {
    StageNode *node = new StageNode(op, stage_id);
    if (op) opToNode_[op] = node;
    return node;
  }

  PrivateNode *createPrivateNode(mlir::Operation *op) {
    PrivateNode *node = new PrivateNode(op);
    if (op) opToNode_[op] = node;
    return node;
  }

  /// Delete a single node and remove it from the map
  /// Note: user is responsible for unlinking the node and/or the subtree below
  /// it.
  void deleteNode(PipelineTreeNode *node) {
    if (!node) return;

    // Remove from operation map if it has an operation
    if (node->getOperation()) {
      opToNode_.erase(node->getOperation());
    }

    // Delete the node
    delete node;
  }

  /// Delete all nodes in the tree
  void deleteAllNodes() {
    llvm::SmallVector<OperationTreeNode *> to_be_deleted;
    if (root_) {
      OperationTreeNode::walk<OperationTreeNode::kPostOrder>(
          root_, [&to_be_deleted](OperationTreeNode *n) {
            to_be_deleted.push_back(n);
            return n;
          });
      root_ = nullptr;
    }
    for (OperationTreeNode *node : to_be_deleted) {
      delete node;
    }
    opToNode_.clear();
  }

  /// Build the tree from an MLIR operation (typically a function)
  void compute(Operation &unit);

  void recompute(Operation &unit) {
    clear();
    compute(unit);
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
  llvm::DenseMap<mlir::Operation *, PipelineTreeNode *> opToNode_;
};

#endif  // KTIR_SCHEDULER_TRANSFORM_PIPELINETREE_HPP
