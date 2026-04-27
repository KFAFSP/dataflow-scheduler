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

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"
#include <string>

using llvm::raw_ostream;
using llvm::StringRef;
using mlir::function_ref;

namespace mlir {
namespace ktdf {

// Forward declarations
class PipelineTree;
class PipelineTreeNode;
class RootNode;
class LoopNode;
class PipelineNode;
class StageNode;
class PrivateNode;

/// Base class for all nodes in the pipeline tree
/// Similar to OperationNode in dcc/src/Analysis/OperationTree.hpp
class PipelineTreeNode {
  friend class PipelineTree;

 public:
  PipelineTreeNode(Operation *op) : operation_op_(op) {}
  PipelineTreeNode(PipelineTreeNode &) = delete;
  PipelineTreeNode(const PipelineTreeNode &) = delete;
  virtual ~PipelineTreeNode() {}

  bool operator==(const PipelineTreeNode &n) const { return this == &n; }
  bool operator!=(const PipelineTreeNode &n) const { return this != &n; }

  Operation *getOperation() const { return operation_op_; }

  template <class OpTy>
  const OpTy getOpAs() const {
    return dyn_cast<const OpTy>(getOperation());
  }
  template <class OpTy>
  OpTy getOpAs() {
    return const_cast<const PipelineTreeNode *>(this)->getOpAs<OpTy>();
  }

  PipelineTreeNode *getParentNode() const { return parent_node_; }
  unsigned getNumberOfChildren() const;

  /// Get the first/last child in syntactic order.
  PipelineTreeNode *getFirstChild() const { return first_child_; }
  PipelineTreeNode *getLastChild() const;

  /// Get the next/previous sibling in syntactic order.
  PipelineTreeNode *getNextSibling() const { return next_sibling_; }
  PipelineTreeNode *getPrevSibling() const;

  // The depth of an outermost node (in a rooted tree) is one.
  unsigned getDepth() const {
    unsigned depth = 0;
    const PipelineTreeNode *node = this;
    while (node && (node = node->getParentNode())) ++depth;
    return depth;
  }

  bool isOutermost() const { return getDepth() == 1; }
  bool isLeaf() const { return getFirstChild() == nullptr; }

  /// Traversal action function type
  using ActionFuncTy = function_ref<PipelineTreeNode *(PipelineTreeNode *n)>;

  enum WalkOrder {
    kPostOrder = 1,
    kPreOrder,
    kPostOrderGuided,
    kPreOrderGuided,
    kBFS,
    kReverseBFS,
    kKeepOrderRBFS,
  };

  template <WalkOrder>
  static void walk(PipelineTreeNode *n, ActionFuncTy action);

  // Print the tree structure starting from this node.
  virtual void print(raw_ostream &OS, bool showDependencies = false) const;

  // Unlinks the subtree rooted at this node, but does not free any storage.
  void unlink() {
    PipelineTreeNode *prev_sibling = getPrevSibling();
    if (prev_sibling)
      prev_sibling->setNextSibling(getNextSibling());
    else  // this was the first child of the parent
      getParentNode()->setFirstChild(getNextSibling());
    setNextSibling(nullptr);
    setParentNode(nullptr);
  }

  // Node type queries
  virtual bool isRootNode() const { return false; }
  virtual bool isLoopNode() const { return false; }
  virtual bool isPipelineNode() const { return false; }
  virtual bool isStageNode() const { return false; }
  virtual bool isPrivateNode() const { return false; }

  // Get a string representation of the node
  virtual std::string getNodeName() const = 0;

 protected:
  static void preOrderWalk(PipelineTreeNode *n, ActionFuncTy action);
  static void postOrderWalk(PipelineTreeNode *n, ActionFuncTy action);
  static PipelineTreeNode *preOrderGuidedWalk(PipelineTreeNode *n,
                                               ActionFuncTy action);
  static PipelineTreeNode *postOrderGuidedWalk(PipelineTreeNode *n,
                                                ActionFuncTy action);
  static void breadthFirstWalk(PipelineTreeNode *n, ActionFuncTy action);
  static void reverseBreadthFirstWalk(PipelineTreeNode *n, ActionFuncTy action,
                                      bool keep_order = false);

  void setParentNode(PipelineTreeNode *p) { parent_node_ = p; }
  void setFirstChild(PipelineTreeNode *c) { first_child_ = c; }
  void setNextSibling(PipelineTreeNode *s) { next_sibling_ = s; }

  /// Inserts a child node right after the given position \p pos. If no
  /// position is given, then it will be added at the end (ie as the last
  /// child).
  void insertChildNode(PipelineTreeNode *child, PipelineTreeNode *pos = nullptr);

 protected:
  Operation *operation_op_;
  PipelineTreeNode *parent_node_ = nullptr;
  PipelineTreeNode *first_child_ = nullptr;
  PipelineTreeNode *next_sibling_ = nullptr;
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

  scf::ForOp getForOp() const {
    return cast<scf::ForOp>(getOperation());
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
  SmallVector<StageNode *> getStages() const;
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

  const SmallVector<StageNode *> &getDependencies() const {
    return dependencies_;
  }

  void print(raw_ostream &OS, bool showDependencies = false) const override;

 private:
  int stage_id_;
  SmallVector<StageNode *> dependencies_;  // Stages this stage depends on
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

  const SmallVector<Operation *> &getExpandedBuffers() const {
    return expanded_buffers_;
  }

  void print(raw_ostream &OS, bool showDependencies = false) const override;

 private:
  SmallVector<Operation *> expanded_buffers_;  // memref allocation operations
};

/// Pipeline tree representing the complete hierarchical structure
/// Similar to OperationTreeBase in dcc/src/Analysis/OperationTree.hpp
class PipelineTree {
 public:
  PipelineTree() {}
  PipelineTree(PipelineTree &) = delete;
  PipelineTree(const PipelineTree &) = delete;
  virtual ~PipelineTree() { clear(); }

  const RootNode *getRoot() const { return root_; }
  RootNode *getRoot() { return root_; }

  bool empty() const { return !root_ || !root_->getFirstChild(); }

  void remove(PipelineTreeNode *start) { clear(start); }

  virtual void print(raw_ostream &OS, bool showDependencies = false) const {
    PipelineTreeNode *n = getRoot()->getFirstChild();
    while (n) {
      n->print(OS, showDependencies);
      n = n->getNextSibling();
    }
  }

  /// Build the tree from an MLIR operation (typically a function)
  void compute(Operation &unit);

  void recompute(Operation &unit) {
    clear();
    compute(unit);
  }

 protected:
  void clear();

  /// Delete a subtree rooted at \p start.
  void clear(PipelineTreeNode *start);

  /// Filter function to decide whether an operation should be included
  virtual bool isOperationSelected(const Operation &op) const;

 protected:
  RootNode *root_ = nullptr;
};

}  // namespace ktdf
}  // namespace mlir

#endif  // KTIR_SCHEDULER_TRANSFORM_PIPELINETREE_HPP

