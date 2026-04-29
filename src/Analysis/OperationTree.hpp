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
// Operation Tree Node Base Abstraction
//
// This file defines a generic, reusable base class for building tree
// abstractions over MLIR operations. It provides:
// - Common tree navigation (parent, children, siblings)
// - Reusable walk algorithms (pre-order, post-order, BFS, etc.)
// - Tree manipulation utilities (unlink, insert)
// 
// This tree uses a LCRS (Left-Child Right-Sibling) representation.
//
// These base classes are designed to be extended by domain-specific tree
// implementations (e.g., PipelineTree, ControlFlowTree) that add their
// own node types and type queries.
//===----------------------------------------------------------------------===//

#ifndef KTIR_SCHEDULER_ANALYSIS_OPERATIONTREE_HPP
#define KTIR_SCHEDULER_ANALYSIS_OPERATIONTREE_HPP

#include "mlir/IR/Operation.h"
#include "mlir/Support/LLVM.h"
#include "llvm/Support/raw_ostream.h"
#include <string>

using llvm::raw_ostream;
using mlir::function_ref;
using mlir::Operation;

/// Base class for all nodes in operation tree abstractions.
/// This class provides generic tree structure and traversal algorithms
/// that can be reused across different tree types (pipeline trees,
/// control flow trees, etc.).
///
/// Derived classes should:
/// 1. Create an intermediate base (e.g., PipelineTreeNode) that adds
///    domain-specific type queries (isPipelineNode(), isStageNode(), etc.)
/// 2. Create concrete node classes that implement specific node types
///
/// This design allows multiple tree abstractions to share walk algorithms
/// while maintaining clean separation of domain-specific concerns.
class OperationTreeNode {
 public:
  OperationTreeNode(Operation *op) : operation_op_(op) {}
  OperationTreeNode(OperationTreeNode &) = delete;
  OperationTreeNode(const OperationTreeNode &) = delete;
  virtual ~OperationTreeNode() {}

  bool operator==(const OperationTreeNode &n) const { return this == &n; }
  bool operator!=(const OperationTreeNode &n) const { return this != &n; }

  Operation *getOperation() const { return operation_op_; }

  template <class OpTy>
  const OpTy getOpAs() const {
    return mlir::dyn_cast<const OpTy>(getOperation());
  }
  template <class OpTy>
  OpTy getOpAs() {
    return const_cast<const OperationTreeNode *>(this)->getOpAs<OpTy>();
  }

  OperationTreeNode *getParentNode() const { return parent_node_; }
  unsigned getNumberOfChildren() const;

  /// Get the first/last child in syntactic order.
  OperationTreeNode *getFirstChild() const { return first_child_; }
  OperationTreeNode *getLastChild() const;

  /// Get the next/previous sibling in syntactic order.
  OperationTreeNode *getNextSibling() const { return next_sibling_; }
  OperationTreeNode *getPrevSibling() const;

  // The depth of an outermost node (in a rooted tree) is one.
  unsigned getDepth() const {
    unsigned depth = 0;
    const OperationTreeNode *node = this;
    while (node && (node = node->getParentNode())) ++depth;
    return depth;
  }

  bool isOutermost() const { return getDepth() == 1; }
  bool isLeaf() const { return getFirstChild() == nullptr; }

  /// Traversal action function type
  using ActionFuncTy = function_ref<OperationTreeNode *(OperationTreeNode *n)>;

  /// Walk order enumeration
  enum WalkOrder {
    kPostOrder = 1,
    kPreOrder,
    kPostOrderGuided,
    kPreOrderGuided,
    kBFS,
    kReverseBFS,
    kKeepOrderRBFS,
  };

  /// Template-based walk dispatch
  template <WalkOrder>
  static void walk(OperationTreeNode *n, ActionFuncTy action);

  /// Print the tree structure starting from this node.
  /// Derived classes can override to add domain-specific information.
  virtual void print(raw_ostream &OS) const;

  /// Unlinks the subtree rooted at this node, but does not free any storage.
  void unlink() {
    OperationTreeNode *prev_sibling = getPrevSibling();
    if (prev_sibling)
      prev_sibling->setNextSibling(getNextSibling());
    else  // this was the first child of the parent
      getParentNode()->setFirstChild(getNextSibling());
    setNextSibling(nullptr);
    setParentNode(nullptr);
  }

  /// Get a string representation of the node type.
  /// Derived classes must implement this to identify their tree type.
  virtual std::string getNodeType() const = 0;

  /// Get a string representation of the node name.
  /// Derived classes must implement this to identify specific nodes.
  virtual std::string getNodeName() const = 0;

  /// Inserts a child node right after the given position \p pos. If no
  /// position is given, then it will be added at the end (ie as the last
  /// child).
  void insertChildNode(OperationTreeNode *child, OperationTreeNode *pos = nullptr);

 protected:
  static void preOrderWalk(OperationTreeNode *n, ActionFuncTy action);
  static void postOrderWalk(OperationTreeNode *n, ActionFuncTy action);
  static OperationTreeNode *preOrderGuidedWalk(OperationTreeNode *n,
                                                ActionFuncTy action);
  static OperationTreeNode *postOrderGuidedWalk(OperationTreeNode *n,
                                                 ActionFuncTy action);
  static void breadthFirstWalk(OperationTreeNode *n, ActionFuncTy action);
  static void reverseBreadthFirstWalk(OperationTreeNode *n, ActionFuncTy action,
                                      bool keep_order = false);

  void setParentNode(OperationTreeNode *p) { parent_node_ = p; }
  void setFirstChild(OperationTreeNode *c) { first_child_ = c; }
  void setNextSibling(OperationTreeNode *s) { next_sibling_ = s; }

 protected:
  Operation *operation_op_;
  OperationTreeNode *parent_node_ = nullptr;
  OperationTreeNode *first_child_ = nullptr;
  OperationTreeNode *next_sibling_ = nullptr;
};

/// Base class for operation tree containers.
/// Provides tree management (construction, destruction, clearing).
/// Derived classes decide how to build their specific tree.
class OperationTree {
 public:
  OperationTree() {}
  OperationTree(OperationTree &) = delete;
  OperationTree(const OperationTree &) = delete;
  virtual ~OperationTree() { clear(); }

  const OperationTreeNode *getRoot() const { return root_; }
  OperationTreeNode *getRoot() { return root_; }

  bool empty() const { return !root_ || !root_->getFirstChild(); }

  void remove(OperationTreeNode *start) { clear(start); }

  virtual void print(raw_ostream &OS) const {
    OperationTreeNode *n = getRoot()->getFirstChild();
    while (n) {
      n->print(OS);
      n = n->getNextSibling();
    }
  }

  /// Convenience method to walk the tree starting from root
  template <OperationTreeNode::WalkOrder Order>
  void walk(OperationTreeNode::ActionFuncTy action) {
    if (root_) {
      OperationTreeNode::walk<Order>(root_, action);
    }
  }

 protected:
  void clear();

  /// Delete a subtree rooted at \p start.
  void clear(OperationTreeNode *start);

  /// Filter function to decide whether an operation should be included.
  /// Derived classes should override this to select relevant operations.
  virtual bool isOperationSelected(const Operation &op) const = 0;

 protected:
  OperationTreeNode *root_ = nullptr;
};

#endif  // KTIR_SCHEDULER_ANALYSIS_OPERATIONTREE_HPP

