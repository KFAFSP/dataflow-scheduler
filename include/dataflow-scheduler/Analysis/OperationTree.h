//===-- OperationTree.h -----------------------------------------*- c++ -*-===//
//
// Part of the Dataflow Scheduler project.
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
//
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
//
//===----------------------------------------------------------------------===//

#ifndef DATAFLOW_SCHEDULER_ANALYSIS_OPERATIONTREE_H_
#define DATAFLOW_SCHEDULER_ANALYSIS_OPERATIONTREE_H_

#include <llvm/Support/raw_ostream.h>
#include <mlir/IR/Operation.h>
#include <mlir/Support/LLVM.h>

#include <string>

namespace scheduler {

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
  OperationTreeNode(mlir::Operation* op) : operation_op_(op) {}
  OperationTreeNode(OperationTreeNode&) = delete;
  OperationTreeNode(const OperationTreeNode&) = delete;
  virtual ~OperationTreeNode() {}

  bool operator==(const OperationTreeNode& n) const { return this == &n; }
  bool operator!=(const OperationTreeNode& n) const { return this != &n; }

  mlir::Operation* getOperation() const { return operation_op_; }

  template <class OpTy>
  const OpTy getOpAs() const {
    return mlir::dyn_cast<const OpTy>(getOperation());
  }
  template <class OpTy>
  OpTy getOpAs() {
    return const_cast<const OperationTreeNode*>(this)->getOpAs<OpTy>();
  }

  OperationTreeNode* getParentNode() const { return parent_node_; }
  unsigned getNumberOfChildren() const;

  /// Get the first/last child in syntactic order.
  OperationTreeNode* getFirstChild() const { return first_child_; }
  OperationTreeNode* getLastChild() const;

  /// Get the next/previous sibling in syntactic order.
  OperationTreeNode* getNextSibling() const { return next_sibling_; }
  OperationTreeNode* getPrevSibling() const;

  // The depth of an outermost node (in a rooted tree) is one.
  unsigned getDepth() const {
    unsigned depth = 0;
    const OperationTreeNode* node = this;
    while (node && (node = node->getParentNode())) ++depth;
    return depth;
  }

  bool isOutermost() const { return getDepth() == 1; }
  bool isLeaf() const { return getFirstChild() == nullptr; }

  /// The traversal functions below will call back an action function on each
  /// node visited. The action takes a node as input and operates on it. For
  /// guided walks the action would return the node where the traversal should
  /// continue from. For all other walks, the action's return value is ignored,
  /// so the function can return nullptr in those cases.
  using ActionFuncTy =
      mlir::function_ref<OperationTreeNode*(OperationTreeNode* n)>;

  /// This enum indicates the types of walks supported. In the comments that
  /// follow, 'n' is the node being operated on.
  enum WalkOrder {
    /// Caller must ensure action does not remove 'n' itself or any parent or
    /// ancestor nodes. The action can add or remove sibling or children of
    /// 'n', but new children won't be further visited.
    kPostOrder = 1,

    /// Caller must ensure action does not remove 'n' itself or any parent or
    /// ancestor nodes. The action can add or remove sibling or children of 'n'.
    kPreOrder,

    /// Similar to above functions, except action also determines which
    /// sibling to visit next.
    kPostOrderGuided,
    kPreOrderGuided,

    /// BFS walk of the tree rooted at 'n'. The action must not remove
    /// itself, its siblings or any parent or ancestor. The action can only
    /// remove descendents of 'n'.
    kBFS,

    /// Bottom-up breadth-first traversal.
    ///
    /// This walk visits sibilings in reverse order (right to left).
    ///
    /// The action must not remove parent nodes or nodes to left of the current
    /// node. If action removes nodes, it can only remove nodes to the right of
    /// the given node (since they have already been visited).
    ///
    /// The action can add new nodes, but the new nodes won't be further
    /// visited.
    kReverseBFS,

    /// Bottom-up breadth-first traversal.
    ///
    /// This walk visits sibilings from left to right (program order), at the
    /// expense of more compile-time and memory.
    ///
    /// The action must not remove parent nodes or nodes to right of the current
    /// node. If action removes nodes, it can only remove nodes to the left of
    /// the given node (since they have already been visited).
    ///
    /// The action can add new nodes, but the new nodes won't be further
    /// visited.
    kKeepOrderRBFS,
  };

  /// Template-based walk dispatch
  template <WalkOrder>
  static void walk(OperationTreeNode* n, ActionFuncTy action);

  /// Print the tree structure starting from this node.
  /// Derived classes can override to add domain-specific information.
  virtual void print(llvm::raw_ostream& OS) const;

  /// Dump the tree structure for debugging.
  void dump() const;

  /// Unlinks the subtree rooted at this node, but does not free any storage.
  void unlink() {
    OperationTreeNode* prev_sibling = getPrevSibling();
    if (prev_sibling)
      prev_sibling->setNextSibling(getNextSibling());
    else  // this was the first child of the parent
      getParentNode()->setFirstChild(getNextSibling());
    setNextSibling(nullptr);
    setParentNode(nullptr);
  }

  /// Unlinks only this node (not its subtree) by moving its children to its
  /// parent. The children are inserted in place of this node, maintaining tree
  /// structure.
  void unlinkNode();

  /// Get a string representation of the node type.
  /// Derived classes must implement this to identify their tree type.
  virtual std::string getNodeType() const = 0;

  /// Get a string representation of the node name.
  /// Derived classes must implement this to identify specific nodes.
  virtual std::string getNodeName() const = 0;

  /// Inserts a child node right after the given position \p pos. If no
  /// position is given, then it will be added at the end (ie as the last
  /// child).
  void insertChildNode(OperationTreeNode* child,
                       OperationTreeNode* pos = nullptr);

  /// Inserts a child node as the first child of this node.
  /// Note: When this node is a leaf (has no children), this method behaves
  /// identically to insertChildNode(child, nullptr).
  void insertAsFirstChild(OperationTreeNode* child);

  /// Walk up the parent chain and find the first parent that matches the
  /// predicate. Returns nullptr if no matching parent is found.
  template <typename PredicateFn>
  const OperationTreeNode* findParent(PredicateFn predicate) const {
    const OperationTreeNode* current = getParentNode();
    while (current) {
      if (predicate(current)) {
        return current;
      }
      current = current->getParentNode();
    }
    return nullptr;
  }

  /// Non-const version that calls the const version
  template <typename PredicateFn>
  OperationTreeNode* findParent(PredicateFn predicate) {
    return const_cast<OperationTreeNode*>(
        static_cast<const OperationTreeNode*>(this)->findParent(predicate));
  }

 protected:
  static void preOrderWalk(OperationTreeNode* n, ActionFuncTy action);
  static void postOrderWalk(OperationTreeNode* n, ActionFuncTy action);
  static OperationTreeNode* preOrderGuidedWalk(OperationTreeNode* n,
                                               ActionFuncTy action);
  static OperationTreeNode* postOrderGuidedWalk(OperationTreeNode* n,
                                                ActionFuncTy action);
  static void breadthFirstWalk(OperationTreeNode* n, ActionFuncTy action);
  static void reverseBreadthFirstWalk(OperationTreeNode* n, ActionFuncTy action,
                                      bool keep_order = false);

  void setParentNode(OperationTreeNode* p) { parent_node_ = p; }
  void setFirstChild(OperationTreeNode* c) { first_child_ = c; }
  void setNextSibling(OperationTreeNode* s) { next_sibling_ = s; }

 protected:
  mlir::Operation* operation_op_;
  OperationTreeNode* parent_node_ = nullptr;
  OperationTreeNode* first_child_ = nullptr;
  OperationTreeNode* next_sibling_ = nullptr;
};

/// Conceptually a program consists of a forest of operation trees with
/// OperationNodes being the building blocks of such trees. The OperationTree
/// class introduces a synthetic root node and collects all such trees under
/// that single root. Clients would not directly instantiate this class. Instead
/// they should derived from it, and provide functions that iterate over the IR
/// and construct the tree.
/// Derived classes decide how to build their specific tree.
class OperationTree {
 public:
  OperationTree() {}
  OperationTree(OperationTree&) = delete;
  OperationTree(const OperationTree&) = delete;
  virtual ~OperationTree() { clear(); }

  const OperationTreeNode* getRoot() const {
    assert(root_ && root_->getNextSibling() == nullptr &&
           root_->getParentNode() == nullptr && "invalid root");
    return root_;
  }
  OperationTreeNode* getRoot() { return root_; }

  bool empty() const { return !root_ || !root_->getFirstChild(); }

  void remove(OperationTreeNode* start) { clear(start); }

  virtual void print(llvm::raw_ostream& OS) const {
    OperationTreeNode* n = getRoot()->getFirstChild();
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
  void clear(OperationTreeNode* start);

  /// Filter function to decide whether an operation should be included.
  /// Derived classes should override this to select relevant operations.
  virtual bool isOperationSelected(const mlir::Operation& op) const = 0;

 protected:
  OperationTreeNode* root_ = nullptr;
};

}  // namespace scheduler

#endif  // DATAFLOW_SCHEDULER_ANALYSIS_OPERATIONTREE_H_
