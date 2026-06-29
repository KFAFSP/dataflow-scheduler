//===----------------------------------------------------------------------===//
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
// Operation Tree Node Implementation
//
//===----------------------------------------------------------------------===//

#include "dataflow-scheduler/Analysis/OperationTree.h"

#include <cassert>
#include <queue>

#include "dataflow-scheduler/Analysis/Utils.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

using namespace scheduler;

//===----------------------------------------------------------------------===//
// OperationTreeNode Implementation
//===----------------------------------------------------------------------===//

unsigned OperationTreeNode::getNumberOfChildren() const {
  unsigned count = 0;
  OperationTreeNode* child = getFirstChild();
  while (child) {
    child = child->getNextSibling();
    ++count;
  }
  return count;
}

OperationTreeNode* OperationTreeNode::getLastChild() const {
  OperationTreeNode* child = getFirstChild();
  if (!child) return nullptr;
  while (child->getNextSibling()) {
    child = child->getNextSibling();
  }
  return child;
}

OperationTreeNode* OperationTreeNode::getPrevSibling() const {
  assert(getParentNode() && "expected a parent");
  OperationTreeNode* prev = getParentNode()->getFirstChild();
  if (prev == this) return nullptr;
  while (prev) {
    if (prev->getNextSibling() == this) break;
    prev = prev->getNextSibling();
  }
  return prev;
}

void OperationTreeNode::insertChildNode(OperationTreeNode* child,
                                        OperationTreeNode* pos) {
  assert(child && "Cannot insert null child");
  if (isLeaf()) {
    assert(pos == nullptr && "position incorrectly specified");
    setFirstChild(child);
  } else if (pos) {
    assert(pos->getParentNode() == this &&
           "expected position to be one of the existing children of this node");
    OperationTreeNode* old_pos_next = pos->getNextSibling();
    pos->setNextSibling(child);
    child->setNextSibling(old_pos_next);
  } else
    getLastChild()->setNextSibling(child);
  child->setParentNode(this);
}

void OperationTreeNode::insertAsFirstChild(OperationTreeNode* child) {
  assert(child && "Cannot insert null child");
  OperationTreeNode* old_first_child = getFirstChild();
  if (old_first_child) {
    child->setNextSibling(old_first_child);
  }
  setFirstChild(child);
  child->setParentNode(this);
}

void OperationTreeNode::print(llvm::raw_ostream& OS) const {
  preOrderWalk(const_cast<OperationTreeNode*>(this),
               [&OS](OperationTreeNode* n) {
                 auto* op = n->getOperation();
                 unsigned depth = n->getDepth();
                 OS.indent(depth - 1);
                 printLocation(OS, op);
                 return nullptr;
               });
}

void OperationTreeNode::dump() const { print(llvm::outs()); }

void OperationTreeNode::unlinkNode() {
  OperationTreeNode* parent = getParentNode();
  if (!parent) return;  // Cannot unlink root node

  OperationTreeNode* const prev_sibling = getPrevSibling();
  OperationTreeNode* const next_sibling = getNextSibling();
  OperationTreeNode* const first_child = getFirstChild();
  OperationTreeNode* const last_child = getLastChild();

  // If this node has children, move them to parent
  if (first_child) {
    // Connect children's chain to siblings
    if (prev_sibling) {
      prev_sibling->setNextSibling(first_child);
    } else {
      // This was the first child, so first child becomes new first child
      parent->setFirstChild(first_child);
    }

    // Update parent pointers for all children
    OperationTreeNode* child = first_child;
    while (child) {
      child->setParentNode(parent);
      if (child == last_child) {
        child->setNextSibling(next_sibling);
        break;
      }
      child = child->getNextSibling();
    }
  } else {
    // No children, just unlink this node
    if (prev_sibling) {
      prev_sibling->setNextSibling(next_sibling);
    } else {
      parent->setFirstChild(next_sibling);
    }
  }

  // Clear this node's links
  setFirstChild(nullptr);
  setNextSibling(nullptr);
  setParentNode(nullptr);
}

//===----------------------------------------------------------------------===//
// Walk Methods
//===----------------------------------------------------------------------===//

void OperationTreeNode::preOrderWalk(OperationTreeNode* n,
                                     ActionFuncTy action) {
  assert(n && "expected valid node");
  (void)action(n);
  OperationTreeNode* child = n->getFirstChild();
  while (child) {
    preOrderWalk(child, action);
    child = child->getNextSibling();
  }
}

void OperationTreeNode::postOrderWalk(OperationTreeNode* n,
                                      ActionFuncTy action) {
  assert(n && "expected valid node");
  OperationTreeNode* child = n->getFirstChild();
  while (child) {
    postOrderWalk(child, action);
    child = child->getNextSibling();
  }
  (void)action(n);
}

OperationTreeNode* OperationTreeNode::preOrderGuidedWalk(OperationTreeNode* n,
                                                         ActionFuncTy action) {
  assert(n && "expected valid node");
  OperationTreeNode* next_node = action(n);
  if (next_node)
    assert(n->getParentNode() == next_node->getParentNode() &&
           "make sure action returns a true sibling");
  OperationTreeNode* child = n->getFirstChild();
  while (child) {
    child = preOrderGuidedWalk(child, action);
  }
  return next_node;
}

OperationTreeNode* OperationTreeNode::postOrderGuidedWalk(OperationTreeNode* n,
                                                          ActionFuncTy action) {
  assert(n && "expected valid node");
  OperationTreeNode* child = n->getFirstChild();
  while (child) {
    child = postOrderGuidedWalk(child, action);
  }
  OperationTreeNode* next_node = action(n);
  if (next_node)
    assert(n->getParentNode() == next_node->getParentNode() &&
           "make sure action returns a true sibling");
  return next_node;
}

void OperationTreeNode::breadthFirstWalk(OperationTreeNode* n,
                                         ActionFuncTy action) {
  assert(n && "expected valid node");
  std::queue<OperationTreeNode*> queue;
  queue.push(n);
  while (!queue.empty()) {
    OperationTreeNode* curr_node = queue.front();
    assert(curr_node && "expected valid node");
    (void)action(curr_node);
    queue.pop();
    OperationTreeNode* child = curr_node->getFirstChild();
    while (child) {
      queue.push(child);
      child = child->getNextSibling();
    }
  }
}

void OperationTreeNode::reverseBreadthFirstWalk(OperationTreeNode* n,
                                                ActionFuncTy action,
                                                bool keep_order) {
  assert(n && "expected valid node");
  llvm::SmallVector<OperationTreeNode*> stack;
  std::queue<OperationTreeNode*> queue;
  queue.push(n);
  while (!queue.empty()) {
    OperationTreeNode* curr_node = queue.front();
    assert(curr_node && "expected valid node");
    stack.push_back(curr_node);
    queue.pop();
    OperationTreeNode* child = curr_node->getFirstChild();
    if (!keep_order) {
      while (child) {
        queue.push(child);
        child = child->getNextSibling();
      }
    } else {
      // Add sibilings to the queue in reverse order to counter the reversal
      // effect of the main stack.
      llvm::SmallVector<OperationTreeNode*> s;
      while (child) {
        s.push_back(child);
        child = child->getNextSibling();
      }
      while (!s.empty()) {
        queue.push(s.back());
        s.pop_back();
      }
    }
  }

  while (!stack.empty()) {
    (void)action(stack.back());
    stack.pop_back();
  }
}

//===----------------------------------------------------------------------===//
// Template Specializations
//===----------------------------------------------------------------------===//

/// Caller must ensure action does not remove \\p n itself or any parent or
/// ancestor nodes. The action can add or remove sibling or children of \\p n,
/// but new children won't be further visited.

template <>
void OperationTreeNode::walk<OperationTreeNode::kPreOrder>(
    OperationTreeNode* n, ActionFuncTy action) {
  preOrderWalk(n, action);
}

template <>
void OperationTreeNode::walk<OperationTreeNode::kPostOrder>(
    OperationTreeNode* n, ActionFuncTy action) {
  postOrderWalk(n, action);
}

template <>
void OperationTreeNode::walk<OperationTreeNode::kPreOrderGuided>(
    OperationTreeNode* n, ActionFuncTy action) {
  preOrderGuidedWalk(n, action);
}

template <>
void OperationTreeNode::walk<OperationTreeNode::kPostOrderGuided>(
    OperationTreeNode* n, ActionFuncTy action) {
  postOrderGuidedWalk(n, action);
}

template <>
void OperationTreeNode::walk<OperationTreeNode::kBFS>(OperationTreeNode* n,
                                                      ActionFuncTy action) {
  breadthFirstWalk(n, action);
}

template <>
void OperationTreeNode::walk<OperationTreeNode::kReverseBFS>(
    OperationTreeNode* n, ActionFuncTy action) {
  reverseBreadthFirstWalk(n, action, false);
}

template <>
void OperationTreeNode::walk<OperationTreeNode::kKeepOrderRBFS>(
    OperationTreeNode* n, ActionFuncTy action) {
  reverseBreadthFirstWalk(n, action, true);
}

void OperationTree::clear() {
  if (!empty()) {
    llvm::SmallVector<OperationTreeNode*> to_be_deleted;
    OperationTreeNode::walk<OperationTreeNode::WalkOrder::kPostOrder>(
        getRoot(), [&](OperationTreeNode* n) {
          to_be_deleted.push_back(n);
          return nullptr;
        });
    for (OperationTreeNode* n : to_be_deleted) delete n;
  } else if (root_)
    delete root_;
  root_ = nullptr;
}

void OperationTree::clear(OperationTreeNode* start) {
  assert(start && "expected valid node");
  if (root_ && start == root_) clear();
  start->unlink();
  llvm::SmallVector<OperationTreeNode*> to_be_deleted;
  OperationTreeNode::walk<OperationTreeNode::WalkOrder::kPostOrder>(
      start, [&](OperationTreeNode* n) {
        to_be_deleted.push_back(n);
        return nullptr;
      });
  for (OperationTreeNode* n : to_be_deleted) delete n;
}