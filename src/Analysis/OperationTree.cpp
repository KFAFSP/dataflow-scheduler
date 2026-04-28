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
// Operation Tree Node Implementation
//===----------------------------------------------------------------------===//

#include "OperationTree.hpp"

#include <cassert>
#include <queue>

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/BuiltinOps.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// OperationTreeNode Implementation
//===----------------------------------------------------------------------===//

unsigned OperationTreeNode::getNumberOfChildren() const {
  unsigned count = 0;
  OperationTreeNode* child = getFirstChild();
  while (child) {
    ++count;
    child = child->getNextSibling();
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
  if (!parent_node_) return nullptr;
  OperationTreeNode* prev = nullptr;
  OperationTreeNode* curr = parent_node_->getFirstChild();
  while (curr && curr != this) {
    prev = curr;
    curr = curr->getNextSibling();
  }
  return prev;
}

void OperationTreeNode::insertChildNode(OperationTreeNode* child,
                                        OperationTreeNode* pos) {
  assert(child && "Cannot insert null child");
  child->setParentNode(this);

  if (!pos) {
    // Insert at the end
    OperationTreeNode* last = getLastChild();
    if (last) {
      last->setNextSibling(child);
    } else {
      setFirstChild(child);
    }
  } else {
    // Insert after pos
    child->setNextSibling(pos->getNextSibling());
    pos->setNextSibling(child);
  }
}

void OperationTreeNode::print(raw_ostream& OS) const {
  unsigned depth = getDepth();
  for (unsigned i = 0; i < depth; ++i) {
    OS << "  ";
  }
  OS << getNodeName();

  if (operation_op_) {
    // Print location information if available
    auto loc = operation_op_->getLoc();
    if (auto fileLoc = dyn_cast<FileLineColLoc>(loc)) {
      OS << " @" << fileLoc.getLine();
    }
  }
  OS << "\n";

  // Print children
  OperationTreeNode* child = getFirstChild();
  while (child) {
    child->print(OS);
    child = child->getNextSibling();
  }
}

//===----------------------------------------------------------------------===//
// Walk Methods
//===----------------------------------------------------------------------===//

void OperationTreeNode::preOrderWalk(OperationTreeNode* n,
                                     ActionFuncTy action) {
  if (!n) return;
  action(n);
  OperationTreeNode* child = n->getFirstChild();
  while (child) {
    preOrderWalk(child, action);
    child = child->getNextSibling();
  }
}

void OperationTreeNode::postOrderWalk(OperationTreeNode* n,
                                      ActionFuncTy action) {
  if (!n) return;
  OperationTreeNode* child = n->getFirstChild();
  while (child) {
    postOrderWalk(child, action);
    child = child->getNextSibling();
  }
  action(n);
}

OperationTreeNode* OperationTreeNode::preOrderGuidedWalk(OperationTreeNode* n,
                                                         ActionFuncTy action) {
  if (!n) return nullptr;
  OperationTreeNode* result = action(n);
  if (result) return result;

  OperationTreeNode* child = n->getFirstChild();
  while (child) {
    result = preOrderGuidedWalk(child, action);
    if (result) return result;
    child = child->getNextSibling();
  }
  return nullptr;
}

OperationTreeNode* OperationTreeNode::postOrderGuidedWalk(OperationTreeNode* n,
                                                          ActionFuncTy action) {
  if (!n) return nullptr;

  OperationTreeNode* child = n->getFirstChild();
  while (child) {
    OperationTreeNode* result = postOrderGuidedWalk(child, action);
    if (result) return result;
    child = child->getNextSibling();
  }

  return action(n);
}

void OperationTreeNode::breadthFirstWalk(OperationTreeNode* n,
                                         ActionFuncTy action) {
  if (!n) return;

  std::queue<OperationTreeNode*> queue;
  queue.push(n);

  while (!queue.empty()) {
    OperationTreeNode* curr = queue.front();
    queue.pop();
    action(curr);

    OperationTreeNode* child = curr->getFirstChild();
    while (child) {
      queue.push(child);
      child = child->getNextSibling();
    }
  }
}

void OperationTreeNode::reverseBreadthFirstWalk(OperationTreeNode* n,
                                                ActionFuncTy action,
                                                bool keep_order) {
  if (!n) return;

  std::queue<OperationTreeNode*> queue;
  SmallVector<OperationTreeNode*> nodes;

  queue.push(n);

  while (!queue.empty()) {
    OperationTreeNode* curr = queue.front();
    queue.pop();
    nodes.push_back(curr);

    OperationTreeNode* child = curr->getFirstChild();
    while (child) {
      queue.push(child);
      child = child->getNextSibling();
    }
  }

  // Process in reverse order
  for (auto it = nodes.rbegin(); it != nodes.rend(); ++it) {
    action(*it);
  }
}

//===----------------------------------------------------------------------===//
// Template Specializations
//===----------------------------------------------------------------------===//

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

//===----------------------------------------------------------------------===//
// OperationTree Implementation
//===----------------------------------------------------------------------===//

void OperationTree::clear() {
  if (!root_) return;
  clear(root_->getFirstChild());
  delete root_;
  root_ = nullptr;
}

void OperationTree::clear(OperationTreeNode* start) {
  if (!start) return;

  // Post-order deletion to avoid dangling pointers
  OperationTreeNode::walk<OperationTreeNode::kPostOrder>(
      start, [](OperationTreeNode* n) {
        delete n;
        return n;
      });
}
