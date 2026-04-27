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
// Pipeline Tree Implementation
//===----------------------------------------------------------------------===//

#include "src/Transform/PipelineTree.hpp"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "src/Dialects/KTDF/KTDFOps.hpp"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"
#include <queue>

using namespace mlir;
using namespace mlir::ktdf;

//===----------------------------------------------------------------------===//
// PipelineTreeNode Implementation
//===----------------------------------------------------------------------===//

unsigned PipelineTreeNode::getNumberOfChildren() const {
  unsigned count = 0;
  PipelineTreeNode *child = getFirstChild();
  while (child) {
    ++count;
    child = child->getNextSibling();
  }
  return count;
}

PipelineTreeNode *PipelineTreeNode::getLastChild() const {
  PipelineTreeNode *child = getFirstChild();
  if (!child) return nullptr;
  while (child->getNextSibling()) {
    child = child->getNextSibling();
  }
  return child;
}

PipelineTreeNode *PipelineTreeNode::getPrevSibling() const {
  if (!parent_node_) return nullptr;
  PipelineTreeNode *prev = nullptr;
  PipelineTreeNode *curr = parent_node_->getFirstChild();
  while (curr && curr != this) {
    prev = curr;
    curr = curr->getNextSibling();
  }
  return prev;
}

void PipelineTreeNode::insertChildNode(PipelineTreeNode *child,
                                        PipelineTreeNode *pos) {
  assert(child && "Cannot insert null child");
  child->setParentNode(this);

  if (!pos) {
    // Insert at the end
    PipelineTreeNode *last = getLastChild();
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

void PipelineTreeNode::print(raw_ostream &OS, bool showDependencies) const {
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
  PipelineTreeNode *child = getFirstChild();
  while (child) {
    child->print(OS, showDependencies);
    child = child->getNextSibling();
  }
}

//===----------------------------------------------------------------------===//
// Walk Methods
//===----------------------------------------------------------------------===//

void PipelineTreeNode::preOrderWalk(PipelineTreeNode *n, ActionFuncTy action) {
  if (!n) return;
  action(n);
  PipelineTreeNode *child = n->getFirstChild();
  while (child) {
    preOrderWalk(child, action);
    child = child->getNextSibling();
  }
}

void PipelineTreeNode::postOrderWalk(PipelineTreeNode *n, ActionFuncTy action) {
  if (!n) return;
  PipelineTreeNode *child = n->getFirstChild();
  while (child) {
    postOrderWalk(child, action);
    child = child->getNextSibling();
  }
  action(n);
}

PipelineTreeNode *PipelineTreeNode::preOrderGuidedWalk(PipelineTreeNode *n,
                                                        ActionFuncTy action) {
  if (!n) return nullptr;
  PipelineTreeNode *result = action(n);
  if (result) return result;
  
  PipelineTreeNode *child = n->getFirstChild();
  while (child) {
    result = preOrderGuidedWalk(child, action);
    if (result) return result;
    child = child->getNextSibling();
  }
  return nullptr;
}

PipelineTreeNode *PipelineTreeNode::postOrderGuidedWalk(PipelineTreeNode *n,
                                                         ActionFuncTy action) {
  if (!n) return nullptr;
  
  PipelineTreeNode *child = n->getFirstChild();
  while (child) {
    PipelineTreeNode *result = postOrderGuidedWalk(child, action);
    if (result) return result;
    child = child->getNextSibling();
  }
  
  return action(n);
}

void PipelineTreeNode::breadthFirstWalk(PipelineTreeNode *n,
                                         ActionFuncTy action) {
  if (!n) return;
  
  std::queue<PipelineTreeNode *> queue;
  queue.push(n);
  
  while (!queue.empty()) {
    PipelineTreeNode *curr = queue.front();
    queue.pop();
    action(curr);
    
    PipelineTreeNode *child = curr->getFirstChild();
    while (child) {
      queue.push(child);
      child = child->getNextSibling();
    }
  }
}

void PipelineTreeNode::reverseBreadthFirstWalk(PipelineTreeNode *n,
                                                ActionFuncTy action,
                                                bool keep_order) {
  if (!n) return;
  
  std::queue<PipelineTreeNode *> queue;
  SmallVector<PipelineTreeNode *> nodes;
  
  queue.push(n);
  
  while (!queue.empty()) {
    PipelineTreeNode *curr = queue.front();
    queue.pop();
    nodes.push_back(curr);
    
    PipelineTreeNode *child = curr->getFirstChild();
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

template <>
void PipelineTreeNode::walk<PipelineTreeNode::kPreOrder>(
    PipelineTreeNode *n, ActionFuncTy action) {
  preOrderWalk(n, action);
}

template <>
void PipelineTreeNode::walk<PipelineTreeNode::kPostOrder>(
    PipelineTreeNode *n, ActionFuncTy action) {
  postOrderWalk(n, action);
}

template <>
void PipelineTreeNode::walk<PipelineTreeNode::kPreOrderGuided>(
    PipelineTreeNode *n, ActionFuncTy action) {
  preOrderGuidedWalk(n, action);
}

template <>
void PipelineTreeNode::walk<PipelineTreeNode::kPostOrderGuided>(
    PipelineTreeNode *n, ActionFuncTy action) {
  postOrderGuidedWalk(n, action);
}

template <>
void PipelineTreeNode::walk<PipelineTreeNode::kBFS>(PipelineTreeNode *n,
                                                     ActionFuncTy action) {
  breadthFirstWalk(n, action);
}

template <>
void PipelineTreeNode::walk<PipelineTreeNode::kReverseBFS>(
    PipelineTreeNode *n, ActionFuncTy action) {
  reverseBreadthFirstWalk(n, action, false);
}

template <>
void PipelineTreeNode::walk<PipelineTreeNode::kKeepOrderRBFS>(
    PipelineTreeNode *n, ActionFuncTy action) {
  reverseBreadthFirstWalk(n, action, true);
}

//===----------------------------------------------------------------------===//
// StageNode Implementation
//===----------------------------------------------------------------------===//

void StageNode::print(raw_ostream &OS, bool showDependencies) const {
  unsigned depth = getDepth();
  for (unsigned i = 0; i < depth; ++i) {
    OS << "  ";
  }
  OS << getNodeName();
  
  if (operation_op_) {
    auto loc = operation_op_->getLoc();
    if (auto fileLoc = dyn_cast<FileLineColLoc>(loc)) {
      OS << " @" << fileLoc.getLine();
    }
  }
  
  if (showDependencies && !dependencies_.empty()) {
    OS << " [depends on: ";
    for (size_t i = 0; i < dependencies_.size(); ++i) {
      if (i > 0) OS << ", ";
      OS << "Stage " << dependencies_[i]->getStageId();
    }
    OS << "]";
  }
  
  OS << "\n";

  // Print children
  PipelineTreeNode *child = getFirstChild();
  while (child) {
    child->print(OS, showDependencies);
    child = child->getNextSibling();
  }
}

//===----------------------------------------------------------------------===//
// PrivateNode Implementation
//===----------------------------------------------------------------------===//

void PrivateNode::print(raw_ostream &OS, bool showDependencies) const {
  unsigned depth = getDepth();
  for (unsigned i = 0; i < depth; ++i) {
    OS << "  ";
  }
  OS << getNodeName();
  
  if (operation_op_) {
    auto loc = operation_op_->getLoc();
    if (auto fileLoc = dyn_cast<FileLineColLoc>(loc)) {
      OS << " @" << fileLoc.getLine();
    }
  }
  
  if (!expanded_buffers_.empty()) {
    OS << " [expanded buffers: " << expanded_buffers_.size() << "]";
  }
  
  OS << "\n";

  // Print children
  PipelineTreeNode *child = getFirstChild();
  while (child) {
    child->print(OS, showDependencies);
    child = child->getNextSibling();
  }
}

//===----------------------------------------------------------------------===//
// PipelineNode Implementation
//===----------------------------------------------------------------------===//

PrivateNode *PipelineNode::getPrivateNode() const {
  PipelineTreeNode *child = getFirstChild();
  while (child) {
    if (child->isPrivateNode()) {
      return static_cast<PrivateNode*>(child);
    }
    child = child->getNextSibling();
  }
  return nullptr;
}

SmallVector<StageNode *> PipelineNode::getStages() const {
  SmallVector<StageNode *> stages;
  PipelineTreeNode *child = getFirstChild();
  while (child) {
    if (child->isStageNode()) {
      stages.push_back(static_cast<StageNode*>(child));
    }
    child = child->getNextSibling();
  }
  return stages;
}

//===----------------------------------------------------------------------===//
// PipelineTree Implementation
//===----------------------------------------------------------------------===//

void PipelineTree::clear() {
  if (!root_) return;
  clear(root_->getFirstChild());
  delete root_;
  root_ = nullptr;
}

void PipelineTree::clear(PipelineTreeNode *start) {
  if (!start) return;
  
  // Post-order deletion to avoid dangling pointers
  PipelineTreeNode::walk<PipelineTreeNode::kPostOrder>(
      start, [](PipelineTreeNode *n) {
        delete n;
        return n;
      });
}

bool PipelineTree::isOperationSelected(const Operation &op) const {
  // Select operations that are relevant to the pipeline structure
  return isa<scf::ForOp, PipelineOp, StageOp, PrivateOp>(op);
}

void PipelineTree::compute(Operation &unit) {
  // Create root node
  root_ = new RootNode();
  
  // Map to track operation to node mapping for dependency tracking
  DenseMap<Operation *, PipelineTreeNode *> opToNode;
  
  // Map to track stage IDs
  DenseMap<Operation *, int> stageIds;
  int nextStageId = 0;
  
  // Helper function to recursively build the tree
  std::function<void(Operation *, PipelineTreeNode *)> buildTree;
  buildTree = [&](Operation *op, PipelineTreeNode *parent) {
    if (!op) return;
    
    PipelineTreeNode *node = nullptr;
    
    if (auto forOp = dyn_cast<scf::ForOp>(op)) {
      // Create loop node
      node = new LoopNode(op);
      parent->insertChildNode(node);
      opToNode[op] = node;
      
      // Process body
      for (auto &bodyOp : forOp.getBody()->getOperations()) {
        buildTree(&bodyOp, node);
      }
    } else if (auto pipelineOp = dyn_cast<PipelineOp>(op)) {
      // Create pipeline node
      node = new PipelineNode(op);
      parent->insertChildNode(node);
      opToNode[op] = node;
      
      // Process body - iterate through blocks and their operations
      for (auto &block : pipelineOp.getBody().getBlocks()) {
        for (auto &bodyOp : block.getOperations()) {
          buildTree(&bodyOp, node);
        }
      }
    } else if (auto stageOp = dyn_cast<StageOp>(op)) {
      // Create stage node
      int stageId = nextStageId++;
      stageIds[op] = stageId;
      auto *stageNode = new StageNode(op, stageId);
      node = stageNode;
      parent->insertChildNode(node);
      opToNode[op] = node;
      
      // Process body - iterate through blocks and their operations
      for (auto &block : stageOp.getBody().getBlocks()) {
        for (auto &bodyOp : block.getOperations()) {
          buildTree(&bodyOp, node);
        }
      }
    } else if (auto privateOp = dyn_cast<PrivateOp>(op)) {
      // Create private node
      node = new PrivateNode(op);
      parent->insertChildNode(node);
      opToNode[op] = node;
      
      // Process body - iterate through blocks and their operations
      for (auto &block : privateOp.getBody().getBlocks()) {
        for (auto &bodyOp : block.getOperations()) {
          buildTree(&bodyOp, node);
        }
      }
    } else if (isOperationSelected(*op)) {
      // For other selected operations, continue recursively
      for (auto &region : op->getRegions()) {
        for (auto &block : region.getBlocks()) {
          for (auto &bodyOp : block.getOperations()) {
            buildTree(&bodyOp, parent);
          }
        }
      }
    }
  };
  
  // Start building from the unit operation
  for (auto &region : unit.getRegions()) {
    for (auto &block : region.getBlocks()) {
      for (auto &op : block.getOperations()) {
        buildTree(&op, root_);
      }
    }
  }
  
  // TODO: Build stage dependencies by analyzing token flow
  // This would require analyzing ConsumeTokenOp and ProduceTokenOp
  // to determine which stages depend on which other stages
}

//===----------------------------------------------------------------------===//
// Template Instantiations
//===----------------------------------------------------------------------===//

// Explicit template instantiations for walk methods
template void PipelineTreeNode::walk<PipelineTreeNode::kPreOrder>(
    PipelineTreeNode *, ActionFuncTy);
template void PipelineTreeNode::walk<PipelineTreeNode::kPostOrder>(
    PipelineTreeNode *, ActionFuncTy);
template void PipelineTreeNode::walk<PipelineTreeNode::kPreOrderGuided>(
    PipelineTreeNode *, ActionFuncTy);
template void PipelineTreeNode::walk<PipelineTreeNode::kPostOrderGuided>(
    PipelineTreeNode *, ActionFuncTy);
template void PipelineTreeNode::walk<PipelineTreeNode::kBFS>(PipelineTreeNode *,
                                                              ActionFuncTy);
template void PipelineTreeNode::walk<PipelineTreeNode::kReverseBFS>(
    PipelineTreeNode *, ActionFuncTy);
template void PipelineTreeNode::walk<PipelineTreeNode::kKeepOrderRBFS>(
    PipelineTreeNode *, ActionFuncTy);

// Made with Bob
