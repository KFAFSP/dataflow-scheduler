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

#include "PipelineTree.hpp"

#include <queue>

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "src/Dialects/KTDF/KTDFOps.hpp"
#include "src/Utils/OperationUtils.hpp"

using namespace mlir;
using namespace mlir::ktdf;
using namespace llvm;

//===----------------------------------------------------------------------===//
// StageNode Implementation
//===----------------------------------------------------------------------===//

void StageNode::print(raw_ostream& OS, bool showDependencies) const {
  unsigned depth = getDepth();
  for (unsigned i = 0; i < depth; ++i) {
    OS << "  ";
  }
  OS << getNodeName();

  if (!isMaterialized()) {
    OS << " [unmaterialized]";
  } else if (operation_op_) {
    OS << " ";
    scheduler::utils::printLocation(OS, operation_op_);
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
  OperationTreeNode* child = getFirstChild();
  while (child) {
    static_cast<PipelineTreeNode*>(child)->print(OS, showDependencies);
    child = child->getNextSibling();
  }
}

//===----------------------------------------------------------------------===//
// PrivateNode Implementation
//===----------------------------------------------------------------------===//

void PrivateNode::print(raw_ostream& OS, bool showDependencies) const {
  unsigned depth = getDepth();
  for (unsigned i = 0; i < depth; ++i) {
    OS << "  ";
  }
  OS << getNodeName();

  if (!isMaterialized()) {
    OS << " [unmaterialized]";
  } else if (operation_op_) {
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
  OperationTreeNode* child = getFirstChild();
  while (child) {
    static_cast<PipelineTreeNode*>(child)->print(OS, showDependencies);
    child = child->getNextSibling();
  }
}

//===----------------------------------------------------------------------===//
// PipelineNode Implementation
//===----------------------------------------------------------------------===//

PrivateNode* PipelineNode::getPrivateNode() const {
  OperationTreeNode* child = getFirstChild();
  while (child) {
    if (static_cast<PipelineTreeNode*>(child)->isPrivateNode()) {
      return static_cast<PrivateNode*>(child);
    }
    child = child->getNextSibling();
  }
  return nullptr;
}

SmallVector<StageNode*> PipelineNode::getStages() const {
  SmallVector<StageNode*> stages;
  OperationTreeNode* child = getFirstChild();
  while (child) {
    if (static_cast<PipelineTreeNode*>(child)->isStageNode()) {
      stages.push_back(static_cast<StageNode*>(child));
    }
    child = child->getNextSibling();
  }
  return stages;
}

//===----------------------------------------------------------------------===//
// PipelineTree Implementation
//===----------------------------------------------------------------------===//
// clear() methods are now inherited from OperationTree

bool PipelineTree::isOperationSelected(const Operation& op) const {
  // Select operations that are relevant to the pipeline structure
  return isa<scf::ForOp, PipelineOp, StageOp, PrivateOp>(op);
}

void PipelineTree::compute(Operation& unit) {
  // Create root node
  root_ = new RootNode();

  // Clear the operation-to-node mapping
  opToNode_.clear();

  // Map to track stage IDs
  int nextStageId = 0;

  // Helper to create node and link to parent
  auto createAndLinkNode = [&](Operation* op, PipelineTreeNode* node) {
    opToNode_[op] = node;

    // Find parent: either another selected operation or root
    Operation* parentOp = op->getParentOp();
    while (parentOp && !opToNode_.count(parentOp)) {
      parentOp = parentOp->getParentOp();
    }

    PipelineTreeNode* parent =
        parentOp ? opToNode_[parentOp] : static_cast<PipelineTreeNode*>(root_);
    parent->insertChildNode(node);
  };

  // Use MLIR's walk to traverse all operations in pre-order
  unit.walk<WalkOrder::PreOrder>([&](Operation* op) {
    if (!isOperationSelected(*op)) return;

    PipelineTreeNode* node = nullptr;

    // Create appropriate node type based on operation
    if (auto forOp = dyn_cast<scf::ForOp>(op)) {
      node = new LoopNode(op);
    } else if (auto pipelineOp = dyn_cast<PipelineOp>(op)) {
      node = new PipelineNode(op);
    } else if (auto stageOp = dyn_cast<StageOp>(op)) {
      node = new StageNode(op, nextStageId++);
    } else if (auto privateOp = dyn_cast<PrivateOp>(op)) {
      node = new PrivateNode(op);
    }

    if (node) {
      createAndLinkNode(op, node);
    }
  });

  // TODO: Build stage dependencies by analyzing token flow
  // This would require analyzing ConsumeTokenOp and ProduceTokenOp
  // to determine which stages depend on which other stages
}
