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
// PipelineTreeNode Implementation
//===----------------------------------------------------------------------===//

void PipelineTreeNode::printNodeHeader(raw_ostream& os) const {
  // Print indentation based on depth
  unsigned depth = getDepth();
  for (unsigned i = 0; i < depth; ++i) {
    os << "  ";
  }

  // Print node name
  os << getNodeName();

  // Print materialization status and location
  if (!isMaterialized()) {
    os << " [unmaterialized]";
    if (template_op_) {
      os << " - template op ";
      scheduler::utils::printLocation(os, template_op_);
    }
  } else if (operation_op_) {
    os << " ";
    scheduler::utils::printLocation(os, operation_op_);
  }
}

void PipelineTreeNode::printChildren(raw_ostream& os) const {
  OperationTreeNode* child = getFirstChild();
  while (child) {
    static_cast<PipelineTreeNode*>(child)->print(os);
    child = child->getNextSibling();
  }
}

//===----------------------------------------------------------------------===//
// StageNode Implementation
//===----------------------------------------------------------------------===//

void StageNode::print(raw_ostream& os) const {
  constexpr bool kShowDependencies = false;
  printNodeHeader(os);

  if (kShowDependencies && !dependencies_.empty()) {
    os << " [depends on: ";
    for (size_t i = 0; i < dependencies_.size(); ++i) {
      if (i > 0) os << ", ";
      os << "Stage " << dependencies_[i]->getStageId();
    }
    os << "]";
  }

  os << "\n";
  printChildren(os);
}

//===----------------------------------------------------------------------===//
// PrivateNode Implementation
//===----------------------------------------------------------------------===//

void PrivateNode::print(raw_ostream& os) const {
  printNodeHeader(os);
  os << "\n";
  printChildren(os);
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
  // Create root node if not present already.
  root_ = root_ ? root_ : new RootNode();

  // Map to track stage IDs
  int next_stage_id = 0;

  // Use MLIR's walk to traverse all operations in pre-order
  unit.walk<WalkOrder::PreOrder>([&](Operation* op) {
    if (!isOperationSelected(*op)) return;

    // Create appropriate node type based on operation using factory methods
    PipelineTreeNode* node = nullptr;
    if (auto forOp = dyn_cast<scf::ForOp>(op)) {
      node = createLoopNode(op);
    } else if (auto pipelineOp = dyn_cast<PipelineOp>(op)) {
      node = createPipelineNode(op);
    } else if (auto stageOp = dyn_cast<StageOp>(op)) {
      node = createStageNode(op, next_stage_id++);
    } else if (auto privateOp = dyn_cast<PrivateOp>(op)) {
      node = createPrivateNode(op);
    }
    if (!node) return;

    // Find parent: either another selected operation or root
    OperationTreeNode* parent = root_;
    Operation* curr_op = op;
    while ((curr_op = curr_op->getParentOp())) {
      if (!isOperationSelected(*curr_op)) continue;
      auto parent_it = opToNode_.find(curr_op);
      assert(parent_it != opToNode_.end() &&
             "expected parent to be in the map (due to pre-order traversal)");
      parent = parent_it->getSecond();
      break;
    }
    parent->insertChildNode(node);
  });

  // TODO: Build stage dependencies by analyzing token flow
  // This would require analyzing ConsumeTokenOp and ProduceTokenOp
  // to determine which stages depend on which other stages
}

//===----------------------------------------------------------------------===//
// PipelineTreeNode Helper Methods
//===----------------------------------------------------------------------===//

const PipelineTreeNode* PipelineTreeNode::findOutermostLoop(
    const PipelineTreeNode* start_node,
    std::function<bool(const PipelineTreeNode*)> stop) {
  if (!start_node) return nullptr;

  const PipelineTreeNode* outermost_loop = nullptr;
  const PipelineTreeNode* current = start_node;

  while (current && !stop(current)) {
    if (current->isLoopNode()) {
      outermost_loop = current;
    }
    current = static_cast<const PipelineTreeNode*>(current->getParentNode());
  }

  return outermost_loop;
}

const PipelineTreeNode* PipelineTreeNode::findInnermostLoop(
    const PipelineTreeNode* start_node) {
  if (!start_node || !start_node->isLoopNode()) return nullptr;

  const PipelineTreeNode* innermost_loop = start_node;

  while (true) {
    const PipelineTreeNode* child =
        static_cast<const PipelineTreeNode*>(innermost_loop->getFirstChild());
    if (!child || !child->isLoopNode()) break;

    innermost_loop = child;
  }

  return innermost_loop;
}
