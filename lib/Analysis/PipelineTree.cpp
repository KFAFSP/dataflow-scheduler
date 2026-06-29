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
// Pipeline Tree Implementation
//
//===----------------------------------------------------------------------===//

#include "dataflow-scheduler/Analysis/PipelineTree.h"

#include <queue>

#include "dataflow-scheduler/Analysis/Utils.h"
#include "dataflow-scheduler/Dialect/KTDF/KTDF.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

using namespace scheduler;

//===----------------------------------------------------------------------===//
// PipelineTreeNode Implementation
//===----------------------------------------------------------------------===//

void PipelineTreeNode::printNodeHeader(mlir::raw_ostream& os) const {
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
      printLocation(os, template_op_);
    }
  } else if (operation_op_) {
    os << " ";
    printLocation(os, operation_op_);
  }
}

void PipelineTreeNode::printChildren(mlir::raw_ostream& os) const {
  OperationTreeNode* child = getFirstChild();
  while (child) {
    static_cast<PipelineTreeNode*>(child)->print(os);
    child = child->getNextSibling();
  }
}

//===----------------------------------------------------------------------===//
// StageNode Implementation
//===----------------------------------------------------------------------===//

void StageNode::print(mlir::raw_ostream& os) const {
  constexpr bool kShowDependencies = true;
  printNodeHeader(os);

  if (kShowDependencies && !dependencies_.empty()) {
    os << " [unblocks: ";
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

void PrivateNode::print(mlir::raw_ostream& os) const {
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

llvm::SmallVector<StageNode*> PipelineNode::getStages() const {
  llvm::SmallVector<StageNode*> stages;
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

bool PipelineTree::isOperationSelected(const mlir::Operation& op) const {
  // Select operations that are relevant to the pipeline structure
  return mlir::isa<mlir::scf::ForOp, mlir::ktdf::PipelineOp,
                   mlir::ktdf::StageOp, mlir::ktdf::PrivateOp>(op);
}

void PipelineTree::compute(mlir::Operation& unit) {
  // Create root node if not present already.
  root_ = root_ ? root_ : new RootNode();

  // Map to track stage IDs
  int next_stage_id = 0;

  // Use MLIR's walk to traverse all operations in pre-order
  unit.walk<mlir::WalkOrder::PreOrder>([&](mlir::Operation* op) {
    if (!isOperationSelected(*op)) return;

    // Create appropriate node type based on operation using factory methods
    PipelineTreeNode* node = nullptr;
    if (mlir::isa<mlir::scf::ForOp>(op)) {
      node = createLoopNode(op);
    } else if (mlir::isa<mlir::ktdf::PipelineOp>(op)) {
      node = createPipelineNode(op);
    } else if (mlir::isa<mlir::ktdf::StageOp>(op)) {
      node = createStageNode(op, next_stage_id++);
    } else if (mlir::isa<mlir::ktdf::PrivateOp>(op)) {
      node = createPrivateNode(op);
    }
    if (!node) return;

    if (op == &unit) {
      root_->insertChildNode(node);
    } else {
      // Find parent: either another selected operation or root
      OperationTreeNode* parent = root_;
      mlir::Operation* curr_op = op;
      while ((curr_op = curr_op->getParentOp())) {
        if (!isOperationSelected(*curr_op)) continue;
        if (curr_op == &unit) break;  // reached the top where we
        auto parent_it = opToNode_.find(curr_op);
        assert(parent_it != opToNode_.end() &&
               "expected parent to be in the map (due to pre-order traversal)");
        parent = parent_it->getSecond();
        break;
      }
      parent->insertChildNode(node);
    }
  });

  // Build stage dependencies by analyzing token flow
  // For each pipeline, analyze the token dependencies between stages
  unit.walk([&](mlir::ktdf::PipelineOp pipeline_op) {
    // Map from token values to the stages that produce them (depends_out)
    llvm::DenseMap<mlir::Value, StageNode*> token_producers;

    // First pass: collect all token producers
    for (auto stage_op : pipeline_op.getStages()) {
      StageNode* stage_node = static_cast<StageNode*>(getNodeForOp(stage_op));
      assert(stage_node);

      // Record this stage as a producer for all its output tokens
      for (mlir::Value token : stage_op.getDependsOut()) {
        token_producers[token] = stage_node;
      }
    }

    // Second pass: build dependencies by matching consumers to producers
    for (auto stage_op : pipeline_op.getStages()) {
      StageNode* consumer_stage =
          static_cast<StageNode*>(getNodeForOp(stage_op));
      assert(consumer_stage);

      // For each input token, find the producer stage and add this stage
      // as a dependency of that producer
      for (mlir::Value token : stage_op.getDependsIn()) {
        auto it = token_producers.find(token);
        if (it != token_producers.end()) {
          StageNode* producer_stage = it->second;
          // Producer stage adds consumer as a dependency
          producer_stage->addDependency(consumer_stage);
        }
      }
    }
  });
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

  // Track the deepest loop found and its depth
  const PipelineTreeNode* innermost_loop = start_node;
  unsigned max_depth = 0;

  // Helper function to recursively find the deepest loop
  std::function<void(const PipelineTreeNode*, unsigned)> findDeepestLoop =
      [&](const PipelineTreeNode* node, unsigned current_depth) {
        if (!node) return;

        // If this is a loop and it's deeper than what we've seen, update
        if (node->isLoopNode() && current_depth > max_depth) {
          max_depth = current_depth;
          innermost_loop = node;
        }

        // Recursively check all children
        const OperationTreeNode* child = node->getFirstChild();
        while (child) {
          const PipelineTreeNode* pipeline_child =
              static_cast<const PipelineTreeNode*>(child);
          unsigned next_depth =
              current_depth + (pipeline_child->isLoopNode() ? 1 : 0);
          findDeepestLoop(pipeline_child, next_depth);
          child = child->getNextSibling();
        }
      };

  // Start the search from start_node with depth 0
  findDeepestLoop(start_node, 0);

  return innermost_loop;
}

//===----------------------------------------------------------------------===//
// Stage DAG Analysis Methods
//===----------------------------------------------------------------------===//

llvm::FailureOr<llvm::SmallVector<StageNode*>>
PipelineTree::topologicalSortStages(PipelineNode* pipeline) const {
  if (!pipeline) return mlir::failure();

  llvm::SmallVector<StageNode*> stages = pipeline->getStages();
  if (stages.empty()) return llvm::SmallVector<StageNode*>();

  // Build in-degree map (count of incoming dependencies for each stage)
  llvm::DenseMap<StageNode*, int> in_degree;
  for (StageNode* stage : stages) {
    in_degree[stage] = 0;
  }

  // Count incoming edges by examining all stages' outgoing dependencies
  for (StageNode* stage : stages) {
    for (StageNode* dependent : stage->getDependencies()) {
      in_degree[dependent]++;
    }
  }

  // Initialize queue with all source nodes (in-degree == 0)
  std::queue<StageNode*> worklist;
  for (StageNode* stage : stages) {
    if (in_degree[stage] == 0) {
      worklist.push(stage);
    }
  }

  // Perform topological sort using Kahn's algorithm
  llvm::SmallVector<StageNode*> sorted;
  while (!worklist.empty()) {
    StageNode* current = worklist.front();
    worklist.pop();
    sorted.push_back(current);

    // Reduce in-degree for all dependents
    for (StageNode* dependent : current->getDependencies()) {
      in_degree[dependent]--;
      if (in_degree[dependent] == 0) {
        worklist.push(dependent);
      }
    }
  }

  // Check for cycles
  if (sorted.size() != stages.size()) {
    return mlir::failure();  // Cycle detected
  }

  return sorted;
}

llvm::SmallVector<StageNode*> PipelineTree::identifySourceStages(
    PipelineNode* pipeline) const {
  llvm::SmallVector<StageNode*> sources;
  if (!pipeline) return sources;

  llvm::SmallVector<StageNode*> stages = pipeline->getStages();

  // Build set of all stages that have incoming dependencies
  llvm::DenseSet<StageNode*> has_incoming;
  for (StageNode* stage : stages) {
    for (StageNode* dependent : stage->getDependencies()) {
      has_incoming.insert(dependent);
    }
  }

  // Source stages are those not in the has_incoming set
  for (StageNode* stage : stages) {
    if (!has_incoming.contains(stage)) {
      sources.push_back(stage);
    }
  }

  return sources;
}

llvm::SmallVector<StageNode*> PipelineTree::identifySinkStages(
    PipelineNode* pipeline) const {
  llvm::SmallVector<StageNode*> sinks;
  if (!pipeline) return sinks;

  llvm::SmallVector<StageNode*> stages = pipeline->getStages();

  // Sink stages are those with no outgoing dependencies
  for (StageNode* stage : stages) {
    if (stage->getDependencies().empty()) {
      sinks.push_back(stage);
    }
  }

  return sinks;
}
