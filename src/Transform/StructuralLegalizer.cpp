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
// Structural Legalizer Implementation
//===----------------------------------------------------------------------===//

#include "StructuralLegalizer.hpp"

#include "llvm/Support/raw_ostream.h"
#include "src/Analysis/PipelineTree.hpp"

namespace scheduler {
namespace structural_legality {

//===----------------------------------------------------------------------===//
// Rule Implementations
//===----------------------------------------------------------------------===//

bool PipelineChildrenRule::isViolated(PipelineTreeNode* node,
                                      PipelineTree& tree) const {
  // Check if this is a pipeline with illegal children
  if (!node->isPipelineNode()) {
    return false;
  }
  
  OperationTreeNode* child = node->getFirstChild();
  while (child) {
    PipelineTreeNode* child_pnode = static_cast<PipelineTreeNode*>(child);
    if (!child_pnode->isPrivateNode() && !child_pnode->isStageNode()) {
      return true;
    }
    child = child->getNextSibling();
  }
  
  return false;
}

bool PipelineChildrenRule::fixViolation(PipelineTreeNode* violating_node,
                                        PipelineTree& tree,
                                        int& next_stage_id) {
  llvm::outs() << "      Creating stage wrapper for "
               << violating_node->getNodeName() << "\n";
  
  // Create a new stage
  StageNode* new_stage =
      tree.createStageNode(nullptr, next_stage_id++);
  
  // Get the pipeline parent
  OperationTreeNode* pipeline_parent = violating_node->getParentNode();
  assert(pipeline_parent && "Violating node should have a parent");
  
  // Unlink the violating node from pipeline
  violating_node->unlink();
  
  // Insert coarsened stage as child of pipeline
  pipeline_parent->insertChildNode(new_stage);
  
  // Insert violating node as child of coarsened stage
  new_stage->insertChildNode(violating_node);
  
  llvm::outs() << "      Created coarsened stage "
               << (next_stage_id - 1) << "\n";
  return true;
}

void PipelineChildrenRule::print(llvm::raw_ostream& os) const {
  os << "Immediate children of pipeline must be stage or private";
}

bool StageParentRule::isViolated(PipelineTreeNode* node,
                                 PipelineTree& tree) const {
  // Check if this is a stage with non-pipeline parent
  if (!node->isStageNode()) {
    return false;
  }
  
  OperationTreeNode* parent = node->getParentNode();
  if (!parent) {
    return false;
  }
  
  PipelineTreeNode* parent_pnode = static_cast<PipelineTreeNode*>(parent);
  return !parent_pnode->isPipelineNode();
}

bool StageParentRule::fixViolation(PipelineTreeNode* violating_node,
                                   PipelineTree& tree,
                                   int& next_coarsened_stage_id) {
  llvm::outs() << "      Creating nested pipeline for stage "
               << violating_node->getNodeName() << "\n";
  
  assert(violating_node->isStageNode() && "Expected stage node");
  
  // Get the non-pipeline parent
  OperationTreeNode* parent = violating_node->getParentNode();
  assert(parent && "Stage should have a parent");
  PipelineTreeNode* parent_pnode = static_cast<PipelineTreeNode*>(parent);
  assert(!parent_pnode->isPipelineNode() && "Parent should not be a pipeline since expected violation");
  
  // Create a new nested pipeline node (unmaterialized)
  PipelineNode* nested_pipeline = tree.createPipelineNode(nullptr);
  
  // Get the template from the nearest ancestor pipeline
  OperationTreeNode* ancestor = parent_pnode->getParentNode();
  while (ancestor) {
    PipelineTreeNode* ancestor_pnode = static_cast<PipelineTreeNode*>(ancestor);
    if (ancestor_pnode->isPipelineNode()) {
      PipelineNode* ancestor_pipeline = static_cast<PipelineNode*>(ancestor_pnode);
      if (ancestor_pipeline->getOperation()) {
        nested_pipeline->setTemplateOp(ancestor_pipeline->getOperation());
      } else {
        assert(ancestor_pipeline->getTemplateOp());
        nested_pipeline->setTemplateOp(ancestor_pipeline->getTemplateOp());
      }
      llvm::outs() << "      Set template from ancestor pipeline\n";
      break;
    }
    ancestor = ancestor->getParentNode();
  }
  
  llvm::outs() << "      Created nested pipeline node\n";
  
  // Collect all stage siblings that should move to the nested pipeline
  // (all stages that are children of the same non-pipeline parent)
  llvm::SmallVector<StageNode*> stages_to_move;
  OperationTreeNode* child = parent_pnode->getFirstChild();
  while (child) {
    PipelineTreeNode* child_pnode = static_cast<PipelineTreeNode*>(child);
    if (child_pnode->isStageNode()) {
      stages_to_move.push_back(static_cast<StageNode*>(child_pnode));
    }
    child = child->getNextSibling();
  }
  
  llvm::outs() << "      Moving " << stages_to_move.size()
               << " stage(s) to nested pipeline\n";
  
  // Insert the nested pipeline as a child of the parent
  // Get the position before the first stage
  OperationTreeNode* insert_before = nullptr;
  if (!stages_to_move.empty()) {
    insert_before = stages_to_move[0]->getPrevSibling();
  }
  
  parent_pnode->insertChildNode(nested_pipeline, insert_before);
  
  // Move all stages to be children of the nested pipeline
  for (StageNode* stage : stages_to_move) {
    stage->unlink();
    nested_pipeline->insertChildNode(stage);
  }
  
  // TODO: Privatize buffers for nested pipeline
  // This requires:
  // 1. Identify buffers used by stages in the nested pipeline
  // 2. Create a new PrivateNode for the nested pipeline
  // 3. Move or duplicate buffer allocations
  // 4. Update data transfer operations
  
  // TODO: Create new tokens for nested pipeline
  // This requires:
  // 1. Analyze stage dependencies within the nested pipeline
  // 2. Create token operations for the nested pipeline
  // 3. Update stage operations to use new tokens
  
  llvm::outs() << "      Nested pipeline created successfully\n";
  llvm::outs() << "      TODO: Buffer privatization and token creation not yet implemented\n";
  
  return true;
}

void StageParentRule::print(llvm::raw_ostream& os) const {
  os << "Stage must have pipeline as immediate parent";
}

//===----------------------------------------------------------------------===//
// Legalizer Implementation
//===----------------------------------------------------------------------===//

Legalizer::Legalizer() : next_coarsened_stage_id_(1000) {
  // Initialize all rules
  rules_.push_back(std::make_unique<PipelineChildrenRule>());
  rules_.push_back(std::make_unique<StageParentRule>());
}

const llvm::SmallVector<Violation>& Legalizer::identifyViolations(PipelineTree& tree) {
  violations_.clear();
  collectViolations(tree);
  
  return violations_;
}

void Legalizer::correctViolations(PipelineTree& tree) {
  for (auto& violation : violations_) {
    llvm::outs() << "    Correcting: ";
    violation.print(llvm::outs());
    llvm::outs() << "\n";
    
    // Let the rule fix its own violation
    violation.rule->fixViolation(violation.violating_node, tree,
                                 next_coarsened_stage_id_);
  }
  violations_.clear();
}

void Legalizer::collectViolations(PipelineTree& tree) {
  OperationTreeNode::walk<OperationTreeNode::kPreOrder>(
      tree.getRoot(), [&](OperationTreeNode* node) {
        PipelineTreeNode* pnode = static_cast<PipelineTreeNode*>(node);

        // Check all rules against this node
        for (const auto& rule : rules_) {
          if (rule->isViolated(pnode, tree)) {
            // Clone the rule for this violation
            std::unique_ptr<Rule> rule_copy;
            if (llvm::isa<PipelineChildrenRule>(rule.get())) {
              rule_copy = std::make_unique<PipelineChildrenRule>();
            } else if (llvm::isa<StageParentRule>(rule.get())) {
              rule_copy = std::make_unique<StageParentRule>();
            }
            
            violations_.emplace_back(std::move(rule_copy), pnode);
          }
        }

        return node;
      });
}

}  // namespace structural_legality
}  // namespace scheduler

// Made with Bob
