//===-- PipelineTreeLegalizer.h ---------------------------------*- c++ -*-===//
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
// Pipeline Tree Legalizer
//
// This file defines a legalizer for discovering and correcting structure
// violations in pipeline trees.
//
//===----------------------------------------------------------------------===//

#ifndef DATAFLOW_SCHEDULER_TRANSFORMS_UTILS_PIPELINETREELEGALIZER_H_
#define DATAFLOW_SCHEDULER_TRANSFORMS_UTILS_PIPELINETREELEGALIZER_H_

#include <memory>

#include "dataflow-scheduler/Analysis/PipelineTree.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

namespace scheduler::pipeline_tree {

/// Rule kind enumeration for RTTI
enum class RuleKind {
  kPipelineChildrenMustBePrivateOrStage,
  kStageMustHavePipelineParent,
  kPrivateMustExistForPipelinesWithStageDeps
};

/// Base class for structure legality rules
class Rule {
 public:
  virtual ~Rule() = default;

  /// Check if this rule is violated by the given node
  virtual bool isViolated(PipelineTreeNode* node, PipelineTree& tree) const = 0;

  /// Fix a violation of this rule at the given node
  virtual bool fixViolation(PipelineTreeNode* violating_node,
                            PipelineTree& tree,
                            int& next_coarsened_stage_id) = 0;

  /// Print the rule-specific part of a violation description
  virtual void print(llvm::raw_ostream& os) const = 0;

  RuleKind getKind() const { return kind_; }

 protected:
  Rule(RuleKind kind) : kind_(kind) {}

 private:
  RuleKind kind_;
};

/// Rule 1: A pipeline can only have ktdf.private or ktdf.stage as immediate
/// children
class PipelineChildrenRule : public Rule {
 public:
  PipelineChildrenRule()
      : Rule(RuleKind::kPipelineChildrenMustBePrivateOrStage) {}

  bool isViolated(PipelineTreeNode* node, PipelineTree& tree) const override;

  bool fixViolation(PipelineTreeNode* violating_node, PipelineTree& tree,
                    int& next_coarsened_stage_id) override;

  void print(llvm::raw_ostream& os) const override;

  static bool classof(const Rule* rule) {
    return rule->getKind() == RuleKind::kPipelineChildrenMustBePrivateOrStage;
  }
};

/// Rule 2: A stage must have a pipeline as immediate parent
class StageParentRule : public Rule {
 public:
  StageParentRule() : Rule(RuleKind::kStageMustHavePipelineParent) {}

  bool isViolated(PipelineTreeNode* node, PipelineTree& tree) const override;

  bool fixViolation(PipelineTreeNode* violating_node, PipelineTree& tree,
                    int& next_coarsened_stage_id) override;

  void print(llvm::raw_ostream& os) const override;

  static bool classof(const Rule* rule) {
    return rule->getKind() == RuleKind::kStageMustHavePipelineParent;
  }
};

/// Rule 3: A pipeline with stages that have dependencies must have a private
/// node as first child
class PrivateMustExistRule : public Rule {
 public:
  PrivateMustExistRule()
      : Rule(RuleKind::kPrivateMustExistForPipelinesWithStageDeps) {}

  bool isViolated(PipelineTreeNode* node, PipelineTree& tree) const override;

  bool fixViolation(PipelineTreeNode* violating_node, PipelineTree& tree,
                    int& next_coarsened_stage_id) override;

  void print(llvm::raw_ostream& os) const override;

  static bool classof(const Rule* rule) {
    return rule->getKind() ==
           RuleKind::kPrivateMustExistForPipelinesWithStageDeps;
  }
};

/// Represents a structure violation
struct Violation {
  std::unique_ptr<Rule> rule;
  PipelineTreeNode* violating_node;

  Violation(std::unique_ptr<Rule> r, PipelineTreeNode* node)
      : rule(std::move(r)), violating_node(node) {}

  /// Print a description of this violation (node-specific + rule-specific)
  void print(llvm::raw_ostream& os) const {
    os << "violation of rule: ";
    rule->print(os);
    os << ", violating node:" << violating_node->getNodeName();
  }
};

/// Legalizer class for discovering and correcting structure violations
class Legalizer {
 public:
  Legalizer();

  /// Find and fix all structure violations iteratively until none remain
  /// Returns the number of iterations required
  int findAndFixViolations(PipelineTree& tree);

 private:
  /// Collect all structure violations in the tree
  void collectViolations(PipelineTree& tree);

  int next_coarsened_stage_id_;              // Counter for coarsened stage IDs
  llvm::SmallVector<Violation> violations_;  // Collected violations
  llvm::SmallVector<std::unique_ptr<Rule>> rules_;  // All rules to check
};

}  // namespace scheduler::pipeline_tree

#endif  // DATAFLOW_SCHEDULER_TRANSFORMS_UTILS_PIPELINETREELEGALIZER_H_
