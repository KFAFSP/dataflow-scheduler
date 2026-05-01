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
// Stage Coarsening Transformation
//
// This pass implements a stage-coarsening algorithm as follows.
// After identification of stage groups (using loop distribution legality
// rules):
// Step 1a: Distribute and sink loops into pipeline
// Step 1b: Further sink into stage groups containing a single stage
//          (expand buffers)
// Step 2: Discover and correct illegal structures iteratively
// Step 3: Correct stage dependencies that cross pipeline boundaries
//
// Note: Legality of stage coarsening follows loop distribution rules.
//       Data dependence analysis will be added in future.
//===----------------------------------------------------------------------===//

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "src/Analysis/PipelineTree.hpp"
#include "src/Analysis/StageGrouping.hpp"
#include "src/Dialects/KTDF/KTDFOps.hpp"
#include "src/Transform/Passes.h"
#include "src/Transform/StructuralLegalizer.hpp"
#include "src/Utils/OperationUtils.hpp"

namespace ktdf {

#define GEN_PASS_DEF_STAGECOARSENINGPASS
#include "src/Transform/Passes.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// Transform Candidate Structure
//===----------------------------------------------------------------------===//

/// Represents a candidate for stage coarsening transformation
struct TransformCandidate {
  llvm::SmallVector<mlir::scf::ForOp>
      loop_nest_;  // Perfectly nested loops to sink (outermost to innermost)
  mlir::ktdf::PipelineOp pipeline_;    // Target pipeline
  mlir::Operation* loop_nest_parent_;  // Parent operation of the outermost loop
                                       // (will be tree root)
};

//===----------------------------------------------------------------------===//
// Stage Coarsening Pass
//===----------------------------------------------------------------------===//

/// Module pass that implements the 4-stage algorithm for creating hierarchical
/// pipelines
struct StageCoarseningPass
    : public impl::StageCoarseningPassBase<StageCoarseningPass> {
  using impl::StageCoarseningPassBase<
      StageCoarseningPass>::StageCoarseningPassBase;

  void runOnOperation() override {
    auto module_op = getOperation();

    llvm::outs() << "\n=== Stage Coarsening Pass ===\n";
    llvm::outs() << "Loop nest depth: " << loopNestDepth << "\n\n";

    // Step 1: Identify all transform candidates
    llvm::SmallVector<TransformCandidate> candidates;
    identifyTransformCandidates(module_op, candidates);

    if (candidates.empty()) {
      llvm::outs() << "No transform candidates found.\n";
      return;
    }

    llvm::outs() << "Found " << candidates.size()
                 << " transform candidate(s)\n\n";

    // Step 2: Apply 4-step transformation to each candidate
    for (TransformCandidate& candidate : candidates) {
      applyTransformation(candidate);
    }

    llvm::outs() << "=== Stage Coarsening Pass Complete ===\n\n";
  }

 private:
  /// Identify all transform candidates in the module
  /// Walks IR to find pipelines with 'n' perfectly nested loops above them
  void identifyTransformCandidates(
      mlir::ModuleOp module_op,
      llvm::SmallVector<TransformCandidate>& candidates) {
    // Walk all functions in the module
    module_op.walk([&](mlir::func::FuncOp func_op) {
      llvm::outs() << "Analyzing function: " << func_op.getName() << "\n";

      // Find all pipelines in this function
      func_op.walk([&](mlir::ktdf::PipelineOp pipeline) {
        llvm::outs()
            << "  Found pipeline, checking for loop nest above it...\n";

        // Walk up the parent chain to find perfectly nested loops
        mlir::Operation* current_op = pipeline.getOperation();
        llvm::SmallVector<mlir::scf::ForOp> loop_nest;

        // Collect up to 'loopNestDepth' perfectly nested loops
        for (unsigned i = 0; i < loopNestDepth; ++i) {
          mlir::Operation* parent_op = current_op->getParentOp();
          if (!parent_op) break;

          if (auto forOp = mlir::dyn_cast<mlir::scf::ForOp>(parent_op)) {
            // Check if this is a perfectly nested loop (single child in body)
            auto& body_ops = forOp.getBody()->getOperations();
            if (body_ops.size() == 2) {  // Loop body + terminator
              loop_nest.push_back(forOp);
              current_op = forOp.getOperation();
              continue;
            }
          }
          break;
        }

        // If we found any loops, this is a candidate
        if (!loop_nest.empty()) {
          // Reverse to get outermost-to-innermost order
          std::reverse(loop_nest.begin(), loop_nest.end());

          // Parent of outermost loop becomes tree root
          mlir::Operation* loop_nest_parent =
              loop_nest.empty()
                  ? pipeline.getOperation()->getParentOp()
                  : loop_nest.front().getOperation()->getParentOp();

          candidates.push_back({loop_nest, pipeline, loop_nest_parent});

          llvm::outs() << "    -> Found candidate with " << loop_nest.size()
                       << " nested loop(s) at ";
          scheduler::utils::printLocation(llvm::outs(),
                                          loop_nest.front().getOperation());
          llvm::outs() << "\n";
        }
      });
    });
  }

  /// Apply the 4-step transformation algorithm to a candidate
  void applyTransformation(TransformCandidate& candidate) {
    llvm::outs() << "\n--- Applying Transformation in steps ---\n";
    llvm::outs() << "Loop nest depth: " << candidate.loop_nest_.size() << "\n";

    // Build pipeline tree once - it will be manipulated throughout the
    // transformation
    PipelineTree tree;
    tree.compute(*candidate.loop_nest_parent_);
    llvm::outs() << "=== Initial Pipeline Tree ===\n";
    tree.print(llvm::outs());
    llvm::outs() << "\n";

    // Perform stage grouping analysis once for both step 1a and 1b
    llvm::outs() << "\n  Performing stage grouping analysis...\n";
    StageGroupingAnalysis grouping(candidate.pipeline_);
    grouping.print(llvm::outs());

    // Step 1a: Distribute and sink loops into pipeline
    llvm::outs() << "\n--- Step 1a: Distribute and Sink into Pipeline ---\n";
    step1a_DistributeAndSinkIntoPipeline(candidate, tree, grouping);
    llvm::outs() << "\n=== Pipeline Tree after Step 1a ===\n";
    tree.print(llvm::outs());
    llvm::outs() << "\n";

    // Step 1b: Further sink into stage groups containing a single stage
    // (expand buffers)
    llvm::outs() << "\n--- Step 1b: Sink into Single-Stage Groups ---\n";
    step1b_SinkIntoSingleStage(candidate, tree, grouping);
    llvm::outs() << "\n=== Pipeline Tree after Step 1b ===\n";
    tree.print(llvm::outs());
    llvm::outs() << "\n";

    // Step 2: Discover and correct illegal structures iteratively
    llvm::outs()
        << "\n--- Step 2: Discover and Correct Illegal Structures ---\n";
    step2_DiscoverAndCorrectStructure(candidate, tree);
    llvm::outs() << "\n=== Pipeline Tree after Step 2 ===\n";
    tree.print(llvm::outs());
    llvm::outs() << "\n";

    // Step 3: Correct stage dependencies that cross pipeline boundaries
    llvm::outs() << "\n--- Step 3: Correct Stage Dependencies ---\n";
    step3_CorrectStageDependencies(candidate, tree);
    llvm::outs() << "\n=== Pipeline Tree after Step 3 (Final) ===\n";
    tree.print(llvm::outs());
    llvm::outs() << "\n";
  }

  //===--------------------------------------------------------------------===//
  // Step 1a: Distribute and Sink Loops into Pipeline
  //===--------------------------------------------------------------------===//

  void step1a_DistributeAndSinkIntoPipeline(
      TransformCandidate& candidate, PipelineTree& tree,
      const StageGroupingAnalysis& grouping) {
    llvm::outs() << "Step 1a: Distributing loops into pipeline...\n";

    // Validate candidate
    if (!candidate.pipeline_ || !candidate.loop_nest_parent_) {
      llvm::outs() << "  ERROR: Invalid candidate (null pipeline or parent)\n";
      return;
    }

    // For each stage group, create unmaterialized loop nest and insert
    // as children of pipeline
    llvm::outs() << "\n  Distributing loop nests to stage groups...\n";

    // Process each group
    const llvm::SmallVector<StageGroup>& groups = grouping.getGroups();
    for (size_t group_idx = 0; group_idx < groups.size(); ++group_idx) {
      const StageGroup& group = groups[group_idx];
      const llvm::SmallVector<mlir::ktdf::StageOp>& stages = group.getStages();

      if (stages.empty()) continue;

      llvm::outs() << "  Group " << group_idx << ": Creating loop nest for "
                   << stages.size() << " stage(s)\n";

      // Find stage nodes for this group using the operation-to-node mapping
      llvm::SmallVector<StageNode*> stage_nodes;
      PipelineTreeNode* pipeline_parent_node =
          tree.getNodeForOp(candidate.pipeline_.getOperation());
      assert(pipeline_parent_node && pipeline_parent_node->isPipelineNode());

      for (mlir::ktdf::StageOp stage : stages) {
        PipelineTreeNode* node = tree.getNodeForOp(stage);
        assert(node);
        assert(node->isStageNode());
        auto* stage_node = static_cast<StageNode*>(node);
        stage_nodes.push_back(stage_node);
      }
      assert(!stage_nodes.empty());

      // Create nested loop nodes (outermost to innermost)
      // The outermost loop will be inserted as a child of the pipeline
      // Each subsequent loop will be a child of the previous loop
      // The stages will be moved to be children of the innermost loop
      OperationTreeNode* current_parent = pipeline_parent_node;

      for (size_t loop_idx = 0; loop_idx < candidate.loop_nest_.size();
           ++loop_idx) {
        // Create unmaterialized loop node using factory method
        LoopNode* loop_node = tree.createLoopNode(nullptr);  // Unmaterialized
        loop_node->setTemplateOp(candidate.loop_nest_[loop_idx].getOperation());

        // Insert loop node as child of current parent (pipeline or outer loop)
        current_parent->insertChildNode(loop_node);

        // This loop becomes the parent for the next loop or the stages
        current_parent = loop_node;
      }

      // Move all stages in this group to be children of the innermost loop
      for (StageNode* stage_node : stage_nodes) {
        stage_node->unlink();
        current_parent->insertChildNode(stage_node);
      }

      llvm::outs() << "    Moved " << stage_nodes.size()
                   << " stage(s) under loop nest\n";
    }

    // Now remove the original loops from the tree
    // Note that references to their operations still exist as template ops in
    // the unmaterialized nodes
    for (mlir::scf::ForOp loop_op : candidate.loop_nest_) {
      PipelineTreeNode* loop_node = tree.getNodeForOp(loop_op.getOperation());
      assert(loop_node);
      assert(loop_node->isLoopNode() && "Expected loop node");
      // Unlink only the node itself, not its subtree
      // This moves the loop's children (the pipeline) up to the loop's parent
      loop_node->unlinkNode();
      // Delete the now-unlinked loop node
      tree.deleteNode(loop_node);
    }

    llvm::outs() << "\n  Loop distribution complete.\n";
  }

  //===--------------------------------------------------------------------===//
  // Step 1b: Sink into Stage Groups Containing a Single Stage
  //===--------------------------------------------------------------------===//

  void step1b_SinkIntoSingleStage(TransformCandidate& candidate,
                                  PipelineTree& tree,
                                  const StageGroupingAnalysis& grouping) {
    llvm::outs()
        << "Step 1b: Conditionally sinking into single-stage groups...\n";

    const llvm::SmallVector<StageGroup>& groups = grouping.getGroups();

    // Find the pipeline node in the tree
    PipelineTreeNode* pipeline_node =
        tree.getNodeForOp(candidate.pipeline_.getOperation());
    assert(pipeline_node && pipeline_node->isPipelineNode());

    // Process each stage group
    for (size_t group_idx = 0; group_idx < groups.size(); ++group_idx) {
      const StageGroup& group = groups[group_idx];

      // Only process single-stage groups
      if (!group.isSingleStage()) {
        llvm::outs() << "  Group " << group_idx << ": Skipping (contains "
                     << group.getStages().size() << " stages)\n";
        continue;
      }

      mlir::ktdf::StageOp stage = group.getFirstStage();
      llvm::outs() << "  Group " << group_idx
                   << ": Processing single-stage group with stage at ";
      scheduler::utils::printLocation(llvm::outs(), stage.getOperation());
      llvm::outs() << "\n";

      // Find the stage node in the tree
      PipelineTreeNode* stage_node = tree.getNodeForOp(stage.getOperation());
      assert(stage_node && stage_node->isStageNode());

      // The stage's parent should be the innermost loop from step 1a
      OperationTreeNode* innermost_loop = stage_node->getParentNode();
      assert(innermost_loop && "Stage should have a parent by now");

      PipelineTreeNode* innermost_loop_pipeline =
          static_cast<PipelineTreeNode*>(innermost_loop);
      assert(innermost_loop_pipeline->isLoopNode() &&
             "Stage's parent should be a loop node from step 1a");

      // Find the outermost loop in the nest by walking up to the pipeline
      PipelineTreeNode* outermost_loop = innermost_loop_pipeline;
      OperationTreeNode* parent = outermost_loop->getParentNode();
      while (parent) {
        PipelineTreeNode* parent_pipeline =
            static_cast<PipelineTreeNode*>(parent);
        if (parent_pipeline->isPipelineNode()) {
          // Reached the pipeline, outermost_loop is correct
          assert(parent == pipeline_node &&
                 "Outermost loop should be a child of the pipeline");
          break;
        }
        assert(parent_pipeline->isLoopNode() &&
               "Expected loop node while walking up to pipeline");
        outermost_loop = parent_pipeline;
        parent = outermost_loop->getParentNode();
      }

      llvm::outs() << "    Sinking loop nest into stage\n";

      // Sink the loop nest into the stage:
      assert(outermost_loop->getParentNode() == pipeline_node);
      OperationTreeNode* insert = outermost_loop->getPrevSibling();
      outermost_loop->unlink();
      stage_node->unlink();
      stage_node->insertChildNode(outermost_loop);
      pipeline_node->insertChildNode(stage_node, insert);

      llvm::outs() << "    Successfully sank loop nest into stage\n";

      // TODO: Expand LX buffers from 1D to tile-sized
      // This requires:
      // 1. Identifying which buffers in the private region are used by this
      // stage
      // 2. Determining the tile size from the loop bounds
      // 3. Marking these buffers as "expanded" in the PrivateNode
      // 4. Recording which data transfers and operations need to be updated
      llvm::outs() << "    TODO: Buffer expansion not yet implemented\n";
    }

    llvm::outs() << "  Step 1b complete.\n";
  }

  //===--------------------------------------------------------------------===//
  // Step 2: Discover and Correct Illegal Structures
  //===--------------------------------------------------------------------===//

  void step2_DiscoverAndCorrectStructure(TransformCandidate& candidate,
                                         PipelineTree& tree) {
    llvm::outs()
        << "Step 2: Discovering and correcting illegal structures...\n";

    scheduler::structural_legality::Legalizer legalizer;

    // Find and fix all violations iteratively
    legalizer.findAndFixViolations(tree);

    llvm::outs() << "  Step 2 complete.\n";
  }

  //===--------------------------------------------------------------------===//
  // Step 3: Correct Stage Dependencies Crossing Pipeline Boundaries
  //===--------------------------------------------------------------------===//

  void step3_CorrectStageDependencies(TransformCandidate& candidate,
                                      PipelineTree& tree) {
    llvm::outs() << "Step 3: Correcting stage dependencies...\n";

    // TODO: Implement dependency correction
    // For each stage dependency that crosses pipeline boundaries:
    //   1. Identify the dependency
    //   2. Create nested pipeline if needed
    //   3. Move stages to correct pipeline level

    llvm::outs() << "  (Implementation pending)\n";
  }
};

}  // namespace
}  // namespace ktdf

std::unique_ptr<mlir::Pass> ktdf::createStageCoarseningPass() {
  return std::make_unique<ktdf::StageCoarseningPass>();
}
