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

    // Step 2: Apply 4-stage transformation to each candidate
    for (TransformCandidate& candidate : candidates) {
      applyFourStageTransformation(candidate);
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

  /// Apply the 4-stage transformation algorithm to a candidate
  void applyFourStageTransformation(TransformCandidate& candidate) {
    llvm::outs() << "\n--- Applying 4-Stage Transformation ---\n";
    llvm::outs() << "Loop nest depth: " << candidate.loop_nest_.size() << "\n";

    // Build pipeline tree once - it will be manipulated throughout the
    // transformation
    PipelineTree tree;
    tree.compute(*candidate.loop_nest_parent_);
    llvm::outs() << "=== Initial Pipeline Tree ===\n";
    tree.print(llvm::outs());
    llvm::outs() << "\n";

    // Stage 1a: Distribute and sink loops into pipeline
    llvm::outs() << "\n--- Stage 1a: Distribute and Sink into Pipeline ---\n";
    stage1a_DistributeAndSinkIntoPipeline(candidate, tree);
    llvm::outs() << "\n=== Pipeline Tree after Stage 1a ===\n";
    tree.print(llvm::outs());
    llvm::outs() << "\n";

    // Stage 1b: Further sink into stage groups containing a single stage
    // (expand buffers)
    llvm::outs() << "\n--- Stage 1b: Sink into Single-Stage Groups ---\n";
    stage1b_SinkIntoSingleStage(candidate, tree);
    llvm::outs() << "\n=== Pipeline Tree after Stage 1b ===\n";
    tree.print(llvm::outs());
    llvm::outs() << "\n";

    // Stage 2: Discover and correct illegal structures iteratively
    llvm::outs()
        << "\n--- Stage 2: Discover and Correct Illegal Structures ---\n";
    stage2_DiscoverAndCorrectStructure(candidate, tree);
    llvm::outs() << "\n=== Pipeline Tree after Stage 2 ===\n";
    tree.print(llvm::outs());
    llvm::outs() << "\n";

    // Stage 3: Correct stage dependencies that cross pipeline boundaries
    llvm::outs() << "\n--- Stage 3: Correct Stage Dependencies ---\n";
    stage3_CorrectStageDependencies(candidate, tree);
    llvm::outs() << "\n=== Pipeline Tree after Stage 3 (Final) ===\n";
    tree.print(llvm::outs());
    llvm::outs() << "\n";
  }

  //===--------------------------------------------------------------------===//
  // Stage 1a: Distribute and Sink Loops into Pipeline
  //===--------------------------------------------------------------------===//

  void stage1a_DistributeAndSinkIntoPipeline(TransformCandidate& candidate,
                                             PipelineTree& tree) {
    llvm::outs() << "Stage 1a: Distributing loops into pipeline...\n";

    // Validate candidate
    if (!candidate.pipeline_ || !candidate.loop_nest_parent_) {
      llvm::outs() << "  ERROR: Invalid candidate (null pipeline or parent)\n";
      return;
    }

    // Step 1: Perform stage grouping analysis
    llvm::outs() << "\n  Performing stage grouping analysis...\n";
    StageGroupingAnalysis grouping(candidate.pipeline_);
    // Print stage grouping analysis results
    grouping.print(llvm::outs());

    // Step 2: For each stage group, create unmaterialized loop nest and insert
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
  // Stage 1b: Sink into Stage Groups Containing a Single Stage
  //===--------------------------------------------------------------------===//

  void stage1b_SinkIntoSingleStage(TransformCandidate& candidate,
                                   PipelineTree& tree) {
    llvm::outs()
        << "Stage 1b: Conditionally sinking into single-stage groups...\n";

    // TODO: Implement conditional sinking with buffer expansion
    // For each stage group containing only a single stage:
    //   1. Check if sinking is legal (dependence analysis)
    //   2. If legal, sink loop into that stage
    //   3. Expand LX buffers from 1D to tile-sized

    llvm::outs() << "  (Implementation pending)\n";
  }

  //===--------------------------------------------------------------------===//
  // Stage 2: Discover and Correct Illegal Structures
  //===--------------------------------------------------------------------===//

  void stage2_DiscoverAndCorrectStructure(TransformCandidate& candidate,
                                          PipelineTree& tree) {
    llvm::outs()
        << "Stage 2: Discovering and correcting illegal structures...\n";

    // TODO: Implement iterative structure discovery and correction
    // Repeat until no illegal structures found:
    //   1. Rebuild pipeline tree
    //   2. Discover illegal patterns:
    //      - Pattern 1: Pipeline → Loop → Multiple Stages (missing wrapping
    //      stage)
    //      - Pattern 2: Stage without Pipeline parent (missing nested pipeline)
    //   3. Correct each illegal structure
    //   4. Check for new illegal structures introduced by corrections

    bool found_illegal_structure = true;
    int iteration = 0;

    while (found_illegal_structure && iteration < 10) {  // Safety limit
      iteration++;
      llvm::outs() << "  Iteration " << iteration << ":\n";

      // Rebuild tree to get current structure
      PipelineTree tree;
      tree.compute(*candidate.loop_nest_parent_);
      found_illegal_structure = false;

      // TODO: Traverse tree to discover illegal patterns
      // TODO: Correct each illegal structure found
      // TODO: Set found_illegal_structure = true if corrections were made

      if (!found_illegal_structure) {
        llvm::outs() << "    No illegal structures found.\n";
      }
    }

    if (iteration >= 10) {
      llvm::outs() << "  WARNING: Reached iteration limit, may have unresolved "
                      "structures\n";
    }

    llvm::outs() << "  (Implementation pending)\n";
  }

  //===--------------------------------------------------------------------===//
  // Stage 3: Correct Stage Dependencies Crossing Pipeline Boundaries
  //===--------------------------------------------------------------------===//

  void stage3_CorrectStageDependencies(TransformCandidate& candidate,
                                       PipelineTree& tree) {
    llvm::outs() << "Stage 3: Correcting stage dependencies...\n";

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
