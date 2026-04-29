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
      loopNest;  // Perfectly nested loops to sink (outermost to innermost)
  mlir::ktdf::PipelineOp pipeline;  // Target pipeline
  mlir::Operation* loopNestParent;  // Parent operation of the outermost loop
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
    auto moduleOp = getOperation();

    llvm::outs() << "\n=== Stage Coarsening Pass ===\n";
    llvm::outs() << "Loop nest depth: " << loopNestDepth << "\n\n";

    // Step 1: Identify all transform candidates
    llvm::SmallVector<TransformCandidate> candidates;
    identifyTransformCandidates(moduleOp, candidates);

    if (candidates.empty()) {
      llvm::outs() << "No transform candidates found.\n";
      return;
    }

    llvm::outs() << "Found " << candidates.size()
                 << " transform candidate(s)\n\n";

    // Step 2: Apply 4-stage transformation to each candidate
    for (auto& candidate : candidates) {
      applyFourStageTransformation(candidate);
    }

    llvm::outs() << "=== Stage Coarsening Pass Complete ===\n\n";
  }

 private:
  /// Identify all transform candidates in the module
  /// Walks IR to find pipelines with 'n' perfectly nested loops above them
  void identifyTransformCandidates(
      mlir::ModuleOp moduleOp,
      llvm::SmallVector<TransformCandidate>& candidates) {
    // Walk all functions in the module
    moduleOp.walk([&](mlir::func::FuncOp funcOp) {
      llvm::outs() << "Analyzing function: " << funcOp.getName() << "\n";

      // Find all pipelines in this function
      funcOp.walk([&](mlir::ktdf::PipelineOp pipeline) {
        llvm::outs()
            << "  Found pipeline, checking for loop nest above it...\n";

        // Walk up the parent chain to find perfectly nested loops
        mlir::Operation* currentOp = pipeline.getOperation();
        llvm::SmallVector<mlir::scf::ForOp> loopNest;

        // Collect up to 'loopNestDepth' perfectly nested loops
        for (unsigned i = 0; i < loopNestDepth; ++i) {
          mlir::Operation* parentOp = currentOp->getParentOp();
          if (!parentOp) break;

          if (auto forOp = mlir::dyn_cast<mlir::scf::ForOp>(parentOp)) {
            // Check if this is a perfectly nested loop (single child in body)
            auto& bodyOps = forOp.getBody()->getOperations();
            if (bodyOps.size() == 2) {  // Loop body + terminator
              loopNest.push_back(forOp);
              currentOp = forOp.getOperation();
              continue;
            }
          }
          break;
        }

        // If we found any loops, this is a candidate
        if (!loopNest.empty()) {
          // Reverse to get outermost-to-innermost order
          std::reverse(loopNest.begin(), loopNest.end());

          // Parent of outermost loop becomes tree root
          mlir::Operation* loopNestParent =
              loopNest.empty() ? pipeline.getOperation()->getParentOp()
                               : loopNest.front().getOperation()->getParentOp();

          candidates.push_back({loopNest, pipeline, loopNestParent});

          llvm::outs() << "    -> Found candidate with " << loopNest.size()
                       << " nested loop(s) at ";
          scheduler::utils::printLocation(llvm::outs(),
                                          loopNest.front().getOperation());
          llvm::outs() << "\n";
        }
      });
    });
  }

  /// Apply the 4-stage transformation algorithm to a candidate
  void applyFourStageTransformation(TransformCandidate& candidate) {
    llvm::outs() << "\n--- Applying 4-Stage Transformation ---\n";
    llvm::outs() << "Loop nest depth: " << candidate.loopNest.size() << "\n";

    // Build pipeline tree once - it will be manipulated throughout the
    // transformation
    PipelineTree tree;
    tree.compute(*candidate.loopNestParent);
    llvm::outs() << "=== Initial Pipeline Tree ===\n";
    tree.print(llvm::outs(), false);
    llvm::outs() << "\n";

    // Stage 1a: Distribute and sink loops into pipeline
    llvm::outs() << "\n--- Stage 1a: Distribute and Sink into Pipeline ---\n";
    stage1a_DistributeAndSinkIntoPipeline(candidate, tree);
    llvm::outs() << "\n=== Pipeline Tree after Stage 1a ===\n";
    tree.print(llvm::outs(), false);
    llvm::outs() << "\n";

    // Stage 1b: Further sink into stage groups containing a single stage
    // (expand buffers)
    llvm::outs() << "\n--- Stage 1b: Sink into Single-Stage Groups ---\n";
    stage1b_SinkIntoSingleStage(candidate, tree);
    llvm::outs() << "\n=== Pipeline Tree after Stage 1b ===\n";
    tree.print(llvm::outs(), false);
    llvm::outs() << "\n";

    // Stage 2: Discover and correct illegal structures iteratively
    llvm::outs()
        << "\n--- Stage 2: Discover and Correct Illegal Structures ---\n";
    stage2_DiscoverAndCorrectStructure(candidate, tree);
    llvm::outs() << "\n=== Pipeline Tree after Stage 2 ===\n";
    tree.print(llvm::outs(), false);
    llvm::outs() << "\n";

    // Stage 3: Correct stage dependencies that cross pipeline boundaries
    llvm::outs() << "\n--- Stage 3: Correct Stage Dependencies ---\n";
    stage3_CorrectStageDependencies(candidate, tree);
    llvm::outs() << "\n=== Pipeline Tree after Stage 3 (Final) ===\n";
    tree.print(llvm::outs(), false);
    llvm::outs() << "\n";
  }

  //===--------------------------------------------------------------------===//
  // Stage 1a: Distribute and Sink Loops into Pipeline
  //===--------------------------------------------------------------------===//

  void stage1a_DistributeAndSinkIntoPipeline(TransformCandidate& candidate,
                                             PipelineTree& tree) {
    llvm::outs() << "Stage 1a: Distributing loops into pipeline...\n";

    // Validate candidate
    if (!candidate.pipeline || !candidate.loopNestParent) {
      llvm::outs() << "  ERROR: Invalid candidate (null pipeline or parent)\n";
      return;
    }

    // Step 1: Perform stage grouping analysis
    llvm::outs() << "\n  Performing stage grouping analysis...\n";
    StageGroupingAnalysis grouping(candidate.pipeline);

    // Print the stage groups
    llvm::outs() << "\n  Stage Groups:\n";
    grouping.dump();
    llvm::outs() << "\n";

    const auto& groups = grouping.getGroups();
    llvm::outs() << "  Found " << groups.size() << " stage group(s)\n";

    // Analyze each group
    for (size_t i = 0; i < groups.size(); ++i) {
      const auto& group = groups[i];
      if (group.isSingleStage()) {
        llvm::outs() << "  Group " << i
                     << ": Single stage (can sink further in Stage 1b)\n";
      } else {
        llvm::outs() << "  Group " << i
                     << ": Multiple stages (must stay together)\n";
      }
    }

    // Step 2: For each stage group, create unmaterialized loop nest and insert
    // as children of pipeline
    llvm::outs() << "\n  Distributing loop nests to stage groups...\n";

    // Process each group
    for (size_t groupIdx = 0; groupIdx < groups.size(); ++groupIdx) {
      const auto& group = groups[groupIdx];
      const llvm::SmallVector<mlir::ktdf::StageOp> stages = group.getStages();

      if (stages.empty()) continue;

      llvm::outs() << "  Group " << groupIdx << ": Creating loop nest for "
                   << stages.size() << " stage(s)\n";

      // Find stage nodes for this group using the operation-to-node mapping
      llvm::SmallVector<StageNode*> stageNodes;
      PipelineTreeNode* pipelineParentNode =
          tree.getNodeForOp(candidate.pipeline.getOperation());
      assert(pipelineParentNode && pipelineParentNode->isPipelineNode());

      for (auto stage : stages) {
        PipelineTreeNode* node = tree.getNodeForOp(stage);
        assert(node);
        assert(node->isStageNode());
        auto* stageNode = static_cast<StageNode*>(node);
        stageNodes.push_back(stageNode);
      }
      assert(!stageNodes.empty());

      // Create nested loop nodes (outermost to innermost)
      // The outermost loop will be inserted as a child of the pipeline
      // Each subsequent loop will be a child of the previous loop
      // The stages will be moved to be children of the innermost loop
      OperationTreeNode* currentParent = pipelineParentNode;

      for (size_t loopIdx = 0; loopIdx < candidate.loopNest.size(); ++loopIdx) {
        // Create unmaterialized loop node using factory method
        LoopNode* loopNode = tree.createLoopNode(nullptr);  // Unmaterialized
        loopNode->setTemplateOp(candidate.loopNest[loopIdx].getOperation());

        // Insert loop node as child of current parent (pipeline or outer loop)
        currentParent->insertChildNode(loopNode);

        llvm::outs() << "    Created unmaterialized loop node (level "
                     << loopIdx << ") with template from ";
        scheduler::utils::printLocation(
            llvm::outs(), candidate.loopNest[loopIdx].getOperation());
        llvm::outs() << "\n";

        // This loop becomes the parent for the next loop or the stages
        currentParent = loopNode;
      }

      // Move all stages in this group to be children of the innermost loop
      for (StageNode* stageNode : stageNodes) {
        stageNode->unlink();
        currentParent->insertChildNode(stageNode);
      }

      llvm::outs() << "    Moved " << stageNodes.size()
                   << " stage(s) under loop nest\n";
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

    bool foundIllegalStructure = true;
    int iteration = 0;

    while (foundIllegalStructure && iteration < 10) {  // Safety limit
      iteration++;
      llvm::outs() << "  Iteration " << iteration << ":\n";

      // Rebuild tree to get current structure
      PipelineTree tree;
      tree.compute(*candidate.loopNestParent);
      foundIllegalStructure = false;

      // TODO: Traverse tree to discover illegal patterns
      // TODO: Correct each illegal structure found
      // TODO: Set foundIllegalStructure = true if corrections were made

      if (!foundIllegalStructure) {
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
