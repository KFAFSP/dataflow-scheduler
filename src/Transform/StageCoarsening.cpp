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
// This pass implements a stage-coarsening algorithm using the following steps:
//
// 1: Identify stage groups using loop distribution legality rules.
// 2: Construct a PipelineTree to represent the currect structure of the code
// and transform it to the desired structure (this step does not modify IR)
//   a. Distribute and sink loops into pipeline.
//   b. Further sink into stage groups containing a single stage (expand
//      buffers).
//   c. Discover and correct illegal structures iteratively.
//   d. Correct stage dependencies that cross pipeline boundaries.
// 3: Use the manipulated PipelineTree abstraction with reference to
//    operations in the input IR to generate the transformed IR.
//
// Note: Legality of stage coarsening follows loop distribution rules.
//       Data dependence analysis may be added in future.
//===----------------------------------------------------------------------===//

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "src/Analysis/PipelineTree.hpp"
#include "src/Analysis/StageGrouping.hpp"
#include "src/Dialects/KTDF/KTDFOps.hpp"
#include "src/Transform/BufferExpansion.hpp"
#include "src/Transform/Passes.h"
#include "src/Transform/PipelineTree/Legalizer.hpp"
#include "src/Transform/PipelineTree/Materializer.hpp"
#include "src/Utils/OperationUtils.hpp"

namespace ktdf {

#define GEN_PASS_DEF_STAGECOARSENINGPASS
#include "src/Transform/Passes.h.inc"

#define PASS_NAME "stage-coarsening"
#define DEBUG_TYPE PASS_NAME

namespace {
const char VerboseDebug[] = DEBUG_TYPE "-verbose";

//===----------------------------------------------------------------------===//
// Stage Coarsening Pass
//===----------------------------------------------------------------------===//

/// Module pass that implements the algorithm for creating hierarchical
/// pipelines
struct StageCoarseningPass
    : public impl::StageCoarseningPassBase<StageCoarseningPass> {
  using impl::StageCoarseningPassBase<
      StageCoarseningPass>::StageCoarseningPassBase;

  void runOnOperation() override;

 private:
  /// Represents a candidate for stage coarsening transformation
  struct TransformCandidate {
    llvm::SmallVector<mlir::scf::ForOp>
        loop_nest_;  // Perfectly nested loops to sink (outermost to innermost)
    mlir::ktdf::PipelineOp pipeline_;    // Target pipeline
    mlir::Operation* loop_nest_parent_;  // Parent operation of the outermost
                                         // loop (will be tree root)
  };

  /// Identify all transform candidates in the module
  /// Walks IR to find pipelines with 'n' perfectly nested loops above them
  void identifyTransformCandidates(
      mlir::ModuleOp module_op,
      llvm::SmallVector<TransformCandidate>& candidates);

  /// Apply the transformation algorithm to a candidate
  void applyTransformation(TransformCandidate& candidate);

  //===--------------------------------------------------------------------===//
  // Distribute and Sink Loops into Pipeline
  //===--------------------------------------------------------------------===//

  void DistributeAndSinkIntoPipeline(
      TransformCandidate& candidate, PipelineTree& tree,
      const StageGroupingAnalysis& grouping,
      llvm::SmallVectorImpl<scheduler::BufferExpansionInfo*>& expansion_infos);

  /// Perform buffer expansion analysis and record expansion information

  //===--------------------------------------------------------------------===//
  // Sink into Stage Groups Containing a Single Stage
  //===--------------------------------------------------------------------===//

  void SinkIntoSingleStage(TransformCandidate& candidate, PipelineTree& tree,
                           const StageGroupingAnalysis& grouping);

  //===--------------------------------------------------------------------===//
  // Discover and Correct Illegal Structures
  //===--------------------------------------------------------------------===//

  void DiscoverAndCorrectStructure(TransformCandidate& candidate,
                                   PipelineTree& tree);

  //===--------------------------------------------------------------------===//
  // Correct Stage Dependencies Crossing Pipeline Boundaries
  //===--------------------------------------------------------------------===//

  void CorrectStageDependencies(TransformCandidate& candidate,
                                PipelineTree& tree);

  //===--------------------------------------------------------------------===//
  // Step 3: Generate Transformed IR from PipelineTree
  //===--------------------------------------------------------------------===//

  /// Generate the transformed IR from the manipulated PipelineTree
  void GenerateTransformedIR(
      TransformCandidate& candidate, PipelineTree& tree,
      llvm::ArrayRef<const scheduler::BufferExpansionInfo*> expansion_infos);

  //===--------------------------------------------------------------------===//
  // Fix Private Result Scoping
  //===--------------------------------------------------------------------===//

  /// Fix scoping of private results that are only used in inner pipelines
  void fixPrivateResultScoping(mlir::OpBuilder& builder,
                               mlir::Operation* outer_pipeline_op);

  /// Analyze which private results are only used in specific inner pipelines
  void analyzePrivateResultUsage(
      mlir::ktdf::PrivateOp outer_private,
      mlir::ktdf::PipelineOp outer_pipeline,
      llvm::ArrayRef<mlir::ktdf::PipelineOp> inner_pipelines,
      llvm::DenseMap<mlir::ktdf::PipelineOp, llvm::SmallVector<unsigned>>&
          pipeline_to_result_indices);

  /// Move specified private results from outer private to inner pipeline
  void movePrivateResultsToInnerPipeline(
      mlir::OpBuilder& builder, mlir::ktdf::PrivateOp outer_private,
      mlir::ktdf::PipelineOp inner_pipeline,
      llvm::ArrayRef<unsigned> result_indices);

  /// Extend an existing inner private node with moved allocations
  void extendInnerPrivate(mlir::OpBuilder& builder,
                          mlir::ktdf::PrivateOp inner_private,
                          llvm::ArrayRef<mlir::Operation*> ops_to_move,
                          llvm::SmallVectorImpl<mlir::Value>& old_results);

  /// Create a new inner private node with moved allocations
  void createInnerPrivate(mlir::OpBuilder& builder,
                          mlir::ktdf::PipelineOp inner_pipeline,
                          llvm::ArrayRef<mlir::Operation*> ops_to_move,
                          llvm::SmallVectorImpl<mlir::Value>& old_results);

  /// Update the outer private to remove moved results
  void updateOuterPrivate(mlir::OpBuilder& builder,
                          mlir::ktdf::PrivateOp outer_private,
                          llvm::ArrayRef<unsigned> removed_indices);
};

//===----------------------------------------------------------------------===//
// StageCoarseningPass Implementation
//===----------------------------------------------------------------------===//

void StageCoarseningPass::runOnOperation() {
  auto module_op = getOperation();

  LLVM_DEBUG({
    llvm::dbgs() << "\n=== Stage Coarsening Pass ===\n";
    llvm::dbgs() << "Loop nest depth: " << loopNestDepth << "\n\n";
  });

  // Identify all transform candidates
  llvm::SmallVector<TransformCandidate> candidates;
  identifyTransformCandidates(module_op, candidates);

  if (candidates.empty()) {
    LLVM_DEBUG(llvm::dbgs() << "No transform candidates found.\n");
    return;
  }

  LLVM_DEBUG(llvm::dbgs() << "Found " << candidates.size()
                          << " transform candidate(s)\n\n");

  // Apply transformation to each candidate
  for (TransformCandidate& candidate : candidates) {
    applyTransformation(candidate);
  }

  LLVM_DEBUG(llvm::dbgs() << "=== Stage Coarsening Pass Complete ===\n\n");
}

void StageCoarseningPass::identifyTransformCandidates(
    mlir::ModuleOp module_op,
    llvm::SmallVector<TransformCandidate>& candidates) {
  // Walk all functions in the module
  module_op.walk([&](mlir::func::FuncOp func_op) {
    LLVM_DEBUG(llvm::dbgs()
               << "Analyzing function: " << func_op.getName() << "\n");

    // Find all pipelines in this function
    func_op.walk([&](mlir::ktdf::PipelineOp pipeline) {
      LLVM_DEBUG(llvm::dbgs()
                 << "  Found pipeline, checking for loop nest above it...\n");

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
            loop_nest.empty() ? pipeline.getOperation()->getParentOp()
                              : loop_nest.front().getOperation()->getParentOp();

        candidates.push_back({loop_nest, pipeline, loop_nest_parent});

        LLVM_DEBUG({
          llvm::dbgs() << "    -> Found candidate with " << loop_nest.size()
                       << " nested loop(s) at ";
          scheduler::utils::printLocation(llvm::dbgs(),
                                          loop_nest.front().getOperation());
          llvm::dbgs() << "\n";
        });
      }
    });
  });
}

void StageCoarseningPass::applyTransformation(TransformCandidate& candidate) {
  LLVM_DEBUG({
    llvm::dbgs() << "\n--- Applying Transformation in steps ---\n";
    llvm::dbgs() << "Loop nest depth: " << candidate.loop_nest_.size() << "\n";
  });

  // Build pipeline tree once - it will be manipulated throughout the
  // transformation
  PipelineTree tree;
  tree.compute(*candidate.loop_nest_parent_);
  LLVM_DEBUG({
    llvm::dbgs() << "=== Initial Pipeline Tree ===\n";
    tree.print(llvm::dbgs());
    llvm::dbgs() << "\n";
  });

  // Perform stage grouping analysis once
  LLVM_DEBUG(llvm::dbgs() << "\n  Performing stage grouping analysis...\n");
  StageGroupingAnalysis grouping(candidate.pipeline_);
  LLVM_DEBUG(grouping.print(llvm::dbgs()));

  // Step 2a: Distribute and sink loops into pipeline
  LLVM_DEBUG(
      llvm::dbgs() << "\n--- Step 2a: Distribute and Sink into Pipeline ---\n");
  llvm::SmallVector<scheduler::BufferExpansionInfo*> buffer_expansion_infos;
  DistributeAndSinkIntoPipeline(candidate, tree, grouping,
                                buffer_expansion_infos);
  LLVM_DEBUG({
    llvm::dbgs() << "\n=== Pipeline Tree after Step 2a ===\n";
    tree.print(llvm::dbgs());
    llvm::dbgs() << "\n";
  });

  // Step 2b: Further sink into stage groups containing a single stage
  // (expand buffers)
  LLVM_DEBUG(
      llvm::dbgs() << "\n--- Step 2b: Sink into Single-Stage Groups ---\n");
  SinkIntoSingleStage(candidate, tree, grouping);
  LLVM_DEBUG({
    llvm::dbgs() << "\n=== Pipeline Tree after Step 2b ===\n";
    tree.print(llvm::dbgs());
    llvm::dbgs() << "\n";
  });

  // Step 2c: Discover and correct illegal structures iteratively
  LLVM_DEBUG(llvm::dbgs()
             << "\n--- Step 2c: Discover and Correct Illegal Structures ---\n");
  DiscoverAndCorrectStructure(candidate, tree);
  LLVM_DEBUG({
    llvm::dbgs() << "\n=== Pipeline Tree after Step 2c ===\n";
    tree.print(llvm::dbgs());
    llvm::dbgs() << "\n";
  });

  // Step 2d: Correct stage dependencies that cross pipeline boundaries
  LLVM_DEBUG(llvm::dbgs() << "\n--- Step 2d: Correct Stage Dependencies ---\n");
  CorrectStageDependencies(candidate, tree);
  LLVM_DEBUG({
    llvm::dbgs() << "\n=== Pipeline Tree after Step 2d (Final) ===\n";
    tree.print(llvm::dbgs());
    llvm::dbgs() << "\n";
  });

  // Step 3: Generate transformed IR from the manipulated tree
  GenerateTransformedIR(candidate, tree, buffer_expansion_infos);

  // Clean up allocated expansion infos
  for (scheduler::BufferExpansionInfo* info : buffer_expansion_infos) {
    delete info;
  }
}

void StageCoarseningPass::DistributeAndSinkIntoPipeline(
    TransformCandidate& candidate, PipelineTree& tree,
    const StageGroupingAnalysis& grouping,
    llvm::SmallVectorImpl<scheduler::BufferExpansionInfo*>& expansion_infos) {
  LLVM_DEBUG(llvm::dbgs() << "Distributing loops into pipeline...\n");

  // Validate candidate
  if (!candidate.pipeline_ || !candidate.loop_nest_parent_) {
    LLVM_DEBUG(llvm::dbgs()
               << "  ERROR: Invalid candidate (null pipeline or parent)\n");
    return;
  }

  // For each stage group, create unmaterialized loop nest and insert
  // as children of pipeline
  LLVM_DEBUG(
      llvm::dbgs() << "\n  Distributing loop nests to stage groups...\n");

  // Process each group
  const llvm::SmallVector<StageGroup>& groups = grouping.getGroups();
  for (size_t group_idx = 0; group_idx < groups.size(); ++group_idx) {
    const StageGroup& group = groups[group_idx];
    const llvm::SmallVector<mlir::ktdf::StageOp>& stages = group.getStages();

    if (stages.empty()) continue;

    LLVM_DEBUG(llvm::dbgs()
               << "  Group " << group_idx << ": Creating loop nest for "
               << stages.size() << " stage(s)\n");

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

    LLVM_DEBUG(llvm::dbgs() << "    Moved " << stage_nodes.size()
                            << " stage(s) under loop nest\n");
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

  scheduler::PerformBufferExpansionAnalysis(
      candidate.loop_nest_, candidate.pipeline_, grouping, expansion_infos);

  LLVM_DEBUG(llvm::dbgs() << "\n  Loop distribution complete.\n");
}

void StageCoarseningPass::SinkIntoSingleStage(
    TransformCandidate& candidate, PipelineTree& tree,
    const StageGroupingAnalysis& grouping) {
  LLVM_DEBUG(llvm::dbgs() << "Sinking into single-stage groups...\n");

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
      LLVM_DEBUG(llvm::dbgs()
                 << "  Group " << group_idx << ": Skipping (contains "
                 << group.getStages().size() << " stages)\n");
      continue;
    }

    mlir::ktdf::StageOp stage = group.getFirstStage();
    LLVM_DEBUG({
      llvm::dbgs() << "  Group " << group_idx
                   << ": Processing single-stage group with stage at ";
      scheduler::utils::printLocation(llvm::dbgs(), stage.getOperation());
      llvm::dbgs() << "\n";
    });

    // Find the stage node in the tree
    PipelineTreeNode* stage_node = tree.getNodeForOp(stage.getOperation());
    assert(stage_node && stage_node->isStageNode());

    // The stage's parent should now be the innermost loop from the candidate
    OperationTreeNode* innermost_loop = stage_node->getParentNode();
    assert(innermost_loop && "Stage should have a parent by now");

    PipelineTreeNode* innermost_loop_pipeline =
        static_cast<PipelineTreeNode*>(innermost_loop);
    assert(innermost_loop_pipeline->isLoopNode() &&
           "Stage's parent should be a loop node now");

    // Find the outermost loop in the nest by walking up to the pipeline
    const PipelineTreeNode* outermost_loop =
        PipelineTreeNode::findOutermostLoop(innermost_loop_pipeline,
                                            [](const PipelineTreeNode* node) {
                                              return node->isPipelineNode();
                                            });
    assert(outermost_loop && "Expected to find an outermost loop");

    LLVM_DEBUG(llvm::dbgs() << "    Sinking loop nest into stage\n");

    // Sink the loop nest into the stage:
    PipelineTreeNode* mutable_outermost =
        const_cast<PipelineTreeNode*>(outermost_loop);
    assert(mutable_outermost->getParentNode() == pipeline_node);
    OperationTreeNode* insert = mutable_outermost->getPrevSibling();
    mutable_outermost->unlink();
    stage_node->unlink();
    stage_node->insertChildNode(mutable_outermost);
    pipeline_node->insertChildNode(stage_node, insert);

    LLVM_DEBUG(llvm::dbgs() << "    Successfully sank loop nest into stage\n");
  }

  LLVM_DEBUG(llvm::dbgs() << "  Sinking into single-stage groups complete.\n");
}

void StageCoarseningPass::DiscoverAndCorrectStructure(
    TransformCandidate& candidate, PipelineTree& tree) {
  LLVM_DEBUG(
      llvm::dbgs() << "Discovering and correcting illegal structures...\n");

  scheduler::pipeline_tree::Legalizer legalizer;

  // Find and fix all violations iteratively
  legalizer.findAndFixViolations(tree);

  LLVM_DEBUG(llvm::dbgs() << "  Structure legalization complete.\n");
}

void StageCoarseningPass::CorrectStageDependencies(
    TransformCandidate& candidate, PipelineTree& tree) {
  LLVM_DEBUG(llvm::dbgs() << "Correcting stage dependencies...\n");

  // Step 1: Build pipeline-to-depth map and collect all stages
  llvm::DenseMap<const PipelineNode*, unsigned> pipeline_to_depth;
  llvm::SmallVector<StageNode*> all_stages;
  unsigned pipeline_depth = 0;

  PipelineTreeNode::walk<PipelineTreeNode::kPreOrder>(
      tree.getRoot(), [&](OperationTreeNode* node) {
        auto* pnode = static_cast<PipelineTreeNode*>(node);
        if (pnode->isPipelineNode()) {
          pipeline_to_depth[static_cast<const PipelineNode*>(pnode)] =
              pipeline_depth;
          pipeline_depth++;
        } else if (pnode->isStageNode()) {
          all_stages.push_back(static_cast<StageNode*>(pnode));
        }
        return node;
      });

  // Helper lambda to find a stage at the same level as the target pipeline by
  // walking up from start_stage
  auto findStageAtPipeline = [&](StageNode* start_stage,
                                 PipelineNode* target_pipeline) -> StageNode* {
    // Find the stage whose containing pipeline is at the target depth
    OperationTreeNode* result_stage =
        start_stage->findParent([&](const OperationTreeNode* node) {
          if (!static_cast<const PipelineTreeNode*>(node)->isStageNode()) {
            return false;
          }
          const StageNode* stage = static_cast<const StageNode*>(node);
          const PipelineTreeNode* parent =
              static_cast<const PipelineTreeNode*>(stage->getParentNode());
          assert(parent->isPipelineNode() && "Stage parent must be a pipeline");
          const PipelineNode* stage_pipeline =
              static_cast<const PipelineNode*>(parent);
          return stage_pipeline == target_pipeline;
        });

    return static_cast<StageNode*>(result_stage);
  };

  // Step 2: Process each stage and correct boundary-crossing dependencies
  for (StageNode* source_stage : all_stages) {
    PipelineTreeNode* source_parent =
        static_cast<PipelineTreeNode*>(source_stage->getParentNode());
    assert(source_parent->isPipelineNode() &&
           "Stage parent must be a pipeline");
    PipelineNode* source_pipeline = static_cast<PipelineNode*>(source_parent);
    unsigned source_depth = pipeline_to_depth[source_pipeline];

    for (StageNode* sink_stage : source_stage->getDependencies()) {
      PipelineTreeNode* sink_parent =
          static_cast<PipelineTreeNode*>(sink_stage->getParentNode());
      assert(sink_parent->isPipelineNode() &&
             "Stage parent must be a pipeline");
      PipelineNode* sink_pipeline = static_cast<PipelineNode*>(sink_parent);
      unsigned sink_depth = pipeline_to_depth[sink_pipeline];

      // If source and sink are in the same pipeline, nothing to do
      if (source_pipeline == sink_pipeline) continue;

      // If source is at lower depth than sink (outer -> inner)
      if (source_depth < sink_depth) {
        LLVM_DEBUG(llvm::dbgs()
                   << "    Correcting outer->inner dependency: Stage "
                   << source_stage->getStageId() << " (depth " << source_depth
                   << ") -> Stage " << sink_stage->getStageId() << " (depth "
                   << sink_depth << ")\n");

        // Walk up from sink to find stage at source's depth
        StageNode* new_sink = findStageAtPipeline(sink_stage, source_pipeline);
        if (new_sink) {
          LLVM_DEBUG(llvm::dbgs() << "      Redirected to Stage "
                                  << new_sink->getStageId() << "\n");
          source_stage->replaceDependency(sink_stage, new_sink);
        }
      }
      // If source is at higher depth than sink (inner -> outer)
      else if (source_depth > sink_depth) {
        LLVM_DEBUG(llvm::dbgs()
                   << "    Correcting inner->outer dependency: Stage "
                   << source_stage->getStageId() << " (depth " << source_depth
                   << ") -> Stage " << sink_stage->getStageId() << " (depth "
                   << sink_depth << ")\n");

        // Walk up from source to find stage at sink's depth
        StageNode* new_source =
            findStageAtPipeline(source_stage, sink_pipeline);
        if (new_source) {
          LLVM_DEBUG(llvm::dbgs() << "      Moved dependency to Stage "
                                  << new_source->getStageId() << "\n");

          // Add dependency to new source (if not already present)
          const llvm::SmallVector<StageNode*>& new_source_deps =
              new_source->getDependencies();

          bool already_has_dep = false;
          for (StageNode* existing_dep : new_source_deps) {
            if (existing_dep == sink_stage) {
              already_has_dep = true;
              break;
            }
          }

          if (!already_has_dep) {
            new_source->addDependency(sink_stage);
          }

          // Mark dependency for removal by nullifying it
          source_stage->nullifyDependency(sink_stage);
        }
      }
    }

    // Remove all nullified dependencies
    source_stage->removeNullifiedDependencies();
  }

  LLVM_DEBUG(llvm::dbgs() << "  Stage dependency correction complete.\n");
}

//===----------------------------------------------------------------------===//
// Step 3: Generate Transformed IR from PipelineTree
//===----------------------------------------------------------------------===//

void StageCoarseningPass::GenerateTransformedIR(
    TransformCandidate& candidate, PipelineTree& tree,
    llvm::ArrayRef<const scheduler::BufferExpansionInfo*> expansion_infos) {
  LLVM_DEBUG(llvm::dbgs() << "\n--- Step 3: Generate Transformed IR ---\n");

  // Get the outer pipeline node from the tree
  PipelineTreeNode* outer_pipeline =
      tree.getNodeForOp(candidate.pipeline_.getOperation());
  assert(outer_pipeline);
  assert(outer_pipeline->isPipelineNode());

  // Create a builder at the location of the root (ie right before outermost
  // loop in the original candidate loop nest)
  Operation* root = candidate.loop_nest_.front();
  mlir::OpBuilder builder(root);

  // Create an IRMapping to track value mappings during cloning
  mlir::IRMapping value_map;

  // Create buffer cloner if we have buffers to expand
  scheduler::materializer::BufferCloner* buffer_cloner = nullptr;
  std::optional<scheduler::StageCoarseningBufferCloner> cloner_storage;
  if (!expansion_infos.empty()) {
    LLVM_DEBUG({
      llvm::dbgs() << "  Creating buffer cloner for " << expansion_infos.size()
                   << " buffer(s)\n";
    });
    cloner_storage.emplace(expansion_infos);
    buffer_cloner = &cloner_storage.value();
  }

  // Use the materializer to generate IR from the tree
  scheduler::PipelineTreeMaterializer materializer(builder, value_map,
                                                   buffer_cloner);
  mlir::Operation* new_outer_pipeline =
      materializer.materialize(*outer_pipeline);

  // Erase the original candidate loop nest and all the old ops under it
  root->erase();

  DEBUG_WITH_TYPE(VerboseDebug, {
    llvm::dbgs() << "IR after materialization:\n";
    new_outer_pipeline->dump();
  });

  // Fix scoping of private results that are only used in inner pipelines
  fixPrivateResultScoping(builder, new_outer_pipeline);

  DEBUG_WITH_TYPE(VerboseDebug, {
    llvm::dbgs() << "IR after private scoping correction:\n";
    new_outer_pipeline->dump();
  });

  LLVM_DEBUG(llvm::dbgs() << "  IR generation complete.\n");
}

//===----------------------------------------------------------------------===//
// Fix Private Result Scoping
//===----------------------------------------------------------------------===//

/// Analyze and fix the scoping of SSA values from outer pipeline's ktdf.private
/// that are only used in inner pipeline stages. Such values should be moved to
/// the inner pipeline's private node.
void StageCoarseningPass::fixPrivateResultScoping(
    mlir::OpBuilder& builder, mlir::Operation* outer_pipeline_op) {
  LLVM_DEBUG(llvm::dbgs() << "\n--- Fixing Private Result Scoping ---\n");

  // Find the outer pipeline's private operation
  auto outer_pipeline = mlir::cast<mlir::ktdf::PipelineOp>(outer_pipeline_op);
  mlir::ktdf::PrivateOp outer_private = nullptr;
  outer_pipeline.walk([&](mlir::ktdf::PrivateOp priv_op) {
    if (priv_op->getParentOp() == outer_pipeline.getOperation()) {
      outer_private = priv_op;
      return mlir::WalkResult::interrupt();
    }
    return mlir::WalkResult::advance();
  });

  if (!outer_private) {
    LLVM_DEBUG(llvm::dbgs() << "  No outer private node found, skipping.\n");
    return;
  }

  // Find all inner pipelines
  llvm::SmallVector<mlir::ktdf::PipelineOp> inner_pipelines;
  outer_pipeline.walk([&](mlir::ktdf::PipelineOp inner_pipeline) {
    if (inner_pipeline != outer_pipeline) {
      inner_pipelines.push_back(inner_pipeline);
    }
    return mlir::WalkResult::advance();
  });

  if (inner_pipelines.empty()) {
    LLVM_DEBUG(llvm::dbgs() << "  No inner pipelines found, skipping.\n");
    return;
  }

  // Analyze which results need to be moved
  llvm::DenseMap<mlir::ktdf::PipelineOp, llvm::SmallVector<unsigned>>
      pipeline_to_result_indices;

  analyzePrivateResultUsage(outer_private, outer_pipeline, inner_pipelines,
                            pipeline_to_result_indices);

  if (pipeline_to_result_indices.empty()) {
    LLVM_DEBUG(llvm::dbgs() << "  No results need to be moved.\n");
    return;
  }

  // Move results to their respective inner pipelines
  for (auto& [inner_pipeline, result_indices] : pipeline_to_result_indices) {
    movePrivateResultsToInnerPipeline(builder, outer_private, inner_pipeline,
                                      result_indices);
  }
}

/// Analyze which private results are only used in specific inner pipelines
void StageCoarseningPass::analyzePrivateResultUsage(
    mlir::ktdf::PrivateOp outer_private, mlir::ktdf::PipelineOp outer_pipeline,
    llvm::ArrayRef<mlir::ktdf::PipelineOp> inner_pipelines,
    llvm::DenseMap<mlir::ktdf::PipelineOp, llvm::SmallVector<unsigned>>&
        pipeline_to_result_indices) {
  for (auto [idx, result] : llvm::enumerate(outer_private.getResults())) {
    // Skip tokens - they're for stage dependencies
    if (mlir::isa<mlir::ktdf::TokenType>(result.getType())) {
      continue;
    }

    // Check all uses of this result
    bool only_used_in_inner = true;
    mlir::ktdf::PipelineOp target_inner_pipeline = nullptr;

    for (mlir::OpOperand& use : result.getUses()) {
      mlir::Operation* user = use.getOwner();

      // Find which pipeline (if any) contains this use
      mlir::ktdf::PipelineOp containing_pipeline =
          user->getParentOfType<mlir::ktdf::PipelineOp>();

      // If use is not in an inner pipeline, we can't move this result
      if (!containing_pipeline || containing_pipeline == outer_pipeline) {
        only_used_in_inner = false;
        break;
      }

      // Check if all uses are in the same inner pipeline
      if (!target_inner_pipeline) {
        target_inner_pipeline = containing_pipeline;
      } else if (target_inner_pipeline != containing_pipeline) {
        // Used in multiple inner pipelines, can't move
        only_used_in_inner = false;
        break;
      }
    }

    if (only_used_in_inner && target_inner_pipeline) {
      LLVM_DEBUG(llvm::dbgs()
                 << "  Result #" << idx << " only used in inner pipeline\n");
      pipeline_to_result_indices[target_inner_pipeline].push_back(idx);
    }
  }
}

/// Move specified private results from outer private to inner pipeline's
/// private
void StageCoarseningPass::movePrivateResultsToInnerPipeline(
    mlir::OpBuilder& builder, mlir::ktdf::PrivateOp outer_private,
    mlir::ktdf::PipelineOp inner_pipeline,
    llvm::ArrayRef<unsigned> result_indices) {
  LLVM_DEBUG(llvm::dbgs() << "  Moving " << result_indices.size()
                          << " result(s) to inner pipeline\n");

  // Find or create the inner pipeline's private operation
  mlir::ktdf::PrivateOp inner_private = nullptr;
  for (mlir::Operation& op : inner_pipeline.getBody().front()) {
    if (auto priv = mlir::dyn_cast<mlir::ktdf::PrivateOp>(&op)) {
      inner_private = priv;
      break;
    }
  }

  // Collect the operations to move and their corresponding results
  llvm::SmallVector<mlir::Operation*> ops_to_move;
  llvm::SmallVector<mlir::Value> old_results_to_move;

  auto yield_op = mlir::cast<mlir::ktdf::PrivateYieldOp>(
      outer_private.getBody().front().getTerminator());

  for (unsigned idx : result_indices) {
    mlir::Value result = outer_private.getResult(idx);
    old_results_to_move.push_back(result);

    // Find the operation that produces this yielded value
    mlir::Value yielded_val = yield_op.getOperand(idx);
    if (mlir::Operation* def_op = yielded_val.getDefiningOp()) {
      ops_to_move.push_back(def_op);
    }
  }

  if (inner_private) {
    // Inner pipeline already has a private node - extend it
    extendInnerPrivate(builder, inner_private, ops_to_move,
                       old_results_to_move);
  } else {
    // Create a new private node for the inner pipeline
    createInnerPrivate(builder, inner_pipeline, ops_to_move,
                       old_results_to_move);
  }

  // Update the outer private to remove the moved results
  updateOuterPrivate(builder, outer_private, result_indices);
}

/// Extend an existing inner private node with moved allocations
void StageCoarseningPass::extendInnerPrivate(
    mlir::OpBuilder& builder, mlir::ktdf::PrivateOp inner_private,
    llvm::ArrayRef<mlir::Operation*> ops_to_move,
    llvm::SmallVectorImpl<mlir::Value>& old_results) {
  LLVM_DEBUG(llvm::dbgs() << "    Extending existing inner private\n");

  // Get the old yield first to preserve existing operands
  auto old_yield = mlir::cast<mlir::ktdf::PrivateYieldOp>(
      inner_private.getBody().front().getTerminator());
  llvm::SmallVector<mlir::Value> existing_yield_operands(
      old_yield.getOperands().begin(), old_yield.getOperands().end());

  // Clone moved operations at the beginning of the inner private body
  builder.setInsertionPointToStart(&inner_private.getBody().front());
  mlir::IRMapping value_map;
  llvm::SmallVector<mlir::Value> new_values;

  for (mlir::Operation* op : ops_to_move) {
    mlir::Operation* cloned = builder.clone(*op, value_map);
    new_values.push_back(cloned->getResult(0));
  }

  // Build new yield operands: moved results first, then existing results
  llvm::SmallVector<mlir::Value> new_yield_operands;
  new_yield_operands.append(new_values.begin(), new_values.end());
  new_yield_operands.append(existing_yield_operands.begin(),
                            existing_yield_operands.end());

  // Build new result types
  llvm::SmallVector<mlir::Type> new_result_types;
  for (mlir::Value val : new_yield_operands) {
    new_result_types.push_back(val.getType());
  }

  // Create new private op with extended results
  builder.setInsertionPoint(inner_private);
  auto new_inner_private = mlir::ktdf::PrivateOp::create(
      builder, inner_private.getLoc(), new_result_types);

  // Move operations from old body to new body (excluding terminator)
  mlir::Block& new_body = new_inner_private.getBody().front();
  mlir::Block& old_body = inner_private.getBody().front();
  new_body.getOperations().splice(new_body.begin(), old_body.getOperations(),
                                  old_body.begin(), std::prev(old_body.end()));

  // Create new yield in the new body
  builder.setInsertionPointToEnd(&new_body);
  mlir::ktdf::PrivateYieldOp::create(builder, old_yield.getLoc(),
                                     new_yield_operands);

  // Map moved results to new results (now at the beginning)
  for (unsigned i = 0; i < old_results.size(); ++i) {
    old_results[i].replaceAllUsesWith(new_inner_private.getResult(i));
  }

  // Replace uses of old existing results with new results (shifted by moved
  // count)
  unsigned moved_count = old_results.size();
  for (unsigned i = 0; i < inner_private.getNumResults(); ++i) {
    inner_private.getResult(i).replaceAllUsesWith(
        new_inner_private.getResult(moved_count + i));
  }

  inner_private.erase();
}

/// Create a new inner private node with moved allocations
void StageCoarseningPass::createInnerPrivate(
    mlir::OpBuilder& builder, mlir::ktdf::PipelineOp inner_pipeline,
    llvm::ArrayRef<mlir::Operation*> ops_to_move,
    llvm::SmallVectorImpl<mlir::Value>& old_results) {
  LLVM_DEBUG(llvm::dbgs() << "    Creating new inner private\n");

  // Determine result types from operations to move
  llvm::SmallVector<mlir::Type> result_types;
  for (mlir::Operation* op : ops_to_move) {
    result_types.push_back(op->getResult(0).getType());
  }

  // Create the private operation at the start of inner pipeline
  builder.setInsertionPointToStart(&inner_pipeline.getBody().front());
  auto new_private = mlir::ktdf::PrivateOp::create(
      builder, inner_pipeline.getLoc(), result_types);

  // Clone operations into the private body
  mlir::Block& private_body = new_private.getBody().front();
  builder.setInsertionPointToStart(&private_body);
  mlir::IRMapping value_map;
  llvm::SmallVector<mlir::Value> new_values;

  for (mlir::Operation* op : ops_to_move) {
    mlir::Operation* cloned = builder.clone(*op, value_map);
    new_values.push_back(cloned->getResult(0));
  }

  // Create yield operation
  mlir::ktdf::PrivateYieldOp::create(builder, inner_pipeline.getLoc(),
                                     new_values);

  // Replace all uses of old results with new results
  for (unsigned i = 0; i < old_results.size(); ++i) {
    old_results[i].replaceAllUsesWith(new_private.getResult(i));
  }
}

/// Update the outer private to remove moved results
void StageCoarseningPass::updateOuterPrivate(
    mlir::OpBuilder& builder, mlir::ktdf::PrivateOp outer_private,
    llvm::ArrayRef<unsigned> removed_indices) {
  LLVM_DEBUG(llvm::dbgs() << "    Updating outer private\n");

  // Build set of indices to remove for O(1) lookup
  llvm::DenseSet<unsigned> to_remove(removed_indices.begin(),
                                     removed_indices.end());

  // Get the yield operation
  auto old_yield = mlir::cast<mlir::ktdf::PrivateYieldOp>(
      outer_private.getBody().front().getTerminator());

  // Identify operations that define removed results (to be excluded from new
  // body)
  llvm::DenseSet<mlir::Operation*> ops_to_exclude;
  for (unsigned idx : removed_indices) {
    mlir::Value yielded_val = old_yield.getOperand(idx);
    if (mlir::Operation* def_op = yielded_val.getDefiningOp()) {
      ops_to_exclude.insert(def_op);
    }
  }

  // Build new yield operands and result types (excluding removed ones)
  llvm::SmallVector<mlir::Value> new_yield_operands;
  llvm::SmallVector<mlir::Type> new_result_types;

  for (auto [idx, operand] : llvm::enumerate(old_yield.getOperands())) {
    if (!to_remove.contains(idx)) {
      new_yield_operands.push_back(operand);
      new_result_types.push_back(operand.getType());
    }
  }

  // Create new private op with reduced results
  builder.setInsertionPoint(outer_private);
  auto new_outer_private = mlir::ktdf::PrivateOp::create(
      builder, outer_private.getLoc(), new_result_types);

  // Clone operations from old body to new body (excluding removed ops and
  // terminator)
  mlir::Block& new_body = new_outer_private.getBody().front();
  mlir::Block& old_body = outer_private.getBody().front();
  builder.setInsertionPointToStart(&new_body);
  mlir::IRMapping value_map;

  for (mlir::Operation& op : old_body.without_terminator()) {
    // Skip operations that produce removed results
    if (ops_to_exclude.contains(&op)) {
      LLVM_DEBUG(llvm::dbgs()
                 << "      Skipping stale operation: " << op << "\n");
      continue;
    }
    // Clone the operation
    builder.clone(op, value_map);
  }

  // Create new yield in the new body with mapped operands
  builder.setInsertionPointToEnd(&new_body);
  llvm::SmallVector<mlir::Value> mapped_yield_operands;
  for (mlir::Value operand : new_yield_operands) {
    mapped_yield_operands.push_back(value_map.lookupOrDefault(operand));
  }
  mlir::ktdf::PrivateYieldOp::create(builder, old_yield.getLoc(),
                                     mapped_yield_operands);

  // Map old results to new results (skipping removed ones)
  unsigned new_idx = 0;
  for (unsigned old_idx = 0; old_idx < outer_private.getNumResults();
       ++old_idx) {
    if (!to_remove.contains(old_idx)) {
      outer_private.getResult(old_idx).replaceAllUsesWith(
          new_outer_private.getResult(new_idx));
      new_idx++;
    }
  }

  outer_private.erase();
}

}  // namespace
}  // namespace ktdf

std::unique_ptr<mlir::Pass> ktdf::createStageCoarseningPass() {
  return std::make_unique<ktdf::StageCoarseningPass>();
}
