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
// Path Expansion Pass
//
// This pass legalizes data movement in ktdf.pipeline operations using an
// explicit architecture graph. It expands illegal direct transfers into
// legal multi-hop paths by inserting intermediate stages and resources.
//===----------------------------------------------------------------------===//

#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "src/Analysis/ArchitectureGraph.hpp"
#include "src/Analysis/PipelineTree.hpp"
#include "src/Dialects/KTDF/KTDFOps.hpp"
#include "src/Transform/Passes.h"
#include "src/Transform/PathExpansion/Materializer.hpp"
#include "src/Transform/PathExpansion/Planner.hpp"
#include "src/Transform/Utils/Utils.h"
#include "src/Utils/OperationUtils.hpp"

namespace ktdf {

#define GEN_PASS_DEF_PATHEXPANSIONPASS
#include "src/Transform/Passes.h.inc"

#define PASS_NAME "path-expansion"
#define DEBUG_TYPE PASS_NAME

namespace {

//===----------------------------------------------------------------------===//
// Path Expansion Pass
//===----------------------------------------------------------------------===//

struct PathExpansionPass
    : public impl::PathExpansionPassBase<PathExpansionPass> {
  using impl::PathExpansionPassBase<PathExpansionPass>::PathExpansionPassBase;

  void runOnOperation() override;

 private:
  /// Process a single pipeline operation
  mlir::LogicalResult processPipeline(
      mlir::ktdf::PipelineOp pipeline_op,
      const scheduler::ArchitectureGraph& arch_graph);
};

void PathExpansionPass::runOnOperation() {
  mlir::ModuleOp module_op = getOperation();

  LLVM_DEBUG(llvm::dbgs() << "=== Path Expansion Pass ===\n");

  // Create the built-in Spyre architecture graph
  scheduler::ArchitectureGraph arch_graph =
      scheduler::ArchitectureGraph::createBuiltInSpyre();

  LLVM_DEBUG({
    llvm::dbgs() << "Architecture Graph:\n";
    arch_graph.dump(llvm::dbgs());
  });

  // Walk all pipeline operations in the module
  llvm::SmallVector<mlir::ktdf::PipelineOp> candidates;
  module_op.walk<mlir::WalkOrder::PostOrder>(
      [&](mlir::ktdf::PipelineOp pipeline_op) {
        candidates.push_back(pipeline_op);
      });
  for (mlir::ktdf::PipelineOp cand : candidates) {
    if (mlir::failed(processPipeline(cand, arch_graph))) {
      LLVM_DEBUG(llvm::dbgs() << "Unable to process pipeline \n";
                 scheduler::utils::printLocation(llvm::dbgs(), cand));
    }
  }
}

mlir::LogicalResult PathExpansionPass::processPipeline(
    mlir::ktdf::PipelineOp pipeline_op,
    const scheduler::ArchitectureGraph& arch_graph) {
  LLVM_DEBUG(llvm::dbgs() << "Processing pipeline at " << pipeline_op.getLoc()
                          << "\n");

  // Build PipelineTree for this pipeline
  PipelineTree tree;
  mlir::Operation* parent = pipeline_op->getParentOp();
  if (!parent) {
    return mlir::failure();
  }

  tree.compute(*parent);

  // Find the pipeline node in the tree
  PipelineNode* pipeline_node =
      static_cast<PipelineNode*>(tree.getNodeForOp(pipeline_op));

  // Plan path expansion
  auto plan = planPathExpansion(tree, pipeline_node, arch_graph);
  if (!plan) {
    LLVM_DEBUG(llvm::dbgs() << "  Planning failed\n");
    return mlir::failure();
  }

  if (!plan->changed) {
    LLVM_DEBUG(llvm::dbgs() << "  Pipeline already legal, no changes needed\n");
    return mlir::success();
  }

  LLVM_DEBUG(llvm::dbgs() << "  Pipeline needs expansion\n");

  // Materialize the modified pipeline tree
  mlir::OpBuilder builder(pipeline_op.getContext());
  mlir::IRMapping value_map;

  scheduler::PathExpansionMaterializer materializer(
      builder, value_map, plan->stage_info, plan->resource_factory);

  // Set insertion point before the original pipeline
  builder.setInsertionPoint(pipeline_op);

  // Materialize the new pipeline from the modified tree
  mlir::Operation* new_pipeline_op = materializer.materialize(*pipeline_node);

  if (!new_pipeline_op) {
    LLVM_DEBUG(llvm::dbgs() << "  Materialization failed\n");
    return mlir::failure();
  }

  LLVM_DEBUG(llvm::dbgs() << "  Materialization succeeded\n");

  auto new_pipeline = mlir::dyn_cast<mlir::ktdf::PipelineOp>(new_pipeline_op);
  if (!new_pipeline) {
    return mlir::failure();
  }

  if (mlir::failed(
          scheduler::utils::cleanupPrivateOpsInPipeline(new_pipeline))) {
    LLVM_DEBUG(llvm::dbgs() << "  Private cleanup failed\n");
    return mlir::failure();
  }

  // Erase the original pipeline
  pipeline_op.erase();

  return mlir::success();
}

}  // namespace

//===----------------------------------------------------------------------===//
// Pass Registration
//===----------------------------------------------------------------------===//

std::unique_ptr<mlir::Pass> createPathExpansionPass() {
  return std::make_unique<PathExpansionPass>();
}

}  // namespace ktdf

// Made with Bob