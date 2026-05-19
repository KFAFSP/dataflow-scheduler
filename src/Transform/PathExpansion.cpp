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
#include "src/Transform/PathExpansion/Planner.hpp"

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
  mlir::WalkResult result = module_op.walk<mlir::WalkOrder::PostOrder>(
      [&](mlir::ktdf::PipelineOp pipeline_op) {
        if (mlir::failed(processPipeline(pipeline_op, arch_graph))) {
          return mlir::WalkResult::skip();
        }
        return mlir::WalkResult::advance();
      });
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

  // TODO: Implement materialization
  // For now, just report that changes are needed
  LLVM_DEBUG(llvm::dbgs() << "  Materialization not yet implemented\n");

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