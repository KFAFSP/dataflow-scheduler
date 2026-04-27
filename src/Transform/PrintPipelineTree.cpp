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
// Pass to print the Pipeline Tree structure for debugging and visualization
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#include "src/Transform/Passes.h"
#include "src/Transform/PipelineTree.hpp"
#include "llvm/Support/raw_ostream.h"

namespace ktdf {

#define GEN_PASS_DEF_PRINTPIPELINETREEPASS
#include "src/Transform/Passes.h.inc"

namespace {

struct PrintPipelineTreePass
    : public impl::PrintPipelineTreePassBase<PrintPipelineTreePass> {
  using impl::PrintPipelineTreePassBase<
      PrintPipelineTreePass>::PrintPipelineTreePassBase;

  void runOnOperation() override {
    auto moduleOp = getOperation();
    
    // Iterate over all functions in the module
    moduleOp.walk([&](mlir::func::FuncOp funcOp) {
      llvm::outs() << "=== Pipeline Tree for function: "
                   << funcOp.getName() << " ===\n";
      
      // Build the pipeline tree
      mlir::ktdf::PipelineTree tree;
      tree.compute(*funcOp.getOperation());
      
      // Print the tree structure
      if (tree.empty()) {
        llvm::outs() << "(empty tree)\n";
      } else {
        tree.print(llvm::outs(), showDependencies);
      }
      
      llvm::outs() << "=== End Pipeline Tree ===\n\n";
    });
  }
};

}  // namespace

std::unique_ptr<mlir::Pass> createPrintPipelineTreePass() {
  return std::make_unique<PrintPipelineTreePass>();
}

}  // namespace ktdf

// Made with Bob
