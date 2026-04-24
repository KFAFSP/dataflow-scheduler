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
// This pass tiles perfectly nested scf.for loops.
// Example: scf.for %i = 0 to 100 step 1 {
//            scf.for %j = 0 to 100 step 1 { ... }
//          }
// becomes:
//   scf.for %i0 = 0 to 100 step 32 {
//     scf.for %j0 = 0 to 100 step 32 {
//       scf.for %i1 = %i0 to min(%i0+32, 100) step 1 {
//         scf.for %j1 = %j0 to min(%j0+32, 100) step 1 { ... }
//       }
//     }
//   }
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "src/Transform/Passes.h"

namespace ktdf {

#define GEN_PASS_DEF_TILESCFFORLOOPSPASS
#include "src/Transform/Passes.h.inc"

namespace {

struct TileSCFForLoopsPass
    : public impl::TileSCFForLoopsPassBase<TileSCFForLoopsPass> {
  using impl::TileSCFForLoopsPassBase<
      TileSCFForLoopsPass>::TileSCFForLoopsPassBase;

  void runOnOperation() override {
    mlir::Operation* op = getOperation();
    mlir::OpBuilder builder(op->getContext());

    // Collect all outermost scf.for loops (roots of perfectly nested loops)
    llvm::SmallVector<mlir::scf::ForOp> rootLoops;
    op->walk([&](mlir::scf::ForOp forOp) {
      // Check if this is a root (not nested inside another scf.for)
      if (!forOp->getParentOfType<mlir::scf::ForOp>()) {
        rootLoops.push_back(forOp);
      }
    });

    // Tile each perfectly nested loop nest
    for (mlir::scf::ForOp rootLoop : rootLoops) {
      tileLoopNest(rootLoop, builder);
    }
  }

 private:
  void tileLoopNest(mlir::scf::ForOp rootLoop, mlir::OpBuilder& builder) {
    // Collect the perfectly nested loops starting from root
    llvm::SmallVector<mlir::scf::ForOp> nestedLoops;
    mlir::getPerfectlyNestedLoops(nestedLoops, rootLoop);

    if (nestedLoops.empty()) return;

    // Prepare tile sizes as Values
    llvm::SmallVector<mlir::Value> tileSizeValues;
    builder.setInsertionPoint(rootLoop);
    mlir::Location loc = rootLoop.getLoc();

    // Use configured tile sizes, or default to 32 for each loop
    for (size_t i = 0; i < nestedLoops.size(); ++i) {
      int64_t tileSize = (i < tileSizes.size()) ? tileSizes[i] : 32;
      tileSizeValues.push_back(
          mlir::arith::ConstantIndexOp::create(builder, loc, tileSize));
    }

    // Use MLIR's tilePerfectlyNested utility to tile the loop nest
    // This will create the outer tiled loops and inner point loops
    mlir::tilePerfectlyNested(rootLoop, tileSizeValues);
  }
};

}  // namespace

std::unique_ptr<mlir::Pass> createTileSCFForLoopsPass() {
  return std::make_unique<TileSCFForLoopsPass>();
}

}  // namespace ktdf

// Made with Bob
