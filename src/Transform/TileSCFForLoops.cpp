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
// Example:
//   scf.for %i = 0 to 100 step 1 {
//     scf.for %j = 0 to 100 step 1 { ... }
//       use (%i, %j)
//   }
// becomes:
//   scf.for %i0 = 0 to 100 step 32 {
//     scf.for %j0 = 0 to 100 step 32 {
//       %ub_i = min(%i0 + 32, 100) - %i0 // ~= 32 (i's tile size)
//       scf.for %i1 = 0 to %ub_i step 1 {
//         %ub_j = min(%j0 + 32, 200) - %j0 // ~= 32 (j's tile size)
//         scf.for %j1 = %0 to ub_j step 1 { ... }
//            use (%i0 + %i1, %j0 + %j1)
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
    mlir::IRRewriter rewriter(op->getContext());

    // Collect all outermost scf.for loops (roots of perfectly nested loops)
    llvm::SmallVector<mlir::scf::ForOp> root_loops;
    op->walk([&](mlir::scf::ForOp for_op) {
      // Check if this is a root (not nested inside another scf.for)
      if (!for_op->getParentOfType<mlir::scf::ForOp>()) {
        root_loops.push_back(for_op);
      }
    });

    // Tile each perfectly nested loop nest
    for (mlir::scf::ForOp root_loop : root_loops) {
      tileLoopNest(root_loop, rewriter);
    }
  }

 private:
  void tileLoopNest(mlir::scf::ForOp root_loop, mlir::IRRewriter& rewriter) {
    // Collect the perfectly nested loops starting from root
    llvm::SmallVector<mlir::scf::ForOp> nested_loops;
    mlir::getPerfectlyNestedLoops(nested_loops, root_loop);

    if (nested_loops.empty()) return;

    // Prepare tile sizes as Values
    llvm::SmallVector<mlir::Value> tile_size_values;
    rewriter.setInsertionPoint(root_loop);
    mlir::Location loc = root_loop.getLoc();

    // Use configured tile sizes, or default to 32 for each loop
    for (size_t i = 0; i < nested_loops.size(); ++i) {
      int64_t tile_size = (i < tileSizes.size()) ? tileSizes[i] : 32;
      tile_size_values.push_back(
          mlir::arith::ConstantIndexOp::create(rewriter, loc, tile_size));
    }

    // Use MLIR's tilePerfectlyNested utility to tile the loop nest
    // This will create the outer tiled loops and inner point loops
    auto inner_loops = mlir::tilePerfectlyNested(root_loop, tile_size_values);

    // If normalization is enabled, normalize the inner loops
    if (normalizeInnerLoops && !inner_loops.empty()) {
      normalizeInnerTiledLoops(nested_loops, inner_loops, rewriter);
    }
  }

  void normalizeInnerTiledLoops(llvm::ArrayRef<mlir::scf::ForOp> original_loops,
                                llvm::ArrayRef<mlir::scf::ForOp> inner_loops,
                                mlir::IRRewriter& rewriter) {
    // For each inner loop, normalize it to start at 0 and denormalize the IV
    for (size_t i = 0; i < inner_loops.size() && i < original_loops.size();
         ++i) {
      mlir::scf::ForOp inner_loop = inner_loops[i];

      // Get the original lower bound and step before normalization
      mlir::Value orig_lb = inner_loop.getLowerBound();
      mlir::Value orig_step = inner_loop.getStep();
      mlir::Location loc = inner_loop.getLoc();

      // Use MLIR's emitNormalizedLoopBounds to compute normalized bounds
      rewriter.setInsertionPoint(inner_loop);
      mlir::Value ub = inner_loop.getUpperBound();
      auto normalized_range =
          mlir::emitNormalizedLoopBounds(rewriter, loc, orig_lb, ub, orig_step);

      // Update the loop to use normalized bounds (0 to size, step 1)
      inner_loop.setLowerBound(mlir::getValueOrCreateConstantIndexOp(
          rewriter, loc, normalized_range.offset));
      inner_loop.setUpperBound(mlir::getValueOrCreateConstantIndexOp(
          rewriter, loc, normalized_range.size));
      inner_loop.setStep(mlir::getValueOrCreateConstantIndexOp(
          rewriter, loc, normalized_range.stride));

      // Use MLIR's denormalizeInductionVariable to replace uses of the
      // normalized IV with: orig_lb + iv * orig_step
      rewriter.setInsertionPointToStart(inner_loop.getBody());
      mlir::denormalizeInductionVariable(
          rewriter, loc, inner_loop.getInductionVar(), orig_lb, orig_step);
    }
  }
};

}  // namespace

std::unique_ptr<mlir::Pass> createTileSCFForLoopsPass() {
  return std::make_unique<TileSCFForLoopsPass>();
}

}  // namespace ktdf

// Made with Bob
