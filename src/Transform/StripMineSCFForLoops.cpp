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
// This pass performs strip-mining on scf.for loops.
// Strip-mining splits a loop into an outer loop and an inner loop,
// where the outer loop iterates over blocks and the inner loop
// iterates within each block.
//
// Example: scf.for %i = 0 to 100 step 1 { ... }
// becomes: scf.for %i0 = 0 to 100 step 32 {
//            scf.for %i1 = %i0 to min(%i0+32, 100) step 1 { ... }
//          }
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "src/Dialects/KTDF/KTDFOps.hpp"
#include "src/Transform/Passes.h"

namespace ktdf {

#define GEN_PASS_DEF_STRIPMINESCFFORLOOPSPASS
#include "src/Transform/Passes.h.inc"

namespace {

struct StripMineSCFForLoopsPass
    : public impl::StripMineSCFForLoopsPassBase<StripMineSCFForLoopsPass> {
  using impl::StripMineSCFForLoopsPassBase<
      StripMineSCFForLoopsPass>::StripMineSCFForLoopsPassBase;

  void runOnOperation() override {
    mlir::Operation* op = getOperation();
    mlir::IRRewriter rewriter(op->getContext());

    // Collect all scf.for loops
    llvm::SmallVector<mlir::scf::ForOp> allLoops;
    op->walk([&](mlir::scf::ForOp for_op) { allLoops.push_back(for_op); });

    // Strip-mine each loop individually
    for (mlir::scf::ForOp for_op : allLoops) {
      stripMineLoop(for_op, rewriter);
    }
  }

 private:
  void stripMineLoop(mlir::scf::ForOp for_op, mlir::IRRewriter& rewriter) {
    // Determine strip-mine factor (tile size)
    int64_t strip_mine_factor = stripMineSize.empty() ? 32 : stripMineSize[0];

    // Create tile size as a Value
    rewriter.setInsertionPoint(for_op);
    mlir::Location loc = for_op.getLoc();
    mlir::Value tile_size_val;
    
    // If strip-mine factor is negative, search for ktdf.create_tile_size
    if (strip_mine_factor < 0) {
      // Collect all create_tile_size ops before this loop
      llvm::SmallVector<mlir::Value> available_tile_sizes;
      mlir::Block* block = for_op->getBlock();
      
      for (mlir::Operation& op : block->getOperations()) {
        if (&op == for_op.getOperation()) break;
        if (auto create_tile_op = mlir::dyn_cast<mlir::ktdf::CreateTileSizeOp>(&op)) {
          available_tile_sizes.push_back(create_tile_op.getResult());
        }
      }
      
      // If not enough in current block, search parent regions
      if (available_tile_sizes.empty()) {
        mlir::Operation* parent_op = block->getParentOp();
        while (parent_op && !mlir::isa<mlir::ModuleOp>(parent_op) &&
               available_tile_sizes.empty()) {
          mlir::Block* parent_block = parent_op->getBlock();
          for (mlir::Operation& op : parent_block->getOperations()) {
            if (&op == parent_op) break;
            if (auto create_tile_op = mlir::dyn_cast<mlir::ktdf::CreateTileSizeOp>(&op)) {
              available_tile_sizes.push_back(create_tile_op.getResult());
            }
          }
          parent_op = parent_block->getParentOp();
        }
      }
      
      // Use the first available create_tile_size, or default to 32
      if (!available_tile_sizes.empty()) {
        tile_size_val = available_tile_sizes[0];
      } else {
        tile_size_val = mlir::arith::ConstantIndexOp::create(rewriter, loc, 32);
      }
    } else {
      tile_size_val = mlir::arith::ConstantIndexOp::create(rewriter, loc, strip_mine_factor);
    }

    // Save original bounds and step for potential normalization
    mlir::Value orig_lb = for_op.getLowerBound();
    mlir::Value orig_step = for_op.getStep();

    // Use MLIR's tile utility to strip-mine this single loop
    // Pass the loop as both the loop to tile and the target
    llvm::SmallVector<mlir::scf::ForOp> loops_to_tile = {for_op};
    llvm::SmallVector<mlir::Value> tile_sizes = {tile_size_val};

    // The tile function performs strip-mining by creating an outer loop
    // that iterates over blocks and an inner loop that iterates within blocks
    auto inner_loops = mlir::tile(loops_to_tile, tile_sizes, for_op);

    // If normalization is enabled and we got an inner loop, normalize it
    if (normalizeInnerLoop && !inner_loops.empty()) {
      mlir::scf::ForOp inner_loop = inner_loops[0];

      // Get the current bounds of the inner loop
      mlir::Value inner_lb = inner_loop.getLowerBound();
      mlir::Value inner_ub = inner_loop.getUpperBound();
      mlir::Value inner_step = inner_loop.getStep();

      // Use MLIR's emitNormalizedLoopBounds to compute normalized bounds
      rewriter.setInsertionPoint(inner_loop);
      auto normalized_range = mlir::emitNormalizedLoopBounds(
          rewriter, loc, inner_lb, inner_ub, inner_step);

      // Update the inner loop to use normalized bounds (0 to size, step 1)
      inner_loop.setLowerBound(mlir::getValueOrCreateConstantIndexOp(
          rewriter, loc, normalized_range.offset));
      inner_loop.setUpperBound(mlir::getValueOrCreateConstantIndexOp(
          rewriter, loc, normalized_range.size));
      inner_loop.setStep(mlir::getValueOrCreateConstantIndexOp(
          rewriter, loc, normalized_range.stride));

      // Use MLIR's denormalizeInductionVariable to replace uses of the
      // normalized IV with: inner_lb + iv * inner_step
      rewriter.setInsertionPointToStart(inner_loop.getBody());
      mlir::denormalizeInductionVariable(
          rewriter, loc, inner_loop.getInductionVar(), inner_lb, inner_step);
    }
  }
};

}  // namespace

std::unique_ptr<mlir::Pass> createStripMineSCFForLoopsPass() {
  return std::make_unique<StripMineSCFForLoopsPass>();
}

}  // namespace ktdf

// Made with Bob
