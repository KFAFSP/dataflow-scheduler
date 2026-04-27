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
    op->walk([&](mlir::scf::ForOp forOp) { allLoops.push_back(forOp); });

    // Strip-mine each loop individually
    for (mlir::scf::ForOp forOp : allLoops) {
      stripMineLoop(forOp, rewriter);
    }
  }

 private:
  void stripMineLoop(mlir::scf::ForOp forOp, mlir::IRRewriter& rewriter) {
    // Determine strip-mine factor (tile size)
    int64_t stripMineFactor = stripMineSize.empty() ? 32 : stripMineSize[0];

    // Create tile size as a Value
    rewriter.setInsertionPoint(forOp);
    mlir::Location loc = forOp.getLoc();
    mlir::Value tileSizeVal =
        mlir::arith::ConstantIndexOp::create(rewriter, loc, stripMineFactor);

    // Save original bounds and step for potential normalization
    mlir::Value origLb = forOp.getLowerBound();
    mlir::Value origStep = forOp.getStep();

    // Use MLIR's tile utility to strip-mine this single loop
    // Pass the loop as both the loop to tile and the target
    llvm::SmallVector<mlir::scf::ForOp> loopsToTile = {forOp};
    llvm::SmallVector<mlir::Value> tileSizes = {tileSizeVal};

    // The tile function performs strip-mining by creating an outer loop
    // that iterates over blocks and an inner loop that iterates within blocks
    auto innerLoops = mlir::tile(loopsToTile, tileSizes, forOp);

    // If normalization is enabled and we got an inner loop, normalize it
    if (normalizeInnerLoop && !innerLoops.empty()) {
      mlir::scf::ForOp innerLoop = innerLoops[0];
      
      // Get the current bounds of the inner loop
      mlir::Value innerLb = innerLoop.getLowerBound();
      mlir::Value innerUb = innerLoop.getUpperBound();
      mlir::Value innerStep = innerLoop.getStep();
      
      // Use MLIR's emitNormalizedLoopBounds to compute normalized bounds
      rewriter.setInsertionPoint(innerLoop);
      auto normalizedRange = mlir::emitNormalizedLoopBounds(
          rewriter, loc, innerLb, innerUb, innerStep);
      
      // Update the inner loop to use normalized bounds (0 to size, step 1)
      innerLoop.setLowerBound(
          mlir::getValueOrCreateConstantIndexOp(rewriter, loc, normalizedRange.offset));
      innerLoop.setUpperBound(
          mlir::getValueOrCreateConstantIndexOp(rewriter, loc, normalizedRange.size));
      innerLoop.setStep(
          mlir::getValueOrCreateConstantIndexOp(rewriter, loc, normalizedRange.stride));
      
      // Use MLIR's denormalizeInductionVariable to replace uses of the
      // normalized IV with: innerLb + iv * innerStep
      rewriter.setInsertionPointToStart(innerLoop.getBody());
      mlir::denormalizeInductionVariable(
          rewriter, loc, innerLoop.getInductionVar(), innerLb, innerStep);
    }
  }
};

}  // namespace

std::unique_ptr<mlir::Pass> createStripMineSCFForLoopsPass() {
  return std::make_unique<StripMineSCFForLoopsPass>();
}

}  // namespace ktdf

// Made with Bob
