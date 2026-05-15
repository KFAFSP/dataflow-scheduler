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
// Pass: -parallelize-loops-across-instances
//
// Rewrites parallel scf.for loops enclosing per-corelet ktdf.pipelines into
// ktdf.parallel ops that distribute iterations across hardware instances.
// Target abstraction is via SchedulerExtContext (getNumCorelets,
// isPerCoreletUnit); the pass code itself names no hardware specifics.
//===----------------------------------------------------------------------===//

#include <optional>

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "src/Dialects/KTDF/KTDFOps.hpp"
#include "src/Transform/Common/ApplicableUnits.hpp"
#include "src/Transform/Common/PipelineScope.hpp"
#include "src/Transform/Passes.h"
#include "src/Utils/SchedulerExtContext.h"

#define PASS_NAME "parallelize-loops-across-instances"
#define DEBUG_TYPE PASS_NAME

namespace ktdf {
#define GEN_PASS_DEF_PARALLELIZELOOPSACROSSINSTANCESPASS
#include "src/Transform/Passes.h.inc"
}  // namespace ktdf

using namespace mlir;

namespace {

//===----------------------------------------------------------------------===//
// Candidate
//===----------------------------------------------------------------------===//

struct Candidate {
  mlir::ktdf::PipelineOp pipeline;
  mlir::scf::ForOp loop;
  int64_t trip_count;
  bool balanced;
};

//===----------------------------------------------------------------------===//
// Analysis helpers
//===----------------------------------------------------------------------===//

/// Returns the static trip count of `loop` iff lb, ub, step are all
/// arith.constant index ops. step must be positive.
std::optional<int64_t> getStaticTripCount(scf::ForOp loop) {
  auto lb_const = loop.getLowerBound().getDefiningOp<arith::ConstantIndexOp>();
  auto ub_const = loop.getUpperBound().getDefiningOp<arith::ConstantIndexOp>();
  auto step_const = loop.getStep().getDefiningOp<arith::ConstantIndexOp>();
  if (!lb_const || !ub_const || !step_const) {
    return std::nullopt;
  }
  int64_t lb = lb_const.value();
  int64_t ub = ub_const.value();
  int64_t step = step_const.value();
  if (step <= 0) {
    return std::nullopt;
  }
  if (ub <= lb) {
    return 0;
  }
  return (ub - lb + step - 1) / step;
}

/// True iff `loop` carries `loop_type = #ktdf.loop_type<parallel_loop>`.
bool isParallelLoop(scf::ForOp loop) {
  auto attr = loop->getAttrOfType<mlir::ktdf::LoopTypeAttr>("loop_type");
  if (!attr) {
    return false;
  }
  return attr.getValue() == mlir::ktdf::LoopType::ParallelLoop;
}

/// True iff every unit in the pipeline's effective applicable_units is
/// per-corelet per `ctx`. Returns false if the effective set is empty.
/// Triggers report_fatal_error if any immediate stage lacks the attribute.
bool allUnitsAreCoreletLocal(mlir::ktdf::PipelineOp pipeline,
                             const scheduler::SchedulerExtContext& ctx) {
  auto units = ::ktdf::common::computePipelineApplicableUnits(pipeline);
  if (units.empty()) {
    return false;
  }
  for (mlir::ktdf::ComponentType unit : units) {
    if (!ctx.isPerCoreletUnit(unit)) {
      return false;
    }
  }
  return true;
}

/// Walk scope_loops outermost-to-innermost; return the first loop matching
/// the predicate `pred(loop, trip_count)`.
template <typename Pred>
std::optional<std::pair<scf::ForOp, int64_t>> firstMatching(
    llvm::ArrayRef<scf::ForOp> scope_loops_innermost_first, Pred pred) {
  for (scf::ForOp loop : llvm::reverse(scope_loops_innermost_first)) {
    if (!isParallelLoop(loop)) {
      continue;
    }
    auto trip = getStaticTripCount(loop);
    if (!trip.has_value()) {
      continue;
    }
    if (pred(loop, *trip)) {
      return std::make_pair(loop, *trip);
    }
  }
  return std::nullopt;
}

/// Two-pass loop selection: balanced first, imbalanced fallback.
std::optional<std::pair<scf::ForOp, int64_t>> selectLoop(
    llvm::ArrayRef<scf::ForOp> scope_loops_innermost_first,
    unsigned num_instances) {
  // Pass 1: trip >= N and trip % N == 0.
  auto balanced = firstMatching(
      scope_loops_innermost_first,
      [num_instances](scf::ForOp, int64_t trip) {
        return trip >= static_cast<int64_t>(num_instances) &&
               trip % static_cast<int64_t>(num_instances) == 0;
      });
  if (balanced.has_value()) {
    return balanced;
  }
  // Pass 2: trip >= N (drop divisibility).
  return firstMatching(
      scope_loops_innermost_first,
      [num_instances](scf::ForOp, int64_t trip) {
        return trip >= static_cast<int64_t>(num_instances);
      });
}

std::optional<Candidate> findCandidate(
    mlir::ktdf::PipelineOp pipeline,
    const scheduler::SchedulerExtContext& ctx) {
  // Step 1+2: applicable_units.
  if (!allUnitsAreCoreletLocal(pipeline, ctx)) {
    LLVM_DEBUG(llvm::dbgs() << "[" PASS_NAME "] skip pipeline at "
                            << pipeline.getLoc()
                            << ": empty or non-corelet applicable_units\n");
    return std::nullopt;
  }

  // Step 3: enclosing scope.
  auto scope = ::ktdf::common::computePipelineEnclosingScope(pipeline);
  if (scope.loops.empty()) {
    LLVM_DEBUG(llvm::dbgs() << "[" PASS_NAME "] skip pipeline at "
                            << pipeline.getLoc()
                            << ": no enclosing scf.for scope\n");
    return std::nullopt;
  }

  // Step 4: loop selection.
  unsigned num_instances = ctx.getNumCorelets();
  auto picked = selectLoop(scope.loops, num_instances);
  if (!picked.has_value()) {
    LLVM_DEBUG(llvm::dbgs() << "[" PASS_NAME "] skip pipeline at "
                            << pipeline.getLoc()
                            << ": no parallel loop in scope satisfies trip >= "
                            << num_instances << "\n");
    return std::nullopt;
  }
  scf::ForOp loop = picked->first;
  int64_t trip = picked->second;
  bool balanced = (trip % static_cast<int64_t>(num_instances)) == 0;

  if (!balanced) {
    int64_t chunk = (trip + num_instances - 1) / num_instances;
    int64_t leftover = trip - (num_instances - 1) * chunk;
    LLVM_DEBUG(llvm::dbgs()
               << "[" PASS_NAME "] imbalanced (fallback) at "
               << loop.getLoc() << ": trip=" << trip
               << ", num_instances=" << num_instances
               << ", chunks=" << chunk << "," << leftover << "\n");
  } else {
    LLVM_DEBUG(llvm::dbgs() << "[" PASS_NAME "] balanced candidate at "
                            << loop.getLoc() << ": trip=" << trip
                            << ", num_instances=" << num_instances << "\n");
  }

  return Candidate{pipeline, loop, trip, balanced};
}

//===----------------------------------------------------------------------===//
// Transformation
//===----------------------------------------------------------------------===//

void rewrite(const Candidate& c, unsigned num_instances) {
  scf::ForOp loop = c.loop;
  Location loc = loop.getLoc();
  Value lb = loop.getLowerBound();
  Value ub = loop.getUpperBound();
  Value step = loop.getStep();
  Value old_iv = loop.getInductionVar();
  Block* old_body = loop.getBody();

  // Build ktdf.parallel at the scf.for's position. The custom builder
  // creates the body block with two index args (IV + instance id).
  OpBuilder builder(loop);
  auto parallel = mlir::ktdf::ParallelOp::create(
      builder, loc,
      /*lowerBounds=*/ValueRange{lb},
      /*upperBounds=*/ValueRange{ub},
      /*steps=*/ValueRange{step},
      /*numInstances=*/static_cast<int64_t>(num_instances));

  Block* new_body = &parallel.getBody().front();

  // Splice the old body (excluding the trailing scf.yield terminator) into
  // the new ktdf.parallel body.
  new_body->getOperations().splice(
      new_body->begin(), old_body->getOperations(), old_body->begin(),
      std::prev(old_body->end()));

  // RAUW the original IV with the new parallel's first block argument.
  Value new_iv = parallel.getInductionVar(0);
  old_iv.replaceAllUsesWith(new_iv);

  // Append ktdf.parallel_yield as the new body's terminator.
  builder.setInsertionPointToEnd(new_body);
  mlir::ktdf::ParallelYieldOp::create(builder, loc);

  // Erase the original scf.for. Its body is now empty (only the orphaned
  // scf.yield remains, which is erased with the parent).
  loop.erase();

#ifndef NDEBUG
  if (failed(parallel.verify())) {
    llvm::report_fatal_error("ktdf.parallel verifier failed after rewrite");
  }
#endif
}

//===----------------------------------------------------------------------===//
// Pass entry
//===----------------------------------------------------------------------===//

struct ParallelizeLoopsAcrossInstancesPass
    : public ::ktdf::impl::ParallelizeLoopsAcrossInstancesPassBase<
          ParallelizeLoopsAcrossInstancesPass> {
  ParallelizeLoopsAcrossInstancesPass(
      const scheduler::SchedulerExtContext& scheduler_ctx)
      : scheduler_ctx_(scheduler_ctx) {}

  void runOnOperation() override {
    ModuleOp module = getOperation();
    unsigned num_instances = scheduler_ctx_.getNumCorelets();

    LLVM_DEBUG(llvm::dbgs() << "[" PASS_NAME "] num_instances="
                            << num_instances << "\n");

    if (num_instances < 2) {
      LLVM_DEBUG(llvm::dbgs() << "[" PASS_NAME "] num_instances < 2; "
                                 "no-op\n");
      return;
    }

    // Pre-order walk over pipelines. Collect candidates first, then rewrite,
    // so the walk's iterator is not invalidated by op erasure.
    SmallVector<Candidate> candidates;
    module.walk<WalkOrder::PreOrder>([&](mlir::ktdf::PipelineOp pipeline) {
      if (auto c = findCandidate(pipeline, scheduler_ctx_)) {
        candidates.push_back(*c);
      }
    });

    for (const Candidate& c : candidates) {
      rewrite(c, num_instances);
    }
  }

 private:
  const scheduler::SchedulerExtContext& scheduler_ctx_;
};

}  // namespace

namespace ktdf {

std::unique_ptr<mlir::Pass> createParallelizeLoopsAcrossInstancesPass(
    const scheduler::SchedulerExtContext& scheduler_ctx) {
  return std::make_unique<ParallelizeLoopsAcrossInstancesPass>(scheduler_ctx);
}

std::unique_ptr<mlir::Pass> createParallelizeLoopsAcrossInstancesPass() {
  return std::make_unique<ParallelizeLoopsAcrossInstancesPass>(
      scheduler::SchedulerExtContext::dummyContext());
}

}  // namespace ktdf
