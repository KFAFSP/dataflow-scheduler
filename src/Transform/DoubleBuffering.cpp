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
// Pass: -double-buffering
//
// Detects scratchpad memrefs handed off between sibling stages of a
// ktdf.pipeline (single producer, >=1 consumer via ktdf.data_transfer) and
// rewrites them into a ping-pong form: paired allocs hoisted outside the
// pipeline scope, ktdf.buffer_phase + per-buffer ktdf.select_memref
// inserted before the pipeline, and modulo(size: 2) set on the pipeline.
//===----------------------------------------------------------------------===//

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "src/Dialects/KTDF/KTDFOps.hpp"
#include "src/Transform/Common/PipelineScope.hpp"
#include "src/Transform/Passes.h"
#include "src/Utils/SchedulerExtContext.h"

#define PASS_NAME "double-buffering"
#define DEBUG_TYPE PASS_NAME

namespace ktdf {
#define GEN_PASS_DEF_DOUBLEBUFFERINGPASS
#include "src/Transform/Passes.h.inc"
}  // namespace ktdf

using namespace mlir;

namespace {

constexpr int64_t kDefaultNumPhases = 2;

//===----------------------------------------------------------------------===//
// Per-buffer shape-eligibility helpers
//===----------------------------------------------------------------------===//

struct CandidateShape {
  /// SSA value yielded by ktdf.private (a result of the PrivateOp).
  mlir::Value yielded;
  /// Index into PrivateOp's results / PrivateYieldOp's operands.
  unsigned slot;
  /// The single memref.alloc op inside the private region producing this value.
  mlir::memref::AllocOp alloc;
  /// Memref type carried by `yielded`.
  mlir::MemRefType memref_ty;
};

/// Extract the KTDF MemorySpace enum from a memref's memory space attribute.
/// Returns std::nullopt if the memory space is not a KTDF MemorySpaceCustomAttr.
std::optional<mlir::ktdf::MemorySpace> extractKtdfMemorySpace(
    mlir::MemRefType memref_ty) {
  Attribute mspace_attr = memref_ty.getMemorySpace();
  if (!mspace_attr) {
    return std::nullopt;
  }
  if (auto ktdf_attr =
          llvm::dyn_cast<mlir::ktdf::MemorySpaceCustomAttr>(mspace_attr)) {
    return ktdf_attr.getValue();
  }
  return std::nullopt;
}

/// True iff every dynamic-size SSA operand of `alloc` is defined at a location
/// that dominates `insertion_point_op` (using strict op-level dominance —
/// "defined before" `insertion_point_op` in any common enclosing block).
bool dynamicSizesDominate(mlir::memref::AllocOp alloc,
                          mlir::Operation* insertion_point_op,
                          const mlir::DominanceInfo& dom) {
  for (Value sz : alloc.getDynamicSizes()) {
    if (!dom.properlyDominates(sz, insertion_point_op)) {
      return false;
    }
  }
  return true;
}

/// Apply spec §3.3 a–f checks to one yielded value of a pipeline's
/// ktdf.private region. Returns a populated CandidateShape on success;
/// std::nullopt on rejection (with a reason logged via LLVM_DEBUG).
std::optional<CandidateShape> checkBufferShape(
    mlir::ktdf::PrivateOp private_op,
    unsigned slot, mlir::Operation* hoist_target,
    const scheduler::SchedulerExtContext& ctx, const mlir::DominanceInfo& dom) {
  Value yielded = private_op.getResults()[slot];

  // a. Type check: memref.
  auto memref_ty = llvm::dyn_cast<mlir::MemRefType>(yielded.getType());
  if (!memref_ty) {
    LLVM_DEBUG(llvm::dbgs()
               << "[" PASS_NAME "]   slot " << slot << " skipped: not a memref\n");
    return std::nullopt;
  }

  // b. Memory-space eligibility.
  auto mspace = extractKtdfMemorySpace(memref_ty);
  if (!mspace.has_value() || !ctx.isEligibleForDoubleBuffering(*mspace)) {
    LLVM_DEBUG(llvm::dbgs()
               << "[" PASS_NAME "]   slot " << slot
               << " skipped: memory space ineligible\n");
    return std::nullopt;
  }

  // c. Defining op check: memref.alloc inside the private region.
  Value yield_operand = private_op.getYieldOp().getOperands()[slot];
  auto alloc = yield_operand.getDefiningOp<mlir::memref::AllocOp>();
  if (!alloc) {
    LLVM_DEBUG(llvm::dbgs()
               << "[" PASS_NAME "]   slot " << slot
               << " skipped: defining op is not memref.alloc\n");
    return std::nullopt;
  }

  // d. Single-use check: alloc result has exactly one use, and that use is
  // the private_yield op.
  if (!alloc.getResult().hasOneUse()) {
    LLVM_DEBUG(llvm::dbgs()
               << "[" PASS_NAME "]   slot " << slot
               << " skipped: alloc has multiple uses\n");
    return std::nullopt;
  }
  Operation* sole_user = *alloc.getResult().getUsers().begin();
  if (sole_user != private_op.getYieldOp().getOperation()) {
    LLVM_DEBUG(llvm::dbgs()
               << "[" PASS_NAME "]   slot " << slot
               << " skipped: alloc's sole use is not private_yield\n");
    return std::nullopt;
  }

  // e. Same-block check: alloc and private_yield in the same block of the
  // private region.
  if (alloc->getBlock() != private_op.getYieldOp()->getBlock()) {
    LLVM_DEBUG(llvm::dbgs()
               << "[" PASS_NAME "]   slot " << slot
               << " skipped: alloc and yield are in different blocks\n");
    return std::nullopt;
  }

  // f. Dynamic-size domination check.
  if (!dynamicSizesDominate(alloc, hoist_target, dom)) {
    LLVM_DEBUG(llvm::dbgs()
               << "[" PASS_NAME "]   slot " << slot
               << " skipped: dynamic size operand does not dominate hoist target\n");
    return std::nullopt;
  }

  return CandidateShape{yielded, slot, alloc, memref_ty};
}

/// Find the PrivateOp that is the immediate child of `pipeline`. Returns
/// nullptr if the pipeline has no PrivateOp.
mlir::ktdf::PrivateOp getPipelinePrivateOp(mlir::ktdf::PipelineOp pipeline) {
  for (Operation& op : pipeline.getBodyBlock()->getOperations()) {
    if (auto p = llvm::dyn_cast<mlir::ktdf::PrivateOp>(op)) {
      return p;
    }
  }
  return nullptr;
}

//===----------------------------------------------------------------------===//
// Producer/consumer scan and token precedence
//===----------------------------------------------------------------------===//

struct Candidate {
  CandidateShape shape;
  /// The unique sibling stage of the pipeline that writes the buffer.
  mlir::ktdf::StageOp producer;
  /// All sibling stages that read the buffer.
  SmallVector<mlir::ktdf::StageOp> consumers;
};

/// Find the immediate sibling stage of `pipeline` containing `op` (op may be
/// nested arbitrarily deep inside the stage's region). Returns nullptr if no
/// sibling-stage ancestor is found.
mlir::ktdf::StageOp findEnclosingSiblingStage(
    mlir::ktdf::PipelineOp pipeline, mlir::Operation* op) {
  Operation* cur = op;
  while (cur && cur->getParentOp() != pipeline.getOperation()) {
    if (auto stage = llvm::dyn_cast<mlir::ktdf::StageOp>(cur)) {
      if (stage->getParentOp() == pipeline.getOperation()) {
        return stage;
      }
    }
    cur = cur->getParentOp();
  }
  if (auto stage = llvm::dyn_cast_or_null<mlir::ktdf::StageOp>(cur)) {
    return stage;
  }
  return nullptr;
}

/// Walk the entire region of each sibling stage of `pipeline` recursively,
/// classifying every ktdf.data_transfer that references `buffer` as producer
/// (buffer is dest) or consumer (buffer is source). Stages are deduplicated.
/// Returns std::nullopt iff a sibling stage is both producer and consumer.
std::optional<std::pair<SmallVector<mlir::ktdf::StageOp>,
                         SmallVector<mlir::ktdf::StageOp>>>
scanProducersConsumers(mlir::ktdf::PipelineOp pipeline, mlir::Value buffer) {
  llvm::SmallSetVector<mlir::ktdf::StageOp, 4> producers;
  llvm::SmallSetVector<mlir::ktdf::StageOp, 4> consumers;

  for (Operation& sibling : pipeline.getBodyBlock()->getOperations()) {
    auto stage = llvm::dyn_cast<mlir::ktdf::StageOp>(sibling);
    if (!stage) {
      continue;
    }
    stage->walk([&](mlir::ktdf::DataTransferOp xfer) {
      if (xfer.getDestination() == buffer) {
        producers.insert(stage);
      }
      if (xfer.getSource() == buffer) {
        consumers.insert(stage);
      }
    });
  }

  // Disjointness: no sibling can be both producer and consumer.
  for (auto stage : producers) {
    if (consumers.contains(stage)) {
      return std::nullopt;
    }
  }

  return std::make_pair(SmallVector<mlir::ktdf::StageOp>(producers.begin(),
                                                          producers.end()),
                        SmallVector<mlir::ktdf::StageOp>(consumers.begin(),
                                                          consumers.end()));
}

/// True iff `producer` transitively precedes `target` in `pipeline`'s token
/// dependency graph. Edges are stage-to-stage: stage S' has an incoming edge
/// from stage S iff S produces (depends_out) a token that S' consumes
/// (depends_in).
bool producerTransitivelyPrecedes(mlir::ktdf::PipelineOp pipeline,
                                  mlir::ktdf::StageOp producer,
                                  mlir::ktdf::StageOp target) {
  if (producer == target) {
    return false;
  }

  // Build a map from a token SSA value → defining stage.
  llvm::DenseMap<mlir::Value, mlir::ktdf::StageOp> token_producer_stage;
  for (Operation& op : pipeline.getBodyBlock()->getOperations()) {
    auto stage = llvm::dyn_cast<mlir::ktdf::StageOp>(op);
    if (!stage) {
      continue;
    }
    for (mlir::Value t : stage.getDependsOut()) {
      token_producer_stage.try_emplace(t, stage);
    }
  }

  // BFS from producer over token edges.
  llvm::SmallSetVector<mlir::ktdf::StageOp, 8> visited;
  visited.insert(producer);
  SmallVector<mlir::ktdf::StageOp> worklist{producer};
  while (!worklist.empty()) {
    mlir::ktdf::StageOp s = worklist.pop_back_val();
    // For every other sibling stage S' that consumes a token produced by S,
    // S' is reachable from S.
    for (Operation& op : pipeline.getBodyBlock()->getOperations()) {
      auto succ = llvm::dyn_cast<mlir::ktdf::StageOp>(op);
      if (!succ || succ == s) {
        continue;
      }
      bool depends_on_s = false;
      for (mlir::Value t : succ.getDependsIn()) {
        auto it = token_producer_stage.find(t);
        if (it != token_producer_stage.end() && it->second == s) {
          depends_on_s = true;
          break;
        }
      }
      if (!depends_on_s) {
        continue;
      }
      if (succ == target) {
        return true;
      }
      if (visited.insert(succ)) {
        worklist.push_back(succ);
      }
    }
  }
  return false;
}

/// Apply spec §3.3 g–k checks to a shape-eligible buffer in `pipeline`. On
/// success, returns a Candidate. On rejection, logs the reason and returns
/// std::nullopt.
std::optional<Candidate> checkProducerConsumer(
    mlir::ktdf::PipelineOp pipeline, const CandidateShape& shape) {
  auto scan = scanProducersConsumers(pipeline, shape.yielded);
  if (!scan.has_value()) {
    LLVM_DEBUG(llvm::dbgs()
               << "[" PASS_NAME "]   slot " << shape.slot
               << " skipped: a sibling stage both writes and reads the buffer\n");
    return std::nullopt;
  }
  auto& [producers, consumers] = *scan;

  // h. Producer cardinality.
  if (producers.size() != 1) {
    LLVM_DEBUG(llvm::dbgs()
               << "[" PASS_NAME "]   slot " << shape.slot << " skipped: "
               << producers.size() << " producer stages (need exactly 1)\n");
    return std::nullopt;
  }

  // i. Consumer cardinality.
  if (consumers.empty()) {
    LLVM_DEBUG(llvm::dbgs()
               << "[" PASS_NAME "]   slot " << shape.slot
               << " skipped: no consumer stages\n");
    return std::nullopt;
  }

  // k. Producer must transitively precede every consumer.
  for (mlir::ktdf::StageOp consumer : consumers) {
    if (!producerTransitivelyPrecedes(pipeline, producers.front(), consumer)) {
      LLVM_DEBUG(llvm::dbgs()
                 << "[" PASS_NAME "]   slot " << shape.slot
                 << " skipped: producer does not transitively precede consumer\n");
      return std::nullopt;
    }
  }

  return Candidate{shape, producers.front(), consumers};
}

//===----------------------------------------------------------------------===//
// Per-pipeline guards
//===----------------------------------------------------------------------===//

/// True iff the pipeline is eligible for analysis (no pre-existing modulo
/// and a non-empty enclosing scope of scf.for loops). On rejection, logs
/// the reason via LLVM_DEBUG.
bool isPipelineEligible(
    mlir::ktdf::PipelineOp pipeline,
    const ::ktdf::common::PipelineEnclosingScope& scope) {
  if (pipeline.getModuloAttr()) {
    LLVM_DEBUG(llvm::dbgs()
               << "[" PASS_NAME "] skip pipeline at " << pipeline.getLoc()
               << ": modulo already set\n");
    return false;
  }
  if (scope.loops.empty()) {
    LLVM_DEBUG(llvm::dbgs()
               << "[" PASS_NAME "] skip pipeline at " << pipeline.getLoc()
               << ": no enclosing scf.for scope\n");
    return false;
  }
  return true;
}

struct DoubleBufferingPass
    : public ::ktdf::impl::DoubleBufferingPassBase<DoubleBufferingPass> {
  DoubleBufferingPass()
      : scheduler_ctx_(scheduler::SchedulerExtContext::dummyContext()) {}
  explicit DoubleBufferingPass(const scheduler::SchedulerExtContext& ctx)
      : scheduler_ctx_(ctx) {}

  void runOnOperation() override {
    ModuleOp module = getOperation();

    LLVM_DEBUG(llvm::dbgs() << "[" PASS_NAME "] starting\n");

    DominanceInfo dom(module);

    SmallVector<std::pair<mlir::ktdf::PipelineOp, SmallVector<CandidateShape>>>
        per_pipeline_shapes;
    module.walk<WalkOrder::PreOrder>([&](mlir::ktdf::PipelineOp pipeline) {
      auto scope = ::ktdf::common::computePipelineEnclosingScope(pipeline);
      if (!isPipelineEligible(pipeline, scope)) {
        return;
      }

      // Hoist target: the position immediately before scope.loops.back()
      // (the outermost loop in scope).
      Operation* hoist_target = scope.loops.back().getOperation();

      auto private_op = getPipelinePrivateOp(pipeline);
      if (!private_op) {
        LLVM_DEBUG(llvm::dbgs()
                   << "[" PASS_NAME "] skip pipeline at " << pipeline.getLoc()
                   << ": no ktdf.private region\n");
        return;
      }

      SmallVector<CandidateShape> shapes;
      for (unsigned slot = 0; slot < private_op.getNumResults(); ++slot) {
        auto shape = checkBufferShape(private_op, slot, hoist_target,
                                       scheduler_ctx_, dom);
        if (shape.has_value()) {
          shapes.push_back(*shape);
        }
      }

      if (shapes.empty()) {
        LLVM_DEBUG(llvm::dbgs()
                   << "[" PASS_NAME "] skip pipeline at " << pipeline.getLoc()
                   << ": no shape-eligible buffers\n");
        return;
      }

      LLVM_DEBUG(llvm::dbgs()
                 << "[" PASS_NAME "] pipeline at " << pipeline.getLoc()
                 << " has " << shapes.size() << " shape-eligible buffer(s)\n");
      per_pipeline_shapes.emplace_back(pipeline, std::move(shapes));
    });

    // Per pipeline, filter shape-eligible buffers down to producer/consumer-
    // valid candidates.
    SmallVector<std::pair<mlir::ktdf::PipelineOp, SmallVector<Candidate>>>
        per_pipeline_candidates;
    for (auto& [pipeline, shapes] : per_pipeline_shapes) {
      SmallVector<Candidate> candidates;
      for (const CandidateShape& shape : shapes) {
        if (auto c = checkProducerConsumer(pipeline, shape)) {
          candidates.push_back(*c);
        }
      }
      if (candidates.empty()) {
        LLVM_DEBUG(llvm::dbgs()
                   << "[" PASS_NAME "] skip pipeline at " << pipeline.getLoc()
                   << ": no candidates passed producer/consumer scan\n");
        continue;
      }
      LLVM_DEBUG(llvm::dbgs()
                 << "[" PASS_NAME "] pipeline at " << pipeline.getLoc()
                 << " has " << candidates.size() << " candidate(s)\n");
      per_pipeline_candidates.emplace_back(pipeline, std::move(candidates));
    }

    // Transformation arrives in the next task.
  }

 private:
  const scheduler::SchedulerExtContext& scheduler_ctx_;
};

}  // namespace

namespace ktdf {

std::unique_ptr<mlir::Pass> createDoubleBufferingPass() {
  return std::make_unique<DoubleBufferingPass>();
}

std::unique_ptr<mlir::Pass> createDoubleBufferingPass(
    const scheduler::SchedulerExtContext& scheduler_ctx) {
  return std::make_unique<DoubleBufferingPass>(scheduler_ctx);
}

}  // namespace ktdf
