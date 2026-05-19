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

    // Pre-order walk over pipelines.
    SmallVector<mlir::ktdf::PipelineOp> eligible;
    module.walk<WalkOrder::PreOrder>([&](mlir::ktdf::PipelineOp pipeline) {
      auto scope = ::ktdf::common::computePipelineEnclosingScope(pipeline);
      if (!isPipelineEligible(pipeline, scope)) {
        return;
      }
      LLVM_DEBUG(llvm::dbgs()
                 << "[" PASS_NAME "] eligible pipeline at " << pipeline.getLoc()
                 << "\n");
      eligible.push_back(pipeline);
    });

    // Candidate detection and transformation arrive in subsequent tasks.
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
