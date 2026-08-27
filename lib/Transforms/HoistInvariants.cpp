//===-- HoistInvariants.cpp -------------------------------------*- c++ -*-===//
//
// Part of the Dataflow Scheduler project.
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

#include <llvm/Support/DebugLog.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/Dominance.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/IR/Visitors.h>
#include <mlir/Interfaces/LoopLikeInterface.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Transforms/LoopInvariantCodeMotionUtils.h>

#include "dataflow-scheduler/Dialect/KTDF/KTDF.h"
#include "dataflow-scheduler/Dialect/KTDF/Transforms/CodeMotion.h"
#include "dataflow-scheduler/Transforms/Passes.h"  // IWYU pragma: keep
#include "dataflow-scheduler/Transforms/Utils/Hoisting.h"

#define PASS_NAME "hoist-invariants"
#define DEBUG_TYPE PASS_NAME

static llvm::cl::opt<bool> disable_this_pass(
    "disable-" PASS_NAME, llvm::cl::desc("Disable Hoist Invariants pass"),
    llvm::cl::init(false));

namespace scheduler {
#define GEN_PASS_DEF_HOISTINVARIANTSPASS
#include "dataflow-scheduler/Transforms/Passes.h.inc"
}  // namespace scheduler

using namespace scheduler;

namespace {

static const auto kSkipRegions = mlir::OpPrintingFlags().skipRegions();

struct HoistInvariantsPass
    : impl::HoistInvariantsPassBase<HoistInvariantsPass> {
  using HoistInvariantsPassBase::HoistInvariantsPassBase;

  void runOnOperation() override {
    if (disable_this_pass) {
      return;
    }

    // Hoist entirely pure invariants.
    getOperation()->walk([&](mlir::Operation* op) { hoistInvariants(op); });

    // Hoist allocations with dominating invariant writes (constants).
    getOperation()->walk(
        [&](mlir::ktdf::PipelineOp pipeline) { hoistConstants(pipeline); });
  }

 private:
  void hoistInvariants(mlir::Operation* op) {
    if (auto iface = mlir::dyn_cast<mlir::LoopLikeOpInterface>(op); iface) {
      // Hoist all pure operations without inter-iteration dependencies directly
      // in front of the loop.
      num_hoisted += mlir::moveLoopInvariantCode(iface);
      return;
    }

    if (auto generic = mlir::dyn_cast<mlir::linalg::GenericOp>(op); generic) {
      // Hoist all pure operations without inter-iteration dependencies directly
      // in front of the 'linalg.generic' operation.
      num_hoisted += mlir::moveLoopInvariantCode(
          {&generic.getBodyRegion()},
          [&](mlir::Value value, mlir::Region* /*region*/) -> bool {
            return value.getParentRegion()->isProperAncestor(
                &generic.getBodyRegion());
          },
          [&](mlir::Operation* op, mlir::Region* /*region*/) -> bool {
            return mlir::isPure(op);
          },
          [&](mlir::Operation* op, mlir::Region* /*region*/) {
            op->moveBefore(generic);
          });
      return;
    }

    if (auto pipeline = mlir::dyn_cast<mlir::ktdf::PipelineOp>(op); pipeline) {
      // Hoist all pure operations without dependencies directly in front of the
      // 'ktdf.pipeline' operation.
      num_hoisted += mlir::ktdf::hoistPipelineContents(
          pipeline, [&](mlir::Operation* op) -> mlir::ktdf::PipelineAnchor {
            return mlir::isPure(op) ? mlir::ktdf::PipelineAnchor::Parent
                                    : mlir::ktdf::PipelineAnchor::Stage;
          });
      return;
    }
  }

  void hoistConstants(mlir::ktdf::PipelineOp pipeline) {
    mlir::OpBuilder builder(pipeline);

    mlir::DominanceInfo dominance;
    llvm::DenseSet<mlir::Operation*> invariant_writes;
    const auto hoist = [&](mlir::Operation* op) {
      // Find the target scope to hoist to.
      auto* const target =
          findHoistingTarget(op, [](mlir::Region* region) -> bool {
            return !region->getParentOp()
                        ->mightHaveTrait<mlir::OpTrait::HasParallelRegion>();
          });
      assert(target->isProperAncestor(op));

      if (!invariant_writes.contains(op)) {
        op->moveBefore(target);
        return;
      }

      // Create a pipeline with a single stage and hoist into it.
      builder.setInsertionPoint(target);
      mlir::ktdf::PipelineOp::create(
          builder, pipeline.getLoc(),
          [&](mlir::OpBuilder& builder, mlir::Location /*loc*/) {
            auto stage =
                mlir::ktdf::StageOp::create(builder, op->getLoc(), {}, {});
            stage.setApplicableUnitsAttr(
                op->getParentOfType<mlir::ktdf::StageOp>()
                    .getApplicableUnitsAttr());
            op->moveBefore(stage.getBody(), stage.getBody()->end());
          });
    };

    // Hoist allocations and their invariant writes, as well as pure ops so
    // that we munch as many as possible.
    num_hoisted += mlir::ktdf::hoistPipelineContents(
        pipeline,
        [&](mlir::Operation* op) -> mlir::ktdf::PipelineAnchor {
          if (mlir::isPure(op) || invariant_writes.contains(op)) {
            return mlir::ktdf::PipelineAnchor::Parent;
          }

          if (auto alloc = llvm::dyn_cast<mlir::memref::AllocaOp>(op); alloc) {
            auto* const write = findInvariantWriteIn(
                alloc.getResult(), &pipeline.getBodyRegion(), dominance);
            if (write) {
              LDBG() << "found invariant write "
                     << mlir::OpWithFlags(write, kSkipRegions);
              LDBG() << "hoisting allocation "
                     << mlir::OpWithFlags(alloc, kSkipRegions);
              invariant_writes.insert(write);
              ++this->invariant_writes;
              return mlir::ktdf::PipelineAnchor::Parent;
            }
          }

          return mlir::ktdf::PipelineAnchor::Stage;
        },
        hoist);
  }
};

}  // namespace
