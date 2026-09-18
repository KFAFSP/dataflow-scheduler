//===-- FuseLinalg.cpp ------------------------------------------*- c++ -*-===//
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

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/LogicalResult.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/Linalg/Transforms/Transforms.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include "dataflow-scheduler/Conversion/frontend/KTIRToScheduleIR/Passes.h"  // IWYU pragma: keep
#include "dataflow-scheduler/Dialect/KTDFArch/KTDFArch.h"
#include "dataflow-scheduler/Dialect/KTDFArch/KTDFArchAttributes.h"
#include "dataflow-scheduler/Dialect/KTDFArch/KTDFArchIntrinsics.h"

#define PASS_NAME "fuse-linalg"
#define DEBUG_TYPE PASS_NAME

static llvm::cl::opt<bool> disable_this_pass(
    "disable-" PASS_NAME, llvm::cl::desc("Disable Fuse Linalg pass"),
    llvm::cl::init(false));

using namespace scheduler;

namespace scheduler {
#define GEN_PASS_DEF_FUSELINALGPASS
#include "dataflow-scheduler/Conversion/frontend/KTIRToScheduleIR/Passes.h.inc"
}  // namespace scheduler

namespace {

/// Detects the implied mapping of @p generic and explicitly sets it.
///
/// @returns  Succeeds when @p generic explicitly declares the correct mapping.
auto propagateImpliedMapping(mlir::linalg::GenericOp generic)
    -> llvm::FailureOr<mlir::ktdf_arch::MapsToAttr> {
  auto mappable =
      llvm::dyn_cast<mlir::ktdf_arch::Mappable>(generic.getOperation());
  if (!mappable) {
    return llvm::success(nullptr);
  }
  if (const auto maps_to = mappable.getMapsTo()) {
    return llvm::success(maps_to);
  }

  // Try to determine a mapping from the contents of the operation.
  llvm::SetVector<mlir::ktdf_arch::ResourceSpecAttr> mapping;
  for (auto& child : *generic.getBody()) {
    if (const auto child_mapping = mlir::ktdf_arch::Mappable::getMapsTo(&child);
        child_mapping) {
      mapping.insert_range(child_mapping.getValue());
    }
  }
  if (mapping.empty()) {
    return llvm::success(nullptr);
  }

  // There are mapping constraints on the contents of this op that will
  // have to be respected. Set them on the op, and if that fails, don't
  // attempt to fuse it.
  const auto maps_to = mlir::ktdf_arch::MapsToAttr::get(generic->getContext(),
                                                        mapping.getArrayRef());
  if (llvm::succeeded(mappable.setMapsTo(maps_to))) {
    return llvm::success(maps_to);
  }

  return llvm::failure();
}

/// Determines whether @p lhs and @p rhs are subsets of one another.
[[nodiscard]] auto setEqual(
    llvm::MutableArrayRef<mlir::ktdf_arch::ResourceSpecAttr> lhs,
    llvm::MutableArrayRef<mlir::ktdf_arch::ResourceSpecAttr> rhs) {
  const auto compare = [](mlir::ktdf_arch::ResourceSpecAttr l,
                          mlir::ktdf_arch::ResourceSpecAttr r) -> bool {
    return l.getAsOpaquePointer() < r.getAsOpaquePointer();
  };

  llvm::sort(lhs, compare);
  lhs = {lhs.begin(), llvm::unique(lhs)};
  llvm::sort(rhs, compare);
  rhs = {rhs.begin(), llvm::unique(rhs)};

  return lhs == rhs;
}

[[nodiscard]] auto areCompatible(mlir::ktdf_arch::MapsToAttr lhs,
                                 mlir::ktdf_arch::MapsToAttr rhs) -> bool {
  if (lhs == rhs || !lhs || !rhs) {
    return true;
  }

  llvm::SmallVector<mlir::ktdf_arch::ResourceSpecAttr> buffer;
  buffer.reserve(lhs.getValue().size() + rhs.getValue().size());
  llvm::append_range(buffer, lhs.getValue());
  const auto split = buffer.size();
  llvm::append_range(buffer, rhs.getValue());
  return setEqual({buffer.data(), buffer.data() + split},
                  {buffer.data() + split, buffer.end()});
}

struct FuseLinalgPass : public impl::FuseLinalgPassBase<FuseLinalgPass> {
  using FuseLinalgPassBase<FuseLinalgPass>::FuseLinalgPassBase;

  void runOnOperation() override {
    if (disable_this_pass) {
      return;
    }

    // Collect all fusable linalg.generic candidate operations.
    llvm::SetVector<mlir::Operation*> candidates;
    const auto collect = [&](mlir::linalg::GenericOp generic) {
      if (succeeded(propagateImpliedMapping(generic))) {
        candidates.insert(generic);
      }
    };
    getOperation()->walk(collect);

    mlir::RewritePatternSet patterns(&getContext());

    llvm::DenseSet<mlir::Location> fused_locs;
    const auto control_fusion = [&](mlir::OpOperand* through) -> bool {
      auto* const producer = through->get().getDefiningOp();
      auto* const consumer = through->getOwner();
      if (!candidates.contains(producer)) {
        // We know that the consumer is a candidate, but the producer might not
        // be, and then we're not allowed to fuse with it.
        return false;
      }

      const auto producer_mapping =
          mlir::ktdf_arch::Mappable::getMapsTo(producer);
      const auto consumer_mapping =
          mlir::ktdf_arch::Mappable::getMapsTo(consumer);
      // Only fuse ops that have compatible mappings.
      if (!areCompatible(producer_mapping, consumer_mapping)) {
        return false;
      }

      // Annotate the mapping in the op location, so that we will be able to
      // recover it after fusion. We will undo this if the op is not fused.
      if (const auto fused_mapping =
              consumer_mapping ? consumer_mapping : producer_mapping;
          fused_mapping) {
        const auto fused_loc = mlir::FusedLoc::get(
            consumer->getContext(), {consumer->getLoc()}, fused_mapping);
        fused_locs.insert(fused_loc);
        consumer->setLoc(fused_loc);
      }
      return true;
    };
    mlir::linalg::populateElementwiseOpsFusionPatterns(patterns,
                                                       control_fusion);

    // Fuse as many of the candidates as possible.
    auto changed = false;
    const auto rewrite_config =
        mlir::GreedyRewriteConfig()
            .setStrictness(mlir::GreedyRewriteStrictness::ExistingAndNewOps)
            .enableConstantCSE();
    if (mlir::failed(mlir::applyOpPatternsGreedily(candidates.getArrayRef(),
                                                   std::move(patterns),
                                                   rewrite_config, &changed))) {
      signalPassFailure();
    }
    if (!changed) {
      markAllAnalysesPreserved();
    }

    // Restore the locations and update the fused op's mapping.
    const auto fix = [&](mlir::linalg::GenericOp generic) {
      if (!fused_locs.contains(generic.getLoc())) {
        return;
      }
      const auto loc =
          llvm::cast<mlir::FusedLocWith<mlir::ktdf_arch::MapsToAttr>>(
              generic.getLoc());
      generic->setLoc(loc.getLocations()[0]);

      if (!candidates.contains(generic)) {
        std::ignore =
            mlir::ktdf_arch::Mappable::setMapsTo(generic, loc.getMetadata());
      }
    };
    getOperation()->walk(fix);
  }
};

}  // namespace
