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
// BroadcastPromotion: hoist invariant ktdf.data_transfer ops as sibling
// pipelines. See
// docs/superpowers/specs/2026-05-12-broadcast-promotion-design.md
//===----------------------------------------------------------------------===//

#include <memory>

#include "llvm/Support/Debug.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "src/Dialects/KTDF/KTDFOps.hpp"
#include "src/Transform/Passes.h"
#include "src/Transform/ReusePromotion/Common/CandidateSearch.hpp"
#include "src/Transform/ReusePromotion/Common/RegionClonePrune.hpp"

namespace ktdf {

#define GEN_PASS_DEF_BROADCASTPROMOTIONPASS
#include "src/Transform/Passes.h.inc"

#define PASS_NAME "broadcast-promotion"
#define DEBUG_TYPE PASS_NAME

namespace {

/// Find the underlying memref.alloc that defines `dest`. If `dest` is a
/// direct alloc result, return it. If `dest` is a result of a
/// ktdf.private op, walk the corresponding private_yield operand back to
/// the alloc inside the private body. Returns nullptr if no alloc can be
/// found through the chain.
mlir::memref::AllocOp findUnderlyingAlloc(mlir::Value dest) {
  if (auto alloc = dest.getDefiningOp<mlir::memref::AllocOp>()) {
    return alloc;
  }
  if (auto private_op = dest.getDefiningOp<mlir::ktdf::PrivateOp>()) {
    auto result_idx = mlir::cast<mlir::OpResult>(dest).getResultNumber();
    mlir::ktdf::PrivateYieldOp yield_op = private_op.getYieldOp();
    if (!yield_op || result_idx >= yield_op.getNumOperands()) {
      return nullptr;
    }
    return findUnderlyingAlloc(yield_op.getOperand(result_idx));
  }
  return nullptr;
}

/// Clone the defining memref.alloc of `orig_dest` immediately before
/// `insert_before_op`. Returns the cloned alloc's SSA result value.
/// Handles the case where `orig_dest` is a result of ktdf.private by
/// finding the underlying alloc inside the private body.
mlir::Value cloneDestAllocAbove(mlir::Value orig_dest,
                                mlir::Operation* insert_before_op) {
  mlir::memref::AllocOp alloc = findUnderlyingAlloc(orig_dest);
  assert(alloc && "destination must trace back to a memref.alloc");
  mlir::OpBuilder builder(insert_before_op);
  mlir::Operation* new_alloc = builder.clone(*alloc.getOperation());
  return new_alloc->getResult(0);
}

/// Perform one broadcast hoist for `c`.
void performHoist(const reuse::Candidate& c) {
  mlir::ktdf::DataTransferOp transfer = c.transfer;
  mlir::ktdf::StageOp donor_stage = c.donor_stage;
  mlir::scf::ForOp target_loop = c.target_loop;
  mlir::Value orig_dest = transfer.getDestination();

  LLVM_DEBUG(llvm::dbgs() << "[" << PASS_NAME << "] hoisting "
                          << *transfer.getOperation() << "\n");

  // 1. Fresh destination alloc immediately before target_loop.
  mlir::Value new_dest =
      cloneDestAllocAbove(orig_dest, target_loop.getOperation());

  // 2. Sibling pipeline + stage immediately after the new alloc, still
  //    before target_loop.
  mlir::OpBuilder builder(target_loop);
  auto new_pipeline =
      mlir::ktdf::PipelineOp::create(builder, target_loop.getLoc());
  builder.setInsertionPointToStart(new_pipeline.getBodyBlock());
  auto new_stage =
      mlir::ktdf::StageOp::create(builder, target_loop.getLoc(),
                                  /*depends_in=*/mlir::ValueRange{},
                                  /*depends_out=*/mlir::ValueRange{});

  // 3. Seed IRMapping: original dest -> new_dest.
  mlir::IRMapping value_map;
  value_map.map(orig_dest, new_dest);

  // 4. Clone donor stage body into new stage body; prune to the transfer.
  mlir::Block* new_stage_body = &new_stage.getBody().front();
  reuse::cloneRegionAndPruneToAnchor(donor_stage.getBody(), new_stage_body,
                                     new_stage_body->end(),
                                     transfer.getOperation(), value_map);

  // 5. Rewire uses of orig_dest outside donor_stage to new_dest.
  orig_dest.replaceUsesWithIf(new_dest, [&](mlir::OpOperand& use) {
    mlir::Operation* user = use.getOwner();
    return !donor_stage.getOperation()->isAncestor(user);
  });

  // 6. Erase original transfer.
  transfer.erase();

  // NOTE: if the donor stage's body becomes empty after erasing the
  // transfer, the in-memory IR is well-formed but the printer/parser
  // round-trip fails (an empty ktdf.stage body without explicit ^bb0:
  // re-parses as a 0-block region, which fails the SizedRegion<1>
  // verifier). This is a dialect-level printer/parser asymmetry, not
  // a pass bug. Round-trip tests that pipe pass output through
  // scheduler again will hit it. Tracking as a known limitation.
}

/// Walk pipelines in post-order; return true iff a hoist was performed.
bool tryHoistOneBroadcast(mlir::ModuleOp module) {
  bool did_hoist = false;
  module.walk<mlir::WalkOrder::PostOrder>(
      [&](mlir::ktdf::PipelineOp pipeline) -> mlir::WalkResult {
        if (did_hoist) {
          return mlir::WalkResult::interrupt();
        }
        if (auto c = reuse::findFirstCandidate(pipeline)) {
          performHoist(*c);
          did_hoist = true;
          return mlir::WalkResult::interrupt();
        }
        return mlir::WalkResult::advance();
      });
  return did_hoist;
}

struct BroadcastPromotionPass
    : public impl::BroadcastPromotionPassBase<BroadcastPromotionPass> {
  using impl::BroadcastPromotionPassBase<
      BroadcastPromotionPass>::BroadcastPromotionPassBase;

  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();
    bool changed = true;
    while (changed) {
      changed = tryHoistOneBroadcast(module);
    }
  }
};

}  // namespace

std::unique_ptr<mlir::Pass> createBroadcastPromotionPass() {
  return std::make_unique<BroadcastPromotionPass>();
}

}  // namespace ktdf
