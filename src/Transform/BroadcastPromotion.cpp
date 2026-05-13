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
// pipelines. See docs/superpowers/specs/2026-05-12-broadcast-promotion-design.md
//===----------------------------------------------------------------------===//

#include <memory>

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "src/Dialects/KTDF/KTDFOps.hpp"
#include "src/Transform/Passes.h"

namespace ktdf {

#define GEN_PASS_DEF_BROADCASTPROMOTIONPASS
#include "src/Transform/Passes.h.inc"

#define PASS_NAME "broadcast-promotion"
#define DEBUG_TYPE PASS_NAME

namespace {

struct BroadcastPromotionPass
    : public impl::BroadcastPromotionPassBase<BroadcastPromotionPass> {
  using impl::BroadcastPromotionPassBase<
      BroadcastPromotionPass>::BroadcastPromotionPassBase;

  void runOnOperation() override {
    // TODO(Task 7): drive fixed-point loop calling tryHoistOneBroadcast.
  }
};

}  // namespace

std::unique_ptr<mlir::Pass> createBroadcastPromotionPass() {
  return std::make_unique<BroadcastPromotionPass>();
}

}  // namespace ktdf
