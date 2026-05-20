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

#include "src/Transform/ReusePromotion/Common/CandidateSearch.hpp"

#include "mlir/IR/Operation.h"
#include "src/Transform/Common/PipelineScope.hpp"
#include "src/Transform/ReusePromotion/Common/InvarianceCheck.hpp"
#include "src/Transform/ReusePromotion/Common/WriteSetScan.hpp"

namespace ktdf {
namespace reuse {

namespace {

/// True iff `v`'s defining op (if any) is located inside `region`
/// (transitively: at any depth, including nested regions).
bool definedInsideRegion(mlir::Value v, mlir::Region& region) {
  mlir::Operation* def = v.getDefiningOp();
  if (!def) {
    return false;
  }
  mlir::Operation* ancestor = def;
  while (ancestor) {
    if (ancestor->getParentRegion() == &region) {
      return true;
    }
    ancestor = ancestor->getParentOp();
  }
  return false;
}

/// Walk `scope.loops` innermost-to-outermost. For each loop, check all
/// five conditions. Return the index in `scope.loops` of the outermost
/// loop where every condition still holds, or -1 if even the innermost
/// loop fails. The index doubles as a depth metric: higher = farther
/// out from the pipeline.
int findOutermostLegalTargetDepth(
    mlir::ktdf::DataTransferOp transfer,
    const ktdf::common::PipelineEnclosingScope& scope) {
  int best = -1;
  for (int i = 0; i < static_cast<int>(scope.loops.size()); ++i) {
    mlir::scf::ForOp loop = scope.loops[i];
    if (!transferIsInvariantWrt(transfer, loop.getInductionVar())) {
      break;
    }
    if (regionWritesTo(loop.getRegion(), transfer.getSource())) {
      break;
    }
    if (regionWritesTo(loop.getRegion(), transfer.getDestination(),
                       /*ignore=*/transfer.getOperation())) {
      break;
    }
    if (definedInsideRegion(transfer.getSource(), loop.getRegion())) {
      break;
    }
    best = i;
  }
  return best;
}

}  // namespace

std::optional<Candidate> findFirstCandidate(mlir::ktdf::PipelineOp pipeline) {
  ktdf::common::PipelineEnclosingScope scope =
      ktdf::common::getPipelineEnclosingScope(pipeline);
  if (scope.loops.empty()) {
    return std::nullopt;
  }

  // Scan every data_transfer in the pipeline, then pick the one whose
  // legal hoist target is deepest (i.e., farthest out from the pipeline).
  // Picking source-order-first would let an early shallow-target hoist
  // poison the pipeline-enclosing-scope chain for later transfers that
  // could have hoisted further. Picking the deepest target avoids that.
  std::optional<Candidate> best_candidate;
  int best_depth = -1;
  for (mlir::Operation& child : pipeline.getBodyBlock()->getOperations()) {
    auto stage = mlir::dyn_cast<mlir::ktdf::StageOp>(&child);
    if (!stage) {
      continue;
    }
    stage.walk([&](mlir::ktdf::DataTransferOp transfer) {
      if (transfer.isDestFifo()) {
        return mlir::WalkResult::advance();
      }
      int depth = findOutermostLegalTargetDepth(transfer, scope);
      if (depth < 0) {
        return mlir::WalkResult::advance();
      }
      if (depth > best_depth) {
        best_depth = depth;
        best_candidate = Candidate{transfer, stage, scope.loops[depth]};
      }
      return mlir::WalkResult::advance();
    });
  }
  return best_candidate;
}

}  // namespace reuse
}  // namespace ktdf
