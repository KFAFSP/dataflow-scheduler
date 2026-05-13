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
#include "src/Transform/ReusePromotion/Common/InvarianceCheck.hpp"
#include "src/Transform/ReusePromotion/Common/PipelineScope.hpp"
#include "src/Transform/ReusePromotion/Common/WriteSetScan.hpp"

namespace ktdf {
namespace reuse {

namespace {

/// True iff `v`'s defining op (if any) is located inside `region`
/// (transitively: at any depth, including nested regions).
bool definedInsideRegion(mlir::Value v, mlir::Region &region) {
  mlir::Operation *def = v.getDefiningOp();
  if (!def) {
    return false;
  }
  mlir::Operation *ancestor = def;
  while (ancestor) {
    if (ancestor->getParentRegion() == &region) {
      return true;
    }
    ancestor = ancestor->getParentOp();
  }
  return false;
}

/// Walk `scope.loops` innermost-to-outermost. For each loop, check all
/// five conditions. Return the outermost loop where every condition
/// still holds. Nullptr if even the innermost loop fails.
mlir::scf::ForOp findOutermostLegalTarget(
    mlir::ktdf::DataTransferOp transfer,
    const PipelineEnclosingScope &scope) {
  mlir::scf::ForOp best = nullptr;
  for (mlir::scf::ForOp loop : scope.loops) {
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
    best = loop;
  }
  return best;
}

}  // namespace

std::optional<Candidate> findFirstCandidate(
    mlir::ktdf::PipelineOp pipeline) {
  PipelineEnclosingScope scope = computePipelineEnclosingScope(pipeline);
  if (scope.loops.empty()) {
    return std::nullopt;
  }

  std::optional<Candidate> found;
  for (mlir::Operation &child : pipeline.getBodyBlock()->getOperations()) {
    if (found) {
      break;
    }
    auto stage = mlir::dyn_cast<mlir::ktdf::StageOp>(&child);
    if (!stage) {
      continue;
    }
    stage.walk([&](mlir::ktdf::DataTransferOp transfer) {
      if (found) {
        return mlir::WalkResult::interrupt();
      }
      if (transfer.isDestFifo()) {
        return mlir::WalkResult::advance();
      }
      mlir::scf::ForOp target = findOutermostLegalTarget(transfer, scope);
      if (!target) {
        return mlir::WalkResult::advance();
      }
      found = Candidate{transfer, stage, target};
      return mlir::WalkResult::interrupt();
    });
  }
  return found;
}

}  // namespace reuse
}  // namespace ktdf
