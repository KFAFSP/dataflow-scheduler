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

#include "src/Transform/ReusePromotion/Common/PipelineScope.hpp"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"

namespace ktdf {
namespace reuse {

namespace {

/// True iff the body of `loop` contains nothing except `child` (and the
/// loop's required scf.yield terminator).
bool loopBodyContainsOnly(mlir::scf::ForOp loop, mlir::Operation *child) {
  mlir::Block &body = loop.getRegion().front();
  int non_terminator_count = 0;
  mlir::Operation *only_non_terminator = nullptr;
  for (mlir::Operation &op : body) {
    if (op.hasTrait<mlir::OpTrait::IsTerminator>()) {
      continue;
    }
    ++non_terminator_count;
    only_non_terminator = &op;
  }
  return non_terminator_count == 1 && only_non_terminator == child;
}

}  // namespace

PipelineEnclosingScope computePipelineEnclosingScope(
    mlir::ktdf::PipelineOp pipeline) {
  PipelineEnclosingScope scope;
  mlir::Operation *current = pipeline.getOperation();
  mlir::Operation *parent = current->getParentOp();
  while (parent) {
    auto for_op = mlir::dyn_cast<mlir::scf::ForOp>(parent);
    if (!for_op) {
      break;
    }
    if (!loopBodyContainsOnly(for_op, current)) {
      break;
    }
    scope.loops.push_back(for_op);
    current = for_op.getOperation();
    parent = current->getParentOp();
  }
  return scope;
}

}  // namespace reuse
}  // namespace ktdf
