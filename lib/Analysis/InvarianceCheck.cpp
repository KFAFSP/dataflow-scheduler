//===-- InvarianceCheck.cpp -------------------------------------*- c++ -*-===//
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

#include "dataflow-scheduler/Analysis/InvarianceCheck.h"

#include <llvm/ADT/DenseSet.h>

using namespace scheduler;

namespace {

bool dependsOnImpl(mlir::Value v, mlir::Value iv,
                   llvm::DenseSet<mlir::Value>& visited) {
  if (v == iv) {
    return true;
  }
  if (!visited.insert(v).second) {
    return false;
  }
  mlir::Operation* def = v.getDefiningOp();
  if (!def) {
    return false;
  }
  for (mlir::Value operand : def->getOperands()) {
    if (dependsOnImpl(operand, iv, visited)) {
      return true;
    }
  }
  return false;
}

}  // namespace

auto scheduler::ssaDependsOn(mlir::Value v, mlir::Value iv) -> bool {
  llvm::DenseSet<mlir::Value> visited;
  return dependsOnImpl(v, iv, visited);
}

auto scheduler::transferIsInvariantWrt(mlir::ktdf::DataTransferOp transfer,
                                       mlir::Value iv) -> bool {
  auto any_depends = [&](mlir::ValueRange values) {
    for (mlir::Value v : values) {
      if (ssaDependsOn(v, iv)) {
        return true;
      }
    }
    return false;
  };
  if (any_depends(transfer.getSourceIndices())) return false;
  if (any_depends(transfer.getDestIndices())) return false;
  if (any_depends(transfer.getSourceSizes())) return false;
  if (any_depends(transfer.getDestSizes())) return false;
  return true;
}
