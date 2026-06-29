//===-- WriteSetScan.cpp ----------------------------------------*- c++ -*-===//
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

#include "dataflow-scheduler/Analysis/WriteSetScan.h"

#include <llvm/ADT/DenseSet.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>

#include "dataflow-scheduler/Dialect/KTDF/KTDF.h"

using namespace scheduler;

namespace {

/// Build the alias set reachable from `seed` via memref view/cast ops
/// within the IR. This is a fixed-point forward walk through uses of the
/// seed and any op we consider aliasing-producing.
void buildAliasSet(mlir::Value seed, llvm::DenseSet<mlir::Value>& aliases) {
  llvm::SmallVector<mlir::Value> worklist{seed};
  while (!worklist.empty()) {
    mlir::Value v = worklist.pop_back_val();
    if (!aliases.insert(v).second) continue;
    for (mlir::OpOperand& use : v.getUses()) {
      mlir::Operation* user = use.getOwner();
      // Ops that produce an aliased memref from their input.
      if (mlir::isa<mlir::memref::CastOp, mlir::memref::ViewOp,
                    mlir::memref::SubViewOp, mlir::memref::ReinterpretCastOp,
                    mlir::memref::MemorySpaceCastOp>(user)) {
        for (mlir::Value result : user->getResults()) {
          worklist.push_back(result);
        }
      }
      // Other ops (stores, linalg, transfers) are consumers, not
      // producers of new aliases — skip.
    }
  }
}

bool opIsWriter(mlir::Operation* op,
                const llvm::DenseSet<mlir::Value>& aliases) {
  // ktdf.data_transfer writes to its destination when dest is a memref.
  if (auto transfer = mlir::dyn_cast<mlir::ktdf::DataTransferOp>(op)) {
    if (transfer.isDestMemRef() &&
        aliases.contains(transfer.getDestination())) {
      return true;
    }
    return false;
  }
  // memref.store writes to its destination memref.
  if (auto store = mlir::dyn_cast<mlir::memref::StoreOp>(op)) {
    return aliases.contains(store.getMemRef());
  }
  // linalg structured ops: any output operand whose type is a memref is a
  // writer.
  if (auto linalg_op = mlir::dyn_cast<mlir::linalg::LinalgOp>(op)) {
    for (mlir::OpOperand& out : linalg_op.getDpsInitsMutable()) {
      if (aliases.contains(out.get())) {
        return true;
      }
    }
    return false;
  }
  return false;
}

}  // namespace

auto scheduler::regionWritesTo(mlir::Region& region, mlir::Value memref,
                               mlir::Operation* ignore) -> bool {
  llvm::DenseSet<mlir::Value> aliases;
  buildAliasSet(memref, aliases);
  bool found_writer = false;
  region.walk([&](mlir::Operation* op) {
    if (found_writer) return mlir::WalkResult::interrupt();
    if (op == ignore) return mlir::WalkResult::advance();
    if (opIsWriter(op, aliases)) {
      found_writer = true;
      return mlir::WalkResult::interrupt();
    }
    return mlir::WalkResult::advance();
  });
  return found_writer;
}
