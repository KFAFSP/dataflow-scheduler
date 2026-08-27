//===-------------------------------------------------------------*- c++ -*-==//
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

#ifndef DATAFLOW_SCHEDULER_TRANSFORMS_UTILS_HOISTING_H_
#define DATAFLOW_SCHEDULER_TRANSFORMS_UTILS_HOISTING_H_

#include <llvm/ADT/SmallVector.h>
#include <mlir/IR/Dominance.h>
#include <mlir/IR/Operation.h>

namespace scheduler {

/// Obtains the smallest set of SSA dominators of @p op .
///
/// A region SSA dominates @p op if it or one of its parents defines a value
/// used by @p op or any of its descendants.
[[nodiscard]]
auto getSSADominators(mlir::Operation* op) -> llvm::SmallVector<mlir::Region*>;

/// Finds the topmost operation to which @p op can be hoisted according to SSA.
[[nodiscard]]
auto findHoistingTarget(
    mlir::Operation* op,
    llvm::function_ref<bool(mlir::Region*)> should_hoist_out_of)
    -> mlir::Operation*;

/// Finds the smallest set of dominating writes to @p restricted in @p region .
///
/// The location @p restricted is assumed to not be aliased by any other value.
/// If the result is present, it indicates the smallest set of ops writing to
/// @p restricgted that together dominate all others.
[[nodiscard]]
auto findDominatingWritesIn(mlir::Value restricted, mlir::Region* region,
                            mlir::DominanceInfo& dominance)
    -> std::optional<llvm::SmallVector<mlir::Operation*>>;

/// Finds the single dominating invariant write to @p restricted in @p region .
///
/// The location @p restricted is assumed to not be aliased by any other value.
/// The write is considered invariant when neither it nor its children depend
/// on any values defined within the region, and don't have side effects.
[[nodiscard]]
auto findInvariantWriteIn(mlir::Value restricted, mlir::Region* region,
                          mlir::DominanceInfo& dominance) -> mlir::Operation*;

}  // namespace scheduler

#endif  // DATAFLOW_SCHEDULER_TRANSFORMS_UTILS_HOISTING_H_
