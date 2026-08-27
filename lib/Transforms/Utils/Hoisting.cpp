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

#include "dataflow-scheduler/Transforms/Utils/Hoisting.h"

#include <llvm/ADT/STLExtras.h>
#include <llvm/Support/DebugLog.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/Visitors.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>
#include <mlir/Support/WalkResult.h>

#define DEBUG_TYPE "dataflow-scheduler-hoisting"

using namespace scheduler;

namespace {

const auto kSkipRegions = mlir::OpPrintingFlags().skipRegions();

}  // namespace

auto scheduler::getSSADominators(mlir::Operation* op)
    -> llvm::SmallVector<mlir::Region*> {
  // Maintain a linear-scan set of dominating regions.
  llvm::SmallVector<mlir::Region*> dominators;
  const auto add_dominator = [&](mlir::Region* region) {
    auto it = dominators.begin();
    for (; it != dominators.end(); ++it) {
      if (region->isAncestor(*it)) {
        // The new region is the same as or above a known dominator, so no
        // changes are necessary.
        return;
      }
      if ((*it)->isAncestor(region)) {
        // We've found a dominator above the new region, which means we can
        // reduce our set of dominators.
        break;
      }
    }

    if (it == dominators.end()) {
      // We record a new, unrelated dominator.
      dominators.push_back(region);
      return;
    }

    // Replace the old dominator with the more narrow region.
    *it = region;

    // We know that all dominators preceeding it aren't descendants of region.
    // But dominators after it might still be, in which case they are now
    // obsolete and can be removed.
    dominators.erase(std::remove_if(std::next(it), dominators.end(),
                                    [&](mlir::Region* dominator) -> bool {
                                      return dominator->isAncestor(region);
                                    }),
                     dominators.end());
  };

  op->walk<mlir::WalkOrder::PreOrder>(
      [&](mlir::Operation* child) -> mlir::WalkResult {
        // Record all sources of operands originating from outside the op.
        for (auto operand : child->getOperands()) {
          auto* const source = operand.getParentRegion();
          if (!op->isAncestor(source->getParentOp())) {
            add_dominator(source);
          }
        }

        if (child->hasTrait<mlir::OpTrait::IsIsolatedFromAbove>()) {
          return mlir::WalkResult::skip();
        }
        return mlir::WalkResult::advance();
      });

  return dominators;
}

auto scheduler::findHoistingTarget(
    mlir::Operation* op,
    llvm::function_ref<bool(mlir::Region*)> should_hoist_out_of)
    -> mlir::Operation* {
  if (op->mightHaveTrait<mlir::OpTrait::IsTerminator>()) {
    // Terminators can never be hoisted.
    return op;
  }

  auto* result = op;

  const auto has_uses = !op->use_empty();
  const auto dominators = getSSADominators(op);

  while (auto* const target = result->getParentOp()) {
    if (has_uses &&
        target->mightHaveTrait<mlir::OpTrait::IsIsolatedFromAbove>()) {
      // The definitions would no longer be able to reach the uses if we hoist
      // to the parent region.
      LDBG() << "can't hoist " << mlir::OpWithFlags(op, kSkipRegions);
      LDBG() << "  above " << mlir::OpWithFlags(target, kSkipRegions);
      LDBG() << "  target is isolated from above";
      break;
    }

    if (llvm::any_of(dominators, [&](mlir::Region* dominator) -> bool {
          return !dominator->isAncestor(target->getParentRegion());
        })) {
      // The uses would no longer be reached by the definitions if we hoist to
      // the parent region.
      LDBG() << "can't hoist " << mlir::OpWithFlags(op, kSkipRegions);
      LDBG() << "  above " << mlir::OpWithFlags(target, kSkipRegions);
      LDBG() << "  uses are not dominated";
      break;
    }

    if (!should_hoist_out_of(result->getParentRegion())) {
      break;
    }

    result = target;
  }

  return result;
}

auto scheduler::findDominatingWritesIn(mlir::Value restricted,
                                       mlir::Region* region,
                                       mlir::DominanceInfo& dominance)
    -> std::optional<llvm::SmallVector<mlir::Operation*>> {
  // Maintain a linear-scan set of candidates.
  llvm::SmallVector<mlir::Operation*> candidates;
  const auto add_candidate = [&](mlir::Operation* op) {
    auto it = candidates.begin();
    for (; it != candidates.end(); ++it) {
      if (dominance.dominates(*it, op)) {
        // An existing candidate is the same as or dominates op, so no change
        // is necessary.
        return;
      }
      if (dominance.dominates(op, *it)) {
        // We've found a candidate that is dominated by the new op, which means
        // we can reduce our set of candidates.
        break;
      }
    }

    if (it == candidates.end()) {
      // We record a new, unrelated candidate.
      candidates.push_back(op);
      return;
    }

    // Replace the old candidate with its dominator.
    *it = op;

    // We know that all candidates preceeding it aren't dominated by op. But
    // candidates after it might still be, in which case they are now obsolete
    // and can be removed.
    candidates.erase(std::remove_if(std::next(it), candidates.end(),
                                    [&](mlir::Operation* candidate) -> bool {
                                      return dominance.dominates(op, candidate);
                                    }),
                     candidates.end());
  };

  // Process just the users, since the location is restricted.
  for (auto* const user : restricted.getUsers()) {
    if (user->getParentRegion()->isProperAncestor(region)) {
      // User is outside the region.
      continue;
    }

    const auto effects = mlir::getEffectsRecursively(user);
    if (!effects) {
      // We can't judge the effects of this operation.
      LDBG() << "giving up on " << restricted;
      LDBG() << "  unknown user " << mlir::OpWithFlags(user, kSkipRegions);
      return std::nullopt;
    }
    if (!llvm::any_of(
            *effects,
            [&](const mlir::MemoryEffects::EffectInstance& effect) -> bool {
              return llvm::isa<mlir::MemoryEffects::Write>(
                         effect.getEffect()) &&
                     effect.getValue() == restricted;
            })) {
      // The operation does not write to the value.
      continue;
    }

    add_candidate(user);
  }

  return candidates;
}

auto scheduler::findInvariantWriteIn(mlir::Value restricted,
                                     mlir::Region* region,
                                     mlir::DominanceInfo& dominance)
    -> mlir::Operation* {
  // Find the single dominating write to restricted in the region.
  const auto maybe_writes =
      findDominatingWritesIn(restricted, region, dominance);
  if (!maybe_writes) {
    LDBG() << "giving up on " << restricted;
    LDBG() << "  no dominating writes";
    return nullptr;
  }
  if (maybe_writes->size() != 1) {
    LDBG() << "giving up on " << restricted;
    for (auto* write : *maybe_writes) {
      LDBG() << "  written by " << mlir::OpWithFlags(write, kSkipRegions);
    }
    return nullptr;
  }
  auto* const result = maybe_writes->front();

  // Ensure that the write is invariant. We use a simple algorithm that
  // checks whether the operands to the write are defined outside of the
  // region of interest, but ignoring the restricted location and the children
  // of the write we found.
  const auto is_invariant_value = [&](mlir::Value value) -> bool {
    if (value == restricted) {
      return true;
    }
    if (result->isAncestor(value.getParentRegion()->getParentOp())) {
      return true;
    }

    if (auto* const definition = value.getDefiningOp();
        definition && !mlir::isMemoryEffectFree(definition)) {
      // Give up.
      return false;
    }

    return value.getParentRegion()->isProperAncestor(region);
  };
  const auto is_invariant_op = [&](mlir::Operation* op) -> bool {
    for (auto operand : op->getOperands()) {
      if (!is_invariant_value(operand)) {
        LDBG() << "giving up on " << mlir::OpWithFlags(result, kSkipRegions);
        LDBG() << "  " << operand << " is not invariant";
        return false;
      }
    }

    return true;
  };
  const auto visitor = [&](mlir::Operation* child) -> mlir::WalkResult {
    if (!is_invariant_op(child)) {
      return mlir::WalkResult::interrupt();
    }

    if (child->hasTrait<mlir::OpTrait::IsIsolatedFromAbove>()) {
      return mlir::WalkResult::skip();
    }
    return mlir::WalkResult::advance();
  };
  if (result->walk<mlir::WalkOrder::PreOrder>(visitor).wasInterrupted()) {
    return nullptr;
  }

  return result;
}
