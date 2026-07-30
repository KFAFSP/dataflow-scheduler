//===-- SliceAnalysis.cpp ---------------------------------------*- c++ -*-===//
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

#include "dataflow-scheduler/Analysis/SliceAnalysis.h"

#include <llvm/ADT/STLExtras.h>
#include <llvm/Support/Casting.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Interfaces/ControlFlowInterfaces.h>
#include <mlir/Interfaces/FunctionInterfaces.h>
#include <mlir/Interfaces/LoopLikeInterface.h>
#include <mlir/Pass/AnalysisManager.h>

using namespace scheduler;

namespace {

void visitControlFlow(mlir::SelectLikeOpInterface select,
                      BackwardSliceAnalysis::map_type& result) {
  if (select->getNumResults() != 1) {
    return;
  }

  result.emplace_or_assign(
      select->getResult(0),
      BackwardSliceAnalysis::Predecessors::exhaustive(
          {select.getTrueValue(), select.getFalseValue()}));
}

void visitControlFlow(mlir::RegionBranchOpInterface branch,
                      BackwardSliceAnalysis::map_type& result) {
  for (auto branch_point : branch.getAllRegionBranchPoints()) {
    llvm::SmallVector<mlir::RegionSuccessor> successors;
    branch.getSuccessorRegions(branch_point, successors);
    for (auto& successor : successors) {
      const auto entry_values = successor.getSuccessorInputs();
      const auto exit_values =
          branch.getSuccessorOperands(branch_point, successor);
      for (auto [entry, exit] : llvm::zip_equal(entry_values, exit_values)) {
        auto& cached =
            result
                .try_emplace(entry,
                             BackwardSliceAnalysis::Predecessors::exhaustive())
                .first->getSecond();
        cached.unite(exit);
      }
    }
  }
}

void visitControlFlow(mlir::BranchOpInterface branch,
                      BackwardSliceAnalysis::map_type& result) {
  for (auto& successor : branch->getBlockOperands()) {
    const auto operands =
        branch.getSuccessorOperands(successor.getOperandNumber());

    for (auto argument : successor.get()->getArguments()) {
      auto& cached =
          result
              .try_emplace(argument,
                           BackwardSliceAnalysis::Predecessors::exhaustive())
              .first->getSecond();
      if (const auto forwarded = operands[argument.getArgNumber()]; forwarded) {
        cached.unite(forwarded);
      }
    }
  }
}

}  // namespace

//===----------------------------------------------------------------------===//
// BackwardSliceAnalysis
//===----------------------------------------------------------------------===//

void BackwardSliceAnalysis::getPredecessors(
    mlir::Value value, bool& is_exhaustive,
    SmallPtrSetImpl<mlir::Value>& predecessors) {
  if (const auto result = llvm::dyn_cast<mlir::OpResult>(value); result) {
    is_exhaustive &= result.getOwner()->isRegistered();
    predecessors.insert_range(result.getOwner()->getOperands());
  }

  const auto& control_flow = getControlFlowPredecessors(value);
  is_exhaustive &= control_flow.is_exhaustive_;
  predecessors.insert_range(control_flow.values_);
}

auto BackwardSliceAnalysis::getControlFlowPredecessors(mlir::Value value)
    -> const Predecessors& {
  {
    // Lookup in cache, emplacing inexact result if none exists.
    auto [it, inserted] = control_flow_.try_emplace(value);
    if (!inserted) {
      return it->second;
    }
  }

  if (const auto argument = llvm::dyn_cast<mlir::BlockArgument>(value);
      argument) {
    if (!argument.getOwner()->hasNoPredecessors()) {
      // Argument is fully determined by predecessors' branch operations.
      auto is_exact = argument.getOwner()->getParentOp()->isRegistered();
      for (auto* const pred : argument.getOwner()->getPredecessors()) {
        if (auto iface = llvm::dyn_cast<mlir::BranchOpInterface>(
                pred->getTerminator())) {
          visitControlFlow(iface, control_flow_);
        } else {
          // We don't understand that one.
          is_exact = false;
        }
      }

      auto& result = control_flow_[value];
      result.is_exhaustive_ = is_exact;
      return result;
    }

    if (auto iface = llvm::dyn_cast<mlir::FunctionOpInterface>(
            argument.getOwner()->getParentOp());
        iface) {
      // We do not perform any inter-procedural analyses.
      return control_flow_[value] = Predecessors::exhaustive();
    }

    if (auto iface = llvm::dyn_cast<mlir::RegionBranchOpInterface>(
            argument.getOwner()->getParentOp());
        iface) {
      // Argument is determined by region branch semantics.
      visitControlFlow(iface, control_flow_);
    }
  } else {
    const auto result = llvm::cast<mlir::OpResult>(value);

    if (auto iface =
            llvm::dyn_cast<mlir::SelectLikeOpInterface>(result.getOwner());
        iface) {
      // Result is determined by select semantics.
      visitControlFlow(iface, control_flow_);
    } else if (auto iface = llvm::dyn_cast<mlir::RegionBranchOpInterface>(
                   result.getOwner());
               iface) {
      // Result is determined by region branch semantics.
      visitControlFlow(iface, control_flow_);
    }
  }

  return control_flow_[value];
}

//===----------------------------------------------------------------------===//
// ForwardSliceAnalysis
//===----------------------------------------------------------------------===//

ForwardSlice::ForwardSlice(BackwardSliceAnalysis& backward,
                           mlir::ValueRange values)
    : backward_(backward) {
  for (auto value : values) {
    cache_[value] = Result::MustContain;
  }
}

auto ForwardSlice::insert(mlir::ValueRange values) -> bool {
  if (llvm::all_of(values, [&](mlir::Value value) -> bool {
        return cache_[value] == Result::MustContain;
      })) {
    return false;
  }

  // This invalidates the cache apart from MustContain.
  map_type temp;
  using std::swap;
  swap(temp, cache_);

  for (auto value : values) {
    cache_[value] = Result::MustContain;
  }
  for (auto [key, value] : temp) {
    if (value == Result::MustContain) {
      cache_[key] = Result::MustContain;
    }
  }

  return true;
}

auto ForwardSlice::contains(mlir::Value value) -> Result {
  if (cache_.empty()) {
    // Short-circuit on the known empty slice.
    return Result::NoContain;
  }

  // Lookup cached result or initialize with MayContain.
  auto [it, invalid] = cache_.try_emplace(value, Result::MayContain);
  if (!invalid) {
    return it->second;
  }

  // Find all predecessors of the value.
  bool is_exhaustive;
  llvm::SmallPtrSet<mlir::Value, 8U> predecessors;
  backward_.getPredecessors(value, is_exhaustive, predecessors);

  // If the set of predecessors is exhaustive, we may assume that the value is
  // independent for now. If it is visited in the recursive search (which can
  // only happen within a graph region, or if we follow block arguments), then
  // it being part of its own cycle should not be an obstacle to independence.
  auto result = it->second =
      is_exhaustive ? Result::NoContain : Result::MayContain;

  for (auto predecessor : predecessors) {
    switch (contains(predecessor)) {
      case Result::MustContain:
        return cache_[value] = Result::MustContain;
      case Result::MayContain:
        result = cache_[value] = Result::MayContain;
        continue;
      case Result::NoContain:
        continue;
    }
  }

  return result;
}

//===----------------------------------------------------------------------===//
// LoopSliceAnalysis
//===----------------------------------------------------------------------===//

namespace {

auto getLoopVariables(mlir::Operation* op) -> llvm::SmallVector<mlir::Value> {
  llvm::SmallVector<mlir::Value> result;

  auto iface = llvm::dyn_cast<mlir::LoopLikeOpInterface>(op);
  if (iface == nullptr) {
    return result;
  }

  if (const auto ivs = iface.getLoopInductionVars(); ivs) {
    llvm::append_range(result, ivs.value());
  }
  llvm::append_range(result, iface.getRegionIterArgs());
  return result;
}

}  // namespace

LoopSliceAnalysis::LoopSliceAnalysis(mlir::Operation* op,
                                     mlir::AnalysisManager& analyses)
    : ForwardSlice(analyses.getAnalysis<BackwardSliceAnalysis>(),
                   getLoopVariables(op)) {}
