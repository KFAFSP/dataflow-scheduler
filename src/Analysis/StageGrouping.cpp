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

#include "StageGrouping.hpp"

#include "llvm/ADT/EquivalenceClasses.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "src/Dialects/KTDF/KTDFTypes.hpp"
#include "src/Utils/OperationUtils.hpp"

#define DEBUG_TYPE "stage-grouping"
namespace ktdf {

//===----------------------------------------------------------------------===//
// StageGroupingAnalysis Implementation
//===----------------------------------------------------------------------===//

StageGroupingAnalysis::StageGroupingAnalysis(mlir::ktdf::PipelineOp pipeline) {
  llvm::SmallVector<mlir::ktdf::StageOp> stages;
  collectStages(stages, pipeline);
  init(stages);
}

StageGroupingAnalysis::StageGroupingAnalysis(
    llvm::ArrayRef<mlir::ktdf::StageOp> stages) {
  init(stages);
}

void StageGroupingAnalysis::init(llvm::ArrayRef<mlir::ktdf::StageOp> stages) {
  if (stages.empty()) return;

  // Use EquivalenceClasses to group stages based on shared resources
  llvm::EquivalenceClasses<unsigned> ec;

  // Initialize: each stage starts in its own equivalence class
  for (size_t i = 0; i < stages.size(); ++i) {
    ec.insert(i);
  }

  // Merge equivalence classes for stages that should be grouped together
  for (size_t i = 0; i < stages.size(); ++i) {
    for (size_t j = i + 1; j < stages.size(); ++j) {
      if (shouldGroupTogether(stages[i], stages[j])) {
        ec.unionSets(i, j);
      }
    }
  }

  // Build groups by iterating over equivalence classes
  // Only process leader entries to avoid creating empty groups
  unsigned group_id = 0;

  for (auto it = ec.begin(), end = ec.end(); it != end; ++it) {
    // Skip non-leader entries
    if (!(*it)->isLeader()) continue;

    // Create a new group for this equivalence class
    StageGroup group;

    // Iterate over all members of this equivalence class
    for (auto mi = ec.member_begin(**it); mi != ec.member_end(); ++mi) {
      unsigned stage_idx = *mi;
      group.addStage(stages[stage_idx]);

      auto mutable_stage = const_cast<mlir::ktdf::StageOp&>(stages[stage_idx]);
      stage_to_group_[mutable_stage.getOperation()] = group_id;
    }

    groups_.push_back(group);
    group_id++;
  }
  LLVM_DEBUG({
    llvm::dbgs() << "Stage grouping analysis complete\n";
    llvm::dbgs() << "  Total groups: " << groups_.size() << "\n";
  });
}

const StageGroup* StageGroupingAnalysis::getGroupForStage(
    mlir::ktdf::StageOp stage) const {
  auto it = stage_to_group_.find(stage.getOperation());
  if (it == stage_to_group_.end()) return nullptr;
  return &groups_[it->second];
}

bool StageGroupingAnalysis::areInSameGroup(mlir::ktdf::StageOp stage1,
                                           mlir::ktdf::StageOp stage2) const {
  auto it1 = stage_to_group_.find(stage1.getOperation());
  auto it2 = stage_to_group_.find(stage2.getOperation());

  if (it1 == stage_to_group_.end() || it2 == stage_to_group_.end())
    return false;

  return it1->second == it2->second;
}

void StageGroupingAnalysis::print(llvm::raw_ostream& os) const {
  os << "  Total groups: " << groups_.size() << "\n\n";

  for (size_t i = 0; i < groups_.size(); ++i) {
    const StageGroup& group = groups_[i];
    os << "  Group " << i << ":\n";
    os << "    Stages: [";
    bool first = true;
    for (mlir::ktdf::StageOp stage : group.getStages()) {
      if (!first) os << ", ";
      os << "stage ";
      scheduler::utils::printLocation(os, stage.getOperation());
      first = false;
    }
    os << "]\n";
    os << "    Single stage: " << (group.isSingleStage() ? "yes" : "no")
       << "\n";
  }
}

void StageGroupingAnalysis::dump() const {
  llvm::dbgs() << "Stage Grouping Analysis Results:\n";
  print(llvm::dbgs());
}

void StageGroupingAnalysis::collectStages(
    llvm::SmallVector<mlir::ktdf::StageOp>& stages,
    mlir::ktdf::PipelineOp pipeline) {
  // Walk the pipeline body and collect all stage operations
  // getBody() returns a Region&, we need to get its first block
  mlir::Region& region = pipeline.getBody();
  if (!region.empty()) {
    mlir::Block& body = region.front();
    for (auto& op : body.getOperations()) {
      if (auto stage = mlir::dyn_cast<mlir::ktdf::StageOp>(&op)) {
        stages.push_back(stage);
      }
    }
  }
}

bool StageGroupingAnalysis::shouldGroupTogether(mlir::ktdf::StageOp stage1,
                                                mlir::ktdf::StageOp stage2) {
  // Analyze ALL operations in both stages
  // Group stages that:
  // 1. Access the same register file (e.g., both access SFPRegisterFile)
  //    Note: LX and HBM are not register files, so sharing those doesn't
  //    require grouping
  // 2. Access the same FIFO types (just the enum, not sizes/element types)

  llvm::SmallSet<mlir::ktdf::MemorySpace, 4> stage1_reg_files;
  llvm::SmallSet<mlir::ktdf::MemorySpace, 4> stage2_reg_files;
  llvm::SmallSet<mlir::ktdf::FifoType, 4> stage1_fifos;
  llvm::SmallSet<mlir::ktdf::FifoType, 4> stage2_fifos;

  // Helper to check if a memory space is a register file
  auto isRegisterFile = [](mlir::ktdf::MemorySpace mem_space) {
    // Currently only SFPRegisterFile is a register file
    return mem_space == mlir::ktdf::MemorySpace::SFPRegisterFile;
  };

  // Helper to analyze a value for register files or FIFO types
  auto analyzeValue = [&](mlir::Value val,
                          llvm::SmallSet<mlir::ktdf::MemorySpace, 4>& reg_files,
                          llvm::SmallSet<mlir::ktdf::FifoType, 4>& fifos) {
    if (auto memref_type = mlir::dyn_cast<mlir::MemRefType>(val.getType())) {
      auto mem_space = memref_type.getMemorySpace();
      if (mem_space) {
        // Try to cast to the custom MemorySpaceCustomAttr
        if (auto mem_space_attr =
                mlir::dyn_cast<mlir::ktdf::MemorySpaceCustomAttr>(mem_space)) {
          auto mem_space_enum = mem_space_attr.getValue();
          if (isRegisterFile(mem_space_enum)) {
            reg_files.insert(mem_space_enum);
          }
        }
      }
    } else if (auto fifo_slot_type =
                   mlir::dyn_cast<mlir::ktdf::FifoSlotType>(val.getType())) {
      fifos.insert(fifo_slot_type.getFifoType());
    }
  };

  // Analyze ALL operations in stage1 - check all operands
  stage1.walk([&](mlir::Operation* op) {
    for (auto operand : op->getOperands()) {
      analyzeValue(operand, stage1_reg_files, stage1_fifos);
    }
  });

  // Analyze ALL operations in stage2 - check all operands
  stage2.walk([&](mlir::Operation* op) {
    for (auto operand : op->getOperands()) {
      analyzeValue(operand, stage2_reg_files, stage2_fifos);
    }
  });

  // Group if they share any register file
  for (mlir::ktdf::MemorySpace reg_file : stage1_reg_files) {
    if (stage2_reg_files.contains(reg_file)) {
      return true;
    }
  }

  // Group if they share any FIFO types
  for (mlir::ktdf::FifoType fifo_type : stage1_fifos) {
    if (stage2_fifos.contains(fifo_type)) {
      return true;
    }
  }

  return false;
}

}  // namespace ktdf

// Made with Bob
