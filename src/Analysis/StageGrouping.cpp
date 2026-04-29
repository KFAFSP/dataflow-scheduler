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
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "src/Dialects/KTDF/KTDFTypes.hpp"
#include "src/Utils/OperationUtils.hpp"

namespace ktdf {

//===----------------------------------------------------------------------===//
// StageGroupingAnalysis Implementation
//===----------------------------------------------------------------------===//

StageGroupingAnalysis::StageGroupingAnalysis(mlir::ktdf::PipelineOp pipeline) {
  llvm::SmallVector<mlir::ktdf::StageOp> stages;
  collectStages(stages, pipeline);

  // Delegate to ArrayRef constructor
  *this = StageGroupingAnalysis(llvm::ArrayRef<mlir::ktdf::StageOp>(stages));
}

StageGroupingAnalysis::StageGroupingAnalysis(
    llvm::ArrayRef<mlir::ktdf::StageOp> stages) {
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
  unsigned groupId = 0;

  for (auto it = ec.begin(), end = ec.end(); it != end; ++it) {
    // Skip non-leader entries
    if (!(*it)->isLeader()) continue;

    // Create a new group for this equivalence class
    StageGroup group;

    // Iterate over all members of this equivalence class
    for (auto mi = ec.member_begin(**it); mi != ec.member_end(); ++mi) {
      unsigned stageIdx = *mi;
      group.addStage(stages[stageIdx]);

      auto mutableStage = const_cast<mlir::ktdf::StageOp&>(stages[stageIdx]);
      stageToGroup[mutableStage.getOperation()] = groupId;
    }

    groups.push_back(group);
    groupId++;
  }
}

void StageGroupingAnalysis::analyze() {
  llvm::outs() << "Stage grouping analysis complete\n";
  llvm::outs() << "  Total groups: " << groups.size() << "\n";
  llvm::outs() << "  Grouping based on:\n";
  llvm::outs() << "    - Shared SFP register usage\n";
  llvm::outs() << "    - Shared FIFO buffer access\n";
}

const StageGroup* StageGroupingAnalysis::getGroupForStage(
    mlir::ktdf::StageOp stage) const {
  auto it = stageToGroup.find(stage.getOperation());
  if (it == stageToGroup.end()) return nullptr;
  return &groups[it->second];
}

bool StageGroupingAnalysis::areInSameGroup(mlir::ktdf::StageOp stage1,
                                           mlir::ktdf::StageOp stage2) const {
  auto it1 = stageToGroup.find(stage1.getOperation());
  auto it2 = stageToGroup.find(stage2.getOperation());

  if (it1 == stageToGroup.end() || it2 == stageToGroup.end()) return false;

  return it1->second == it2->second;
}

void StageGroupingAnalysis::print(llvm::raw_ostream& OS) const {
  OS << "  Total groups: " << groups.size() << "\n\n";

  for (size_t i = 0; i < groups.size(); ++i) {
    const auto& group = groups[i];
    OS << "  Group " << i << ":\n";
    OS << "    Stages: [";
    bool first = true;
    for (auto stage : group.getStages()) {
      if (!first) OS << ", ";
      OS << "stage ";
      scheduler::utils::printLocation(OS, stage.getOperation());
      first = false;
    }
    OS << "]\n";
    OS << "    Single stage: " << (group.isSingleStage() ? "yes" : "no")
       << "\n";
  }
}

void StageGroupingAnalysis::dump() const {
  llvm::outs() << "Stage Grouping Analysis Results:\n";
  print(llvm::outs());
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

  llvm::SmallSet<mlir::ktdf::MemorySpace, 4> stage1RegFiles;
  llvm::SmallSet<mlir::ktdf::MemorySpace, 4> stage2RegFiles;
  llvm::SmallSet<mlir::ktdf::FifoType, 4> stage1FIFOs;
  llvm::SmallSet<mlir::ktdf::FifoType, 4> stage2FIFOs;

  // Helper to check if a memory space is a register file
  auto isRegisterFile = [](mlir::ktdf::MemorySpace memSpace) {
    // Currently only SFPRegisterFile is a register file
    return memSpace == mlir::ktdf::MemorySpace::SFPRegisterFile;
  };

  // Helper to analyze a value for register files or FIFO types
  auto analyzeValue = [&](mlir::Value val,
                          llvm::SmallSet<mlir::ktdf::MemorySpace, 4>& regFiles,
                          llvm::SmallSet<mlir::ktdf::FifoType, 4>& fifos) {
    if (auto memrefType = mlir::dyn_cast<mlir::MemRefType>(val.getType())) {
      auto memSpace = memrefType.getMemorySpace();
      if (memSpace) {
        // Try to cast to the custom MemorySpaceCustomAttr
        if (auto memSpaceAttr =
                mlir::dyn_cast<mlir::ktdf::MemorySpaceCustomAttr>(memSpace)) {
          auto memSpaceEnum = memSpaceAttr.getValue();
          if (isRegisterFile(memSpaceEnum)) {
            regFiles.insert(memSpaceEnum);
          }
        }
      }
    } else if (auto fifoSlotType =
                   mlir::dyn_cast<mlir::ktdf::FifoSlotType>(val.getType())) {
      fifos.insert(fifoSlotType.getFifoType());
    }
  };

  // Analyze ALL operations in stage1 - check all operands
  stage1.walk([&](mlir::Operation* op) {
    for (auto operand : op->getOperands()) {
      analyzeValue(operand, stage1RegFiles, stage1FIFOs);
    }
  });

  // Analyze ALL operations in stage2 - check all operands
  stage2.walk([&](mlir::Operation* op) {
    for (auto operand : op->getOperands()) {
      analyzeValue(operand, stage2RegFiles, stage2FIFOs);
    }
  });

  // Group if they share any register file
  for (auto regFile : stage1RegFiles) {
    if (stage2RegFiles.contains(regFile)) {
      return true;
    }
  }

  // Group if they share any FIFO types
  for (auto fifoType : stage1FIFOs) {
    if (stage2FIFOs.contains(fifoType)) {
      return true;
    }
  }

  return false;
}

void StageGroupingAnalysis::analyzeDataTransfer(
    mlir::ktdf::DataTransferOp transfer,
    llvm::SmallSet<mlir::ktdf::MemorySpace, 4>& registerFiles,
    llvm::SmallSet<mlir::ktdf::FifoType, 4>& fifoTypes) {
  // Helper to check if a memory space is a register file
  auto isRegisterFile = [](mlir::ktdf::MemorySpace memSpace) {
    // Currently only SFPRegisterFile is a register file
    return memSpace == mlir::ktdf::MemorySpace::SFPRegisterFile;
  };

  // Check source and destination for register files or FIFO types
  auto checkValue = [&](mlir::Value val) {
    if (auto memrefType = mlir::dyn_cast<mlir::MemRefType>(val.getType())) {
      auto memSpace = memrefType.getMemorySpace();
      if (memSpace) {
        // Extract memory space enum from attribute
        if (auto memSpaceAttr = mlir::dyn_cast<mlir::Attribute>(memSpace)) {
          if (auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(memSpaceAttr)) {
            // Convert integer to MemorySpace enum
            auto memSpaceEnum =
                static_cast<mlir::ktdf::MemorySpace>(intAttr.getInt());
            // Only track register files, not LX or HBM
            if (isRegisterFile(memSpaceEnum)) {
              registerFiles.insert(memSpaceEnum);
            }
          }
        }
      }
    } else if (auto fifoSlotType =
                   mlir::dyn_cast<mlir::ktdf::FifoSlotType>(val.getType())) {
      // Store only the FIFO type enum (e.g., "sfp-to-lxsu", "lxlu-to-sfp")
      // not the full type with sizes and element types
      fifoTypes.insert(fifoSlotType.getFifoType());
    }
  };

  checkValue(transfer.getSource());
  checkValue(transfer.getDestination());
}

}  // namespace ktdf

// Made with Bob
