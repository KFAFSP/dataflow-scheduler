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
// Path Expansion Planner
//
// This file defines a lightweight planning layer for path expansion that
// operates directly on PipelineTree. The planner analyzes the original
// pipeline, compares it with the architecture graph's shortest path, and
// updates the PipelineTree in-place while storing only minimal side
// information needed by the materializer.
//
// Key design principles:
// - PipelineTree is the single source of truth for stage structure
// - Planning updates PipelineTree directly (adding/modifying nodes)
// - Side information is minimal and only for materialization needs
// - No redundant parallel abstractions to PipelineTree
//
//===----------------------------------------------------------------------===//

#ifndef KTIR_SCHEDULER_TRANSFORM_PATHEXPANSION_PLANNER_HPP
#define KTIR_SCHEDULER_TRANSFORM_PATHEXPANSION_PLANNER_HPP

#include <optional>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LogicalResult.h"
#include "src/Analysis/ArchitectureGraph.hpp"
#include "src/Analysis/PipelineTree.hpp"
#include "src/Dialects/KTDF/KTDFOps.hpp"

namespace scheduler {

//===----------------------------------------------------------------------===//
// Lightweight Analysis Structures
//===----------------------------------------------------------------------===//

/// Inferred endpoint information for a stage
struct StageEndpoint {
  ArchitectureGraph::Resource resource;
  std::optional<mlir::ktdf::ComponentType> component;
};

/// Summary of a stage's inferred properties (used during analysis only)
struct StageSummary {
  StageNode *node = nullptr;
  std::optional<ArchitectureGraph::Resource> anchor_resource;
  llvm::SmallVector<StageEndpoint> input_endpoints;
  llvm::SmallVector<StageEndpoint> output_endpoints;
  bool is_transfer_only = false;
  bool is_compute_containing = false;
};

//===----------------------------------------------------------------------===//
// Materialization Side Information
//===----------------------------------------------------------------------===//

/// Information needed to materialize a transfer operation
struct TransferMaterializationInfo {
  mlir::ktdf::DataTransferOp template_op;  // Template to derive from
  ArchitectureGraph::EdgeInfo hop;         // Architecture hop this implements
  ArchitectureGraph::Resource source_resource;  // Source resource for this hop
  ArchitectureGraph::Resource dest_resource;    // Destination resource for this hop
};

/// Side information attached to a StageNode for materialization
/// This is stored separately and looked up by the materializer
struct StageMaterializationInfo {
  enum class Kind {
    kPreserveOriginal,      // Keep original stage body unchanged
    kAdaptFifoKinds,        // Preserve body but update FIFO types
    kSyntheticTransfer,     // Generate new transfer stage
  };

  Kind kind = Kind::kPreserveOriginal;
  
  // For AdaptFifoKinds and SyntheticTransfer:
  llvm::SmallVector<TransferMaterializationInfo> transfers;
  
  mlir::ktdf::ComponentType applicable_unit;
};

/// Specification for a new private resource to be allocated
struct PrivateResourceSpec {
  ArchitectureGraph::Resource resource; // resource to use when allocating buffers (eg LX)
  mlir::Type element_type;
  llvm::SmallVector<int64_t> shape;
};

//===----------------------------------------------------------------------===//
// Planning Result
//===----------------------------------------------------------------------===//

/// Result of path expansion planning
/// Contains the modified PipelineTree and minimal side information
struct PathExpansionPlan {
  // The modified PipelineTree is the primary output
  // It contains the final stage DAG with dependencies
  PipelineTree *modified_tree;
  
  // Side information for materialization (indexed by StageNode*)
  llvm::DenseMap<StageNode*, StageMaterializationInfo> stage_info;
  
  // New private resources to allocate
  llvm::SmallVector<PrivateResourceSpec> new_resources;
  
  // Whether any changes were made
  bool changed = false;
  
  // Default constructor
  PathExpansionPlan() = default;
  
  // Delete copy operations since PipelineTree is not copyable
  PathExpansionPlan(const PathExpansionPlan&) = delete;
  PathExpansionPlan& operator=(const PathExpansionPlan&) = delete;
  
  // Delete move operations since PipelineTree is not movable
  PathExpansionPlan(PathExpansionPlan&&) = delete;
  PathExpansionPlan& operator=(PathExpansionPlan&&) = delete;
};

//===----------------------------------------------------------------------===//
// Stage DAG Validation Functions
//===----------------------------------------------------------------------===//

/// Validate that the stage topology is a single linear chain
/// Returns failure if the DAG has branches or multiple paths
mlir::LogicalResult
validateLinearChain(llvm::ArrayRef<StageNode*> sorted_stages);

//===----------------------------------------------------------------------===//
// Planning Functions
//===----------------------------------------------------------------------===//

/// Analyze stages and infer their endpoints and properties
/// This is a pure analysis step that doesn't modify the tree
/// Takes sorted stages to ensure path inference order matches DAG order
mlir::FailureOr<llvm::SmallVector<StageSummary>>
analyzeStages(llvm::ArrayRef<StageNode*> sorted_stages);

/// Infer the original resource path from stage summaries
/// Returns the sequence of resources implied by the current stages
mlir::FailureOr<llvm::SmallVector<ArchitectureGraph::Resource>>
inferOriginalPath(llvm::ArrayRef<StageSummary> summaries);

/// Main planning entry point
/// Analyzes the pipeline, compares with architecture graph, and if needed:
/// 1. Updates the PipelineTree in-place (modifying stage DAG)
/// 2. Attaches materialization info to stages
/// 3. Records new private resources needed
/// Returns a unique_ptr to the plan, or nullptr on failure
std::unique_ptr<PathExpansionPlan>
planPathExpansion(PipelineTree &tree,
                  PipelineNode *pipeline,
                  const ArchitectureGraph &arch_graph);

}  // namespace scheduler

#endif  // KTIR_SCHEDULER_TRANSFORM_PATHEXPANSION_PLANNER_HPP

// Made with Bob
