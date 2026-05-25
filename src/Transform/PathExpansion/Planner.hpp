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
  StageNode* node = nullptr;
  std::optional<ArchitectureGraph::Resource> anchor_resource;
  llvm::SmallVector<StageEndpoint> input_endpoints;
  llvm::SmallVector<StageEndpoint> output_endpoints;
  bool is_transfer_only = false;
  bool is_compute_containing = false;
};

//===----------------------------------------------------------------------===//
// Materialization Side Information
//===----------------------------------------------------------------------===//

/// Forward declarations for pointer-based references
struct PrivateResourceSpec;
struct PrivateResourceAllocation;

/// Specification for a new private resource to be allocated
/// Can represent either a memory buffer or a FIFO
struct PrivateResourceSpec {
  enum class Kind {
    kUnknown,
    kMemoryBuffer,  // Allocate a memref in the specified memory space
    kFifo,          // Allocate a FIFO with the specified type
  };

  Kind kind = Kind::kUnknown;

  // For Memory Buffer:
  ArchitectureGraph::Resource memory_resource;  // LX, HBM, etc.

  // For Fifo:
  mlir::ktdf::FifoType fifo_type;  // The FIFO type to allocate
  // How many elements for each slot in the FIFO allocation.
  // Note the size of this list represents the number of slots in the
  // fifo.allocate. For example:
  //     %slot1, %slot2 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"lxlu-to-sfp",
  //     2>, !ktdf.fifo.slot<"lxlu-to-sfp", 4>
  // \p elements_per_slot would be [2, 4]
  llvm::SmallVector<int64_t> elements_per_slot;

  // For both fifo and memory-buffers:
  llvm::SmallVector<int64_t> shape;
  mlir::Type element_type;

  void print(llvm::raw_ostream& os) const;
  void dump() const;
};

/// Represents an allocated private resource with its SSA value(s)
/// This is created during materialization and tracked by the factory
struct PrivateResourceAllocation {
  const PrivateResourceSpec* spec;  // Pointer to the spec that created this

  // For memory buffers: single SSA value
  // For FIFOs: one SSA value per slot
  llvm::SmallVector<mlir::Value> ssa_values;

  PrivateResourceAllocation(const PrivateResourceSpec* s) : spec(s) {}
};

/// Factory for managing private resource allocations
/// This replaces index-based schemes with pointer-based references
class PrivateResourceFactory {
 public:
  /// Create a new resource spec and return a pointer to it
  /// The factory owns the spec and keeps it alive
  PrivateResourceSpec* createMemoryBuffer(
      ArchitectureGraph::Resource memory_resource,
      llvm::ArrayRef<int64_t> shape, mlir::Type element_type);

  PrivateResourceSpec* createFifo(mlir::ktdf::FifoType fifo_type,
                                  llvm::ArrayRef<int64_t> elements_per_slot,
                                  mlir::Type element_type);

  /// Get all resource specs created by this factory
  llvm::ArrayRef<std::unique_ptr<PrivateResourceSpec>> getSpecs() const {
    return specs_;
  }

  /// Register an allocation for a spec (called during materialization)
  void registerAllocation(const PrivateResourceSpec* spec,
                          llvm::ArrayRef<mlir::Value> ssa_values);

  /// Look up the allocation for a spec
  const PrivateResourceAllocation* getAllocation(
      const PrivateResourceSpec* spec) const;

 private:
  // Owned resource specs
  llvm::SmallVector<std::unique_ptr<PrivateResourceSpec>> specs_;

  // Map from spec pointer to its allocation
  llvm::DenseMap<const PrivateResourceSpec*,
                 std::unique_ptr<PrivateResourceAllocation>>
      allocations_;
};

// Forward declaration
struct TransferMaterializationInfo;

/// Factory for creating and managing TransferMaterializationInfo objects
/// Owns all transfer info objects and provides factory methods
class TransferInfoFactory {
 public:
  /// Factory method: Create transfer from template with intermediate buffer
  TransferMaterializationInfo* createFromTemplateWithBuffer(
      mlir::ktdf::DataTransferOp template_op,
      const ArchitectureGraph::EdgeInfo& edge,
      ArchitectureGraph::Resource intermediate_resource,
      ArchitectureGraph::Resource current_resource, bool intermediate_is_source,
      const PrivateResourceSpec* buffer_spec, mlir::OpBuilder& builder);

  /// Factory method: Create transfer for FIFO operation
  TransferMaterializationInfo* createFromFifoOp(
      mlir::Operation* fifo_op, const ArchitectureGraph::EdgeInfo& edge,
      ArchitectureGraph::Resource source_resource,
      ArchitectureGraph::Resource dest_resource,
      const PrivateResourceSpec* fifo_spec, size_t slot_index, bool is_read);

  /// Factory method: Create synthetic transfer for intermediate stage
  TransferMaterializationInfo* createSynthetic(
      const ArchitectureGraph::EdgeInfo& edge,
      ArchitectureGraph::Resource source_resource,
      ArchitectureGraph::Resource dest_resource,
      const PrivateResourceSpec* source_spec, size_t source_slot_index,
      llvm::ArrayRef<mlir::Value> source_indices,
      llvm::ArrayRef<mlir::OpFoldResult> source_sizes,
      const PrivateResourceSpec* dest_spec, size_t dest_slot_index,
      llvm::ArrayRef<mlir::Value> dest_indices,
      llvm::ArrayRef<mlir::OpFoldResult> dest_sizes,
      mlir::MLIRContext* context);

 private:
  // Owned transfer info objects
  llvm::SmallVector<std::unique_ptr<TransferMaterializationInfo>> transfers_;
};

/// Information needed to materialize a transfer operation
struct TransferMaterializationInfo {
  mlir::Operation* template_op =
      nullptr;  // Template to derive from (can be null for synthetic)
  ArchitectureGraph::EdgeInfo hop;  // Architecture hop this implements
  ArchitectureGraph::Resource source_resource;  // Source resource for this hop
  ArchitectureGraph::Resource
      dest_resource;  // Destination resource for this hop

  // Pointers to private resource specs for source/dest (nullptr if not a
  // private resource)
  const PrivateResourceSpec* source_private_resource = nullptr;
  const PrivateResourceSpec* dest_private_resource = nullptr;

  // For FIFO resources: which slot index to use (0-based)
  // For memory buffers: always 0 (single SSA value)
  size_t source_slot_index = 0;
  size_t dest_slot_index = 0;

  // Indices and sizes for the transfer (captured from template or neighboring
  // transfers) Indices are stored as Values that will be materialized during
  // code generation Sizes are stored as OpFoldResult to support both static
  // constants and dynamic SSA values
  llvm::SmallVector<mlir::Value> source_indices;
  llvm::SmallVector<mlir::OpFoldResult> source_sizes;
  llvm::SmallVector<mlir::Value> dest_indices;
  llvm::SmallVector<mlir::OpFoldResult> dest_sizes;

  void print(llvm::raw_ostream& os) const;
  void dump() const;

  /// Print debug information about this transfer (for LLVM_DEBUG)
  void printDebug(llvm::raw_ostream& os) const;
};

/// Side information attached to a StageNode for materialization
/// This is stored separately and looked up by the materializer
struct StageMaterializationInfo {
  enum class Kind {
    kPreserveOriginal,  // Keep original stage body unchanged (rare)
    kAdaptFifoKinds,  // Preserve body but update FIFO types (usually needed for
                      // compute stages)
    kAdaptTransfer,   // Preserve body but update transfer source or dest types
                      // (needed for transfer stages that are intermdiate-stage
                      // facing)
    kSyntheticTransfer,  // Generate new transfer stage (usually for
                         // intermediate memory stages ie LX)
  };

  Kind kind = Kind::kPreserveOriginal;

  // Resource associated with this stage (determined in Step 1)
  std::optional<ArchitectureGraph::Resource> stage_resource;

  // For AdaptFifoKinds, AdaptTransfer, and SyntheticTransfer:
  // Contains transfer operations with pointers to private resources
  // Pointers are owned by TransferInfoFactory
  llvm::SmallVector<TransferMaterializationInfo*, 4> transfers;

  // which unit this stage runs on
  std::optional<mlir::ktdf::ComponentType> applicable_unit;

  void print(llvm::raw_ostream& os) const;
  void dump() const;
};

//===----------------------------------------------------------------------===//
// Planning Result
//===----------------------------------------------------------------------===//

/// Result of path expansion planning
/// Contains the modified PipelineTree and minimal side information
struct PathExpansionPlan {
  // The modified PipelineTree is the primary output
  // It contains the final stage DAG with dependencies
  PipelineTree* modified_tree;

  // Side information for materialization (indexed by StageNode*)
  llvm::DenseMap<StageNode*, StageMaterializationInfo> stage_info;

  // Resource factory that owns all PrivateResourceSpec objects
  // and tracks their allocations during materialization
  PrivateResourceFactory resource_factory;

  // Transfer info factory that owns all TransferMaterializationInfo objects
  TransferInfoFactory transfer_factory;

  // Whether any changes were made
  bool changed = false;

  // Default constructor
  PathExpansionPlan() = default;

  /// Get stage materialization info for a stage (with assertion)
  StageMaterializationInfo& getStageInfo(StageNode* stage) {
    assert(stage_info.count(stage) && "Stage should have materialization info");
    return stage_info[stage];
  }

  const StageMaterializationInfo& getStageInfo(StageNode* stage) const {
    assert(stage_info.count(stage) && "Stage should have materialization info");
    return stage_info.at(stage);
  }

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
mlir::LogicalResult validateLinearChain(
    llvm::ArrayRef<StageNode*> sorted_stages);

//===----------------------------------------------------------------------===//
// Planning Functions
//===----------------------------------------------------------------------===//

/// Analyze stages and infer their endpoints and properties
/// This is a pure analysis step that doesn't modify the tree
/// Takes sorted stages to ensure path inference order matches DAG order
mlir::FailureOr<llvm::SmallVector<StageSummary>> analyzeStages(
    llvm::ArrayRef<StageNode*> sorted_stages);

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
std::unique_ptr<PathExpansionPlan> planPathExpansion(
    PipelineTree& tree, PipelineNode* pipeline,
    const ArchitectureGraph& arch_graph);

}  // namespace scheduler

#endif  // KTIR_SCHEDULER_TRANSFORM_PATHEXPANSION_PLANNER_HPP

// Made with Bob
