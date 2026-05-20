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

#include "Planner.hpp"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LogicalResult.h"
#include "src/Analysis/ArchitectureGraph.hpp"
#include "src/Analysis/PipelineTree.hpp"
#include "src/Dialects/KTDF/KTDFOps.hpp"

#define DEBUG_TYPE "path-expansion-planner"

namespace scheduler {

namespace {

/// Helper to validate FIFO slot index assignment
/// Checks that the assigned slot index matches the original result index
/// from the template transfer operation
void validateFifoSlotIndex(mlir::Value orig_fifo_value,
                           size_t assigned_slot_idx,
                           const PrivateResourceSpec* fifo_spec) {
  if (auto orig_private_result =
          mlir::dyn_cast<mlir::OpResult>(orig_fifo_value)) {
    // Find which result index this was in the original private op
    unsigned orig_result_idx = orig_private_result.getResultNumber();
    // The slot index we assign should match the original result index
    // This ensures we maintain the same slot ordering as the original
    assert(assigned_slot_idx == orig_result_idx &&
           "expected result index to match our calculated fifo slot index");
  }
  // Sanity check: slot index should be within allocated slots
  assert(assigned_slot_idx < fifo_spec->elements_per_slot.size() &&
         "Slot index should be within the allocated slots for this FIFO type");
}

/// Helper to infer resource from memory space attribute
std::optional<ArchitectureGraph::Resource> inferResourceFromMemorySpace(
    mlir::Attribute memory_space) {
  if (!memory_space) return std::nullopt;

  auto memory_space_attr =
      mlir::dyn_cast<mlir::ktdf::MemorySpaceCustomAttr>(memory_space);
  if (!memory_space_attr) return std::nullopt;

  switch (memory_space_attr.getValue()) {
    case mlir::ktdf::MemorySpace::HBM:
      return ArchitectureGraph::Resource::HBM;
    case mlir::ktdf::MemorySpace::LX:
      return ArchitectureGraph::Resource::LX;
    default:
      return std::nullopt;
  }
}

/// Helper to infer resource from FIFO type
std::optional<ArchitectureGraph::Resource> inferResourceFromFifoType(
    mlir::ktdf::FifoType fifo_type, bool is_producer_side) {
  switch (fifo_type) {
    case mlir::ktdf::FifoType::HbmToSfp:
      return is_producer_side ? ArchitectureGraph::Resource::HBM
                              : ArchitectureGraph::Resource::SFP;
    case mlir::ktdf::FifoType::SfpToHbm:
      return is_producer_side ? ArchitectureGraph::Resource::SFP
                              : ArchitectureGraph::Resource::HBM;
    case mlir::ktdf::FifoType::LxluToSfp:
      return is_producer_side ? ArchitectureGraph::Resource::LX
                              : ArchitectureGraph::Resource::SFP;
    case mlir::ktdf::FifoType::SfpToLxsu:
      return is_producer_side ? ArchitectureGraph::Resource::SFP
                              : ArchitectureGraph::Resource::LX;
  }
  return std::nullopt;
}

//===----------------------------------------------------------------------===//
// Debug Output Helpers
//===----------------------------------------------------------------------===//

/// Print stage list with classifications
static void debugPrintStageList(llvm::raw_ostream& os,
                                llvm::ArrayRef<StageNode*> stages,
                                const PathExpansionPlan* plan,
                                llvm::StringRef step_name) {
  os << "\n=== " << step_name << " ===\n";
  os << "Total stages: " << stages.size() << "\n";
  for (StageNode* stage : stages) {
    os << "  Stage " << stage->getStageId() << ": ";

    // Check if this is an intermediate (synthetic) stage
    bool is_synthetic = (stage->getOperation() == nullptr);

    if (is_synthetic) {
      os << "(synthetic, unmaterialized)";
    } else {
      os << "(original)";
    }

    // Print materialization info if available
    if (plan && plan->stage_info.count(stage)) {
      const StageMaterializationInfo& info = plan->stage_info.at(stage);
      os << " kind=";
      switch (info.kind) {
        case StageMaterializationInfo::Kind::kPreserveOriginal:
          os << "PreserveOriginal";
          break;
        case StageMaterializationInfo::Kind::kAdaptFifoKinds:
          os << "AdaptFifoKinds";
          break;
        case StageMaterializationInfo::Kind::kAdaptTransfer:
          os << "AdaptTransfer";
          break;
        case StageMaterializationInfo::Kind::kSyntheticTransfer:
          os << "SyntheticTransfer";
          break;
      }
      os << ", transfers=" << info.transfers.size();
    }
    os << "\n";
  }
}

/// Print resource specs summary
static void debugPrintResourceSpecs(llvm::raw_ostream& os,
                                    const PrivateResourceFactory& factory) {
  os << "Private resources:\n";
  size_t mem_count = 0, fifo_count = 0;
  for (const auto& spec_ptr : factory.getSpecs()) {
    if (spec_ptr->kind == PrivateResourceSpec::Kind::kMemoryBuffer) {
      mem_count++;
    } else if (spec_ptr->kind == PrivateResourceSpec::Kind::kFifo) {
      fifo_count++;
    }
  }
  os << "  Memory buffers: " << mem_count << "\n";
  os << "  FIFOs: " << fifo_count << "\n";
}

/// Analyze a single stage and create its summary
mlir::FailureOr<StageSummary> analyzeStage(StageNode* stage_node) {
  StageSummary summary;
  summary.node = stage_node;

  auto stage_op =
      mlir::dyn_cast<mlir::ktdf::StageOp>(stage_node->getOperation());
  if (!stage_op) {
    return mlir::failure();
  }

  // Check for applicable_units
  if (auto applicable_units = stage_op.getApplicableUnits()) {
    llvm::ArrayRef<mlir::ktdf::ComponentType> units =
        applicable_units->getValue();
    assert(units.size() == 1 &&
           "path expansion currently does not handle nested pipelines with "
           "multi-unit stages");
    mlir::ktdf::ComponentType component = units[0];
    // Map component to resource
    if (component == mlir::ktdf::ComponentType::SFP) {
      summary.anchor_resource = ArchitectureGraph::Resource::SFP;
    } else if (component == mlir::ktdf::ComponentType::PE) {
      summary.anchor_resource = ArchitectureGraph::Resource::PE;
    } else {
      llvm_unreachable("unimplemented component type");
    }
  }

  // Analyze stage body
  bool has_compute = false;
  bool has_transfer = false;

  stage_op.walk([&](mlir::Operation* op) {
    if (mlir::isa<mlir::ktdf::DataTransferOp>(op)) {
      has_transfer = true;

      auto transfer_op = mlir::cast<mlir::ktdf::DataTransferOp>(op);

      // Analyze source endpoint
      if (auto source_type = transfer_op.getSource().getType()) {
        if (auto memref_type = mlir::dyn_cast<mlir::MemRefType>(source_type)) {
          if (auto resource =
                  inferResourceFromMemorySpace(memref_type.getMemorySpace())) {
            summary.input_endpoints.push_back({*resource, std::nullopt});
          }
        } else if (auto fifo_type =
                       mlir::dyn_cast<mlir::ktdf::FifoSlotType>(source_type)) {
          if (auto resource =
                  inferResourceFromFifoType(fifo_type.getFifoType(), true)) {
            summary.input_endpoints.push_back({*resource, std::nullopt});
          }
        }
      }

      // Analyze destination endpoint
      if (auto dest_type = transfer_op.getDestination().getType()) {
        if (auto memref_type = mlir::dyn_cast<mlir::MemRefType>(dest_type)) {
          if (auto resource =
                  inferResourceFromMemorySpace(memref_type.getMemorySpace())) {
            summary.output_endpoints.push_back({*resource, std::nullopt});
          }
        } else if (auto fifo_type =
                       mlir::dyn_cast<mlir::ktdf::FifoSlotType>(dest_type)) {
          if (auto resource =
                  inferResourceFromFifoType(fifo_type.getFifoType(), false)) {
            summary.output_endpoints.push_back({*resource, std::nullopt});
          }
        }
      }
    } else if (mlir::isa<mlir::linalg::LinalgOp>(op)) {
      // Consider linalg ops as compute
      has_compute = true;
    }
  });

  summary.is_transfer_only = has_transfer && !has_compute;
  summary.is_compute_containing = has_compute;

  return summary;
}

}  // anonymous namespace

//===----------------------------------------------------------------------===//
// Stage DAG Validation Functions
//===----------------------------------------------------------------------===//

mlir::LogicalResult validateLinearChain(
    llvm::ArrayRef<StageNode*> sorted_stages) {
  if (sorted_stages.empty()) {
    return mlir::success();  // Empty is trivially linear
  }

  // For a linear chain, each stage (except the last) should have exactly one
  // dependency and each stage (except the first) should be depended upon by
  // exactly one other stage

  // Count how many stages depend on each stage
  llvm::DenseMap<StageNode*, int> dependent_count;
  for (StageNode* stage : sorted_stages) {
    dependent_count[stage] = 0;
  }

  for (StageNode* stage : sorted_stages) {
    const llvm::SmallVector<StageNode*>& deps = stage->getDependencies();

    // Each stage should have at most one outgoing dependency
    if (deps.size() > 1) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Stage " << stage->getStageId()
                 << " has multiple outgoing dependencies (branching)\n");
      return mlir::failure();
    }

    // Count dependents
    for (StageNode* dep : deps) {
      dependent_count[dep]++;
    }
  }

  // Check that each stage (except source) has exactly one incoming dependency
  int source_count = 0;
  int sink_count = 0;

  for (StageNode* stage : sorted_stages) {
    int incoming = dependent_count[stage];
    const llvm::SmallVector<StageNode*>& outgoing = stage->getDependencies();

    if (incoming == 0) {
      source_count++;
    }
    if (outgoing.empty()) {
      sink_count++;
    }

    // In a linear chain, each non-source stage should have exactly 1 incoming
    // edge
    if (incoming > 1) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Stage " << stage->getStageId()
                 << " has multiple incoming dependencies (merging)\n");
      return mlir::failure();
    }
  }

  // Should have exactly one source and one sink
  if (source_count != 1 || sink_count != 1) {
    LLVM_DEBUG(llvm::dbgs()
               << "Expected 1 source and 1 sink, got " << source_count
               << " sources and " << sink_count << " sinks\n");
    return mlir::failure();
  }

  return mlir::success();
}

//===----------------------------------------------------------------------===//
// Stage Analysis Functions
//===----------------------------------------------------------------------===//

/// Overload that takes sorted stages to ensure path inference order matches DAG
/// order
mlir::FailureOr<llvm::SmallVector<StageSummary>> analyzeStages(
    llvm::ArrayRef<StageNode*> sorted_stages) {
  llvm::SmallVector<StageSummary> summaries;

  for (StageNode* stage : sorted_stages) {
    mlir::FailureOr<StageSummary> summary_or = analyzeStage(stage);
    if (mlir::failed(summary_or)) {
      return mlir::failure();
    }
    summaries.push_back(*summary_or);
  }

  return summaries;
}

mlir::FailureOr<llvm::SmallVector<ArchitectureGraph::Resource>>
inferOriginalPath(llvm::ArrayRef<StageSummary> summaries) {
  llvm::SmallVector<ArchitectureGraph::Resource> path;

  if (summaries.empty()) return mlir::failure();

  // Add anchor resources from compute stages
  for (const auto& summary : summaries) {
    if (summary.anchor_resource) {
      // Only add if different from last resource in path
      if (path.empty() || path.back() != *summary.anchor_resource) {
        path.push_back(*summary.anchor_resource);
      }
    } else if (!(summary.input_endpoints.empty() &&
                 summary.output_endpoints.empty())) {
      assert(summary.input_endpoints.size() == summary.output_endpoints.size());
      for (int i = 0; i < summary.input_endpoints.size(); i++) {
        if (path.empty() || path.back() != summary.input_endpoints[i].resource)
          path.push_back(summary.input_endpoints[i].resource);
      }
      for (int i = 0; i < summary.output_endpoints.size(); i++) {
        if (path.empty() || path.back() != summary.output_endpoints[i].resource)
          path.push_back(summary.output_endpoints[i].resource);
      }
    }
  }

  return path;
}

//===----------------------------------------------------------------------===//
// Step 1: Connection Legalization
//===----------------------------------------------------------------------===//

/// Insert intermediate stages for an illegal hop between two stages
static void insertIntermediateStages(
    PipelineTree& tree, PipelineNode* pipeline,
    ArchitectureGraph::Resource source_resource,
    ArchitectureGraph::Resource dest_resource,
    const ArchitectureGraph& arch_graph, PathExpansionPlan* plan,
    int& next_stage_id, StageNode*& last_inserted_stage) {
  // Find shortest path between resources
  ArchitectureGraph::NodeId source_node =
      arch_graph.getNodeIdForResource(source_resource);
  ArchitectureGraph::NodeId dest_node =
      arch_graph.getNodeIdForResource(dest_resource);

  std::optional<llvm::SmallVector<ArchitectureGraph::NodeId>> path =
      arch_graph.findShortestPath(source_node, dest_node);

  assert(path && "No path found between stages");
  assert(path->size() >= 3 && "Expected at least one intermediate node");

  LLVM_DEBUG(
      llvm::dbgs() << "  Illegal hop from "
                   << ArchitectureGraph::stringifyResource(source_resource)
                   << " to "
                   << ArchitectureGraph::stringifyResource(dest_resource)
                   << ", inserting " << (path->size() - 2)
                   << " intermediate stage(s)\n");

  // Insert intermediate stages for each intermediate node in the path
  // Skip first (source) and last (dest) nodes
  for (size_t i = 1; i < path->size() - 1; ++i) {
    ArchitectureGraph::NodeId intermediate_node = (*path)[i];
    std::optional<ArchitectureGraph::ResourceNode> intermediate_res_node =
        arch_graph.getNode(intermediate_node);

    assert(intermediate_res_node && "Intermediate node not found in graph");

    ArchitectureGraph::Resource intermediate_resource =
        intermediate_res_node->resource;

    // Create new intermediate stage with null operation (unmaterialized)
    StageNode* intermediate_stage =
        tree.createStageNode(nullptr, next_stage_id++);

    LLVM_DEBUG(llvm::dbgs()
               << "    Creating intermediate stage "
               << intermediate_stage->getStageId() << " at "
               << ArchitectureGraph::stringifyResource(intermediate_resource)
               << "\n");

    // Create minimal materialization info for the intermediate stage
    // Kind will be set to kSyntheticTransfer, transfers filled in Step 4
    StageMaterializationInfo intermediate_info;
    intermediate_info.kind = StageMaterializationInfo::Kind::kSyntheticTransfer;
    intermediate_info.stage_resource =
        intermediate_resource;  // Store resource for later steps

    plan->stage_info[intermediate_stage] = intermediate_info;

    // Insert the intermediate stage after the last inserted stage
    pipeline->insertChildNode(intermediate_stage, last_inserted_stage);
    last_inserted_stage = intermediate_stage;
  }
}

/// Step 1: Identify illegal connections and insert intermediate stages
static mlir::FailureOr<llvm::SmallVector<StageNode*>> legalizeStageConnections(
    PipelineTree& tree, PipelineNode* pipeline,
    llvm::ArrayRef<StageNode*> original_sorted_stages,
    llvm::ArrayRef<StageSummary> summaries, const ArchitectureGraph& arch_graph,
    PathExpansionPlan* plan, int& next_stage_id) {
  StageNode* last_inserted_stage = nullptr;

  LLVM_DEBUG(llvm::dbgs() << "Original stages: "
                          << original_sorted_stages.size() << "\n");

  for (size_t i = 0; i < original_sorted_stages.size(); ++i) {
    StageNode* current_stage = original_sorted_stages[i];
    const StageSummary& current_summary = summaries[i];

    // Determine the resource for this stage
    std::optional<ArchitectureGraph::Resource> current_resource;
    if (current_summary.anchor_resource) {
      current_resource = current_summary.anchor_resource;
    } else if (i == 0 && !current_summary.input_endpoints.empty()) {
      current_resource = current_summary.input_endpoints[0].resource;
    } else if (i == original_sorted_stages.size() - 1 &&
               !current_summary.output_endpoints.empty()) {
      current_resource = current_summary.output_endpoints[0].resource;
    }

    assert(current_resource && "Unable to determine resource for stage");

    // Update last_inserted_stage to current
    last_inserted_stage = current_stage;

    // Create minimal materialization info for original stage
    // Kind will be determined in later steps based on what needs adaptation
    StageMaterializationInfo info;
    info.kind =
        StageMaterializationInfo::Kind::kPreserveOriginal;  // Default, may be
                                                            // updated
    info.stage_resource = current_resource;  // Store resource for later steps

    // Set applicable_unit for compute stages
    if (current_summary.anchor_resource &&
        arch_graph.getResourceKind(*current_summary.anchor_resource) ==
            ArchitectureGraph::ResourceKind::Compute) {
      info.applicable_unit =
          (*current_summary.anchor_resource == ArchitectureGraph::Resource::SFP)
              ? mlir::ktdf::ComponentType::SFP
              : mlir::ktdf::ComponentType::PE;
    }

    plan->stage_info[current_stage] = info;

    LLVM_DEBUG(llvm::dbgs()
               << "Stage " << current_stage->getStageId() << " at "
               << ArchitectureGraph::stringifyResource(*current_resource)
               << "\n");

    // Check if we need to expand the path to the next stage
    if (i + 1 < original_sorted_stages.size()) {
      const StageSummary& next_summary = summaries[i + 1];

      // Determine the resource for the next stage
      std::optional<ArchitectureGraph::Resource> next_resource;
      if (next_summary.anchor_resource) {
        next_resource = next_summary.anchor_resource;
      } else if (!next_summary.output_endpoints.empty()) {
        next_resource = next_summary.output_endpoints[0].resource;
      }

      assert(next_resource && "Unable to determine resource for next stage");

      // Check if the hop is legal (direct edge exists)
      ArchitectureGraph::NodeId current_node =
          arch_graph.getNodeIdForResource(*current_resource);
      ArchitectureGraph::NodeId next_node =
          arch_graph.getNodeIdForResource(*next_resource);

      std::optional<ArchitectureGraph::EdgeInfo> direct_edge =
          arch_graph.getEdgeInfo(current_node, next_node);

      if (!direct_edge) {
        // Hop is not legal - insert intermediate stages
        insertIntermediateStages(tree, pipeline, *current_resource,
                                 *next_resource, arch_graph, plan,
                                 next_stage_id, last_inserted_stage);
      } else {
        LLVM_DEBUG(llvm::dbgs() << "  Legal hop to next stage\n");
      }
    }
  }

  // Rebuild dependencies: create linear chain through all stages
  llvm::SmallVector<StageNode*> all_stages = pipeline->getStages();

  // Clear existing dependencies
  for (StageNode* stage : all_stages) {
    stage->nullifyAllDependencies();
    stage->removeNullifiedDependencies();
  }

  // Get new topological sort to establish order
  mlir::FailureOr<llvm::SmallVector<StageNode*>> new_sorted_or =
      tree.topologicalSortStages(pipeline);
  if (mlir::failed(new_sorted_or)) {
    // If sort fails, manually create dependencies based on insertion order
    // This shouldn't happen but provides a fallback
    for (size_t i = 0; i < all_stages.size() - 1; ++i) {
      all_stages[i]->addDependency(all_stages[i + 1]);
    }
    new_sorted_or = tree.topologicalSortStages(pipeline);
    if (mlir::failed(new_sorted_or)) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Failed to topologically sort after expansion\n");
      return mlir::failure();
    }
  }

  llvm::SmallVector<StageNode*> sorted = *new_sorted_or;

  // Create linear chain dependencies
  for (size_t i = 0; i < sorted.size() - 1; ++i) {
    sorted[i]->addDependency(sorted[i + 1]);
    LLVM_DEBUG(llvm::dbgs()
               << "  Added dependency: Stage " << sorted[i]->getStageId()
               << " -> Stage " << sorted[i + 1]->getStageId() << "\n");
  }

  return sorted;
}

//===----------------------------------------------------------------------===//
// Step 2: Analyze Original Stage Transfers
//===----------------------------------------------------------------------===//

/// Step 2: Analyze data transfers in original stages that are
/// intermediate-stage facing, and for each data transfer create a
/// TransferMaterializationInfo in the StageMaterializationInfo::transfers list
/// with PrivateResourceSpec memory resource (according to the resource type of
/// the intermediate stage, eg LX). Assign this to source_private_resource or
/// dest_private_resource depending on whether the intermediate resource is at
/// the souce or sink of this stage.
static mlir::LogicalResult analyzeOriginalStageTransfers(
    llvm::ArrayRef<StageNode*> sorted_stages,
    const ArchitectureGraph& arch_graph, PathExpansionPlan* plan) {
  LLVM_DEBUG(llvm::dbgs() << "Analyzing " << sorted_stages.size()
                          << " stages\n");

  for (size_t i = 0; i < sorted_stages.size(); ++i) {
    StageNode* current_stage = sorted_stages[i];

    // Skip intermediate (synthetic) stages - they have null operations
    if (current_stage->getOperation() == nullptr) {
      continue;
    }

    // Get the stage's materialization info
    assert(plan->stage_info.count(current_stage) &&
           "Stage should have materialization info from Step 1");
    StageMaterializationInfo& stage_info = plan->stage_info[current_stage];

    // Get adjacent stages
    StageNode* prev_stage = (i > 0) ? sorted_stages[i - 1] : nullptr;
    StageNode* next_stage =
        (i + 1 < sorted_stages.size()) ? sorted_stages[i + 1] : nullptr;

    bool prev_is_intermediate =
        prev_stage && (prev_stage->getOperation() == nullptr);
    bool next_is_intermediate =
        next_stage && (next_stage->getOperation() == nullptr);

    // If this stage doesn't face any intermediate stages, skip it
    if (!prev_is_intermediate && !next_is_intermediate) {
      continue;
    }

    // Walk through data transfer operations
    mlir::ktdf::StageOp stage_op =
        mlir::cast<mlir::ktdf::StageOp>(current_stage->getOperation());

    stage_op.walk([&](mlir::ktdf::DataTransferOp transfer) {
      mlir::Type src_type = transfer.getSource().getType();
      mlir::Type dest_type = transfer.getDestination().getType();

      // Determine if this transfer involves an intermediate stage
      // Transfers to/from intermediate stages will have memref types on one
      // side
      bool src_is_memref = mlir::isa<mlir::MemRefType>(src_type);
      bool dest_is_memref = mlir::isa<mlir::MemRefType>(dest_type);

      // Skip if both are memrefs or neither (not facing intermediate)
      if (src_is_memref == dest_is_memref) {
        return mlir::WalkResult::advance();
      }

      // Determine which side is the intermediate memory resource
      ArchitectureGraph::Resource intermediate_resource;
      ArchitectureGraph::Resource current_resource = *stage_info.stage_resource;
      bool intermediate_is_source = false;

      mlir::MemRefType memref_type;

      if (src_is_memref) {
        // Source is memref (intermediate memory), dest is FIFO or other
        auto src_memref = mlir::cast<mlir::MemRefType>(src_type);
        memref_type = src_memref;
        intermediate_is_source = true;

        // Get intermediate resource from prev stage
        if (prev_is_intermediate) {
          assert(plan->stage_info.count(prev_stage) &&
                 "Intermediate stage should have info");
          intermediate_resource = *plan->stage_info[prev_stage].stage_resource;
        } else {
          return mlir::WalkResult::advance();
        }
      } else {
        // Dest is memref (intermediate memory), source is FIFO or other
        auto dest_memref = mlir::cast<mlir::MemRefType>(dest_type);
        memref_type = dest_memref;
        intermediate_is_source = false;

        // Get intermediate resource from next stage
        if (next_is_intermediate) {
          assert(plan->stage_info.count(next_stage) &&
                 "Intermediate stage should have info");
          intermediate_resource = *plan->stage_info[next_stage].stage_resource;
        } else {
          return mlir::WalkResult::advance();
        }
      }

      // Create memory buffer spec for the intermediate resource
      PrivateResourceSpec* buffer_spec =
          plan->resource_factory.createMemoryBuffer(
              intermediate_resource, memref_type.getShape(),
              memref_type.getElementType());

      // Get architecture edge info
      ArchitectureGraph::NodeId intermediate_node =
          arch_graph.getNodeIdForResource(intermediate_resource);
      ArchitectureGraph::NodeId current_node =
          arch_graph.getNodeIdForResource(current_resource);

      ArchitectureGraph::NodeId src_node =
          intermediate_is_source ? intermediate_node : current_node;
      ArchitectureGraph::NodeId dest_node =
          intermediate_is_source ? current_node : intermediate_node;

      std::optional<ArchitectureGraph::EdgeInfo> edge =
          arch_graph.getEdgeInfo(src_node, dest_node);

      assert(edge && "Edge should exist for legal hop");

      // Create transfer materialization info
      TransferMaterializationInfo transfer_info;
      transfer_info.template_op = transfer;
      transfer_info.hop = *edge;
      transfer_info.source_resource =
          intermediate_is_source ? intermediate_resource : current_resource;
      transfer_info.dest_resource =
          intermediate_is_source ? current_resource : intermediate_resource;

      // Link the buffer spec
      if (intermediate_is_source) {
        transfer_info.source_private_resource = buffer_spec;
        transfer_info.source_slot_index = 0;  // Memory buffers use slot 0
      } else {
        transfer_info.dest_private_resource = buffer_spec;
        transfer_info.dest_slot_index = 0;  // Memory buffers use slot 0
      }

      stage_info.transfers.push_back(transfer_info);

      // Update stage kind to kAdaptTransfer since we're modifying transfers
      if (stage_info.kind ==
          StageMaterializationInfo::Kind::kPreserveOriginal) {
        stage_info.kind = StageMaterializationInfo::Kind::kAdaptTransfer;
      }

      LLVM_DEBUG({
        llvm::dbgs() << "  Stage " << current_stage->getStageId()
                     << ": Created buffer spec for "
                     << ArchitectureGraph::stringifyResource(
                            intermediate_resource)
                     << " shape=[";
        for (size_t j = 0; j < buffer_spec->shape.size(); ++j) {
          llvm::dbgs() << buffer_spec->shape[j];
          if (j + 1 < buffer_spec->shape.size()) llvm::dbgs() << "x";
        }
        llvm::dbgs() << "], element=" << memref_type.getElementType() << "\n";
      });

      return mlir::WalkResult::advance();
    });
  }

  return mlir::success();
}

//===----------------------------------------------------------------------===//
// Step 3: Analyze FIFO Operations
//===----------------------------------------------------------------------===//

/// Step 3: Look for fifo_read/fifo_write operations in stages that are facing
/// intermediate stages. For each such operation, query the architecture graph
/// to determine the correct FIFO type based on the resources involved, create a
/// corresponding PrivateResourceSpec for the FIFO, and link it to a new
/// TransferMaterializationInfo object in the
/// StageMaterializationInfo::transfers for that stage.
static mlir::LogicalResult analyzeFifoOperations(
    llvm::ArrayRef<StageNode*> sorted_stages,
    const ArchitectureGraph& arch_graph, PathExpansionPlan* plan) {
  LLVM_DEBUG(llvm::dbgs() << "Analyzing FIFO operations in "
                          << sorted_stages.size() << " stages\n");

  // Track FIFO allocations by fifo_type to group slots
  // Map from fifo_type to the PrivateResourceSpec* created by factory
  llvm::DenseMap<mlir::ktdf::FifoType, PrivateResourceSpec*> fifo_specs;
  // Track next available slot index for each FIFO type
  llvm::DenseMap<mlir::ktdf::FifoType, size_t> fifo_next_slot;

  for (size_t i = 0; i < sorted_stages.size(); ++i) {
    StageNode* current_stage = sorted_stages[i];

    // Skip intermediate (synthetic) stages - they have null operations
    if (current_stage->getOperation() == nullptr) {
      continue;
    }

    // Get the stage's materialization info
    assert(plan->stage_info.count(current_stage) &&
           "Stage should have materialization info from Step 1");
    StageMaterializationInfo& stage_info = plan->stage_info[current_stage];

    // Get adjacent stages
    StageNode* prev_stage = (i > 0) ? sorted_stages[i - 1] : nullptr;
    StageNode* next_stage =
        (i + 1 < sorted_stages.size()) ? sorted_stages[i + 1] : nullptr;

    bool prev_is_intermediate =
        prev_stage && (prev_stage->getOperation() == nullptr);
    bool next_is_intermediate =
        next_stage && (next_stage->getOperation() == nullptr);

    // If this stage doesn't face any intermediate stages, skip it
    if (!prev_is_intermediate && !next_is_intermediate) {
      continue;
    }

    // Walk through FIFO read/write operations
    mlir::ktdf::StageOp stage_op =
        mlir::cast<mlir::ktdf::StageOp>(current_stage->getOperation());

    stage_op.walk([&](mlir::Operation* op) {
      // Check if this is a read_from_fifo or write_to_fifo operation
      auto read_op = mlir::dyn_cast<mlir::ktdf::ReadFromFifoOp>(op);
      auto write_op = mlir::dyn_cast<mlir::ktdf::WriteToFifoOp>(op);

      if (!read_op && !write_op) {
        return mlir::WalkResult::advance();
      }

      // Determine if this is a read or write, and get the fifo slot
      bool is_read = (read_op != nullptr);
      mlir::Value fifo_slot =
          is_read ? read_op.getFifoSlot() : write_op.getFifoSlot();
      auto fifo_slot_type =
          mlir::cast<mlir::ktdf::FifoSlotType>(fifo_slot.getType());

      // Determine which adjacent stage is involved
      // Reads consume from previous stage, writes produce to next stage
      StageNode* adjacent_stage = is_read ? prev_stage : next_stage;
      bool adjacent_is_intermediate =
          is_read ? prev_is_intermediate : next_is_intermediate;

      if (!adjacent_is_intermediate) {
        return mlir::WalkResult::advance();
      }

      // Get resources
      ArchitectureGraph::Resource current_resource = *stage_info.stage_resource;
      ArchitectureGraph::Resource adjacent_resource =
          *plan->stage_info[adjacent_stage].stage_resource;

      // Determine source and destination based on operation type
      ArchitectureGraph::Resource source_resource =
          is_read ? adjacent_resource : current_resource;
      ArchitectureGraph::Resource dest_resource =
          is_read ? current_resource : adjacent_resource;

      // Get edge info from architecture graph
      ArchitectureGraph::NodeId source_node =
          arch_graph.getNodeIdForResource(source_resource);
      ArchitectureGraph::NodeId dest_node =
          arch_graph.getNodeIdForResource(dest_resource);

      std::optional<ArchitectureGraph::EdgeInfo> edge =
          arch_graph.getEdgeInfo(source_node, dest_node);
      assert(edge && "Edge should exist for legal hop");
      assert(edge->fifo_type &&
             "Edge should have FIFO type for FIFO operation");

      mlir::ktdf::FifoType fifo_type = *edge->fifo_type;

      // Get element type and number of elements
      mlir::Type element_type = fifo_slot_type.getElementType();
      assert(!fifo_slot_type.isDynamicNumElements() &&
             "Dynamic FIFO sizes not supported in path expansion");
      int64_t num_elements = fifo_slot_type.getStaticNumElements();

      // Group FIFO allocations by fifo_type
      if (fifo_specs.find(fifo_type) == fifo_specs.end()) {
        // Create new FIFO spec via factory with first slot
        PrivateResourceSpec* spec = plan->resource_factory.createFifo(
            fifo_type, {num_elements},  // First slot
            element_type);
        fifo_specs[fifo_type] = spec;
        fifo_next_slot[fifo_type] = 0;  // Initialize slot counter
      } else {
        // Add another slot to existing FIFO spec
        PrivateResourceSpec* spec = fifo_specs[fifo_type];
        spec->elements_per_slot.push_back(num_elements);
      }

      // Assign and increment slot index for this FIFO type
      size_t slot_idx = fifo_next_slot[fifo_type]++;

      // Validate slot index assignment
      validateFifoSlotIndex(fifo_slot, slot_idx, fifo_specs[fifo_type]);

      // Create transfer materialization info
      TransferMaterializationInfo transfer_info;
      transfer_info.template_op = op;
      transfer_info.hop = *edge;
      transfer_info.source_resource = source_resource;
      transfer_info.dest_resource = dest_resource;

      // Link the FIFO spec to the appropriate side
      if (is_read) {
        transfer_info.source_private_resource = fifo_specs[fifo_type];
        transfer_info.source_slot_index = slot_idx;
      } else {
        transfer_info.dest_private_resource = fifo_specs[fifo_type];
        transfer_info.dest_slot_index = slot_idx;
      }

      stage_info.transfers.push_back(transfer_info);

      // Update stage kind to kAdaptFifoKinds since we're adapting FIFO
      // operations
      if (stage_info.kind ==
          StageMaterializationInfo::Kind::kPreserveOriginal) {
        stage_info.kind = StageMaterializationInfo::Kind::kAdaptFifoKinds;
      }

      LLVM_DEBUG({
        llvm::dbgs() << "  Stage " << current_stage->getStageId() << ": "
                     << (is_read ? "read_from_fifo" : "write_to_fifo")
                     << " - FIFO type "
                     << mlir::ktdf::stringifyFifoType(fifo_type) << ", slot "
                     << slot_idx << ", elements=" << num_elements << "\n";
      });

      return mlir::WalkResult::advance();
    });
  }

  return mlir::success();
}

//===----------------------------------------------------------------------===//
// Step 4: Populate Intermediate Stage Transfers
//===----------------------------------------------------------------------===//

/// Step 4: Walk through intermediate stages and for each one, look at the
/// surrounding stages' transfers. For each data transfer in a neighboring
/// stage, create a corresponding TransferMaterializationInfo for the
/// intermediate stage, reusing the source/dest private resources and slot
/// indexes from the corresponding transfer in the neighboring stage.
static mlir::LogicalResult populateIntermediateStageTransfers(
    llvm::ArrayRef<StageNode*> sorted_stages,
    const ArchitectureGraph& arch_graph, PathExpansionPlan* plan) {
  LLVM_DEBUG(llvm::dbgs() << "Populating intermediate stage transfers for "
                          << sorted_stages.size() << " stages\n");

  for (size_t i = 0; i < sorted_stages.size(); ++i) {
    StageNode* current_stage = sorted_stages[i];

    // Only process intermediate (synthetic) stages - they have null operations
    if (current_stage->getOperation() != nullptr) {
      continue;
    }

    // Get the stage's materialization info
    assert(plan->stage_info.count(current_stage) &&
           "Intermediate stage should have materialization info from Step 1");
    StageMaterializationInfo& stage_info = plan->stage_info[current_stage];

    // Get adjacent stages
    StageNode* prev_stage = (i > 0) ? sorted_stages[i - 1] : nullptr;
    StageNode* next_stage =
        (i + 1 < sorted_stages.size()) ? sorted_stages[i + 1] : nullptr;

    // Intermediate stages should have both prev and next stages
    assert(prev_stage && next_stage &&
           "Intermediate stage should have both neighbors");

    // Get the intermediate resource
    ArchitectureGraph::Resource intermediate_resource =
        *stage_info.stage_resource;

    LLVM_DEBUG(llvm::dbgs()
               << "  Processing intermediate stage "
               << current_stage->getStageId() << " (resource: "
               << ArchitectureGraph::stringifyResource(intermediate_resource)
               << ")\n");

    // Update stage kind to kSyntheticTransfer
    stage_info.kind = StageMaterializationInfo::Kind::kSyntheticTransfer;

    // Look at previous stage's transfers that have this intermediate stage as
    // destination
    if (plan->stage_info.count(prev_stage) == 0) continue;
    StageMaterializationInfo& prev_info = plan->stage_info[prev_stage];

    for (const TransferMaterializationInfo& prev_transfer :
         prev_info.transfers) {
      // Check if this transfer's destination is the intermediate resource
      if (prev_transfer.dest_resource != intermediate_resource) continue;
      // Create corresponding transfer for intermediate stage
      // This transfer goes FROM the intermediate resource TO the next stage

      // Get next stage's resource
      assert(plan->stage_info.count(next_stage) &&
             "Next stage should have info");
      ArchitectureGraph::Resource next_resource =
          *plan->stage_info[next_stage].stage_resource;

      // Get edge info for intermediate -> next hop
      ArchitectureGraph::NodeId intermediate_node =
          arch_graph.getNodeIdForResource(intermediate_resource);
      ArchitectureGraph::NodeId next_node =
          arch_graph.getNodeIdForResource(next_resource);

      std::optional<ArchitectureGraph::EdgeInfo> edge =
          arch_graph.getEdgeInfo(intermediate_node, next_node);

      assert(edge && "Edge should exist for legal hop");

      // Create transfer info for intermediate stage
      TransferMaterializationInfo transfer_info;
      transfer_info.template_op = nullptr;  // Synthetic transfer, no template
      transfer_info.hop = *edge;
      transfer_info.source_resource = intermediate_resource;
      transfer_info.dest_resource = next_resource;

      // Reuse the private resource from prev_transfer's destination as our
      // source
      transfer_info.source_private_resource =
          prev_transfer.dest_private_resource;
      transfer_info.source_slot_index = prev_transfer.dest_slot_index;

      // Look for matching transfer in next stage to get destination resource
      // Find a transfer in next_stage that has intermediate_resource as source
      const PrivateResourceSpec* dest_resource_spec = nullptr;
      size_t dest_slot_idx = 0;

      if (plan->stage_info.count(next_stage)) {
        StageMaterializationInfo& next_info = plan->stage_info[next_stage];
        for (const TransferMaterializationInfo& next_transfer :
             next_info.transfers) {
          if (next_transfer.source_resource == intermediate_resource &&
              next_transfer.source_private_resource ==
                  transfer_info.source_private_resource) {
            // Found matching transfer - use its destination
            dest_resource_spec = next_transfer.dest_private_resource;
            dest_slot_idx = next_transfer.dest_slot_index;
            break;
          }
        }
      }

      transfer_info.dest_private_resource = dest_resource_spec;
      transfer_info.dest_slot_index = dest_slot_idx;

      stage_info.transfers.push_back(transfer_info);

      LLVM_DEBUG({
        llvm::dbgs() << "    Created transfer: "
                     << ArchitectureGraph::stringifyResource(
                            intermediate_resource)
                     << " -> "
                     << ArchitectureGraph::stringifyResource(next_resource);
        if (transfer_info.source_private_resource) {
          llvm::dbgs() << " (source: ";
          if (transfer_info.source_private_resource->kind ==
              PrivateResourceSpec::Kind::kFifo) {
            llvm::dbgs() << "FIFO slot " << transfer_info.source_slot_index;
          } else {
            llvm::dbgs() << "buffer";
          }
          llvm::dbgs() << ")";
        }
        if (transfer_info.dest_private_resource) {
          llvm::dbgs() << " (dest: ";
          if (transfer_info.dest_private_resource->kind ==
              PrivateResourceSpec::Kind::kFifo) {
            llvm::dbgs() << "FIFO slot " << transfer_info.dest_slot_index;
          } else {
            llvm::dbgs() << "buffer";
          }
          llvm::dbgs() << ")";
        }
        llvm::dbgs() << "\n";
      });
    }
  }

  return mlir::success();
}

//===----------------------------------------------------------------------===//
// Main Planning Function
//===----------------------------------------------------------------------===//

std::unique_ptr<PathExpansionPlan> planPathExpansion(
    PipelineTree& tree, PipelineNode* pipeline,
    const ArchitectureGraph& arch_graph) {
  auto plan = std::make_unique<PathExpansionPlan>();
  plan->changed = false;

  // Prep Step 1: Perform topological sort
  mlir::FailureOr<llvm::SmallVector<StageNode*>> sorted_stages_or =
      tree.topologicalSortStages(pipeline);
  if (mlir::failed(sorted_stages_or)) {
    LLVM_DEBUG(llvm::dbgs()
               << "Failed to perform topological sort (cycle detected)\n");
    return nullptr;
  }
  llvm::SmallVector<StageNode*> sorted_stages = *sorted_stages_or;

  // For now we only perform path expansion when we have linear chain of
  // dependencies.
  if (mlir::failed(validateLinearChain(sorted_stages))) {
    LLVM_DEBUG(llvm::dbgs() << "Stage topology is not a linear chain\n");
    return nullptr;
  }

  // Prep Step 2: Analyze stages in topological order
  mlir::FailureOr<llvm::SmallVector<StageSummary>> summaries_or =
      analyzeStages(sorted_stages);
  if (mlir::failed(summaries_or)) {
    return nullptr;
  }
  llvm::SmallVector<StageSummary> summaries = *summaries_or;

  // Prep Step 3: Infer original path
  // todo: do we still need this prep step?
  mlir::FailureOr<llvm::SmallVector<ArchitectureGraph::Resource>>
      original_path_or = inferOriginalPath(summaries);
  if (mlir::failed(original_path_or)) {
    return nullptr;
  }
  llvm::SmallVector<ArchitectureGraph::Resource> original_path =
      *original_path_or;

  if (original_path.size() < 2) {
    // Need at least source and destination
    return nullptr;
  }

  LLVM_DEBUG({
    llvm::dbgs() << "Original PipelineTree before expansion:\n";
    pipeline->print(llvm::dbgs());
    llvm::dbgs() << "\nOriginal topological order: ";
    for (StageNode* stage : sorted_stages) {
      llvm::dbgs() << "Stage " << stage->getStageId() << " ";
    }
    llvm::dbgs() << "\n\n";
  });

  // Prep Step 4: Resource-centric path expansion
  // Key insight: Create one stage per resource node in the expanded path
  // - Original stages get adapted/preserved at their resource positions
  // - New stages get created for newly introduced intermediate resources

  // First, build the full shortest path by querying each segment
  // todo: remove this step, the new algorithm is idempotent by design, or
  // perhaps guard this under debug (could be useful for debugging purposes
  // only)
  llvm::SmallVector<ArchitectureGraph::Resource> full_shortest_path;

  for (size_t i = 0; i < original_path.size() - 1; ++i) {
    ArchitectureGraph::Resource seg_source = original_path[i];
    ArchitectureGraph::Resource seg_dest = original_path[i + 1];

    ArchitectureGraph::NodeId source_node =
        arch_graph.getNodeIdForResource(seg_source);
    ArchitectureGraph::NodeId dest_node =
        arch_graph.getNodeIdForResource(seg_dest);

    std::optional<llvm::SmallVector<ArchitectureGraph::NodeId>>
        segment_path_or = arch_graph.findShortestPath(source_node, dest_node);
    if (!segment_path_or) {
      LLVM_DEBUG(llvm::dbgs()
                 << "No legal path found from "
                 << ArchitectureGraph::stringifyResource(seg_source) << " to "
                 << ArchitectureGraph::stringifyResource(seg_dest) << "\n");
      return nullptr;
    }

    // Convert node IDs to resources and add to full path
    for (size_t j = 0; j < segment_path_or->size(); ++j) {
      ArchitectureGraph::NodeId node_id = (*segment_path_or)[j];
      std::optional<ArchitectureGraph::ResourceNode> node_or =
          arch_graph.getNode(node_id);
      if (!node_or) return nullptr;

      // Avoid duplicating connection points between segments
      if (j == 0 && !full_shortest_path.empty() &&
          full_shortest_path.back() == node_or->resource) {
        continue;
      }
      full_shortest_path.push_back(node_or->resource);
    }
  }

  LLVM_DEBUG({
    llvm::dbgs() << "Original path: ";
    for (auto res : original_path) {
      llvm::dbgs() << ArchitectureGraph::stringifyResource(res) << " -> ";
    }
    llvm::dbgs() << "\nFull shortest path: ";
    for (auto res : full_shortest_path) {
      llvm::dbgs() << ArchitectureGraph::stringifyResource(res) << " -> ";
    }
    llvm::dbgs() << "\n";
  });

  // Check if expansion is needed
  if (original_path == full_shortest_path) {
    plan->changed = false;
    LLVM_DEBUG(llvm::dbgs() << "Pipeline already legal\n");
    return plan;
  }

  plan->changed = true;
  plan->modified_tree = &tree;

  // Progressive 4-step algorithm to build complete plan
  int next_stage_id = sorted_stages.size();

  // STEP 1: Legalize connections and insert intermediate stages
  LLVM_DEBUG(llvm::dbgs() << "\n=== Step 1: Connection Legalization ===\n");
  auto expanded_stages_or =
      legalizeStageConnections(tree, pipeline, sorted_stages, summaries,
                               arch_graph, plan.get(), next_stage_id);
  if (mlir::failed(expanded_stages_or)) {
    return nullptr;
  }
  llvm::SmallVector<StageNode*> expanded_stages = *expanded_stages_or;
  LLVM_DEBUG(debugPrintStageList(llvm::dbgs(), expanded_stages, plan.get(),
                                 "After Step 1"));

  // STEP 2: Analyze original stage transfers
  LLVM_DEBUG(
      llvm::dbgs() << "\n=== Step 2: Original Stage Transfer Analysis ===\n");
  if (mlir::failed(analyzeOriginalStageTransfers(expanded_stages, arch_graph,
                                                 plan.get()))) {
    return nullptr;
  }
  LLVM_DEBUG(debugPrintStageList(llvm::dbgs(), expanded_stages, plan.get(),
                                 "After Step 2"));

  // STEP 3: Analyze FIFO operations
  LLVM_DEBUG(llvm::dbgs() << "\n=== Step 3: FIFO Operation Analysis ===\n");
  if (mlir::failed(
          analyzeFifoOperations(expanded_stages, arch_graph, plan.get()))) {
    return nullptr;
  }
  LLVM_DEBUG(debugPrintStageList(llvm::dbgs(), expanded_stages, plan.get(),
                                 "After Step 3"));

  // STEP 4: Populate intermediate stage transfers
  LLVM_DEBUG(llvm::dbgs()
             << "\n=== Step 4: Intermediate Stage Transfer Population ===\n");
  if (mlir::failed(populateIntermediateStageTransfers(
          expanded_stages, arch_graph, plan.get()))) {
    return nullptr;
  }
  LLVM_DEBUG(debugPrintStageList(llvm::dbgs(), expanded_stages, plan.get(),
                                 "After Step 4"));

  // Final summary
  LLVM_DEBUG({
    llvm::dbgs() << "\n=== Planning Complete ===\n";
    llvm::dbgs() << "Total stages: " << expanded_stages.size() << "\n";
    llvm::dbgs() << "Total private resources: "
                 << plan->resource_factory.getSpecs().size() << "\n";
    debugPrintResourceSpecs(llvm::dbgs(), plan->resource_factory);
  });

  return plan;
}

}  // namespace scheduler

// Made with Bob