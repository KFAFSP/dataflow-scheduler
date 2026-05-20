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
                  inferResourceFromFifoType(fifo_type.getFifoType(), false)) {
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
                  inferResourceFromFifoType(fifo_type.getFifoType(), true)) {
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

std::unique_ptr<PathExpansionPlan> planPathExpansion(
    PipelineTree& tree, PipelineNode* pipeline,
    const ArchitectureGraph& arch_graph) {
  auto plan = std::make_unique<PathExpansionPlan>();
  plan->changed = false;

  // Step 1: Perform topological sort
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

  // Step 2: Analyze stages in topological order
  mlir::FailureOr<llvm::SmallVector<StageSummary>> summaries_or =
      analyzeStages(sorted_stages);
  if (mlir::failed(summaries_or)) {
    return nullptr;
  }
  llvm::SmallVector<StageSummary> summaries = *summaries_or;

  // Step 3: Infer original path
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

  // Step 4-12: Resource-centric path expansion
  // Key insight: Create one stage per resource node in the expanded path
  // - Original stages get adapted/preserved at their resource positions
  // - New stages get created for newly introduced intermediate resources

  // First, build the full shortest path by querying each segment
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

  // Process each segment of the original path and build the stage sequence
  // Key insight: Walk through original stages sequentially, and for each
  // segment:
  // - Reuse original stages where they belong
  // - Insert new stages for newly introduced intermediate resources

  llvm::SmallVector<StageNode*> stage_sequence;
  llvm::DenseMap<ArchitectureGraph::Resource, PrivateResourceSpec>
      resource_specs;
  int next_stage_id = sorted_stages.size();
  size_t original_stage_idx = 0;
  size_t path_position = 0;  // Current position in full_shortest_path
  StageNode* last_inserted_stage =
      nullptr;  // Track last inserted stage for ordering

  for (size_t seg_idx = 0; seg_idx < original_path.size() - 1; ++seg_idx) {
    ArchitectureGraph::Resource seg_source = original_path[seg_idx];
    ArchitectureGraph::Resource seg_dest = original_path[seg_idx + 1];

    // Find this segment in the full shortest path
    size_t seg_start_pos = path_position;
    size_t seg_end_pos = seg_start_pos;
    while (seg_end_pos < full_shortest_path.size() &&
           full_shortest_path[seg_end_pos] != seg_dest) {
      seg_end_pos++;
    }

    LLVM_DEBUG(llvm::dbgs()
               << "\nSegment " << seg_idx << ": "
               << ArchitectureGraph::stringifyResource(seg_source) << " -> "
               << ArchitectureGraph::stringifyResource(seg_dest)
               << " (positions " << seg_start_pos << " to " << seg_end_pos
               << " in shortest path)\n");

    // Check if segment needs expansion
    bool needs_expansion = (seg_end_pos - seg_start_pos > 1);

    if (!needs_expansion) {
      // Segment is already legal - reuse original stage
      if (original_stage_idx < summaries.size()) {
        StageNode* original_stage = summaries[original_stage_idx].node;
        stage_sequence.push_back(original_stage);
        last_inserted_stage = original_stage;  // Track for insertion order
        LLVM_DEBUG(llvm::dbgs() << "  Keeping original stage "
                                << original_stage->getStageId() << "\n");

        // Create materialization info
        StageMaterializationInfo info;
        const StageSummary& summary = summaries[original_stage_idx];
        if (summary.anchor_resource &&
            arch_graph.getResourceKind(*summary.anchor_resource) ==
                ArchitectureGraph::ResourceKind::Compute) {
          info.kind = StageMaterializationInfo::Kind::kAdaptFifoKinds;
          info.applicable_unit =
              (*summary.anchor_resource == ArchitectureGraph::Resource::SFP)
                  ? mlir::ktdf::ComponentType::SFP
                  : mlir::ktdf::ComponentType::PE;
        } else {
          info.kind = StageMaterializationInfo::Kind::kAdaptFifoKinds;
        }
        plan->stage_info[original_stage] = info;
        original_stage_idx++;
      }
    } else {
      // Segment needs expansion
      LLVM_DEBUG(llvm::dbgs()
                 << "  Expanding segment with "
                 << (seg_end_pos - seg_start_pos + 1) << " resources\n");

      // Process each position in the expanded segment
      for (size_t pos = seg_start_pos; pos <= seg_end_pos; ++pos) {
        ArchitectureGraph::Resource res = full_shortest_path[pos];

        // Skip if this is a segment boundary we already processed
        if (pos == seg_start_pos && seg_idx > 0 && !stage_sequence.empty()) {
          // This is the start of a new segment, but it's the same resource as
          // the end of the previous segment - skip it
          LLVM_DEBUG(llvm::dbgs()
                     << "    Skipping "
                     << ArchitectureGraph::stringifyResource(res)
                     << " (already handled as previous segment end)\n");
          continue;
        }

        StageNode* stage = nullptr;
        bool reused_original = false;

        // Check if we should reuse an original stage at this position
        if (original_stage_idx < summaries.size()) {
          const StageSummary& summary = summaries[original_stage_idx];

          // Reuse if this is a compute anchor at this resource
          if (summary.anchor_resource && *summary.anchor_resource == res) {
            stage = summary.node;
            reused_original = true;
            original_stage_idx++;
            last_inserted_stage = stage;  // Track for insertion order

            LLVM_DEBUG(llvm::dbgs()
                       << "    Reusing original stage " << stage->getStageId()
                       << " at " << ArchitectureGraph::stringifyResource(res)
                       << "\n");

            StageMaterializationInfo info;
            info.kind = StageMaterializationInfo::Kind::kAdaptFifoKinds;
            info.applicable_unit = (res == ArchitectureGraph::Resource::SFP)
                                       ? mlir::ktdf::ComponentType::SFP
                                       : mlir::ktdf::ComponentType::PE;
            plan->stage_info[stage] = info;
          }
          // Reuse if this is the start or end of the segment and matches
          // original path
          else if ((pos == seg_start_pos && res == seg_source) ||
                   (pos == seg_end_pos && res == seg_dest)) {
            // This is a transfer stage at segment boundary
            stage = summary.node;
            reused_original = true;
            original_stage_idx++;
            last_inserted_stage = stage;  // Track for insertion order

            LLVM_DEBUG(llvm::dbgs()
                       << "    Reusing original transfer stage "
                       << stage->getStageId() << " at "
                       << ArchitectureGraph::stringifyResource(res) << "\n");

            StageMaterializationInfo info;
            info.kind = StageMaterializationInfo::Kind::kAdaptFifoKinds;
            plan->stage_info[stage] = info;
          }
        }

        if (!reused_original) {
          // Create new stage for intermediate resource
          stage = tree.createStageNode(nullptr, next_stage_id++);

          LLVM_DEBUG(llvm::dbgs()
                     << "    Creating new stage " << stage->getStageId()
                     << " at " << ArchitectureGraph::stringifyResource(res)
                     << "\n");

          StageMaterializationInfo info;
          info.kind = StageMaterializationInfo::Kind::kSyntheticTransfer;

          // Add transfer for incoming edge
          if (pos > seg_start_pos ||
              (pos == seg_start_pos && !stage_sequence.empty())) {
            size_t src_pos = (pos > 0) ? pos - 1 : 0;
            ArchitectureGraph::Resource src_res = full_shortest_path[src_pos];
            ArchitectureGraph::NodeId src_node =
                arch_graph.getNodeIdForResource(src_res);
            ArchitectureGraph::NodeId dst_node =
                arch_graph.getNodeIdForResource(res);
            std::optional<ArchitectureGraph::EdgeInfo> edge_info =
                arch_graph.getEdgeInfo(src_node, dst_node);
            if (edge_info) {
              TransferMaterializationInfo transfer_info;
              transfer_info.template_op = nullptr;
              transfer_info.hop = *edge_info;
              transfer_info.source_resource = src_res;
              transfer_info.dest_resource = res;
              info.transfers.push_back(transfer_info);
            }
          }

          plan->stage_info[stage] = info;

          // Insert stage in correct position (after last inserted stage)
          pipeline->insertChildNode(stage, last_inserted_stage);
          last_inserted_stage = stage;  // Track for next insertion

          // Track intermediate memory resources
          if (pos > seg_start_pos && pos < seg_end_pos &&
              arch_graph.getResourceKind(res) ==
                  ArchitectureGraph::ResourceKind::Memory) {
            if (!summaries.empty()) {
              mlir::ktdf::StageOp stage_op = mlir::cast<mlir::ktdf::StageOp>(
                  summaries[0].node->getOperation());
              stage_op.walk([&](mlir::ktdf::DataTransferOp transfer) {
                mlir::Type src_type = transfer.getSource().getType();
                if (mlir::MemRefType memref_type =
                        mlir::dyn_cast<mlir::MemRefType>(src_type)) {
                  if (!resource_specs.contains(res)) {
                    PrivateResourceSpec spec;
                    spec.resource = res;
                    spec.element_type = memref_type.getElementType();
                    spec.shape.assign(memref_type.getShape().begin(),
                                      memref_type.getShape().end());
                    resource_specs[res] = spec;
                  }
                }
                return mlir::WalkResult::advance();
              });
            }
          }
        }

        stage_sequence.push_back(stage);
      }
    }

    // Move to next segment's starting position
    path_position = seg_end_pos;
  }

  // Build dependencies between stages in sequence
  for (StageNode* stage : pipeline->getStages()) {
    stage->nullifyAllDependencies();
    stage->removeNullifiedDependencies();
  }

  for (size_t i = 0; i < stage_sequence.size() - 1; ++i) {
    StageNode* current = stage_sequence[i];
    StageNode* next = stage_sequence[i + 1];
    if (current && next && current != next) {
      current->addDependency(next);
      LLVM_DEBUG(llvm::dbgs()
                 << "  Added dependency: Stage " << current->getStageId()
                 << " -> Stage " << next->getStageId() << "\n");
    }
  }

  // Rebuild the sorted stages list
  mlir::FailureOr<llvm::SmallVector<StageNode*>> new_sorted_or =
      tree.topologicalSortStages(pipeline);
  if (mlir::failed(new_sorted_or)) {
    LLVM_DEBUG(llvm::dbgs()
               << "Failed to topologically sort after expansion\n");
    return nullptr;
  }

  LLVM_DEBUG({
    llvm::dbgs() << "Modified PipelineTree after expansion:\n";
    pipeline->print(llvm::dbgs());
    llvm::dbgs() << "\nTopological order: ";
    for (StageNode* stage : *new_sorted_or) {
      llvm::dbgs() << "Stage " << stage->getStageId() << " ";
    }
    llvm::dbgs() << "\n\n";
  });

  // Record new private resources
  for (const auto& [res, spec] : resource_specs) {
    plan->new_resources.push_back(spec);
  }

  // Count synthetic stages
  size_t synthetic_stage_count = 0;
  for (const auto& [stage, info] : plan->stage_info) {
    if (info.kind == StageMaterializationInfo::Kind::kSyntheticTransfer) {
      synthetic_stage_count++;
    }
  }

  LLVM_DEBUG(llvm::dbgs() << "Plan created with " << plan->stage_info.size()
                          << " stage entries (including "
                          << synthetic_stage_count << " synthetic stages) and "
                          << plan->new_resources.size() << " new resources\n");

  return plan;
}

}  // namespace scheduler

// Made with Bob