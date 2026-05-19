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

  // Step 4: Query architecture graph for shortest path
  ArchitectureGraph::Resource source_resource = original_path.front();
  ArchitectureGraph::Resource dest_resource = original_path.back();

  ArchitectureGraph::NodeId source_node =
      arch_graph.getNodeIdForResource(source_resource);
  ArchitectureGraph::NodeId dest_node =
      arch_graph.getNodeIdForResource(dest_resource);

  std::optional<llvm::SmallVector<ArchitectureGraph::NodeId>> shortest_path_or =
      arch_graph.findShortestPath(source_node, dest_node);
  if (!shortest_path_or) {
    // No legal path exists
    LLVM_DEBUG(llvm::dbgs() << "No legal path found in architecture graph\n");
    return nullptr;
  }

  // Step 5: Convert node IDs back to resources
  llvm::SmallVector<ArchitectureGraph::Resource> shortest_path;
  for (ArchitectureGraph::NodeId node_id : *shortest_path_or) {
    std::optional<ArchitectureGraph::ResourceNode> node_or =
        arch_graph.getNode(node_id);
    if (!node_or) return nullptr;
    shortest_path.push_back(node_or->resource);
  }

  LLVM_DEBUG({
    llvm::dbgs() << "Original path: ";
    for (ArchitectureGraph::Resource res : original_path) {
      llvm::dbgs() << ArchitectureGraph::stringifyResource(res) << " -> ";
    }
    llvm::dbgs() << "\nShortest path: ";
    for (ArchitectureGraph::Resource res : shortest_path) {
      llvm::dbgs() << ArchitectureGraph::stringifyResource(res) << " -> ";
    }
    llvm::dbgs() << "\n";
  });

  // Step 6: Compare paths
  if (original_path == shortest_path) {
    // Already legal, no changes needed
    plan->changed = false;
    LLVM_DEBUG(llvm::dbgs() << "Pipeline already legal\n");
    return plan;
  }

  // Step 7: Path needs expansion - modify the PipelineTree
  plan->changed = true;
  plan->modified_tree = tree;

  LLVM_DEBUG(llvm::dbgs() << "Planning path expansion for "
                          << sorted_stages.size() << " stages, expanding to "
                          << shortest_path.size() << " hops\n");

  // Step 8: Identify intermediate memory resources that need buffers
  llvm::DenseSet<ArchitectureGraph::Resource> intermediate_resources;
  for (size_t i = 1; i < shortest_path.size() - 1; i++) {
    ArchitectureGraph::Resource res = shortest_path[i];
    if (arch_graph.getResourceKind(res) ==
        ArchitectureGraph::ResourceKind::Memory) {
      intermediate_resources.insert(res);
    }
  }

  // Step 9: Build stage-to-summary map for quick lookup
  llvm::DenseMap<StageNode*, StageSummary> stage_to_summary;
  for (const StageSummary& summary : summaries) {
    stage_to_summary[summary.node] = summary;
  }

  // Step 10: Modify PipelineTree by inserting new stages for intermediate hops

  LLVM_DEBUG({
    llvm::dbgs() << "Original PipelineTree before expansion:\n";
    pipeline->print(llvm::dbgs());
    llvm::dbgs() << "\nOriginal topological order: ";
    for (StageNode* stage : sorted_stages) {
      llvm::dbgs() << "Stage " << stage->getStageId() << " ";
    }
    llvm::dbgs() << "\n\n";
  });

  // Map each hop in shortest path to the stage that covers it (if any)
  // Also collect buffer shape information for intermediate resources
  llvm::SmallVector<StageNode*> hop_to_stage(shortest_path.size() - 1, nullptr);
  llvm::DenseMap<ArchitectureGraph::Resource, PrivateResourceSpec>
      resource_specs;

  for (size_t hop_idx = 0; hop_idx < shortest_path.size() - 1; hop_idx++) {
    ArchitectureGraph::Resource src_res = shortest_path[hop_idx];
    ArchitectureGraph::Resource dst_res = shortest_path[hop_idx + 1];

    // Find which existing stage covers this hop
    for (StageNode* stage : sorted_stages) {
      const StageSummary& summary = stage_to_summary[stage];

      // Check if this stage's transfers cover this hop
      bool covers_hop = false;
      for (const StageEndpoint& input_ep : summary.input_endpoints) {
        for (const StageEndpoint& output_ep : summary.output_endpoints) {
          if (input_ep.resource == src_res && output_ep.resource == dst_res) {
            covers_hop = true;
            break;
          }
        }
        if (covers_hop) break;
      }

      if (covers_hop) {
        hop_to_stage[hop_idx] = stage;
        if (!(intermediate_resources.contains(src_res) ||
              intermediate_resources.contains(dst_res)))
          break;
        // If this hop involves an intermediate memory resource, extract buffer
        // shape
        mlir::ktdf::StageOp stage_op =
            mlir::cast<mlir::ktdf::StageOp>(stage->getOperation());

        stage_op.walk([&](mlir::ktdf::DataTransferOp transfer) {
          mlir::Type src_type = transfer.getSource().getType();
          mlir::Type dst_type = transfer.getDestination().getType();

          // Extract shape from memory resource (memref)
          auto extract_from_memref = [&](mlir::Type type,
                                         ArchitectureGraph::Resource res) {
            if (mlir::MemRefType memref_type =
                    mlir::dyn_cast<mlir::MemRefType>(type)) {
              if (!resource_specs.contains(res)) {
                PrivateResourceSpec spec;
                spec.resource = res;
                spec.element_type = memref_type.getElementType();
                spec.shape.assign(memref_type.getShape().begin(),
                                  memref_type.getShape().end());
                resource_specs[res] = spec;
              }
            }
          };

          // Check if source or destination is an intermediate memory resource
          if (intermediate_resources.contains(src_res)) {
            extract_from_memref(src_type, src_res);
          }
          if (intermediate_resources.contains(dst_res)) {
            extract_from_memref(dst_type, dst_res);
          }

          return mlir::WalkResult::advance();
        });
        break;
      }
    }
  }

  // Create synthetic stages for uncovered hops
  int next_stage_id = sorted_stages.size();
  StageNode* insert_point = nullptr;
  for (size_t hop_idx = 0; hop_idx < hop_to_stage.size(); hop_idx++) {
    if (hop_to_stage[hop_idx] != nullptr) {
      insert_point = hop_to_stage[hop_idx];
      continue;
    }

    ArchitectureGraph::Resource src_res = shortest_path[hop_idx];
    ArchitectureGraph::Resource dst_res = shortest_path[hop_idx + 1];

    LLVM_DEBUG(llvm::dbgs()
               << "  Inserting synthetic stage for hop "
               << ArchitectureGraph::stringifyResource(src_res) << " -> "
               << ArchitectureGraph::stringifyResource(dst_res) << "\n");

    // Create unmaterialized stage node (no operation yet)
    StageNode* new_stage = tree.createStageNode(nullptr, next_stage_id++);
    hop_to_stage[hop_idx] = new_stage;

    // Create materialization info for this synthetic stage
    StageMaterializationInfo info;
    info.kind = StageMaterializationInfo::Kind::kSyntheticTransfer;

    // Get edge info for this hop
    ArchitectureGraph::NodeId src_node =
        arch_graph.getNodeIdForResource(src_res);
    ArchitectureGraph::NodeId dst_node =
        arch_graph.getNodeIdForResource(dst_res);
    std::optional<ArchitectureGraph::EdgeInfo> edge_info =
        arch_graph.getEdgeInfo(src_node, dst_node);

    if (edge_info) {
      TransferMaterializationInfo transfer_info;
      transfer_info.template_op = nullptr;
      transfer_info.hop = *edge_info;
      transfer_info.source_resource = src_res;
      transfer_info.dest_resource = dst_res;
      info.transfers.push_back(transfer_info);
    }

    plan->stage_info[new_stage] = info;

    // Add to pipeline as child
    pipeline->insertChildNode(new_stage, insert_point);
  }

  // Step 11: Build linear chain dependencies based on hop sequence
  // Clear existing dependencies and rebuild based on shortest path order
  for (StageNode* stage : pipeline->getStages()) {
    // Clear all dependencies
    while (!stage->getDependencies().empty()) {
      stage->nullifyDependency(stage->getDependencies().front());
    }
    stage->removeNullifiedDependencies();
  }

  // Add dependencies to form linear chain following shortest path
  for (size_t hop_idx = 0; hop_idx < hop_to_stage.size() - 1; hop_idx++) {
    StageNode* current_stage = hop_to_stage[hop_idx];
    StageNode* next_stage = hop_to_stage[hop_idx + 1];

    if (current_stage && next_stage && current_stage != next_stage) {
      current_stage->addDependency(next_stage);
      LLVM_DEBUG(llvm::dbgs()
                 << "  Added dependency: Stage " << current_stage->getStageId()
                 << " -> Stage " << next_stage->getStageId() << "\n");
    }
  }

  // Rebuild the sorted stages list with new topology
  mlir::FailureOr<llvm::SmallVector<StageNode*>> new_sorted_or =
      tree.topologicalSortStages(pipeline);
  if (mlir::failed(new_sorted_or)) {
    LLVM_DEBUG(
        llvm::dbgs()
        << "Failed to topologically sort after adding synthetic stages\n");
    return nullptr;
  }
  sorted_stages = *new_sorted_or;

  LLVM_DEBUG({
    llvm::dbgs() << "Modified PipelineTree after expansion:\n";
    pipeline->print(llvm::dbgs());
    llvm::dbgs() << "\nTopological order: ";
    for (StageNode* stage : sorted_stages) {
      llvm::dbgs() << "Stage " << stage->getStageId() << " ";
    }
    llvm::dbgs() << "\n\n";
  });

  // Step 12: Create materialization info for original stages
  for (StageNode* stage : sorted_stages) {
    // Skip if already processed (synthetic stages)
    if (plan->stage_info.count(stage)) continue;

    const StageSummary& summary = stage_to_summary[stage];
    StageMaterializationInfo info;

    if (summary.anchor_resource) {
      // Stage has explicit applicable_units attribute
      ArchitectureGraph::ResourceKind kind =
          arch_graph.getResourceKind(*summary.anchor_resource);

      if (kind == ArchitectureGraph::ResourceKind::Compute) {
        // Compute stage - adapt FIFO kinds to match shortest path
        info.kind = StageMaterializationInfo::Kind::kAdaptFifoKinds;
        info.applicable_unit =
            (*summary.anchor_resource == ArchitectureGraph::Resource::SFP)
                ? mlir::ktdf::ComponentType::SFP
                : mlir::ktdf::ComponentType::PE;
      } else {
        // Memory resource as anchor - preserve as-is
        info.kind = StageMaterializationInfo::Kind::kPreserveOriginal;
      }
    } else if (summary.is_compute_containing) {
      // Compute stage without explicit applicable_units - adapt FIFO kinds
      info.kind = StageMaterializationInfo::Kind::kAdaptFifoKinds;
    } else if (summary.is_transfer_only) {
      // Transfer-only stage - check if already legal, otherwise adapt
      bool all_legal = true;

      mlir::ktdf::StageOp stage_op =
          mlir::cast<mlir::ktdf::StageOp>(stage->getOperation());
      stage_op.walk([&](mlir::ktdf::DataTransferOp transfer) {
        // Infer source and destination from this transfer
        mlir::Type src_type = transfer.getSource().getType();
        mlir::Type dst_type = transfer.getDestination().getType();

        std::optional<ArchitectureGraph::Resource> src_res;
        std::optional<ArchitectureGraph::Resource> dst_res;

        if (mlir::MemRefType src_memref =
                mlir::dyn_cast<mlir::MemRefType>(src_type)) {
          src_res = inferResourceFromMemorySpace(src_memref.getMemorySpace());
        } else if (mlir::ktdf::FifoSlotType src_fifo =
                       mlir::dyn_cast<mlir::ktdf::FifoSlotType>(src_type)) {
          src_res = inferResourceFromFifoType(src_fifo.getFifoType(), false);
        }

        if (mlir::MemRefType dst_memref =
                mlir::dyn_cast<mlir::MemRefType>(dst_type)) {
          dst_res = inferResourceFromMemorySpace(dst_memref.getMemorySpace());
        } else if (mlir::ktdf::FifoSlotType dst_fifo =
                       mlir::dyn_cast<mlir::ktdf::FifoSlotType>(dst_type)) {
          dst_res = inferResourceFromFifoType(dst_fifo.getFifoType(), true);
        }

        if (src_res && dst_res) {
          ArchitectureGraph::NodeId src_node =
              arch_graph.getNodeIdForResource(*src_res);
          ArchitectureGraph::NodeId dst_node =
              arch_graph.getNodeIdForResource(*dst_res);
          std::optional<ArchitectureGraph::EdgeInfo> edge_info =
              arch_graph.getEdgeInfo(src_node, dst_node);

          if (edge_info) {
            // Legal hop - store for potential adaptation
            TransferMaterializationInfo transfer_info;
            transfer_info.template_op = transfer;
            transfer_info.hop = *edge_info;
            transfer_info.source_resource = *src_res;
            transfer_info.dest_resource = *dst_res;
            info.transfers.push_back(transfer_info);
          } else {
            // Illegal hop
            all_legal = false;
          }
        } else {
          all_legal = false;
        }
      });

      // If all transfers are legal, preserve; otherwise adapt
      info.kind = all_legal ? StageMaterializationInfo::Kind::kPreserveOriginal
                            : StageMaterializationInfo::Kind::kAdaptFifoKinds;
    } else {
      // No clear classification - preserve as-is
      info.kind = StageMaterializationInfo::Kind::kPreserveOriginal;
    }

    plan->stage_info[stage] = info;
  }

  // Step 12: Record new private resources for intermediate buffers
  // Use the buffer specs collected during hop-to-stage analysis
  for (ArchitectureGraph::Resource res : intermediate_resources) {
    if (resource_specs.contains(res)) {
      plan->new_resources.push_back(resource_specs[res]);
    } else {
      // Fallback: if we couldn't infer shape, log a warning
      LLVM_DEBUG(
          llvm::dbgs()
          << "Warning: Could not infer buffer shape for intermediate resource "
          << ArchitectureGraph::stringifyResource(res) << "\n");
    }
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