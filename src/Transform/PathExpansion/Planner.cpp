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

#include "llvm/Support/Debug.h"
#include "mlir/IR/BuiltinTypes.h"
#include "src/Dialects/KTDF/KTDFOps.hpp"

#define DEBUG_TYPE "path-expansion-planner"

namespace scheduler {

namespace {

/// Helper to infer resource from memory space attribute
std::optional<ArchitectureGraph::Resource> inferResourceFromMemorySpace(
    mlir::Attribute memory_space) {
  if (!memory_space) return std::nullopt;

  auto memory_space_attr =
      mlir::dyn_cast<mlir::ktdf::MemorySpaceAttr>(memory_space);
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
        applicable_units->getUnits();
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

mlir::FailureOr<llvm::SmallVector<StageSummary>> analyzeStages(
    PipelineNode* pipeline) {
  llvm::SmallVector<StageSummary> summaries;

  auto stages = pipeline->getStages();
  for (auto* stage : stages) {
    auto summary_or = analyzeStage(stage);
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

mlir::FailureOr<PathExpansionPlan> planPathExpansion(
    PipelineTree& tree, PipelineNode* pipeline,
    const ArchitectureGraph& arch_graph) {
  PathExpansionPlan plan;
  plan.changed = false;

  // Step 1: Analyze stages
  auto summaries_or = analyzeStages(pipeline);
  if (mlir::failed(summaries_or)) {
    return mlir::failure();
  }
  auto summaries = *summaries_or;

  // Step 2: Infer original path
  auto original_path_or = inferOriginalPath(summaries);
  if (mlir::failed(original_path_or)) {
    return mlir::failure();
  }
  auto original_path = *original_path_or;

  if (original_path.size() < 2) {
    // Need at least source and destination
    return mlir::failure();
  }

  // Step 3: Query architecture graph for shortest path
  ArchitectureGraph::Resource source_resource = original_path.front();
  ArchitectureGraph::Resource dest_resource = original_path.back();

  ArchitectureGraph::NodeId source_node =
      arch_graph.getNodeIdForResource(source_resource);
  ArchitectureGraph::NodeId dest_node =
      arch_graph.getNodeIdForResource(dest_resource);

  auto shortest_path_or = arch_graph.findShortestPath(source_node, dest_node);
  if (!shortest_path_or) {
    // No legal path exists
    return mlir::failure();
  }

  // Step 4: Convert node IDs back to resources
  llvm::SmallVector<ArchitectureGraph::Resource> shortest_path;
  for (ArchitectureGraph::NodeId node_id : *shortest_path_or) {
    auto node_or = arch_graph.getNode(node_id);
    if (!node_or) return mlir::failure();
    shortest_path.push_back(node_or->resource);
  }

  // Step 5: Compare paths
  if (original_path == shortest_path) {
    // Already legal, no changes needed
    plan.changed = false;
    plan.modified_tree = tree;
    return plan;
  }

  // Step 6: Path needs expansion - mark as changed
  plan.changed = true;

  // TODO: Implement tree modification to insert intermediate stages
  // For now, just copy the tree and mark stages for preservation
  plan.modified_tree = tree;

  // Mark all existing stages as preserve original for now
  auto stages = pipeline->getStages();
  for (StageNode* stage : stages) {
    StageMaterializationInfo info;
    info.kind = StageMaterializationInfo::Kind::kPreserveOriginal;
    plan.stage_info[stage] = info;
  }

  return plan;
}

}  // namespace scheduler

// Made with Bob