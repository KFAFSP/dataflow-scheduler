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
void validateFifoSlotIndex(mlir::Value orig_fifo_value, size_t assigned_slot_idx,
                           const PrivateResourceSpec* fifo_spec) {
  if (auto orig_private_result = mlir::dyn_cast<mlir::OpResult>(orig_fifo_value)) {
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

  // Step 4: Resource-centric path expansion
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

  // Simplified algorithm: Walk through sorted stages and check if each hop is legal
  // If a hop is illegal, insert an intermediate stage for the memory resource
  
  llvm::SmallVector<StageNode*> stage_sequence;
  int next_stage_id = sorted_stages.size();
  StageNode* last_inserted_stage = nullptr;

  for (size_t i = 0; i < sorted_stages.size(); ++i) {
    StageNode* current_stage = sorted_stages[i];
    const StageSummary& current_summary = summaries[i];
    
    // Determine the resource for this stage
    std::optional<ArchitectureGraph::Resource> current_resource;
    if (current_summary.anchor_resource) {
      current_resource = current_summary.anchor_resource;
    } else if (i == 0 && !current_summary.input_endpoints.empty()) {
      current_resource = current_summary.input_endpoints[0].resource;
    } else if (i == sorted_stages.size() - 1 && !current_summary.output_endpoints.empty()) {
      current_resource = current_summary.output_endpoints[0].resource;
    }
    
    assert(current_resource && "Unable to determine resource for stage");
    
    // Add the current stage to the sequence
    stage_sequence.push_back(current_stage);
    last_inserted_stage = current_stage;
    
    // Create materialization info for current stage
    StageMaterializationInfo info;
    info.kind = StageMaterializationInfo::Kind::kAdaptFifoKinds;
    if (current_summary.anchor_resource &&
        arch_graph.getResourceKind(*current_summary.anchor_resource) ==
            ArchitectureGraph::ResourceKind::Compute) {
      info.applicable_unit =
          (*current_summary.anchor_resource == ArchitectureGraph::Resource::SFP)
              ? mlir::ktdf::ComponentType::SFP
              : mlir::ktdf::ComponentType::PE;
    }
    
    // Track FIFO allocations needed for this stage
    // Map from fifo_type to the PrivateResourceSpec* created by factory
    llvm::DenseMap<mlir::ktdf::FifoType, PrivateResourceSpec*> fifo_specs;
    // Track next available slot index for each FIFO type
    llvm::DenseMap<mlir::ktdf::FifoType, size_t> fifo_next_slot;
    
    // For existing stages, record transfer info for FIFO adaptation
    // Walk through the stage's data transfers and record edge info
    mlir::ktdf::StageOp current_stage_op =
        mlir::cast<mlir::ktdf::StageOp>(current_stage->getOperation());
    current_stage_op.walk([&](mlir::ktdf::DataTransferOp transfer) {
      // Determine source and destination resources for this transfer
      mlir::Type src_type = transfer.getSource().getType();
      mlir::Type dest_type = transfer.getDestination().getType();
      
      std::optional<ArchitectureGraph::Resource> src_res;
      std::optional<ArchitectureGraph::Resource> dest_res;
      mlir::Type element_type;
      int64_t num_elements = 1;
      
      // Infer source resource
      if (auto memref = mlir::dyn_cast<mlir::MemRefType>(src_type)) {
        src_res = inferResourceFromMemorySpace(memref.getMemorySpace());
        element_type = memref.getElementType();
        for (int64_t dim : memref.getShape()) {
          num_elements *= dim;
        }
      } else if (auto fifo = mlir::dyn_cast<mlir::ktdf::FifoSlotType>(src_type)) {
        src_res = inferResourceFromFifoType(fifo.getFifoType(), true);
        element_type = fifo.getElementType();
        assert(!fifo.isDynamicNumElements() && "Dynamic FIFO sizes not supported in path expansion");
        num_elements = fifo.getStaticNumElements();
      }
      
      // Infer destination resource
      if (auto memref = mlir::dyn_cast<mlir::MemRefType>(dest_type)) {
        dest_res = inferResourceFromMemorySpace(memref.getMemorySpace());
        if (!element_type) {
          element_type = memref.getElementType();
          num_elements = 1;
          for (int64_t dim : memref.getShape()) {
            num_elements *= dim;
          }
        }
      } else if (auto fifo = mlir::dyn_cast<mlir::ktdf::FifoSlotType>(dest_type)) {
        dest_res = inferResourceFromFifoType(fifo.getFifoType(), false);
        if (!element_type) {
          element_type = fifo.getElementType();
          assert(!fifo.isDynamicNumElements() && "Dynamic FIFO sizes not supported in path expansion");
          num_elements = fifo.getStaticNumElements();
        }
      }
      
      // If we have both resources, look up the edge info for adaptation
      if (src_res && dest_res) {
        ArchitectureGraph::NodeId src_node = arch_graph.getNodeIdForResource(*src_res);
        ArchitectureGraph::NodeId dest_node = arch_graph.getNodeIdForResource(*dest_res);
        std::optional<ArchitectureGraph::EdgeInfo> edge =
            arch_graph.getEdgeInfo(src_node, dest_node);
        
        assert(edge && "Edge should exist for transfer");
        
        // Check if this transfer involves a FIFO
        bool src_is_fifo = mlir::isa<mlir::ktdf::FifoSlotType>(src_type);
        bool dest_is_fifo = mlir::isa<mlir::ktdf::FifoSlotType>(dest_type);
        
        if (src_is_fifo || dest_is_fifo) {
          // Determine the FIFO type from the edge
          mlir::ktdf::FifoType fifo_type;
          if (edge->fifo_type) {
            fifo_type = *edge->fifo_type;
          } else {
            // Fallback: infer from resources
            if (*src_res == ArchitectureGraph::Resource::HBM &&
                *dest_res == ArchitectureGraph::Resource::SFP) {
              fifo_type = mlir::ktdf::FifoType::HbmToSfp;
            } else if (*src_res == ArchitectureGraph::Resource::SFP &&
                       *dest_res == ArchitectureGraph::Resource::HBM) {
              fifo_type = mlir::ktdf::FifoType::SfpToHbm;
            } else if (*src_res == ArchitectureGraph::Resource::LX &&
                       *dest_res == ArchitectureGraph::Resource::SFP) {
              fifo_type = mlir::ktdf::FifoType::LxluToSfp;
            } else if (*src_res == ArchitectureGraph::Resource::SFP &&
                       *dest_res == ArchitectureGraph::Resource::LX) {
              fifo_type = mlir::ktdf::FifoType::SfpToLxsu;
            } else {
              llvm_unreachable("Unsupported FIFO edge");
            }
          }
          
          // Group FIFO allocations by fifo_type
          // All transfers using the same FIFO type will share one fifo.allocate
          if (fifo_specs.find(fifo_type) == fifo_specs.end()) {
            // Create new FIFO spec via factory with first slot
            PrivateResourceSpec* spec = plan->resource_factory.createFifo(
                fifo_type,
                {num_elements},  // First slot
                element_type);
            fifo_specs[fifo_type] = spec;
            fifo_next_slot[fifo_type] = 0;  // Initialize slot counter
          } else {
            // Add another slot to existing FIFO spec
            PrivateResourceSpec* spec = fifo_specs[fifo_type];
            spec->elements_per_slot.push_back(num_elements);
          }
        }
        
        TransferMaterializationInfo transfer_info;
        transfer_info.template_op = transfer;
        transfer_info.hop = *edge;
        transfer_info.source_resource = *src_res;
        transfer_info.dest_resource = *dest_res;
        
        // Link FIFO specs to transfer info and assign slot indices
        if (src_is_fifo) {
          mlir::ktdf::FifoType fifo_type;
          if (edge->fifo_type) {
            fifo_type = *edge->fifo_type;
          } else {
            // Infer fifo type from resources (same logic as above)
            if (*src_res == ArchitectureGraph::Resource::HBM &&
                *dest_res == ArchitectureGraph::Resource::SFP) {
              fifo_type = mlir::ktdf::FifoType::HbmToSfp;
            } else if (*src_res == ArchitectureGraph::Resource::SFP &&
                       *dest_res == ArchitectureGraph::Resource::HBM) {
              fifo_type = mlir::ktdf::FifoType::SfpToHbm;
            } else if (*src_res == ArchitectureGraph::Resource::LX &&
                       *dest_res == ArchitectureGraph::Resource::SFP) {
              fifo_type = mlir::ktdf::FifoType::LxluToSfp;
            } else if (*src_res == ArchitectureGraph::Resource::SFP &&
                       *dest_res == ArchitectureGraph::Resource::LX) {
              fifo_type = mlir::ktdf::FifoType::SfpToLxsu;
            } else {
              llvm_unreachable("Unsupported FIFO edge");
            }
          }
          transfer_info.source_private_resource = fifo_specs[fifo_type];
          // Assign and increment slot index for this FIFO type
          size_t slot_idx = fifo_next_slot[fifo_type]++;
          transfer_info.source_slot_index = slot_idx;
          
          // Validate slot index assignment if this is from the original template
          if (mlir::isa<mlir::ktdf::FifoSlotType>(src_type)) {
            validateFifoSlotIndex(transfer.getSource(), slot_idx, fifo_specs[fifo_type]);
          }
        }
        
        if (dest_is_fifo) {
          mlir::ktdf::FifoType fifo_type;
          if (edge->fifo_type) {
            fifo_type = *edge->fifo_type;
          } else {
            // Infer fifo type from resources (same logic as above)
            if (*src_res == ArchitectureGraph::Resource::HBM &&
                *dest_res == ArchitectureGraph::Resource::SFP) {
              fifo_type = mlir::ktdf::FifoType::HbmToSfp;
            } else if (*src_res == ArchitectureGraph::Resource::SFP &&
                       *dest_res == ArchitectureGraph::Resource::HBM) {
              fifo_type = mlir::ktdf::FifoType::SfpToHbm;
            } else if (*src_res == ArchitectureGraph::Resource::LX &&
                       *dest_res == ArchitectureGraph::Resource::SFP) {
              fifo_type = mlir::ktdf::FifoType::LxluToSfp;
            } else if (*src_res == ArchitectureGraph::Resource::SFP &&
                       *dest_res == ArchitectureGraph::Resource::LX) {
              fifo_type = mlir::ktdf::FifoType::SfpToLxsu;
            } else {
              llvm_unreachable("Unsupported FIFO edge");
            }
          }
          transfer_info.dest_private_resource = fifo_specs[fifo_type];
          // Assign and increment slot index for this FIFO type
          size_t slot_idx = fifo_next_slot[fifo_type]++;
          transfer_info.dest_slot_index = slot_idx;
          
          // Validate slot index assignment if this is from the original template
          if (mlir::isa<mlir::ktdf::FifoSlotType>(dest_type)) {
            validateFifoSlotIndex(transfer.getDestination(), slot_idx, fifo_specs[fifo_type]);
          }
        }
        
        info.transfers.push_back(transfer_info);
      }
      
      return mlir::WalkResult::advance();
    });
    
    plan->stage_info[current_stage] = info;
    
    LLVM_DEBUG(llvm::dbgs() << "Processing stage " << current_stage->getStageId()
               << " at " << ArchitectureGraph::stringifyResource(*current_resource)
               << "\n");
    
    // Check if we need to expand the path to the next stage
    if (i + 1 < sorted_stages.size()) {
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
        // Hop is not legal - need to find shortest path and insert intermediate stage
        std::optional<llvm::SmallVector<ArchitectureGraph::NodeId>> path =
            arch_graph.findShortestPath(current_node, next_node);
        
        assert(path && "No path found between stages");
        assert(path->size() == 3 &&
               "Path expansion currently assumes at most one intermediate node");
        
        // Path has intermediate nodes - insert a stage for the intermediate memory
        LLVM_DEBUG(llvm::dbgs() << "  Illegal hop from "
                   << ArchitectureGraph::stringifyResource(*current_resource)
                   << " to "
                   << ArchitectureGraph::stringifyResource(*next_resource)
                   << ", inserting intermediate stage\n");
        
        // Get the intermediate node (should be exactly one)
        ArchitectureGraph::NodeId intermediate_node = (*path)[1];
        std::optional<ArchitectureGraph::ResourceNode> intermediate_res_node =
            arch_graph.getNode(intermediate_node);
        
        assert(intermediate_res_node && "Intermediate node not found in graph");
        
        ArchitectureGraph::Resource intermediate_resource =
            intermediate_res_node->resource;
        
        // Create new intermediate stage
        StageNode* intermediate_stage = tree.createStageNode(nullptr, next_stage_id++);
        
        LLVM_DEBUG(llvm::dbgs() << "  Creating intermediate stage "
                   << intermediate_stage->getStageId() << " at "
                   << ArchitectureGraph::stringifyResource(intermediate_resource)
                   << "\n");
        
        // Create materialization info for the intermediate stage
        StageMaterializationInfo intermediate_info;
        intermediate_info.kind = StageMaterializationInfo::Kind::kSyntheticTransfer;
        
        // Get edge info for the hop
        std::optional<ArchitectureGraph::EdgeInfo> edge1 =
            arch_graph.getEdgeInfo(current_node, intermediate_node);
        assert(edge1 && "Edge should exist in shortest path");
        
        // For each data transfer in the current stage, create a corresponding transfer
        // to the intermediate resource
        mlir::ktdf::StageOp current_stage_op =
            mlir::cast<mlir::ktdf::StageOp>(current_stage->getOperation());
        
        current_stage_op.walk([&](mlir::ktdf::DataTransferOp transfer) {
          // Create transfer materialization info
          TransferMaterializationInfo transfer_info;
          transfer_info.template_op = transfer;
          transfer_info.hop = *edge1;
          transfer_info.source_resource = *current_resource;
          transfer_info.dest_resource = intermediate_resource;
          
          // If intermediate resource is memory, create buffer spec for this transfer
          if (arch_graph.getResourceKind(intermediate_resource) ==
              ArchitectureGraph::ResourceKind::Memory) {
            // Extract shape/type from the transfer
            mlir::Type dest_type = transfer.getDestination().getType();
            mlir::Type src_type = transfer.getSource().getType();
            
            // Try to get memref type from either source or destination
            mlir::MemRefType memref_type;
            if (auto src_memref = mlir::dyn_cast<mlir::MemRefType>(src_type)) {
              memref_type = src_memref;
            } else if (auto dest_memref = mlir::dyn_cast<mlir::MemRefType>(dest_type)) {
              memref_type = dest_memref;
            }
            
            assert(memref_type && "Unable to extract memref type from transfer");
            
            // Create memory buffer spec using factory
            PrivateResourceSpec* spec = plan->resource_factory.createMemoryBuffer(
                intermediate_resource,
                memref_type.getShape(),
                memref_type.getElementType());
            
            // Link this transfer to the buffer resource
            // The intermediate resource is the destination of the first hop
            // (from current_resource to intermediate_resource)
            transfer_info.dest_private_resource = spec;
            
            LLVM_DEBUG({
              llvm::dbgs() << "  Recording buffer spec for "
                           << ArchitectureGraph::stringifyResource(intermediate_resource)
                           << " with shape [";
              for (size_t j = 0; j < spec->shape.size(); ++j) {
                llvm::dbgs() << spec->shape[j];
                if (j + 1 < spec->shape.size()) llvm::dbgs() << ", ";
              }
              llvm::dbgs() << "]\n";
            });
          }
          
          intermediate_info.transfers.push_back(transfer_info);
          return mlir::WalkResult::advance();
        });
        
        plan->stage_info[intermediate_stage] = intermediate_info;
        
        // Insert the intermediate stage after the current stage
        pipeline->insertChildNode(intermediate_stage, last_inserted_stage);
        stage_sequence.push_back(intermediate_stage);
        last_inserted_stage = intermediate_stage;
      } else {
        LLVM_DEBUG(llvm::dbgs() << "  Legal hop to next stage\n");
      }
    }
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

  // Count synthetic stages (new_resources are already populated during transfer creation)
  size_t synthetic_stage_count = 0;
  for (const auto& [stage, info] : plan->stage_info) {
    if (info.kind == StageMaterializationInfo::Kind::kSyntheticTransfer) {
      synthetic_stage_count++;
    }
  }

  LLVM_DEBUG(llvm::dbgs() << "Plan created with " << plan->stage_info.size()
                          << " stage entries (including "
                          << synthetic_stage_count << " synthetic stages) and "
                          << plan->resource_factory.getSpecs().size() << " new resources\n");

  return plan;
}

}  // namespace scheduler

// Made with Bob