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
// Path Expansion Pipeline Tree Materializer Implementation
//===----------------------------------------------------------------------===//

#include "Materializer.hpp"

#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "src/Dialects/KTDF/KTDFOps.hpp"

#define DEBUG_TYPE "path-expansion-materializer"

namespace scheduler {

//===----------------------------------------------------------------------===//
// Main Materialization Entry Point
//===----------------------------------------------------------------------===//

mlir::Operation* PathExpansionMaterializer::materialize(
    const PipelineTreeNode& root_node) {
  // Clear any previous materialization state
  node_to_materialized_op_.clear();
  stage_to_token_map_.clear();
  resource_index_to_value_.clear();

  // Materialize the root and all its children
  materializeTreeNode(root_node);

  // Return the materialized operation for the root
  return getMaterializedOp(root_node);
}

mlir::Operation* PathExpansionMaterializer::getMaterializedOp(
    const PipelineTreeNode& node) const {
  auto it = node_to_materialized_op_.find(&node);
  return it != node_to_materialized_op_.end() ? it->second : nullptr;
}

//===----------------------------------------------------------------------===//
// Tree Node Materialization Dispatcher
//===----------------------------------------------------------------------===//

void PathExpansionMaterializer::materializeTreeNode(
    const PipelineTreeNode& node) {
  // Handle different node types
  if (node.isLoopNode()) {
    const LoopNode& loop_node = static_cast<const LoopNode&>(node);
    materializeLoopNode(loop_node);
  } else if (node.isStageNode()) {
    const StageNode& stage_node = static_cast<const StageNode&>(node);
    materializeStageNode(stage_node);
  } else if (node.isPipelineNode()) {
    const PipelineNode& pipeline_node = static_cast<const PipelineNode&>(node);
    materializePipelineNode(pipeline_node);
  } else if (node.isPrivateNode()) {
    // Private nodes are handled specially within pipeline materialization
    // They should not be materialized independently
    llvm_unreachable(
        "Private nodes should be materialized within pipeline context");
  } else {
    llvm_unreachable("unimplemented node type");
  }
}

void PathExpansionMaterializer::materializeChildren(
    const PipelineTreeNode& parent_node) {
  OperationTreeNode* child = parent_node.getFirstChild();
  while (child) {
    materializeTreeNode(static_cast<const PipelineTreeNode&>(*child));
    child = child->getNextSibling();
  }
}

//===----------------------------------------------------------------------===//
// Helper Methods
//===----------------------------------------------------------------------===//

mlir::Value PathExpansionMaterializer::materializeValue(mlir::Value value) {
  return value_map_.lookupOrDefault(value);
}

mlir::Attribute PathExpansionMaterializer::getMemorySpaceFromResource(
    ArchitectureGraph::Resource resource) {
  mlir::MLIRContext* ctx = builder_.getContext();

  switch (resource) {
    case ArchitectureGraph::Resource::HBM:
      return mlir::ktdf::MemorySpaceCustomAttr::get(
          ctx, mlir::ktdf::MemorySpace::HBM);
    case ArchitectureGraph::Resource::LX:
      return mlir::ktdf::MemorySpaceCustomAttr::get(
          ctx, mlir::ktdf::MemorySpace::LX);
    default:
      llvm_unreachable("Unsupported memory resource");
  }
}

mlir::ktdf::FifoSlotType PathExpansionMaterializer::getFifoTypeFromEdge(
    const ArchitectureGraph::EdgeInfo& edge,
    ArchitectureGraph::Resource source_resource,
    ArchitectureGraph::Resource dest_resource, mlir::Type element_type,
    llvm::ArrayRef<int64_t> shape) {
  mlir::MLIRContext* ctx = builder_.getContext();

  // Determine FIFO type from edge
  mlir::ktdf::FifoType fifo_type;
  if (edge.fifo_type) {
    fifo_type = *edge.fifo_type;
  } else {
    // Default based on source/dest resources
    if (source_resource == ArchitectureGraph::Resource::HBM &&
        dest_resource == ArchitectureGraph::Resource::SFP) {
      fifo_type = mlir::ktdf::FifoType::HbmToSfp;
    } else if (source_resource == ArchitectureGraph::Resource::SFP &&
               dest_resource == ArchitectureGraph::Resource::HBM) {
      fifo_type = mlir::ktdf::FifoType::SfpToHbm;
    } else if (source_resource == ArchitectureGraph::Resource::LX &&
               dest_resource == ArchitectureGraph::Resource::SFP) {
      fifo_type = mlir::ktdf::FifoType::LxluToSfp;
    } else if (source_resource == ArchitectureGraph::Resource::SFP &&
               dest_resource == ArchitectureGraph::Resource::LX) {
      fifo_type = mlir::ktdf::FifoType::SfpToLxsu;
    } else {
      llvm_unreachable("Unsupported FIFO edge");
    }
  }

  // Calculate total number of elements
  int64_t num_elements = 1;
  for (int64_t dim : shape) {
    num_elements *= dim;
  }

  return mlir::ktdf::FifoSlotType::get(ctx, fifo_type, num_elements,
                                       element_type);
}

//===----------------------------------------------------------------------===//
// Loop Node Materialization
//===----------------------------------------------------------------------===//

mlir::scf::ForOp PathExpansionMaterializer::materializeLoopNode(
    const LoopNode& loop_node) {
  // Get the template operation to copy loop bounds from
  mlir::Operation* template_op = loop_node.getOperationOrTemplate();

  assert(template_op && "Loop node must have either operation or template");
  auto template_for = mlir::cast<mlir::scf::ForOp>(template_op);

  // Materialize the loop bounds
  mlir::Value lower_bound = materializeValue(template_for.getLowerBound());
  mlir::Value upper_bound = materializeValue(template_for.getUpperBound());
  mlir::Value step = materializeValue(template_for.getStep());

  // Create the new loop with empty body
  auto new_loop = mlir::scf::ForOp::create(builder_, template_for.getLoc(),
                                           lower_bound, upper_bound, step);

  // Copy attributes from template loop to new loop
  for (mlir::NamedAttribute attr : template_for->getAttrs()) {
    new_loop->setAttr(attr.getName(), attr.getValue());
  }

  // Map the induction variable
  value_map_.map(template_for.getInductionVar(), new_loop.getInductionVar());

  // Track the materialized operation
  node_to_materialized_op_[&loop_node] = new_loop.getOperation();

  // Set insertion point inside the loop body and materialize children
  builder_.setInsertionPointToStart(new_loop.getBody());
  materializeChildren(loop_node);
  builder_.setInsertionPointAfter(new_loop);

  LLVM_DEBUG(llvm::dbgs() << "  Materialized loop node\n");
  return new_loop;
}

//===----------------------------------------------------------------------===//
// Pipeline Node Materialization
//===----------------------------------------------------------------------===//

mlir::ktdf::PipelineOp PathExpansionMaterializer::materializePipelineNode(
    const PipelineNode& pipeline_node) {
  mlir::Operation* template_op = pipeline_node.getOperationOrTemplate();
  assert(template_op && "Pipeline node must have template");

  auto template_pipeline = mlir::cast<mlir::ktdf::PipelineOp>(template_op);
  mlir::Location loc = template_pipeline.getLoc();

  // Create new pipeline operation
  auto new_pipeline = mlir::ktdf::PipelineOp::create(builder_, loc);

  // Track the materialized operation
  node_to_materialized_op_[&pipeline_node] = new_pipeline.getOperation();

  // Set insertion point inside the pipeline body
  builder_.setInsertionPointToStart(&new_pipeline.getBody().front());

  // Find the private node child (should be first child)
  OperationTreeNode* child = pipeline_node.getFirstChild();
  const PrivateNode* private_node = nullptr;

  while (child) {
    const auto pipeline_child = static_cast<const PipelineTreeNode*>(child);
    if (pipeline_child->isPrivateNode()) {
      private_node = static_cast<const PrivateNode*>(pipeline_child);
      break;
    }
    child = child->getNextSibling();
  }

  assert(private_node && "Pipeline must have a private node");

  // Materialize the private node first to create SSA values for resources
  materializePrivateNode(*private_node, pipeline_node);

  // Now materialize all stage children
  child = pipeline_node.getFirstChild();
  while (child) {
    const auto pipeline_child = static_cast<const PipelineTreeNode*>(child);
    if (pipeline_child->isStageNode()) {
      materializeTreeNode(*pipeline_child);
    }
    child = child->getNextSibling();
  }

  builder_.setInsertionPointAfter(new_pipeline);

  LLVM_DEBUG(llvm::dbgs() << "  Materialized pipeline node\n");
  return new_pipeline;
}

//===----------------------------------------------------------------------===//
// Private Node Materialization
//===----------------------------------------------------------------------===//

void PathExpansionMaterializer::materializePrivateNode(
    const PrivateNode& private_node, const PipelineNode& pipeline_node) {
  mlir::Operation* template_op = private_node.getOperationOrTemplate();
  mlir::Location loc =
      template_op ? template_op->getLoc() : builder_.getUnknownLoc();

  // Collect all stages with dependencies to determine token count
  llvm::SmallVector<StageNode*> stages_with_deps;
  OperationTreeNode* child = pipeline_node.getFirstChild();
  while (child) {
    auto pipeline_child = static_cast<PipelineTreeNode*>(child);
    if (pipeline_child->isStageNode()) {
      StageNode* stage_node = static_cast<StageNode*>(pipeline_child);
      // A stage needs a token if it has dependencies or is depended upon
      bool needs_token = !stage_node->getDependencies().empty();

      // Check if any other stage depends on this one
      OperationTreeNode* other_child = pipeline_node.getFirstChild();
      while (other_child && !needs_token) {
        auto other_pipeline_child = static_cast<PipelineTreeNode*>(other_child);
        if (other_pipeline_child->isStageNode() && other_child != child) {
          auto other_stage = static_cast<StageNode*>(other_pipeline_child);
          for (StageNode* dep : other_stage->getDependencies()) {
            if (dep == stage_node) {
              needs_token = true;
              break;
            }
          }
        }
        other_child = other_child->getNextSibling();
      }

      if (needs_token) {
        stages_with_deps.push_back(stage_node);
      }
    }
    child = child->getNextSibling();
  }

  // Build result types: new resources + FIFOs + tokens
  llvm::SmallVector<mlir::Type> result_types;

  // Add new buffer resources
  for (const PrivateResourceSpec& spec : new_resources_) {
    mlir::Attribute memory_space = getMemorySpaceFromResource(spec.resource);
    mlir::MemRefType memref_type =
        mlir::MemRefType::get(spec.shape, spec.element_type,
                              mlir::MemRefLayoutAttrInterface{}, memory_space);
    result_types.push_back(memref_type);
  }

  size_t non_token_result_count = result_types.size();

  // Add FIFO slots - collect from stage materialization info
  llvm::DenseSet<mlir::Type> fifo_types_set;
  llvm::SmallVector<mlir::Type> fifo_types;

  for (const std::pair<StageNode* const, StageMaterializationInfo>& entry :
       stage_info_) {
    const StageMaterializationInfo& info = entry.second;
    for (const TransferMaterializationInfo& transfer_info : info.transfers) {
      if (transfer_info.hop.fifo_type) {
        // Extract element type and shape from template if available
        mlir::Type element_type;
        llvm::SmallVector<int64_t> shape;

        if (transfer_info.template_op) {
          // Get from template transfer op (need to make it mutable)
          mlir::ktdf::DataTransferOp template_op = transfer_info.template_op;
          mlir::Type src_type = template_op.getSource().getType();
          if (auto memref_type = mlir::dyn_cast<mlir::MemRefType>(src_type)) {
            element_type = memref_type.getElementType();
            shape.assign(memref_type.getShape().begin(),
                         memref_type.getShape().end());
          } else if (auto fifo_type =
                         mlir::dyn_cast<mlir::ktdf::FifoSlotType>(src_type)) {
            auto tensor_type = fifo_type.getElementType();
            if (auto ranked =
                    mlir::dyn_cast<mlir::RankedTensorType>(tensor_type)) {
              element_type = ranked.getElementType();
              shape.assign(ranked.getShape().begin(), ranked.getShape().end());
            }
          }
        }

        if (element_type && !shape.empty()) {
          mlir::ktdf::FifoSlotType fifo_type = getFifoTypeFromEdge(
              transfer_info.hop, transfer_info.source_resource,
              transfer_info.dest_resource, element_type, shape);

          // Avoid duplicates
          if (!fifo_types_set.contains(fifo_type)) {
            fifo_types_set.insert(fifo_type);
            fifo_types.push_back(fifo_type);
          }
        }
      }
    }
  }

  result_types.append(fifo_types.begin(), fifo_types.end());
  non_token_result_count = result_types.size();

  // Add tokens for stages with dependencies
  mlir::Type token_type = mlir::ktdf::TokenType::get(builder_.getContext());
  for (size_t i = 0; i < stages_with_deps.size(); ++i) {
    result_types.push_back(token_type);
  }

  // Create the new private operation
  auto new_private = mlir::ktdf::PrivateOp::create(builder_, loc, result_types);

  // Set insertion point inside private body
  mlir::Block* private_body = &new_private.getBody().front();
  builder_.setInsertionPointToStart(private_body);

  // Allocate new buffer resources
  for (size_t i = 0; i < new_resources_.size(); ++i) {
    const PrivateResourceSpec& spec = new_resources_[i];
    mlir::Attribute memory_space = getMemorySpaceFromResource(spec.resource);
    mlir::MemRefType memref_type =
        mlir::MemRefType::get(spec.shape, spec.element_type,
                              mlir::MemRefLayoutAttrInterface{}, memory_space);

    auto alloc_op = mlir::memref::AllocOp::create(builder_, loc, memref_type);

    // Store mapping from resource index to SSA value
    resource_index_to_value_[i] = alloc_op.getResult();
  }

  // Allocate FIFO slots - allocate all at once
  size_t fifo_start_idx = new_resources_.size();
  if (!fifo_types.empty()) {
    llvm::SmallVector<mlir::Value> dynamic_sizes;  // No dynamic sizes for now
    mlir::ktdf::FifoAllocateOp fifo_alloc = mlir::ktdf::FifoAllocateOp::create(
        builder_, loc, fifo_types, dynamic_sizes);
    for (size_t i = 0; i < fifo_types.size(); ++i) {
      resource_index_to_value_[fifo_start_idx + i] = fifo_alloc.getResult(i);
    }
  }

  // Create tokens
  size_t token_start_idx = non_token_result_count;
  for (size_t i = 0; i < stages_with_deps.size(); ++i) {
    auto token_op = mlir::ktdf::CreateTokenOp::create(builder_, loc);
    resource_index_to_value_[token_start_idx + i] = token_op.getResult();
  }

  // Build yield operands
  llvm::SmallVector<mlir::Value> yield_operands;
  for (size_t i = 0; i < result_types.size(); ++i) {
    yield_operands.push_back(resource_index_to_value_[i]);
  }

  // Create private_yield
  mlir::ktdf::PrivateYieldOp::create(builder_, loc, yield_operands);

  // Build stage-to-token mapping
  buildStageToTokenMapping(stages_with_deps, new_private,
                           non_token_result_count);

  // Map private results to value map for later use
  for (size_t i = 0; i < new_private.getNumResults(); ++i) {
    resource_index_to_value_[i] = new_private.getResult(i);
  }

  // Reset insertion point to after private op
  builder_.setInsertionPointAfter(new_private);

  LLVM_DEBUG(llvm::dbgs() << "  Materialized private node with "
                          << new_resources_.size() << " buffers, "
                          << fifo_types.size() << " FIFOs, and "
                          << stages_with_deps.size() << " tokens\n");
}

void PathExpansionMaterializer::buildStageToTokenMapping(
    const llvm::SmallVector<StageNode*>& stages_with_deps,
    mlir::ktdf::PrivateOp new_private, size_t non_token_result_count) {
  // Map each stage to its token (results after non-token results)
  for (size_t i = 0; i < stages_with_deps.size(); ++i) {
    StageNode* stage = stages_with_deps[i];
    mlir::Value token = new_private.getResult(non_token_result_count + i);
    stage_to_token_map_[stage] = token;
  }
}

//===----------------------------------------------------------------------===//
// Stage Node Materialization
//===----------------------------------------------------------------------===//

mlir::ktdf::StageOp PathExpansionMaterializer::materializeStageNode(
    const StageNode& stage_node) {
  // Look up materialization info for this stage
  auto info_it = stage_info_.find(const_cast<StageNode*>(&stage_node));

  mlir::Operation* template_op = stage_node.getOperationOrTemplate();
  mlir::Location loc =
      template_op ? template_op->getLoc() : builder_.getUnknownLoc();

  // Determine depends_in and depends_out tokens
  mlir::Value depends_in_token;
  mlir::Value depends_out_token;

  // Find which stage depends on this one (for depends_out)
  const llvm::SmallVector<StageNode*>& deps = stage_node.getDependencies();
  if (!deps.empty() && deps[0] != nullptr) {
    llvm::DenseMap<StageNode*, mlir::Value>::iterator token_it =
        stage_to_token_map_.find(deps[0]);
    if (token_it != stage_to_token_map_.end()) {
      depends_out_token = token_it->second;
    }
  }

  // Find which stage this one depends on (for depends_in)
  // Need to search through all stages to find predecessor
  for (const std::pair<StageNode* const, mlir::Value>& entry :
       stage_to_token_map_) {
    StageNode* other_stage = entry.first;
    mlir::Value token = entry.second;
    for (StageNode* dep : other_stage->getDependencies()) {
      if (dep == &stage_node) {
        depends_in_token = token;
        break;
      }
    }
    if (depends_in_token) break;
  }

  // Create the stage operation
  auto new_stage = mlir::ktdf::StageOp::create(builder_, loc, depends_in_token,
                                               depends_out_token);

  // Set applicable_units if present in materialization info
  if (info_it != stage_info_.end() &&
      info_it->second.kind !=
          StageMaterializationInfo::Kind::kPreserveOriginal) {
    if (info_it->second.applicable_unit != mlir::ktdf::ComponentType{}) {
      llvm::SmallVector<mlir::ktdf::ComponentType> units = {
          info_it->second.applicable_unit};
      auto applicable_units_attr = mlir::ktdf::ApplicableUnitsCustomAttr::get(
          builder_.getContext(), units);
      new_stage.setApplicableUnitsAttr(applicable_units_attr);
    }
  } else if (template_op) {
    // Copy applicable_units from template
    auto template_stage = mlir::cast<mlir::ktdf::StageOp>(template_op);
    if (auto applicable_units = template_stage.getApplicableUnits()) {
      new_stage.setApplicableUnitsAttr(*applicable_units);
    }
  }

  // Track the materialized operation
  node_to_materialized_op_[&stage_node] = new_stage.getOperation();

  // Set insertion point inside stage body
  builder_.setInsertionPointToStart(&new_stage.getBody().front());

  // Materialize stage body based on kind
  if (info_it != stage_info_.end()) {
    const StageMaterializationInfo& info = info_it->second;

    switch (info.kind) {
      case StageMaterializationInfo::Kind::kPreserveOriginal:
        if (template_op) {
          cloneStageBody(mlir::cast<mlir::ktdf::StageOp>(template_op));
        }
        break;

      case StageMaterializationInfo::Kind::kAdaptFifoKinds:
        if (template_op) {
          adaptStageBodyFifoKinds(mlir::cast<mlir::ktdf::StageOp>(template_op),
                                  info);
        }
        break;

      case StageMaterializationInfo::Kind::kSyntheticTransfer:
        synthesizeTransferStage(info);
        break;
    }
  } else if (template_op) {
    // No info found, preserve original
    cloneStageBody(mlir::cast<mlir::ktdf::StageOp>(template_op));
  }

  builder_.setInsertionPointAfter(new_stage);

  LLVM_DEBUG(llvm::dbgs() << "  Materialized stage " << stage_node.getStageId()
                          << "\n");
  return new_stage;
}

void PathExpansionMaterializer::cloneStageBody(mlir::ktdf::StageOp orig_stage) {
  mlir::Block* orig_body = &orig_stage.getBody().front();
  for (mlir::Operation& op : *orig_body) {
    if (op.hasTrait<mlir::OpTrait::IsTerminator>()) continue;
    builder_.clone(op, value_map_);
  }
}

void PathExpansionMaterializer::adaptStageBodyFifoKinds(
    mlir::ktdf::StageOp orig_stage, const StageMaterializationInfo& info) {
  // Clone the stage body but adapt FIFO types based on transfer info
  mlir::Block* orig_body = &orig_stage.getBody().front();

  for (mlir::Operation& op : *orig_body) {
    if (op.hasTrait<mlir::OpTrait::IsTerminator>()) continue;

    // Check if this is a read_from_fifo or write_to_fifo that needs adaptation
    if (auto read_op = mlir::dyn_cast<mlir::ktdf::ReadFromFifoOp>(&op)) {
      // Find corresponding transfer info
      // For now, clone as-is (full adaptation would require matching with
      // transfer info)
      builder_.clone(op, value_map_);
    } else if (auto write_op = mlir::dyn_cast<mlir::ktdf::WriteToFifoOp>(&op)) {
      // Find corresponding transfer info
      // For now, clone as-is
      builder_.clone(op, value_map_);
    } else {
      // Clone other operations as-is
      builder_.clone(op, value_map_);
    }
  }
}

void PathExpansionMaterializer::synthesizeTransferStage(
    const StageMaterializationInfo& info) {
  // Synthesize new data transfer operations based on transfer info
  for (const TransferMaterializationInfo& transfer_info : info.transfers) {
    if (transfer_info.template_op) {
      // Clone and adapt the template transfer
      mlir::ktdf::DataTransferOp template_transfer = transfer_info.template_op;
      mlir::Location loc = template_transfer.getLoc();

      // Get source and destination from template, but adapt types based on hop
      mlir::Value source = materializeValue(template_transfer.getSource());
      mlir::Value destination =
          materializeValue(template_transfer.getDestination());

      // Get indices and sizes from template
      llvm::SmallVector<mlir::Value> source_indices;
      for (mlir::Value idx : template_transfer.getSourceIndices()) {
        source_indices.push_back(materializeValue(idx));
      }

      llvm::SmallVector<mlir::OpFoldResult> source_sizes;
      for (mlir::Value size : template_transfer.getSourceSizes()) {
        source_sizes.push_back(materializeValue(size));
      }

      llvm::SmallVector<mlir::Value> dest_indices;
      for (mlir::Value idx : template_transfer.getDestIndices()) {
        dest_indices.push_back(materializeValue(idx));
      }

      llvm::SmallVector<mlir::OpFoldResult> dest_sizes;
      for (mlir::Value size : template_transfer.getDestSizes()) {
        dest_sizes.push_back(materializeValue(size));
      }

      // Create the transfer operation with all required parameters
      mlir::ktdf::DataTransferOp::create(builder_, loc, source, source_indices,
                                         source_sizes, destination,
                                         dest_indices, dest_sizes);
    }
  }
}

}  // namespace scheduler

// Made with Bob