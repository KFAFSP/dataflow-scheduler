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
// Planner Factory Implementations
// - PrivateResourceFactory: Manages private resource specs
// - TransferInfoFactory: Manages transfer materialization info
//===----------------------------------------------------------------------===//

#include "PlannerFactories.hpp"

#include "Planner.hpp"
#include "mlir/IR/Builders.h"

namespace scheduler {

//===----------------------------------------------------------------------===//
// Helper function for deriving sizes from resource specs
//===----------------------------------------------------------------------===//

/// Helper to derive sizes from a private resource spec
/// Used to populate source_sizes or dest_sizes when they're empty
static void deriveSizesFromResourceSpec(
    const PrivateResourceSpec* spec, size_t slot_index,
    mlir::MLIRContext* context,
    llvm::SmallVectorImpl<mlir::OpFoldResult>& sizes) {
  if (!spec) return;

  mlir::OpBuilder builder(context);

  if (spec->kind == PrivateResourceSpec::Kind::kFifo) {
    // For FIFO, use the specific slot's capacity
    assert(slot_index < spec->elements_per_slot.size() &&
           "Slot index out of bounds");
    int64_t slot_capacity = spec->elements_per_slot[slot_index];
    sizes.push_back(builder.getI64IntegerAttr(slot_capacity));
  } else if (spec->kind == PrivateResourceSpec::Kind::kMemoryBuffer) {
    // For memory buffer, use the shape
    for (int64_t dim : spec->shape) {
      sizes.push_back(builder.getI64IntegerAttr(dim));
    }
  }
}

//===----------------------------------------------------------------------===//
// PrivateResourceFactory Implementation
//===----------------------------------------------------------------------===//

PrivateResourceSpec* PrivateResourceFactory::createMemoryBuffer(
    ArchitectureGraph::Resource memory_resource, llvm::ArrayRef<int64_t> shape,
    mlir::Type element_type) {
  auto spec = std::make_unique<PrivateResourceSpec>();
  spec->kind = PrivateResourceSpec::Kind::kMemoryBuffer;
  spec->memory_resource = memory_resource;
  spec->shape.assign(shape.begin(), shape.end());
  spec->element_type = element_type;

  PrivateResourceSpec* ptr = spec.get();
  specs_.push_back(std::move(spec));
  return ptr;
}

PrivateResourceSpec* PrivateResourceFactory::createFifo(
    mlir::ktdf::FifoType fifo_type, llvm::ArrayRef<int64_t> elements_per_slot,
    mlir::Type element_type) {
  auto spec = std::make_unique<PrivateResourceSpec>();
  spec->kind = PrivateResourceSpec::Kind::kFifo;
  spec->fifo_type = fifo_type;
  spec->elements_per_slot.assign(elements_per_slot.begin(),
                                 elements_per_slot.end());
  spec->element_type = element_type;

  PrivateResourceSpec* ptr = spec.get();
  specs_.push_back(std::move(spec));
  return ptr;
}

void PrivateResourceFactory::registerAllocation(
    const PrivateResourceSpec* spec, llvm::ArrayRef<mlir::Value> ssa_values) {
  auto alloc = std::make_unique<PrivateResourceAllocation>(spec);
  alloc->ssa_values.assign(ssa_values.begin(), ssa_values.end());
  allocations_[spec] = std::move(alloc);
}

const PrivateResourceAllocation* PrivateResourceFactory::getAllocation(
    const PrivateResourceSpec* spec) const {
  auto it = allocations_.find(spec);
  return it != allocations_.end() ? it->second.get() : nullptr;
}

//===----------------------------------------------------------------------===//
// TransferInfoFactory Implementation
//===----------------------------------------------------------------------===//

TransferMaterializationInfo* TransferInfoFactory::createFromTemplateWithBuffer(
    mlir::ktdf::DataTransferOp template_op,
    const ArchitectureGraph::EdgeInfo& edge,
    ArchitectureGraph::Resource intermediate_resource,
    ArchitectureGraph::Resource current_resource, bool intermediate_is_source,
    const PrivateResourceSpec* buffer_spec, mlir::OpBuilder& builder) {
  auto transfer = std::make_unique<TransferMaterializationInfo>();
  transfer->template_op = template_op;
  transfer->hop = edge;
  transfer->source_resource =
      intermediate_is_source ? intermediate_resource : current_resource;
  transfer->dest_resource =
      intermediate_is_source ? current_resource : intermediate_resource;

  // Set up indices and sizes based on which side is the intermediate buffer
  if (intermediate_is_source) {
    // Source is the intermediate buffer, dest is FIFO
    transfer->source_private_resource = buffer_spec;
    transfer->source_indices.append(template_op.getSourceIndices().begin(),
                                    template_op.getSourceIndices().end());
    // Source sizes: derive from buffer spec
    deriveSizesFromResourceSpec(buffer_spec, 0, builder.getContext(),
                                transfer->source_sizes);
    // Dest indices and sizes: copy from template (FIFO side)
    transfer->dest_indices.append(template_op.getDestIndices().begin(),
                                  template_op.getDestIndices().end());
    transfer->dest_sizes = template_op.getMixedDestSizes();
  } else {
    // Source is FIFO, dest is the intermediate buffer
    transfer->dest_private_resource = buffer_spec;
    transfer->source_indices.append(template_op.getSourceIndices().begin(),
                                    template_op.getSourceIndices().end());
    transfer->source_sizes = template_op.getMixedSourceSizes();
    // Dest indices: copy from template (memref side)
    transfer->dest_indices.append(template_op.getDestIndices().begin(),
                                  template_op.getDestIndices().end());
    // Dest sizes: derive from buffer spec
    deriveSizesFromResourceSpec(buffer_spec, 0, builder.getContext(),
                                transfer->dest_sizes);
  }

  TransferMaterializationInfo* result = transfer.get();
  transfers_.push_back(std::move(transfer));
  return result;
}

TransferMaterializationInfo* TransferInfoFactory::createFromFifoOp(
    mlir::Operation* fifo_op, const ArchitectureGraph::EdgeInfo& edge,
    ArchitectureGraph::Resource source_resource,
    ArchitectureGraph::Resource dest_resource,
    const PrivateResourceSpec* fifo_spec, size_t slot_index, bool is_read) {
  auto transfer = std::make_unique<TransferMaterializationInfo>();
  transfer->template_op = fifo_op;
  transfer->hop = edge;
  transfer->source_resource = source_resource;
  transfer->dest_resource = dest_resource;

  // Link the FIFO spec to the appropriate side
  if (is_read) {
    transfer->source_private_resource = fifo_spec;
    transfer->source_slot_index = slot_index;
  } else {
    transfer->dest_private_resource = fifo_spec;
    transfer->dest_slot_index = slot_index;
  }

  // Note: FIFO operations don't have indices and sizes - they work with tensors
  // The indices and sizes will remain empty for these operations

  TransferMaterializationInfo* result = transfer.get();
  transfers_.push_back(std::move(transfer));
  return result;
}

TransferMaterializationInfo* TransferInfoFactory::createSynthetic(
    const ArchitectureGraph::EdgeInfo& edge,
    ArchitectureGraph::Resource source_resource,
    ArchitectureGraph::Resource dest_resource,
    const PrivateResourceSpec* source_spec, size_t source_slot_index,
    llvm::ArrayRef<mlir::Value> source_indices,
    llvm::ArrayRef<mlir::OpFoldResult> source_sizes,
    const PrivateResourceSpec* dest_spec, size_t dest_slot_index,
    llvm::ArrayRef<mlir::Value> dest_indices,
    llvm::ArrayRef<mlir::OpFoldResult> dest_sizes, mlir::MLIRContext* context) {
  auto transfer = std::make_unique<TransferMaterializationInfo>();
  transfer->template_op = nullptr;  // Synthetic transfer, no template
  transfer->hop = edge;
  transfer->source_resource = source_resource;
  transfer->dest_resource = dest_resource;
  transfer->source_private_resource = source_spec;
  transfer->source_slot_index = source_slot_index;
  transfer->source_indices.append(source_indices.begin(), source_indices.end());
  transfer->source_sizes.append(source_sizes.begin(), source_sizes.end());
  transfer->dest_private_resource = dest_spec;
  transfer->dest_slot_index = dest_slot_index;
  transfer->dest_indices.append(dest_indices.begin(), dest_indices.end());
  transfer->dest_sizes.append(dest_sizes.begin(), dest_sizes.end());

  // Derive sizes from resource specs if empty
  if (transfer->dest_sizes.empty() && dest_spec) {
    deriveSizesFromResourceSpec(dest_spec, dest_slot_index, context,
                                transfer->dest_sizes);
  }

  if (transfer->source_sizes.empty() && source_spec) {
    deriveSizesFromResourceSpec(source_spec, source_slot_index, context,
                                transfer->source_sizes);
  }

  TransferMaterializationInfo* result = transfer.get();
  transfers_.push_back(std::move(transfer));
  return result;
}

}  // namespace scheduler

// Made with Bob