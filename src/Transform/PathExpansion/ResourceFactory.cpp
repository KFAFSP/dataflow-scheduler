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
// Private Resource Factory Implementation
//===----------------------------------------------------------------------===//

#include "Planner.hpp"

namespace scheduler {

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

}  // namespace scheduler

// Made with Bob