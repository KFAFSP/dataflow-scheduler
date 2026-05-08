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
// Buffer Expansion Implementation
//
// We need to expand buffers to make loop distribution legal. This
// implementation helps with identifying the buffers that need expansion and
// recording relevant information for the materializer to use. It identifies
// which buffers are candidate for expansion as follows:
//   1. Use the result of the stage grouping analysis and look for common
//   memrefs across groups of stages. Consider a memref as candidate for
//   expansion if it is written to in any of the stages under this pipeline.
//   2. Further check to see if the candidate is allocated in the private region
//   of the parent pipeline, otherwise ignore it.
//   3. Collect the loop variables on which the candidate memref has
//   dependency (see explaination below)...only collect loops that are in the
//   nest of candidates being distributed.
// Determine how to expand the buffers (ie what dimensions to add, what sizes,
// etc) and record relevant information
//   1. Expand the memref only in dimensions corresponding to the loop's
//   collected (ie do not expand in dimensions that are loop invariant)... also
//   check if the memref already contains those dims assert/skip them.
//   2: Expand the memref in dims to the left of its current buffer and in the
//   same order as the candidate loop nest.
//
// Where the rules above come from? Consider this example:
//
// C[N] = A + B[M]
//
// S0:  for n...N
// S1:    t1 = load A
// S2:    t2 = load B[n]
// S3:    t3 = t1 + t2
// S4:    store t3, C[n]
//
// Suppose we want to distribute to {[S1,S2],S3,S4}. The questions are:
// 1. which of t1, t2, or t3 need expansion?
// 2. in what dimensions to expand so that we preserve loop invariance for reuse
// promotion opportunity in later passes.
//
// Focusing on S1, we notice that all the n iterations are using the same value
// of A (ie A's subscripts (can think of [0]) are invariant to loop-N). Each
// iteration of N reads the same A into t1, so t1 need not be expanded to enable
// loop distribution. Focusing on S2, the load of B[n] is dependent on loop-N,
// so t2 receives different data in each iteration of loop-N. To enable
// distribution t2 needs to be expanded in the loop-N dimension. Similarly, t3
// needs to be expanded in the N dimension:
//
// S1':  t1 = load A
// S2':  t2[n] = load B[n]
// S3':  t3[n] = t1 + t2[n]
// S4':  store t3[n], C[n]
//
// Without above rules, a naive transformation would have unnecessarily expanded
// t1 to t[n] as well, making it difficult to recover the invariance.
//
//===----------------------------------------------------------------------===//

#include "BufferExpansion.hpp"

#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "src/Analysis/StageGrouping.hpp"
#include "src/Utils/OperationUtils.hpp"

#define DEBUG_TYPE "buffer-expansion"

namespace scheduler {

//===----------------------------------------------------------------------===//
// StageCoarseningBufferCloner Implementation
//===----------------------------------------------------------------------===//

StageCoarseningBufferCloner::StageCoarseningBufferCloner(
    llvm::ArrayRef<const BufferExpansionInfo*> expansion_infos)
    : expansion_infos_(expansion_infos.begin(), expansion_infos.end()) {
  // Build a map from original buffer to expansion info for quick lookup
  for (const BufferExpansionInfo* info : expansion_infos_) {
    assert(info);
    buffer_to_info_[info->buffer] = info;
  }
}

mlir::memref::AllocOp StageCoarseningBufferCloner::cloneAllocOp(
    mlir::memref::AllocOp alloc_op, mlir::OpBuilder& builder,
    mlir::IRMapping& value_map) {
  // Get the result of the allocation
  mlir::Value original_result = alloc_op.getResult();

  // Check if this buffer needs expansion
  // The buffer might be identified by its yielded value from ktdf.private,
  // so we need to check if this allocation is yielded and matches a candidate
  const BufferExpansionInfo* info = nullptr;
  auto it = buffer_to_info_.find(original_result);
  if (it != buffer_to_info_.end()) {
    info = it->second;
  } else {
    // Check if this allocation is yielded by a private operation
    // and the yielded value is a candidate for expansion
    for (mlir::Operation* user : original_result.getUsers()) {
      auto yield_op = mlir::dyn_cast<mlir::ktdf::PrivateYieldOp>(user);
      if (!yield_op) continue;
      // Find which operand index this is
      for (auto [idx, operand] : llvm::enumerate(yield_op.getOperands())) {
        if (operand != original_result) continue;
        // Get the corresponding result from the parent private operation
        auto private_op =
            mlir::cast<mlir::ktdf::PrivateOp>(yield_op->getParentOp());
        mlir::Value yielded_result = private_op.getResult(idx);

        // Check if this yielded result is a candidate
        auto yielded_it = buffer_to_info_.find(yielded_result);
        if (yielded_it != buffer_to_info_.end()) {
          info = yielded_it->second;
          break;
        }
      }
      if (info) break;
    }
  }

  if (!info) {
    // No expansion needed, use default behavior
    return nullptr;
  }

  auto original_type = alloc_op.getType();

  LLVM_DEBUG({
    llvm::dbgs() << "  Expanding buffer allocation with " << info->sizes.size()
                 << " new dimension(s)\n";
  });

  // Build new shape: [expansion_dims..., original_dims...]
  llvm::SmallVector<int64_t> new_shape;
  llvm::SmallVector<mlir::Value> dynamic_sizes;

  // Add expansion dimensions (to the left)
  for (const mlir::OpFoldResult& size : info->sizes) {
    if (auto attr = mlir::dyn_cast<mlir::Attribute>(size)) {
      // Constant size
      auto int_attr = mlir::cast<mlir::IntegerAttr>(attr);
      new_shape.push_back(int_attr.getInt());
    } else {
      // Dynamic size
      new_shape.push_back(mlir::ShapedType::kDynamic);
      mlir::Value size_val = mlir::cast<mlir::Value>(size);
      // Map the size value if it's been cloned
      if (value_map.contains(size_val)) {
        size_val = value_map.lookup(size_val);
      }
      dynamic_sizes.push_back(size_val);
    }
  }

  // Add original dimensions
  for (int64_t dim : original_type.getShape()) {
    new_shape.push_back(dim);
    if (!mlir::ShapedType::isDynamic(dim)) continue;
    // Original dimension was dynamic - need to get its size
    // For memref.alloc, dynamic sizes are operands
    // Find the corresponding dynamic size operand
    unsigned dynamic_idx = 0;
    for (int64_t orig_dim : original_type.getShape()) {
      if (!mlir::ShapedType::isDynamic(orig_dim)) continue;
      if (dynamic_idx < alloc_op.getDynamicSizes().size()) {
        mlir::Value orig_size = alloc_op.getDynamicSizes()[dynamic_idx];
        if (value_map.contains(orig_size)) {
          orig_size = value_map.lookup(orig_size);
        }
        dynamic_sizes.push_back(orig_size);
      }
      dynamic_idx++;
    }
  }

  // Create new memref type with expanded dimensions
  mlir::MemRefType new_type = mlir::MemRefType::get(
      new_shape, original_type.getElementType(),
      mlir::MemRefLayoutAttrInterface(), original_type.getMemorySpace());

  // Create new allocation with expanded type
  auto new_alloc = mlir::memref::AllocOp::create(builder, alloc_op.getLoc(),
                                                 new_type, dynamic_sizes);

  LLVM_DEBUG({
    llvm::dbgs() << "    Original type: " << original_type << "\n";
    llvm::dbgs() << "    Expanded type: " << new_type << "\n";
  });

  return new_alloc;
}

mlir::ktdf::DataTransferOp StageCoarseningBufferCloner::cloneDataTransfer(
    mlir::ktdf::DataTransferOp transfer_op, mlir::OpBuilder& builder,
    mlir::IRMapping& value_map) {
  // Get source and destination buffers
  mlir::Value src = transfer_op.getSource();
  mlir::Value dst = transfer_op.getDestination();

  // Map to cloned values
  if (value_map.contains(src)) {
    src = value_map.lookup(src);
  }
  if (value_map.contains(dst)) {
    dst = value_map.lookup(dst);
  }

  // Check if either buffer was expanded
  const BufferExpansionInfo* src_info = nullptr;
  const BufferExpansionInfo* dst_info = nullptr;

  for (const BufferExpansionInfo* info : expansion_infos_) {
    mlir::Value mapped_buffer = info->buffer;
    if (value_map.contains(mapped_buffer)) {
      mapped_buffer = value_map.lookup(mapped_buffer);
    }
    if (mapped_buffer == src) {
      src_info = info;
    }
    if (mapped_buffer == dst) {
      dst_info = info;
    }
  }

  if (!src_info && !dst_info) {
    // No expansion needed
    return nullptr;
  }

  LLVM_DEBUG({
    llvm::dbgs()
        << "  Adjusting data transfer indices for expanded buffer(s)\n";
  });

  // Build new index lists with loop IV values prepended
  llvm::SmallVector<mlir::Value> new_src_indices;
  llvm::SmallVector<mlir::Value> new_dst_indices;

  // Lambda to compute normalized indices for expanded dimensions
  auto computeExpandedIndices = [&](const BufferExpansionInfo* info,
                                    llvm::SmallVector<mlir::Value>& indices) {
    if (!info) return;

    // Prepend loop IV values for expanded dimensions
    // Need to compute normalized index: (iv - lower_bound) / step
    for (size_t i = 0; i < info->loop_ivs.size(); ++i) {
      mlir::Value loop_iv = info->loop_ivs[i];
      mlir::Value lower_bound = info->loop_lower_bounds[i];
      mlir::Value step = info->loop_steps[i];

      // Map values if they've been cloned
      if (value_map.contains(loop_iv)) {
        loop_iv = value_map.lookup(loop_iv);
      }
      if (value_map.contains(lower_bound)) {
        lower_bound = value_map.lookup(lower_bound);
      }
      if (value_map.contains(step)) {
        step = value_map.lookup(step);
      }

      // Compute normalized index: (iv - lower_bound) / step
      mlir::Location loc = transfer_op.getLoc();
      mlir::Value offset =
          mlir::arith::SubIOp::create(builder, loc, loop_iv, lower_bound)
              .getResult();
      mlir::Value normalized_idx =
          mlir::arith::DivSIOp::create(builder, loc, offset, step).getResult();

      indices.push_back(normalized_idx);
    }
  };

  // Compute expanded indices for source
  computeExpandedIndices(src_info, new_src_indices);

  // Add original source indices
  for (mlir::Value idx : transfer_op.getSourceIndices()) {
    mlir::Value mapped_idx = idx;
    if (value_map.contains(idx)) {
      mapped_idx = value_map.lookup(idx);
    }
    new_src_indices.push_back(mapped_idx);
  }

  // Compute expanded indices for destination
  computeExpandedIndices(dst_info, new_dst_indices);

  // Add original destination indices
  for (mlir::Value idx : transfer_op.getDestIndices()) {
    mlir::Value mapped_idx = idx;
    if (value_map.contains(idx)) {
      mapped_idx = value_map.lookup(idx);
    }
    new_dst_indices.push_back(mapped_idx);
  }

  // Create new data transfer with adjusted indices
  // Get the mixed sizes (OpFoldResult) from the original transfer
  llvm::SmallVector<mlir::OpFoldResult> src_sizes =
      transfer_op.getMixedSourceSizes();
  llvm::SmallVector<mlir::OpFoldResult> dst_sizes =
      transfer_op.getMixedDestSizes();

  auto new_transfer = mlir::ktdf::DataTransferOp::create(
      builder, transfer_op.getLoc(), src, new_src_indices, src_sizes, dst,
      new_dst_indices, dst_sizes);

  return new_transfer;
}

//===----------------------------------------------------------------------===//
// Buffer Expansion Analysis
//===----------------------------------------------------------------------===//

void PerformBufferExpansionAnalysis(
    llvm::ArrayRef<mlir::scf::ForOp> loop_nest, mlir::ktdf::PipelineOp pipeline,
    const ktdf::StageGroupingAnalysis& grouping,
    llvm::SmallVectorImpl<BufferExpansionInfo*>& expansion_infos) {
  LLVM_DEBUG(llvm::dbgs() << "  Performing buffer expansion analysis...\n");

  // Step 1: Identify common memrefs across stage groups
  // Map from memref value to the set of group indices that use it
  llvm::DenseMap<mlir::Value, llvm::SmallSet<size_t, 4>> memref_to_groups;

  const llvm::SmallVector<ktdf::StageGroup>& groups = grouping.getGroups();

  for (size_t group_idx = 0; group_idx < groups.size(); ++group_idx) {
    const ktdf::StageGroup& group = groups[group_idx];

    for (mlir::ktdf::StageOp stage : group.getStages()) {
      // Walk the stage body to find all memref uses
      stage.walk([&](mlir::Operation* op) {
        for (mlir::Value operand : op->getOperands()) {
          if (mlir::isa<mlir::MemRefType>(operand.getType())) {
            memref_to_groups[operand].insert(group_idx);
          }
        }
      });
    }
  }

  LLVM_DEBUG({
    llvm::dbgs() << "    Found " << memref_to_groups.size()
                 << " unique memrefs across all stages\n";
  });

  // Step 2: Filter for buffers that are:
  // a) Used across multiple groups (shared)
  // b) Written to in at least one stage
  // c) Defined in the private region of the parent pipeline
  llvm::SmallVector<mlir::Value> candidate_buffers;

  for (const auto& entry : memref_to_groups) {
    mlir::Value memref = entry.first;
    const llvm::SmallSet<size_t, 4>& group_set = entry.second;

    // Check if used across multiple groups
    if (group_set.size() <= 1) {
      continue;
    }

    // Check if written to in any stage
    bool is_written = false;
    for (size_t group_idx : group_set) {
      const ktdf::StageGroup& group = groups[group_idx];
      for (mlir::ktdf::StageOp stage : group.getStages()) {
        if (scheduler::utils::isMemrefWrittenTo(memref, stage.getOperation())) {
          is_written = true;
          break;
        }
      }
      if (is_written) break;
    }

    if (!is_written) {
      continue;
    }

    // Check if allocated in private region
    auto defining_op = memref.getDefiningOp();
    if (!defining_op) {
      continue;
    }

    // Check if it's coming from a ktdf.private operation
    if (!mlir::isa<mlir::ktdf::PrivateOp>(defining_op)) {
      continue;
    }

    // This is a valid candidate
    if (std::find(candidate_buffers.begin(), candidate_buffers.end(), memref) ==
        candidate_buffers.end())
      candidate_buffers.push_back(memref);

    LLVM_DEBUG({
      llvm::dbgs() << "    Candidate buffer found: used in " << group_set.size()
                   << " groups\n";
    });
  }

  LLVM_DEBUG({
    llvm::dbgs() << "    Identified " << candidate_buffers.size()
                 << " candidate buffers for expansion\n";
  });

  // Step 3: Analyze loop dependencies for each candidate buffer
  // Map from buffer to the list of loop IVs it depends on
  llvm::DenseMap<mlir::Value, llvm::SmallVector<mlir::Value>>
      buffer_to_loop_ivs;

  // Build set of loop IVs from candidate loop nest for quick lookup
  llvm::SmallVector<mlir::Value> candidate_loop_ivs;
  for (mlir::scf::ForOp loop : loop_nest) {
    candidate_loop_ivs.push_back(loop.getInductionVar());
  }

  for (mlir::Value buffer : candidate_buffers) {
    llvm::SmallVector<mlir::Value> dependent_ivs;

    // Find all data transfer operations that use this buffer
    pipeline.walk([&](mlir::ktdf::DataTransferOp transfer) {
      // Check if this transfer uses the buffer
      if (transfer.getSource() != buffer &&
          transfer.getDestination() != buffer) {
        return;
      }

      // Analyze the indices used in the transfer
      // For source indices
      for (mlir::Value idx : transfer.getSourceIndices()) {
        // Check if this index depends on any candidate loop IV
        for (mlir::Value loop_iv : candidate_loop_ivs) {
          if (isValueDependentOn(idx, loop_iv)) {
            // Add to dependent IVs if not already present
            if (llvm::find(dependent_ivs, loop_iv) == dependent_ivs.end()) {
              dependent_ivs.push_back(loop_iv);
            }
          }
        }
      }

      // For destination indices
      for (mlir::Value idx : transfer.getDestIndices()) {
        for (mlir::Value loop_iv : candidate_loop_ivs) {
          if (isValueDependentOn(idx, loop_iv)) {
            if (llvm::find(dependent_ivs, loop_iv) == dependent_ivs.end()) {
              dependent_ivs.push_back(loop_iv);
            }
          }
        }
      }
    });

    if (!dependent_ivs.empty()) {
      buffer_to_loop_ivs[buffer] = dependent_ivs;
      LLVM_DEBUG({
        llvm::dbgs() << "    Buffer depends on " << dependent_ivs.size()
                     << " loop IV(s)\n";
      });
    }
  }

  // Step 4: Calculate expansion dimensions and sizes

  for (const auto& entry : buffer_to_loop_ivs) {
    mlir::Value buffer = entry.first;
    const llvm::SmallVector<mlir::Value>& dependent_ivs = entry.second;

    BufferExpansionInfo* info = new BufferExpansionInfo();
    info->buffer = buffer;

    // Order the dependent IVs according to the loop nest order
    for (mlir::scf::ForOp loop : loop_nest) {
      mlir::Value loop_iv = loop.getInductionVar();
      if (llvm::find(dependent_ivs, loop_iv) == dependent_ivs.end()) continue;
      info->loop_ivs.push_back(loop_iv);

      // Calculate the size for this dimension
      // Size = (upper_bound - lower_bound + step - 1) / step
      mlir::Value lower = loop.getLowerBound();
      mlir::Value upper = loop.getUpperBound();
      mlir::Value step = loop.getStep();

      // Store loop bounds and step for later index computation
      info->loop_lower_bounds.push_back(lower);
      info->loop_steps.push_back(step);

      // Try to compute constant size if possible
      std::optional<int64_t> lower_const, upper_const, step_const;
      if (auto cst = lower.getDefiningOp<mlir::arith::ConstantIndexOp>()) {
        lower_const = cst.value();
      }
      if (auto cst = upper.getDefiningOp<mlir::arith::ConstantIndexOp>()) {
        upper_const = cst.value();
      }
      if (auto cst = step.getDefiningOp<mlir::arith::ConstantIndexOp>()) {
        step_const = cst.value();
      }

      if (lower_const && upper_const && step_const) {
        // Compute constant trip count
        int64_t trip_count =
            (*upper_const - *lower_const + *step_const - 1) / *step_const;
        mlir::OpBuilder builder(loop);
        mlir::IntegerAttr size_attr = builder.getI64IntegerAttr(trip_count);
        info->sizes.push_back(mlir::OpFoldResult(size_attr));
        LLVM_DEBUG({
          llvm::dbgs() << "      Dimension size (constant): " << trip_count
                       << "\n";
        });
        continue;
      }

      // Dynamic size - compute at runtime
      // Create: ceildiv(upper - lower, step)
      mlir::OpBuilder builder(loop);
      mlir::Location loc = loop.getLoc();
      mlir::Value diff =
          mlir::arith::SubIOp::create(builder, loc, upper, lower).getResult();
      mlir::Value size_val =
          mlir::arith::CeilDivSIOp::create(builder, loc, diff, step)
              .getResult();
      info->sizes.push_back(mlir::OpFoldResult(size_val));
      LLVM_DEBUG({ llvm::dbgs() << "      Dimension size (dynamic)\n"; });
    }

    if (!info->loop_ivs.empty()) {
      LLVM_DEBUG({
        llvm::dbgs() << "    Buffer expansion: " << info->loop_ivs.size()
                     << " new dimension(s)\n";
      });
      
      // Assert that all users of this buffer are data transfer operations.
      // If other operations use a buffer that is a candidate for expansion,
      // we would likely require custom cloners for them as well.
      for (mlir::Operation* user : buffer.getUsers()) {
        assert(mlir::isa<mlir::ktdf::DataTransferOp>(user) &&
               "Expected all users of buffers marked for expansion to be "
               "ktdf::DataTransferOp.");
      }
      
      expansion_infos.push_back(info);
    } else {
      delete info;
    }
  }

  if (expansion_infos.empty()) {
    LLVM_DEBUG(llvm::dbgs() << "    No buffers need expansion\n");
  } else {
    LLVM_DEBUG({
      llvm::dbgs() << "    Total buffers to expand: " << expansion_infos.size()
                   << "\n";
    });
  }
}

//===----------------------------------------------------------------------===//
// Helper Functions
//===----------------------------------------------------------------------===//

bool isValueDependentOn(mlir::Value value, mlir::Value target) {
  if (value == target) {
    return true;
  }

  // Check if value is defined by an operation
  mlir::Operation* defining_op = value.getDefiningOp();
  if (!defining_op) {
    return false;
  }

  // Recursively check operands
  for (mlir::Value operand : defining_op->getOperands()) {
    if (isValueDependentOn(operand, target)) {
      return true;
    }
  }

  return false;
}

}  // namespace scheduler

// Made with Bob
