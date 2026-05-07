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
// Buffer Expansion Support for Stage Coarsening
//
// This file provides utilities for expanding buffers when sinking loops into
// pipeline stages. Buffer expansion is necessary when a buffer is shared
// across multiple stage groups and has dependency on loop-dependent accesses.
//===----------------------------------------------------------------------===//

#ifndef KTIR_SCHEDULER_TRANSFORM_BUFFER_EXPANSION_HPP
#define KTIR_SCHEDULER_TRANSFORM_BUFFER_EXPANSION_HPP

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Value.h"
#include "src/Dialects/KTDF/KTDFOps.hpp"
#include "src/Transform/PipelineTree/Materializer.hpp"

// Forward declaration
namespace ktdf {
class StageGroupingAnalysis;
}  // namespace ktdf

namespace scheduler {

//===----------------------------------------------------------------------===//
// BufferExpansionInfo
//===----------------------------------------------------------------------===//

/// Stores information about how to expand a buffer
struct BufferExpansionInfo {
  mlir::Value buffer;
  llvm::SmallVector<mlir::Value> loop_ivs;  // Ordered by loop nest
  llvm::SmallVector<mlir::Value>
      loop_lower_bounds;                      // Lower bounds for each loop
  llvm::SmallVector<mlir::Value> loop_steps;  // Step values for each loop
  llvm::SmallVector<mlir::OpFoldResult>
      sizes;  // Expansion sizes (constant or dynamic)
};

//===----------------------------------------------------------------------===//
// StageCoarseningBufferHandler
//===----------------------------------------------------------------------===//

/// Buffer handler for stage coarsening that expands buffers with additional
/// dimensions based on loop dependencies
class StageCoarseningBufferHandler : public scheduler::BufferHandler {
 public:
  /// Construct a buffer handler with expansion information
  explicit StageCoarseningBufferHandler(
      llvm::ArrayRef<const BufferExpansionInfo*> expansion_infos);

  /// Handle allocation operations - expand memref type with additional
  /// dimensions
  mlir::memref::AllocOp handleAllocOp(mlir::memref::AllocOp alloc_op,
                                      mlir::OpBuilder& builder,
                                      mlir::IRMapping& value_map) override;

  /// Handle data transfer operations - adjust indices for expanded dimensions
  mlir::ktdf::DataTransferOp handleDataTransfer(
      mlir::ktdf::DataTransferOp transfer_op, mlir::OpBuilder& builder,
      mlir::IRMapping& value_map) override;

 private:
  llvm::SmallVector<const BufferExpansionInfo*> expansion_infos_;
  llvm::DenseMap<mlir::Value, const BufferExpansionInfo*> buffer_to_info_;
};

//===----------------------------------------------------------------------===//
// Buffer Expansion Analysis
//===----------------------------------------------------------------------===//

/// Analyze buffers and determine which ones need expansion when sinking loops
/// into pipeline stages.
///
/// This function identifies buffers that:
/// - Are used across multiple stage groups (shared)
/// - Are written to in at least one stage
/// - Are defined in the private region of the parent pipeline
/// - Have loop-dependent accesses
///
/// For each candidate buffer, it computes the expansion dimensions and sizes
///
/// \param loop_nest The loop nest being sunk into the pipeline
/// \param pipeline The pipeline operation
/// \param grouping The stage grouping analysis
/// \param expansion_infos Output vector to store expansion information
void PerformBufferExpansionAnalysis(
    llvm::ArrayRef<mlir::scf::ForOp> loop_nest,
    mlir::ktdf::PipelineOp pipeline,
    const ktdf::StageGroupingAnalysis& grouping,
    llvm::SmallVectorImpl<BufferExpansionInfo*>& expansion_infos);

//===----------------------------------------------------------------------===//
// Helper Functions
//===----------------------------------------------------------------------===//

/// Check if a value depends on a target value through its defining operation
/// and operands (recursively)
bool isValueDependentOn(mlir::Value value, mlir::Value target);

}  // namespace scheduler

#endif  // KTIR_SCHEDULER_TRANSFORM_BUFFER_EXPANSION_HPP

// Made with Bob
