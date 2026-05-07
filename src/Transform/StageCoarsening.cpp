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
// Stage Coarsening Transformation
//
// This pass implements a stage-coarsening algorithm using the following steps:
//
// 1: Identify stage groups using loop distribution legality rules.
// 2: Construct a PipelineTree to represent the currect structure of the code
// and transform it to the desired structure (this step does not modify IR)
//   a. Distribute and sink loops into pipeline.
//   b. Further sink into stage groups containing a single stage (expand
//      buffers).
//   c. Discover and correct illegal structures iteratively.
//   d. Correct stage dependencies that cross pipeline boundaries.
// 3: Use the manipulated PipelineTree abstraction with reference to
//    operations in the input IR to generate the transformed IR.
//
// Note: Legality of stage coarsening follows loop distribution rules.
//       Data dependence analysis may be added in future.
//===----------------------------------------------------------------------===//

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "src/Analysis/PipelineTree.hpp"
#include "src/Analysis/StageGrouping.hpp"
#include "src/Dialects/KTDF/KTDFOps.hpp"
#include "src/Transform/Passes.h"
#include "src/Transform/PipelineTree/Legalizer.hpp"
#include "src/Transform/PipelineTree/Materializer.hpp"
#include "src/Utils/OperationUtils.hpp"

namespace ktdf {

#define GEN_PASS_DEF_STAGECOARSENINGPASS
#include "src/Transform/Passes.h.inc"

#define DEBUG_TYPE "stage-coarsening"

namespace {

//===----------------------------------------------------------------------===//
// Buffer Expansion Support
//===----------------------------------------------------------------------===//

/// Stores information about how to expand a buffer
struct BufferExpansionInfo {
  mlir::Value buffer;
  llvm::SmallVector<mlir::Value> loop_ivs;  // Ordered by loop nest
  llvm::SmallVector<mlir::Value> loop_lower_bounds;  // Lower bounds for each loop
  llvm::SmallVector<mlir::Value> loop_steps;  // Step values for each loop
  llvm::SmallVector<mlir::OpFoldResult>
      sizes;  // Expansion sizes (constant or dynamic)
};

//===----------------------------------------------------------------------===//
// Stage Coarsening Pass
//===----------------------------------------------------------------------===//

/// Module pass that implements the algorithm for creating hierarchical
/// pipelines
struct StageCoarseningPass
    : public impl::StageCoarseningPassBase<StageCoarseningPass> {
  using impl::StageCoarseningPassBase<
      StageCoarseningPass>::StageCoarseningPassBase;

  void runOnOperation() override;

 private:
  /// Represents a candidate for stage coarsening transformation
  struct TransformCandidate {
    llvm::SmallVector<mlir::scf::ForOp>
        loop_nest_;  // Perfectly nested loops to sink (outermost to innermost)
    mlir::ktdf::PipelineOp pipeline_;    // Target pipeline
    mlir::Operation* loop_nest_parent_;  // Parent operation of the outermost
                                         // loop (will be tree root)
  };

  /// Identify all transform candidates in the module
  /// Walks IR to find pipelines with 'n' perfectly nested loops above them
  void identifyTransformCandidates(
      mlir::ModuleOp module_op,
      llvm::SmallVector<TransformCandidate>& candidates);

  /// Apply the transformation algorithm to a candidate
  void applyTransformation(TransformCandidate& candidate);

  //===--------------------------------------------------------------------===//
  // Distribute and Sink Loops into Pipeline
  //===--------------------------------------------------------------------===//

  void DistributeAndSinkIntoPipeline(
      TransformCandidate& candidate, PipelineTree& tree,
      const StageGroupingAnalysis& grouping,
      llvm::SmallVectorImpl<BufferExpansionInfo>& expansion_infos);

  /// Perform buffer expansion analysis and record expansion information
  void PerformBufferExpansion(
      TransformCandidate& candidate, PipelineTree& tree,
      const StageGroupingAnalysis& grouping,
      llvm::SmallVectorImpl<BufferExpansionInfo>& expansion_infos);

  //===--------------------------------------------------------------------===//
  // Sink into Stage Groups Containing a Single Stage
  //===--------------------------------------------------------------------===//

  void SinkIntoSingleStage(TransformCandidate& candidate, PipelineTree& tree,
                           const StageGroupingAnalysis& grouping);

  //===--------------------------------------------------------------------===//
  // Discover and Correct Illegal Structures
  //===--------------------------------------------------------------------===//

  void DiscoverAndCorrectStructure(TransformCandidate& candidate,
                                   PipelineTree& tree);

  //===--------------------------------------------------------------------===//
  // Correct Stage Dependencies Crossing Pipeline Boundaries
  //===--------------------------------------------------------------------===//

  void CorrectStageDependencies(TransformCandidate& candidate,
                                PipelineTree& tree);

  //===--------------------------------------------------------------------===//
  // Step 3: Generate Transformed IR from PipelineTree
  //===--------------------------------------------------------------------===//

  /// Generate the transformed IR from the manipulated PipelineTree
  void GenerateTransformedIR(
      TransformCandidate& candidate, PipelineTree& tree,
      llvm::ArrayRef<BufferExpansionInfo> expansion_infos);
};

//===----------------------------------------------------------------------===//
// Buffer Expansion Handler
//===----------------------------------------------------------------------===//

// Forward declaration
static bool isValueDependentOn(mlir::Value value, mlir::Value target);

/// Handler for buffer expansion during IR materialization
class StageCoarseningBufferHandler : public scheduler::BufferHandler {
 public:
  explicit StageCoarseningBufferHandler(
      llvm::ArrayRef<BufferExpansionInfo> expansion_infos)
      : expansion_infos_(expansion_infos.begin(), expansion_infos.end()) {
    // Build a map from original buffer to expansion info for quick lookup
    for (const BufferExpansionInfo& info : expansion_infos_) {
      buffer_to_info_[info.buffer] = &info;
    }
  }

  /// Handle allocation operations - expand memref type with additional
  /// dimensions
  mlir::memref::AllocOp handleAllocOp(mlir::memref::AllocOp alloc_op,
                                      mlir::OpBuilder& builder,
                                      mlir::IRMapping& value_map) override {
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
      llvm::dbgs() << "  Expanding buffer allocation with "
                   << info->sizes.size() << " new dimension(s)\n";
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

  /// Handle data transfer operations - adjust indices for expanded dimensions
  mlir::ktdf::DataTransferOp handleDataTransfer(
      mlir::ktdf::DataTransferOp transfer_op, mlir::OpBuilder& builder,
      mlir::IRMapping& value_map) override {
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

    for (const BufferExpansionInfo& info : expansion_infos_) {
      mlir::Value mapped_buffer = info.buffer;
      if (value_map.contains(mapped_buffer)) {
        mapped_buffer = value_map.lookup(mapped_buffer);
      }
      if (mapped_buffer == src) {
        src_info = &info;
      }
      if (mapped_buffer == dst) {
        dst_info = &info;
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

    if (src_info) {
      // Prepend loop IV values for expanded dimensions
      // Need to compute normalized index: (iv - lower_bound) / step
      for (size_t i = 0; i < src_info->loop_ivs.size(); ++i) {
        mlir::Value loop_iv = src_info->loop_ivs[i];
        mlir::Value lower_bound = src_info->loop_lower_bounds[i];
        mlir::Value step = src_info->loop_steps[i];
        
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
        mlir::Value offset = mlir::arith::SubIOp::create(
            builder, loc, loop_iv, lower_bound).getResult();
        mlir::Value normalized_idx = mlir::arith::DivSIOp::create(
            builder, loc, offset, step).getResult();
        
        new_src_indices.push_back(normalized_idx);
      }
    }

    // Add original source indices
    for (mlir::Value idx : transfer_op.getSourceIndices()) {
      mlir::Value mapped_idx = idx;
      if (value_map.contains(idx)) {
        mapped_idx = value_map.lookup(idx);
      }
      new_src_indices.push_back(mapped_idx);
    }

    if (dst_info) {
      // Prepend loop IV values for expanded dimensions
      // Need to compute normalized index: (iv - lower_bound) / step
      for (size_t i = 0; i < dst_info->loop_ivs.size(); ++i) {
        mlir::Value loop_iv = dst_info->loop_ivs[i];
        mlir::Value lower_bound = dst_info->loop_lower_bounds[i];
        mlir::Value step = dst_info->loop_steps[i];
        
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
        mlir::Value offset = mlir::arith::SubIOp::create(
            builder, loc, loop_iv, lower_bound).getResult();
        mlir::Value normalized_idx = mlir::arith::DivSIOp::create(
            builder, loc, offset, step).getResult();
        
        new_dst_indices.push_back(normalized_idx);
      }
    }

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

 private:
  llvm::SmallVector<BufferExpansionInfo> expansion_infos_;
  llvm::DenseMap<mlir::Value, const BufferExpansionInfo*> buffer_to_info_;
};

//===----------------------------------------------------------------------===//
// StageCoarseningPass Implementation
//===----------------------------------------------------------------------===//

void StageCoarseningPass::runOnOperation() {
  auto module_op = getOperation();

  LLVM_DEBUG({
    llvm::dbgs() << "\n=== Stage Coarsening Pass ===\n";
    llvm::dbgs() << "Loop nest depth: " << loopNestDepth << "\n\n";
  });

  // Identify all transform candidates
  llvm::SmallVector<TransformCandidate> candidates;
  identifyTransformCandidates(module_op, candidates);

  if (candidates.empty()) {
    LLVM_DEBUG(llvm::dbgs() << "No transform candidates found.\n");
    return;
  }

  LLVM_DEBUG(llvm::dbgs() << "Found " << candidates.size()
                          << " transform candidate(s)\n\n");

  // Apply transformation to each candidate
  for (TransformCandidate& candidate : candidates) {
    applyTransformation(candidate);
  }

  LLVM_DEBUG(llvm::dbgs() << "=== Stage Coarsening Pass Complete ===\n\n");
}

void StageCoarseningPass::identifyTransformCandidates(
    mlir::ModuleOp module_op,
    llvm::SmallVector<TransformCandidate>& candidates) {
  // Walk all functions in the module
  module_op.walk([&](mlir::func::FuncOp func_op) {
    LLVM_DEBUG(llvm::dbgs()
               << "Analyzing function: " << func_op.getName() << "\n");

    // Find all pipelines in this function
    func_op.walk([&](mlir::ktdf::PipelineOp pipeline) {
      LLVM_DEBUG(llvm::dbgs()
                 << "  Found pipeline, checking for loop nest above it...\n");

      // Walk up the parent chain to find perfectly nested loops
      mlir::Operation* current_op = pipeline.getOperation();
      llvm::SmallVector<mlir::scf::ForOp> loop_nest;

      // Collect up to 'loopNestDepth' perfectly nested loops
      for (unsigned i = 0; i < loopNestDepth; ++i) {
        mlir::Operation* parent_op = current_op->getParentOp();
        if (!parent_op) break;

        if (auto forOp = mlir::dyn_cast<mlir::scf::ForOp>(parent_op)) {
          // Check if this is a perfectly nested loop (single child in body)
          auto& body_ops = forOp.getBody()->getOperations();
          if (body_ops.size() == 2) {  // Loop body + terminator
            loop_nest.push_back(forOp);
            current_op = forOp.getOperation();
            continue;
          }
        }
        break;
      }

      // If we found any loops, this is a candidate
      if (!loop_nest.empty()) {
        // Reverse to get outermost-to-innermost order
        std::reverse(loop_nest.begin(), loop_nest.end());

        // Parent of outermost loop becomes tree root
        mlir::Operation* loop_nest_parent =
            loop_nest.empty() ? pipeline.getOperation()->getParentOp()
                              : loop_nest.front().getOperation()->getParentOp();

        candidates.push_back({loop_nest, pipeline, loop_nest_parent});

        LLVM_DEBUG({
          llvm::dbgs() << "    -> Found candidate with " << loop_nest.size()
                       << " nested loop(s) at ";
          scheduler::utils::printLocation(llvm::dbgs(),
                                          loop_nest.front().getOperation());
          llvm::dbgs() << "\n";
        });
      }
    });
  });
}

void StageCoarseningPass::applyTransformation(TransformCandidate& candidate) {
  LLVM_DEBUG({
    llvm::dbgs() << "\n--- Applying Transformation in steps ---\n";
    llvm::dbgs() << "Loop nest depth: " << candidate.loop_nest_.size() << "\n";
  });

  // Build pipeline tree once - it will be manipulated throughout the
  // transformation
  PipelineTree tree;
  tree.compute(*candidate.loop_nest_parent_);
  LLVM_DEBUG({
    llvm::dbgs() << "=== Initial Pipeline Tree ===\n";
    tree.print(llvm::dbgs());
    llvm::dbgs() << "\n";
  });

  // Perform stage grouping analysis once
  LLVM_DEBUG(llvm::dbgs() << "\n  Performing stage grouping analysis...\n");
  StageGroupingAnalysis grouping(candidate.pipeline_);
  LLVM_DEBUG(grouping.print(llvm::dbgs()));

  // Step 2a: Distribute and sink loops into pipeline
  LLVM_DEBUG(
      llvm::dbgs() << "\n--- Step 2a: Distribute and Sink into Pipeline ---\n");
  llvm::SmallVector<BufferExpansionInfo> buffer_expansion_infos;
  DistributeAndSinkIntoPipeline(candidate, tree, grouping,
                                buffer_expansion_infos);
  LLVM_DEBUG({
    llvm::dbgs() << "\n=== Pipeline Tree after Step 2a ===\n";
    tree.print(llvm::dbgs());
    llvm::dbgs() << "\n";
  });

  // Step 2b: Further sink into stage groups containing a single stage
  // (expand buffers)
  LLVM_DEBUG(
      llvm::dbgs() << "\n--- Step 2b: Sink into Single-Stage Groups ---\n");
  SinkIntoSingleStage(candidate, tree, grouping);
  LLVM_DEBUG({
    llvm::dbgs() << "\n=== Pipeline Tree after Step 2b ===\n";
    tree.print(llvm::dbgs());
    llvm::dbgs() << "\n";
  });

  // Step 2c: Discover and correct illegal structures iteratively
  LLVM_DEBUG(llvm::dbgs()
             << "\n--- Step 2c: Discover and Correct Illegal Structures ---\n");
  DiscoverAndCorrectStructure(candidate, tree);
  LLVM_DEBUG({
    llvm::dbgs() << "\n=== Pipeline Tree after Step 2c ===\n";
    tree.print(llvm::dbgs());
    llvm::dbgs() << "\n";
  });

  // Step 2d: Correct stage dependencies that cross pipeline boundaries
  LLVM_DEBUG(llvm::dbgs() << "\n--- Step 2d: Correct Stage Dependencies ---\n");
  CorrectStageDependencies(candidate, tree);
  LLVM_DEBUG({
    llvm::dbgs() << "\n=== Pipeline Tree after Step 2d (Final) ===\n";
    tree.print(llvm::dbgs());
    llvm::dbgs() << "\n";
  });

  // Step 3: Generate transformed IR from the manipulated tree
  GenerateTransformedIR(candidate, tree, buffer_expansion_infos);
}

void StageCoarseningPass::DistributeAndSinkIntoPipeline(
    TransformCandidate& candidate, PipelineTree& tree,
    const StageGroupingAnalysis& grouping,
    llvm::SmallVectorImpl<BufferExpansionInfo>& expansion_infos) {
  LLVM_DEBUG(llvm::dbgs() << "Distributing loops into pipeline...\n");

  // Validate candidate
  if (!candidate.pipeline_ || !candidate.loop_nest_parent_) {
    LLVM_DEBUG(llvm::dbgs()
               << "  ERROR: Invalid candidate (null pipeline or parent)\n");
    return;
  }

  // For each stage group, create unmaterialized loop nest and insert
  // as children of pipeline
  LLVM_DEBUG(
      llvm::dbgs() << "\n  Distributing loop nests to stage groups...\n");

  // Process each group
  const llvm::SmallVector<StageGroup>& groups = grouping.getGroups();
  for (size_t group_idx = 0; group_idx < groups.size(); ++group_idx) {
    const StageGroup& group = groups[group_idx];
    const llvm::SmallVector<mlir::ktdf::StageOp>& stages = group.getStages();

    if (stages.empty()) continue;

    LLVM_DEBUG(llvm::dbgs()
               << "  Group " << group_idx << ": Creating loop nest for "
               << stages.size() << " stage(s)\n");

    // Find stage nodes for this group using the operation-to-node mapping
    llvm::SmallVector<StageNode*> stage_nodes;
    PipelineTreeNode* pipeline_parent_node =
        tree.getNodeForOp(candidate.pipeline_.getOperation());
    assert(pipeline_parent_node && pipeline_parent_node->isPipelineNode());

    for (mlir::ktdf::StageOp stage : stages) {
      PipelineTreeNode* node = tree.getNodeForOp(stage);
      assert(node);
      assert(node->isStageNode());
      auto* stage_node = static_cast<StageNode*>(node);
      stage_nodes.push_back(stage_node);
    }
    assert(!stage_nodes.empty());

    // Create nested loop nodes (outermost to innermost)
    // The outermost loop will be inserted as a child of the pipeline
    // Each subsequent loop will be a child of the previous loop
    // The stages will be moved to be children of the innermost loop
    OperationTreeNode* current_parent = pipeline_parent_node;

    for (size_t loop_idx = 0; loop_idx < candidate.loop_nest_.size();
         ++loop_idx) {
      // Create unmaterialized loop node using factory method
      LoopNode* loop_node = tree.createLoopNode(nullptr);  // Unmaterialized
      loop_node->setTemplateOp(candidate.loop_nest_[loop_idx].getOperation());

      // Insert loop node as child of current parent (pipeline or outer loop)
      current_parent->insertChildNode(loop_node);

      // This loop becomes the parent for the next loop or the stages
      current_parent = loop_node;
    }

    // Move all stages in this group to be children of the innermost loop
    for (StageNode* stage_node : stage_nodes) {
      stage_node->unlink();
      current_parent->insertChildNode(stage_node);
    }

    LLVM_DEBUG(llvm::dbgs() << "    Moved " << stage_nodes.size()
                            << " stage(s) under loop nest\n");
  }

  // Now remove the original loops from the tree
  // Note that references to their operations still exist as template ops in
  // the unmaterialized nodes
  for (mlir::scf::ForOp loop_op : candidate.loop_nest_) {
    PipelineTreeNode* loop_node = tree.getNodeForOp(loop_op.getOperation());
    assert(loop_node);
    assert(loop_node->isLoopNode() && "Expected loop node");
    // Unlink only the node itself, not its subtree
    // This moves the loop's children (the pipeline) up to the loop's parent
    loop_node->unlinkNode();
    // Delete the now-unlinked loop node
    tree.deleteNode(loop_node);
  }

  PerformBufferExpansion(candidate, tree, grouping, expansion_infos);

  LLVM_DEBUG(llvm::dbgs() << "\n  Loop distribution complete.\n");
}

void StageCoarseningPass::PerformBufferExpansion(
    TransformCandidate& candidate, PipelineTree& tree,
    const StageGroupingAnalysis& grouping,
    llvm::SmallVectorImpl<BufferExpansionInfo>& expansion_infos) {
  LLVM_DEBUG(llvm::dbgs() << "  Performing buffer expansion analysis...\n");

  // Step 1: Identify common memrefs across stage groups
  // Map from memref value to the set of group indices that use it
  llvm::DenseMap<mlir::Value, llvm::SmallSet<size_t, 4>> memref_to_groups;

  const llvm::SmallVector<StageGroup>& groups = grouping.getGroups();

  for (size_t group_idx = 0; group_idx < groups.size(); ++group_idx) {
    const StageGroup& group = groups[group_idx];

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
      const StageGroup& group = groups[group_idx];
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
  for (mlir::scf::ForOp loop : candidate.loop_nest_) {
    candidate_loop_ivs.push_back(loop.getInductionVar());
  }

  for (mlir::Value buffer : candidate_buffers) {
    llvm::SmallVector<mlir::Value> dependent_ivs;

    // Find all data transfer operations that use this buffer
    candidate.pipeline_.walk([&](mlir::ktdf::DataTransferOp transfer) {
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

    BufferExpansionInfo info;
    info.buffer = buffer;

    // Order the dependent IVs according to the loop nest order
    for (mlir::scf::ForOp loop : candidate.loop_nest_) {
      mlir::Value loop_iv = loop.getInductionVar();
      if (llvm::find(dependent_ivs, loop_iv) == dependent_ivs.end()) continue;
      info.loop_ivs.push_back(loop_iv);

      // Calculate the size for this dimension
      // Size = (upper_bound - lower_bound + step - 1) / step
      mlir::Value lower = loop.getLowerBound();
      mlir::Value upper = loop.getUpperBound();
      mlir::Value step = loop.getStep();
      
      // Store loop bounds and step for later index computation
      info.loop_lower_bounds.push_back(lower);
      info.loop_steps.push_back(step);

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
        info.sizes.push_back(mlir::OpFoldResult(size_attr));
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
      info.sizes.push_back(mlir::OpFoldResult(size_val));
      LLVM_DEBUG({ llvm::dbgs() << "      Dimension size (dynamic)\n"; });
    }

    if (!info.loop_ivs.empty()) {
      expansion_infos.push_back(std::move(info));

      LLVM_DEBUG({
        llvm::dbgs() << "    Buffer expansion: " << info.loop_ivs.size()
                     << " new dimension(s)\n";
      });
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

// Helper function to check if a value depends on another value
static bool isValueDependentOn(mlir::Value value, mlir::Value target) {
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

void StageCoarseningPass::SinkIntoSingleStage(
    TransformCandidate& candidate, PipelineTree& tree,
    const StageGroupingAnalysis& grouping) {
  LLVM_DEBUG(llvm::dbgs() << "Sinking into single-stage groups...\n");

  const llvm::SmallVector<StageGroup>& groups = grouping.getGroups();

  // Find the pipeline node in the tree
  PipelineTreeNode* pipeline_node =
      tree.getNodeForOp(candidate.pipeline_.getOperation());
  assert(pipeline_node && pipeline_node->isPipelineNode());

  // Process each stage group
  for (size_t group_idx = 0; group_idx < groups.size(); ++group_idx) {
    const StageGroup& group = groups[group_idx];

    // Only process single-stage groups
    if (!group.isSingleStage()) {
      LLVM_DEBUG(llvm::dbgs()
                 << "  Group " << group_idx << ": Skipping (contains "
                 << group.getStages().size() << " stages)\n");
      continue;
    }

    mlir::ktdf::StageOp stage = group.getFirstStage();
    LLVM_DEBUG({
      llvm::dbgs() << "  Group " << group_idx
                   << ": Processing single-stage group with stage at ";
      scheduler::utils::printLocation(llvm::dbgs(), stage.getOperation());
      llvm::dbgs() << "\n";
    });

    // Find the stage node in the tree
    PipelineTreeNode* stage_node = tree.getNodeForOp(stage.getOperation());
    assert(stage_node && stage_node->isStageNode());

    // The stage's parent should now be the innermost loop from the candidate
    OperationTreeNode* innermost_loop = stage_node->getParentNode();
    assert(innermost_loop && "Stage should have a parent by now");

    PipelineTreeNode* innermost_loop_pipeline =
        static_cast<PipelineTreeNode*>(innermost_loop);
    assert(innermost_loop_pipeline->isLoopNode() &&
           "Stage's parent should be a loop node now");

    // Find the outermost loop in the nest by walking up to the pipeline
    const PipelineTreeNode* outermost_loop =
        PipelineTreeNode::findOutermostLoop(innermost_loop_pipeline,
                                            [](const PipelineTreeNode* node) {
                                              return node->isPipelineNode();
                                            });
    assert(outermost_loop && "Expected to find an outermost loop");

    LLVM_DEBUG(llvm::dbgs() << "    Sinking loop nest into stage\n");

    // Sink the loop nest into the stage:
    PipelineTreeNode* mutable_outermost =
        const_cast<PipelineTreeNode*>(outermost_loop);
    assert(mutable_outermost->getParentNode() == pipeline_node);
    OperationTreeNode* insert = mutable_outermost->getPrevSibling();
    mutable_outermost->unlink();
    stage_node->unlink();
    stage_node->insertChildNode(mutable_outermost);
    pipeline_node->insertChildNode(stage_node, insert);

    LLVM_DEBUG(llvm::dbgs() << "    Successfully sank loop nest into stage\n");
  }

  LLVM_DEBUG(llvm::dbgs() << "  Sinking into single-stage groups complete.\n");
}

void StageCoarseningPass::DiscoverAndCorrectStructure(
    TransformCandidate& candidate, PipelineTree& tree) {
  LLVM_DEBUG(
      llvm::dbgs() << "Discovering and correcting illegal structures...\n");

  scheduler::pipeline_tree::Legalizer legalizer;

  // Find and fix all violations iteratively
  legalizer.findAndFixViolations(tree);

  LLVM_DEBUG(llvm::dbgs() << "  Structure legalization complete.\n");
}

void StageCoarseningPass::CorrectStageDependencies(
    TransformCandidate& candidate, PipelineTree& tree) {
  LLVM_DEBUG(llvm::dbgs() << "Correcting stage dependencies...\n");

  // TODO: Implement dependency correction
  // For each stage dependency that crosses pipeline boundaries:
  //   1. Identify the dependency
  //   2. Create nested pipeline if needed
  //   3. Move stages to correct pipeline level

  LLVM_DEBUG(llvm::dbgs() << "  (Implementation pending)\n");
}

//===----------------------------------------------------------------------===//
// Step 3: Generate Transformed IR from PipelineTree
//===----------------------------------------------------------------------===//

void StageCoarseningPass::GenerateTransformedIR(
    TransformCandidate& candidate, PipelineTree& tree,
    llvm::ArrayRef<BufferExpansionInfo> expansion_infos) {
  LLVM_DEBUG(llvm::dbgs() << "\n--- Step 3: Generate Transformed IR ---\n");

  // Get the outer pipeline node from the tree
  PipelineTreeNode* outer_pipeline =
      tree.getNodeForOp(candidate.pipeline_.getOperation());
  assert(outer_pipeline);
  assert(outer_pipeline->isPipelineNode());

  // Create a builder at the location of the root (ie right before outermost
  // loop in the original candidate loop nest)
  Operation* root = candidate.loop_nest_.front();
  mlir::OpBuilder builder(root);

  // Create an IRMapping to track value mappings during cloning
  mlir::IRMapping value_map;

  // Create buffer handler if we have buffers to expand
  scheduler::BufferHandler* buffer_handler = nullptr;
  std::optional<StageCoarseningBufferHandler> handler_storage;
  if (!expansion_infos.empty()) {
    LLVM_DEBUG({
      llvm::dbgs() << "  Creating buffer handler for " << expansion_infos.size()
                   << " buffer(s)\n";
    });
    handler_storage.emplace(expansion_infos);
    buffer_handler = &handler_storage.value();
  }

  // Use the materializer to generate IR from the tree
  scheduler::PipelineTreeMaterializer materializer(builder, value_map,
                                                   buffer_handler);
  materializer.materialize(*outer_pipeline);

  // Erase the original candidate loop nest and all the old ops under it
  root->erase();

  LLVM_DEBUG(llvm::dbgs() << "  IR generation complete.\n");
}

}  // namespace
}  // namespace ktdf

std::unique_ptr<mlir::Pass> ktdf::createStageCoarseningPass() {
  return std::make_unique<ktdf::StageCoarseningPass>();
}
