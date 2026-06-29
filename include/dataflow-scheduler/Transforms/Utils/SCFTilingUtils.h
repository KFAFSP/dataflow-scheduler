//===-------------------------------------------------------------*- c++ -*-==//
//
// Part of the Dataflow Scheduler project.
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
//
// Utility functions for SCF loop tiling and strip-mining transformations.
//
//===----------------------------------------------------------------------===//

#ifndef DATAFLOW_SCHEDULER_TRANSFORMS_UTILS_SCFTILINGUTILS_H_
#define DATAFLOW_SCHEDULER_TRANSFORMS_UTILS_SCFTILINGUTILS_H_

#include <llvm/ADT/ArrayRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/PatternMatch.h>

namespace scheduler {

/// Normalize a collection of SCF loops to start at 0 with step 1, and
/// denormalize their induction variables.
///
/// This function:
/// 1. Normalizes each loop to have bounds [0, size) with step 1
/// 2. Denormalizes the induction variable (IV) to maintain original semantics
/// 3. Sinks affine.apply operations closer to their uses for better performance
///
/// The denormalization replaces uses of the normalized IV with:
///   denormalized_iv = orig_lb + normalized_iv * orig_step
///
/// This is particularly useful after tiling or strip-mining, where loops may
/// have non-zero lower bounds or non-unit steps.
///
/// @param loops The loops to normalize (can include both outer and inner loops)
/// @param rewriter The IR rewriter to use for modifications
void normalizeSCFLoops(llvm::ArrayRef<mlir::scf::ForOp> loops,
                       mlir::IRRewriter& rewriter);

/// Sink affine.apply operations closer to their uses within a loop.
///
/// After denormalization, affine.apply operations are created at the loop start
/// but may not be used until deep in nested structures (e.g., inside
/// ktdf.pipeline blocks). This function clones each affine.apply operation
/// right before each of its uses, ensuring all uses have a dominating
/// definition.
///
/// This optimization is beneficial for:
/// - Reducing register pressure
/// - Improving instruction scheduling
/// - Better locality of computation
///
/// @param loop The loop containing affine.apply operations to sink
/// @param rewriter The IR rewriter to use for modifications
void sinkAffineApplyOpsToUses(mlir::scf::ForOp loop,
                              mlir::IRRewriter& rewriter);

/// Propagate loop_type attribute from original loops to inner point loops
/// after tiling or strip-mining.
///
/// When a loop with a loop_type attribute (e.g., parallel_loop) is tiled,
/// the resulting inner "point" loops should inherit this attribute to preserve
/// the semantic information about loop parallelism.
///
/// @param original_loops The original loops before tiling (outer tile loops)
/// @param inner_loops The inner point loops created by tiling
void propagateLoopTypeAttribute(llvm::ArrayRef<mlir::scf::ForOp> original_loops,
                                llvm::ArrayRef<mlir::scf::ForOp> inner_loops);

}  // namespace scheduler

#endif  // DATAFLOW_SCHEDULER_TRANSFORMS_UTILS_SCFTILINGUTILS_H_

// Made with Bob
