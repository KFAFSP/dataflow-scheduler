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
// Utility for tiling normalized perfectly nested scf.for loops into
// ktdf.tiling.derive_size / ktdf.tiling.linearize_index form.
//
//===----------------------------------------------------------------------===//

#ifndef DATAFLOW_SCHEDULER_DIALECT_KTDF_TRANSFORMS_TILENORMALIZED_H_
#define DATAFLOW_SCHEDULER_DIALECT_KTDF_TRANSFORMS_TILENORMALIZED_H_

#include <llvm/ADT/SmallVector.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LogicalResult.h>

namespace mlir::ktdf {

/// Result of customTileNormalizedPerfectlyNested.
struct TileNestResult {
  /// Outer tile loops, outermost-first (one per tiled dimension).
  SmallVector<scf::ForOp> tile_loops;
  /// Inner point loops, outermost-first (one per tiled dimension).
  SmallVector<scf::ForOp> point_loops;
};

/// Tile a perfectly nested, normalized scf.for loop nest using
/// ktdf.tiling.derive_size and ktdf.tiling.linearize_index.
///
/// Preconditions (returns failure if violated):
///   - All loops in the nest starting from root_loop have lb=0 and step=1.
///   - tile_sizes.size() == number of perfectly nested loops from root_loop.
///
/// On success, the original loop nest is replaced by:
///   - tile_loops: outer tile loops (lb=0, step=1, trip=ceildivui(ub, ts))
///   - point_loops: inner point loops (lb=0, step=1,
///   ub=ktdf.tiling.derive_size)
///
/// All tile loops appear outermost; all point loops appear innermost.
/// Loop attributes (e.g., loop_type) carry forward to both tile and point
/// loops. ktdf.tiling.linearize_index ops using tiled IVs are updated to
/// include the new tile level. Bare IV uses are wrapped in new
/// ktdf.tiling.linearize_index ops.
FailureOr<TileNestResult> customTileNormalizedPerfectlyNested(
    scf::ForOp root_loop, ArrayRef<Value> tile_sizes, IRRewriter& rewriter);

}  // namespace mlir::ktdf

#endif  // DATAFLOW_SCHEDULER_DIALECT_KTDF_TRANSFORMS_TILENORMALIZED_H_

// Made with Bob
