//===-- RegionClonePrune.h --------------------------------------*- c++ -*-===//
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
// Clone a region into a target block and prune down to a designated anchor.
// Used by reuse-promotion passes to emit the slice of a donor stage into a
// new sibling pipeline's stage body.
//
//===----------------------------------------------------------------------===//

#ifndef DATAFLOW_SCHEDULER_TRANSFORMS_UTILS_REGIONCLONEPRUNE_H_
#define DATAFLOW_SCHEDULER_TRANSFORMS_UTILS_REGIONCLONEPRUNE_H_

#include <mlir/IR/Block.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/Region.h>

namespace scheduler {

/// Clone every op from `source_region`'s entry block into `target_block`
/// at `ip`, in source order, then erase every cloned op not needed by
/// `keep_anchor`.
///
/// Kept set (the ops that survive pruning):
///   - the cloned image of `keep_anchor`
///   - every cloned op whose result is transitively consumed by a kept op
///     (via SSA operands — so operand-producing chains are preserved)
///   - every cloned op that is a structural ancestor of a kept op (so the
///     scf.for / scf.if skeleton around the anchor survives)
///   - terminators (scf.yield, etc.) of kept regions
///
/// Erasure proceeds in reverse-topological order so no dangling uses arise.
///
/// `value_map` is updated with old→new mappings for every cloned value.
/// Callers can seed it in advance (e.g., original dest buffer → new outer
/// alloc) so the cloned anchor's operands are remapped automatically.
///
/// Returns the cloned image of `keep_anchor` (i.e., the surviving anchor
/// in the target block's region tree).
mlir::Operation* cloneRegionAndPruneToAnchor(mlir::Region& source_region,
                                             mlir::Block* target_block,
                                             mlir::Block::iterator ip,
                                             mlir::Operation* keep_anchor,
                                             mlir::IRMapping& value_map);

}  // namespace scheduler

#endif  // DATAFLOW_SCHEDULER_TRANSFORMS_UTILS_REGIONCLONEPRUNE_H_
