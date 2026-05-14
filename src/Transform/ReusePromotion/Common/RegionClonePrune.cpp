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

#include "src/Transform/ReusePromotion/Common/RegionClonePrune.hpp"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Operation.h"

namespace ktdf {
namespace reuse {

namespace {

/// Walk original op + cloned op lockstep, populating origin_map for every
/// op inside (including nested region ops).
void populateOriginMap(
    mlir::Operation *orig, mlir::Operation *clone,
    llvm::DenseMap<mlir::Operation *, mlir::Operation *> &origin_map) {
  origin_map[orig] = clone;
  // Assumes the clone was produced by OpBuilder::clone, which preserves
  // the structural shape (region count, block count, op count per block).
  // The llvm::zip below would silently truncate if shapes differed, but
  // that would manifest later as a missing origin_map lookup and a clear
  // assert in cloneRegionAndPruneToAnchor.
  for (auto region_pair :
       llvm::zip(orig->getRegions(), clone->getRegions())) {
    mlir::Region &orig_region = std::get<0>(region_pair);
    mlir::Region &clone_region = std::get<1>(region_pair);
    for (auto block_pair :
         llvm::zip(orig_region.getBlocks(), clone_region.getBlocks())) {
      mlir::Block &orig_block = std::get<0>(block_pair);
      mlir::Block &clone_block = std::get<1>(block_pair);
      auto orig_it = orig_block.begin();
      auto clone_it = clone_block.begin();
      while (orig_it != orig_block.end() && clone_it != clone_block.end()) {
        populateOriginMap(&*orig_it, &*clone_it, origin_map);
        ++orig_it;
        ++clone_it;
      }
    }
  }
}

/// Compute the set of cloned ops to keep, starting from `anchor` and
/// walking transitively (a) operand-producing defs and (b) parent ops,
/// stopping at `prune_root` (never add prune_root itself or anything
/// above it).
void computeKeptSet(mlir::Operation *anchor, mlir::Operation *prune_root,
                    llvm::DenseSet<mlir::Operation *> &kept) {
  llvm::SmallVector<mlir::Operation *> worklist;
  worklist.push_back(anchor);
  while (!worklist.empty()) {
    mlir::Operation *op = worklist.pop_back_val();
    if (op == prune_root || op == nullptr) {
      continue;
    }
    if (!kept.insert(op).second) {
      continue;
    }
    for (mlir::Value operand : op->getOperands()) {
      if (mlir::Operation *def = operand.getDefiningOp()) {
        worklist.push_back(def);
      }
    }
    if (mlir::Operation *parent = op->getParentOp()) {
      worklist.push_back(parent);
    }
  }
}

/// Is `op` inside the region tree rooted at `root` (but not root itself)?
bool isInsideRootRegions(mlir::Operation *op, mlir::Operation *root) {
  mlir::Operation *p = op->getParentOp();
  while (p) {
    if (p == root) {
      return true;
    }
    p = p->getParentOp();
  }
  return false;
}

}  // namespace

mlir::Operation *cloneRegionAndPruneToAnchor(
    mlir::Region &source_region, mlir::Block *target_block,
    mlir::Block::iterator ip, mlir::Operation *keep_anchor,
    mlir::IRMapping &value_map) {
  llvm::DenseMap<mlir::Operation *, mlir::Operation *> origin_map;
  mlir::OpBuilder builder(target_block, ip);

  mlir::Operation *prune_root = target_block->getParentOp();
  assert(prune_root && "target_block must have a parent op");

  // Clone top-level ops from source_region's entry block.
  mlir::Block &source_block = source_region.front();
  llvm::SmallVector<mlir::Operation *> cloned_top_level;
  for (mlir::Operation &op : source_block) {
    mlir::Operation *cloned = builder.clone(op, value_map);
    cloned_top_level.push_back(cloned);
    populateOriginMap(&op, cloned, origin_map);
  }

  mlir::Operation *cloned_anchor = origin_map.lookup(keep_anchor);
  assert(cloned_anchor && "anchor not found in cloned region");

  // Compute kept set rooted in the cloned anchor.
  llvm::DenseSet<mlir::Operation *> kept;
  computeKeptSet(cloned_anchor, prune_root, kept);

  // Collect erasure candidates: every cloned op inside the cloned top-level
  // ops' region tree that is NOT in kept and NOT a terminator.
  // walk() visits in post-order: children before parents, which means
  // inner uses are visited before outer defs — good for erasure ordering.
  llvm::SmallVector<mlir::Operation *> erasable;
  for (mlir::Operation *top : cloned_top_level) {
    top->walk([&](mlir::Operation *op) {
      if (kept.contains(op)) {
        return;
      }
      if (op->hasTrait<mlir::OpTrait::IsTerminator>()) {
        return;
      }
      // Only erase things inside the region tree we cloned (paranoia).
      if (op != top && !isInsideRootRegions(op, top)) {
        return;
      }
      erasable.push_back(op);
    });
  }

  // Erase. Drop remaining uses first to avoid dangling-use asserts from
  // MLIR's op-erase machinery.
  for (mlir::Operation *op : erasable) {
    op->dropAllUses();
  }
  for (mlir::Operation *op : erasable) {
    op->erase();
  }

  return cloned_anchor;
}

}  // namespace reuse
}  // namespace ktdf
