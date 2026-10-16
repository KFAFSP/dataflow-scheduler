//===-- ReductionFootprint.h ------------------------------------*- c++ -*-===//
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
// ReductionFootprintAnalysis: the storage the input tiles of one chunk of a
// reduction occupy, and the chunk sizes that fit a budget.
//
//===----------------------------------------------------------------------===//

#ifndef DATAFLOW_SCHEDULER_DIALECT_KTDF_ANALYSIS_REDUCTIONFOOTPRINT_H_
#define DATAFLOW_SCHEDULER_DIALECT_KTDF_ANALYSIS_REDUCTIONFOOTPRINT_H_

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>
#include <mlir/Dialect/Utils/StructuredOpsUtils.h>
#include <mlir/IR/AffineMap.h>
#include <mlir/IR/Operation.h>

#include <cstdint>
#include <optional>

namespace mlir::ktdf {

/// Models the bytes the input tiles of a chunk of a reduction `linalg.generic`
/// occupy, as a function of its tile sizes.
///
/// Each input takes a tile shaped the way tiling slices it: by its indexing map
/// at the last point of the chunk. Accumulators are not counted.
///
/// Only reduction dimensions are chunked, and never the innermost loop
/// dimension or the innermost dimension of an input.
class ReductionFootprintAnalysis {
 public:
  explicit ReductionFootprintAnalysis(Operation* op);

  /// Whether the op could be modelled: a `linalg.generic` with static loop
  /// ranges, whose inputs are ranked, of a fixed-width element type, and
  /// indexed by maps without symbols.
  [[nodiscard]] auto isValid() const -> bool { return valid_; }

  /// Returns the bytes the input tiles of a chunk of @p tile_sizes , given per
  /// loop dimension, occupy.
  [[nodiscard]] auto getFootprint(llvm::ArrayRef<int64_t> tile_sizes) const
      -> int64_t;

  /// Returns the tile sizes per loop dimension whose footprint fits @p budget
  /// bytes, or std::nullopt if none does.
  ///
  /// Each tile size divides its loop range. Dimensions are shrunk from the
  /// outermost to the innermost in the layout of the inputs, each to the
  /// largest size that fits or, failing that, the one that saves the most;
  /// the ones shrunk before the last are then grown back as far as the budget
  /// allows.
  [[nodiscard]] auto getTileSizes(int64_t budget) const
      -> std::optional<llvm::SmallVector<int64_t>>;

 private:
  struct Input {
    AffineMap map;
    llvm::SmallVector<int64_t> shape;
    int64_t elem_bytes;
  };

  /// Returns the dimensions that may be shrunk, outermost first.
  [[nodiscard]] auto getShrinkOrder() const -> llvm::SmallVector<unsigned>;

  bool valid_ = false;
  llvm::SmallVector<int64_t> loop_ranges_;
  llvm::SmallVector<utils::IteratorType> iterator_types_;
  llvm::SmallVector<Input> inputs_;
  /// The dimensions that are never chunked.
  llvm::SmallVector<unsigned> pinned_;
};

}  // namespace mlir::ktdf

#endif  // DATAFLOW_SCHEDULER_DIALECT_KTDF_ANALYSIS_REDUCTIONFOOTPRINT_H_
