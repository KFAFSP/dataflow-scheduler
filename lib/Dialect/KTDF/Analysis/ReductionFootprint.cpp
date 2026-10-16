//===-- ReductionFootprint.cpp ----------------------------------*- c++ -*-===//
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

#include "dataflow-scheduler/Dialect/KTDF/Analysis/ReductionFootprint.h"

#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/Sequence.h>
#include <llvm/Support/DebugLog.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/IR/BuiltinTypes.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <utility>

#define DEBUG_TYPE "reduction-footprint"

namespace {

/// Returns the divisors of @p value , largest first.
[[nodiscard]] auto getDivisors(int64_t value) -> llvm::SmallVector<int64_t> {
  llvm::SmallVector<int64_t> divisors;
  for (auto divisor = value; divisor > 0; --divisor) {
    if (value % divisor == 0) {
      divisors.push_back(divisor);
    }
  }
  return divisors;
}

}  // namespace

mlir::ktdf::ReductionFootprintAnalysis::ReductionFootprintAnalysis(
    Operation* op) {
  auto generic = llvm::dyn_cast<linalg::GenericOp>(op);
  if (!generic) {
    return;
  }
  loop_ranges_ = generic.getStaticLoopRanges();
  iterator_types_ = generic.getIteratorTypesArray();
  if (llvm::any_of(loop_ranges_, ShapedType::isDynamic)) {
    LDBG() << "dynamic loop range";
    return;
  }

  if (const auto num_loops = generic.getNumLoops(); num_loops != 0) {
    pinned_.push_back(num_loops - 1);
  }
  for (auto* const input : generic.getDpsInputOperands()) {
    const auto map = generic.getMatchingIndexingMap(input);
    const auto type = llvm::dyn_cast<RankedTensorType>(input->get().getType());
    if (!type || !type.getElementType().isIntOrFloat() ||
        map.getNumSymbols() != 0) {
      LDBG() << "input cannot be modelled: " << input->get();
      return;
    }
    if (map.getNumResults() != 0) {
      if (auto innermost =
              llvm::dyn_cast<AffineDimExpr>(map.getResults().back());
          innermost && !llvm::is_contained(pinned_, innermost.getPosition())) {
        pinned_.push_back(innermost.getPosition());
      }
    }
    inputs_.push_back(Input{map, llvm::to_vector(type.getShape()),
                            type.getElementType().getIntOrFloatBitWidth() / 8});
  }
  valid_ = true;
}

auto mlir::ktdf::ReductionFootprintAnalysis::getFootprint(
    llvm::ArrayRef<int64_t> tile_sizes) const -> int64_t {
  assert(valid_ && "the op could not be modelled");

  // Tiling sizes a slice by the map at the last point of the tile.
  const auto last_point =
      llvm::map_to_vector(tile_sizes, [](int64_t size) { return size - 1; });
  int64_t total = 0;
  for (const auto& input : inputs_) {
    int64_t elements = 1;
    for (auto [last, size] :
         llvm::zip_equal(input.map.compose(last_point), input.shape)) {
      elements *= std::min(size, last + 1);
    }
    total += elements * input.elem_bytes;
  }
  return total;
}

auto mlir::ktdf::ReductionFootprintAnalysis::getShrinkOrder() const
    -> llvm::SmallVector<unsigned> {
  // How outer a dimension is is the logarithm of its stride in the row-major
  // layout of the inputs, weighed by their size where they disagree.
  const auto num_dims = static_cast<unsigned>(loop_ranges_.size());
  llvm::SmallVector<double> outerness(num_dims, 0.0);
  llvm::SmallVector<double> weights(num_dims, 0.0);
  for (const auto& input : inputs_) {
    const auto linearize = [&](llvm::ArrayRef<int64_t> point) {
      int64_t offset = 0;
      for (auto [index, size] :
           llvm::zip_equal(input.map.compose(point), input.shape)) {
        offset = offset * size + index;
      }
      return offset;
    };
    int64_t elements = 1;
    for (auto size : input.shape) {
      elements *= size;
    }
    const auto weight = static_cast<double>(elements * input.elem_bytes);

    llvm::SmallVector<int64_t> point(num_dims, 0);
    const auto origin = linearize(point);
    for (auto dim : llvm::seq(num_dims)) {
      point[dim] = 1;
      if (const auto stride = std::abs(linearize(point) - origin);
          stride != 0) {
        outerness[dim] += weight * std::log2(static_cast<double>(stride));
        weights[dim] += weight;
      }
      point[dim] = 0;
    }
  }

  llvm::SmallVector<unsigned> order;
  for (auto dim : llvm::seq(num_dims)) {
    if (iterator_types_[dim] == utils::IteratorType::reduction &&
        weights[dim] > 0.0 && loop_ranges_[dim] > 1 &&
        !llvm::is_contained(pinned_, dim)) {
      outerness[dim] /= weights[dim];
      order.push_back(dim);
    }
  }
  llvm::stable_sort(order, [&](unsigned lhs, unsigned rhs) {
    return outerness[lhs] > outerness[rhs];
  });
  return order;
}

auto mlir::ktdf::ReductionFootprintAnalysis::getTileSizes(int64_t budget) const
    -> std::optional<llvm::SmallVector<int64_t>> {
  assert(valid_ && "the op could not be modelled");

  auto tile_sizes = loop_ranges_;
  auto current = getFootprint(tile_sizes);
  if (current <= budget) {
    return tile_sizes;
  }

  const auto footprint_with = [&](unsigned dim, int64_t size) {
    auto trial = tile_sizes;
    trial[dim] = size;
    return getFootprint(trial);
  };

  llvm::SmallVector<unsigned> shrunk;
  auto fits = false;
  for (auto dim : getShrinkOrder()) {
    std::optional<std::pair<int64_t, int64_t>> smallest;
    for (auto size : getDivisors(loop_ranges_[dim])) {
      const auto footprint = footprint_with(dim, size);
      if (footprint <= budget) {
        smallest = {size, footprint};
        fits = true;
        break;
      }
      if (!smallest || footprint < smallest->second) {
        smallest = {size, footprint};
      }
    }
    if (!fits && smallest->second >= current) {
      continue;
    }
    tile_sizes[dim] = smallest->first;
    current = smallest->second;
    shrunk.push_back(dim);
    LDBG() << "chunking d" << dim << " by " << smallest->first << ": "
           << smallest->second << " bytes";
    if (fits) {
      break;
    }
  }
  if (!fits) {
    return std::nullopt;
  }

  // The last dimension shrunk took what it needed; the ones before may have
  // been shrunk further than that.
  for (auto dim : llvm::reverse(llvm::ArrayRef(shrunk).drop_back())) {
    for (auto size : getDivisors(loop_ranges_[dim])) {
      if (size <= tile_sizes[dim]) {
        break;
      }
      if (footprint_with(dim, size) <= budget) {
        LDBG() << "growing d" << dim << " back to " << size;
        tile_sizes[dim] = size;
        break;
      }
    }
  }
  return tile_sizes;
}
