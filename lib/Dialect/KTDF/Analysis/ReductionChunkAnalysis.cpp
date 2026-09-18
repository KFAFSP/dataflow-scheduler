//===-- ReductionChunkAnalysis.cpp ------------------------------*- c++ -*-===//
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

#include "dataflow-scheduler/Dialect/KTDF/Analysis/ReductionChunkAnalysis.h"

#include <numeric>

#include "llvm/Support/DebugLog.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/BuiltinTypes.h"

#define DEBUG_TYPE "reduction-chunk-analysis"

using namespace mlir;
using namespace mlir::ktdf;

std::optional<ReductionChunkResult> mlir::ktdf::analyzeReductionChunks(
    linalg::GenericOp generic_op, int64_t chunk_size_threshold) {
  // Collect reduction dimension indices.
  SmallVector<int64_t> reduction_dims;
  auto iter_types = generic_op.getIteratorTypesArray();
  for (int64_t i = 0; i < static_cast<int64_t>(iter_types.size()); ++i)
    if (iter_types[i] == utils::IteratorType::reduction)
      reduction_dims.push_back(i);

  if (reduction_dims.empty()) {
    LDBG(1) << DEBUG_TYPE ": no reduction iterators found";
    return std::nullopt;
  }

  // Inspect the first input tensor.
  auto input_type =
      dyn_cast<RankedTensorType>(generic_op.getInputs().front().getType());
  if (!input_type) {
    LDBG(1) << DEBUG_TYPE ": first input is not a ranked tensor";
    return std::nullopt;
  }

  // Require a fixed element bit-width (rules out index type).
  if (!input_type.getElementType().isIntOrFloat()) {
    LDBG(1) << DEBUG_TYPE ": element type has no fixed bit-width";
    return std::nullopt;
  }
  int64_t element_bytes =
      input_type.getElementType().getIntOrFloatBitWidth() / 8;

  // Compute total input bytes; bail on dynamic dimensions.
  int64_t total_bytes = element_bytes;
  for (int64_t sz : input_type.getShape()) {
    if (sz == ShapedType::kDynamic) {
      LDBG(1) << DEBUG_TYPE ": dynamic input dimension — cannot infer chunks";
      return std::nullopt;
    }
    total_bytes *= sz;
  }

  // Collect reduction-dimension sizes.
  SmallVector<int64_t> red_dim_sizes;
  for (int64_t dim : reduction_dims) {
    int64_t sz = input_type.getDimSize(dim);
    if (sz == ShapedType::kDynamic) {
      LDBG(1) << DEBUG_TYPE
          ": dynamic reduction dimension — cannot infer "
          "chunks";
      return std::nullopt;
    }
    red_dim_sizes.push_back(sz);
  }

  // Upper bound: product of all reduction-dim sizes (each chunk = 1 element
  // along every reduction dim — the smallest meaningful chunk).
  int64_t max_n = 1;
  for (int64_t sz : red_dim_sizes) max_n *= sz;

  // Find the smallest N ≥ 1 that brings the per-chunk byte count within the
  // threshold.
  unsigned inferred_n = 0;
  for (int64_t n = 1; n <= max_n; ++n) {
    if (total_bytes / n <= chunk_size_threshold) {
      inferred_n = static_cast<unsigned>(n);
      break;
    }
  }

  if (inferred_n == 0) {
    LDBG(1) << DEBUG_TYPE ": no valid num_chunks found within threshold "
            << chunk_size_threshold << " bytes (total_bytes=" << total_bytes
            << ")";
    return std::nullopt;
  }

  LDBG(1) << DEBUG_TYPE ": inferred num_chunks=" << inferred_n
          << " (total_bytes=" << total_bytes
          << ", threshold=" << chunk_size_threshold << ")";

  return ReductionChunkResult{inferred_n, /*chunk_sizes=*/{},
                              std::move(reduction_dims)};
}

std::optional<llvm::SmallVector<int64_t>> mlir::ktdf::computeChunkDims(
    llvm::ArrayRef<int64_t> red_dim_sizes, int64_t chunk_size) {
  assert(chunk_size >= 1 && "chunk_size must be >= 1");

  llvm::SmallVector<int64_t> new_sizes(red_dim_sizes.begin(),
                                       red_dim_sizes.end());
  int64_t remaining = chunk_size;
  const size_t n = red_dim_sizes.size();

  for (size_t i = 0; i < n && remaining > 1; ++i) {
    int64_t dim_size = red_dim_sizes[i];

    // Skip dimensions with a symbolic (dynamic) size.
    if (dim_size == mlir::ShapedType::kDynamic) continue;

    int64_t g = std::gcd(dim_size, remaining);
    new_sizes[i] = dim_size / g;
    remaining /= g;
  }

  if (remaining > 1) return std::nullopt;

  return new_sizes;
}
