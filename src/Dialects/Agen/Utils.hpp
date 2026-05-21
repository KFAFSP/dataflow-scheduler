//===--------------------------- Utils.hpp --------------------------------===//
//
// IBM Confidential
// (C) Copyright IBM Corp. 2025
//
// =============================================================================
//
// Utility functions extracted from Agen dialect for AgenToLLVM conversion.
//
//===----------------------------------------------------------------------===//

#ifndef SRC_CONVERSION_AGENTOLLVM_UTILS_H
#define SRC_CONVERSION_AGENTOLLVM_UTILS_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Value.h"

using namespace mlir;

namespace dialect {
namespace agen {
namespace utils {

/// @brief Get the coefficients of all dims for \p result of \p map .
/// @param coeffs The vector to fill with the coefficients.
/// @param map The AffineMap to collect the coefficients from.
/// @param result The map result to collect the coefficients from.
void getMapCoefficients(SmallVectorImpl<int64_t> &coeffs, const AffineMap &map,
                        int result);

/// @brief Construct a map of iterator coefficients from loop iterators to
/// their constant coefficients calculated using the layout map.
/// @param subscripts_map The subscripts map to calculate coefficients for.
/// @param transfer_order A map describing how the subscripts map should be
/// ordered.
/// @param mem_view_layout_map The 1D map describing how to stride through
/// memory dimensions.
/// @param indices SmallVector<Value> containing the Values representing the
/// subscripts map dimensions.
/// @return DenseMap<Value, int64_t> containing the indices to composed
/// coefficients.
DenseMap<Value, int64_t> constructIteratorCoeffDict(
    const AffineMap &subscripts_map, const AffineMap &transfer_order,
    const AffineMap &mem_view_layout_map, SmallVectorImpl<Value> &indices);

/// @brief Calculates time offsets given an initial \p time_offsets ,
/// \p mem_view_layout_map , \p time_addr_map , and a \p time_order .
/// @param time_offsets The initialized time_offsets to calculate. Will contain
/// the final time_offsets once calculated.
/// @param mem_view_layout_map The memory view layout map to use for the
/// calculation.
/// @param time_addr_map The time address map to use for the calculation.
/// @param time_order The time order to use for the final \p time_offsets .
/// @return LogicalResult::success() if the calculation was successful,
/// LogicalResult::failure() otherwise.
LogicalResult calculateTimeOffsets(SmallVectorImpl<int64_t> &time_offsets,
                                   const AffineMap &mem_view_layout_map,
                                   const AffineMap &time_addr_map,
                                   const AffineMap &time_order);

}  // namespace utils
}  // namespace agen
}  // namespace dialect

#endif  // SRC_CONVERSION_AGENTOLLVM_UTILS_H

// Made with Bob
