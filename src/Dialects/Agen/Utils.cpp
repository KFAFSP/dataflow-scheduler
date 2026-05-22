//===----------------------- Utils.cpp ------------------------------------===//
//
// IBM Confidential
// (C) Copyright IBM Corp. 2025
//
// =============================================================================
//
// Utility functions extracted from Agen dialect for AgenToLLVM conversion.
//
//===----------------------------------------------------------------------===//

#include "src/Dialects/Agen/Utils.hpp"

#include "mlir/Dialect/Affine/Analysis/AffineStructures.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "src/dt_exception.hpp"

using namespace mlir;

void dialect::agen::utils::getMapCoefficients(SmallVectorImpl<int64_t>& coeffs,
                                              const AffineMap& map,
                                              int result) {
  affine::FlatAffineValueConstraints csts;
  auto res = getFlattenedAffineExpr(map.getResult(result), map.getNumDims(),
                                    map.getNumSymbols(), &coeffs, &csts);
  DT_CHECK(succeeded(res));
}

/// @brief Given \p composed_map , construct the composed coefficients.
/// @param composed_map AffineMap representing an ordered subscripts map
/// composed with a layout map.
/// @return SmallVector<int64_t> containing the individual composed iterator
/// coefficients.
static SmallVector<int64_t> constructDimCoefficients(AffineMap& composed_map) {
  composed_map = compressUnusedSymbols(composed_map);
  composed_map = simplifyAffineMap(composed_map);

  SmallVector<int64_t> composed_coeffs;
  affine::FlatAffineValueConstraints composed_flat_csts;

  DT_CHECK(composed_map.getResults().size() == 1);
  AffineExpr expr = composed_map.getResult(0);

  auto flat_result = getFlattenedAffineExpr(
      expr, composed_map.getNumDims(), composed_map.getNumSymbols(),
      &composed_coeffs, &composed_flat_csts);

  DT_CHECK(flat_result.succeeded());

  return composed_coeffs;
}

DenseMap<Value, int64_t> dialect::agen::utils::constructIteratorCoeffDict(
    const AffineMap& subscripts_map, const AffineMap& transfer_order,
    const AffineMap& mem_view_layout_map, SmallVectorImpl<Value>& indices) {
  auto composed_load_order_map = transfer_order.compose(subscripts_map);
  auto composed_layout_map =
      mem_view_layout_map.compose(composed_load_order_map);
  affine::fullyComposeAffineMapAndOperands(&composed_layout_map, &indices);
  auto composed_layout_coeffs = constructDimCoefficients(composed_layout_map);

  // Sort the indices from outermost to innnermost
  DenseMap<Value, int64_t> indices_coeff_dict;
  for (std::size_t i = 0; i < indices.size(); ++i)
    indices_coeff_dict[indices[i]] = composed_layout_coeffs[i];

  int const_value =
      composed_layout_coeffs.size() != composed_layout_map.getNumDims()
          ? composed_layout_coeffs.back()
          : 0;

  // indices_coeff_dict is a map from loop iterators to coefficients associated
  // with them. nullptr refers to constant offset
  // for, e.g., 2xi + 3xj + 10
  // coefficient with i is 2, j is 3, and nullptr is 10.
  indices_coeff_dict[nullptr] = const_value;

  return indices_coeff_dict;
}

LogicalResult dialect::agen::utils::calculateTimeOffsets(
    SmallVectorImpl<int64_t>& time_offsets,
    const AffineMap& mem_view_layout_map, const AffineMap& time_addr_map,
    const AffineMap& time_order) {
  affine::FlatAffineValueConstraints* cst = nullptr;
  SmallVector<int64_t> tmp_time_offsets;
  AffineMap tmp_time_addr_map = mem_view_layout_map.compose(time_addr_map);
  tmp_time_addr_map = simplifyAffineMap(tmp_time_addr_map);
  if (getFlattenedAffineExpr(
          tmp_time_addr_map.getResult(0), tmp_time_addr_map.getNumDims(),
          tmp_time_addr_map.getNumSymbols(), &tmp_time_offsets, cst)
          .failed())
    return failure();

  SmallVector<int64_t> tmp_non_const_time_offsets;
  for (int i = 0, e = tmp_time_offsets.size() - 1; i < e; ++i)
    tmp_non_const_time_offsets.push_back(tmp_time_offsets[i]);

  time_offsets = time_order.compose(tmp_non_const_time_offsets);
  time_offsets.push_back(tmp_time_offsets.back());

  return success();
}

// Made with Bob
