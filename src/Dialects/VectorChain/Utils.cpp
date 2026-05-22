//===----------------------- Utils.cpp ---------------------------===//
//
// IBM Confidential
// (C) Copyright IBM Corp. 2022, 2025
//
// =============================================================================
//
// General utility functions and classes
//
//===----------------------------------------------------------------------===//

#include "src/Dialects/VectorChain/Utils.hpp"

using namespace mlir;

bool mlir::vectorchain::utils::isShuffleIndicesValid(
    vectorchain::ShuffleOp shuffle_op) {
  for (auto index_iter : shuffle_op.getIndices().getValue()) {
    if (auto index_attr = mlir::dyn_cast<IntegerAttr>(index_iter)) {
      int index = index_attr.getInt();
      if (index < -1) {
        shuffle_op->emitError("Indices should be integers >= -1");
        return false;
      }
    } else {
      shuffle_op->emitError("Indices should be integers");
      return false;
    }
  }
  return true;
}

std::vector<int> mlir::vectorchain::utils::getShuffleIndicesAsVector(
    vectorchain::ShuffleOp shuffle_op) {
  std::vector<int> indices;
  if (!isShuffleIndicesValid(shuffle_op)) return indices;
  for (auto index_iter : shuffle_op.getIndices().getValue()) {
    indices.push_back(mlir::dyn_cast<IntegerAttr>(index_iter).getInt());
  }
  return indices;
}

std::vector<int> mlir::vectorchain::utils::getExpandedVector(
    std::vector<int> indices, int repetition) {
  std::vector<int> indices_expanded;
  for (int i = 0; i < repetition; i++) {
    indices_expanded.insert(indices_expanded.end(), indices.begin(),
                            indices.end());
  }
  return indices_expanded;
}

bool mlir::vectorchain::utils::isFirstElemSplat(
    vectorchain::ShuffleOp shuffle_op) {
  std::vector<int> indices = getShuffleIndicesAsVector(shuffle_op);
  std::vector<int> indices_expanded =
      getExpandedVector(indices, shuffle_op.getRepetition());
  for (auto val : indices_expanded) {
    if (val != 0) return false;
  }
  return true;
}

bool mlir::vectorchain::utils::isPadLeftFor8FirstElemSplat(
    vectorchain::ShuffleOp shuffle_op) {
  std::vector<int> indices = getShuffleIndicesAsVector(shuffle_op);
  std::vector<int> indices_expanded =
      getExpandedVector(indices, shuffle_op.getRepetition());
  for (std::size_t i = 0; i < indices_expanded.size(); i++) {
    if (i % 8 == 0) {
      if (indices_expanded[i] != 0) return false;
    } else {
      if (indices_expanded[i] != -1) return false;
    }
  }
  return true;
}

bool mlir::vectorchain::utils::isShuffleNFWDVersion0(
    vectorchain::ShuffleOp shuffle_op) {
  std::vector<int> indices = getShuffleIndicesAsVector(shuffle_op);
  std::vector<int> indices_expanded =
      getExpandedVector(indices, shuffle_op.getRepetition());
  if (indices_expanded.size() == 32) {
    // FP32 case
    int nfwd0[4] = {1, 0, 3, 3};
    for (std::size_t i = 0; i < indices_expanded.size(); i++) {
      if (indices_expanded[i] != nfwd0[i % 4]) return false;
    }
  } else {
    // FP16 case
    int nfwd0[8] = {2, 3, 0, 1, 6, 7, 6, 7};
    for (std::size_t i = 0; i < indices_expanded.size(); i++) {
      if (indices_expanded[i] != nfwd0[i % 8]) return false;
    }
  }
  return true;
}

bool mlir::vectorchain::utils::isShuffleNFWDVersion2(
    vectorchain::ShuffleOp shuffle_op) {
  std::vector<int> indices = getShuffleIndicesAsVector(shuffle_op);
  std::vector<int> indices_expanded =
      getExpandedVector(indices, shuffle_op.getRepetition());
  if (indices_expanded.size() == 32) {
    // FP32 case
    int nfwd2[4] = {2, 0, 1, 3};
    for (std::size_t i = 0; i < indices_expanded.size(); i++) {
      if (indices_expanded[i] != nfwd2[i % 4]) return false;
    }
  } else {
    // FP16 case
    int nfwd2[8] = {4, 5, 3, 3, 5, 5, 6, 7};
    for (std::size_t i = 0; i < indices_expanded.size(); i++) {
      if (indices_expanded[i] != nfwd2[i % 8]) return false;
    }
  }
  return true;
}

bool mlir::vectorchain::utils::isSplatFromFirstElem(
    vectorchain::ShuffleOp shuffle_op, unsigned num_bytes) {
  std::vector<int> indices = getShuffleIndicesAsVector(shuffle_op);

  // A splat is defined as an indices attribute consisting of n elements
  // from the beginning of the vector, where n is the number of elements
  // equating to the number of bytes specified by num_bytes.
  //
  // num_bytes can be constructed from multiple elements of various data types.
  auto input_type = mlir::cast<VectorType>(shuffle_op.getInput().getType());
  if (!input_type.getElementType().isIntOrFloat()) return false;
  unsigned input_element_bitwidth =
      input_type.getElementType().getIntOrFloatBitWidth();
  unsigned num_bits = num_bytes * 8;
  unsigned num_elements = num_bits / input_element_bitwidth;

  if (num_elements == 0 || (num_bits % input_element_bitwidth != 0) ||
      indices.size() != num_elements)
    return false;

  for (unsigned i = 0; i < num_elements; ++i) {
    if (indices[i] != static_cast<int>(i)) return false;
  }

  return true;
}

bool mlir::vectorchain::utils::isRightZeroPadFromFirstElem(
    vectorchain::ShuffleOp shuffle_op, unsigned num_bytes) {
  std::vector<int> indices = getShuffleIndicesAsVector(shuffle_op);

  // Right zero pad is defined as an indices attribute consisting of n
  // elements from the beginning of the vector, where n is the number of
  // elements equating to a number of bytes specified by num_bytes, followed by
  // repeating -1 to indicate a padding element.
  //
  // num_bytes can be constructed from multiple elements of various data types.
  auto input_type = mlir::cast<VectorType>(shuffle_op.getInput().getType());
  if (!input_type.getElementType().isIntOrFloat()) return false;
  unsigned input_element_bitwidth =
      input_type.getElementType().getIntOrFloatBitWidth();
  unsigned num_bits = num_bytes * 8;
  unsigned num_elements = num_bits / input_element_bitwidth;

  if ((num_elements == 0) || (num_bits % input_element_bitwidth != 0))
    return false;

  for (unsigned i = 0; i < num_elements; ++i)
    if (indices[i] != static_cast<int>(i)) return false;

  auto indices_size = indices.size();
  for (unsigned i = num_elements; i < indices_size; ++i)
    if (indices[i] != -1) return false;

  return true;
}
