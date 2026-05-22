//===----------------------- Utils.hpp ---------------------------===//
//
// IBM Confidential
// (C) Copyright IBM Corp. 2022, 2025
//
// =============================================================================
//
// General utility functions and classes
//
//===----------------------------------------------------------------------===//

#ifndef SRC_DIALECT_VECTORCHAIN_UTILS_H
#define SRC_DIALECT_VECTORCHAIN_UTILS_H

#include "src/Dialects/VectorChain/VectorChainOps.hpp"

using namespace mlir;

namespace mlir::vectorchain::utils {

/// @brief Verifies the indices values of \p shuffle_op are valid.
/// @param shuffle_op The vectorchain::ShuffleOp to analyze for conformity.
/// @return Returns true if the indices values are valid, false otherwise.
bool isShuffleIndicesValid(vectorchain::ShuffleOp shuffle_op);

/// @brief Converts the indices of \p shuffle_op to a vector of ints.
/// @param shuffle_op The vectorchain::ShuffleOp to analyze for conformity.
/// @return Returns the indices of \p shuffle_op as a vector of ints.
std::vector<int> getShuffleIndicesAsVector(vectorchain::ShuffleOp shuffle_op);

/// @brief Expands \p indices into a vector of ints by repeating \p indices the
/// number of times specified by \p repetition.
/// @param indices The input vector of ints to expand based on \p repetition.
/// @param repetition The number of times to repeat \indices.
/// @return Returns an expanded vector consisting of \p indices repeated the
/// number of times specified by \p repetition.
std::vector<int> getExpandedVector(std::vector<int> indices, int repetition);

/// @brief Determines if \p shuffle_op represents a splat of only the first
/// element of the input to \p shuffle_op.
/// @param shuffle_op The vectorchain::ShuffleOp to analyze.
/// @return Returns true if \p shuffle_op matches a splat of the first element
/// of the input to \p shuffle_op and false otherwise.
bool isFirstElemSplat(vectorchain::ShuffleOp shuffle_op);

/// @brief Determines if \p shuffle_op represents a splat of 7 padded elements
/// followed by the first element of the input to \p shuffle_op .
/// @param shuffle_op The vectorchain::ShuffleOp to analyze.
/// @return Returns true if \p shuffle_op matches a splat of 7 padded elements
/// followed by the first element of the input to \p shuffle_op .
bool isPadLeftFor8FirstElemSplat(vectorchain::ShuffleOp shuffle_op);

bool isShuffleNFWDVersion0(vectorchain::ShuffleOp shuffle_op);
bool isShuffleNFWDVersion2(vectorchain::ShuffleOp shuffle_op);

/// @brief Determines if \p shuffle_op is a right zero pad of \p num_bytes
/// contiguous bytes starting from the first element of the input to
/// \p shuffle_op .
/// @param shuffle_op The vectorchain::ShuffleOp to analyze.
/// @param num_bytes The number of bytes \p shuffle_op is expected to use from
/// the input to \p shuffle_op before padding.
/// @return Returns true if \p shuffle_op is a right zero pad of \p num_bytes
/// contiguous bytes starting from the first element of the input to
/// \p shuffle_op .
bool isRightZeroPadFromFirstElem(vectorchain::ShuffleOp shuffle_op,
                                 unsigned num_bytes);

/// @brief Determines if \p shuffle_op is a splat of \p num_bytes contiguous
/// bytes starting from the first element of the input to \p shuffle_op .
/// @param shuffle_op The vectorchain::ShuffleOp to analyze.
/// @param num_bytes The number of bytes \p shuffle_op is expected to splat.
/// @return Returns true if \p shuffle_op is a splat of \p num_bytes contiguous
/// bytes starting from the first element of the input to \p shuffle_op .
bool isSplatFromFirstElem(vectorchain::ShuffleOp shuffle_op,
                          unsigned num_bytes);

}  // namespace mlir::vectorchain::utils

#endif  // SRC_DIALECT_VECTORCHAIN_UTILS_H
