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

#include "mlir/Pass/Pass.h"
#include "src/Dialects/VectorChain/VectorChainOps.hpp"

using namespace mlir;

namespace mlir {
namespace vectorchain {
namespace utils {

bool isShuffleIndicesValid(vectorchain::ShuffleOp shuffle_op);
std::vector<int> getShuffleIndicesAsVector(vectorchain::ShuffleOp shuffle_op);
std::vector<int> getExpandedVector(std::vector<int> indices, int repetition);
bool isFirstElemSplat(vectorchain::ShuffleOp shuffle_op);
bool isPadLeftFor8FirstElemSplat(vectorchain::ShuffleOp shuffle_op);
bool isShuffleNFWDVersion0(vectorchain::ShuffleOp shuffle_op);
bool isShuffleNFWDVersion2(vectorchain::ShuffleOp shuffle_op);
bool isRightZeroPadFromFirstElem(vectorchain::ShuffleOp shuffle_op,
                                 unsigned num_bytes);
bool isSplatFromFirstElem(vectorchain::ShuffleOp shuffle_op,
                          unsigned num_bytes);

}  // namespace utils
}  // namespace vectorchain
}  // namespace mlir

#endif  // SRC_DIALECT_VECTORCHAIN_UTILS_H
