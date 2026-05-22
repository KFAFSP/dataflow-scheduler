//===--------------------------- Utils.hpp --------------------------------===//
//
// IBM Confidential
// (C) Copyright IBM Corp. 2022, 2023
//
// =============================================================================
//
// General utility functions and classes for the dataflow dialect
//
//===----------------------------------------------------------------------===//

#ifndef DIALECT_DATAFLOW_UTILS_H
#define DIALECT_DATAFLOW_UTILS_H

#include <llvm/ADT/StringRef.h>

#include <string>

#include "mlir/IR/Value.h"
#include "src/Dialects/Dataflow/DataflowOps.hpp"
#include "src/dt_exception.hpp"

namespace dialects::dataflow::utils {

int getCoreId(const mlir::dataflow::GetUnitOp op);
int getCoreletId(const mlir::dataflow::GetUnitOp op);

/// Return the component referenced by the get_unit operation \p op.
llvm::StringRef getUnitType(mlir::dataflow::GetUnitOp op);

/// Generate a unique string encoding for a get_unit operation
std::string getUniqueUnitString(mlir::dataflow::GetUnitOp op);

bool equivalent(const mlir::dataflow::GetUnitOp,
                const mlir::dataflow::GetUnitOp);

/// Given a create_multicast_group operation, return the GTR encoding of the
/// corresponding information.
uint32_t encodeMulticastGroupInfo(
    ::mlir::dataflow::CreateMulticastGroupOp multicast);

/// Given a GTR encoding decode it into individual pieces of information.
std::string decodeMulticastGroupInfo(uint32_t gtr);

/// @brief Returns the iter_arg index of \p iter_arg in \p loop .
/// @tparam LoopTy The operation type of \p loop .
/// @param loop The loop expected to contain \p iter_arg .
/// @param iter_arg The iterator argument to search for.
/// @return int representing the iter_arg operand index \p iter_arg can be found
/// at.
template <typename LoopTy>
int getIterArgIndex(LoopTy& loop, mlir::BlockArgument& iter_arg) {
  DT_CHECK(iter_arg.getOwner()->getParentOp() == loop);
  auto region_args = loop.getRegionIterArgs();
  int index = -1;
  for (int i = 0, e = region_args.size(); i < e; ++i) {
    if (region_args[i] == iter_arg) {
      index = i;
      break;
    }
  }
  return index;
}

/// @brief Returns the Value initializing \p iter_arg from \p loop .
/// @tparam LoopTy The operation type of \p loop .
/// @param loop The loop expected to contain \p iter_arg .
/// @param iter_arg The iterator argument to search for.
/// @return Value initializing \p iter_arg .
template <typename LoopTy>
mlir::Value getIterArgInit(LoopTy& loop, mlir::BlockArgument& iter_arg) {
  DT_CHECK(iter_arg.getOwner()->getParentOp() == loop);
  int index = getIterArgIndex<LoopTy>(loop, iter_arg);
  DT_CHECK(index >= 0);
  return loop.getOperand(loop.getNumControlOperands() + index);
}

/// @brief Sets the initial value of the iter_arg chain originating from
/// \p iter_arg to \p replace_val .
/// @param iter_arg
/// @param replace_val
void setIterArgChainInit(mlir::BlockArgument& iter_arg,
                         mlir::Value replace_val);

/// @brief Returns the Value initializing the chain of iterator arguments
/// originating from \p iter_arg .
/// @param iter_arg The iterator argument at the bottom of the chain to
/// traverse.
/// @param req_one_use If true, requires each iter_arg in the chain to be used
/// only once. When set, if multiple uses are detected, the function will
/// assert. Defaults to false.
/// @return Value initializing the top of the chain from \p iter_arg .
mlir::Value getIterArgChainInit(mlir::BlockArgument& iter_arg,
                                bool req_one_use = false);

/// @brief Sets the initial value of \p iter_arg to \p replace_val .
/// @param iter_arg
/// @param replace_val
void setIterArgInit(mlir::BlockArgument& iter_arg, mlir::Value replace_val);

/// @brief Returns the iter_arg chain originating from \p iter_arg and sets
/// \p init_val to the value initializing the chain.
/// @param iter_arg The iterator argument at the bottom of the chain to analyze.
/// @param init_val This function will set this parameter to the value
/// initializing the chain.
/// @return The iter_arg chain originating from \p iter_arg .
llvm::SmallVector<mlir::BlockArgument> getIterArgChain(
    mlir::BlockArgument& iter_arg, mlir::Value& init_val);

}  // namespace dialects::dataflow::utils

#endif  // DIALECT_DATAFLOW_UTILS_H
