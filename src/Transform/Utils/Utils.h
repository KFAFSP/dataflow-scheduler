//===----------------------------------------------------------------------===//
//
// IBM Confidential
// (C) Copyright IBM Corp. 2026
//
//===----------------------------------------------------------------------===//
//
// Utility functions for transform passes.
//
//===----------------------------------------------------------------------===//

#ifndef SCHEDULER_TRANSFORM_UTILS_UTILS_H
#define SCHEDULER_TRANSFORM_UTILS_UTILS_H

#include <optional>

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Value.h"

namespace scheduler {
namespace utils {

/// Check if a value is a constant with a specific target value.
/// Returns true if the value is defined by an arith.constant operation
/// with an integer value equal to target.
bool isTargetConstant(int target, mlir::Value val);

/// Convert ktdp.spyre_memory_space attribute to ktdf.memory_space attribute.
/// Returns nullptr for unsupported memory space kinds.
mlir::Attribute convertKtdpToKtdfMemorySpace(mlir::OpBuilder& builder,
                                             mlir::Attribute ktdp_memory_space);

/// Returns the static trip count of `loop` iff lb, ub, step are all
/// arith.constant index ops. step must be positive.
std::optional<int64_t> getStaticTripCount(mlir::scf::ForOp loop);

/// True iff `loop` carries `loop_type = #ktdf.loop_type<parallel_loop>`.
bool isParallelLoop(mlir::scf::ForOp loop);

}  // namespace utils
}  // namespace scheduler

#endif  // SCHEDULER_TRANSFORM_UTILS_UTILS_H

// Made with Bob
