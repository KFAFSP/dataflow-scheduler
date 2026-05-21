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
#include "src/Dialects/KTDF/KTDFOps.hpp"

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

/// Rebuild a ktdf.private op with only the results that are used and the
/// operations needed to produce the corresponding yield operands.
mlir::LogicalResult cleanupPrivateOp(mlir::ktdf::PrivateOp private_op);

/// Apply cleanupPrivateOp to all immediate private ops in the given pipeline.
mlir::LogicalResult cleanupPrivateOpsInPipeline(
    mlir::ktdf::PipelineOp pipeline_op);

}  // namespace utils
}  // namespace scheduler

#endif  // SCHEDULER_TRANSFORM_UTILS_UTILS_H

// Made with Bob
