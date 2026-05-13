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

#include "mlir/IR/Value.h"

namespace scheduler {
namespace utils {

/// Check if a value is a constant with a specific target value.
/// Returns true if the value is defined by an arith.constant operation
/// with an integer value equal to target.
bool isTargetConstant(int target, mlir::Value val);

}  // namespace utils
}  // namespace scheduler

#endif  // SCHEDULER_TRANSFORM_UTILS_UTILS_H

// Made with Bob
