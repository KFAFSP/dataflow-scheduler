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

#include "Utils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"

namespace scheduler {
namespace utils {

bool isTargetConstant(int target, mlir::Value val) {
  if (auto const_op = val.getDefiningOp<mlir::arith::ConstantIndexOp>()) {
    return const_op.value() == target;
  }
  return false;
}

}  // namespace utils
}  // namespace scheduler

// Made with Bob
