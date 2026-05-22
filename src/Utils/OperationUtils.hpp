//===----------------------------------------------------------------------===//
//
// Operation Utility Functions
//
//===----------------------------------------------------------------------===//

#ifndef SRC_UTILS_OPERATIONUTILS_HPP
#define SRC_UTILS_OPERATIONUTILS_HPP

#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"

namespace scheduler {
namespace utils {

/// Print the source location of an operation to the output stream.
/// If the operation has a FileLineColLoc, prints "@line_number".
/// Otherwise, prints "@operation_pointer".
void printLocation(llvm::raw_ostream& OS, mlir::Operation* op);

/// Check if a memref value is written to within the given operation's region.
/// Returns true if any data transfer operation writes to the memref as a
/// destination.
/// \param memref The memref value to check
/// \param root_op The root operation to search within
/// \return true if the memref is written to, false otherwise
bool isMemrefWrittenTo(mlir::Value memref, mlir::Operation* root_op);

}  // namespace utils
}  // namespace scheduler

#endif  // SRC_UTILS_OPERATIONUTILS_HPP
