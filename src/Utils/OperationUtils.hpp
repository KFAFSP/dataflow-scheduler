//===----------------------------------------------------------------------===//
//
// Operation Utility Functions
//
//===----------------------------------------------------------------------===//

#ifndef SRC_UTILS_OPERATIONUTILS_HPP
#define SRC_UTILS_OPERATIONUTILS_HPP

#include "mlir/IR/Operation.h"
#include "llvm/Support/raw_ostream.h"

namespace ktir {
namespace utils {

/// Print the source location of an operation to the output stream.
/// If the operation has a FileLineColLoc, prints "@line_number".
/// Otherwise, prints "@operation_pointer".
void printLocation(llvm::raw_ostream& OS, mlir::Operation* op);

} // namespace utils
} // namespace ktir

#endif // SRC_UTILS_OPERATIONUTILS_HPP

