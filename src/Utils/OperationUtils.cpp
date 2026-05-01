//===----------------------------------------------------------------------===//
//
// Operation Utility Functions Implementation
//
//===----------------------------------------------------------------------===//

#include "src/Utils/OperationUtils.hpp"

#include "mlir/IR/Location.h"

namespace scheduler {
namespace utils {

void printLocation(llvm::raw_ostream& OS, mlir::Operation* op) {
  if (!op) {
    OS << "<nullptr op>";
    return;
  }

  auto loc = op->getLoc();
  if (auto file_loc = mlir::dyn_cast<mlir::FileLineColLoc>(loc)) {
    OS << "@ line: " << file_loc.getLine();
  } else {
    OS << "@ line: " << op;
  }
}

}  // namespace utils
}  // namespace scheduler

// Made with Bob
