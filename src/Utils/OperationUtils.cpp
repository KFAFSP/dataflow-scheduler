//===----------------------------------------------------------------------===//
//
// Operation Utility Functions Implementation
//
//===----------------------------------------------------------------------===//

#include "src/Utils/OperationUtils.hpp"

#include "mlir/IR/Location.h"

namespace ktir {
namespace utils {

void printLocation(llvm::raw_ostream& OS, mlir::Operation* op) {
  if (!op) {
    OS << "@nullptr";
    return;
  }

  auto loc = op->getLoc();
  if (auto fileLoc = mlir::dyn_cast<mlir::FileLineColLoc>(loc)) {
    OS << "@" << fileLoc.getLine();
  } else {
    OS << "@" << op;
  }
}

}  // namespace utils
}  // namespace ktir

// Made with Bob
