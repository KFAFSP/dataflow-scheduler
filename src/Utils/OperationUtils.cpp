//===----------------------------------------------------------------------===//
//
// Operation Utility Functions Implementation
//
//===----------------------------------------------------------------------===//

#include "src/Utils/OperationUtils.hpp"

#include "mlir/IR/Location.h"
#include "src/Dialects/KTDF/KTDFOps.hpp"

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

bool isMemrefWrittenTo(mlir::Value memref, mlir::Operation* root_op) {
  if (!root_op) {
    return false;
  }

  bool is_written = false;

  root_op->walk([&](mlir::Operation* op) {
    // Check if this is a data transfer operation
    if (auto transfer = mlir::dyn_cast<mlir::ktdf::DataTransferOp>(op)) {
      // Check if memref is the destination
      if (transfer.getDestination() == memref) {
        is_written = true;
        return mlir::WalkResult::interrupt();
      }
    }
    return mlir::WalkResult::advance();
  });

  return is_written;
}

}  // namespace utils
}  // namespace scheduler

// Made with Bob
