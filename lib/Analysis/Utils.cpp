//===-- Utils.h -------------------------------------------------*- c++ -*-===//
//
// Part of the Dataflow Scheduler project.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//
//
// Implements some common utilities for scheduler analyses.
//
//===----------------------------------------------------------------------===//

#include "dataflow-scheduler/Analysis/Utils.h"

#include <mlir/IR/Location.h>
#include <mlir/IR/Operation.h>

using namespace scheduler;

void scheduler::printLocation(llvm::raw_ostream& OS, mlir::Operation* op) {
  if (!op) {
    OS << "<nullptr op>";
    return;
  }

  if (auto file_loc = op->getLoc()->findInstanceOf<mlir::FileLineColLoc>()) {
    OS << "@ line: " << file_loc.getLine();
  } else {
    OS << "@ line: " << op;
  }
}

auto scheduler::getElementSizeBytes(mlir::Type element_type) -> int64_t {
  unsigned element_size_bits = element_type.getIntOrFloatBitWidth();
  return (element_size_bits + 7) / 8;
}
