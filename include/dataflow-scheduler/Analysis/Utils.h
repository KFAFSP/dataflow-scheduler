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
// Declares some common utilities for scheduler analyses.
//
//===----------------------------------------------------------------------===//

#ifndef DATAFLOW_SCHEDULER_ANALYSIS_UTILS_H_
#define DATAFLOW_SCHEDULER_ANALYSIS_UTILS_H_

#include <mlir/IR/Operation.h>
#include <mlir/IR/Types.h>
#include <mlir/Support/LLVM.h>

namespace scheduler {

/// Print the source location of an operation to the output stream.
/// If the operation has a FileLineColLoc, prints "@line_number".
/// Otherwise, prints "@operation_pointer".
void printLocation(llvm::raw_ostream& os, mlir::Operation* op);

/// Get element size in bytes for an int or float memref element type.
auto getElementSizeBytes(mlir::Type element_type) -> int64_t;

}  // namespace scheduler

#endif  // DATAFLOW_SCHEDULER_ANALYSIS_UTILS_H_
