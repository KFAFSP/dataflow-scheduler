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
// Declares some common utilities for ktdf dialect analyses.
//
//===----------------------------------------------------------------------===//

#ifndef DATAFLOW_SCHEDULER_DIALECT_KTDF_ANALYSIS_UTILS_H_
#define DATAFLOW_SCHEDULER_DIALECT_KTDF_ANALYSIS_UTILS_H_

#include <mlir/Dialect/SCF/IR/SCF.h>

#include "dataflow-scheduler/Dialect/KTDF/KTDF.h"

namespace mlir::ktdf {

/// True iff `loop` carries `loop_type = #ktdf.loop_type<parallel_loop>`.
auto isParallelLoop(scf::ForOp loop) -> bool;

/// Find the parent ktdf.parallel operation for a given stage
/// Returns nullptr if the stage is not inside a parallel operation
auto findParallelParent(StageOp stage) -> Operation*;

}  // namespace mlir::ktdf

#endif  // DATAFLOW_SCHEDULER_DIALECT_KTDF_ANALYSIS_UTILS_H_
