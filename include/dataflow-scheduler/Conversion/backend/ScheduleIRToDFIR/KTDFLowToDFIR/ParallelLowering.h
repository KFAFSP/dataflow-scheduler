//===------------------------------------------------------------*- c++ -*-===//
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

#ifndef DATAFLOW_SCHEDULER_CONVERSION_KTDFLOWTODFIR_PARALLELLOWERING_H_
#define DATAFLOW_SCHEDULER_CONVERSION_KTDFLOWTODFIR_PARALLELLOWERING_H_

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Support/LogicalResult.h"

namespace scheduler {

/// Lower ktdf.parallel operations to scf.for loops.
/// This should be called after all other lowerings are complete, particularly
/// after get_logical_memory_view operations are created.
/// @param func The function containing ktdf.parallel operations to lower
/// @return success if all parallel operations were successfully lowered
mlir::LogicalResult lowerParallelOps(mlir::func::FuncOp func);

}  // namespace scheduler

#endif  // DATAFLOW_SCHEDULER_CONVERSION_KTDFLOWTODFIR_PARALLELLOWERING_H_

// Made with Bob