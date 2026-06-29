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
//
/// Partition a func::FuncOp's entry-block ops into "prelude" and "work" sets.
///
/// Prelude = dataflow.get_unit, arith.constant, uniform.def_immutable_mapping,
///           uniform.query_map (anywhere at function-body level).
/// Work    = every other op at function-body level, excluding the terminator.
///
//
//===----------------------------------------------------------------------===//

#ifndef DATAFLOW_SCHEDULER_CONVERSION_KTDFLOWTODFIR_PRELUDEWORKPARTITION_H_
#define DATAFLOW_SCHEDULER_CONVERSION_KTDFLOWTODFIR_PRELUDEWORKPARTITION_H_

#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Operation.h"

namespace scheduler {

struct PreludeWorkSplit {
  llvm::SmallVector<mlir::Operation*, 16> prelude_ops;
  llvm::SmallVector<mlir::Operation*, 16> work_ops;
};

PreludeWorkSplit partitionPreludeAndWork(mlir::func::FuncOp func);

}  // namespace scheduler

#endif  // DATAFLOW_SCHEDULER_CONVERSION_KTDFLOWTODFIR_PRELUDEWORKPARTITION_H_
