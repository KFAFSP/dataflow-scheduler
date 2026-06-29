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
/// Walk a list of work ops, collect every ktdf_lowering.execute_on, and build
/// a MapVector mapping each resource type to the unit Values used by ops of
/// that type. Insertion order = first-encounter order during the walk.
///
//
//===----------------------------------------------------------------------===//

#ifndef DATAFLOW_SCHEDULER_CONVERSION_KTDFLOWTODFIR_UNITTYPEDISCOVERY_H_
#define DATAFLOW_SCHEDULER_CONVERSION_KTDFLOWTODFIR_UNITTYPEDISCOVERY_H_

#include "dataflow-scheduler/Utils/SchedulerExtContext.h"  // scheduler::ResourceType
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"

namespace scheduler {

using ResourceToUnits =
    llvm::MapVector<scheduler::ResourceType, llvm::SmallVector<mlir::Value, 4>>;

/// Walk every ktdf_lowering.execute_on op contained in `work_ops` (recursively)
/// and resolve each unit operand to its resource type via
/// scheduler::getUnitResourceType.
///
/// On success, `result` contains one entry per discovered resource type (in
/// first-encounter order) with that type's unit Values appended in the order
/// they first appeared, with no duplicates within a single type.
///
/// On failure, emits an error on the offending op and returns failure.
mlir::LogicalResult discoverUnitTypes(llvm::ArrayRef<mlir::Operation*> work_ops,
                                      ResourceToUnits& result);

}  // namespace scheduler

#endif  // DATAFLOW_SCHEDULER_CONVERSION_KTDFLOWTODFIR_UNITTYPEDISCOVERY_H_
