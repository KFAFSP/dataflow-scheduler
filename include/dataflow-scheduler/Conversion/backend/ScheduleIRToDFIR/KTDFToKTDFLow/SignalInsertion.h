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

#ifndef DATAFLOW_SCHEDULER_CONVERSION_KTDFTOKTDFLOW_SIGNALINSERTION_H_
#define DATAFLOW_SCHEDULER_CONVERSION_KTDFTOKTDFLOW_SIGNALINSERTION_H_

#include <map>

#include "dataflow-scheduler/Conversion/backend/ScheduleIRToDFIR/KTDFToKTDFLow/StageToUnitsMap.h"
#include "dataflow-scheduler/Dialect/KTDF/Analysis/GlobalStageDAG.h"
#include "dataflow-scheduler/Dialect/KTDF/KTDF.h"
#include "dataflow-scheduler/Utils/SchedulerExtContext.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/Builders.h"

namespace scheduler {

/// Insert signal operations between stages with scratchpad conflicts.
/// Narrows signal units to immediate leaf/root stages when producer or
/// consumer wraps a nested pipeline.
mlir::LogicalResult insertSignalsInPipeline(
    mlir::ktdf::PipelineOp pipeline,
    const llvm::SmallVector<mlir::ktdf::StageOp, 8>& sorted_stages,
    const StageToUnitsMap& stage_to_units,
    const std::map<std::pair<mlir::Operation*, mlir::Operation*>,
                   llvm::SmallVector<scheduler::ResourceType, 2>>& conflicts,
    mlir::OpBuilder& builder);

}  // namespace scheduler

#endif  // DATAFLOW_SCHEDULER_CONVERSION_KTDFTOKTDFLOW_SIGNALINSERTION_H_
