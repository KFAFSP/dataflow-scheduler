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
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Operation.h"

namespace scheduler {

/// Metadata for a cross-iteration pipeline back-edge.
/// store_stage writes a shared scratchpad; load_stage reads it on the next
/// iteration.  dependent_loops are the non-trivial scf.for loops whose
/// induction variables index into the scratchpad; their bounds drive the
/// signal guards.
struct BackEdgeInfo {
  mlir::Operation* store_stage;  ///< leaf stage (producer, writes scratchpad)
  mlir::Operation* load_stage;   ///< root stage (consumer, reads scratchpad)
  /// Non-trivial loops whose IVs appear in the def-use chains of
  /// load_stage_transfers.  Each loop contributes one first/last-iter guard.
  llvm::SmallVector<mlir::scf::ForOp, 2> dependent_loops;
  /// Data transfers in load_stage whose destination memref is written by
  /// store_stage (the scratchpad values that cross loop iterations).
  llvm::SmallVector<mlir::ktdf::DataTransferOp, 4> load_stage_transfers;

  BackEdgeInfo(
      mlir::Operation* store_stage, mlir::Operation* load_stage,
      llvm::SmallVector<mlir::scf::ForOp, 2> dependent_loops,
      llvm::SmallVector<mlir::ktdf::DataTransferOp, 4> load_stage_transfers)
      : store_stage(store_stage),
        load_stage(load_stage),
        dependent_loops(std::move(dependent_loops)),
        load_stage_transfers(std::move(load_stage_transfers)) {}
};

/// Insert signal operations for all scratchpad conflicts found in global_dag.
///
/// For normal (intra-iteration) edges a single unconditional SignalOp is
/// placed after the producer stage.
///
/// For cross-iteration back-edges recorded in `back_edges` two guarded
/// SignalOps are emitted instead:
///   - at the start of load_stage body:
///       scf.if (iv0 != lb0 && iv1 != lb1 && ...) { signal }
///   - at the end   of store_stage body:
///       scf.if (iv0 != ub0-step0 && iv1 != ub1-step1 && ...) { signal }
/// where the conjunction runs over all loops in dependent_loops.
mlir::LogicalResult insertSignals(
    mlir::Location loc, const StageToUnitsMap& stage_to_units,
    const mlir::ktdf::StageDependencyDAG& global_dag,
    const std::map<std::pair<mlir::Operation*, mlir::Operation*>,
                   llvm::SmallVector<scheduler::ResourceType, 2>>& conflicts,
    llvm::ArrayRef<BackEdgeInfo> back_edges);

}  // namespace scheduler

#endif  // DATAFLOW_SCHEDULER_CONVERSION_KTDFTOKTDFLOW_SIGNALINSERTION_H_
