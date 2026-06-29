//===-- DataPath.h ----------------------------------------------*- c++ -*-===//
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
// DataPathAnalysis: SSA-based data path tracing within ktdf.pipeline.
//
// Traces the data path for a linalg.generic input operand backward through
// SSA within an enclosing ktdf.pipeline, building an ordered sequence of
// DataPathHop entries from source memref to target value.
//
//===----------------------------------------------------------------------===//

#ifndef DATAFLOW_SCHEDULER_DIALECT_KTDF_ANALYSIS_DATAPATH_H_
#define DATAFLOW_SCHEDULER_DIALECT_KTDF_ANALYSIS_DATAPATH_H_

#include <llvm/ADT/SmallVector.h>

#include <optional>

#include "dataflow-scheduler/Dialect/KTDF/KTDF.h"

namespace mlir::ktdf {

/// One hop in the data path: a data_transfer op and the private resource
/// it writes into (memref buffer or FIFO slot value).
struct DataPathHop {
  DataTransferOp transfer;
  Value private_resource;  // memref buffer or FIFO slot value
};

/// A complete data path from a source memref to a target value
/// (the linalg.generic input operand), with all intermediate hops
/// in source→target order.
struct DataPath {
  Value source;                   // originating memref (e.g. DDR)
  Value target;                   // linalg.generic input operand
  SmallVector<DataPathHop> hops;  // ordered source→target
};

/// Trace the data path for a single linalg.generic input operand backward
/// through SSA within the enclosing ktdf.pipeline.
///
/// Returns nullopt if the path does not match the expected pipeline
/// structure (read_from_fifo → fifo.allocate → data_transfer chain → DDR).
/// This is not a hard error — the caller decides whether to warn or skip.
auto traceDataPath(Value generic_input, PipelineOp pipeline)
    -> std::optional<DataPath>;

}  // namespace mlir::ktdf

#endif  // DATAFLOW_SCHEDULER_DIALECT_KTDF_ANALYSIS_DATAPATHANALYSIS_H_
