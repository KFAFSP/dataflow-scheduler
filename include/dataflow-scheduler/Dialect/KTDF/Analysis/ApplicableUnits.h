//===-- ApplicableUnits.h ---------------------------------------*- c++ -*-===//
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
// Compute the union of applicable_units over a pipeline's immediate stages.
//
//===----------------------------------------------------------------------===//

#ifndef DATAFLOW_SCHEDULER_DIALECT_KTDF_ANALYSIS_APPLICABLEUNITS_H_
#define DATAFLOW_SCHEDULER_DIALECT_KTDF_ANALYSIS_APPLICABLEUNITS_H_

#include <llvm/ADT/SetVector.h>

#include "dataflow-scheduler/Dialect/KTDF/KTDF.h"

namespace mlir::ktdf {

using ResourceType = mlir::Attribute;

/// Union the `applicable_units` of every immediate `ktdf.stage` child of
/// `pipeline`. Does not descend into nested pipelines or stage bodies.
/// Skips non-stage children (e.g., `ktdf.private`).
///
/// Asserts that every immediate stage has its `applicable_units` attribute
/// set; absence is treated as an upstream-pass invariant violation and
/// triggers `llvm::report_fatal_error`.
///
/// Returns an empty SetVector when the pipeline has no immediate stages
/// (e.g., a pipeline containing only `ktdf.private`).
auto collectPipelineApplicableUnits(PipelineOp pipeline)
    -> llvm::SmallSetVector<ResourceType, 4>;

}  // namespace mlir::ktdf

#endif  // DATAFLOW_SCHEDULER_DIALECT_KTDF_ANALYSIS_APPLICABLEUNITS_H_
