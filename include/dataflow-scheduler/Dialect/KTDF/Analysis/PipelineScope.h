//===-- PipelineScope.h -----------------------------------------*- c++ -*-===//
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
// Pipeline Enclosing Scope computation, shared across transform passes.
//
//===----------------------------------------------------------------------===//

#ifndef DATAFLOW_SCHEDULER_DIALECT_KTDF_ANALYSIS_PIPELINESCOPE_H_
#define DATAFLOW_SCHEDULER_DIALECT_KTDF_ANALYSIS_PIPELINESCOPE_H_

#include <llvm/ADT/SmallVector.h>
#include <mlir/Dialect/SCF/IR/SCF.h>

#include "dataflow-scheduler/Dialect/KTDF/KTDF.h"

namespace mlir::ktdf {

/// The set of scf.for loops enclosing a pipeline, up to the scope boundary.
/// Elements are ordered innermost-to-outermost (index 0 is the loop directly
/// containing the pipeline, last element is the outermost loop still inside
/// the scope).
struct PipelineEnclosingScope {
  SmallVector<scf::ForOp> loops;
};

/// Walk outward from `pipeline` through enclosing ops. An scf.for qualifies
/// as part of the scope iff its body contains no ktdf.pipeline op other than
/// those transitively inside the chain member leading back to `pipeline`.
/// Imperfectly-nested loops (with sibling index-computation ops) are allowed
/// as long as they introduce no peer pipelines.
///
/// The walk stops at (returns without extending) any of:
///   - a non-scf.for parent: ktdf.stage, ktdf.pipeline, ktdf.parallel,
///     scf.parallel, func.func, or any other non-scf.for op
///   - a parent scf.for whose body contains a peer ktdf.pipeline (anywhere
///     in the subtree, excluding the chain back to `pipeline`)
///
/// Returns an empty scope if the pipeline has no enclosing scf.for that
/// qualifies.
auto getPipelineEnclosingScope(PipelineOp pipeline) -> PipelineEnclosingScope;

}  // namespace mlir::ktdf

#endif  // DATAFLOW_SCHEDULER_TRANSFORMS_COMMON_PIPELINESCOPE_H_
