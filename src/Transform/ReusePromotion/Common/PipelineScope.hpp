//===----------------------------------------------------------------------===//
//
// Copyright 2026 The KTIR Scheduler Authors.
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
// Pipeline Enclosing Scope computation for reuse-promotion passes.
//===----------------------------------------------------------------------===//

#ifndef KTIR_SCHEDULER_TRANSFORM_REUSEPROMOTION_PIPELINESCOPE_HPP
#define KTIR_SCHEDULER_TRANSFORM_REUSEPROMOTION_PIPELINESCOPE_HPP

#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "src/Dialects/KTDF/KTDFOps.hpp"

namespace ktdf {
namespace reuse {

/// The set of scf.for loops enclosing a pipeline, up to the scope boundary.
/// Elements are ordered innermost-to-outermost (index 0 is the loop directly
/// containing the pipeline, last element is the outermost loop still inside
/// the scope).
struct PipelineEnclosingScope {
  llvm::SmallVector<mlir::scf::ForOp> loops;
};

/// Walk outward from `pipeline` through enclosing ops. An scf.for qualifies
/// as part of the scope iff its body contains nothing except the chain that
/// leads back to the pipeline (i.e., the pipeline itself, or another
/// qualifying scf.for, plus that loop's terminator scf.yield).
///
/// The walk stops at (returns without extending) any of:
///   - a non-scf.for parent (e.g. scf.if, func.func, ktdf.stage)
///   - a parent whose body contains siblings beside the chain
///
/// Returns an empty scope if the pipeline has no enclosing scf.for that
/// qualifies.
PipelineEnclosingScope computePipelineEnclosingScope(
    mlir::ktdf::PipelineOp pipeline);

}  // namespace reuse
}  // namespace ktdf

#endif  // KTIR_SCHEDULER_TRANSFORM_REUSEPROMOTION_PIPELINESCOPE_HPP
