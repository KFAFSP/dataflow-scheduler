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
/// Collapse arith binops computed over uniform.query_map results into a single
/// query_map over a folded constant map.
//
//===----------------------------------------------------------------------===//

#ifndef DATAFLOW_SCHEDULER_CONVERSION_KTDFLOWTODFIR_QUERYMAPARITHCOLLAPSE_H_
#define DATAFLOW_SCHEDULER_CONVERSION_KTDFLOWTODFIR_QUERYMAPARITHCOLLAPSE_H_

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Support/LogicalResult.h"

namespace scheduler {

/// Greedily collapse arith binops over uniform.query_map results within `func`:
///   Rule A: arith.binop(query_map(map, k), const) -> query_map(map', k)
///   Rule B: arith.binop(query_map(mapA, k), query_map(mapB, k)) ->
///           query_map(mapAB, k)   (same key k, both maps all-constant)
/// Folded constant values are materialized at the rewrite site via the
/// PatternRewriter (the patterns are stateless); identical constants across
/// program_units are left for a later CSE pass to dedup.
mlir::LogicalResult collapseArithOverQueryMaps(mlir::func::FuncOp func);

}  // namespace scheduler

#endif  // DATAFLOW_SCHEDULER_CONVERSION_KTDFLOWTODFIR_QUERYMAPARITHCOLLAPSE_H_
