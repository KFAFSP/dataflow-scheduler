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

#ifndef DATAFLOW_SCHEDULER_CONVERSION_KTDFTOKTDFLOW_COMPONENTCLASSIFIER_H_
#define DATAFLOW_SCHEDULER_CONVERSION_KTDFTOKTDFLOW_COMPONENTCLASSIFIER_H_

#include <map>

#include "dataflow-scheduler/Dialect/KTDF/KTDF.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallSet.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinOps.h"

namespace scheduler {

using ResourceType = mlir::Attribute;

/// Classification result: components grouped by parallel/non-parallel
struct ComponentClassification {
  llvm::SetVector<ResourceType> non_parallel_components;
  std::map<mlir::Operation*, llvm::SetVector<ResourceType>>
      parallel_components_map;  // parallel_op -> set of components in that
                                // parallel
};

/// Single-walk classifier that traverses IR once to identify all components
/// and classify them based on whether they're used inside ktdf.parallel
class ComponentClassifier {
 public:
  explicit ComponentClassifier(mlir::func::FuncOp func) : func_(func) {}

  /// Perform classification: collect components from stages, classify
  mlir::LogicalResult classify(
      const llvm::SmallVector<mlir::ktdf::StageOp, 8>& stages,
      ComponentClassification& result);

 private:
  mlir::func::FuncOp func_;
};

}  // namespace scheduler

#endif  // DATAFLOW_SCHEDULER_CONVERSION_KTDFTOKTDFLOW_COMPONENTCLASSIFIER_H_
