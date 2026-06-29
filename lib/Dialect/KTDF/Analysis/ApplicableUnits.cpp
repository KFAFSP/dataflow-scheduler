//===-- ApplicableUnits.cpp -------------------------------------*- c++ -*-===//
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
// This file implements the applicable units analysis.
//
//===----------------------------------------------------------------------===//

#include "dataflow-scheduler/Dialect/KTDF/Analysis/ApplicableUnits.h"

#include <llvm/Support/ErrorHandling.h>
#include <llvm/Support/raw_ostream.h>

using namespace mlir;
using namespace mlir::ktdf;
using ResourceType = mlir::Attribute;

auto mlir::ktdf::collectPipelineApplicableUnits(PipelineOp pipeline)
    -> llvm::SmallSetVector<ResourceType, 4> {
  llvm::SmallSetVector<ResourceType, 4> result;

  for (StageOp stage : pipeline.getStages()) {
    auto attr = stage.getApplicableUnitsAttr();
    if (!attr) {
      std::string msg;
      llvm::raw_string_ostream os(msg);
      os << "ktdf.stage at " << stage.getLoc()
         << " is missing required `applicable_units` attribute "
            "(needed by ktdf::common::collectPipelineApplicableUnits)";
      llvm::report_fatal_error(llvm::Twine(os.str()));
    }
    for (mlir::Attribute unit : attr.getValue()) {
      result.insert(unit);
    }
  }

  return result;
}