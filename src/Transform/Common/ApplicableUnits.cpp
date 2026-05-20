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

#include "src/Transform/Common/ApplicableUnits.hpp"

#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

namespace ktdf {
namespace common {

llvm::SmallSetVector<mlir::ktdf::ComponentType, 4>
collectPipelineApplicableUnits(mlir::ktdf::PipelineOp pipeline) {
  llvm::SmallSetVector<mlir::ktdf::ComponentType, 4> result;

  for (mlir::ktdf::StageOp stage : pipeline.getStages()) {
    auto attr = stage.getApplicableUnitsAttr();
    if (!attr) {
      std::string msg;
      llvm::raw_string_ostream os(msg);
      os << "ktdf.stage at " << stage.getLoc()
         << " is missing required `applicable_units` attribute "
            "(needed by ktdf::common::collectPipelineApplicableUnits)";
      llvm::report_fatal_error(llvm::Twine(os.str()));
    }
    for (mlir::ktdf::ComponentType unit : attr.getValue()) {
      result.insert(unit);
    }
  }

  return result;
}

}  // namespace common
}  // namespace ktdf
