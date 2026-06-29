//===-- Utils.cpp -----------------------------------------------*- c++ -*-===//
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
// Implements some common utilities for ktdf dialect analyses.
//
//===----------------------------------------------------------------------===//

#include "dataflow-scheduler/Dialect/KTDF/Analysis/Utils.h"

#include "dataflow-scheduler/Dialect/KTDF/KTDF.h"

using namespace mlir;
using namespace mlir::ktdf;

auto mlir::ktdf::isParallelLoop(scf::ForOp loop) -> bool {
  auto attr = loop->getAttrOfType<LoopTypeAttr>("loop_type");
  if (!attr) {
    return false;
  }
  return attr.getValue() == LoopType::ParallelLoop;
}

auto mlir::ktdf::findParallelParent(StageOp stage) -> Operation* {
  return stage->getParentOfType<ParallelOp>();
}
