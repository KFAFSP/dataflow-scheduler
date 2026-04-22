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
//
// This file defines the operations in the KTDF operation set.
//
//===----------------------------------------------------------------------===//
#include "KTDFOps.hpp"

#include <iostream>
#include <iterator>
#include <regex>

#include "KTDFTypes.hpp"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "src/Dialects/KTDF/KTDFEnums.cpp.inc"
#include "src/Dialects/KTDF/KTDFOpsDialect.cpp.inc"

using namespace mlir;
using namespace mlir::ktdf;

//===----------------------------------------------------------------------===//
// KTDFDialect
//===----------------------------------------------------------------------===//

void KTDFDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "src/Dialects/KTDF/KTDFOps.cpp.inc"

      >();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "src/Dialects/KTDF/KTDFOpsAttributes.cpp.inc"

      >();
  registerTypes();
}

//===----------------------------------------------------------------------===//
// Tablegen Definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "src/Dialects/KTDF/KTDFOps.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "src/Dialects/KTDF/KTDFOpsAttributes.cpp.inc"

//===----------------------------------------------------------------------===//
// PipelineOp
//===----------------------------------------------------------------------===//

PipelineOp::StageRange PipelineOp::getStages() {
  return getBodyBlock()->getOps<StageOp>();
}

unsigned PipelineOp::getNumStages() {
  auto stages = getStages();
  return std::distance(stages.begin(), stages.end());
}

//===----------------------------------------------------------------------===//
// FifoAllocateOp
//===----------------------------------------------------------------------===//

LogicalResult FifoAllocateOp::verify() {
  auto dynamicSizes = getDynamicSizes();
  auto slots = getSlots();

  // Count the number of dynamic sizes in the result types
  unsigned numDynamicSizes = 0;
  for (auto slot : slots) {
    auto slotType = mlir::cast<FifoSlotType>(slot.getType());
    if (slotType.isDynamicSize()) {
      numDynamicSizes++;
    }
  }

  // Verify that the number of dynamic size operands matches the number of
  // dynamic sizes in the result types
  if (dynamicSizes.size() != numDynamicSizes) {
    return emitOpError("number of dynamic size operands (")
           << dynamicSizes.size()
           << ") must match number of dynamic sizes in result types ("
           << numDynamicSizes << ")";
  }

  return success();
}
