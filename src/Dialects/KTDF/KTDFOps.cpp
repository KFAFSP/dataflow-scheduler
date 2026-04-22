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
    if (slotType.isDynamicNumElements()) {
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

//===----------------------------------------------------------------------===//
// DataTransferOp
//===----------------------------------------------------------------------===//

LogicalResult DataTransferOp::verify() {
  auto sourceType = getSource().getType();
  auto destType = getDestination().getType();

  // Verify that source and destination are not both FIFOs
  bool sourceFifo = llvm::isa<FifoSlotType>(sourceType);
  bool destFifo = llvm::isa<FifoSlotType>(destType);

  if (sourceFifo && destFifo) {
    return emitOpError("cannot transfer directly from FIFO to FIFO");
  }

  // Verify that at least one is a memref or FIFO
  bool sourceMemRef = llvm::isa<MemRefType>(sourceType);
  bool destMemRef = llvm::isa<MemRefType>(destType);

  if (!sourceMemRef && !sourceFifo) {
    return emitOpError("source must be a memref or FIFO slot");
  }

  if (!destMemRef && !destFifo) {
    return emitOpError("destination must be a memref or FIFO slot");
  }

  // Verify that FIFO slots have no indices specified
  if (sourceFifo && !getSourceIndices().empty()) {
    return emitOpError("FIFO slot source cannot have indices specified");
  }

  if (destFifo && !getDestIndices().empty()) {
    return emitOpError("FIFO slot destination cannot have indices specified");
  }

  // Get element types for compatibility checking
  Type sourceElemType;
  Type destElemType;

  if (sourceMemRef) {
    sourceElemType = llvm::cast<MemRefType>(sourceType).getElementType();
  } else if (sourceFifo) {
    sourceElemType = llvm::cast<FifoSlotType>(sourceType).getElementType();
  }

  if (destMemRef) {
    destElemType = llvm::cast<MemRefType>(destType).getElementType();
  } else if (destFifo) {
    destElemType = llvm::cast<FifoSlotType>(destType).getElementType();
  }

  // Verify element type compatibility
  if (sourceElemType && destElemType && sourceElemType != destElemType) {
    return emitOpError("source and destination element types must match, got ")
           << sourceElemType << " and " << destElemType;
  }

  // Verify indices and sizes are consistent for memrefs
  auto sourceMemRefType = llvm::dyn_cast<MemRefType>(sourceType);
  if (sourceMemRefType) {
    unsigned rank = sourceMemRefType.getRank();
    if (!getSourceIndices().empty() && getSourceIndices().size() != rank) {
      return emitOpError("source indices count (")
             << getSourceIndices().size() << ") must match source memref rank ("
             << rank << ")";
    }
  }

  auto destMemRefType = llvm::dyn_cast<MemRefType>(destType);
  if (destMemRefType) {
    unsigned rank = destMemRefType.getRank();
    if (!getDestIndices().empty() && getDestIndices().size() != rank) {
      return emitOpError("destination indices count (")
             << getDestIndices().size()
             << ") must match destination memref rank (" << rank << ")";
    }
  }

  return success();
}
