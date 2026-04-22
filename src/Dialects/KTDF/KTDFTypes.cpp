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
// This file defines the types in the KTDF dialect.
//
//===----------------------------------------------------------------------===//

#include "KTDFTypes.hpp"

#include "KTDFOps.hpp"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/TypeSupport.h"

using namespace mlir;
using namespace mlir::ktdf;

//===----------------------------------------------------------------------===//
// Tablegen Type Definitions
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "src/Dialects/KTDF/KTDFTypes.cpp.inc"

//===----------------------------------------------------------------------===//
// FifoSlotType
//===----------------------------------------------------------------------===//

void FifoSlotType::print(AsmPrinter& printer) const {
  printer << "<" << getFifoType() << ", ";
  if (mlir::ShapedType::isDynamic(getNumElements())) {
    printer << "?";
  } else {
    printer << getNumElements();
  }
  printer << "x";
  printer.printType(getElementType());
  printer << ">";
}

Type FifoSlotType::parse(AsmParser& parser) {
  if (parser.parseLess()) return Type();

  // Parse the FifoType enum as a string keyword
  std::string fifoTypeStr;
  if (parser.parseKeywordOrString(&fifoTypeStr)) return Type();

  // Convert string to FifoType enum
  auto fifoType = symbolizeFifoType(fifoTypeStr);
  if (!fifoType) {
    parser.emitError(parser.getCurrentLocation(), "invalid FIFO type: ")
        << fifoTypeStr;
    return Type();
  }

  if (parser.parseComma()) return Type();

  // Parse dimension (NxType or ?xType)
  int64_t numElements;
  Type elementType;

  // Try to parse '?' for dynamic dimension
  if (succeeded(parser.parseOptionalQuestion())) {
    numElements = mlir::ShapedType::kDynamic;
  } else {
    // Parse static dimension
    if (parser.parseInteger(numElements)) return Type();
  }

  // Parse 'x' - use parseXInDimensionList which handles 'x' in dimension
  // context
  if (parser.parseXInDimensionList()) return Type();

  // Parse element type
  if (parser.parseType(elementType)) return Type();

  if (parser.parseGreater()) return Type();

  return FifoSlotType::getChecked(
      [&] { return parser.emitError(parser.getCurrentLocation()); },
      parser.getContext(), *fifoType, numElements, elementType);
}

//===----------------------------------------------------------------------===//
// KTDFDialect Type Registration
//===----------------------------------------------------------------------===//

void KTDFDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "src/Dialects/KTDF/KTDFTypes.cpp.inc"
      >();
}
