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
// Tablegen Attribute Definitions
//===----------------------------------------------------------------------===//

#define GET_ATTRDEF_CLASSES
#include "src/Dialects/KTDF/KTDFTypesAttributes.cpp.inc"

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
  std::string fifo_type_str;
  if (parser.parseKeywordOrString(&fifo_type_str)) return Type();

  // Convert string to FifoType enum
  auto fifo_type = symbolizeFifoType(fifo_type_str);
  if (!fifo_type) {
    parser.emitError(parser.getCurrentLocation(), "invalid FIFO type: ")
        << fifo_type_str;
    return Type();
  }

  if (parser.parseComma()) return Type();

  // Parse dimension (NxType or ?xType)
  int64_t num_elements;
  Type element_type;

  // Try to parse '?' for dynamic dimension
  if (succeeded(parser.parseOptionalQuestion())) {
    num_elements = mlir::ShapedType::kDynamic;
  } else {
    // Parse static dimension
    if (parser.parseInteger(num_elements)) return Type();
  }

  // Parse 'x' - use parseXInDimensionList which handles 'x' in dimension
  // context
  if (parser.parseXInDimensionList()) return Type();

  // Parse element type
  if (parser.parseType(element_type)) return Type();

  if (parser.parseGreater()) return Type();

  return FifoSlotType::getChecked(
      [&] { return parser.emitError(parser.getCurrentLocation()); },
      parser.getContext(), *fifo_type, num_elements, element_type);
}

//===----------------------------------------------------------------------===//
// KTDFDialect Type Registration

//===----------------------------------------------------------------------===//
// KTDFDialect Attribute Registration
//===----------------------------------------------------------------------===//

void KTDFDialect::registerAttributes() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "src/Dialects/KTDF/KTDFTypesAttributes.cpp.inc"
      >();
}
//===----------------------------------------------------------------------===//

void KTDFDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "src/Dialects/KTDF/KTDFTypes.cpp.inc"
      >();
}
