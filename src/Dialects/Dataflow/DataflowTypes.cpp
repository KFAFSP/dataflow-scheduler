//===------- DataflowTypes.cpp - Dataflow Dialect Types -------------------===//
//
// IBM Confidential
// (C) Copyright IBM Corp. 2023, 2024
//
//===----------------------------------------------------------------------===//
//
// This file contains types defined by the Dataflow Dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Dialects/Dataflow/DataflowTypes.hpp"

#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "src/Dialects/Dataflow/DataflowOps.hpp"

using namespace mlir;
using namespace dataflow;

//===----------------------------------------------------------------------===//
// IBMVectorFloatType
//===----------------------------------------------------------------------===//

Type DataflowDialect::parseType(DialectAsmParser& parser) const {
  mlir::Type elementType;
  mlir::Type elementType_stored;
  auto& builder = parser.getBuilder();
  if (!parser.parseOptionalKeyword("vector")) {
    if (parser.parseLess()) return Type();
    size_t numOfElement;
    if (parser.parseInteger(numOfElement)) return Type();
    if (parser.parseKeyword("xibm") || parser.parseLess()) return Type();
    if (parser.parseType(elementType)) return Type();
    if (isa<Float16Type>(elementType)) {
      elementType_stored = builder.getI16Type();
    }
    if (parser.parseGreater()) return Type();
    if (parser.parseGreater()) return Type();
    return IBMVectorFloatType::get(
        std::make_pair(elementType_stored, numOfElement));
  } else if (!(parser.parseKeyword("ibm") || parser.parseLess())) {
    if (parser.parseType(elementType)) return Type();
    if (isa<Float16Type>(elementType)) {
      elementType_stored = builder.getI16Type();
    }
    if (parser.parseGreater()) return Type();
    return IBMFloatType::get(elementType_stored);
  }
  return Type();
}

void DataflowDialect::printType(mlir::Type type,
                                mlir::DialectAsmPrinter& printer) const {
  if (isa<IBMFloatType>(type)) {
    IBMFloatType ibm_float = mlir::cast<IBMFloatType>(type);
    printer << "ibm<";
    if (auto ibm_float_stored =
            llvm::dyn_cast<IntegerType>(ibm_float.getElementType())) {
      if (ibm_float_stored.getIntOrFloatBitWidth() == 16) {
        printer << "f16";
      }
    }
    printer << ">";
  } else {
    IBMVectorFloatType ibm_vfloat = mlir::cast<IBMVectorFloatType>(type);
    printer << "vector<";
    printer << ibm_vfloat.getNumElementTypes();
    printer << "xibm<";
    if (auto ibm_vfloat_stored =
            llvm::dyn_cast<IntegerType>(ibm_vfloat.getElementType())) {
      if (ibm_vfloat_stored.getIntOrFloatBitWidth() == 16) {
        printer << "f16";
      }
    }
    printer << ">>";
  }
}
