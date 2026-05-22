//===------- DataflowTypes.hpp - Dataflow Dialect Types -------------------===//
//
// IBM Confidential
// (C) Copyright IBM Corp. 2023, 2024
//
//===----------------------------------------------------------------------===//
//
// This file contains types defined by the Dataflow Dialect.
//
//===----------------------------------------------------------------------===//

#ifndef DIALECT_DATAFLOW_TYPES_H
#define DIALECT_DATAFLOW_TYPES_H

#include "mlir/AsmParser/AsmParser.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Types.h"

/// Auto-generated includes
#include "src/Dialects/Dataflow/DataflowOpsTypes.hpp.inc"

namespace mlir::dataflow {

//===----------------------------------------------------------------------===//
// IBMVectorFloatType
//===----------------------------------------------------------------------===//

struct IBMVectorFloatTypeStorage : public mlir::TypeStorage {
  using KeyTy = std::pair<mlir::Type, size_t>;
  IBMVectorFloatTypeStorage(std::pair<mlir::Type, size_t> elementTypes)
      : elementTypes(elementTypes) {}
  bool operator==(const KeyTy& key) const { return key == elementTypes; }
  static KeyTy getKey(std::pair<mlir::Type, size_t> elementTypes) {
    return KeyTy(elementTypes);
  }
  static IBMVectorFloatTypeStorage* construct(
      mlir::TypeStorageAllocator& allocator, const KeyTy& key) {
    return new (allocator.allocate<IBMVectorFloatTypeStorage>())
        IBMVectorFloatTypeStorage(key);
  }
  std::pair<mlir::Type, size_t> elementTypes;
};

class IBMVectorFloatType
    : public mlir::Type::TypeBase<IBMVectorFloatType, mlir::Type,
                                  IBMVectorFloatTypeStorage> {
 public:
  static constexpr ::llvm::StringLiteral name = "dataflow.vector<#xibm<f8/16>";
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"vector<#xibm<f8/16>"};
  }

  using Base::Base;
  static IBMVectorFloatType get(std::pair<mlir::Type, size_t> elementTypes) {
    mlir::MLIRContext* ctx = elementTypes.first.getContext();
    return Base::get(ctx, elementTypes);
  }
  mlir::Type getElementType() { return getImpl()->elementTypes.first; }
  size_t getNumElementTypes() { return getImpl()->elementTypes.second; }
  static bool classof(Type type) {
    return type.getImpl()->getAbstractType().getTypeID() ==
           IBMVectorFloatType::getTypeID();
  }
};

struct IBMFloatTypeStorage : public mlir::TypeStorage {
  using KeyTy = mlir::Type;
  IBMFloatTypeStorage(mlir::Type elementTypes) : elementTypes(elementTypes) {}
  bool operator==(const KeyTy& key) const { return key == elementTypes; }
  static KeyTy getKey(mlir::Type elementTypes) { return KeyTy(elementTypes); }
  static IBMFloatTypeStorage* construct(mlir::TypeStorageAllocator& allocator,
                                        const KeyTy& key) {
    return new (allocator.allocate<IBMFloatTypeStorage>())
        IBMFloatTypeStorage(key);
  }
  mlir::Type elementTypes;
};

class IBMFloatType : public mlir::Type::TypeBase<IBMFloatType, mlir::Type,
                                                 IBMFloatTypeStorage> {
 public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "dataflow.ibm<f8/16>";
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"ibm<f8/16>"};
  }
  static IBMFloatType get(mlir::Type elementTypes) {
    mlir::MLIRContext* ctx = elementTypes.getContext();
    return Base::get(ctx, elementTypes);
  }
  mlir::Type getElementType() { return getImpl()->elementTypes; }
  static bool classof(Type type) {
    return type.getImpl()->getAbstractType().getTypeID() ==
           IBMFloatType::getTypeID();
  }
};

}  // namespace mlir::dataflow

#endif  // DIALECT_DATAFLOW_TYPES_H
