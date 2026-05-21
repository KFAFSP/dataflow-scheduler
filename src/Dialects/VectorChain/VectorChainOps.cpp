//===------------------------ VectorChainOps.cpp -------------------------===//
//
// IBM Confidential
// (C) Copyright IBM Corp. 2021, 2025
//
// =============================================================================
//
// This file defines the Dataflow operations in the MLIR operation set.
//
//===----------------------------------------------------------------------===//
#ifndef SRC_DIALECT_VECTORCHAIN_OPS_H
#define SRC_DIALECT_VECTORCHAIN_OPS_H

#include "src/Dialects/VectorChain/VectorChainOps.hpp"

#include "llvm/ADT/TypeSwitch.h"
#include "mlir/IR/DialectImplementation.h"
#include "src/Dialects/Dataflow/DataflowOps.hpp"

/// Auto-generated includes.
#include "src/Dialects/VectorChain/VectorChainEnums.cpp.inc"
#include "src/Dialects/VectorChain/VectorChainOpsDialect.cpp.inc"

using namespace mlir;
using namespace mlir::vectorchain;

//===----------------------------------------------------------------------===//
// VectorChainOpsDialect
//===----------------------------------------------------------------------===//

/// Dialect creation, the instance will be owned by the context. This is the
/// point of registration of custom types and operations for the dialect.
void VectorChainDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "src/Dialects/VectorChain/VectorChainOps.cpp.inc"
      >();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "src/Dialects/VectorChain/VectorChainAttributes.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// ConstantBitstreamOp
//===----------------------------------------------------------------------===//
ParseResult ConstantBitstreamOp::parse(OpAsmParser& parser,
                                       OperationState& state) {
  auto& builder = parser.getBuilder();

  // Parse the attributes and the result type.
  Type op_type;
  auto op_result = parser.parseOptionalAttrDict(state.attributes) ||
                   parser.parseColonType(op_type) ||
                   parser.addTypeToList(op_type, state.types);

  return failure(op_result);
}

void ConstantBitstreamOp::print(OpAsmPrinter& p) {
  auto& op = *this;

  p << " {value = [";

  // Print each of the interpreted_value elements as hex.
  auto values = op.getValue().getValue();
  unsigned values_size = values.size();
  for (unsigned i = 0; i < values_size; ++i) {
    auto int_attr = mlir::cast<IntegerAttr>(values[i]);
    int64_t int_val = int_attr.getInt();
    p << "0x" << llvm::Twine::utohexstr((uint64_t)int_val);
    if (i < values_size - 1) p << ", ";
  }

  p << "]}";

  // Print the return type.
  p << " : " << op.getType();
}

LogicalResult ConstantBitstreamOp::verify() {
  auto& op = *this;

  // The number of elements in the interpreted_value array should match the
  // number of elements in the result type. The result type is expected to be a
  // vector.
  int value_size = op.getValue().size();
  auto result_type = op.getResult().getType();
  mlir::Type elements_type;
  int num_of_elements = 0;
  if (auto const_vtype = mlir::dyn_cast<VectorType>(result_type)) {
    elements_type = const_vtype.getElementType();
    num_of_elements = const_vtype.getNumElements();
  } else if (auto const_ibmvtype =
                 mlir::dyn_cast<dataflow::IBMVectorFloatType>(result_type)) {
    elements_type = const_ibmvtype.getElementType();
    num_of_elements = const_ibmvtype.getNumElementTypes();
  } else {
    op->emitOpError("result element type is not vector type.");
    return failure();
  }
  if (!elements_type.isIntOrFloat()) {
    op->emitOpError("result element type is not ints or floats");
    return failure();
  }

  if (num_of_elements != value_size) {
    op->emitOpError("result does not match value array");
    return failure();
  }

  // The bitwidths of the elements of the value attribute should not exceed the
  // bitwidth of the output vector element type.
  auto values = op.getValue().getValue();
  unsigned result_bitwidth = elements_type.getIntOrFloatBitWidth();
  for (int i = 0; i < value_size; ++i) {
    auto int_attr = mlir::dyn_cast<IntegerAttr>(values[i]);
    if (!int_attr) {
      op->emitOpError(
          "value attribute should only contain integer representations");
      return failure();
    }

    int64_t int_val = int_attr.getInt();
    int64_t mask = pow(2, result_bitwidth) - 1;
    if ((int_val & mask) != int_val) {
      op->emitOpError(
          "value attribute contains element that exceeds bitwidth of output "
          "element type");
      return failure();
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// CastOp
//===----------------------------------------------------------------------===//
LogicalResult CastOp::verify() {
  auto& op = *this;

  auto result_type = mlir::dyn_cast<VectorType>(op.getResult().getType());
  if (!result_type) {
    op->emitOpError("result type is not a vector");
    return failure();
  }
  auto input_type = mlir::dyn_cast<VectorType>(op.getInput().getType());
  if (!input_type) {
    op->emitOpError("input type is not a vector");
    return failure();
  }

  // The number of elements in the result type should be equivalent to the
  // number of elements in the input type.
  int result_size = result_type.getNumElements();
  int input_size = input_type.getNumElements();
  if (result_size != input_size) {
    op->emitOpError("result size does not match input size");
    return failure();
  }
  return success();
}

//===----------------------------------------------------------------------===//
// ShuffleOp
//===----------------------------------------------------------------------===//
LogicalResult ShuffleOp::verify() {
  auto& op = *this;

  auto result_type = mlir::dyn_cast<VectorType>(op.getResult().getType());
  if (!result_type) {
    op->emitOpError("result type is not a vector");
    return failure();
  }

  // The number of elements in the result type should be equivalent to the
  // product of the number of elements in the indices and the number of
  // repetitions.
  auto indices = op.getIndices();
  int indices_size = indices.size();
  int result_size = result_type.getNumElements();
  if (result_size != indices_size * op.getRepetition()) {
    op->emitOpError("result does not scale with indices and repetitions");
    return failure();
  }

  auto input_type = op.getInput().getType();
  mlir::Type input_elements_type;
  int input_num_of_elements = 0;
  // The input to shuffle is expected to be a VectorTyep or IBMVectorFloatType.
  if (auto const_vtype = mlir::dyn_cast<VectorType>(input_type)) {
    input_elements_type = const_vtype.getElementType();
    input_num_of_elements = const_vtype.getNumElements();
  } else if (auto const_ibmvtype =
                 mlir::dyn_cast<dataflow::IBMVectorFloatType>(input_type)) {
    input_elements_type = const_ibmvtype.getElementType();
    input_num_of_elements = const_ibmvtype.getNumElementTypes();
  } else {
    op->emitOpError(
        "input to vectorchain.shuffle should be VectorType or "
        "IBMVectorFloatType.");
    return failure();
  }
  // The element type of the input should match the type of the output (type
  // only, not necessarily number of elements).
  if (!isa<dataflow::IBMVectorFloatType>(input_type) &&
      result_type.getElementType() != input_elements_type) {
    op->emitOpError(
        "input element type does not match output element "
        "type");
    return failure();
  }
  // The elements referred to in the indices should not refer to out of bounds
  // elements from the input. Element values of -1 are still valid as these are
  // used to indicate padding.
  int max_element = input_num_of_elements - 1;
  for (int i = 0; i < indices_size; ++i) {
    auto element = mlir::dyn_cast<IntegerAttr>(indices[i]);
    if (!element) {
      op->emitOpError("indices element is not integer");
      return failure();
    }

    int64_t element_value = element.getInt();
    if (element_value < -1 || element_value > max_element) {
      op->emitOpError("out of bounds indices element");
      return failure();
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// CoalesceOp
//===----------------------------------------------------------------------===//
#ifdef COMMENT_OUT_COALESCE_STORE
#else
static LogicalResult verify(CoalesceOp op) {
  // drop_frist_map can only be constant or mod expr
  auto drop_first_map = op.drop_first_map();
  if (drop_first_map.getNumResults() != 1) {
    op->emitOpError("drop_first_map must have 1 result.");
    return failure();
  }
  auto drop_first_expr = drop_first_map.getResult(0);
  if (auto binary_expr = mlir::dyn_cast<AffineBinaryOpExpr>(drop_first_expr)) {
    if (binary_expr.getKind() != AffineExprKind::Mod) {
      op->emitOpError("drop_first_map must be Mod expression.");
      return failure();
    }
    auto rhs = binary_expr.getRHS();
    if (auto rhs_const = dyn_cast<AffineConstantExpr>(rhs)) {
      if (rhs_const.getValue() != op.subword_len()) {
        op->emitOpError("modulus must be equal to subword_len.");
        return failure();
      }
    }
  } else if (!mlir::dyn_cast<AffineConstantExpr>(drop_first_expr)) {
    op->emitOpError("drop_first_map must be either constant or mod.");
    return failure();
  }
  // check input and output vectors' sizes match the coalesce requirement
  auto subword_len = op.subword_len();
  auto stride = op.stride();
  auto elements_per_stick = 1024 / op.getType().getElementTypeBitWidth();
  auto output_element_num = op.getType().getNumElements();
  auto input_element_num =
      mlir::cast<VectorType>(op.data().getType()).getNumElements();
  auto coalesced_element_num =
      (input_element_num - elements_per_stick) / stride;
  if (output_element_num != coalesced_element_num) {
    op->emitOpError(
        "output vector size is not consistent with coalesce constrains.");
    return failure();
  }
  return success();
}
#endif
//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//
#define GET_ATTRDEF_CLASSES
#include "src/Dialects/VectorChain/VectorChainAttributes.cpp.inc"
#define GET_OP_CLASSES
#include "src/Dialects/VectorChain/VectorChainOps.cpp.inc"
#endif
