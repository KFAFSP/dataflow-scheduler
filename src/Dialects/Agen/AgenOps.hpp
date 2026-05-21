//===--------------------------- AgenOps.hpp-------------------------------===//
//
// IBM Confidential
// (C) Copyright IBM Corp. 2021, 2024
//
// =============================================================================
//
// This file declares the Agen operations in the MLIR operation set.
//
//===----------------------------------------------------------------------===//

#ifndef DIALECT_AGEN_HPP_
#define DIALECT_AGEN_HPP_

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Support/LLVM.h"

/// Auto-generated includes.
#include "src/Dialects/Agen/AgenEnums.hpp.inc"
#include "src/Dialects/Agen/AgenOpsDialect.hpp.inc"
#define GET_ATTRDEF_CLASSES
#include "src/Dialects/Agen/AgenAttributes.hpp.inc"

static llvm::StringRef getDbgNameAttrStrName() { return "dbgName"; }

/// Verify common indexing invariants of affine.load, affine.store,
/// affine.vector_load and affine.vector_store.
/// Present as part of the AffineOps.cpp
extern mlir::LogicalResult verifyMemoryOpIndexing(
    mlir::Operation* op, mlir::AffineMapAttr map_attr,
    mlir::Operation::operand_range map_operands, mlir::MemRefType mem_ref_type,
    unsigned num_index_operands);

/// Include the auto-generated header file containing the declarations of the
/// Dataflow operations.
#define GET_OP_CLASSES
#include "src/Dialects/Agen/AgenOps.hpp.inc"

#endif  // DIALECT_AGEN_HPP_
