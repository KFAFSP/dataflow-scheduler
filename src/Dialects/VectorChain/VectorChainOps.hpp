//===------------------------- VectorChainOps.hpp -------------------------===//
//
// IBM Confidential
// (C) Copyright IBM Corp. 2021, 2023
//
// =============================================================================
//
// This file defines the VectorChain operations in the MLIR operation set.
//
//===----------------------------------------------------------------------===//

#ifndef DIALECT_VECTORCHAIN_HPP_
#define DIALECT_VECTORCHAIN_HPP_

#define COMMENT_OUT_COALESCE_STORE

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"

/// Auto-generated includes.
#include "src/Dialects/VectorChain/VectorChainEnums.hpp.inc"
#include "src/Dialects/VectorChain/VectorChainOpsDialect.hpp.inc"
#define GET_ATTRDEF_CLASSES
#include "src/Dialects/VectorChain/VectorChainAttributes.hpp.inc"

namespace mlir::vectorchain {

std::unique_ptr<Pass> createVectorChainToSentientPTPass();
std::unique_ptr<Pass> createVectorchainToSentientPESFPPass();

}  // namespace mlir::vectorchain

/// Include the auto-generated header file containing the declarations of the
/// VectorChain operations.
#define GET_OP_CLASSES
#include "src/Dialects/VectorChain/VectorChainOps.hpp.inc"

#endif  // DIALECT_VECTORCHAIN_HPP_