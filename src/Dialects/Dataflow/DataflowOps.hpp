//===--------------------------- DataflowOps.hpp --------------------------===//
//
// IBM Confidential
// (C) Copyright IBM Corp. 2021, 2024
//
// =============================================================================
//
// This file declares the Dataflow operations in the MLIR operation set.
//
//===----------------------------------------------------------------------===//

#ifndef DIALECT_DATAFLOW_HPP_
#define DIALECT_DATAFLOW_HPP_

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "src/Dialects/Dataflow/DataflowTypes.hpp"

/// Auto-generated includes.
#include "src/Dialects/Dataflow/DataflowEnums.hpp.inc"
#include "src/Dialects/Dataflow/DataflowOpsDialect.hpp.inc"
#define GET_ATTRDEF_CLASSES
#include "src/Dialects/Dataflow/DataflowAttributes.hpp.inc"

namespace mlir {
namespace dataflow {

static StringRef getDbgNameAttrStrName() { return "dbgName"; }

static void print(OpAsmPrinter& printer, Operation* op);

}  // namespace dataflow
}  // end namespace mlir

/// Include the auto-generated header file containing the declarations of the
/// Dataflow operations.
#define GET_OP_CLASSES
#include "src/Dialects/Dataflow/DataflowOps.hpp.inc"

#endif  // DIALECT_DATAFLOW_HPP_
