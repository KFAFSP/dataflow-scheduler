//===-- DataflowTypes.td -- DataflowTypes Dialect Operation Definitions -*- tablegen ----==//
//
// IBM Confidential
// (C) Copyright IBM Corp. 2022, 2023
//
// =============================================================================
//
// This file defines the dataflow types in the MLIR operation set.
//
//===----------------------------------------------------------------------===//

#ifndef DIALECT_DATAFLOW_TYPES_TD
#define DIALECT_DATAFLOW_TYPES_TD

#ifndef OP_BASE
include "mlir/IR/OpBase.td"
#endif // OP_BASE
include "mlir/IR/EnumAttr.td"
include "src/Dialects/Dataflow/DataflowIBMFPTypes.td"
include "src/Dialects/Dataflow/DataflowEnums.td"

def DataflowRoutingDirectionAttr : EnumAttr<Dataflow_Dialect, DataflowRoutingDirection, "direction">; 

#endif // DIALECT_DATAFLOW_TYPES_TD
