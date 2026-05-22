//===-- AgenTypes.td -- AgenTypes Dialect Operation Definitions -*- tablegen ----==//
//
// Copyright 2022- The IBM Research Authors.
//  -- Deep/Agen Code Constructor using MLIR infrastructure
// for IBM's Sentient AI Accelerator
// =============================================================================
//
// This file defines the Agen types in the MLIR operation set.
//
//===----------------------------------------------------------------------===//

#ifndef DIALECT_AGEN_TYPES_TD
#define DIALECT_AGEN_TYPES_TD

#ifndef OP_BASE
include "mlir/IR/OpBase.td"
#endif // OP_BASE
include "mlir/IR/EnumAttr.td"

include "mlir/Dialect/LLVMIR/LLVMOpBase.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/LoopLikeInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "src/Dialects/Agen/AgenEnums.td"

def Agen_Dialect : Dialect {
  let name = "agen";
  let cppNamespace = "::mlir::agen";

  let useDefaultAttributePrinterParser = 1;

  let dependentDialects = [
    "dataflow::DataflowDialect",
    "vectorchain::VectorChainDialect"
  ];
}

// Base class for Agen dialect operations. This operation inherits from the
// base `Op` class in OpBase.td, and provides:
//   * The parent dialect of the operation.
//   * The mnemonic for the operation, or the name without the dialect prefix.
//   * A list of traits for the operation.
class Agen_Op<string mnemonic, list<Trait> traits = []> :
    Op<Agen_Dialect, mnemonic, traits> {
}

def AgenRoutingDirectionAttr : EnumAttr<Agen_Dialect, AgenRoutingDirection, "direction">;

#endif // DIALECT_AGEN_TYPES_TD
