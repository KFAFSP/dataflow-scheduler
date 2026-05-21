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

#ifndef DIALECT_DATAFLOW_IBMFP_TYPES_TD
#define DIALECT_DATAFLOW_IBMFP_TYPES_TD

#ifndef OP_BASE
include "mlir/IR/OpBase.td"
#endif // OP_BASE

def Dataflow_Dialect : Dialect {
  let name = "dataflow";
  let cppNamespace = "::mlir::dataflow";
  let useDefaultAttributePrinterParser = 1;
  let extraClassDeclaration =  [{ 
    mlir::Type parseType(mlir::DialectAsmParser &parser) const override;
    void printType(mlir::Type type, mlir::DialectAsmPrinter &printer) const override;
  }];
}

def Dataflow_IBMVectorFloatType : TypeDef<Dataflow_Dialect, "IBMVectorFloat"> {
  let mnemonic = "vector<#xibm<f8/16>";
}

def Dataflow_IBMFloatType : TypeDef<Dataflow_Dialect, "IBMFloat"> {
  let mnemonic = "ibm<f8/16>";
}

def IBMFloat_Type : AnyTypeOf<[Dataflow_IBMFloatType]>;
def IBMVectorFloat_Type : AnyTypeOf<[Dataflow_IBMVectorFloatType]>;

#endif // DIALECT_DATAFLOW_IBMFP_TYPES_TD
