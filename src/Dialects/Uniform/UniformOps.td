//===-- UniformOps.td -- UniformOps Dialect Operation Definitions -*- tablegen ----==//
//
// IBM Confidential
// (C) Copyright IBM Corp. 2023, 2025
//
// =============================================================================
//
// This file defines the sentient operations in the MLIR operation set.
//
//===----------------------------------------------------------------------===//

#ifndef DIALECTS_DIALECT_UNIFORM_OPS_TD

#ifndef OP_BASE
include "mlir/IR/OpBase.td"
#endif // OP_BASE


//include "mlir/Dialect/LLVMIR/LLVMOpBase.td"
include "mlir/Dialect/Affine/IR/AffineMemoryOpInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/LoopLikeInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/BuiltinAttributes.td"


def Uniform_Dialect : Dialect {
  let name = "uniform";
  let cppNamespace = "::mlir::uniform";
}

// Base class for Uniform dialect operations. This operation inherits from the
// base `Op` class in OpBase.td, and provides:
//   * The parent dialect of the operation.
//   * The mnemonic for the operation, or the name without the dialect prefix.
//   * A list of traits for the operation.
class Uniform_Op<string mnemonic, list<Trait> traits = []> :
    Op<Uniform_Dialect, mnemonic, traits> {

}

// Require regions to have uniform.yield.
def ImplicitUniformTerminator
    : SingleBlockImplicitTerminator<"YieldOp">;

def YieldOp : Uniform_Op<"yield", [NoMemoryEffect, Terminator, ReturnLike,
    MemRefsNormalizable]> {
  let summary = "Yield values to parent operation";
  let description = [{
    "uniform.yield yields zero or more SSA values to parent op."
  }];

  let arguments = (ins Variadic<AnyType>:$operands);
  let builders = [
    OpBuilder<(ins), [{ build($_builder, $_state, ValueRange{}); }]>
  ];

  let assemblyFormat = "attr-dict ($operands^ `:` type($operands))?";
}

def UniformizeRegionsOp : Uniform_Op<"uniformize_regions",
      [ImplicitUniformTerminator]> {
  let summary = "uniformize_regions operation";
  let description = [{
    This is the main operation of the uniform dialect to map parallel regions to different units, aiming to consolidate programs from multiple units into a single program.
  }];

  let arguments = (ins Variadic<AnyType>:$units,
                       I32ArrayAttr:$list_sizes,
                       OptionalAttr<I32ArrayAttr>:$regIndices,
                       OptionalAttr<ArrayAttr>:$regLocales);
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region VariadicRegion<AnyRegion>:$regions);

  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = [{
    static StringRef getListSizesAttrStrName() {return "list_sizes";}
    Value getRegionArg(int i) {return getRegion(i).getArgument(0);}
    ValueRange getRegionUnitList(int);
    std::optional<ValueRange> getRegionUnitList(Value);
    std::optional<Region *> getSpecificRegion(Value);
    Region *getRegionFromUnit(Value);

    ArrayAttr getRegIndicesIfExist() {
      if (getRegIndices().has_value()) return getRegIndices().value();
      return nullptr;
    }
    ArrayAttr getRegLocalesIfExist() {
      if (getRegLocales().has_value()) return getRegLocales().value();
      return nullptr;
    }
  }];

  let hasVerifier = 1;
}

def DefImmutableMappingOp : Uniform_Op<"def_immutable_mapping", 
  [NoMemoryEffect, IsolatedFromAbove, AttrSizedOperandSegments]> {
  let summary = "def_immutable_mapping operation";
  let description = [{
    This operation constructs a key-value dictionary to map a SSA variable to another.
  }];

  let arguments = (ins Variadic<AnyType>:$keys,
                       Variadic<AnyType>:$values);
  let results = (outs Index:$result);
  
  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = [{
    std::optional<Value> getValue(Value key);
    SmallVector<std::optional<Value>> getValuesFromKeys(SmallVector<Value>& keys);
    std::vector<std::optional<Value>> getValuesFromKeys(std::vector<Value>& keys);
    SmallVector<mlir::Value> getNonNullValuesFromKeys(SmallVector<mlir::Value> &keys);
    std::vector<mlir::Value> getNonNullValuesFromKeys(std::vector<mlir::Value> &keys);
  }];
  let hasVerifier = 1;
}

def QueryMapOp : Uniform_Op<"query_map", 
  [NoMemoryEffect, IsolatedFromAbove]> {
  let summary = "query_map operation";
  let description = [{
    This operation queries def_immutable_mapping.
  }];

  let arguments = (ins Index:$map,
                       Index:$key); 
  let results = (outs AnyType:$result);

  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = [{
    void getAllQueriedValues(SmallVector<Value>&);
    unsigned getMapOperandIndex() { return 0; }
  }];

  let hasVerifier = 1;
}

def EqualizePatternOp : Uniform_Op<"equalize_pattern",
      [ImplicitUniformTerminator]> {
  let summary = "equalize_pattern operation";
  let description = [{
    This operation captures the variations in transfer or compute patterns 
    between units and encapsulates them within parallel regions.
    All regions must exhibit an identical pattern of operation sequence.
  }];

  let arguments = (ins Variadic<AnyType>:$units,
                       I32ArrayAttr:$list_sizes);
  let regions = (region VariadicRegion<AnyRegion>:$regions);

  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = [{
    static StringRef getListSizesAttrStrName() {return "list_sizes";}
    Value getRegionArg(int i) {return getRegion(i).getArgument(0);}
    ValueRange getRegionUnitList(int);
    std::optional<ValueRange> getRegionUnitList(Value);
    std::optional<Region *> getSpecificRegion(Value);
    Region *getRegionFromUnit(Value unit);
  }];

  let hasVerifier = 1;
}

#endif // DIALECTS_DIALECT_UNIFORM_OPS_TD
