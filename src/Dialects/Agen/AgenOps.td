//===-- AgenOps.td -- AgenOps Dialect Operation Definitions -*- tablegen ----==//
//
// IBM Confidential
// (C) Copyright IBM Corp. 2021, 2025
//
// =============================================================================
//
// This file defines the sentient operations in the MLIR operation set.
//
//===----------------------------------------------------------------------===//

#ifndef DIALECT_AGEN_OPS_TD
#define DIALECT_AGEN_OPS_TD

#ifndef OP_BASE
include "mlir/IR/OpBase.td"
#endif // OP_BASE
include "src/Dialects/Agen/AgenTypes.td"

include "mlir/Dialect/LLVMIR/LLVMOpBase.td"
include "mlir/Dialect/Affine/IR/AffineMemoryOpInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/LoopLikeInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/BuiltinAttributes.td"


// Require regions to have sentient.yield.
def ImplicitAgenTerminator
    : SingleBlockImplicitTerminator<"YieldOp">;

def YieldOp : Agen_Op<"yield", [NoMemoryEffect, Terminator, ReturnLike,
    MemRefsNormalizable]> {
  let summary = "Yield values to parent operation";
  let description = [{
    "agen.yield yields zero or more SSA values to parent op."
  }];

  let arguments = (ins Variadic<AnyType>:$operands);
  
  let builders = [
    OpBuilder<(ins), [{ build($_builder, $_state, ValueRange{}); }]>
  ];

  let assemblyFormat = "attr-dict ($operands^ `:` type($operands))?";
}

def Agen_VectorLoadOp:Agen_Op<"vector_load",
[DeclareOpInterfaceMethods<AffineReadOpInterface>,
        DeclareOpInterfaceMethods<AffineMapAccessInterface>,
        MemRefsNormalizable]> {

  let summary = "agen vector load operation";
  let arguments = (ins Arg<AnyMemRef, "the reference to load from",
        [MemRead]>:$memref,
        OptionalAttr<StrAttr>:$dbgName,
        Builtin_IntegerSetAttr:$load_set,
        AffineMapAttr:$load_order,
        I32Attr:$num_memref_indices,
        Variadic<Index>:$operands1);

  let results = (outs AnyVectorOfAnyRank:$result);
  let skipDefaultBuilders = 1;
  let hasCustomAssemblyFormat = 1;

  let builders = [
    /// Builds an agen vector load op with the specified map and operands.
    OpBuilder<(ins "VectorType":$resultType, "Value":$memref, "mlir::StringAttr":$dbgName, "mlir::AffineMap":$map,
      "ValueRange":$operands1, "IntegerSet":$load_set, "mlir::AffineMap":$load_order)>,
    /// Builds an agen vector load op with the specified map and operands (including multicast info operand)
    OpBuilder<(ins "VectorType":$resultType, "Value":$memref, "mlir::StringAttr":$dbgName, "mlir::AffineMap":$map,
      "ValueRange":$operands1, "IntegerSet":$load_set, "mlir::AffineMap":$load_order,
      "uint32_t":$num_memref_indices)>
  ];

  let extraClassDeclaration =  [{
    VectorType getVectorType() {
      return mlir::cast<VectorType>(getResult().getType());
    }
    /// Returns the operand index of the memref.
    unsigned getMemRefOperandIndex() { return 0; }

    /// Returns the operand index of the operands1 operands list.
    unsigned getOperands1OperandIndex() { return 1; }

    void setMemRef(Value value) { setOperand(getMemRefOperandIndex(), value); }

    /// Returns the affine map used to index the memref for this operation.
    AffineMapAttr getAffineMapAttr() {
      return mlir::cast<AffineMapAttr>((*this)->getAttr(getMapAttrStrName()));
    }

    /// Returns the affine map for the agen vector load.
    IntegerSetAttr getAffineLoadSetAttr() {
      return mlir::cast<IntegerSetAttr>((*this)->getAttr(getLoadSetAttrStrName()));
    }

    static StringRef getMapAttrStrName() { return "map"; }
    static StringRef getLoadSetAttrStrName() { return "load_set"; }
    static StringRef getLoadOrderMapAttrStrName() { return "load_order"; }
    static StringRef getNumMemrefIndicesAttrStrName() { return "num_memref_indices"; }

    ::mlir::Operation::operand_range getMapIndices() {
      return getOperands1().slice(0, getNumMemrefIndices());
    }

    ::mlir::Operation::operand_range indices() {
      return getMapIndices();
    }

    std::optional<Value> getMulticastInfo() {
      auto multicast_size = getOperands1().size() - getNumMemrefIndices();
      if (multicast_size == 1)
        return getOperands1().slice(getNumMemrefIndices(), 1)[0];
      else if (multicast_size == 0)
        return std::optional<Value>();
      else
        llvm_unreachable("expect one or no multicast info");
    }

    SmallVector<Operation *> getUseChain();
    void cloneUseChainToNewOp(OpBuilder &builder, Operation *new_op);
    VectorLoadOp cloneWithNewAccessInfo(OpBuilder &builder, const Value mem_view,
        const AffineMap &subscripts_map, const SmallVectorImpl<Value> &indices);
    void eraseOpAndUseChain();
  }];
}

def Agen_VectorStoreOp : Agen_Op<"vector_store",
[DeclareOpInterfaceMethods<AffineWriteOpInterface>,
        DeclareOpInterfaceMethods<AffineMapAccessInterface>,
        MemRefsNormalizable]> {

  let summary = "agen vector store operation";
  let arguments = (ins AnyType:$value,
      Arg<AnyMemRef, "the reference to store to",
      [MemWrite]>:$memref,
      OptionalAttr<StrAttr>:$dbgName,
      Builtin_IntegerSetAttr:$store_set,
      AffineMapAttr:$store_order,
      Variadic<Index>:$indices);

  let skipDefaultBuilders = 1;
  let hasCustomAssemblyFormat = 1;
  let builders = [
    OpBuilder<(ins "Value":$valueToStore, "Value":$memref, "mlir::StringAttr":$dbgName,
      "ValueRange":$indices, "IntegerSet":$store_set, "mlir::AffineMap":$store_order)>,
    OpBuilder<(ins "Value":$valueToStore, "Value":$memref, "mlir::StringAttr":$dbgName, "AffineMap":$map,
      "ValueRange":$mapOperands, "IntegerSet":$store_set, "mlir::AffineMap":$store_order)>
  ];

  let extraClassDeclaration =  [{

    /// Returns the operand index of the value to be stored.
    unsigned getStoredValOperandIndex() { return 0; }

    VectorType getVectorType() {
        return mlir::cast<VectorType>(getValue().getType());
    }

    /// Returns the operand index of the memref.
    unsigned getMemRefOperandIndex() { return 1; }

    // Returns the operand index of the indices operands.
    unsigned getIndicesOperandIndex() { return 2; }

    void setMemRef(Value value) { setOperand(getMemRefOperandIndex(), value); }

    /// Returns the affine map used to index the memref for this operation.
    AffineMapAttr getAffineMapAttr() {
        return mlir::cast<AffineMapAttr>((*this)->getAttr(getMapAttrStrName()));
    }

    /// Returns the affine map for then agen vector store.
    IntegerSetAttr getAffineStoreSetAttr() {
        return mlir::cast<IntegerSetAttr>((*this)->getAttr(getStoreSetAttrStrName()));
    }

    int getNumIndices() {
      return getIndices().size();
    }

    // Match other op functions to allow better generalization.
    ::mlir::Operation::operand_range getMapIndices() {
      return getIndices();
    }

    static StringRef getMapAttrStrName() { return "map"; }
    static StringRef getStoreSetAttrStrName() { return "store_set"; }
    static StringRef getStoreOrderMapAttrStrName() { return "store_order"; }

    SmallVector<Operation *> getUseChain();
    void cloneUseChainToNewOp(OpBuilder &builder, Operation *new_op);
    VectorStoreOp cloneWithNewAccessInfo(OpBuilder &builder, const Value &mem_view,
        const AffineMap &subscripts_map, const SmallVectorImpl<Value> &indices);
    void eraseOpAndUseChain();
  }];
}

def Agen_CompositeLoadOp:Agen_Op<"composite_load",
        [DeclareOpInterfaceMethods<AffineMapAccessInterface>,
        MemRefsNormalizable,
        ImplicitAgenTerminator]> {

  let summary = "agen composite load operation";
  let arguments = (ins Arg<AnyMemRef, "the reference to load from",
        [MemRead]>:$memref,
        OptionalAttr<StrAttr>:$dbgName,
        Builtin_IntegerSetAttr:$load_set,
        AffineMapAttr:$load_order,
        Builtin_IntegerSetAttr:$time_set,
        AffineMapAttr:$time_order,
        AffineMapAttr:$time_addr_map,
        I32Attr:$num_memref_indices,
        Variadic<Index>:$operands1);
  let regions = (region SizedRegion<1>:$region);

  let skipDefaultBuilders = 1;
  let hasCustomAssemblyFormat = 1;

  let builders = [
    /// Builds a composite load op with the specified map and operands.
    OpBuilder<(ins "Value":$memref, "mlir::StringAttr":$dbgName, "mlir::AffineMap":$map, "ValueRange":$operands1,
      "VectorType":$load_type, "IntegerSet":$load_set, "mlir::AffineMap":$load_order,
      "IntegerSet":$time_set, "mlir::AffineMap":$time_order, "mlir::AffineMap":$time_addr_map,
      "uint32_t":$num_memref_indices,
      CArg<"function_ref<void(OpBuilder &, Location, Value, ValueRange)>",
           "nullptr">:$bodyBuilder)>
  ];

  let extraClassDeclaration =  [{
    using BodyBuilderFn =
        function_ref<void(OpBuilder &, Location, Value, ValueRange)>;

    // Returns the operand index of the memref.
    unsigned getMemRefOperandIndex() { return 0; }

    // Returns the memref ssa.
    Value getMemRef() {return getMemref(); }

    // Returns the memref value type.
    MemRefType getMemRefType() {return mlir::cast<MemRefType>(getMemRef().getType()); }

    // Returns the affine map used to index the memref for this operation.
    AffineMapAttr getAffineMapAttr() {
      return mlir::cast<AffineMapAttr>((*this)->getAttr(getMapAttrStrName()));
    }

    static StringRef getMapAttrStrName() { return "map"; }
    static StringRef getLoadSetAttrStrName() { return "load_set"; }
    static StringRef getLoadOrderMapAttrStrName() { return "load_order"; }
    static StringRef getTimeSetAttrStrName() { return "time_set"; }
    static StringRef getTimeOrderMapAttrStrName() { return "time_order"; }
    static StringRef getTimeAddrMapAttrStrName() { return "time_addr_map"; }
    static StringRef getNumMemrefIndicesAttrStrName() { return "num_memref_indices"; }

    Value getLoadInductionVar() { return getRegion().getBlocks().begin()->getArgument(0); }

    Type getLoadInductionVarType() { return getRegion().getBlocks().begin()->getArgumentTypes()[0]; }

    ::mlir::Operation::operand_range getMapIndices() {
      return getOperands1().slice(0, getNumMemrefIndices());
    }

    const ::mlir::Operation::operand_range getTimeSymbols() {
      auto time_symbols_size = getOperands1().size() - getNumMemrefIndices();
      return getOperands1().slice(getNumMemrefIndices(), time_symbols_size);
    }

    CompositeLoadOp cloneWithNewAccessInfo(OpBuilder &builder,
        const Value &mem_view, const AffineMap &subscripts_map,
        const SmallVectorImpl<Value> &indices, const IntegerSet &time_set);
  }];
}

def Agen_CompositeStoreOp:Agen_Op<"composite_store",
        [DeclareOpInterfaceMethods<AffineMapAccessInterface>,
        MemRefsNormalizable,
        ImplicitAgenTerminator]> {

  let summary = "agen composite store operation";
  let arguments = (ins Arg<AnyMemRef, "the reference to store to",
        [MemWrite]>:$memref,
        OptionalAttr<StrAttr>:$dbgName,
        OptionalAttr<Builtin_IntegerSetAttr>:$store_set,
        OptionalAttr<AffineMapAttr>:$store_order,
        Builtin_IntegerSetAttr:$time_set,
        AffineMapAttr:$time_order,
        AffineMapAttr:$time_addr_map,
        I32Attr:$num_memref_indices,
        BoolAttr:$have_input_vector,
        Variadic<AnyType>:$operands1);
  let regions = (region AnyRegion:$region);

  
  let skipDefaultBuilders = 1;
  let hasCustomAssemblyFormat = 1;

  let builders = [
    /// Builds a composite store op with store_set, store_order and region but no input vector.
    OpBuilder<(ins "Value":$memref, "mlir::StringAttr":$dbgName, "mlir::AffineMap":$map, "ValueRange":$operands1, "IntegerSet":$store_set,
      "mlir::AffineMap":$store_order, "IntegerSet":$time_set, "mlir::AffineMap":$time_order,
      "mlir::AffineMap":$time_addr_map, "uint32_t":$num_memref_indices,
      CArg<"function_ref<void(OpBuilder &, Location, Value, ValueRange)>",
           "nullptr">:$bodyBuilder)>,
    /// Builds a composite store op with input vector but no store_set, store_order or region.
    OpBuilder<(ins "Value":$memref, "mlir::StringAttr":$dbgName, "mlir::AffineMap":$map, "ValueRange":$operands1, "IntegerSet":$time_set,
      "mlir::AffineMap":$time_order, "mlir::AffineMap":$time_addr_map, "uint32_t":$num_memref_indices)>
  ];

  let extraClassDeclaration =  [{
    using BodyBuilderFn =
        function_ref<void(OpBuilder &, Location, Value, ValueRange)>;

    // Returns the operand index of the memref.
    unsigned getMemRefOperandIndex() { return 0; }

    // Returns the memref ssa.
    Value getMemRef() {return getMemref(); }

    // Returns the memref value type.
    MemRefType getMemRefType() {return mlir::cast<MemRefType>(getMemRef().getType()); }

    // Returns the affine map used to index the memref for this operation.
    AffineMapAttr getAffineMapAttr() {
      return mlir::cast<AffineMapAttr>((*this)->getAttr(getMapAttrStrName()));
    }

    static StringRef getMapAttrStrName() { return "map"; }
    static StringRef getStoreSetAttrStrName() { return "store_set"; }
    static StringRef getStoreOrderMapAttrStrName() { return "store_order"; }
    static StringRef getTimeSetAttrStrName() { return "time_set"; }
    static StringRef getTimeOrderMapAttrStrName() { return "time_order"; }
    static StringRef getTimeAddrMapAttrStrName() { return "time_addr_map"; }
    static StringRef getNumMemrefIndicesAttrStrName() { return "num_memref_indices"; }
    static StringRef getHaveInputVectorAttrStrName() { return "have_input_vector"; }
    
    ::mlir::Operation::operand_range getMapIndices() {
      return getOperands1().slice(0, getNumMemrefIndices());
    }

    const ::mlir::Operation::operand_range getTimeSymbols() {
      auto time_symbols_size = getOperands1().size() - (getHaveInputVector() ? 1 : 0) - getNumMemrefIndices();
      return getOperands1().slice(getNumMemrefIndices() + (getHaveInputVector() ? 1 : 0), time_symbols_size);
    }

    std::optional<Value> getInputVector() {
      if (getHaveInputVector()) {
        return getOperands1()[getNumMemrefIndices()];
      }
      return std::nullopt;
    }

    std::optional<VectorType> getInputVectorType() {
      if (getHaveInputVector()) {
        return mlir::cast<VectorType>(getInputVector().value().getType());
      }
      return std::nullopt;
    }

    // TODO need to consider Non-constant time symbols...
    unsigned int getTimeIterNum() {
      unsigned num = 1;
      for (auto t : getTimeSymbols()) {
        num *= mlir::dyn_cast<IntegerAttr>(t.getDefiningOp<arith::ConstantOp>().getValue()).getInt();
      }
      return num;
    }

    CompositeStoreOp cloneWithNewAccessInfo(OpBuilder &builder,
        const Value &mem_view, AffineMap &subscripts_map,
        const SmallVectorImpl<Value> &indices, const IntegerSet &time_set);
  }];
}

def Agen_CompositeLoadAndStoreOp:Agen_Op<"composite_load_and_store",
        [MemRefsNormalizable,
        ImplicitAgenTerminator]> {
  // TODO:
  // The `AffineMapAccessInterface` trait (and its uses in MLIR) expect a single
  // dereferenceable memref, whereas this operations dereferences two memrefs.
  // For safety we cannot inherit this trait, which means MLIR may assume that
  // memrefs used in `composite_load_and_store` operations escape. To improve
  // this we'd need to enhance the trait (or add a new one) and update a few
  // places in MLIR where they check for this trait.

  let summary = "agen composite load_and_store operation";
  let description = [{
    composite_load_and_store operations are used for loading from one memory
    address and storing into another as part of each time step. For instance,
    this operation is needed to model core-to-memory or memory-to-core data
    transfers with burst in the L3 unit.
  }];
  // $operands renamed to operands1 to fix
  // this  error: generated accessor for `operands` overlaps with a default one; please rename to avoid overlap
  let arguments = (ins Arg<AnyMemRef, "the reference to load from",
        [MemRead]>:$src_memref,
        Arg<AnyMemRef, "the reference to store into",
        [MemWrite]>:$dst_memref,
        OptionalAttr<StrAttr>:$dbgName,
        Builtin_IntegerSetAttr:$load_set,
        AffineMapAttr:$load_order,
        Builtin_IntegerSetAttr:$store_set,
        AffineMapAttr:$store_order,
        Builtin_IntegerSetAttr:$time_set,
        AffineMapAttr:$time_order,
        AffineMapAttr:$load_time_addr_map,
        AffineMapAttr:$store_time_addr_map,
        I32Attr:$num_src_memref_indices,
        I32Attr:$num_dst_memref_indices,
        I32Attr:$num_time_symbols,
        OptionalAttr<AgenRoutingDirectionAttr>:$dir,
        Variadic<Index>:$operands1);
  let regions = (region SizedRegion<1>:$region);

  let skipDefaultBuilders = 1;
  let hasCustomAssemblyFormat = 1;
  let builders = [
    /// Builds a composite load_and_store op with the specified map and operands.
    OpBuilder<(ins "Value":$src_memref, "Value":$dst_memref, "mlir::StringAttr":$dbgName, "mlir::AffineMap":$src_map, "mlir::AffineMap":$dst_map, "ValueRange":$operands,
      "VectorType":$type, "IntegerSet":$load_set, "mlir::AffineMap":$load_order,  "IntegerSet":$store_set, "mlir::AffineMap":$store_order,
      "IntegerSet":$time_set, "mlir::AffineMap":$time_order, "mlir::AffineMap":$load_time_addr_map, "mlir::AffineMap":$store_time_addr_map,
      "uint32_t":$num_src_memref_indices, "uint32_t":$num_dst_memref_indices, "uint32_t":$num_time_symbols, "AgenRoutingDirectionAttr":$dir,
      CArg<"function_ref<void(OpBuilder &, Location, Value, ValueRange)>",
           "nullptr">:$bodyBuilder)>
  ];

  let extraClassDeclaration =  [{
    using BodyBuilderFn =
        function_ref<void(OpBuilder &, Location, Value, ValueRange)>;

    // Returns the src/dst memref value type.
    MemRefType getSrcMemRefType() {return mlir::cast<MemRefType>(getSrcMemref().getType()); }
    MemRefType getDstMemRefType() {return mlir::cast<MemRefType>(getDstMemref().getType()); }

    // Returns the affine map used to index the src/dst memref for this operation.
    AffineMapAttr getSrcAffineMapAttr() {
      return mlir::cast<AffineMapAttr>((*this)->getAttr(getSrcMapAttrStrName()));
    }
    AffineMapAttr getDstAffineMapAttr() {
      return mlir::cast<AffineMapAttr>((*this)->getAttr(getDstMapAttrStrName()));
    }

    static StringRef getSrcMapAttrStrName() { return "src_map"; }
    static StringRef getDstMapAttrStrName() { return "dst_map"; }
    static StringRef getLoadSetAttrStrName() { return "load_set"; }
    static StringRef getLoadOrderMapAttrStrName() { return "load_order"; }
    static StringRef getStoreSetAttrStrName() { return "store_set"; }
    static StringRef getStoreOrderMapAttrStrName() { return "store_order"; }
    static StringRef getTimeSetAttrStrName() { return "time_set"; }
    static StringRef getTimeOrderMapAttrStrName() { return "time_order"; }
    static StringRef getLoadTimeAddrMapAttrStrName() { return "load_time_addr_map"; }
    static StringRef getStoreTimeAddrMapAttrStrName() { return "store_time_addr_map"; }
    static StringRef getNumSrcMemrefIndicesAttrStrName() { return "num_src_memref_indices"; }
    static StringRef getNumDstMemrefIndicesAttrStrName() { return "num_dst_memref_indices"; }
    static StringRef getNumTimeSymbolsAttrStrName() { return "num_time_symbols"; }
    static StringRef getDirAttrStrName() { return "dir"; }
    
    Value getLoadInductionVar() { return getRegion().getBlocks().begin()->getArgument(0); }
    Type getLoadInductionVarType() { return getRegion().getBlocks().begin()->getArgumentTypes()[0]; }

    ::mlir::Operation::operand_range getSrcMapIndices() {
      return getOperands1().slice(0, getNumSrcMemrefIndices());
    }
    ::mlir::Operation::operand_range getDstMapIndices() {
      return getOperands1().slice(getNumSrcMemrefIndices(), getNumDstMemrefIndices());
    }
    ::mlir::Operation::operand_range src_indices() {
      return getSrcMapIndices();
    }
    ::mlir::Operation::operand_range dst_indices() {
      return getDstMapIndices();
    }

    const ::mlir::Operation::operand_range getTimeSymbols() {
      return getOperands1().slice(getNumSrcMemrefIndices() + getNumDstMemrefIndices(), getNumTimeSymbols());
    }

    std::optional<Value> getMulticastInfo() {
      auto num_operands_before_multicast = getNumSrcMemrefIndices() +
        getNumDstMemrefIndices() + getNumTimeSymbols();
      // multicast is the last operand
      auto multicast_size = getOperands1().size() - num_operands_before_multicast;
      if (multicast_size == 1)
        return getOperands1().slice(num_operands_before_multicast, 1)[0];
      else if (multicast_size == 0)
        return std::optional<Value>();
      else
        llvm_unreachable("expect one or no multicast info");
    }

    CompositeLoadAndStoreOp cloneWithNewAccessInfo(OpBuilder &builder,
        const Value &src_mem_view, const Value &dst_mem_view,
        const AffineMap &src_subscripts_map, const AffineMap &dst_subscripts_map,
        const SmallVectorImpl<Value> &src_indices,
        const SmallVectorImpl<Value> &dst_indices,
        const IntegerSet time_set);
  }];
}

def Agen_CompositeIndirectLoadAndStoreOp:Agen_Op<"composite_indirect_load_and_store",
        [MemRefsNormalizable,
        ImplicitAgenTerminator,
        AttrSizedOperandSegments]> {
  // TODO:
  // The `AffineMapAccessInterface` trait (and its uses in MLIR) expect a single
  // dereferenceable memref, whereas this operations dereferences upto four memrefs.
  // For safety we cannot inherit this trait, which means MLIR may assume that
  // memrefs used in `composite_indirect_load_and_store` operations escape. To improve
  // this we'd need to enhance the trait (or add a new one) and update a few
  // places in MLIR where they check for this trait.

  let summary = "agen composite indirect load_and_store operation";
  let description = [{
    composite_indirect_load_and_store operations are used for [indirectly] loading from one memory
    address and [indirectly] storing into another as part of each time step. For instance,
    this operation is needed to model gather/scatter data transfers in the L3 unit.
    All of the indirect access functions are optional.
  }];
  let arguments = (ins 
        Optional<AnyMemRef>:$indirect_src_memref,
        AnyMemRef:$direct_src_memref,
        Optional<AnyMemRef>:$indirect_dst_memref,
        AnyMemRef:$direct_dst_memref,
        Variadic<Index>:$indirect_src_map_indices,
        Variadic<Index>:$direct_src_map_indices,
        Variadic<Index>:$indirect_dst_map_indices,
        Variadic<Index>:$direct_dst_map_indices,
        Optional<Index>:$multicast_info,
        Variadic<Index>:$time_symbols,
        // attrs:
        OptionalAttr<StrAttr>:$dbgName,
        OptionalAttr<AffineMapAttr>:$indirect_src_map,
        OptionalAttr<AffineMapAttr>:$direct_src_map,
        OptionalAttr<AffineMapAttr>:$indirect_dst_map,
        OptionalAttr<AffineMapAttr>:$direct_dst_map,
        Builtin_IntegerSetAttr:$load_set,
        AffineMapAttr:$load_order,
        Builtin_IntegerSetAttr:$store_set,
        AffineMapAttr:$store_order,
        Builtin_IntegerSetAttr:$time_set,
        AffineMapAttr:$time_order,
        OptionalAttr<AffineMapAttr>:$load_indirect_time_addr_map,
        AffineMapAttr:$load_direct_time_addr_map,
        OptionalAttr<AffineMapAttr>:$store_indirect_time_addr_map,
        AffineMapAttr:$store_direct_time_addr_map
  );
  let regions = (region SizedRegion<1>:$region);

  let skipDefaultBuilders = 1;
  let hasCustomAssemblyFormat = 1;
  let builders = [
    /// Builds a composite indirect load_and_store op with the specified map and operands.
    /// The $operands argument is the list of operands corresponding to $indirect_src_map_indices, $direct_src_map_indices,$indirect_dst_map_indices, $direct_dst_map_indices, $multicast_info, and $time_symbols in that order.
    /// Since some operands are optional and/or variadic, the number of SSA values in each segment of the operand list is passed to the builder as argument (eg. see $num_ind_src_memref_indices, etc). 
    OpBuilder<(ins "Value":$indirect_src_memref, "Value":$direct_src_memref, "Value":$indirect_dst_memref, "Value":$direct_dst_memref, "mlir::StringAttr":$dbgName,
      "mlir::AffineMap":$indirect_src_map, "mlir::AffineMap":$direct_src_map, "mlir::AffineMap":$indirect_dst_map, "mlir::AffineMap":$direct_dst_map, "ValueRange":$operands,
      "VectorType":$type, "IntegerSet":$load_set, "mlir::AffineMap":$load_order,  "IntegerSet":$store_set, "mlir::AffineMap":$store_order,
      "IntegerSet":$time_set, "mlir::AffineMap":$time_order, "mlir::AffineMap":$indirect_load_time_addr_map, "mlir::AffineMap":$direct_load_time_addr_map, 
      "mlir::AffineMap":$indirect_store_time_addr_map, "mlir::AffineMap":$direct_store_time_addr_map,
      "uint32_t":$num_ind_src_memref_indices, "uint32_t":$num_dir_src_memref_indices, "uint32_t":$num_ind_dst_memref_indices, "uint32_t":$num_dir_dst_memref_indices, 
      "uint32_t":$num_multicast_info, "uint32_t":$num_time_symbols,
      CArg<"function_ref<void(OpBuilder &, Location, Value, ValueRange)>",
           "nullptr">:$bodyBuilder)>
  ];

  let extraClassDeclaration =  [{
    using BodyBuilderFn =
        function_ref<void(OpBuilder &, Location, Value, ValueRange)>;

    // Returns the src/dst memref value type.
    MemRefType getIndirectSrcMemRefType() {return mlir::cast<MemRefType>(getIndirectSrcMemref().getType()); }
    MemRefType getDirectSrcMemRefType() {return mlir::cast<MemRefType>(getDirectSrcMemref().getType()); }
    MemRefType getIndirectDstMemRefType() {return mlir::cast<MemRefType>(getIndirectDstMemref().getType()); }
    MemRefType getDirectDstMemRefType() {return mlir::cast<MemRefType>(getDirectDstMemref().getType()); }
    IndexType getMulticastInfoType() { return mlir::cast<IndexType>(getMulticastInfo().getType()); }

    // Returns the affine map used to index the src/dst memref for this operation.
    AffineMapAttr getIndirectSrcAffineMapAttr() {
      return mlir::cast<AffineMapAttr>((*this)->getAttr(getIndirectSrcMapAttrStrName()));
    }
    AffineMapAttr getDirectSrcAffineMapAttr() {
      return mlir::cast<AffineMapAttr>((*this)->getAttr(getDirectSrcMapAttrStrName()));
    }
    AffineMapAttr getIndirectDstAffineMapAttr() {
      return mlir::cast<AffineMapAttr>((*this)->getAttr(getIndirectDstMapAttrStrName()));
    }
    AffineMapAttr getDirectDstAffineMapAttr() {
      return mlir::cast<AffineMapAttr>((*this)->getAttr(getDirectDstMapAttrStrName()));
    }

    static StringRef getIndirectSrcMapAttrStrName() { return "ind_src_map"; }
    static StringRef getDirectSrcMapAttrStrName() { return "dir_src_map"; }
    static StringRef getIndirectDstMapAttrStrName() { return "ind_dst_map"; }
    static StringRef getDirectDstMapAttrStrName() { return "dir_dst_map"; }
    static StringRef getLoadSetAttrStrName() { return "load_set"; }
    static StringRef getLoadOrderMapAttrStrName() { return "load_order"; }
    static StringRef getStoreSetAttrStrName() { return "store_set"; }
    static StringRef getStoreOrderMapAttrStrName() { return "store_order"; }
    static StringRef getTimeSetAttrStrName() { return "time_set"; }
    static StringRef getTimeOrderMapAttrStrName() { return "time_order"; }
    static StringRef getLoadIndirectTimeAddrMapAttrStrName() { return "load_indirect_time_addr_map"; }
    static StringRef getLoadDirectTimeAddrMapAttrStrName() { return "load_direct_time_addr_map"; }
    static StringRef getStoreIndirectTimeAddrMapAttrStrName() { return "store_indirect_time_addr_map"; }
    static StringRef getStoreDirectTimeAddrMapAttrStrName() { return "store_direct_time_addr_map"; }
    
    Value getLoadInductionVar() { return getRegion().getBlocks().begin()->getArgument(0); }
    Type getLoadInductionVarType() { return getRegion().getBlocks().begin()->getArgumentTypes()[0]; }

    bool hasIndirectSrc() {
      return static_cast<bool>(getIndirectSrcMemref());
    }
    bool hasIndirectDst() {
      return static_cast<bool>(getIndirectDstMemref());
    }
    bool hasMulticastInfo() {
      return static_cast<bool>(getMulticastInfo());
    }

    CompositeIndirectLoadAndStoreOp cloneWithNewAccessInfo(
        OpBuilder &builder, const Value &ind_src_mem_view,
        const Value &dir_src_mem_view,
        const Value &ind_dst_mem_view, const Value &dir_dst_mem_view,
        const AffineMap &ind_src_map, const AffineMap &dir_src_map,
        const AffineMap &ind_dst_map, const AffineMap &dir_dst_map,
        const SmallVectorImpl<Value> &dir_src_indices,
        const SmallVectorImpl<Value> &dir_dst_indices, const IntegerSet time_set);
  }];
}

def Agen_CompositeIndirectLoadOp:Agen_Op<"composite_indirect_load",
        [MemRefsNormalizable,
        ImplicitAgenTerminator,
        AttrSizedOperandSegments]> {
  // TODO:
  // The `AffineMapAccessInterface` trait (and its uses in MLIR) expect a single
  // dereferenceable memref, whereas this operations dereferences two memrefs.
  // For safety we cannot inherit this trait, which means MLIR may assume that
  // memrefs used in `composite_indirect_load` operations escape. To improve
  // this we'd need to enhance the trait (or add a new one) and update a few
  // places in MLIR where they check for this trait.

  let summary = "agen composite indirect load operation";
  let description = [{
    composite_indirect_load operations are used for [indirectly] loading from a memory
    address as part of each time step. For instance, this operation is needed to
    dynamically load data in the LX unit.
  }];
  let arguments = (ins 
        AnyMemRef:$indirect_memref,
        AnyMemRef:$direct_memref,
        Variadic<Index>:$indirect_map_indices,
        Variadic<Index>:$direct_map_indices,
        Optional<Index>:$multicast_info,
        Variadic<Index>:$time_symbols,
        // attrs:
        OptionalAttr<StrAttr>:$dbgName,
        AffineMapAttr:$indirect_map,
        AffineMapAttr:$direct_map,
        Builtin_IntegerSetAttr:$load_set,
        AffineMapAttr:$load_order,
        Builtin_IntegerSetAttr:$time_set,
        AffineMapAttr:$time_order,
        AffineMapAttr:$time_addr_map
  );
  let regions = (region SizedRegion<1>:$region);

  let skipDefaultBuilders = 1;
  let hasCustomAssemblyFormat = 1;
  let builders = [
    /// Builds a composite indirect load op with the specified map and operands.
    /// The $operands argument is the list of operands corresponding to $indirect_map_indices, $direct_map_indices, $multicast_info, and $time_symbols in that order.
    /// Since some operands are optional and/or variadic, the number of SSA values in each segment of the operand list is passed to the builder as argument (eg. see $num_ind_memref_indices, etc). 
    OpBuilder<(ins "Value":$indirect_memref, "Value":$direct_memref, "mlir::StringAttr":$dbgName,
      "mlir::AffineMap":$indirect_map, "mlir::AffineMap":$direct_map, "ValueRange":$operands,
      "VectorType":$type, "IntegerSet":$load_set, "mlir::AffineMap":$load_order,
      "IntegerSet":$time_set, "mlir::AffineMap":$time_order, "mlir::AffineMap":$time_addr_map,
      "uint32_t":$num_ind_memref_indices, "uint32_t":$num_dir_memref_indices,
      "uint32_t":$num_multicast_info, "uint32_t":$num_time_symbols,
      CArg<"function_ref<void(OpBuilder &, Location, Value, ValueRange)>",
           "nullptr">:$body_builder)>
  ];

  let extraClassDeclaration =  [{
    using BodyBuilderFn =
        function_ref<void(OpBuilder &, Location, Value, ValueRange)>;

    // Returns the memref value type.
    MemRefType getIndirectMemrefType() {return mlir::cast<MemRefType>(getIndirectMemref().getType()); }
    MemRefType getDirectMemrefType() {return mlir::cast<MemRefType>(getDirectMemref().getType()); }
    IndexType getMulticastInfoType() { return mlir::cast<IndexType>(getMulticastInfo().getType()); }

    // Returns the affine map used to index the src memref for this operation.
    AffineMapAttr getIndirectAffineMapAttr() {
      return mlir::cast<AffineMapAttr>((*this)->getAttr(getIndirectMapAttrStrName()));
    }
    AffineMapAttr getDirectAffineMapAttr() {
      return mlir::cast<AffineMapAttr>((*this)->getAttr(getDirectMapAttrStrName()));
    }

    static StringRef getIndirectMapAttrStrName() { return "indirect_map"; }
    static StringRef getDirectMapAttrStrName() { return "direct_map"; }
    static StringRef getLoadSetAttrStrName() { return "load_set"; }
    static StringRef getLoadOrderMapAttrStrName() { return "load_order"; }
    static StringRef getTimeSetAttrStrName() { return "time_set"; }
    static StringRef getTimeOrderMapAttrStrName() { return "time_order"; }
    static StringRef getTimeAddrMapAttrStrName() { return "time_addr_map"; }
    
    Value getLoadInductionVar() { return getRegion().getBlocks().begin()->getArgument(0); }
    Type getLoadInductionVarType() { return getRegion().getBlocks().begin()->getArgumentTypes()[0]; }

    bool hasMulticastInfo() {
      return static_cast<bool>(getMulticastInfo());
    }
  }];
}

def Agen_CompositeIndirectStoreOp:Agen_Op<"composite_indirect_store",
        [MemRefsNormalizable,
        ImplicitAgenTerminator,
        AttrSizedOperandSegments]> {
  // TODO:
  // The `AffineMapAccessInterface` trait (and its uses in MLIR) expect a single
  // dereferenceable memref, whereas this operations dereferences two memrefs.
  // For safety we cannot inherit this trait, which means MLIR may assume that
  // memrefs used in `composite_indirect_store` operations escape. To improve
  // this we'd need to enhance the trait (or add a new one) and update a few
  // places in MLIR where they check for this trait.

  let summary = "agen composite indirect store operation";
  let description = [{
    composite_indirect_store operations are used for [indirectly] storing into a memory
    address as part of each time step. For instance, this operation is needed to
    dynamically load data in the LX unit.
  }];
  let arguments = (ins
        
        AnyMemRef:$indirect_memref,
        AnyMemRef:$direct_memref,
        Variadic<Index>:$indirect_map_indices,
        Variadic<Index>:$direct_map_indices,
        Optional<Index>:$multicast_info,
        Variadic<Index>:$time_symbols,
        // attrs:
        OptionalAttr<StrAttr>:$dbgName,
        AffineMapAttr:$indirect_map,
        AffineMapAttr:$direct_map,
        Builtin_IntegerSetAttr:$store_set,
        AffineMapAttr:$store_order,
        Builtin_IntegerSetAttr:$time_set,
        AffineMapAttr:$time_order,
        AffineMapAttr:$time_addr_map
  );
  let regions = (region SizedRegion<1>:$region);

  let skipDefaultBuilders = 1;
  let hasCustomAssemblyFormat = 1;
  let builders = [
    /// Builds a composite indirect store op with the specified map and operands.
    /// The $operands argument is the list of operands corresponding to $indirect_map_indices, $direct_map_indices, $multicast_info, and $time_symbols in that order.
    /// Since some operands are optional and/or variadic, the number of SSA values in each segment of the operand list is passed to the builder as argument (eg. see $num_ind_memref_indices, etc). 
    OpBuilder<(ins "Value":$indirect_memref, "Value":$direct_memref, "mlir::StringAttr":$dbgName,
      "mlir::AffineMap":$indirect_map, "mlir::AffineMap":$direct_map, "ValueRange":$operands,
      "VectorType":$type, "IntegerSet":$store_set, "mlir::AffineMap":$store_order,
      "IntegerSet":$time_set, "mlir::AffineMap":$time_order,
      "mlir::AffineMap":$time_addr_map,
      "uint32_t":$num_ind_memref_indices, "uint32_t":$num_dir_memref_indices,
      "uint32_t":$num_multicast_info, "uint32_t":$num_time_symbols,
      CArg<"function_ref<void(OpBuilder &, Location, Value, ValueRange)>",
           "nullptr">:$body_builder)>
  ];

  let extraClassDeclaration =  [{
    using BodyBuilderFn =
        function_ref<void(OpBuilder &, Location, Value, ValueRange)>;

    // Returns the memref value type.
    MemRefType getIndirectMemrefType() {return mlir::cast<MemRefType>(getIndirectMemref().getType()); }
    MemRefType getDirectMemrefType() {return mlir::cast<MemRefType>(getDirectMemref().getType()); }
    IndexType getMulticastInfoType() { return mlir::cast<IndexType>(getMulticastInfo().getType()); }

    // Returns the affine map used to index the memref for this operation.
    AffineMapAttr getIndirectAffineMapAttr() {
      return mlir::cast<AffineMapAttr>((*this)->getAttr(getIndirectMapAttrStrName()));
    }
    AffineMapAttr getDirectAffineMapAttr() {
      return mlir::cast<AffineMapAttr>((*this)->getAttr(getDirectMapAttrStrName()));
    }

    static StringRef getIndirectMapAttrStrName() { return "indirect_map"; }
    static StringRef getDirectMapAttrStrName() { return "direct_map"; }
    static StringRef getStoreSetAttrStrName() { return "store_set"; }
    static StringRef getStoreOrderMapAttrStrName() { return "store_order"; }
    static StringRef getTimeSetAttrStrName() { return "time_set"; }
    static StringRef getTimeOrderMapAttrStrName() { return "time_order"; }
    static StringRef getTimeAddrMapAttrStrName() { return "time_addr_map"; }

    bool hasMulticastInfo() {
      return static_cast<bool>(getMulticastInfo());
    }
  }];
}

def Agen_IndirectVectorLoadOp:Agen_Op<"indirect_vector_load",
        [MemRefsNormalizable,
        ImplicitAgenTerminator,
        AttrSizedOperandSegments]> {
  // TODO:
  // The `AffineMapAccessInterface` trait (and its uses in MLIR) expect a single
  // dereferenceable memref, whereas this operations dereferences two memrefs.
  // For safety we cannot inherit this trait, which means MLIR may assume that
  // memrefs used in `indirect_vector_load` operations escape. To improve
  // this we'd need to enhance the trait (or add a new one) and update a few
  // places in MLIR where they check for this trait.

  let summary = "agen indirect vector_load operation";
  let description = [{
    indirect_vector_load operations are used for [indirectly] loading from a memory
    address. For instance, this operation is needed to dynamically load data in the LX unit.
  }];
  let arguments = (ins
        AnyMemRef:$indirect_memref,
        AnyMemRef:$direct_memref,
        Variadic<Index>:$indirect_map_indices,
        Variadic<Index>:$direct_map_indices,
        Optional<Index>:$multicast_info,
        // attrs:
        OptionalAttr<StrAttr>:$dbgName,
        AffineMapAttr:$indirect_map,
        AffineMapAttr:$direct_map,
        Builtin_IntegerSetAttr:$load_set,
        AffineMapAttr:$load_order
  );

  let results = (outs AnyVectorOfAnyRank:$result);
  let skipDefaultBuilders = 1;
  let hasCustomAssemblyFormat = 1;
  let builders = [
    /// Builds an indirect vector_load op with the specified map and operands.
    /// The $operands argument is the list of operands corresponding to $indirect_map_indices, $direct_map_indices, $multicast_info
    /// in that order.
    /// Since some operands are optional and/or variadic, the number of SSA values in each segment of the operand list is passed to
    /// the builder as argument (eg. see $num_ind_memref_indices, etc). 
    OpBuilder<(ins "VectorType":$resultType, "Value":$indirect_memref, "Value":$direct_memref, "mlir::StringAttr":$dbgName,
      "mlir::AffineMap":$indirect_map, "mlir::AffineMap":$direct_map,
      "ValueRange":$operands, "VectorType":$type, "IntegerSet":$load_set, "mlir::AffineMap":$load_order,
      "uint32_t":$num_ind_memref_indices, "uint32_t":$num_dir_memref_indices, "uint32_t":$num_multicast_info)>
  ];

  let extraClassDeclaration =  [{
    VectorType getVectorType() {
      return mlir::cast<VectorType>(getResult().getType());
    }

    // Returns the memref value type.
    MemRefType getIndirectMemrefType() {return mlir::cast<MemRefType>(getIndirectMemref().getType()); }
    MemRefType getDirectMemrefType() {return mlir::cast<MemRefType>(getDirectMemref().getType()); }
    IndexType getMulticastInfoType() { return mlir::cast<IndexType>(getMulticastInfo().getType()); }

    // Returns the affine map used to index the memref for this operation.
    AffineMapAttr getIndirectAffineMapAttr() {
      return mlir::cast<AffineMapAttr>((*this)->getAttr(getIndirectMapAttrStrName()));
    }
    AffineMapAttr getDirectAffineMapAttr() {
      return mlir::cast<AffineMapAttr>((*this)->getAttr(getDirectMapAttrStrName()));
    }

    static StringRef getIndirectMapAttrStrName() { return "indirect_map"; }
    static StringRef getDirectMapAttrStrName() { return "direct_map"; }
    static StringRef getLoadSetAttrStrName() { return "load_set"; }
    static StringRef getLoadOrderMapAttrStrName() { return "load_order"; }

    bool hasMulticastInfo() {
      return static_cast<bool>(getMulticastInfo());
    }
  }];
}

def Agen_IndirectVectorStoreOp:Agen_Op<"indirect_vector_store",
        [MemRefsNormalizable,
        ImplicitAgenTerminator,
        AttrSizedOperandSegments]> {
  // TODO:
  // The `AffineMapAccessInterface` trait (and its uses in MLIR) expect a single
  // dereferenceable memref, whereas this operations dereferences two memrefs.
  // For safety we cannot inherit this trait, which means MLIR may assume that
  // memrefs used in `indirect_vector_store` operations escape. To improve
  // this we'd need to enhance the trait (or add a new one) and update a few
  // places in MLIR where they check for this trait.

  let summary = "agen indirect vector_store operation";
  let description = [{
    indirect_vector_store operations are used for [indirectly] storing into a memory
    address. For instance, this operation is needed to dynamically store data in the LX unit.
  }];
  let arguments = (ins
        AnyType:$value,
        AnyMemRef:$indirect_memref,
        AnyMemRef:$direct_memref,
        Variadic<Index>:$indirect_map_indices,
        Variadic<Index>:$direct_map_indices,
        Optional<Index>:$multicast_info,
        // attrs:
        OptionalAttr<StrAttr>:$dbgName,
        AffineMapAttr:$indirect_map,
        AffineMapAttr:$direct_map,
        Builtin_IntegerSetAttr:$store_set,
        AffineMapAttr:$store_order
  );

  let skipDefaultBuilders = 1;
  let hasCustomAssemblyFormat = 1;
  let builders = [
    /// Builds an indirect vector_store op with the specified map and operands.
    /// The $operands argument is the list of operands corresponding to $indirect_map_indices, $direct_map_indices, $multicast_info
    /// in that order.
    /// Since some operands are optional and/or variadic, the number of SSA values in each segment of the operand list is passed to
    /// the builder as argument (eg. see $num_ind_src_memref_indices, etc). 
    OpBuilder<(ins "Value":$value, "Value":$indirect_memref, "Value":$direct_memref, "mlir::StringAttr":$dbgName, "mlir::AffineMap":$indirect_map,
       "mlir::AffineMap":$direct_map, "ValueRange":$operands, "VectorType":$type, "IntegerSet":$store_set, "mlir::AffineMap":$store_order,
      "uint32_t":$num_ind_memref_indices, "uint32_t":$num_dir_memref_indices, "uint32_t":$num_multicast_info)>
  ];

  let extraClassDeclaration =  [{
    VectorType getVectorType() { return mlir::cast<VectorType>(getValue().getType()); }
  
    // Returns the memref value type.
    MemRefType getIndirectMemrefType() {return mlir::cast<MemRefType>(getIndirectMemref().getType()); }
    MemRefType getDirectMemrefType() {return mlir::cast<MemRefType>(getDirectMemref().getType()); }
    IndexType getMulticastInfoType() { return mlir::cast<IndexType>(getMulticastInfo().getType()); }

    // Returns the affine map used to index the memref for this operation.
    AffineMapAttr getIndirectAffineMapAttr() {
      return mlir::cast<AffineMapAttr>((*this)->getAttr(getIndirectMapAttrStrName()));
    }
    AffineMapAttr getDirectAffineMapAttr() {
      return mlir::cast<AffineMapAttr>((*this)->getAttr(getDirectMapAttrStrName()));
    }

    static StringRef getIndirectMapAttrStrName() { return "indirect_map"; }
    static StringRef getDirectMapAttrStrName() { return "direct_map"; }
    static StringRef getStoreSetAttrStrName() { return "store_set"; }
    static StringRef getStoreOrderMapAttrStrName() { return "store_order"; }

    bool hasMulticastInfo() {
      return static_cast<bool>(getMulticastInfo());
    }
  }];
}

def Agen_CompositeMemoryInterleaveOp:Agen_Op<"composite_memory_interleave",
    [ImplicitAgenTerminator]> {
  let summary = "agen composite memory interleaving operation";
  let description = [{
    The composite_memory_interleave operation interleaves the composite memory
    operations contained in its region in bursts specified by the granularity
    attribute. If no granularity is specified, granularity is assumed to be the
    maximum burst amount supported by the hardware.
    
    For example:
    agen.composite_memory_interleave [granularity = 64] {
      composite memory operation A {burst = 80}
      composite memory operation B {burst = 80}
    }
    -->
    composite memory operation A {burst = 64}
    composite memory operation B {burst = 64}
    composite memory operation A {burst = 16}
    composite memory operation B {burst = 16}

    The granularity cannot exceed the burst of the contained operations. If the
    granularity is set to a value greater than the burst of the contained
    operations, it will be interpreted as being equal to the burst. This
    behaviour is required because some of the upstream components may insert
    CompositeMemoryInterleaveOps without knowing the burst of the contained
    operations first.

    For example:
    agen.composite_memory_interleave [granularity = 32] {
      composite memory operation A {burst = 16}
      composite memory operation B {burst = 16}
    }
    -->
    composite memory operation A {burst = 16}
    composite memory operation B {burst = 16}

    Restrictions:
    - only composite memory operations are allowed in containing region
    - the granularity cannot exceed maximum burst amount supported by hardware
    - all composite memory operations inside the containing region need to
      utilize the same burst
    - all composite memory operations inside the containing region need to be
      the same operation type
  }];

  let arguments = (ins OptionalAttr<StrAttr>:$dbgName, OptionalAttr<I32Attr>:$granularity);
  let regions = (region SizedRegion<1>:$region);

  let hasCustomAssemblyFormat = 1;
  let skipDefaultBuilders = 1;

  let builders = [
    // Builds the CompositeMemoryInterleaveOp with a specific granularity.
    OpBuilder<(ins "mlir::StringAttr":$dbgName, "uint32_t":$granularity)>,
    // Builds the CompositeMemoryInterleaveOp with no granularity attribute.
    OpBuilder<(ins "mlir::StringAttr":$dbgName)>
  ];

  let extraClassDeclaration =  [{
    static StringRef getGranularityAttrStrName() { return "granularity"; }
  }];
}

def Agen_SetTransferMaskStateOp:Agen_Op<"set_transfer_mask_state", [MemoryEffects<[MemWrite]>]> {

  let summary = "agen set transfer mask state operation";
  let description = [{
    The set_transfer_mask_state operation collects information from various
    mask patterns and forms one final mask that will be applied to transfers
    for the given unit.

    mask_value: The value that is applied over the elements that are masked.
    num_slices: The mask can be separated into a number of equal sections (slices).
                Each slice may have a different mask.
    slice_mask_map: A description of the mask that will be applied to each
                    slice. Each slice may have a different masking pattern.
                    Some slices may consist of combinations of multiple masks.

                    Each slice is delimited by parentheses. A slice containing
                    a 0 indicates no masking. A slice containing a letter
                    corresponds to the indicated mask. Lettering starts with A
                    and continues sequentially for each mask thereafter.

                    Current support for combinations of masks include:
                      OR operation: (A|B)
                    
                    Example: (0)(0)(A|B)(A)(B)(1) describes a mask containing
                             6 slices. The first two slices will have no masking.
                             The third slice will be masked by a mask pattern
                             determined by OR'ing masks A and B. The fourth slice
                             will be masked by mask A. The fifth slice will be
                             masked by a mask B. The sixth slice will be fully
                             masked.
    
    *num_unmasked_elements: Number of consecutive unmasked elements in the mask
                            pattern. If multiple masks are involved, the number of
                            consecutive unmasked elements for each mask is provided
                            in an ordered list.
    
    *num_masked_elements*: Number of consecutive masked elements in the mask pattern.
                           If multiple masks are involved, the number of consecutive
                           masked elements for each mask is provided in an ordered
                           list.
    
    * If a mask is shorter than the number of elements in a slice, it is repeated to
      fill the slice. The sum of num_unmasked_elements and num_masked elements at a
      given index must evenly divide into the number of elements in a slice.

    Example IR:
      %mask = agen.set_transfer_mask_state mask_unit(%lxlu_c0_cl0) mask_value(%c0) {
                  num_slices = 8 : i32, slice_mask_map = "(A)(A)(A)(A)(A)(A|B)(1)(1)",
                  maskA = "(unmasked = 6 : i32, masked = 2 : i32)",
                  maskB = "(unmasked = 1 : i32, masked = 1 : i32)" } :
                  index, index : vector<64xf16>

    Current uses of this operation are SAMV.
    Partial masking slice_mask_map for SAMV:
      (A)(A)(A)(A)(A)(A|B)(1)(1)
    Where 1 indicates full masking, MaskA represents MaskWsl, and MaskB represents
    MaskXsl.
    For SAMV, MaskA is always interpreted as MaskWsl and MaskB is always interpreted
    as MaskXsl.

    Full masking slice_mask_map for SAMV:
      (1)(1)(1)(1)(1)(1)(1)(1)
    Example IR:
    %mask = agen.set_transfer_mask_state mask_unit(%lxlu_c0_cl0) mask_value(%c0) {
                  num_slices = 8 : i32, slice_mask_map = "(1)(1)(1)(1)(1)(1)(1)(1)" } :
                  index, index : vector<64xf16>
    Note: Full masking does not specify masks as this form will be translated into one
          specific SAMV operation representing full masking.
    Note: To get full masking, this pattern MUST be used. It's the only currently
          supported way to set maskall=yes in the final instruction. Support for
          full masking without this pattern can be added in the future.

    No masking slice_mask_map for SAMV (reset):
      (0)(0)(0)(0)(0)(0)(0)(0)
    Example IR:
    %mask = agen.set_transfer_mask_state mask_unit(%lxlu_c0_cl0) mask_value(%c0) {
                  num_slices = 8 : i32, slice_mask_map = "(0)(0)(0)(0)(0)(0)(0)(0)" } :
                  index, index : vector<64xf16>
    Note: Full unmasking (SAMV reset) does not specify masks as this form will be
          translated into one specific SAMV operation representing masking reset.
  }];
  let arguments = (ins
        Index:$mask_value,
        OptionalAttr<StrAttr>:$dbgName,
        I32Attr:$num_slices,
        StrAttr:$slice_mask_map,
        OptionalAttr<I32ArrayAttr>:$num_unmasked_elements,
        OptionalAttr<I32ArrayAttr>:$num_masked_elements);

  let results = (outs AnyVectorOfAnyRank:$result);

  let hasVerifier = 1;
  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration =  [{
    IndexType getMaskValueType() { return mlir::cast<IndexType>(getMaskValue().getType()); }

    static StringRef getNumSlicesAttrStrName() { return "num_slices"; }
    static StringRef getSliceMaskMapAttrStrName() { return "slice_mask_map"; }
    static StringRef getMaskPrefixAttrStrName() { return "mask"; }

    bool hasValidSAMVPattern();
    bool isGenericSAMV();
    bool isUnmaskSAMV();
    bool isFullMaskSAMV();
    int getSliceIDXsl();
  }];
}

// TODO: Check if any additional traits are required.
def Agen_SymbolicVectorLoadOp:Agen_Op<"symbolic_vector_load",
[MemRefsNormalizable]> {

  let summary = "agen symbolic vector load operation";
  let arguments = (ins Arg<AnyMemRef, "the reference to load from",
        [MemRead]>:$memref,
        OptionalAttr<StrAttr>:$dbgName,
        Builtin_IntegerSetAttr:$load_set,
        AffineMapAttr:$load_order,
        I32Attr:$num_indices,
        I32Attr:$num_strides,
        Variadic<Index>:$operands1);

  let results = (outs AnyVectorOfAnyRank:$result);

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;

  let extraClassDeclaration =  [{
    VectorType getVectorType() {
      return mlir::cast<VectorType>(getResult().getType());
    }
    /// Returns the operand index of the memref.
    unsigned getMemrefOperandIndex() { return 0; }

    /// Returns the operand index of the operands1 operands list.
    unsigned getOperands1OperandIndex() { return 1; }

    void setMemref(Value value) { setOperand(getMemrefOperandIndex(), value); }

    /// Returns the affine map for the agen vector load.
    IntegerSetAttr getAffineLoadSetAttr() {
      return mlir::cast<IntegerSetAttr>((*this)->getAttr(getLoadSetAttrStrName()));
    }

    static StringRef getLoadSetAttrStrName() { return "load_set"; }
    static StringRef getLoadOrderMapAttrStrName() { return "load_order"; }
    static StringRef getNumIndicesAttrStrName() { return "num_indices"; }
    static StringRef getNumStridesAttrStrName() { return "num_strides"; }

    ::mlir::Operation::operand_range getIndices() {
      return getOperands1().slice(0, getNumIndices());
    }

    ::mlir::Operation::operand_range getStrides() {
      return getOperands1().slice(getNumIndices(), getNumStrides());
    }

    std::optional<Value> getMulticastInfo() {
      auto num_other_operands = getNumIndices() + getNumStrides();
      auto multicast_size = getOperands1().size() - num_other_operands;
      if (multicast_size == 1)
        return getOperands1().slice(num_other_operands, 1)[0];
      else if (multicast_size == 0)
        return std::optional<Value>();
      else
        llvm_unreachable("expect one or no multicast info");
    }

  }];
}

// TODO: Check if any additional traits are required.
def Agen_SymbolicVectorStoreOp : Agen_Op<"symbolic_vector_store",
[MemRefsNormalizable]> {

  let summary = "agen symbolic vector store operation";
  let arguments = (ins AnyType:$value_to_store,
      Arg<AnyMemRef, "the reference to store to",
      [MemWrite]>:$memref,
      OptionalAttr<StrAttr>:$dbgName,
      Builtin_IntegerSetAttr:$store_set,
      AffineMapAttr:$store_order,
      I32Attr:$num_indices,
      I32Attr:$num_strides,
      Variadic<Index>:$operands1);

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;

  let extraClassDeclaration =  [{
    /// Returns the operand index of the value to be stored.
    unsigned getStoredValOperandIndex() { return 0; }

    VectorType getVectorType() {
        return mlir::cast<VectorType>(getValueToStore().getType());
    }

    /// Returns the operand index of the memref.
    unsigned getMemrefOperandIndex() { return 1; }

    // Returns the operand index of the indices operands.
    unsigned getIndicesOperandIndex() { return 2; }

    // Returns the operand index of the strides operands.
    unsigned getStridesOperandIndex() { return 2 + getNumIndices(); }

    void setMemref(Value value) { setOperand(getMemrefOperandIndex(), value); }

    /// Returns the affine map for then agen vector store.
    IntegerSetAttr getAffineStoreSetAttr() {
        return mlir::cast<IntegerSetAttr>((*this)->getAttr(getStoreSetAttrStrName()));
    }

    static StringRef getStoreSetAttrStrName() { return "store_set"; }
    static StringRef getStoreOrderMapAttrStrName() { return "store_order"; }
    static StringRef getNumIndicesAttrStrName() { return "num_indices"; }
    static StringRef getNumStridesAttrStrName() { return "num_strides"; }

    ::mlir::Operation::operand_range getIndices() {
      return getOperands1().slice(0, getNumIndices());
    }

    ::mlir::Operation::operand_range getStrides() {
      return getOperands1().slice(getNumIndices(), getNumStrides());
    }
  }];
}

#endif // DIALECT_AGEN_OPS_TD
