//===-- DataflowOps.td -- Dataflow Dialect Operation Definitions -*- tablegen ----==//
//
// IBM Confidential
// (C) Copyright IBM Corp. 2021, 2025
//
// =============================================================================
//
// This file defines the Dataflow operations in the MLIR operation set.
//
//===----------------------------------------------------------------------===//

#ifndef DIALECT_DATAFLOW_TD
#define DIALECT_DATAFLOW_TD

#ifndef OP_BASE
include "mlir/IR/OpBase.td"
#endif // OP_BASE

include "mlir/Dialect/LLVMIR/LLVMOpBase.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/LoopLikeInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/BuiltinAttributes.td"
include "src/Dialects/Dataflow/DataflowTypes.td"


// Base class for Dataflow dialect operations. This operation inherits from the
// base `Op` class in OpBase.td, and provides:
//   * The parent dialect of the operation.
//   * The mnemonic for the operation, or the name without the dialect prefix.
//   * A list of traits for the operation.
class Dataflow_Op<string mnemonic, list<Trait> traits = []> :
    Op<Dataflow_Dialect, mnemonic, traits> {
  // For every standard op, there needs to be a:
  //   * void print(OpAsmPrinter &p, ${C++ class of Op} op)
  //   * LogicalResult verify(${C++ class of Op} op)
  //   * ParseResult parse${C++ class of Op}(OpAsmParser &parser,
  //                                         OperationState &result)
  // functions.

}

def Dataflow_GetUnitOp:Dataflow_Op<"get_unit",
  [NoMemoryEffect, IsolatedFromAbove]> {
    let summary = "Getting a handle to a global unit";
    let description = [{
    "This operation gets a handle to a global unit for operations on the unit."
    ```mlir
        %unit = dataflow.get_unit {name="C0-L0LU", type="L0LU"} : Index
    ```
    }];
    let arguments = (ins StrAttr:$name, StrAttr:$type);

    //Variadic has been added to represent multiple return values of GetUnit operation with each return
    //value corresponding to an instance of program time steps.
    let results = (outs Variadic<Index>:$units);
    let assemblyFormat = [{
        attr-dict `:` type(results)
    }];

    let hasVerifier = 1;

    let extraClassDeclaration = [{
      Value getUnit();
    }];
}

def Dataflow_GetLocalUnitOp:Dataflow_Op<"get_local_unit",
  [NoMemoryEffect, IsolatedFromAbove]> {
    let summary = "Getting a handle to a local unit";
    let description = [{
    "This operation gets a handle to a local unit for operations on the unit."
    ```mlir
        %unit = dataflow.get_local_unit {name="L0-LU"} : Index
    ```
    }];
    let arguments = (ins Index:$unitId, StrAttr:$name);
    let results = (outs Index:$unit);
    let assemblyFormat = [{
         $unitId attr-dict `:` type(results)
    }];
}

def Dataflow_ProgramUnitOp:Dataflow_Op<"program_unit",
  [SingleBlockImplicitTerminator<"ReturnOp">]> {
    let summary = "Definition of a unit";
    let description = [{
    "This operation holds the definition of a unit."
    ```mlir
    dataflow.program_unit %unit_id : (Index) { }
    ```
    }];

    let arguments = (ins Variadic<Index>:$units, OptionalAttr<StrAttr>:$precision);
    let regions = (region SizedRegion<1>:$region);
    let skipDefaultBuilders = 1;
    let builders = [
    OpBuilder<(ins "ValueRange":$units,
      CArg<"function_ref<void(OpBuilder &, Location, ValueRange)>",
           "nullptr">:$bodyBuilder)>,
      OpBuilder<(ins "ValueRange":$units, "StringAttr":$precision,
            CArg<"function_ref<void(OpBuilder &, Location, ValueRange)>",
                 "nullptr">:$bodyBuilder)>
    ];

    let extraClassDeclaration = [{
    using BodyBuilderFn =
        function_ref<void(OpBuilder &, Location, ValueRange)>;
        static StringRef getPrecisionAttrStrName() { return "precision"; }
    }];

    let hasCustomAssemblyFormat = 1;
}

def Dataflow_ProgramUnitTraceOp:Dataflow_Op<"program_unit_trace",
  [SingleBlockImplicitTerminator<"ReturnOp">]> {
    let summary = "Static trace of a unit";
    let description = [{
    "This operation holds the static trace of a unit.  It is symmetric to 
     program_unit but used for holding the `unrolled` form of the function
     that it traces.  
     Attribute `tracename` is a human readable name to annotate the trace, 
     typically set to the name of the pass that was applied prior to the trace.
     Set to `standalone-trace` with `--kEmitTraceIR`.
     Set to `initial-pass` for the first trace in a pipeline.
     Attribute `compare_pass` is only set if a trace comparsion was performed.
     The tracer runs a comparison after generating a trace if a 
     corresponding prior trace was available in the IR prior.
     True if the traces compared equivalent; false if they did not.
    ```mlir
    dataflow.program_unit_trace %unit_id : (Index) { }
    ```
    }];

    let arguments = (ins Variadic<Index>:$units, 
                         StrAttr:$tracename, 
                         OptionalAttr<BoolAttr>:$compare_pass);
    let regions = (region SizedRegion<1>:$region);
    let skipDefaultBuilders = 1;
    let builders = [
    OpBuilder<(ins "ValueRange":$units, "StringAttr":$compare_pass, 
                   CArg<"function_ref<void(OpBuilder &, Location, ValueRange)>", 
                        "nullptr">:$bodyBuilder)>,
    ];

    let extraClassDeclaration = [{
    using BodyBuilderFn =
        function_ref<void(OpBuilder &, Location, ValueRange)>;
        static StringRef getTracenameAttrStrName() { return "tracename"; }
        static StringRef getComparePassAttrStrName() { return "compare_pass"; }
    }];

    let hasCustomAssemblyFormat = 1;
}

def Dataflow_ReturnOp : Dataflow_Op<"return", [Terminator]> {
  let summary = [{
    The `return` operation terminates the definition.
  }];
}


def Dataflow_CreateGroupOp:Dataflow_Op<"create_group",
  [NoMemoryEffect, IsolatedFromAbove]> {
    let summary = "Creates a group of unit IDs";
    let description = [{
    "This operation creats a group of unit IDs."
    ```mlir
        %group = dataflow.create_group (%1 : index %2 : index) : index
    ```
    }];

    let arguments = (ins Variadic<Index>:$unit_ids);
    let results = (outs Index:$group_id);
    let assemblyFormat = [{
       `(` ($unit_ids^ `:` type($unit_ids))? `)`  attr-dict `:` type(results)
    }];
}

def Dataflow_CreateMulticastGroupOp:Dataflow_Op<"create_multicast_group",
  [NoMemoryEffect, IsolatedFromAbove]> {
    let summary = "Creates a multicast group.";
    let description = [{
    "This operation creats a multicast group which specifies the unit ID of the
     producer of data, list of unit IDs of the consumers, and additional 
     information such as `group_id`, `count` and the number of sharers. Note that the
     list of consumers could be left empty, however when it is not empty, the 
     size of that list must match the `num_consumers` (number of sharers) field. 
     `count` field specifies the initial number of outstanding requests."
    ```mlir
        %group = dataflow.create_multicast_group (%1 -> (%2, %3, %4)) {count = 0 : i32, group_id = 1 : i32, num_consumers = 2 : i32} : index
    ```
    }];

    let arguments = (ins Index:$producer,
                     Variadic<Index>:$consumers,
                     I32Attr:$num_consumers,
                     I32Attr:$group_id,
                     I32Attr:$count
                     );
    let results = (outs Index:$result);
    let assemblyFormat = [{
       `(` $producer `->` `(` ($consumers^)? `)` `)`  attr-dict `:` type(results)
    }];

    let extraClassDeclaration = [{
      // To avoid inter-unit analysis in DCC, the DCG-FE sends a hint 
      // indicating if multicast groups can be safely promoted to programHeader. 
      // This flag should not be set on a create_multicast_group op that appears
      // in a loop. PCFG translator would have put it outside of such loops if it could.
      // The flag is only an optimization hint not a directive, meaning that DCC
      // can choose whether or not to promote and (if num candidates > 8) which 
      // ones to promote. This choise exists because of the fact that 
      // create_multicast_group ops (with flag on) cannot appear in loops. If they 
      // cannot appear in loops, then it doesn't matter which one we pick for 
      // promotion, because there is no way for the corresponding GTRIMM to get 
      // re-executed in any of the cores.
      // The location of create_multicast_groups are still significant in the 
      // absence of the flag, so dcc should avoid moving them with respect to 
      // data transfer operations.
      // * "loop" means a loop with iteration count greater than one.
      bool isOptimizable() { return (*this)->hasAttr("init_packet_opt_en"); }
    }];
}

def Dataflow_SyncSendOp:Dataflow_Op<"sync_send",
  []> {
    let summary = "Send synchronization signal for explicit synchronization";
    let description = [{
    "This operation is equivalent to sending a synchronization signal for explicit synchronization."
    ```mlir
        dataflow.sync_send %to {wait_immediately_for_async_transfers = true} : (Index)
    ```
    }];
    let arguments = (ins Index:$to_unit, OptionalAttr<StrAttr>:$dbgName,
                         OptionalAttr<BoolAttr>:$wait_immediately_for_async_transfers);
    let assemblyFormat = [{
         $to_unit attr-dict `:` type($to_unit)
    }];
}

def Dataflow_SyncRecvOp:Dataflow_Op<"sync_recv",
  []> {
    let summary = "Receive synchronization signal for explicit synchronization";
    let description = [{
    "This operation is equivalent to receive synchronization signal for explicit synchronization."
    ```mlir
        dataflow.sync_recv %from : (Index)
    ```
    }];
    let arguments = (ins Index:$from_unit,
                         OptionalAttr<StrAttr>:$dbgName);
    let assemblyFormat = [{
         $from_unit  attr-dict `:` type($from_unit)
    }];
}

def Dataflow_ImplicitSyncOnStreamingBufferOp:Dataflow_Op<"implicit_sync_on_streaming_buffer",
                                      []> {
    let summary = "Performs sync send/recv operations at the buffer granularity level";
    let description = [{
      "This operation is meant to enable synchronization implicitly at the boundary of streaming buffer."
    ```mlir
          %handler = dataflow.implicit_sync_on_streaming_buffer %l0_view, %l0su, %size : (memref<4xi8>, Index, Index, Index)
    ```
    }];
    let arguments = (ins AnyTypeOf<[AnyMemRef]>:$memory_unit, Index:$dst_unit, Index:$buffer_size,
                         OptionalAttr<StrAttr>:$dbgName);
    let assemblyFormat = [{
      $memory_unit `,` $dst_unit `,` $buffer_size attr-dict `:`
                  type($memory_unit) `,` type($dst_unit)  `,` type($buffer_size)
    }];
}

def Dataflow_GetLogicalMemoryViewOp:Dataflow_Op<"get_logical_memory_view",
  [NoMemoryEffect, IsolatedFromAbove]> {
    let summary = "Operation to get a logical view of the memory";
    let description = [{
    "This operation is useful to get a logical view of the memory."
    ```mlir
    %data = dataflow.get_logical_memory_view %from, %start_address {#layout_map} : memref<128xi8>
    ```
    }];
    let arguments = (ins Index:$from_unit, Index:$start_address, AffineMapAttr:$layout_map);
    let results = (outs AnyTypeOf<[AnyMemRef]>:$data);
    let assemblyFormat = [{
         $from_unit `,` $start_address  attr-dict `:` type($from_unit) `,` type($start_address) `,` type(results)
    }];
}

def Dataflow_GetPagedLogicalMemoryViewOp:Dataflow_Op<"get_paged_logical_memory_view",
  [NoMemoryEffect, IsolatedFromAbove]> {
    let summary = "Operation to get logical views of paged memory";
    let description = [{
    "This operation is useful to get logical views of paged memory."
    ```mlir
    %data = dataflow.get_paged_logical_memory_view %from, %start_address {#layout_map}
                {segments =
                    page0: (idx_set = (d0 >= 0, d0 < 64, d1 >= 0, d1 <= 1, d2 = 0>),
                            start_address = %pg0_start_address),
                    page1: (idx_set = (d0 >= 0, d0 < 64, d1 >= 2, d1 <= 3, d2 = 0>),
                            start_address = %pg1_start_address),
                    page2: (idx_set = (d0 >= 0, d0 < 64, d1 >= 0, d1 <= 1, d2 = 1>),
                            start_address = %pg2_start_address),
                    ...
                    page63: (idx_set = (d0 >= 0, d0 < 64, d1 >= 2, d1 <= 3, d2 = 31>),
                            start_address = %pg63_start_address)}
                : index, index, memref<?x?x?xf16>
    ```
    }];
    let arguments = (ins Index:$unit,
                         Index:$start_addr,
                         Variadic<Index>:$page_start_addrs,
                         AffineMapAttr:$layout_map,
                         ArrayAttr:$idx_sets);
    let results = (outs AnyTypeOf<[AnyMemRef]>:$data);
    let hasCustomAssemblyFormat = 1;
    let hasVerifier = 1;

    let extraClassDeclaration = [{
      StringRef getIdxSetsStrName() { return "idx_sets"; }
    }];
}

def Dataflow_SendOp:Dataflow_Op<"send",
  []> {
    let summary = "Send a vector of data to another unit";
    let description = [{
    "This operation is equivalent to sending a vector of data to another unit."
    ```mlir
    "Dataflow.send"(%to_unit, %send_data) {} : vector<128xi8>
    ```
    }];

    let arguments = (ins Index:$to_unit,
                     AnyTypeOf<[AnyVectorOfAnyRank]>:$send_data,
                     OptionalAttr<DataflowRoutingDirectionAttr>:$dir,
                     OptionalAttr<StrAttr>:$dbgName
                     );
    let assemblyFormat = [{
         $to_unit `,` $send_data attr-dict `:` type($send_data)
    }];
}

def Dataflow_ReceiveOp:Dataflow_Op<"receive",
  []> {
    let summary = "Receive a vector of data from another unit";
    let description = [{
    "This operation is equivalent to receiving a vector of data from another unit."
    ```mlir
    %data = "Dataflow.receive"(%from_unit) {} : vector<128xi8>
    ```
    }];
    let arguments = (ins Index:$from_unit,
                         OptionalAttr<StrAttr>:$dbgName);
    let results = (outs AnyTypeOf<[AnyVectorOfAnyRank]>:$data);

    let assemblyFormat = [{
         $from_unit attr-dict `:` type(results)
    }];
}

def Dataflow_OpaqueOp:Dataflow_Op<"opaque",
  []> {
    let summary = "Opaque operation";
    let description = [{
      "Opaque operation for auxiliary functions."
      ```mlir
      dataflow.opaque {func_name = "reciprocal",
                       read_write_register_dictionary = {p0="R0",p1="R1"},
                       read_only_register_dictionary = {c0="R2"},
                       parameter_dictionary = {unroll_factor="2"}}
      ```
    }];
    let arguments = (ins OptionalAttr<StrAttr>:$dbgName,
                         StrAttr:$func_name,
                         DictionaryAttr:$read_write_register_dictionary,
                         DictionaryAttr:$read_only_register_dictionary,
                         DictionaryAttr:$parameter_dictionary);
    let assemblyFormat = [{
        attr-dict
    }];
}

#endif // DIALECT_DATAFLOW_TD
