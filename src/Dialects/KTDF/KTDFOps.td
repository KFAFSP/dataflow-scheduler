//===----------------------------------------------------------------------===//
//
// Copyright 2026 The KTIR Scheduler Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//
//
// This file defines KTDF operations in the MLIR operation set.
//
//===----------------------------------------------------------------------===//

#ifndef KTDF_SRC_DIALECTS_KTDF_OPS_TD
#define KTDF_SRC_DIALECTS_KTDF_OPS_TD

#ifndef OP_BASE
include "mlir/IR/OpBase.td"
#endif // OP_BASE
include "mlir/IR/EnumAttr.td"

include "mlir/Dialect/Affine/IR/AffineMemoryOpInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/LoopLikeInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ViewLikeInterface.td"
include "mlir/IR/BuiltinAttributes.td"

// Import the KTDF dialect definition
include "KTDFDialect.td"

// Import the KTDF enums (shared between types and ops)
include "KTDFEnums.td"

// Import the KTDF type definitions
include "KTDFTypes.td"

// Import the KTDF attribute definitions
include "KTDFAttributes.td"

// Base class for KTDF dialect operations
class KTDF_Op<string mnemonic, list<Trait> traits = []> :
    Op<KTDF_Dialect, mnemonic, traits> {
}

//===----------------------------------------------------------------------===//
// FIFO Operations
//===----------------------------------------------------------------------===//

def KTDF_FifoAllocateOp : KTDF_Op<"fifo.allocate", [Pure]> {
  let summary = "Allocate FIFO slots for dataflow communication";
  let description = [{
    The `ktdf.fifo.allocate` operation allocates one or more FIFO slots for
    dataflow communication between stages. Each slot has a specific FIFO type
    (e.g., "lxlu-to-sfp") and size (capacity).

    ## FIFO Slot Semantics

    From the IR's perspective, FIFOs are modeled as having infinite depth (i.e.,
    an unbounded number of slots) from which slots are allocated and accessed in
    a rolling fashion. This means:

    1. **Ordered Access**: The order in which slots are allocated must match the
       order in which they are used in data transfers.

    2. **Capacity & Frequency Enforcement**: The total amount of data transferred through a
       given slot must exactly match the slot's size before the next slot can be
       used in a subsequent transfer.

    3. **Rolling Behavior**: After all allocated slots have been used, the FIFO
       rolls back to the first slot, allowing it to be reused for the next round
       of transfers.

    For example:
    ```mlir
    %slot1, %slot2 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"lxlu-to-sfp", 2>,
                                              !ktdf.fifo.slot<"lxlu-to-sfp", 4>
    ```

    This allocation means:
    - Exactly 2 elements of data from lxlu must be transferred to sfp via %slot1
      before any data can be transferred via %slot2
    - Exactly 4 elements of data from lxlu must be transferred via %slot2 before
      the FIFO rolls back, allowing data to be transferred via %slot1 again

    **Loop Semantics**: Allocation of FIFO slots within a loop does not affect
    the lifetime of slots allocated in previous iterations. Each iteration's
    allocations are independent, enabling pipelined execution across loop iterations.

    ## Size Specification

    Sizes can be specified either as:
    1. Static values in the result type (e.g., `!ktdf.fifo.slot<"lxlu-to-sfp", 4>`)
    2. Dynamic values using SSA operands, with `?` in the result type

    When using dynamic sizes, provide one index operand per dynamic slot.
    The sizes are matched to the result types in order.

    ## Examples

    ```mlir
    // Allocate two FIFO slots with static sizes
    %slot0, %slot1 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"lxlu-to-sfp", 1>,
                                              !ktdf.fifo.slot<"lxlu-to-sfp", 4>
    
    // Allocate a single FIFO slot with dynamic size
    %size = arith.constant 8 : index
    %slot = ktdf.fifo.allocate(%size) -> !ktdf.fifo.slot<"sfp-to-lxsu", ?>
    
    // Mix static and dynamic sizes
    %dyn_size = arith.constant 16 : index
    %s0, %s1 = ktdf.fifo.allocate(%dyn_size) -> !ktdf.fifo.slot<"lxlu-to-sfp", ?>,
                                                 !ktdf.fifo.slot<"lxlu-to-sfp", 4>
    ```
  }];

  let arguments = (ins
    Variadic<Index>:$dynamic_sizes
  );

  let results = (outs
    Variadic<KTDF_FifoSlotType>:$slots
  );

  let assemblyFormat = [{
    `(` $dynamic_sizes `)` attr-dict `->` type($slots)
  }];

  let hasVerifier = 1;
  
  let extraClassDeclaration = [{
    /// Returns the number of FIFO slots being allocated.
    unsigned getNumSlots() { return getSlots().size(); }
  }];
}

def KTDF_ReadFromFifoOp : KTDF_Op<"read_from_fifo", [Pure]> {
  let summary = "Read data from a FIFO slot into a tensor";
  let description = [{
    The `ktdf.read_from_fifo` operation reads data from a FIFO slot and returns
    it as a tensor. This operation is used to consume data that has been written
    to a FIFO slot by a previous stage in the dataflow pipeline.

    The operation takes a FIFO slot as input and produces a tensor with the same
    element type as the FIFO slot. The shape of the resulting tensor is determined
    by the number of elements in the FIFO slot.

    Example:
    ```mlir
    // Read from a FIFO slot with 64 f16 elements
    %slot = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"lxlu-to-sfp", 64xf16>
    %data = ktdf.read_from_fifo %slot : !ktdf.fifo.slot<"lxlu-to-sfp", 64xf16> -> tensor<64xf16>
    
    // Read from a FIFO slot with dynamic size
    %size = arith.constant 8 : index
    %dyn_slot = ktdf.fifo.allocate(%size) -> !ktdf.fifo.slot<"sfp-to-lxsu", ?xf16>
    %dyn_data = ktdf.read_from_fifo %dyn_slot : !ktdf.fifo.slot<"sfp-to-lxsu", ?xf16> -> tensor<?xf16>
    ```
  }];

  let arguments = (ins
    KTDF_FifoSlotType:$fifo_slot
  );

  let results = (outs
    AnyTensor:$result
  );

  let assemblyFormat = [{
    $fifo_slot attr-dict `:` type($fifo_slot) `->` type($result)
  }];

  let hasVerifier = 1;

  let extraClassDeclaration = [{
    /// Returns the element type of the FIFO slot
    Type getFifoElementType() {
      return llvm::cast<FifoSlotType>(getFifoSlot().getType()).getElementType();
    }
    
    /// Returns the number of elements in the FIFO slot
    int64_t getFifoNumElements() {
      return llvm::cast<FifoSlotType>(getFifoSlot().getType()).getNumElements();
    }
  }];
}

def KTDF_WriteToFifoOp : KTDF_Op<"write_to_fifo"> {
  let summary = "Write data from a tensor to a FIFO slot";
  let description = [{
    The `ktdf.write_to_fifo` operation writes data from a tensor to a FIFO slot.
    This operation is used to produce data that will be consumed by a subsequent
    stage in the dataflow pipeline.

    The operation takes a tensor and a FIFO slot as inputs. The tensor's element
    type and shape must match the FIFO slot's element type and capacity.

    Example:
    ```mlir
    // Write to a FIFO slot with 64 f16 elements
    %data = arith.constant dense<1.0> : tensor<64xf16>
    %slot = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"lxlu-to-sfp", 64xf16>
    ktdf.write_to_fifo %data, %slot : tensor<64xf16>, !ktdf.fifo.slot<"lxlu-to-sfp", 64xf16>
    
    // Write to a FIFO slot with dynamic size
    %size = arith.constant 8 : index
    %dyn_data = tensor.empty(%size) : tensor<?xf16>
    %dyn_slot = ktdf.fifo.allocate(%size) -> !ktdf.fifo.slot<"sfp-to-lxsu", ?xf16>
    ktdf.write_to_fifo %dyn_data, %dyn_slot : tensor<?xf16>, !ktdf.fifo.slot<"sfp-to-lxsu", ?xf16>
    ```
  }];

  let arguments = (ins
    AnyTensor:$data,
    KTDF_FifoSlotType:$fifo_slot
  );

  let assemblyFormat = [{
    $data `,` $fifo_slot attr-dict `:` type($data) `,` type($fifo_slot)
  }];

  let hasVerifier = 1;

  let extraClassDeclaration = [{
    /// Returns the element type of the FIFO slot
    Type getFifoElementType() {
      return llvm::cast<FifoSlotType>(getFifoSlot().getType()).getElementType();
    }
    
    /// Returns the number of elements in the FIFO slot
    int64_t getFifoNumElements() {
      return llvm::cast<FifoSlotType>(getFifoSlot().getType()).getNumElements();
    }
  }];
}

//===----------------------------------------------------------------------===//
// Token Operations
//===----------------------------------------------------------------------===//

def KTDF_CreateTokenOp : KTDF_Op<"create_token", [Pure]> {
  let summary = "Create a new token for dataflow synchronization";
  let description = [{
    The `ktdf.create_token` operation creates a new token value that can be used
    for dataflow synchronization and dependency tracking. Tokens represent
    abstract synchronization points in the dataflow graph and do not carry data.

    Example:
    ```mlir
    %token = ktdf.create_token : !ktdf.token
    ```
  }];

  let results = (outs
    KTDF_TokenType:$result
  );

  let assemblyFormat = "attr-dict `:` type($result)";

  let builders = [
    OpBuilder<(ins), [{
      build($_builder, $_state, TokenType::get($_builder.getContext()));
    }]>
  ];
}

//===----------------------------------------------------------------------===//
// Tile Size Operations
//===----------------------------------------------------------------------===//

def KTDF_CreateTileSizeOp : KTDF_Op<"create_tile_size", [Pure]> {
  let summary = "Create a placeholder for a tile size to be determined by the scheduler";
  let description = [{
    The `ktdf.create_tile_size` operation creates a placeholder value of index type
    that represents a tile size whose concrete value will be determined by the
    scheduler at a later stage. This operation enables loop tiling transformations
    while deferring the actual tile size selection until scratchpad usage can be
    computed more accurately.

    The operation takes two named attributes:
    - `min_value`: Specifies the minimum acceptable value for the tile size
    - `divisibility`: Specifies a value that the selected tile size must evenly divide

    The scheduler will later replace this placeholder with a concrete tile size value
    that satisfies both constraints and optimizes for scratchpad usage.

    Example:
    ```mlir
    // Create a tile size with minimum value 8 that must divide 4
    %tile_size = ktdf.create_tile_size {min_value = 8 : index, divisibility = 4 : index} : index
    
    // Use in loop tiling
    scf.for %i = %c0 to %N step %tile_size {
      // Loop body
    }
    ```
  }];

  let arguments = (ins
    IndexAttr:$min_value,
    IndexAttr:$divisibility
  );

  let results = (outs
    Index:$result
  );

  let assemblyFormat = "attr-dict `:` type($result)";

  let builders = [
    OpBuilder<(ins "int64_t":$minValue, "int64_t":$divisibility), [{
      auto indexType = $_builder.getIndexType();
      $_state.addAttribute("min_value",
                          $_builder.getIndexAttr(minValue));
      $_state.addAttribute("divisibility",
                          $_builder.getIndexAttr(divisibility));
      $_state.addTypes(indexType);
    }]>
  ];

  let extraClassDeclaration = [{
    /// Returns the minimum value for the tile size
    int64_t getMinValueInt() {
      return getMinValue().getSExtValue();
    }
    
    /// Returns the divisibility constraint
    int64_t getDivisibilityInt() {
      return getDivisibility().getSExtValue();
    }
  }];
}

//===----------------------------------------------------------------------===//
// Pipeline and Stage Operations
//===----------------------------------------------------------------------===//

def KTDF_PipelineOp : KTDF_Op<"pipeline", [
    NoTerminator,
    SingleBlock
  ]> {
  let summary = "Defines a pipeline containing stages";
  let description = [{
    The `ktdf.pipeline` operation represents a structured directed acyclic graph
    (DAG) of execution steps that process data in stages. Instead of executing one
    step completely before starting the next, pipelines allow different stages to
    operate concurrently on different iterations of work, thereby overlapping
    computation and data movement. In dataflow architectures, pipelines reflect
    how hardware engines operate: one unit may be loading data, another computing,
    and another storing results simultaneously.

    The immediate children of a pipeline must be `ktdf.stage` operations.

    The optional `modulo` attribute declares the pipeline's rolling-window depth.
    With `modulo(size: N)`, the pipeline rotates through `N` physical resource
    sets per `N` consecutive iterations.

    Example:
    ```mlir
    ktdf.pipeline modulo(size: 2) {
      ktdf.stage {
        // Stage 1 operations
      }
      ktdf.stage {
        // Stage 2 operations
      }
    }
    ```
  }];

  let arguments = (ins
    OptionalAttr<I64Attr>:$modulo
  );

  let regions = (region SizedRegion<1>:$body);

  let assemblyFormat = [{
    (`modulo` `(` `size` `:` $modulo^ `)`)? $body attr-dict
  }];

  let hasVerifier = 1;

  let builders = [
    OpBuilder<(ins), [{
      Region* bodyRegion = $_state.addRegion();
      bodyRegion->emplaceBlock();
    }]>
  ];

  let extraClassDeclaration = [{
    using StageRange = llvm::iterator_range<Block::op_iterator<StageOp>>;

    /// Get the block containing the stages.
    Block *getBodyBlock() { return &getBody().front(); }

    StageRange getStages();
    unsigned getNumStages();

    /// Returns the modulo size if the modulo attribute is set, otherwise
    /// returns std::nullopt.
    std::optional<int64_t> getModuloSize() {
      if (auto m = getModuloAttr()) {
        return m.getInt();
      }
      return std::nullopt;
    }
  }];
}

def KTDF_StageOp : KTDF_Op<"stage", [
    NoTerminator,
    SingleBlock,
    HasParent<"PipelineOp">,
    AttrSizedOperandSegments
  ]> {
  let hasVerifier = 1;
  let summary = "Defines a stage within a pipeline";
  let description = [{
    The `ktdf.stage` operation defines a single stage within a pipeline.
    A stage represents a logical step within a pipeline where a specific
    type of work is performed. Each stage "eventually" corresponds to a
    particular hardware capability such as memory transfer or computation.
    Stages form the building blocks of the pipeline and define how work flows
    through the system.

    Stages can specify dataflow dependencies using tokens:
    - `depends_in`: Stages which must be performed before this stage
    - `depends_out`: Stages which must be performed upon completion of current stage

    The immediate parent of a stage is always a `ktdf.pipeline`, and the immediate
    siblings of a stage are other stages.

    Example:
    ```mlir
    %token_a = ktdf.create_token : !ktdf.token
    %token_b = ktdf.create_token : !ktdf.token
    
    ktdf.pipeline {
      ktdf.stage depends_in(%token_a, %token_b) depends_out(none) {
        // Stage operations
      }
      ktdf.stage depends_in(none) depends_out(%token_a) {
        // Stage operations
      }
      ktdf.stage depends_in(none) depends_out(%token_b) {
        // Stage operations
      }
    }
    ```
  }];

  let arguments = (ins
    Variadic<KTDF_TokenType>:$depends_in,
    Variadic<KTDF_TokenType>:$depends_out,
    OptionalAttr<KTDF_ApplicableUnitsCustomAttr>:$applicable_units
  );

  let regions = (region SizedRegion<1>:$body);

  let hasCustomAssemblyFormat = 1;
  
  let skipDefaultBuilders = 1;

  let builders = [
    OpBuilder<(ins "ValueRange":$depends_in, "ValueRange":$depends_out), [{
      $_state.addOperands(depends_in);
      $_state.addOperands(depends_out);
      $_state.addAttribute("operandSegmentSizes",
                          $_builder.getDenseI32ArrayAttr({
                            static_cast<int32_t>(depends_in.size()),
                            static_cast<int32_t>(depends_out.size())
                          }));
      Region* bodyRegion = $_state.addRegion();
      bodyRegion->emplaceBlock();
    }]>
  ];
}

//===----------------------------------------------------------------------===//
// Data Transfer Operation
//===----------------------------------------------------------------------===//

def KTDF_DataTransferOp : KTDF_Op<"data_transfer", [
    AttrSizedOperandSegments
  ]> {
  let summary = "Transfer data between memory locations and FIFO slots";
  let description = [{
    The `ktdf.data_transfer` operation represents explicit data movement in the
    dataflow pipeline. It supports three types of transfers:
    
    1. **memref to memref**: Transfer data between different memory spaces
       (e.g., DRAM to LX, LX to DRAM)
    2. **memref to fifo**: Transfer data from memory to a FIFO slot
       (e.g., LX to FIFO for streaming to compute units)
    3. **fifo to memref**: Transfer data from a FIFO slot to memory
       (e.g., FIFO from compute units to SFP registers or LX)

    The operation explicitly specifies the source and destination, along with
    the size of data being transferred. This makes data movement visible in the
    IR and enables analysis of memory traffic patterns and pipeline behavior.

    ## Syntax

    ```
    data-transfer-op ::= `ktdf.data_transfer`
                         `from` ssa-value (`[` ssa-use-list `]`)?
                         (`size` `[` ssa-use-list `]`)?
                         `to` ssa-value (`[` ssa-use-list `]`)?
                         (`size` `[` ssa-use-list `]`)?
                         `:` type(source) `,` type(destination)
    ```

    ## Examples

    ```mlir
    // memref to memref: DRAM to LX
    ktdf.data_transfer from %A[%m, %n] size [1, 64] to %lx_A[%c0] size [64]
      : memref<1x64xf16, "DRAM">, memref<64xf16, "LX">

    // memref to fifo: LX to FIFO slot
    ktdf.data_transfer from %lx_A[%c0] size [64] to %fifo_slot_a size [64]
      : memref<64xf16, "LX">, !ktdf.fifo.slot<"lxlu-to-sfp", 64>

    // fifo to memref: FIFO slot to SFP register
    ktdf.data_transfer from %fifo_slot_a to %sfp_A[%c0] size [64]
      : !ktdf.fifo.slot<"lxlu-to-sfp", 64>, memref<64xf16, "SFP_REG">

    // memref to memref: LX to DRAM
    ktdf.data_transfer from %lx_C[%m1, %n1] size [1, 64] to %C[%m0 + %m1, %n0 + %n1] size [1, 64]
      : memref<?x?xf16, "LX">, memref<?x?xf16, "DRAM">
    ```

    ## Semantics

    - The transfer is atomic from the perspective of the stage containing it
    - Source and destination types must be compatible (same element type)
    - For FIFO transfers, the size must not exceed the FIFO slot capacity constraints
    - Indices and sizes are used to specify the exact region being transferred
  }];

  let arguments = (ins
    AnyTypeOf<[AnyMemRef, KTDF_FifoSlotType]>:$source,
    Variadic<Index>:$source_indices,
    Variadic<Index>:$source_sizes,
    AnyTypeOf<[AnyMemRef, KTDF_FifoSlotType]>:$destination,
    Variadic<Index>:$dest_indices,
    Variadic<Index>:$dest_sizes,
    OptionalAttr<DenseI64ArrayAttr>:$static_source_sizes,
    OptionalAttr<DenseI64ArrayAttr>:$static_dest_sizes
  );

  let assemblyFormat = [{
    `from` $source
    (`[` $source_indices^ `]`)?
    (`size` custom<SizesList>($source_sizes, $static_source_sizes)^)?
    `to` $destination
    (`[` $dest_indices^ `]`)?
    (`size` custom<SizesList>($dest_sizes, $static_dest_sizes)^)?
    attr-dict `:` type($source) `,` type($destination)
  }];

  let hasVerifier = 1;

  let builders = [
    // Build with mixed static/dynamic sizes (OpFoldResult can be either Attribute or Value)
    OpBuilder<(ins "Value":$source,
      "ValueRange":$sourceIndices,
      "ArrayRef<OpFoldResult>":$sourceSizes,
      "Value":$destination,
      "ValueRange":$destIndices,
      "ArrayRef<OpFoldResult>":$destSizes)>,
    
    // Build with all static sizes
    OpBuilder<(ins "Value":$source,
      "ValueRange":$sourceIndices,
      "ArrayRef<int64_t>":$sourceSizes,
      "Value":$destination,
      "ValueRange":$destIndices,
      "ArrayRef<int64_t>":$destSizes)>
  ];

  let extraClassDeclaration = [{
    /// Returns true if the source is a memref
    bool isSourceMemRef() {
      return llvm::isa<mlir::MemRefType>(getSource().getType());
    }

    /// Returns true if the source is a FIFO slot
    bool isSourceFifo() {
      return llvm::isa<FifoSlotType>(getSource().getType());
    }

    /// Returns true if the destination is a memref
    bool isDestMemRef() {
      return llvm::isa<mlir::MemRefType>(getDestination().getType());
    }

    /// Returns true if the destination is a FIFO slot
    bool isDestFifo() {
      return llvm::isa<FifoSlotType>(getDestination().getType());
    }

    /// Returns the transfer type (memref-to-memref, memref-to-fifo, fifo-to-memref)
    enum class TransferType { MemRefToMemRef, MemRefToFifo, FifoToMemRef };
    
    TransferType getTransferType() {
      if (isSourceMemRef() && isDestMemRef())
        return TransferType::MemRefToMemRef;
      if (isSourceMemRef() && isDestFifo())
        return TransferType::MemRefToFifo;
      return TransferType::FifoToMemRef;
    }
    
    /// Get the mixed source sizes (combining static and dynamic)
    SmallVector<OpFoldResult> getMixedSourceSizes() {
      SmallVector<OpFoldResult> result;
      if (auto staticSizes = getStaticSourceSizes()) {
        unsigned dynamicIdx = 0;
        for (int64_t size : *staticSizes) {
          if (ShapedType::isDynamic(size)) {
            result.push_back(getSourceSizes()[dynamicIdx++]);
          } else {
            result.push_back(IntegerAttr::get(IndexType::get(getContext()), size));
          }
        }
      } else {
        // All dynamic
        for (Value size : getSourceSizes()) {
          result.push_back(size);
        }
      }
      return result;
    }
    
    /// Get the mixed destination sizes (combining static and dynamic)
    SmallVector<OpFoldResult> getMixedDestSizes() {
      SmallVector<OpFoldResult> result;
      if (auto staticSizes = getStaticDestSizes()) {
        unsigned dynamicIdx = 0;
        for (int64_t size : *staticSizes) {
          if (ShapedType::isDynamic(size)) {
            result.push_back(getDestSizes()[dynamicIdx++]);
          } else {
            result.push_back(IntegerAttr::get(IndexType::get(getContext()), size));
          }
        }
      } else {
        // All dynamic
        for (Value size : getDestSizes()) {
          result.push_back(size);
        }
      }
      return result;
    }
    
    /// Returns true if all source sizes are static (compile-time constants)
    bool hasAllStaticSourceSizes() {
      if (!getStaticSourceSizes())
        return false;
      return llvm::none_of(*getStaticSourceSizes(),
                          [](int64_t size) { return ShapedType::isDynamic(size); });
    }
    
    /// Returns true if all destination sizes are static (compile-time constants)
    bool hasAllStaticDestSizes() {
      if (!getStaticDestSizes())
        return false;
      return llvm::none_of(*getStaticDestSizes(),
                          [](int64_t size) { return ShapedType::isDynamic(size); });
    }
    
    /// Get source sizes as static integers. Returns std::nullopt if any size is dynamic.
    std::optional<SmallVector<int64_t>> getStaticSourceSizesArray() {
      if (!hasAllStaticSourceSizes())
        return std::nullopt;
      ArrayRef<int64_t> sizes = *getStaticSourceSizes();
      return SmallVector<int64_t>(sizes.begin(), sizes.end());
    }
    
    /// Get destination sizes as static integers. Returns std::nullopt if any size is dynamic.
    std::optional<SmallVector<int64_t>> getStaticDestSizesArray() {
      if (!hasAllStaticDestSizes())
        return std::nullopt;
      ArrayRef<int64_t> sizes = *getStaticDestSizes();
      return SmallVector<int64_t>(sizes.begin(), sizes.end());
    }
  }];
}

//===----------------------------------------------------------------------===//
// Private Region Operations
//===----------------------------------------------------------------------===//

def KTDF_PrivateYieldOp : KTDF_Op<"private_yield", [
    Pure,
    ReturnLike,
    Terminator,
    HasParent<"PrivateOp">
  ]> {
  let summary = "Yield operation for private regions";
  let description = [{
    The `ktdf.private_yield` operation is a terminator for `ktdf.private`
    regions. It yields zero or more values from the private region back to
    the parent operation.
  }];

  let arguments = (ins
    Variadic<AnyType>:$operands
  );

  let assemblyFormat = "attr-dict ($operands^ `:` type($operands))?";

  let builders = [
    OpBuilder<(ins), [{
      build($_builder, $_state, ValueRange{});
    }]>
  ];
}

def KTDF_PrivateOp : KTDF_Op<"private", [
    SingleBlock,
    HasParent<"PipelineOp">
  ]> {
  let summary = "Define a private region with yielded results";
  let description = [{
    The `ktdf.private` operation defines a region yielding resources
    (tokens, memref allocations) private to its parent pipeline operation.
    
    Example:
    ```mlir
    ktdf.pipeline {
      %1:2 = ktdf.private -> (memref<64xf16, "SFP_REG">, !ktdf.token) {
        %sfp_A = memref.alloc() : memref<64xf16, "SFP_REG">           
        %t = ktdf.create_token : !ktdf.token
        ktdf.private_yield %sfp_A, %t : memref<64xf16, "SFP_REG">, !ktdf.token
      }

      // Use the token in a stage
      ktdf.stage depends_in(%1#1) depends_out(none) {
        // Stage operations
      }
    }
    ```
  }];

  let results = (outs
    Variadic<AnyType>:$results
  );

  let regions = (region SizedRegion<1>:$body);

  let assemblyFormat = [{
    (`->` `(` type($results)^ `)`)?  $body attr-dict-with-keyword
  }];

  let hasVerifier = 1;

  let builders = [
    OpBuilder<(ins), [{
      Region* bodyRegion = $_state.addRegion();
      bodyRegion->emplaceBlock();
    }]>,
    OpBuilder<(ins "::mlir::TypeRange":$results), [{
     Region* bodyRegion = $_state.addRegion();
     bodyRegion->emplaceBlock();
     $_state.addTypes(results);
    }]>
  ];

  let extraClassDeclaration = [{
    /// Get the block containing the private region body.
    Block *getBodyBlock() { return &getBody().front(); }

    /// Get the yield operation terminating this private region.
    PrivateYieldOp getYieldOp();
  }];
}

//===----------------------------------------------------------------------===//
// Parallel Operation
//===----------------------------------------------------------------------===//

def KTDF_ParallelOp : KTDF_Op<"parallel", [
    SingleBlock,
    AttrSizedOperandSegments
  ]> {
  let summary = "N-dim parallel iteration space distributed across hardware instances";
  let description = [{
    The `ktdf.parallel` operation expresses an N-dimensional parallel
    iteration space together with a pure-block distribution of those
    iterations across `num_instances` hardware instances. It is target
    agnostic — it does not say whether the instances are corelets, cores,
    or anything else; lowering binds the distribution axis to a concrete
    hardware axis.

    The body is a single block whose arguments are, in order:
      - N induction variables (the iteration coordinates), all `index`.
      - One trailing instance id, `index`-typed, ranging over
        `[0, num_instances)`.

    The op produces no SSA results. Control flow falls through after every
    instance has finished its assigned chunk (implicit barrier).

    ## Distribution semantics

    Iterations enumerate the Cartesian product
    `[lb_0, ub_0) x ... x [lb_{N-1}, ub_{N-1})` in lex order. With `T`
    total iterations and `N = num_instances`, the lex enumeration is split
    into `N` contiguous chunks of size `ceil(T / N)` (with the last chunk
    holding any leftover). Instance `i` runs chunk `i`. Distribution is
    always pure block — no cyclic or block-cyclic policy.

    ## Example

    ```mlir
    ktdf.parallel (%m, %n, %inst) = (%c0, %c0) to (%c32, %c1) step (%c1, %c1)
                                    distribute(num_instances = 2) {
      // Body uses %m, %n as induction vars and %inst as the instance id.
      ktdf.pipeline { /* ... */ }
      ktdf.parallel_yield
    }
    ```
  }];

  let arguments = (ins
    Variadic<Index>:$lower_bounds,
    Variadic<Index>:$upper_bounds,
    Variadic<Index>:$steps,
    I64Attr:$num_instances
  );

  let regions = (region SizedRegion<1>:$body);

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
  let skipDefaultBuilders = 1;

  let builders = [
    OpBuilder<(ins
      "::mlir::ValueRange":$lowerBounds,
      "::mlir::ValueRange":$upperBounds,
      "::mlir::ValueRange":$steps,
      "int64_t":$numInstances)>
  ];

  let extraClassDeclaration = [{
    /// Number of induction variables (== rank of the iteration space).
    unsigned getNumLoops() {
      return getLowerBounds().size();
    }

    /// Block argument for induction variable `dim`.
    ::mlir::BlockArgument getInductionVar(unsigned dim) {
      return getBody().front().getArgument(dim);
    }

    /// Block argument for the trailing instance id.
    ::mlir::BlockArgument getInstanceId() {
      return getBody().front().getArgument(getNumLoops());
    }

    /// The single block forming the body.
    ::mlir::Block *getBodyBlock() { return &getBody().front(); }
  }];
}

def KTDF_ParallelYieldOp : KTDF_Op<"parallel_yield", [
    Pure,
    ReturnLike,
    Terminator,
    HasParent<"ParallelOp">
  ]> {
  let summary = "Terminator for ktdf.parallel regions";
  let description = [{
    The `ktdf.parallel_yield` operation is a terminator for `ktdf.parallel`
    regions. It takes no operands and yields no values; the parent
    `ktdf.parallel` op produces no SSA results.

    Example:
    ```mlir
    ktdf.parallel (%m, %inst) = (%c0) to (%c64) step (%c1)
                                distribute(num_instances = 2) {
      ktdf.parallel_yield
    }
    ```
  }];

  let arguments = (ins);

  let assemblyFormat = "attr-dict";
  // No custom builder: the auto-generated `build(OpBuilder&, OperationState&)`
  // is sufficient for a no-operand, no-result op. A previously-defined custom
  // OpBuilder<(ins), [{ build($_builder, $_state); }]> recursed into itself
  // and crashed when invoked via `ParallelYieldOp::create(builder, loc)`.
}

#endif // KTDF_SRC_DIALECTS_KTDF_OPS_TD
