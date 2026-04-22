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

    Example:
    ```mlir
    ktdf.pipeline {
      ktdf.stage {
        // Stage 1 operations
      }
      ktdf.stage {
        // Stage 2 operations
      }
    }
    ```
  }];

  let regions = (region SizedRegion<1>:$body);

  let assemblyFormat = "$body attr-dict";

  let extraClassDeclaration = [{
    using StageRange = llvm::iterator_range<Block::op_iterator<StageOp>>;

    /// Get the block containing the stages.
    Block *getBodyBlock() { return &getBody().front(); }

    StageRange getStages();
    unsigned getNumStages();
  }];
}

def KTDF_StageOp : KTDF_Op<"stage", [
    NoTerminator,
    SingleBlock,
    HasParent<"PipelineOp">
  ]> {
  let summary = "Defines a stage within a pipeline";
  let description = [{
    The  operation defines a single stage within a pipeline.
    A `ktdf.stage` represents a logical step within a pipeline where a specific
    type of work is performed. Each stage "eventually" correspond to a
    particular hardware capability such as memory transfer or computation.
    Stages form the building blocks of the pipeline and define how work flows
    through the system.

    The immediate parent of a stage is always a `ktdf.pipeline`, and the immediate
    siblings of a stage are other stages.

    Example:
    ```mlir
    ktdf.pipeline {
      ktdf.stage {
        // Stage operations
      }
    }
    ```
  }];

  let regions = (region SizedRegion<1>:$body);

  let assemblyFormat = "$body attr-dict";
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
    Variadic<Index>:$dest_sizes
  );

  let assemblyFormat = [{
    `from` $source
    (`[` $source_indices^ `]`)?
    (`size` `[` $source_sizes^ `]`)?
    `to` $destination
    (`[` $dest_indices^ `]`)?
    (`size` `[` $dest_sizes^ `]`)?
    attr-dict `:` type($source) `,` type($destination)
  }];

  let hasVerifier = 1;

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
  }];
}

#endif // KTDF_SRC_DIALECTS_KTDF_OPS_TD