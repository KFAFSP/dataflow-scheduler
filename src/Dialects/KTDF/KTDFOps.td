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
    /// Get the block containing the stages.
    Block* getBodyBlock() { return &getBody().front(); }

    /// Get all stage operations within this pipeline.
    auto getStages() {
      return getBodyBlock()->getOps<StageOp>();
    }

    /// Get the number of stages in this pipeline.
    unsigned getNumStages() {
      return std::distance(getStages().begin(), getStages().end());
    }
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

#endif // KTDF_SRC_DIALECTS_KTDF_OPS_TD