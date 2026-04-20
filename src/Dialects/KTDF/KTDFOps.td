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

    Sizes can be specified either as:
    1. Static values in the result type (e.g., `!ktdf.fifo_slot<"lxlu-to-sfp", 4>`)
    2. Dynamic values using SSA operands, with `?` in the result type

    When using dynamic sizes, provide one index operand per dynamic slot.
    The sizes are matched to the result types in order.

    FIFO slots are allocated and accessed in a rolling fashion.

    Example:

    ```mlir
    // Allocate two FIFO slots with static sizes
    %slot0, %slot1 = ktdf.fifo.allocate() -> !ktdf.fifo_slot<"lxlu-to-sfp", 1>, !ktdf.fifo_slot<"lxlu-to-sfp", 4>
    
    // Allocate a single FIFO slot with dynamic size
    %size = arith.constant 8 : index
    %slot = ktdf.fifo.allocate(%size) -> !ktdf.fifo_slot<"sfp-to-lxsu", ?>
    
    // Mix static and dynamic sizes
    %dyn_size = arith.constant 16 : index
    %s0, %s1 = ktdf.fifo.allocate(%dyn_size) -> !ktdf.fifo_slot<"sfp-to-sfp", ?>, !ktdf.fifo_slot<"sfp-to-sfp", 4>
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

#endif // KTDF_SRC_DIALECTS_KTDF_OPS_TD