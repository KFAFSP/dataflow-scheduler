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
// This file defines KTDF types.
//
//===----------------------------------------------------------------------===//

#ifndef KTDF_SRC_DIALECTS_KTDF_TYPES_TD
#define KTDF_SRC_DIALECTS_KTDF_TYPES_TD

#ifndef OP_BASE
include "mlir/IR/OpBase.td"
#endif // OP_BASE
include "mlir/IR/EnumAttr.td"

include "src/Dialects/KTDF/KTDFDialect.td"
include "src/Dialects/KTDF/KTDFEnums.td"

// Base class for KTDF dialect types
class KTDF_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<KTDF_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

//===----------------------------------------------------------------------===//
// TokenType
//===----------------------------------------------------------------------===//

def KTDF_TokenType : KTDF_Type<"Token", "token"> {
  let summary = "KTDF token type for dataflow synchronization";
  let description = [{
    Syntax:

    ```
    token-type ::= `token`
    ```

    The token type represents a synchronization token in the KTDF dialect.
    It is used to model dataflow dependencies and control flow in dataflow
    graphs. Tokens have no runtime representation and are used purely for
    scheduling and dependency tracking.

    Example:

    ```mlir
    // A token value
    %token = ktdf.create_token : !ktdf.token
    ...
    // Token used for synchronization
    ktdf.stage depend_in(%token) ...
    ```
  }];

  let parameters = (ins);

  let assemblyFormat = "";
}

//===----------------------------------------------------------------------===//
// FifoSlotType
//===----------------------------------------------------------------------===//

def KTDF_FifoSlotType : KTDF_Type<"FifoSlot", "fifo.slot"> {
  let summary = "KTDF FIFO slot type for dataflow buffering";
  let description = [{
    Syntax:

    ```
    fifo.slot-type ::= `fifo.slot` `<` fifo-type-string `,` shaped-element-type `>`
    fifo-type-string ::= `"sfp-to-lxsu"` | `"lxlu-to-sfp"` | `"hbm-to-sfp"` | `"sfp-to-hbm"`
    shaped-element-type ::= (num-elements `x`)? element-type
    num-elements ::= `?` | decimal-literal
    element-type ::= type
    ```

    The fifo.slot type represents a slot in a FIFO buffer used for dataflow
    communication between stages. Each slot has:
    - A specific type that indicates the source and destination of the dataflow
      (e.g., "sfp-to-lxsu" means data flows from SFP to LXSU)
    - A fifo slot capacity in terms of number of elements that can be static or dynamic
    - An element type that specifies the type of data being transferred

    The number of elements can be:
    - A static positive integer (e.g., 4, 8, 16, 64)
    - Dynamic, indicated by `?` (similar to memref dimensions)

    Example:

    ```mlir
    // A FIFO slot from LXLU to SFP with 64 f16 elements
    %slot0 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"lxlu-to-sfp", 64xf16>
    
    // A FIFO slot from SFP to LXSU with some dynamic number of f16 elements
    %size = arith.constant 8 : index
    %slot1 = ktdf.fifo.allocate(%size) -> !ktdf.fifo.slot<"sfp-to-lxsu", ?xf16>
    ```
  }];

  let parameters = (ins
    "FifoType":$fifoType,
    "int64_t":$numElements,
    "Type":$elementType
  );

  let genStorageClass = 1;
  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = [{
    /// Returns true if the number of elements is dynamic
    bool isDynamicNumElements() const {
      return mlir::ShapedType::isDynamic(getNumElements());
    }
    
    /// Returns the static number of elements, or kDynamic if dynamic
    int64_t getStaticNumElements() const { return getNumElements(); }
  }];
}

#endif // KTDF_SRC_DIALECTS_KTDF_TYPES_TD