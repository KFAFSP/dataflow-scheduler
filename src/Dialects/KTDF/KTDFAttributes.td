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
// This file defines KTDF attributes.
//
//===----------------------------------------------------------------------===//

#ifndef KTDF_SRC_DIALECTS_KTDF_ATTRIBUTES_TD
#define KTDF_SRC_DIALECTS_KTDF_ATTRIBUTES_TD

#ifndef OP_BASE
include "mlir/IR/OpBase.td"
#endif // OP_BASE
include "mlir/IR/AttrTypeBase.td"

include "KTDFDialect.td"

// Base class for KTDF dialect attributes
class KTDF_Attr<string name, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<KTDF_Dialect, name, traits> {
  let mnemonic = attrMnemonic;
}

//===----------------------------------------------------------------------===//
// MemorySpaceAttr
//===----------------------------------------------------------------------===//

def KTDF_MemorySpaceAttr : KTDF_Attr<"MemorySpace", "memory_space"> {
  let summary = "KTDF memory space attribute for memref types";
  let description = [{
    Syntax:

    ```
    memory-space-attr ::= `memory_space` `<` string-literal `>`
    ```

    The memory space attribute provides a named memory space for memref types,
    replacing the default integer-based memory space with a human-readable string.
    This allows for more descriptive IR when working with different memory regions
    such as "DDR", "SRAM", "HBM", "L1", "L2", etc.

    Example:

    ```mlir
    // A memref in DDR memory
    %buffer = memref.alloc() : memref<1024xf32, memory_space<"DDR">>
    
    // A memref in SRAM memory
    %local = memref.alloc() : memref<64xf16, memory_space<"SRAM">>
    
    // A memref in HBM memory
    %hbm_buf = memref.alloc() : memref<4096xi32, memory_space<"HBM">>
    ```
  }];

  let parameters = (ins StringRefParameter<>:$name);

  let assemblyFormat = "`<` $name `>`";

  let extraClassDeclaration = [{
    /// Returns the memory space name as a string
    llvm::StringRef getMemorySpaceName() const { return getName(); }
  }];
}

#endif // KTDF_SRC_DIALECTS_KTDF_ATTRIBUTES_TD