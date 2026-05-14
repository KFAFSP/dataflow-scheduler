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
include "mlir/IR/EnumAttr.td"

include "KTDFDialect.td"
include "KTDFEnums.td"

// Base class for KTDF dialect attributes
class KTDF_Attr<string name, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<KTDF_Dialect, name, traits> {
  let mnemonic = attrMnemonic;
}

//===----------------------------------------------------------------------===//
// MemorySpaceAttr
//===----------------------------------------------------------------------===//

def KTDF_MemorySpaceCustomAttr : KTDF_Attr<"MemorySpaceCustom", "memory_space"> {
  let summary = "KTDF memory space attribute for memref types";
  let description = [{
    Syntax:

    ```
    memory-space-attr ::= `memory_space` `<` memory-space-kind `>`
    ```

    The memory space attribute provides an enum-backed memory space for memref
    types, using a fixed set of keywords rather than arbitrary strings.

    Example:

    ```mlir
    %hbm_buf = memref.alloc() : memref<4096xi32, #ktdf.memory_space<HBM>>
    ```
  }];

  let parameters = (ins "::mlir::ktdf::MemorySpace":$value);

  let assemblyFormat = "`<` $value `>`";
}

//===----------------------------------------------------------------------===//
// LoopTypeAttr
//===----------------------------------------------------------------------===//

def KTDF_LoopTypeAttr : EnumAttr<KTDF_Dialect, KTDF_LoopTypeEnum, "loop_type"> {
  let summary = "KTDF loop type attribute for classifying loops";
  let description = [{
    The loop type attribute provides an enum-backed classification for loops,
    distinguishing between parallel and reduction loops.

    Example:

    ```mlir
    scf.for %i = %c0 to %c10 step %c1 {loop_type = #ktdf.loop_type<parallel_loop>} {
      // parallel loop body
    }
    ```
  }];
  let assemblyFormat = "`<` $value `>`";
}

//===----------------------------------------------------------------------===//
// ApplicableUnitsAttr
//===----------------------------------------------------------------------===//

def KTDF_ApplicableUnitsCustomAttr : KTDF_Attr<"ApplicableUnitsCustom", "applicable_units"> {
  let summary = "KTDF applicable units attribute for stage operations";
  let description = [{
    Syntax:

    ```
    applicable-units-attr ::= `applicable_units` `<` `[` applicable-units-list `]` `>`
    applicable-units-list ::= component-type (`,` component-type)*
    ```

    The applicable units attribute specifies a non-empty list of processing units
    that a stage operation can be executed on. This allows flexibility in scheduling
    by indicating which units are capable of executing the operation.

    Example:

    ```mlir
    ktdf.stage {applicable_units = #ktdf.applicable_units<[SFP]>} {
      // stage body
    }
    
    ktdf.stage {applicable_units = #ktdf.applicable_units<[LXLU, LXSU]>} {
      // stage body contains child stages for LXLU, LXSU
    }
    ```
  }];

  let parameters = (ins ArrayRefParameter<"::mlir::ktdf::ComponentType">:$value);

  let assemblyFormat = "`<` `[` $value `]` `>`";
}

#endif // KTDF_SRC_DIALECTS_KTDF_ATTRIBUTES_TD