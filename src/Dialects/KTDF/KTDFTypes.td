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

include "KTDFDialect.td"

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

#endif // KTDF_SRC_DIALECTS_KTDF_TYPES_TD