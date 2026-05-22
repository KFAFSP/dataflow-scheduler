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
// This file defines the KTDF (Kernel Tiled Data Flow) dialect.
//
//===----------------------------------------------------------------------===//

#ifndef KTDF_SRC_DIALECTS_KTDF_DIALECT_TD
#define KTDF_SRC_DIALECTS_KTDF_DIALECT_TD

#ifndef OP_BASE
include "mlir/IR/OpBase.td"
#endif // OP_BASE

def KTDF_Dialect : Dialect {
  let name = "ktdf";
  let cppNamespace = "::mlir::ktdf";
  
  let useDefaultAttributePrinterParser = 1;
  let useDefaultTypePrinterParser = 1;

  let dependentDialects = [
    "scf::SCFDialect"
  ];

  let extraClassDeclaration = [{
    void registerTypes();
    void registerAttributes();
  }];
}

#endif // KTDF_SRC_DIALECTS_KTDF_DIALECT_TD