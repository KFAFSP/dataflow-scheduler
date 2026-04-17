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

// Import the KTDF type definitions
include "KTDFTypes.td"

// Base class for KTDF dialect operations
class KTDF_Op<string mnemonic, list<Trait> traits = []> :
    Op<KTDF_Dialect, mnemonic, traits> {
}

// Placeholder for future KTDF operations
// Example:
// def KTDF_ScheduleOp : KTDF_Op<"schedule", []> {
//   let summary = "Schedule operation";
//   let description = [{
//     Schedules a computation for execution.
//   }];
// }

#endif // KTDF_SRC_DIALECTS_KTDF_OPS_TD