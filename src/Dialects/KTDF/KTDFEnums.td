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
// This file defines enums used in the KTDF dialect.
//
//===----------------------------------------------------------------------===//

#ifndef KTDF_ENUMS
#define KTDF_ENUMS

include "mlir/IR/EnumAttr.td"
include "src/Dialects/KTDF/KTDFDialect.td"

//===----------------------------------------------------------------------===//
// Memory Space Enum
//===----------------------------------------------------------------------===//

def KTDF_MemorySpaceEnum : I32EnumAttr<"MemorySpace",
    "KTDF memory space",
    [
      I32EnumAttrCase<"LX", 0, "LX">,
      I32EnumAttrCase<"HBM", 1, "HBM">
    ]> {
  let cppNamespace = "::mlir::ktdf";
  let genSpecializedAttr = 0;
}

//===----------------------------------------------------------------------===//
// FIFO Type Enum
//===----------------------------------------------------------------------===//

def KTDF_FifoTypeEnum : I32EnumAttr<"FifoType",
    "FIFO connection type",
    [
      I32EnumAttrCase<"SfpToLxsu", 0, "sfp-to-lxsu">,
      I32EnumAttrCase<"LxluToSfp", 1, "lxlu-to-sfp">
    ]> {
  let cppNamespace = "::mlir::ktdf";
  let genSpecializedAttr = 0;
}

//===----------------------------------------------------------------------===//
// Loop Type Enum
//===----------------------------------------------------------------------===//

def KTDF_LoopTypeEnum : I32EnumAttr<"LoopType",
    "Loop type classification",
    [
      I32EnumAttrCase<"ParallelLoop", 0, "parallel_loop">,
      I32EnumAttrCase<"ReductionLoop", 1, "reduction_loop">
    ]> {
  let cppNamespace = "::mlir::ktdf";
  let genSpecializedAttr = 0;
}

#endif // KTDF_ENUMS