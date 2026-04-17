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
// This file declares the KTDF operations in the MLIR operation set.
//
//===----------------------------------------------------------------------===//

#ifndef KTDF_SRC_DIALECTS_KTDF_HPP_
#define KTDF_SRC_DIALECTS_KTDF_HPP_

#include <map>
#include <string>

#include "mlir/Dialect/Affine/IR/AffineMemoryOpInterfaces.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "src/Dialects/KTDF/KTDFOpsDialect.hpp.inc"
#include "src/Dialects/KTDF/KTDFOpsEnums.hpp.inc"
#define GET_ATTRDEF_CLASSES
#include "src/Dialects/KTDF/KTDFOpsAttributes.hpp.inc"

/// Include the auto-generated header file containing the declarations of the
/// KTDF types.
#include "src/Dialects/KTDF/KTDFTypes.hpp"

/// Include the auto-generated header file containing the declarations of the
/// KTDF operations.
#define GET_OP_CLASSES
#include "src/Dialects/KTDF/KTDFOps.hpp.inc"

#endif  // KTDF_SRC_DIALECTS_KTDF_HPP_

