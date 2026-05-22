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
// This file declares the KTDF types in the MLIR type system.
//
//===----------------------------------------------------------------------===//

#ifndef KTDF_SRC_DIALECTS_KTDF_TYPES_HPP_
#define KTDF_SRC_DIALECTS_KTDF_TYPES_HPP_

#include <cstdint>

#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"

namespace mlir::ktdf {

enum class FifoType : uint32_t;
enum class MemorySpace : uint32_t;
enum class ComponentType : uint32_t;
enum class LoopType : uint32_t;
enum class ApplicableUnits : uint32_t;

// Placeholder for future KTDF type declarations

}  // namespace mlir::ktdf

/// Include the auto-generated header file containing the declarations of the
/// KTDF types.
#define GET_TYPEDEF_CLASSES
#include "src/Dialects/KTDF/KTDFOpsTypes.hpp.inc"

/// Include the auto-generated header file containing the declarations of the
/// KTDF attributes.
#define GET_ATTRDEF_CLASSES
#include "src/Dialects/KTDF/KTDFAttributes.hpp.inc"

namespace mlir::ktdf {

// Placeholder for future KTDF type builders and utilities

}  // namespace mlir::ktdf

#endif  // KTDF_SRC_DIALECTS_KTDF_TYPES_HPP_
