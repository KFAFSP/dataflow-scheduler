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
// SSA invariance check for reuse-promotion passes.
//===----------------------------------------------------------------------===//

#ifndef KTIR_SCHEDULER_TRANSFORM_REUSEPROMOTION_INVARIANCECHECK_HPP
#define KTIR_SCHEDULER_TRANSFORM_REUSEPROMOTION_INVARIANCECHECK_HPP

#include "mlir/IR/Value.h"
#include "src/Dialects/KTDF/KTDFOps.hpp"

namespace ktdf {
namespace reuse {

/// Returns true iff `v` transitively depends on `iv` through SSA use-def
/// chains. Handles cycles safely via a visited set.
bool ssaDependsOn(mlir::Value v, mlir::Value iv);

/// Returns true iff neither the source indices/sizes nor the destination
/// indices/sizes of `transfer` transitively depend on `iv`. The transfer's
/// source/destination memref operands themselves are NOT checked — only
/// the index and size SSA operands.
bool transferIsInvariantWrt(mlir::ktdf::DataTransferOp transfer,
                            mlir::Value iv);

}  // namespace reuse
}  // namespace ktdf

#endif  // KTIR_SCHEDULER_TRANSFORM_REUSEPROMOTION_INVARIANCECHECK_HPP
