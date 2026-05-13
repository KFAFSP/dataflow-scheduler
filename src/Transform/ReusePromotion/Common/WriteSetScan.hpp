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
// Region write-set scan for reuse-promotion passes.
//===----------------------------------------------------------------------===//

#ifndef KTIR_SCHEDULER_TRANSFORM_REUSEPROMOTION_WRITESETSCAN_HPP
#define KTIR_SCHEDULER_TRANSFORM_REUSEPROMOTION_WRITESETSCAN_HPP

#include "mlir/IR/Region.h"
#include "mlir/IR/Value.h"

namespace ktdf {
namespace reuse {

/// Returns true iff any op inside `region` (recursively) writes through
/// `memref` or a value provably aliased to it. `ignore`, if non-null, is
/// excluded from the scan (used to exclude the transfer being hoisted
/// when checking its own destination).
///
/// Aliasing model:
///   - The input `memref` and any value obtained from it via memref.cast,
///     memref.view, memref.subview, or memref.reinterpret_cast is
///     considered aliased. This is recursive through intermediate views.
///   - Function arguments are assumed non-aliasing with other function
///     arguments (matches the spec's TODO on ktdf.noalias).
///
/// Writers recognised (all report as writes when their destination alias
/// matches `memref`):
///   - memref.store
///   - linalg ops: any op with memref-type output operand
///   - ktdf.data_transfer with memref destination
///
/// Scans are conservative: if an op's side-effect model is unknown, it is
/// treated as a non-writer (the pass will simply miss an optimisation, not
/// produce wrong IR — except when the op is actually a memref writer not
/// on this list. New writer kinds should be added here as they appear in
/// the codebase.)
bool regionWritesTo(mlir::Region &region, mlir::Value memref,
                    mlir::Operation *ignore = nullptr);

}  // namespace reuse
}  // namespace ktdf

#endif  // KTIR_SCHEDULER_TRANSFORM_REUSEPROMOTION_WRITESETSCAN_HPP
