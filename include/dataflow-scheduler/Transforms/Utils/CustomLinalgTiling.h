//===-------------------------------------------------------------*- c++ -*-==//
//
// Part of the Dataflow Scheduler project.
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
// Custom tiling without iter_args - minimal modification of upstream
// tileLinalgOpImpl.
//
//===----------------------------------------------------------------------===//

#ifndef DATAFLOW_SCHEDULER_TRANSFORMS_CUSTOMLINALGTILING_H_
#define DATAFLOW_SCHEDULER_TRANSFORMS_CUSTOMLINALGTILING_H_

#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/Linalg/Transforms/Transforms.h>
#include <mlir/IR/PatternMatch.h>

namespace scheduler {

/// Custom tiling that creates loops WITHOUT iter_args.
/// This is a minimal modification of the upstream tileLinalgOp that:
/// 1. Returns empty scf::ValueVector from the loop body builder
/// 2. Returns tensorResults directly from the tiled op (not from loop results)
llvm::FailureOr<mlir::linalg::TiledLinalgOp> customTileLinalgOp(
    mlir::RewriterBase& rewriter, mlir::linalg::LinalgOp op,
    const mlir::linalg::LinalgTilingOptions& options);

}  // namespace scheduler

#endif  // DATAFLOW_SCHEDULER_TRANSFORMS_CUSTOMLINALGTILING_H_

// Made with Bob
