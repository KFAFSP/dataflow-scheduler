//===----------------------- CustomTiling.h -------------------------------===//
//
// IBM Confidential
// (C) Copyright IBM Corp. 2026
//
// =============================================================================
//
// Custom tiling without iter_args - minimal modification of upstream
// tileLinalgOpImpl.
//
//===----------------------------------------------------------------------===//

#ifndef KTIR_TRANSFORM_CUSTOMTILING_H
#define KTIR_TRANSFORM_CUSTOMTILING_H

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir {
namespace linalg {

/// Custom tiling that creates loops WITHOUT iter_args.
/// This is a minimal modification of the upstream tileLinalgOp that:
/// 1. Returns empty scf::ValueVector from the loop body builder
/// 2. Returns tensorResults directly from the tiled op (not from loop results)
FailureOr<TiledLinalgOp> customTileLinalgOp(RewriterBase& rewriter, LinalgOp op,
                                            const LinalgTilingOptions& options);

}  // namespace linalg
}  // namespace mlir

#endif  // KTIR_TRANSFORM_CUSTOMTILING_H

// Made with Bob
