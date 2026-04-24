#ifndef SCHEDULER_TRANSFORM_PASSES_H
#define SCHEDULER_TRANSFORM_PASSES_H

#include <memory>

#include "mlir/Pass/PassRegistry.h"

namespace mlir {
class Pass;
class OpPassManager;
}  // namespace mlir

namespace ktdf {

std::unique_ptr<mlir::Pass> createComputeGroupExtractionPass();
std::unique_ptr<mlir::Pass> createTileSCFForLoopsPass();
void buildKTDPToDFIRPipeline(mlir::OpPassManager& pm);

}  // namespace ktdf

#define GEN_PASS_DECL
#define GEN_PASS_REGISTRATION
#include "src/Transform/Passes.h.inc"

#endif  // SCHEDULER_TRANSFORM_PASSES_H
