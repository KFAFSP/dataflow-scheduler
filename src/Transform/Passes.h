#ifndef SCHEDULER_TRANSFORM_PASSES_H
#define SCHEDULER_TRANSFORM_PASSES_H

#include <memory>

namespace mlir {
class Pass;
class OpPassManager;
}  // namespace mlir

namespace ktdf {

std::unique_ptr<mlir::Pass> createDummyTransformPass();
void buildKTDPToDFIRPipeline(mlir::OpPassManager &pm);

}  // namespace ktdf

#define GEN_PASS_DECL
#include "src/Transform/Passes.h.inc"

#endif // SCHEDULER_TRANSFORM_PASSES_H
