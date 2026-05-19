#ifndef SCHEDULER_TRANSFORM_PASSES_H
#define SCHEDULER_TRANSFORM_PASSES_H

#include <memory>

#include "mlir/Pass/PassRegistry.h"

namespace mlir {
class Pass;
class OpPassManager;
}  // namespace mlir

namespace scheduler {
struct SchedulerExtContext;
}  // namespace scheduler

namespace ktdf {

std::unique_ptr<mlir::Pass> createComputeGroupExtractionPass();
std::unique_ptr<mlir::Pass> createConstructThreeStagePipelinePass(
    const scheduler::SchedulerExtContext& scheduler_ctx);
std::unique_ptr<mlir::Pass> createConstructThreeStagePipelinePass();
std::unique_ptr<mlir::Pass> createTileSCFForLoopsPass();
std::unique_ptr<mlir::Pass> createStripMineSCFForLoopsPass();
std::unique_ptr<mlir::Pass> createStageCoarseningPass();
std::unique_ptr<mlir::Pass> createPathExpansionPass();

std::unique_ptr<mlir::Pass> createBroadcastPromotionPass();
std::unique_ptr<mlir::Pass> createParallelizeLoopsAcrossInstancesPass();
std::unique_ptr<mlir::Pass> createParallelizeLoopsAcrossInstancesPass(
    const scheduler::SchedulerExtContext& scheduler_ctx);
std::unique_ptr<mlir::Pass> createDoubleBufferingPass();
std::unique_ptr<mlir::Pass> createDoubleBufferingPass(
    const scheduler::SchedulerExtContext& scheduler_ctx);
void buildKTDPToDFIRPipeline(
    mlir::OpPassManager& pm,
    const scheduler::SchedulerExtContext& scheduler_ctx);

}  // namespace ktdf

#define GEN_PASS_DECL
#define GEN_PASS_REGISTRATION
#include "src/Transform/Passes.h.inc"

#endif  // SCHEDULER_TRANSFORM_PASSES_H
