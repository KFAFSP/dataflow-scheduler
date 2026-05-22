#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "src/Transform/Passes.h"
#include "src/Utils/SchedulerExtContext.h"
#include "tools/scheduler-driver-lib/scheduler-main.h"

// Include MLIR pass registration headers as needed
#include "mlir/Dialect/SCF/Transforms/Passes.h"
#include "mlir/Transforms/Passes.h"  // For SCCP, Canonicalizer, SymbolDCE, LoopInvariantCodeMotion
// #include "mlir/InitAllPasses.h"  // Uncomment to register all MLIR passes

namespace scheduler {

void registerPassesForScheduler() {
  // Register scheduler-specific passes
  registerSchedulerTransformPasses();

  // Register some common opt passes
  mlir::registerSCCPPass();
  mlir::registerSymbolDCEPass();
  mlir::registerCanonicalizerPass();
  mlir::registerLoopInvariantCodeMotionPass();

  // Register MLIR dialect passes as needed
  mlir::registerSCFPasses();

  // mlir::registerAllPasses();  // Or register all MLIR passes at once
}

void registerPassPipelinesForScheduler() {
  mlir::PassPipelineRegistration<>(
      "kEmitDFIR", "Emit DataflowIR", [](mlir::OpPassManager& pm) {
        ::ktdf::buildKTDPToDFIRPipeline(
            pm, scheduler::SchedulerExtContext::dummyContext());
      });
}

}  // namespace scheduler

namespace ktdf {

void buildKTDPToDFIRPipeline(
    mlir::OpPassManager& pm,
    const scheduler::SchedulerExtContext& scheduler_ctx) {
  pm.addPass(createComputeGroupExtractionPass());
  pm.addPass(createConstructThreeStagePipelinePass(scheduler_ctx));
  pm.addPass(createParallelizeLoopsAcrossInstancesPass(scheduler_ctx));
  pm.addPass(createDoubleBufferingPass(scheduler_ctx));
  // todo: pm.addPass(createTileSCFForLoopsPass());
}

}  // namespace ktdf
