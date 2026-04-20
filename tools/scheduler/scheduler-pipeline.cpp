#include "Transform/Passes.h"
#include "tools/scheduler-driver-lib/scheduler-main.h"

#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

namespace scheduler {

void registerPassesForScheduler() { registerSchedulerTransformsPasses(); }

void registerPassPipelinesForScheduler() {
  mlir::PassPipelineRegistration<>(
      "kEmitDFIR", "Emit DataflowIR",
      [](mlir::OpPassManager &pm) { ::ktdf::buildKTDPToDFIRPipeline(pm); });
}

}  // namespace scheduler

namespace ktdf {

void buildKTDPToDFIRPipeline(mlir::OpPassManager &pm) {
  pm.addPass(createDummyTransformPass());
}

}  // namespace ktdf
