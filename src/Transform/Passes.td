#ifndef SCHEDULER_TRANSFORM_PASSES
#define SCHEDULER_TRANSFORM_PASSES

include "mlir/Pass/PassBase.td"

def DummyTransformPass : Pass<"scheduler-dummy-transform", "mlir::ModuleOp"> {
  let summary = "Dummy scheduler transform";
  let description = [{
    A no-op transform pass used to establish the scheduler transform pipeline.
  }];
  let constructor = "ktdf::createDummyTransformPass()";
}

#endif // SCHEDULER_TRANSFORM_PASSES