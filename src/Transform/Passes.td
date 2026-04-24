#ifndef SCHEDULER_TRANSFORM_PASSES
#define SCHEDULER_TRANSFORM_PASSES

include "mlir/Pass/PassBase.td"

def DummyTransformPass : Pass<"scheduler-dummy-transform", "mlir::ModuleOp"> {
  let summary = "Dummy scheduler transform";
  let description = [{
    A no-op transform pass used to establish the scheduler transform pipeline.
  }];
  let constructor = "::ktdf::createDummyTransformPass()";
  let dependentDialects = ["mlir::ktdf::KTDFDialect"];
}

def ComputeGroupExtractionPass : Pass<"compute-group-extraction", "mlir::ModuleOp"> {
  let summary = "Extract main and compute group modules from a module";
  let description = [{
    This pass extracts the main and compute groups from the initial module.
  }];
  let constructor = "ktdf::createComputeGroupExtractionPass()";
}

#endif // SCHEDULER_TRANSFORM_PASSES