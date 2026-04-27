#ifndef SCHEDULER_TRANSFORM_PASSES
#define SCHEDULER_TRANSFORM_PASSES

include "mlir/Pass/PassBase.td"

def ComputeGroupExtractionPass : Pass<"compute-group-extraction", "mlir::ModuleOp"> {
  let summary = "Extract main and compute group modules from a module";
  let description = [{
    This pass extracts the main and compute groups from the initial module.
  }];
  let constructor = "::ktdf::createComputeGroupExtractionPass()";
}

def TileSCFForLoopsPass : Pass<"tile-scf-for-loops", "mlir::ModuleOp"> {
  let summary = "Tile perfectly nested scf.for loops";
  let description = [{
    This pass tiles perfectly nested scf.for loops.
    
    Example transformation:
    ```
    scf.for %i = 0 to 100 step 1 {
      scf.for %j = 0 to 100 step 1 { ... }
    }
    ```
    becomes:
    ```
    scf.for %i0 = 0 to 100 step 32 {
      scf.for %j0 = 0 to 100 step 32 {
        scf.for %i1 = %i0 to min(%i0+32, 100) step 1 {
          scf.for %j1 = %j0 to min(%j0+32, 100) step 1 { ... }
        }
      }
    }
    ```
  }];
  let constructor = "::ktdf::createTileSCFForLoopsPass()";
  let options = [
    ListOption<"tileSizes", "tile-sizes", "int64_t",
               "Tile sizes for each loop dimension (default: 32 for all)">,
    Option<"normalizeInnerLoops", "normalize-inner-loops", "bool",
           /*default=*/"true",
           "Normalize inner tiled loops to start at 0 and add outer IV to uses">
  ];
}

def StripMineSCFForLoopsPass : Pass<"strip-mine-scf-for-loops", "mlir::ModuleOp"> {
  let summary = "Strip-mine scf.for loops";
  let description = [{
    This pass performs strip-mining on scf.for loops.
    Strip-mining splits a loop into an outer loop and an inner loop,
    where the outer loop iterates over blocks and the inner loop
    iterates within each block.
    
    Example transformation:
    ```
    scf.for %i = 0 to 100 step 1 { ... }
    ```
    becomes:
    ```
    scf.for %i0 = 0 to 100 step 32 {
      scf.for %i1 = %i0 to min(%i0+32, 100) step 1 { ... }
    }
    ```
  }];
  let constructor = "::ktdf::createStripMineSCFForLoopsPass()";
  let options = [
    ListOption<"stripMineSize", "strip-mine-size", "int64_t",
               "Strip-mine factor (block size, default: 32)">,
    Option<"normalizeInnerLoop", "normalize-inner-loop", "bool",
           /*default=*/"true",
           "Normalize inner loop to start at 0 and add outer IV to uses">
  ];
}

def PrintPipelineTreePass : Pass<"print-pipeline-tree", "mlir::ModuleOp"> {
  let summary = "Print the pipeline tree structure";
  let description = [{
    This pass builds and prints the internal pipeline tree abstraction
    for all functions in a module. The tree represents the hierarchical
    structure of pipelines, stages, loops, and private operations.
    
    This is primarily a debugging and visualization tool to understand
    the structure before and after transformations like loop sinking
    and stage coarsening.
  }];
  let constructor = "::ktdf::createPrintPipelineTreePass()";
  let options = [
    Option<"showDependencies", "show-dependencies", "bool",
           /*default=*/"false",
           "Show stage dependencies (token-based synchronization)">
  ];
}

#endif // SCHEDULER_TRANSFORM_PASSES