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

def ConstructThreeStagePipelinePass : Pass<"construct-three-stage-pipeline", "mlir::ModuleOp"> {
  let summary = "Construct a three-stage pipeline";
  let description = [{
    This pass constructs a three-stage pipeline.
  }];
  let constructor = "::ktdf::createConstructThreeStagePipelinePass()";
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

def StageCoarseningPass : Pass<"stage-coarsening", "mlir::ModuleOp"> {
  let summary = "Perform stage coarsening transformation";
  let description = [{
    This pass implements a 4-stage stage coarsening algorithm to create
    hierarchical pipeline structures by sinking loops into stages.

    Following identification of stage groups (using loop distribution legality rules):
    - Stage 1a: Distribute and sink loops into pipeline
    - Stage 1b: Further sink into stages with single parent loop (expand buffers)
    - Stage 2: Discover and correct illegal structures iteratively
    - Stage 3: Correct stage dependencies that cross pipeline boundaries

    The transformation creates nested pipelines where:
    - Outer pipelines operate at tile granularity
    - Inner pipelines operate at hardware execution granularity

    This enables better hardware mapping and reduced synchronization overhead.
  }];
  let constructor = "::ktdf::createStageCoarseningPass()";
  let options = [
    Option<"loopNestDepth", "loop-nest-depth", "unsigned",
           /*default=*/"2",
           "Number of perfectly nested loops to consider for sinking">
  ];
}

def BroadcastPromotionPass : Pass<"broadcast-promotion", "mlir::ModuleOp"> {
  let summary = "Hoist invariant ktdf.data_transfer ops as sibling pipelines";
  let description = [{
    This pass identifies ktdf.data_transfer ops inside ktdf.stage bodies
    whose source and destination indices/sizes are invariant w.r.t. at
    least one enclosing scf.for induction variable, and hoists them
    outward as a sibling ktdf.pipeline placed before the invariant loop.

    Only broadcast-style hoists (single-transfer, hoist-up) are handled.
    Reduction hoists (paired pre-load/post-store) and FIFO-destination
    hoists (requiring co-hoisting of ktdf.fifo.allocate and consumer-side
    register-promotion rewrites) are handled by separate future passes.

    The pass runs a fixed-point loop: one hoist per iteration, re-scan
    after each hoist, exit when no candidate remains.
  }];
  let constructor = "::ktdf::createBroadcastPromotionPass()";
  let dependentDialects = ["mlir::ktdf::KTDFDialect", "mlir::scf::SCFDialect"];
}

#endif // SCHEDULER_TRANSFORM_PASSES