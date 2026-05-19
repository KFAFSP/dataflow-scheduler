#ifndef SCHEDULER_TRANSFORM_PASSES
#define SCHEDULER_TRANSFORM_PASSES

include "mlir/Pass/PassBase.td"

def PathExpansionPass : Pass<"path-expansion", "mlir::ModuleOp"> {
  let summary = "Expand data movement paths to match architecture graph";
  let description = [{
    This pass legalizes data movement in ktdf.pipeline operations by expanding
    paths to match the architecture graph. It inserts intermediate transfer
    stages and resources where needed to ensure all data movement follows legal
    hardware routing.
    
    Example: A direct HBM->SFP transfer is expanded to HBM->LX->SFP with
    intermediate LX buffers and stages.
  }];
  let constructor = "::ktdf::createPathExpansionPass()";
}

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

def ParallelizeLoopsAcrossInstancesPass
    : Pass<"parallelize-loops-across-instances", "mlir::ModuleOp"> {
  let summary = "Map parallel loops onto hardware instances via ktdf.parallel";
  let description = [{
    For each ktdf.pipeline whose work is corelet-local (no core-shared
    units in the union of its immediate stages' applicable_units), find
    the outermost parallel-annotated scf.for in the pipeline's enclosing
    scope whose statically-known trip count permits a split across
    `num_instances` (preferring divisible trips; falling back to imbalanced
    when none qualifies), and rewrite that scf.for into a ktdf.parallel op
    with `distribute(num_instances = N)`. `N` is sourced from
    SchedulerExtContext::getNumCorelets() at pass construction; the
    pass itself is target-neutral.

    Body legality (cross-iteration dependencies, LX sharing) is trusted
    via the loop_type=parallel_loop annotation set by upstream lowering.
    Reduction loops (loop_type=reduction_loop or unannotated) are not
    candidates.
  }];
  let constructor = "::ktdf::createParallelizeLoopsAcrossInstancesPass()";
  let dependentDialects = ["mlir::ktdf::KTDFDialect", "mlir::scf::SCFDialect"];
}

def DoubleBufferingPass
    : Pass<"double-buffering", "mlir::ModuleOp"> {
  let summary = "Double-buffer scratchpad memrefs handed off between sibling stages";
  let description = [{
    For each ktdf.pipeline whose enclosing scope contains at least one scf.for
    and which has no pre-existing modulo attribute, find scratchpad memrefs
    yielded by the pipeline's ktdf.private region that are written by exactly
    one sibling stage and read by one or more sibling stages (via
    ktdf.data_transfer ops). For each such buffer, allocate two physical copies
    just outside the pipeline scope, insert one shared ktdf.buffer_phase and
    per-buffer ktdf.select_memref before the pipeline, rewrite the pipeline's
    references to use the selected memrefs, and set modulo(size: 2) on the
    pipeline.

    Memory-space eligibility is delegated to
    SchedulerExtContext::isEligibleForDoubleBuffering, which by default returns
    true only for LX. Targets can override.

    The pass leaves ktdf.private's signature consistent: dropped result slots
    are removed from the op and from ktdf.private_yield, and orphaned
    memref.alloc ops in the private region are erased.
  }];
  let constructor = "::ktdf::createDoubleBufferingPass()";
  let dependentDialects = [
      "mlir::ktdf::KTDFDialect", "mlir::scf::SCFDialect",
      "mlir::memref::MemRefDialect"];
}

#endif // SCHEDULER_TRANSFORM_PASSES
