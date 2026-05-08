# Representing Corelet Parallelism in KTDF

## 1. Motivation

The Spyre DD2 target places two programmable **corelets** inside every core. Each corelet has its own LX load unit (LXLU), store unit (LXSU), and compute engines (SFP, PE). The two corelets of a core share the same LX scratchpad, but execute independently — corelet 0 and corelet 1 can be running their own compute, their own FIFO transfers, and their own loop iterations at the same instant.

Leaving a corelet idle wastes roughly half of each core's achievable throughput. Any kernel that reaches the coarsened-compute region of a KTDF pipeline and runs on only one corelet is paying a 2× cost against the hardware that's physically present. Closing that gap — making the coarsened compute stage run simultaneously on both corelets — is the specific problem this document addresses.

Representing this in the IR is not merely a code-generation concern. The decision of how work is split across corelets changes what the IR *means*: which iterations are assigned where, how the enclosing pipeline's tokens synchronize them, and whether LX accesses are structured to avoid cross-corelet conflicts. The representation has to be explicit enough that the scheduler can reason about it, and generic enough that KTDF — a target-neutral dialect — does not end up with "corelet" baked into its vocabulary.

## 2. Two Choices: Loop Parallelism vs. Dataflow-Style Split

There are two broadly different ways to use the two corelets concurrently.

### 2.1 Mapping loop parallelism to corelets

The kernel exposes parallel loops (for example, independent tile iterations), and those parallel iterations are distributed across corelets. Both corelets execute the *same program text*; they differ only in which iterations they own. This is a classic SPMD-style split, specialized to the two-corelet case, and it applies directly to every kernel in which the compute-trio stage sits inside a parallel loop nest.

### 2.2 Dataflow-style split across corelets

Rather than splitting iterations, the two corelets take on *different roles* in a dataflow pipeline — for example, corelet 0 runs the SFP-based portion of a computation while corelet 1 runs a PE-based accumulation portion, with data moving between them through shared LX. This is a fundamentally different shape: the two corelets execute different stage bodies and coordinate through LX rather than through iteration-range partitioning.

**This document sets the dataflow-style option aside.** It is a legitimate and potentially valuable mode, but the representation it needs (cross-corelet tokens, per-role stage binding, LX-mediated handoff) is separable from what we design now. The current proposal addresses only the loop-parallelism case. The construct introduced below does not preclude a future companion construct for the dataflow-style case; it simply does not attempt to cover both at once.

## 3. Proposal: the `ktdf.parallel` Construct

We introduce a new KTDF operation, `ktdf.parallel`, that expresses a multi-dimensional parallel iteration space together with a pure block distribution of those iterations across a fixed number of hardware instances. The op is deliberately target-agnostic: it declares that iterations are replicated across some number of instances, but does not say whether those instances are corelets, cores, or anything else. Lowering to DFIR binds the instance axis to the concrete Spyre corelet axis.

The construct has one attribute that carries the scheduling decision:

- **`num_instances`** — an `i64` attribute giving the number of hardware instances the iteration space should be distributed across. For Spyre corelet parallelism this is always `2`.

Distribution is always pure block. The lexicographic enumeration of the iteration space is divided into `num_instances` contiguous chunks; instance `i` owns chunk `i`. With `T` total iterations and `N` instances, instance `i` runs `ceil(T / N)` iterations for `i < N-1` and the remaining `T - (N-1) · ceil(T / N)` iterations for `i = N-1`. The body runs once per owned iteration, in lex order, with an implicit barrier at the end of the op.

The choice to support only pure block distribution is deliberate. Lowering a pure-block distribution to per-instance loop bounds requires two arithmetic operations per distributed dimension — a multiply for the lower bound and a min-clamped add for the upper bound. Cyclic and block-cyclic distributions would require either strided iteration in the generated loops or an explicit outer-block/inner-element loop pair, both of which add complexity without meeting a concrete need for the near-term target. For a 2-instance split (the Spyre corelet case) the imbalance from pure block is never more than one iteration regardless of which distribution policy is chosen, so the policy knob would not affect balance in practice.

The op's iteration space and the `num_instances` attribute together specify the distribution completely. The body does not need to compute a corelet id or perform arithmetic to recover its own iteration indices — every iteration's induction variables are bound directly by the op, exactly as with `scf.parallel`.

## 4. 1-D Case: One Parallel Loop

Consider the inner parallel loop from the coarsened compute stage in `test/Transforms/StageCoarsening/basic-out.mlir`. The `n` dimension has 64 iterations, each iteration running the LX→SFP→LX compute trio over one 1×64 tile of data. With two corelets available, we would like each corelet to run 32 of those iterations in parallel.

Before replication, the stage body has a plain `parallel_loop`-annotated `scf.for`:

```mlir
ktdf.stage depends_in(%res_t1) depends_out(%res_t5) {
  scf.for %n = %c0 to %c64 step %c1 {
    ktdf.pipeline { /* SFP trio */ }
  } {loop_type = #ktdf.loop_type<parallel_loop>}
}
```

After applying the `ktdf.parallel` construct with `num_instances = 2`:

```mlir
ktdf.stage depends_in(%res_t1) depends_out(%res_t5) {
  ktdf.parallel (%n) = (%c0) to (%c64) step (%c1)
                       distribute(num_instances = 2) {
    ktdf.pipeline { /* SFP trio */ }
  }
}
```

The 64 iterations are split into two contiguous chunks of 32. Corelet 0 runs `n ∈ [0, 32)`, corelet 1 runs `n ∈ [32, 64)`. The loop body is unchanged; only the enclosing construct is different.

The `num_instances` attribute carries the entire scheduling decision. The fact that `64 / 2 = 32` divides evenly means there is no imbalance in this case — both corelets run exactly 32 iterations.

### Lowering Approach

The lowering uses `uniform.def_immutable_mapping` and `uniform.query_map` to parameterize a single `dataflow.program_unit` body by corelet unit id. The loop is normalized to start at zero and run over a per-corelet trip count; the per-corelet lower bound of the original `ktdf.parallel` range is folded once into the start address handed to `dataflow.get_logical_memory` / `construct_memory_view`, so each corelet's logical memory view is already offset to its own region of the buffer. The normalized induction variable indexes directly into the per-corelet view, satisfying the `agen` constraint that subscripts be iterators or constants without introducing any `arith` operations in the body.

A structural alternative — one region per corelet under `uniform.uniformize_regions` with each region carrying its own constant-bounds loop header — would require the same normalize-and-absorb transformation inside each region to honor the `agen` subscript constraint, so it does not offer a distinct lowering path.

For the 1-D case above (`num_instances = 2`, chunks `[0, 32)` and `[32, 64)`), it is useful to show the IR in two stages: first with the per-corelet divergence expressed as the loop's lower and upper bounds, and then after that divergence is absorbed into the memory-view start address.

**Before absorbing the per-corelet offset.** In the pre-absorption form, the `scf.for` ranges over the raw per-corelet `[lb, ub)` retrieved from two uniform maps, and the induction variable is used directly as the `agen` subscript into a shared logical memory view:

```mlir
dataflow.program_unit %id : {
  %c0  = arith.constant 0  : index
  %c1  = arith.constant 1  : index
  %c32 = arith.constant 32 : index
  %c64 = arith.constant 64 : index

  // Corelet 0 runs [0, 32), corelet 1 runs [32, 64).
  %lb_map = uniform.def_immutable_mapping
              keys(%corelet0, %corelet1)
              values(%c0, %c32)
            : index
  %ub_map = uniform.def_immutable_mapping
              keys(%corelet0, %corelet1)
              values(%c32, %c64)
            : index

  %lb = uniform.query_map %lb_map[%id] : index
  %ub = uniform.query_map %ub_map[%id] : index

  // Shared logical memory view — both corelets see the same buffer.
  %A = dataflow.get_logical_memory %A_base
     : memref<64x...>

  scf.for %n = %lb to %ub step %c1 {
    // agen.vector_load %A[%n, ..]
  }
}
```

The `scf.for` is written exactly once; its per-corelet bounds come from the uniform maps, and `%n` is used directly as the `agen` subscript. The body itself carries no per-corelet arithmetic. Both corelets see the same logical memory view; the per-corelet divergence lives entirely in the loop header.

**After absorbing the per-corelet offset.** The absorbing step replaces the `[lb, ub)` loop with a unit-step loop `[0, trip)` and folds the per-corelet lower bound into the start address handed to `dataflow.get_logical_memory`. Each corelet's logical memory view now covers only its own region of the buffer, and the normalized induction variable indexes directly into that per-corelet view:

```mlir
dataflow.program_unit %id : {
  %c0  = arith.constant 0  : index
  %c1  = arith.constant 1  : index
  %c32 = arith.constant 32 : index

  // Per-corelet chunk length: both corelets get 32 iterations here.
  %trip_map = uniform.def_immutable_mapping
                keys(%corelet0, %corelet1)
                values(%c32, %c32)
              : index

  // Per-corelet start offset into the buffer:
  // corelet 0 starts at 0, corelet 1 starts at 32 (computed from the layout).
  %offset_map = uniform.def_immutable_mapping
                  keys(%corelet0, %corelet1)
                  values(%c0, %c32) <-- should be based on layout.
                : index

  %trip    = uniform.query_map %trip_map[%id] : index
  %offset  = uniform.query_map %offset_map[%id] : index

  // %offset shifts each corelet's view to start at its own region. The body
  // uses %n_norm directly as the agen subscript into this per-corelet view.
  %A_start = arith.addi %A_base, %offset : index
  %A = dataflow.get_logical_memory %A_start
     : memref<32x...>

  scf.for %n_norm = %c0 to %trip step %c1 {
    // agen.vector_load %A[%n_norm, ..]
  }
}
```

The key observation is that **only the map values encode the corelet split**. The `scf.for` is written exactly once and iterates `%n_norm` over `[0, %trip)` — a unit-step loop whose induction variable is used directly as an `agen` subscript. When the program runs on corelet 0, the id `%id` matches `%corelet0`, the queries return `32` and `0`, the logical memory view starts at the base, and the loop iterates over that corelet's 32 elements. When the same program runs on corelet 1, the id matches `%corelet1`, the queries return `32` and `32`, the view starts 32 positions in, and the loop iterates over the other half. The body of the loop is identical in both cases; nothing about the compute sequence branches on the unit id, and no `arith` operations appear inside the loop — the `agen` constraint that subscripts be iterators or constants is preserved.


## 5. Imbalance from a Leftover Block

The 1-D case becomes interesting when the total iteration count does not divide evenly by `num_instances`. Consider a parallel loop of trip count 7 distributed across 2 corelets:

```mlir
ktdf.parallel (%q) = (%c0) to (%c7) step (%c1)
                     distribute(num_instances = 2) {
  ktdf.pipeline { /* body uses %q */ }
}
```

The seven iterations are laid out in lex order as `k = 0, 1, 2, 3, 4, 5, 6`. With pure block distribution the chunk size is `ceil(7 / 2) = 4`, so the first four iterations (`0..3`) go to corelet 0 and the remaining three (`4..6`) go to corelet 1. Corelet 0 runs 4 iterations and corelet 1 runs only 3. The last iteration's worth of work is missing from corelet 1 — the *leftover block* is incomplete.

This imbalance is inherent to the geometry of 7 / 2. It costs roughly `1/7 ≈ 14%` of the available corelet throughput for this particular stage. The construct makes the imbalance visible in the IR without requiring the kernel author or the scheduler to introduce any arithmetic to model it — the leftover iterations simply occupy the tail of the last instance's chunk.

Smaller trip counts produce proportionally worse imbalance. A parallel loop of trip count 3 distributed across 2 corelets gives corelet 0 two iterations and corelet 1 one, and half of corelet 1's capacity for that stage is idle. A trip count of 5 splits 3+2 with 20% imbalance. Whether this is acceptable is a policy decision for the pass that inserts `ktdf.parallel` — but the cost is plainly readable from the op's `num_instances` attribute and the iteration-space trip count.

## 6. Multi-Dimensional Iterators: Reducing Imbalance

When a kernel exposes more than one parallel loop in the nest, the 1-D case above is strictly worse than it needs to be. Splitting only the outermost parallel dimension ignores the iterations available in the inner parallel dimensions, which could have filled out the leftover block.

Consider two nested parallel loops of trip counts 5 and 7, both contained inside a coarsened compute stage. Distributing across 2 corelets by splitting only the outer dimension gives the size-5 loop a chunk of `ceil(5 / 2) = 3` to corelet 0 and 2 to corelet 1, so corelet 0 runs `3 × 7 = 21` iterations and corelet 1 runs `2 × 7 = 14`. The imbalance is `7 / 35 ≈ 20%` — substantial, and it comes entirely from refusing to reach into the inner dimension.

The multi-dimensional form of `ktdf.parallel` avoids this by declaring the whole Cartesian product as the iteration space, and letting the pure-block split cut across that space in lex order:

```mlir
ktdf.parallel (%p, %q) = (%c0, %c0) to (%c5, %c7) step (%c1, %c1)
                         distribute(num_instances = 2) {
  ktdf.pipeline { /* body uses %p and %q */ }
}
```

The lex enumeration of the 5×7 space produces 35 iterations: `(0,0), (0,1), …, (0,6), (1,0), (1,1), …, (4,6)`. With `num_instances = 2` the chunk size is `ceil(35 / 2) = 18`, so the first 18 iterations go to corelet 0 and the remaining 17 go to corelet 1. The imbalance is now `1 / 35 ≈ 3%` — roughly a seventh of the single-dimension cost, and essentially as balanced as the 35-iteration space permits for two instances.

The induction variables `%p` and `%q` are bound by the op exactly as they would be by two nested `scf.for` loops, so the body reads naturally: every iteration sees its `(p, q)` pair in scope. No explicit arithmetic is needed to recover iteration coordinates from a flattened index — the op handles the enumeration. This keeps the IR shape close to the original loop nest while expressing the balanced distribution that a single-dimensional split could not achieve.

In general, including more parallel dimensions in the `ktdf.parallel` iteration space increases the total iteration count the distribution acts over, which in turn makes the pure-block leftover smaller in relative terms. With `T` total iterations and `N` instances, the smallest achievable imbalance is at most `1 / T` per instance under pure block. Exposing more parallel dimensions to the op is the only mechanism for increasing `T`, and therefore the only mechanism for improving balance beyond what any single dimension can provide.

## 7. End-to-End Example

This section shows a complete coarsened-pipeline IR before and after the `ktdf.parallel` construct is introduced, using the same kernel structure as `test/Transforms/StageCoarsening/basic-out.mlir` (the 5-stage matrix-add after stage coarsening) but with a larger tile-row count so the parallel split is meaningful. The kernel processes a 32×64 tile of `C = A + B` on one core's slice, with the `m` loop running 32 iterations over 1×64 tiles and the `n` loop running a single iteration (so the parallel dimension of interest is `m`, and both loops use unit step).

### 7.1 Before: stage-coarsened IR without `ktdf.parallel`

The outer pipeline has three stages: Stage 0 loads tiles from HBM to LX, Stage 5 is the coarsened compute stage (wrapping a nested pipeline of LXLU→SFP→LXSU), and Stage 4 stores tiles from LX to HBM. Each of the three stages carries its own parallel loop nest over `(%m, %n)`.

```mlir
module {
  func.func @local_schedule_1(%i: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %c64 = arith.constant 64 : index
    // ... memory views and reinterpret_casts as in basic-out.mlir ...

    // Outer pipeline — tile granularity
    ktdf.pipeline {
      %res_lx_A, %res_lx_B, %res_lx_C,
      %res_t1, %res_t5, %res_t6 = ktdf.private -> (
        memref<64xf16, #ktdf.memory_space<LX>>,
        memref<64xf16, #ktdf.memory_space<LX>>,
        memref<64xf16, #ktdf.memory_space<LX>>,
        !ktdf.token, !ktdf.token, !ktdf.token
      ) {
        %lx_A = memref.alloc() : memref<64xf16, #ktdf.memory_space<LX>>
        %lx_B = memref.alloc() : memref<64xf16, #ktdf.memory_space<LX>>
        %lx_C = memref.alloc() : memref<64xf16, #ktdf.memory_space<LX>>
        %t1 = ktdf.create_token : !ktdf.token
        %t5 = ktdf.create_token : !ktdf.token
        %t6 = ktdf.create_token : !ktdf.token
        ktdf.private_yield %lx_A, %lx_B, %lx_C, %t1, %t5, %t6
          : memref<64xf16, #ktdf.memory_space<LX>>,
            memref<64xf16, #ktdf.memory_space<LX>>,
            memref<64xf16, #ktdf.memory_space<LX>>,
            !ktdf.token, !ktdf.token, !ktdf.token
      }

      // Stage 0 — HBM -> LX (parallel loops sunk into stage)
      ktdf.stage depends_in(none) depends_out(%res_t1 : !ktdf.token) {
        scf.for %m = %c0 to %c32 step %c1 {
          scf.for %n = %c0 to %c1 step %c1 {
            ktdf.data_transfer from %A[%m, %n] size [1, 64]
                               to %res_lx_A[%c0] size [64]
              : memref<32x64xf16, ..., #ktdf.memory_space<HBM>>,
                memref<64xf16,   #ktdf.memory_space<LX>>
            ktdf.data_transfer from %B[%m, %n] size [1, 64]
                               to %res_lx_B[%c0] size [64]
              : memref<32x64xf16, ..., #ktdf.memory_space<HBM>>,
                memref<64xf16,   #ktdf.memory_space<LX>>
          } {loop_type = #ktdf.loop_type<parallel_loop>}
        } {loop_type = #ktdf.loop_type<parallel_loop>}
      }

      // Stage 5 — coarsened compute stage
      ktdf.stage depends_in(%res_t1 : !ktdf.token) depends_out(%res_t5 : !ktdf.token) {
        scf.for %m = %c0 to %c32 step %c1 {
          scf.for %n = %c0 to %c1 step %c1 {
            // Nested pipeline — hardware execution granularity
            ktdf.pipeline {
              %res_sfp_A, %res_sfp_B, %res_sfp_out,
              %res_t2, %res_t3, %res_t4 = ktdf.private -> (
                memref<64xf16, #ktdf.memory_space<SFP_REG>>,
                memref<64xf16, #ktdf.memory_space<SFP_REG>>,
                memref<64xf16, #ktdf.memory_space<SFP_REG>>,
                !ktdf.token, !ktdf.token, !ktdf.token
              ) {
                %sfp_A   = memref.alloc() : memref<64xf16, #ktdf.memory_space<SFP_REG>>
                %sfp_B   = memref.alloc() : memref<64xf16, #ktdf.memory_space<SFP_REG>>
                %sfp_out = memref.alloc() : memref<64xf16, #ktdf.memory_space<SFP_REG>>
                %t2 = ktdf.create_token : !ktdf.token
                %t3 = ktdf.create_token : !ktdf.token
                %t4 = ktdf.create_token : !ktdf.token
                ktdf.private_yield %sfp_A, %sfp_B, %sfp_out, %t2, %t3, %t4
                  : memref<64xf16, #ktdf.memory_space<SFP_REG>>,
                    memref<64xf16, #ktdf.memory_space<SFP_REG>>,
                    memref<64xf16, #ktdf.memory_space<SFP_REG>>,
                    !ktdf.token, !ktdf.token, !ktdf.token
              }

              // Stage 1 — LX -> SFP
              ktdf.stage depends_in(none) depends_out(%res_t2 : !ktdf.token) {
                ktdf.data_transfer from %res_lx_A[%c0] size [64]
                                   to %res_sfp_A[%c0] size [64]
                  : memref<64xf16, #ktdf.memory_space<LX>>,
                    memref<64xf16, #ktdf.memory_space<SFP_REG>>
                ktdf.data_transfer from %res_lx_B[%c0] size [64]
                                   to %res_sfp_B[%c0] size [64]
                  : memref<64xf16, #ktdf.memory_space<LX>>,
                    memref<64xf16, #ktdf.memory_space<SFP_REG>>
              }

              // Stage 2 — SFP compute
              ktdf.stage depends_in(%res_t2 : !ktdf.token) depends_out(%res_t3 : !ktdf.token) {
                linalg.add
                  ins(%res_sfp_A, %res_sfp_B
                      : memref<64xf16, #ktdf.memory_space<SFP_REG>>,
                        memref<64xf16, #ktdf.memory_space<SFP_REG>>)
                  outs(%res_sfp_out
                      : memref<64xf16, #ktdf.memory_space<SFP_REG>>)
              }

              // Stage 3 — SFP -> LX
              ktdf.stage depends_in(%res_t3 : !ktdf.token) depends_out(none) {
                ktdf.data_transfer from %res_sfp_out size [64]
                                   to %res_lx_C[%c0] size [64]
                  : memref<64xf16, #ktdf.memory_space<SFP_REG>>,
                    memref<64xf16, #ktdf.memory_space<LX>>
              }
            } // nested ktdf.pipeline
          } {loop_type = #ktdf.loop_type<parallel_loop>}
        } {loop_type = #ktdf.loop_type<parallel_loop>}
      }

      // Stage 4 — LX -> HBM (parallel loops sunk into stage)
      ktdf.stage depends_in(%res_t5 : !ktdf.token) depends_out(none) {
        scf.for %m = %c0 to %c32 step %c1 {
          scf.for %n = %c0 to %c1 step %c1 {
            ktdf.data_transfer from %res_lx_C[%c0] size [64]
                               to %C[%m, %n] size [1, 64]
              : memref<64xf16, #ktdf.memory_space<LX>>,
                memref<32x64xf16, ..., #ktdf.memory_space<HBM>>
          } {loop_type = #ktdf.loop_type<parallel_loop>}
        } {loop_type = #ktdf.loop_type<parallel_loop>}
      }
    } // outer ktdf.pipeline
    return
  }
}
```

Each of Stage 0, Stage 5, and Stage 4 carries its own `(m, n)` parallel loop nest. Stage 5's nest wraps the nested pipeline that runs Stages 1–3. Every one of these nests is a candidate for corelet parallelization, but the IR as written says nothing about how iterations are distributed — it would execute sequentially on a single corelet as written.

### 7.2 After: `ktdf.parallel` applied to Stage 5 only

The parallelization pass applies `ktdf.parallel` selectively. Stage 5 — the coarsened compute stage — is the one stage whose work naturally replicates across corelets: each corelet has its own LXLU, SFP, and LXSU, and its own SFP_REG file, so splitting the compute iteration space between the two corelets directly exercises hardware that would otherwise sit idle. Stage 0 and Stage 4 are different in character. They drive HBM-facing memory engines (conceptually one set shared by the two corelets of a core) and they produce or consume LX buffers that Stage 5 reads or writes as a whole. Replicating Stage 0 or Stage 4 across corelets would either duplicate HBM traffic or partition the LX staging in ways the outer pipeline's token flow isn't set up to coordinate. For this kernel we want those two stages to run once per tile iteration on a single-instance schedule, and only Stage 5 to fan out to both corelets.

So the pass leaves Stage 0 and Stage 4 as they were — plain parallel loop nests with `loop_type = parallel_loop` — and rewrites only Stage 5's nest as a `ktdf.parallel`. With `m ∈ [0, 32)` and `n ∈ [0, 1)` as the parallel dimensions inside Stage 5, the total iteration space is 32, evenly divisible by 2, so pure block distribution gives corelet 0 iterations `m ∈ [0, 16)` and corelet 1 iterations `m ∈ [16, 32)`.

```mlir
module {
  func.func @local_schedule_1(%i: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %c64 = arith.constant 64 : index
    // ... memory views and reinterpret_casts as before ...

    ktdf.pipeline {
      %res_lx_A, %res_lx_B, %res_lx_C,
      %res_t1, %res_t5, %res_t6 = ktdf.private -> ( ... ) { ... }

      // Stage 0 — HBM -> LX (unchanged; single-instance schedule)
      ktdf.stage depends_in(none) depends_out(%res_t1 : !ktdf.token) {
        scf.for %m = %c0 to %c32 step %c1 {
          scf.for %n = %c0 to %c1 step %c1 {
            ktdf.data_transfer from %A[%m, %n] size [1, 64]
                               to %res_lx_A[%c0] size [64]
              : memref<32x64xf16, ..., #ktdf.memory_space<HBM>>,
                memref<64xf16,   #ktdf.memory_space<LX>>
            ktdf.data_transfer from %B[%m, %n] size [1, 64]
                               to %res_lx_B[%c0] size [64]
              : memref<32x64xf16, ..., #ktdf.memory_space<HBM>>,
                memref<64xf16,   #ktdf.memory_space<LX>>
          } {loop_type = #ktdf.loop_type<parallel_loop>}
        } {loop_type = #ktdf.loop_type<parallel_loop>}
      }

      // Stage 5 — coarsened compute, distributed across 2 corelets
      ktdf.stage depends_in(%res_t1 : !ktdf.token) depends_out(%res_t5 : !ktdf.token) {
        ktdf.parallel (%m, %n) = (%c0, %c0) to (%c32, %c1) step (%c1, %c1)
                                 distribute(num_instances = 2) {
          ktdf.pipeline {
            %res_sfp_A, %res_sfp_B, %res_sfp_out,
            %res_t2, %res_t3, %res_t4 = ktdf.private -> ( ... ) { ... }

            // Stage 1 — LX -> SFP
            ktdf.stage depends_in(none) depends_out(%res_t2 : !ktdf.token) {
              ktdf.data_transfer from %res_lx_A[%c0] size [64]
                                 to %res_sfp_A[%c0] size [64]
                : memref<64xf16, #ktdf.memory_space<LX>>,
                  memref<64xf16, #ktdf.memory_space<SFP_REG>>
              ktdf.data_transfer from %res_lx_B[%c0] size [64]
                                 to %res_sfp_B[%c0] size [64]
                : memref<64xf16, #ktdf.memory_space<LX>>,
                  memref<64xf16, #ktdf.memory_space<SFP_REG>>
            }

            // Stage 2 — SFP compute
            ktdf.stage depends_in(%res_t2 : !ktdf.token) depends_out(%res_t3 : !ktdf.token) {
              linalg.add
                ins(%res_sfp_A, %res_sfp_B
                    : memref<64xf16, #ktdf.memory_space<SFP_REG>>,
                      memref<64xf16, #ktdf.memory_space<SFP_REG>>)
                outs(%res_sfp_out
                    : memref<64xf16, #ktdf.memory_space<SFP_REG>>)
            }

            // Stage 3 — SFP -> LX
            ktdf.stage depends_in(%res_t3 : !ktdf.token) depends_out(none) {
              ktdf.data_transfer from %res_sfp_out size [64]
                                 to %res_lx_C[%c0] size [64]
                : memref<64xf16, #ktdf.memory_space<SFP_REG>>,
                  memref<64xf16, #ktdf.memory_space<LX>>
            }
          } // nested ktdf.pipeline
        }
      }

      // Stage 4 — LX -> HBM (unchanged; single-instance schedule)
      ktdf.stage depends_in(%res_t5 : !ktdf.token) depends_out(none) {
        scf.for %m = %c0 to %c32 step %c1 {
          scf.for %n = %c0 to %c1 step %c1 {
            ktdf.data_transfer from %res_lx_C[%c0] size [64]
                               to %C[%m, %n] size [1, 64]
              : memref<64xf16, #ktdf.memory_space<LX>>,
                memref<32x64xf16, ..., #ktdf.memory_space<HBM>>
          } {loop_type = #ktdf.loop_type<parallel_loop>}
        } {loop_type = #ktdf.loop_type<parallel_loop>}
      }
    } // outer ktdf.pipeline
    return
  }
}
```

Three things to notice. First, only Stage 5's body changed. Stage 0 and Stage 4 retain their nested `scf.for` nests with the `parallel_loop` attributes they had coming out of stage coarsening; those attributes continue to carry "these iterations are reorderable" information without committing to any cross-corelet distribution. Second, the outer pipeline's token flow — `%res_t1` produced by Stage 0, consumed by Stage 5; `%res_t5` produced by Stage 5, consumed by Stage 4 — is unchanged. Both corelets' work in Stage 5 begins only after Stage 0's `%res_t1` is produced, and Stage 4 begins only after both corelets' Stage 5 work has joined at the `ktdf.parallel` op's implicit barrier and produced `%res_t5`. The pipeline-level synchronization is preserved exactly. Third, the nested pipeline inside Stage 5 — the LXLU→SFP→LXSU trio with its SFP_REG private region — is untouched; corelet replication happens at the `m` loop level, so each corelet enters the nested pipeline with its own SFP_REG allocations and its own iteration range, which is what the per-corelet compute units physically require.

The shared LX buffers (`%res_lx_A`, `%res_lx_B`, `%res_lx_C`) are visible to both corelets. Because Stage 0 runs once and writes all 32 tile rows before either corelet's Stage 5 begins, and because Stage 4 runs once after both corelets' Stage 5 work completes, the LX traffic does not race at the op level — the outer pipeline's tokens make the producer/consumer ordering explicit and the single-instance Stage 0 and Stage 4 serialize the HBM-facing transfers. Formalizing the LX sharing discipline downstream (for example, the `dataflow.sync_send`/`sync_recv` pairs that materialize at the DFIR layer when two corelets touch the same LX region) is a concern for lowering, but the KTDF IR here is enough to express the intended per-corelet iteration ownership cleanly.

## 8. Placement in the Scheduler Pipeline

`ktdf.parallel` is introduced by a dedicated pass that runs after stage-coarsening and before lowering to DFIR. The pass walks each stage's body, collects the parallel loops within it, chooses how many of those loops to include in the iteration space (based on imbalance policy and the fixed `num_instances` supplied as a pass option), and rewrites the loop nest as a single `ktdf.parallel` with the chosen attribute. Inner sequential (non-parallel) loops remain inside the body unchanged.

Because `num_instances` equals the hardware instance count by construction, the IR that emerges from this pass is directly lowering-ready: DFIR lowering binds the op's distribution axis to corelets and emits one program per corelet, each enumerating only its assigned subset of iterations. No further scheduling decisions remain between `ktdf.parallel` and corelet-bound DFIR.

## 9. Summary

`ktdf.parallel` expresses corelet parallelism in KTDF without introducing corelet-specific vocabulary into the dialect. Its single attribute — `num_instances` — specifies a pure block distribution of a multi-dimensional iteration space across a fixed set of hardware instances. The 1-D form handles the common case of a single parallel loop cleanly when the trip count divides evenly, and makes leftover-block imbalance plainly visible when it does not. The multi-dimensional form folds additional parallel dimensions into the distributed space, converting coarse-grained single-dimension imbalance into fine-grained whole-space imbalance that is typically an order of magnitude smaller. Neither form requires the body to perform arithmetic on a flattened index or a corelet id — the op binds induction variables directly, keeping the body identical in shape to the loop nest it replaces.

The dataflow-style corelet split — where the two corelets take on different roles rather than different iteration ranges — is intentionally outside this proposal's scope and is a candidate for a separate future discussion.

## 10. Non-Constant Bounds and DFIR-Level Loop Splitting

When the `ktdf.parallel` bounds are not compile-time constants — for example, when the upper bound depends on a surrounding loop's iteration variable — the per-corelet chunks are no longer integer constants, and the lowered `uniform.def_immutable_mapping` values become SSA results of `arith` operations.

Consider the following input:

```
for j = 0 to 10 {
  x = (j < 9) ? 100 : 50
  ktdf.parallel (%i) = 0 to x step 1 distribute(num_instances = 2) {
    S1..
  }
}
```

The `i`-loop's upper bound `x` is a function of `j`: 100 for `j < 9`, 50 otherwise. The per-corelet chunk size therefore also varies with `j`, and a direct lowering produces `uniform.def_immutable_mapping` values computed with `arith.divui`, `arith.select`, and friends:

```mlir
dataflow.program_unit %id : {
  scf.for %j = %c0 to %c10 step %c1 {
    %j_lt_9 = arith.cmpi slt, %j, %c9 : index
    %x = arith.select %j_lt_9, %c100, %c50 : index

    // chunk = ceil(x / 2)
    %x_plus_1 = arith.addi %x, %c1 : index
    %chunk = arith.divui %x_plus_1, %c2 : index

    %lb_map = uniform.def_immutable_mapping
                keys(%corelet0, %corelet1)
                values(%c0, %chunk)
              : index
    %ub_map = uniform.def_immutable_mapping
                keys(%corelet0, %corelet1)
                values(%chunk, %x)
              : index

    %lb = uniform.query_map %lb_map[%id] : index
    %ub = uniform.query_map %ub_map[%id] : index

    scf.for %i = %lb to %ub step %c1 {
      // S1..
    }
  }
}
```

The map values `%chunk` and `%x` are SSA results, not constants, so this form cannot lower further to Spyre DFIR which requires `uniform.def_immutable_mapping` values to be constants.

The legalization happens at the DFIR level via a loop-splitting pass that splits the parent loop into iteration ranges over which the condition-dependent values become invariant — so the map values fold to integer constants.

## 11. Lowering Multi-Dimensional `ktdf.parallel`

The multi-dimensional form of `ktdf.parallel` from Section 6 distributes iterations in lex order across the whole Cartesian product. Lowering has to traverse that Cartesian product while selecting only the iterations owned by the current corelet. The strategy is to carry a collapsed lex-order counter as a loop-carried value across the nested `scf.for`s and guard the body with a membership check against the corelet's assigned `[lb, ub)` range.

Using the 5×7 example from Section 6:

```mlir
ktdf.parallel (%p, %q) = (%c0, %c0) to (%c5, %c7) step (%c1, %c1)
                         distribute(num_instances = 2) {
  ktdf.pipeline { /* body uses %p and %q */ }
}
```

with corelet 0 owning lex positions `[0, 18)` and corelet 1 owning `[18, 35)`, lowers to:

```mlir
dataflow.program_unit %id : {
  %c0   = arith.constant  0 : index
  %c1   = arith.constant  1 : index
  %c5   = arith.constant  5 : index
  %c7   = arith.constant  7 : index
  %c18  = arith.constant 18 : index
  %c35  = arith.constant 35 : index

  // Corelet 0 owns [0, 18), corelet 1 owns [18, 35).
  %lb_map = uniform.def_immutable_mapping
              keys(%corelet0, %corelet1)
              values(%c0, %c18)
            : index
  %ub_map = uniform.def_immutable_mapping
              keys(%corelet0, %corelet1)
              values(%c18, %c35)
            : index

  %lb = uniform.query_map %lb_map[%id] : index
  %ub = uniform.query_map %ub_map[%id] : index

  // %k is the collapsed lex-order counter across the (p, q) space.
  // It starts at 0, is carried through both loops, and is incremented
  // once per innermost iteration. The p-loop yields the final counter
  // back to continue across outer iterations.
  %k_final = scf.for %p = %c0 to %c5 step %c1
                     iter_args(%k_p = %c0) -> (index) {
    %k_next = scf.for %q = %c0 to %c7 step %c1
                      iter_args(%k_q = %k_p) -> (index) {
      // Membership check: is this (p, q) in this corelet's range?
      %ge_lb = arith.cmpi uge, %k_q, %lb : index
      %lt_ub = arith.cmpi ult, %k_q, %ub : index
      %in_range = arith.andi %ge_lb, %lt_ub : i1

      scf.if %in_range {
        // Body — executes only when %k_q ∈ [%lb, %ub).
        // Uses %p and %q as bound by the enclosing scf.fors.
        ktdf.pipeline { /* body uses %p and %q */ }
      }

      // Increment the collapsed counter for the next innermost iteration.
      %k_q_next = arith.addi %k_q, %c1 : index
      scf.yield %k_q_next : index
    }
    scf.yield %k_next : index
  }
}
```

The `%k` counter threads through both `scf.for`s via `iter_args` / `scf.yield`, incrementing by 1 per innermost iteration. At any innermost step, `%k` equals `p·7 + q` — the lex-order index into the 5×7 space. The `scf.if` guard consults `%k` against the corelet's `[%lb, %ub)` to decide whether to execute the body.

Both corelets traverse the entire 5×7 lex space, but each executes the body only for its assigned 18 or 17 iterations. On corelet 0 the guard fires for `(p, q)` pairs `(0,0)..(2,3)`; on corelet 1 it fires for `(2,4)..(4,6)`. The iterations outside each corelet's range still incur loop-overhead cost (counter increment, compare, branch) but no body-level work. For the KTDF use case where the body is a `ktdf.pipeline` containing compute and data movement, the overhead of the skipped iterations is negligible compared to the work done in the kept iterations.

The loop nest remains in its natural form: `%p` and `%q` are bound the way they would be by plain `scf.for`s, and the body reads identically to the un-parallelized version. No div/rem recovery is needed, and no flattening of the loop structure is performed.

### Collapsibility and the `uniform.uniformize_regions` path

Before emitting the `%k`-counter form above, the code generator first checks whether the loop nest can be collapsed into a single `scf.for` over the lex-order trip count. Collapse is legal when the bounds are rectangular constants and the body's addressing can be rewritten in terms of the collapsed index. When that check succeeds, the multi-dimensional lowering reduces to the 1-D case from Section 4 — a single `scf.for %k = %lb to %ub step %c1` without a membership guard — and the `uniform.uniformize_regions` operation becomes the natural structural home for the per-corelet program bodies, because each region's body is a textually identical `scf.for` whose bounds are retrieved from the same maps used in the 1-D case.

When collapse is not legal as-is but a small amount of loop splitting would make it legal (for example, splitting an outer loop into iteration ranges over which inner bounds become rectangular), the code generator applies that splitting before falling back to the `%k`-counter form. The `%k`-counter lowering above is the general fallback when neither collapse nor splitting applies.

