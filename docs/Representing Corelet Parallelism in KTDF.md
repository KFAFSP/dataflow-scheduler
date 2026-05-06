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

We introduce a new KTDF operation, `ktdf.parallel`, that expresses a multi-dimensional parallel iteration space together with a block-cyclic distribution of those iterations across a fixed number of hardware instances. The op is deliberately target-agnostic: it declares that iterations are replicated across some number of instances, but does not say whether those instances are corelets, cores, or anything else. Lowering to DFIR binds the instance axis to the concrete Spyre corelet axis.

The construct has two attributes that carry the scheduling decision:

- **`num_instances`** — an `i64` attribute giving the number of hardware instances the iteration space should be distributed across. For Spyre corelet parallelism this is always `2`.
- **`block_size`** — an `i64` attribute giving the size of the contiguous block of iterations handed to one instance before the distribution rotates to the next instance.

Iteration `k` in the lexicographic enumeration of the iteration space is owned by instance `(k / block_size) % num_instances`. Pure block distribution arises when `block_size` equals `ceil(total_iters / num_instances)`; pure cyclic distribution arises when `block_size` is `1`; intermediate values give block-cyclic. The body runs once per owned iteration, in lex order, with an implicit barrier at the end of the op.

Importantly, the op's iteration space and the attributes together specify the distribution completely. The body does not need to compute a corelet id or perform arithmetic to recover its own iteration indices — every iteration's induction variables are bound directly by the op, exactly as with `scf.parallel`.

## 4. 1-D Case: One Parallel Loop

Consider the inner parallel loop from the coarsened compute stage in `test/Transforms/StageCoarsening/basic-out.mlir`. The `n` dimension has 64 iterations of a 1×64 tile, each iteration running the LX→SFP→LX compute trio. With two corelets available, we would like each corelet to run 32 of those iterations in parallel.

Before replication, the stage body has a plain `parallel_loop`-annotated `scf.for`:

```mlir
ktdf.stage depends_in(%res_t1) depends_out(%res_t5) {
  scf.for %n = %c0 to %c4096 step %c64 {
    ktdf.pipeline { /* SFP trio */ }
  } {loop_type = #ktdf.loop_type<parallel_loop>}
}
```

After applying the `ktdf.parallel` construct with `num_instances = 2` and `block_size = 32`:

```mlir
ktdf.stage depends_in(%res_t1) depends_out(%res_t5) {
  ktdf.parallel (%n) = (%c0) to (%c4096) step (%c64)
                       distribute(num_instances = 2, block_size = 32) {
    ktdf.pipeline { /* SFP trio */ }
  }
}
```

The 64 iterations are split into two contiguous blocks of 32. Corelet 0 runs iterations `k ∈ [0, 32)`, corresponding to `n ∈ {0, 64, 128, …, 31·64}`. Corelet 1 runs iterations `k ∈ [32, 64)`, corresponding to `n ∈ {32·64, 33·64, …, 63·64}`. The loop body is unchanged; only the enclosing construct is different.

The `num_instances` and `block_size` attributes carry the entire scheduling decision. The fact that `64 / 2 = 32` divides evenly means there is no imbalance in this case — both corelets run exactly 32 iterations.

## 5. Imbalance from a Leftover Block

The 1-D case becomes interesting when the total iteration count does not divide evenly by `num_instances · block_size`. Consider a parallel loop of trip count 7 distributed across 2 corelets. With pure block distribution, `block_size = ceil(7 / 2) = 4`:

```mlir
ktdf.parallel (%q) = (%c0) to (%c7) step (%c1)
                     distribute(num_instances = 2, block_size = 4) {
  ktdf.pipeline { /* body uses %q */ }
}
```

The seven iterations are laid out in lex order as `k = 0, 1, 2, 3, 4, 5, 6`. The first block of 4 (iterations `0..3`) goes to corelet 0; the second block (iterations `4..6`) goes to corelet 1. Corelet 0 runs 4 iterations and corelet 1 runs only 3. The last iteration's worth of work is missing from corelet 1 — the *leftover block* is incomplete.

This imbalance is inherent to the geometry of 7 / 2. It costs roughly `1/7 ≈ 14%` of the available corelet throughput for this particular stage. The construct makes the imbalance visible in the IR without requiring the kernel author or the scheduler to introduce any arithmetic to model it — the leftover iterations simply occupy the tail of the last instance's block.

Smaller trip counts produce proportionally worse imbalance. A parallel loop of trip count 3 distributed across 2 corelets gives `block_size = 2`, corelet 0 runs 2 iterations, corelet 1 runs 1, and half of corelet 1's capacity for that stage is idle. A trip count of 5 gives `block_size = 3`, splitting 3+2 with 20% imbalance. Whether this is acceptable is a policy decision for the pass that inserts `ktdf.parallel` — but the cost is plainly readable from the op's attributes and trip count.

## 6. Multi-Dimensional Iterators: Reducing Imbalance

When a kernel exposes more than one parallel loop in the nest, the 1-D case above is strictly worse than it needs to be. Splitting only the outermost parallel dimension ignores the iterations available in the inner parallel dimensions, which could have filled out the leftover block.

Consider two nested parallel loops of trip counts 5 and 7, both contained inside a coarsened compute stage. Distributing across 2 corelets by splitting only the outer dimension gives `block_size = 3` on the size-5 loop, so corelet 0 runs `3 × 7 = 21` iterations and corelet 1 runs `2 × 7 = 14`. The imbalance is `7 / 35 ≈ 20%` — substantial, and it comes entirely from refusing to reach into the inner dimension.

The multi-dimensional form of `ktdf.parallel` avoids this by declaring the whole Cartesian product as the iteration space, and letting `block_size` cut across that space in lex order:

```mlir
ktdf.parallel (%p, %q) = (%c0, %c0) to (%c5, %c7) step (%c1, %c1)
                         distribute(num_instances = 2, block_size = 18) {
  ktdf.pipeline { /* body uses %p and %q */ }
}
```

The lex enumeration of the 5×7 space produces 35 iterations: `(0,0), (0,1), …, (0,6), (1,0), (1,1), …, (4,6)`. With `block_size = 18`, the first 18 iterations go to corelet 0 and the remaining 17 go to corelet 1. The imbalance is now `1 / 35 ≈ 3%` — roughly a seventh of the single-dimension cost, and essentially as balanced as the 35-iteration space permits for two instances.

The induction variables `%p` and `%q` are bound by the op exactly as they would be by two nested `scf.for` loops, so the body reads naturally: every iteration sees its `(p, q)` pair in scope. No explicit arithmetic is needed to recover iteration coordinates from a flattened index — the op handles the enumeration. This keeps the IR shape close to the original loop nest while expressing the balanced distribution that a single-dimensional split could not achieve.

In general, including more parallel dimensions in the `ktdf.parallel` iteration space increases the total iteration count the distribution acts over, which in turn lets `block_size` land closer to an even split. The upper bound on achievable balance for `N` instances is determined by the total iteration count across all included parallel dimensions: with `T` total iterations and `N` instances, the smallest achievable imbalance is at most `1 / T` per instance. Exposing more parallel dimensions to the op is the only mechanism for reducing `T`'s denominator, and therefore the only mechanism for improving balance beyond what any single dimension can provide.

## 7. End-to-End Example

This section shows a complete coarsened-pipeline IR before and after the `ktdf.parallel` construct is introduced, using the same kernel structure as `test/Transforms/StageCoarsening/basic-out.mlir` (the 5-stage matrix-add after stage coarsening) but with a larger tile-row count so the parallel split is meaningful. The kernel processes a 96×64 tile of `C = A + B` on one core's slice, with the `m` loop running 32 iterations of 1×64 tiles and the `n` loop running 1 iteration at 64-wide granularity (so the parallel dimension of interest is `m`).

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
          scf.for %n = %c0 to %c64 step %c64 {
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
          scf.for %n = %c0 to %c64 step %c64 {
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
          scf.for %n = %c0 to %c64 step %c64 {
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

### 7.2 After: the same IR with `ktdf.parallel` applied

The parallelization pass walks each stage, finds the parallel loop nest inside, and rewrites it as a `ktdf.parallel` with `num_instances = 2` and a `block_size` chosen to balance. Here the `m` dimension has 32 iterations and `n` has 1, so the total iteration space has 32 iterations — evenly divisible by 2 with `block_size = 16`. Corelet 0 runs iterations `m ∈ [0, 16)`, corelet 1 runs `m ∈ [16, 32)`. The nested pipeline inside Stage 5 is not itself parallelized (its stages are sequential within one corelet); the parallelism lives in the `m` loop that wraps it.

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

      // Stage 0 — HBM -> LX, distributed across 2 corelets
      ktdf.stage depends_in(none) depends_out(%res_t1 : !ktdf.token) {
        ktdf.parallel (%m, %n) = (%c0, %c0) to (%c32, %c64) step (%c1, %c64)
                                 distribute(num_instances = 2, block_size = 16) {
          ktdf.data_transfer from %A[%m, %n] size [1, 64]
                             to %res_lx_A[%c0] size [64]
            : memref<32x64xf16, ..., #ktdf.memory_space<HBM>>,
              memref<64xf16,   #ktdf.memory_space<LX>>
          ktdf.data_transfer from %B[%m, %n] size [1, 64]
                             to %res_lx_B[%c0] size [64]
            : memref<32x64xf16, ..., #ktdf.memory_space<HBM>>,
              memref<64xf16,   #ktdf.memory_space<LX>>
        }
      }

      // Stage 5 — coarsened compute, distributed across 2 corelets
      ktdf.stage depends_in(%res_t1 : !ktdf.token) depends_out(%res_t5 : !ktdf.token) {
        ktdf.parallel (%m, %n) = (%c0, %c0) to (%c32, %c64) step (%c1, %c64)
                                 distribute(num_instances = 2, block_size = 16) {
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

      // Stage 4 — LX -> HBM, distributed across 2 corelets
      ktdf.stage depends_in(%res_t5 : !ktdf.token) depends_out(none) {
        ktdf.parallel (%m, %n) = (%c0, %c0) to (%c32, %c64) step (%c1, %c64)
                                 distribute(num_instances = 2, block_size = 16) {
          ktdf.data_transfer from %res_lx_C[%c0] size [64]
                             to %C[%m, %n] size [1, 64]
            : memref<64xf16, #ktdf.memory_space<LX>>,
              memref<32x64xf16, ..., #ktdf.memory_space<HBM>>
        }
      }
    } // outer ktdf.pipeline
    return
  }
}
```

Each stage's parallel nest has been replaced by one `ktdf.parallel` op whose induction variables `%m` and `%n` are still in scope inside the body — the data-transfer and compute operations read identically to the before-IR. What changes is that each `ktdf.parallel` declares the iteration space as a whole, and its `distribute(num_instances = 2, block_size = 16)` attributes specify that the 32-iteration lex enumeration is split as the first 16 to corelet 0 and the last 16 to corelet 1.

Two things to notice. First, the outer pipeline's token flow (`%res_t1` produced by Stage 0, consumed by Stage 5; `%res_t5` produced by Stage 5, consumed by Stage 4) is unchanged — tokens still express pipeline-level ordering and the parallelization happens entirely inside each stage. Second, the nested pipeline inside Stage 5 is untouched: Stages 1–3 and their SFP_REG private region remain as they were, because the corelet replication is expressed at the `m` loop level, not below. Each corelet enters the nested pipeline with its own private SFP_REG allocations and its own iteration range over `m`.

The shared LX buffers (`%res_lx_A`, `%res_lx_B`, `%res_lx_C`) are visible to both corelets. Because the two corelets own disjoint `m` iterations and Stage 5's nested pipeline allocates fresh SFP_REG buffers per instance, the LX traffic does not race at the op level — corelet 0's Stage 1 reads LX data that corelet 0's Stage 0 already wrote for `m ∈ [0, 16)`, and similarly for corelet 1. Formalizing the LX sharing discipline is a concern for lowering, but the KTDF IR here is enough to express the intended per-corelet iteration ownership cleanly.

## 8. Placement in the Scheduler Pipeline

`ktdf.parallel` is introduced by a dedicated pass that runs after stage-coarsening and before lowering to DFIR. The pass walks each stage's body, collects the parallel loops within it, chooses how many of those loops to include in the iteration space (based on imbalance policy and the fixed `num_instances` supplied as a pass option), picks a `block_size`, and rewrites the loop nest as a single `ktdf.parallel` with the chosen attributes. Inner sequential (non-parallel) loops remain inside the body unchanged.

Because `num_instances` equals the hardware instance count by construction, the IR that emerges from this pass is directly lowering-ready: DFIR lowering binds the op's distribution axis to corelets and emits one program per corelet, each enumerating only its assigned subset of iterations. No further scheduling decisions remain between `ktdf.parallel` and corelet-bound DFIR.

## 9. Summary

`ktdf.parallel` expresses corelet parallelism in KTDF without introducing corelet-specific vocabulary into the dialect. Its two attributes — `num_instances` and `block_size` — together specify a block-cyclic distribution of a multi-dimensional iteration space across a fixed set of hardware instances. The 1-D form handles the common case of a single parallel loop cleanly when the trip count divides evenly, and makes leftover-block imbalance plainly visible when it does not. The multi-dimensional form folds additional parallel dimensions into the distributed space, converting coarse-grained single-dimension imbalance into fine-grained whole-space imbalance that is typically an order of magnitude smaller. Neither form requires the body to perform arithmetic on a flattened index or a corelet id — the op binds induction variables directly, keeping the body identical in shape to the loop nest it replaces.

The dataflow-style corelet split — where the two corelets take on different roles rather than different iteration ranges — is intentionally outside this proposal's scope and is a candidate for a separate future discussion.
