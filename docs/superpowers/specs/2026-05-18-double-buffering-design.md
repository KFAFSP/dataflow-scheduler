# Design: `-double-buffering` Pass

**Status:** Proposed
**Date:** 2026-05-18
**Branch:** `pc/coreletparallelism`
**Scope:** A single MLIR pass that detects scratchpad memrefs handed off between sibling stages of a `ktdf.pipeline` and rewrites them into a double-buffered (ping-pong) form. Combined analysis + transformation in one pass. Adds three new IR constructs to the KTDF dialect (`ktdf.buffer_phase`, `ktdf.select_memref`, optional `modulo` attribute on `ktdf.pipeline`) and one new method on `SchedulerExtContext`.

---

## 1. Scope & Goals

**Pass.** `-double-buffering`. Operates on `mlir::ModuleOp`. Constructed with `const scheduler::SchedulerExtContext&` (matches the existing pattern used by `ConstructThreeStagePipeline` and `ParallelizeLoopsAcrossInstances`).

**Goal.** For each `ktdf.pipeline P`, find scratchpad memrefs handed off between sibling stages — one stage writes the buffer, one or more sibling stages read it, via `ktdf.data_transfer` ops. For each such buffer, allocate two physical copies hoisted just outside the pipeline's enclosing scope, insert a phase computation and per-buffer multiplexer just before the pipeline, and rewrite the pipeline's references to use the multiplexed values. Mark the pipeline with `modulo(size: 2)` to declare its rolling-window depth.

The optimization unlocks asynchrony that the access-execute-decoupled hardware already supports: producer-of-iter-(k+1) and consumer-of-iter-(k) reference different physical buffers, so they can run concurrently across pipeline iterations. Without double-buffering, the shared LX buffer between them is a serializing bottleneck.

**Reference example.** `docs/Pipeline.md` Step 9 shows the canonical end-state IR — a 5-stage matrix-add pipeline with three LX buffers (`lx_A`, `lx_B`, `lx_C`) hoisted as paired allocs just outside `scope.loops.back()` (the `scf.for(m0)` loop in that example), a `ktdf.buffer_phase` and three `ktdf.select_memref`s inserted before the pipeline inside `scf.for(m0, n0)`, and `modulo(size: 2)` on the pipeline.

**Non-goals.**
- Not target-specific. Reads `isEligibleForDoubleBuffering(MemorySpace)` from `SchedulerExtContext`. The default returns true only for `LX`; other targets override.
- Only `ktdf.private`-yielded buffers. Function-scope and arg-passed memrefs are out of scope.
- Only `ktdf.data_transfer` ops define producer/consumer relationships. Other ops (e.g., `linalg.add` writing its `outs`) are not analyzed.
- Single producer, ≥1 consumer only. Multi-producer buffers are skipped.
- No buffer-count higher than 2. The `modulo` attribute and ops are designed to permit `size > 2` later, but the pass hardcodes 2.
- No DFIR lowering of the new ops. That's downstream of this repo.
- No re-running on already-`modulo`-set pipelines. Pre-existing `modulo` causes a skip.

---

## 2. New IR Surface

### 2.1 `ktdf.buffer_phase`

```
%phase = ktdf.buffer_phase(%m0, %n0) {num_phases = 2 : i64} : index
```

- **Operands:** variadic `index` (≥ 1) — IVs of `scf.for` loops.
- **Attribute:** `num_phases : i64` (≥ 2).
- **Result:** single `index`.
- **Trait:** `Pure`.
- **Semantics:** `phase = linearized_pipeline_iteration_count % num_phases`. The linearized count is derived from each operand IV's `(iv - lb) / step` and the trip counts of inner loops. The op walks each operand up to its enclosing `scf.for` to obtain bounds and step.
- **Verifier:**
  - Each operand must be the induction variable of an `scf.for`.
  - `num_phases >= 2`.
  - No structural check is required on the relationship among operand loops (the producer of the IR is responsible for picking the right loops; relaxed for forward compatibility).

### 2.2 `ktdf.select_memref`

```
%X_sel = ktdf.select_memref %phase[%X0, %X1] : index -> memref<?x?xf16, #ktdf.memory_space<LX>>
```

- **Operands:**
  - One `index` (the phase).
  - Variadic memref candidates (≥ 2).
- **Result:** single memref.
- **Trait:** `Pure`.
- **Semantics:** `result = candidates[phase]`. Verifier-guaranteed in-range; no wrap.
- **Verifier:**
  - All candidates have identical memref type (element type, shape including dynamic dims, memory space). Result type matches.
  - `num_candidates >= 2`.
  - The phase operand's defining op must be a `ktdf.buffer_phase` whose `num_phases` attribute equals `num_candidates`.

### 2.3 `modulo` optional attribute on `ktdf.pipeline`

```
ktdf.pipeline modulo(size: 2) { ... }
```

- New custom attribute `ModuloAttr` with one named field `size : i64`.
- Attached to `ktdf.pipeline` as `OptionalAttr<ModuloAttr>`.
- **Verifier (when present):** `size >= 2`.
- **Semantics:** declares the pipeline's rolling-window depth. With `size: N`, the pipeline rotates through N physical resource sets per N consecutive iterations.
- **Optional**: existing IR and tests without `modulo` parse, print, and verify identically — no migration required.

### 2.4 `SchedulerExtContext::isEligibleForDoubleBuffering`

```cpp
// In src/Utils/SchedulerExtContext.h, alongside getNumCorelets / isPerCoreletUnit
virtual bool isEligibleForDoubleBuffering(
    mlir::ktdf::MemorySpace mspace) const;
```

```cpp
// In src/Utils/SchedulerExtContext.cpp
bool SchedulerExtContext::isEligibleForDoubleBuffering(
    mlir::ktdf::MemorySpace mspace) const {
  return mspace == mlir::ktdf::MemorySpace::LX;
}
```

`DummySchedulerExtContext` inherits the default behavior.

---

## 3. Detection Algorithm

The pass walks `ktdf.pipeline` ops in pre-order. Each pipeline `P` is analyzed independently.

**Per-pipeline algorithm.**

1. **Pre-existing `modulo` check.** If `P.modulo` is set to anything, skip `P` entirely. No further analysis.

2. **Scope check.** Compute `scope = ktdf::common::computePipelineEnclosingScope(P)` (from `src/Transform/Common/PipelineScope`). If `scope.loops` is empty, skip `P` — no iteration dimension to rotate over.

3. **Per-yielded-value analysis.** For each value `%X` yielded by `P`'s `ktdf.private` region:

   a. **Type check.** `%X` must be a `MemRefType`. Skip otherwise.

   b. **Memory-space eligibility.** Extract memref's memory space; query `ctx.isEligibleForDoubleBuffering(mspace)`. Skip if false.

   c. **Defining op check.** Trace `%X` back through `ktdf.private_yield` to its defining op inside the private region. The defining op must be exactly `memref.alloc` (static or dynamic sizes both fine). Skip otherwise.

   d. **Single-use check.** The `memref.alloc` must have exactly one use, and that use must be the `ktdf.private_yield` op. Skip otherwise.

   e. **Same-block check.** The `memref.alloc` and the `ktdf.private_yield` must be in the same block of the private region. Skip otherwise.

   f. **Dynamic-size domination check.** If the alloc has dynamic size operands, every such operand must be defined at a location that dominates the future hoist target — the insertion point immediately before `scope.loops.back()` in the block that contains `scope.loops.back()`. Skip otherwise.

   g. **Producer / consumer scan.** For each sibling stage `S_i` of `P` (the immediate `ktdf.stage` children of `P`'s region):
      - Walk `S_i`'s region recursively, collecting every `ktdf.data_transfer` op referencing `%X`.
      - `%X` as `dest` ⇒ `S_i` joins `producers(X)`.
      - `%X` as `source` ⇒ `S_i` joins `consumers(X)`.

   h. **Producer cardinality.** `|producers(X)| == 1`. Skip otherwise.

   i. **Consumer cardinality.** `|consumers(X)| >= 1`. Skip otherwise.

   j. **Disjointness.** `producers(X) ∩ consumers(X) == ∅` (no sibling stage both writes and reads `%X`). Skip otherwise.

   k. **Token precedence.** The producer stage must transitively precede every consumer stage in `P`'s token-dependency graph (built from sibling stages' `depends_in` / `depends_out` clauses). Skip otherwise.

   If all checks pass, `%X` is a candidate.

4. **Pipeline-level guard.** If `P` has zero candidates after step 3, skip `P` (no `buffer_phase`, no `modulo`, no IR change).

The output of detection is a list of `(P, scope, [candidates])` tuples, one per pipeline that passed all guards and yielded ≥ 1 candidate.

---

## 4. Transformation Algorithm

For each `(P, scope, candidates)` tuple from §3, in pre-order:

**1. Hoist allocs.** For each candidate `%X`:
- Clone the original `memref.alloc` twice in the block containing `scope.loops.back()`, immediately before that loop. Result: SSA values `%X0`, `%X1`. The cloned allocs reuse the original alloc's dynamic size operands as-is (detection has verified domination).
- Insert matching `memref.dealloc %X0`, `memref.dealloc %X1` immediately after `scope.loops.back()` in the same block.

**2. Insert phase.** Once per pipeline, immediately before `P` in `P`'s parent block:

```
%phase = ktdf.buffer_phase(<scope IVs, outermost-first>) {num_phases = 2 : i64} : index
```

`scope.loops` is stored innermost-first per its existing convention; the operands here are reversed to outermost-first.

**3. Insert selectors.** For each candidate `%X`, immediately before `P` (after the `buffer_phase`), in the same block:

```
%X_sel = ktdf.select_memref %phase[%X0, %X1] : index -> <memref-type-of-X>
```

All selectors share the single `%phase` SSA value.

**4. Rewrite uses.** For each candidate, walk `P`'s region recursively (descending into nested pipelines and stage bodies). Replace every operand reference to `%X` with `%X_sel`.

**5. Set modulo.** Set `P`'s `modulo` attribute to `ModuloAttr{size: 2}`.

**6. Clean up `ktdf.private`.** For each candidate `%X`:
- Drop the result-type slot corresponding to `%X` from `ktdf.private`'s result types.
- Drop the matching operand from `ktdf.private_yield`.
- Erase the orphaned `memref.alloc` inside the region.

The structural cleanup of `ktdf.private` is performed by erasing and recreating the op with the reduced result type list (or via the dialect's existing builder helper for shrinking results), and by patching the `ktdf.private_yield`'s operand list. All uses outside the region of the dropped result are guaranteed to have been rewritten in step 4.

**Iteration order across candidates.** Steps 1, 3, 4, 6 iterate over all candidates of `P`. Steps 2 and 5 happen exactly once per pipeline. The shared `%phase` from step 2 feeds every selector inserted in step 3.

**No structural changes to:**
- The producer/consumer `ktdf.data_transfer` ops (only operand renames).
- The pipeline's stages, tokens, or stage-level dependencies.
- Anything outside `P`'s parent block or the private region.

---

## 5. Pass Plumbing

- **Class:** `DoubleBufferingPass`
- **CLI flag:** `-double-buffering`
- **Anchor op:** `mlir::ModuleOp`
- **Constructor:** `::ktdf::createDoubleBufferingPass(const scheduler::SchedulerExtContext&)`
- **Options:** none
- **TableGen declaration:** `src/Transform/Passes.td`
- **Constructor declaration:** `src/Transform/Passes.h`
- **Implementation:** `src/Transform/DoubleBuffering.cpp` (one file, mirroring `BroadcastPromotion.cpp` and `ParallelizeLoopsAcrossInstances.cpp`)
- **Build target:** updated in `src/Transform/CMakeLists.txt`

**Pipeline registration.** In `tools/scheduler/scheduler-pipeline.cpp::buildKTDPToDFIRPipeline`, append after `createParallelizeLoopsAcrossInstancesPass(scheduler_ctx)`:

```cpp
pm.addPass(createParallelizeLoopsAcrossInstancesPass(scheduler_ctx));
pm.addPass(createDoubleBufferingPass(scheduler_ctx));
```

---

## 6. Test Layout

`test/Transform/DoubleBuffering/` mirrors the pattern established by the `parallelize-loops-across-instances` tests:

**Positive tests:**
- `basic.mlir` — canonical 5-stage matrix-add pipeline → double-buffered shape with three buffers, one shared phase, three selectors.
- `multiple-candidates.mlir` — verify all candidates of one pipeline share a single `buffer_phase`.
- `producer-in-nested-pipeline.mlir` — producer transfer sits inside a nested pipeline of a sibling stage.
- `consumer-in-nested-pipeline.mlir` — consumer transfer sits inside a nested pipeline of a sibling stage.
- `multiple-consumers.mlir` — one producer, two or more consumer stages.

**Negative tests** (pass is a no-op):
- `negative-no-scope.mlir` — pipeline has no enclosing `scf.for`.
- `negative-ineligible-memory-space.mlir` — buffer in HBM or SFP_REG.
- `negative-producer-equals-consumer.mlir` — same sibling stage both writes and reads.
- `negative-multi-producer.mlir` — two sibling stages write the same buffer.
- `negative-non-alloc-defining-op.mlir` — buffer's defining op is not `memref.alloc` (e.g., `memref.cast`).
- `negative-alloc-multi-use.mlir` — alloc used by something inside private region in addition to the yield.
- `negative-dynamic-size-not-dominating.mlir` — dynamic size operand defined inside `scope.loops`.
- `negative-pre-existing-modulo.mlir` — pipeline already has `modulo` set.

**Round-trip tests** for the new IR:
- `roundtrip-buffer-phase.mlir`
- `roundtrip-select-memref.mlir`
- `roundtrip-modulo-attr.mlir`

**Idempotence test:**
- `idempotence.mlir` — running the pass twice produces the same IR (the second run skips because of the pre-existing `modulo`).

**Debug-log test:**
- `debug-log.mlir` — verifies the pass emits expected debug output via `LLVM_DEBUG`.

---

## 7. Open Items / Future Extensions

These are intentionally out of scope for the first cut but the design accommodates them:

- **N-buffering for N > 2.** The `modulo` attribute and ops support arbitrary `size`. The pass currently hardcodes 2.
- **Folding `select_memref` when the phase is statically known.** Verifier rule §2.2 (phase must come from `buffer_phase`) currently precludes this. A future canonicalization can relax that rule.
- **Multiple producers.** Detection currently requires `|producers| == 1`. Supporting multiple producers requires a story for which producer's write the phase tracks.
- **Buffers not allocated in `ktdf.private`.** Function-arg memrefs and function-scope allocs are excluded by §3.3.c.
- **Non-LX scratchpads.** The `SchedulerExtContext` method abstracts the policy; another target's context can return true for additional spaces.
