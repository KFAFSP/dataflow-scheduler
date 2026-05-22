# Design: `-parallelize-loops-across-instances` Pass

**Status:** Proposed
**Date:** 2026-05-15
**Branch:** `pc/coreletparallelism`
**Scope:** A single MLIR pass that finds parallel `scf.for` loops enclosing per-corelet `ktdf.pipeline`s and rewrites them into `ktdf.parallel` to distribute iterations across hardware instances. Combined analysis + transformation in one pass.

---

## 1. Scope & Goals

**Pass.** `-parallelize-loops-across-instances`. Operates on `mlir::ModuleOp`. Constructed with `const scheduler::SchedulerExtContext&` (matches the existing pattern used by `ConstructThreeStagePipeline`).

**Goal.** For each `ktdf.pipeline` whose work is corelet-local (no core-shared units in the union of its immediate stages' `applicable_units`), find a parallel `scf.for` in the pipeline's enclosing scope whose static trip count permits a split across `num_instances`, and rewrite that loop into a `ktdf.parallel` op (introduced by PR #5). The resulting iterations distribute across the per-corelet hardware units at lowering time.

**Reference example.** `test/Transform/tmp.mlir` shows the transformation: an inner pipeline with `applicable_units = [SFP, LXLU, LXSU]` whose enclosing scope has a `parallel_loop`-annotated `scf.for %arg1 = %c0 to %c2 step %c1` becomes `ktdf.parallel (%arg1, %inst) = (%c0) to (%c2) step (%c1) distribute(num_instances = 2)` with the loop body spliced in.

**Non-goals.**
- Not target-specific. Reads `getNumCorelets()` and `isPerCoreletUnit(...)` from `SchedulerExtContext`. No "Spyre" or "corelet" identifiers in the pass code.
- No multi-dim collapse (spec §6 of `docs/Representing Corelet Parallelism in KTDF.md`). One `scf.for` becomes a 1-D `ktdf.parallel`. Multi-dim collapse is a future extension.
- No body-level legality (cross-iteration SSA dependencies, LX sharing semantics). Trusts the `loop_type = parallel_loop` annotation and the upstream lowering's `applicable_units` correctness.
- No reduction handling. `loop_type = reduction_loop` is filtered out structurally.
- No DFIR lowering for `ktdf.parallel`. That work is downstream of this repo.

---

## 2. Inputs, Pre-conditions, and Hard Assumptions

**Pre-conditions on input IR.**
- The IR has been stage-coarsened (hierarchical pipeline shape).
- Every `ktdf.stage` has its `applicable_units` attribute set. The pass derives a pipeline's effective `applicable_units` as the union of its immediate child stages' attributes (see §6 Prerequisite #1). A stage missing the attribute is a programmer error; the utility calls `llvm::report_fatal_error` rather than guessing.
- `scf.for`s in scope-walk position are annotated with `loop_type` if their iteration ordering is known parallel or reduction. Loops without the annotation are treated as non-parallel.

**Target abstraction (queried via `SchedulerExtContext`).**
- `ctx.getNumCorelets()` — returns `num_instances` for the target. Defaults to `2` (Spyre value) in the dummy.
- `ctx.isPerCoreletUnit(mlir::ktdf::ComponentType)` — true iff the unit lives on a corelet (LXLU/LXSU/SFP/PE/PT). False for shared units (L3LU/L3SU). Default Spyre answers in the dummy.
- Both methods are added to `SchedulerExtContext` as part of this work; both are `virtual` so future contexts (richer architecture graphs, test fixtures) can override.

**Hard assumptions (no analysis to verify them).**
- `loop_type = parallel_loop` correctly identifies loops whose iterations are reorderable.
- `applicable_units` accurately enumerates every unit the pipeline engages.
- DFIR lowering binds `%inst` to the corelet id. The pass leaves `%inst` unused at this layer.

---

## 3. Analysis Algorithm

The analysis runs inline in `runOnOperation`, no separate analysis class. A candidate is the tuple:

```cpp
struct Candidate {
  mlir::ktdf::PipelineOp pipeline;
  mlir::scf::ForOp loop;        // the scf.for to rewrite
  int64_t trip_count;           // statically known
  bool balanced;                // trip_count % num_instances == 0
};
```

The pass performs a **pre-order walk** over `ktdf.pipeline` ops in the module. Each pipeline is analyzed independently. Pre-order is preferred so the traversal aligns with the "outermost first" convention used elsewhere in the pass; walk order does not affect the final IR (pipelines do not interact through this transformation).

**Per-pipeline algorithm.**

1. **Compute pipeline's effective `applicable_units`.** Call `ktdf::common::computePipelineApplicableUnits(pipeline)` — unions `applicable_units` across the pipeline's immediate child stages (see §6 Prerequisite #1). Each stage **must** have the attribute; missing attribute triggers `llvm::report_fatal_error`. If the result is the empty set (e.g., the pipeline has no stages, only a `ktdf.private`), skip this pipeline.

2. **Filter by per-corelet membership.** For each `ComponentType` in the result, query `ctx.isPerCoreletUnit(unit)`. If any returns false, skip this pipeline.

3. **Compute enclosing scope.** Call `ktdf::common::computePipelineEnclosingScope(pipeline)` (from `src/Transform/Common/PipelineScope`). Returns `scope.loops` (innermost → outermost) — the chain of `scf.for`s above the pipeline whose body contains *only* the chain to the pipeline. If `scope.loops` is empty, skip.

4. **Loop selection (two passes, both walking outermost → innermost).**

   **Pass 1 — balanced.** First loop satisfying all four:
   - `loop_type = parallel_loop` (read from the `scf.for`'s discardable attributes; missing or `reduction_loop` ⇒ skip this loop, continue inward).
   - Statically-known trip count. Compute via `lb`, `ub`, `step` all being defined by `arith.constant`. If any is dynamic ⇒ skip this loop.
   - `trip_count >= num_instances` (where `num_instances = ctx.getNumCorelets()`). If less ⇒ skip this loop.
   - `trip_count % num_instances == 0`. If not ⇒ skip this loop.

   Accept the first qualifying loop as a balanced candidate; stop.

   **Pass 2 — fallback (imbalanced).** Only if Pass 1 found nothing. Walk outermost → innermost again. Accept the first loop satisfying conditions 1+2+3 (drop divisibility). Stop.

5. If neither pass produced a loop, no candidate; skip the pipeline.

**Trip count computation.** A small private helper `getStaticTripCount(scf::ForOp)` returns `std::optional<int64_t>` — defined iff `lb`, `ub`, and `step` are all defined by `arith.constant index` ops with positive `step`. The helper computes `(ub - lb + step - 1) / step` as the natural ceiling-div.

**Diagnostics (debug log only, no IR change).**

Under `LLVM_DEBUG` with tag `parallelize-loops-across-instances`:
- Per-pipeline result: skipped (with reason: empty effective applicable_units, non-corelet unit, no scope, no parallel loop, no static trip count, all trip counts < num_instances) or candidate accepted (balanced or imbalanced; loop location; trip count).
- For imbalanced acceptance: trip count, num_instances, per-instance chunk sizes (`ceil(T/N)` for instance 0..N-2, leftover for instance N-1).

---

## 4. Transformation

For each `Candidate {pipeline, loop, trip_count, balanced}`:

1. **Capture references.** Bind locals to the `scf.for`'s `lb`, `ub`, `step` operand `Value`s; its IV `BlockArgument`; its body block; its `Location`. No copies — these are forwarded into the new op.

2. **Build `ktdf.parallel` at the `scf.for`'s position.** Use the custom builder added in PR #5:
   ```cpp
   auto parallel = mlir::ktdf::ParallelOp::create(builder, 
       loc,
       /*lowerBounds=*/ValueRange{lb},
       /*upperBounds=*/ValueRange{ub},
       /*steps=*/ValueRange{step},
       /*numInstances=*/ctx.getNumCorelets());
   ```
   The builder creates an empty body block with two `index` block arguments (the IV and `%inst`).

3. **Splice the body.** Move every operation from the `scf.for`'s block (except the trailing `scf.yield`) into the new `ktdf.parallel`'s body, preserving order. MLIR primitive:
   ```cpp
   parallel.getBody().front().getOperations().splice(
       parallel.getBody().front().begin(),
       forOp.getBody()->getOperations(),
       forOp.getBody()->begin(),
       std::prev(forOp.getBody()->end()));
   ```

4. **Replace the IV.** RAUW the original `scf.for`'s IV with `parallel.getInductionVar(0)`.

5. **Append the terminator.** Emit `ktdf.parallel_yield` at the end of the parallel body (no-op builder).

6. **Erase the original `scf.for`.** Safe at this point — its body is empty (only the orphaned `scf.yield` remains and is erased with the parent op).

7. **`%inst` is bound but unused.** No further rewrites; the body does not reference `%inst`.

**Attribute handling.**
- The original `scf.for`'s `loop_type = parallel_loop` attribute disappears with the op. `ktdf.parallel` has no equivalent attribute, and that is correct: the construct subsumes the meaning.
- Inner loops keep their attributes verbatim. The pass touches only the chosen loop.

**Token flow.**
- Untouched. The chosen loop sits inside a stage body; the stage's `depends_in`/`depends_out` are above the loop and remain unchanged. Inner pipeline tokens are inside the loop body; they move with the body and remain valid.

**No fixed-point iteration.**
- One walk over the module. Each pipeline produces at most one candidate. The transformation does not create new candidates: a `ktdf.parallel` is not an `scf.for`, so it cannot be re-discovered by Step 4.

**End state.** Structurally legal by construction — `ktdf.parallel` is a valid op anywhere a `scf.for` was legal (no `HasParent` constraint), the spliced body is the same MLIR ops in the same order, the IV substitution preserves SSA, and the new terminator satisfies the parallel op's verifier.

---

## 5. Legality Boundaries and Rejection Paths

This pass is best-effort. Rejection is not an error — it just means no rewrite for that pipeline. The pass exits cleanly even if no candidate is found anywhere.

**Per-pipeline rejection cases (each silently skips the pipeline; the pass continues):**

| Cause | Where it's checked |
|---|---|
| Pipeline's effective `applicable_units` (union over immediate stages) is empty | §3 step 1 |
| Any unit in the effective `applicable_units` is not per-corelet (e.g., L3LU/L3SU) | §3 step 2 |
| Pipeline has no qualifying enclosing scope (empty `scope.loops`) | §3 step 3 |
| No loop in scope is `parallel_loop`-annotated | §3 step 4 (Pass 1 + Pass 2 both find nothing) |
| All parallel loops in scope have non-static trip counts | §3 step 4 |
| All parallel loops in scope have trip count `< num_instances` | §3 step 4 |

**Hard failures (not silently skipped):**
- A `ktdf.stage` inside a pipeline being analyzed lacks the `applicable_units` attribute. Triggers `llvm::report_fatal_error` from the `ApplicableUnits` utility. Indicates an upstream-pass invariant violation.

**Cases that look like they need legality checks but don't (within this pass's contract):**
- *Cross-iteration data dependencies in the body.* The `parallel_loop` annotation is the contract.
- *LX sharing across iterations.* Inside a coarsened compute stage's body, iterations of the chosen loop are independent by the parallel-loop assumption. LX-sharing semantics materialize at DFIR.
- *FIFO usage inside the body.* FIFOs are pipeline-private (`ktdf.private` is pipeline-scoped). Iterations of the chosen loop run separate instances of the inner pipeline; each instance allocates its own slots. Spec §3.4 of the corelet doc covers this.
- *Step ≠ 1.* The pass forwards whatever `step` operand the `scf.for` had to `ktdf.parallel`. The construct accepts arbitrary `index` step operands per PR #5.

**Verifier and structural integrity.**
- The MLIR verifier runs after the pass via the standard pass-pipeline mechanism. Any structural regression surfaces there.
- Structural invariants preserved by construction:
  - The new `ktdf.parallel` has exactly two block arguments — built by the custom builder.
  - The body block has exactly one terminator (`ktdf.parallel_yield`) — appended explicitly.
  - The original `scf.for`'s parent block layout is preserved (`ktdf.parallel` lands at the same position).
  - SSA validity: the IV RAUW substitutes `scf.for`'s IV with the parallel's first block arg, both `index`-typed.
- Debug-build assertion: after each rewrite, `assert(succeeded(parallel.verify()))`.

**No structural changes outside the rewritten loop.** Stage tokens, pipeline private regions, and ops outside the rewritten `scf.for`'s body are untouched.

---

## 6. Dialect & Context Prerequisites

These changes are not strictly part of the pass; they must land before the pass code can compile or test against realistic IR.

### Prerequisite #1 — `ApplicableUnits` utility (no dialect change)

Today `applicable_units` lives on `ktdf.stage` only (`KTDFOps.td:445`). Rather than add a redundant attribute to `ktdf.pipeline`, the pass derives a pipeline's effective `applicable_units` by unioning over its immediate child stages.

- New utility at `src/Transform/Common/ApplicableUnits.{cpp,hpp}`, namespace `ktdf::common`:
  ```cpp
  /// Union the `applicable_units` of all immediate child stages of `pipeline`.
  /// Does not descend into nested pipelines (those are analyzed independently).
  /// Skips non-stage children (e.g., ktdf.private). Asserts that every
  /// immediate stage has its applicable_units attribute set; absence is an
  /// upstream-pass invariant violation and triggers llvm::report_fatal_error.
  llvm::SmallSetVector<mlir::ktdf::ComponentType, 4>
  computePipelineApplicableUnits(mlir::ktdf::PipelineOp pipeline);
  ```
- No dialect change. `KTDFOps.td` is not modified.
- Unit-tested via the pass's lit tests (no separate harness today).

### Prerequisite #2 — `SchedulerExtContext` API extensions

Add to `src/Utils/SchedulerExtContext.{h,cpp}`:

```cpp
/// Number of hardware instances iterations should be distributed across when
/// emitting ktdf.parallel. Spyre = 2 (per corelet).
virtual unsigned getNumCorelets() const;

/// True iff `unit` is a per-corelet unit (LXLU/LXSU/SFP/PE/PT). False for
/// core-shared units (L3LU/L3SU). Used to decide whether a pipeline's work
/// is corelet-local.
virtual bool isPerCoreletUnit(mlir::ktdf::ComponentType unit) const;
```

- Both methods are `virtual` (the existing methods are non-virtual; we don't change them, but new ones gain virtual dispatch so future contexts and tests can override).
- Default (real-context) implementations return Spyre values:
  - `getNumCorelets()` → `2`.
  - `isPerCoreletUnit(SFP|LXLU|LXSU)` → `true`; `isPerCoreletUnit(L3LU|L3SU)` → `false`.
- `DummySchedulerExtContext` inherits these defaults — Spyre-flavored, matching the rest of the dummy's existing behavior.

### Prerequisite #3 — Pipeline integration

Wire the pass into `buildKTDPToDFIRPipeline` (in `tools/scheduler/scheduler-pipeline.cpp`) **after** the stage-coarsening invocation. The `scheduler_ctx` parameter is already in scope:

```cpp
pm.addPass(createStageCoarseningPass(/*loop-nest-depth=*/2));
pm.addPass(createParallelizeLoopsAcrossInstancesPass(scheduler_ctx));
```

(The exact insertion point follows whichever pass produces the post-coarsening hierarchical pipeline shape; today this is `StageCoarsening`, but if the upstream pipeline-builder gains additional passes between coarsening and DFIR lowering, the order is "any time after `applicable_units` is set on stages and before DFIR lowering.")

**No other dialect work required.** `ktdf.parallel` and `ktdf.parallel_yield` already exist in the dialect (PR #5).

---

## 7. File Structure & Components

| Path | Action | Purpose |
|---|---|---|
| `src/Transform/ReusePromotion/Common/PipelineScope.{cpp,hpp}` | Move | Relocate to `src/Transform/Common/PipelineScope.{cpp,hpp}`; rename namespace `ktdf::reuse` → `ktdf::common`. |
| `src/Transform/Common/ApplicableUnits.{cpp,hpp}` | Create | New utility: `computePipelineApplicableUnits` (Prereq #1). Namespace `ktdf::common`. |
| `src/Transform/Common/CMakeLists.txt` | Create | Builds the moved/new utilities into a library `KTDFTransformCommon` (or similar). |
| `src/Transform/ReusePromotion/Common/CMakeLists.txt` | Modify | Drop `PipelineScope.cpp` from the source list; link against `KTDFTransformCommon` for callers. |
| `src/Transform/BroadcastPromotion.cpp` | Modify | Update include path (`ReusePromotion/Common/PipelineScope.hpp` → `Common/PipelineScope.hpp`) and namespace usage (`ktdf::reuse::computePipelineEnclosingScope` → `ktdf::common::computePipelineEnclosingScope`). |
| `src/Utils/SchedulerExtContext.h` | Modify | Declare `getNumCorelets()` / `isPerCoreletUnit(ComponentType)` as `virtual`. |
| `src/Utils/SchedulerExtContext.cpp` | Modify | Implement default Spyre values. |
| `src/Transform/Passes.td` | Modify | Register `def ParallelizeLoopsAcrossInstancesPass`. |
| `src/Transform/Passes.h` | Modify | Declare `createParallelizeLoopsAcrossInstancesPass(const scheduler::SchedulerExtContext&)`. |
| `src/Transform/ParallelizeLoopsAcrossInstances.cpp` | Create | Pass entry, analysis (private static helpers), transformation, debug logging. |
| `src/Transform/CMakeLists.txt` | Modify | Add the new source file; add subdirectory `Common`. |
| `tools/scheduler/scheduler-pipeline.cpp` | Modify | Insert the pass into `buildKTDPToDFIRPipeline`. |
| `test/Transforms/ParallelizeLoopsAcrossInstances/*.mlir` | Create | Lit tests (positive transformations, rejection cases, round-trip, idempotence, debug). |

**Component layering inside `ParallelizeLoopsAcrossInstances.cpp`:**

```cpp
namespace {

struct Candidate {
  mlir::ktdf::PipelineOp pipeline;
  mlir::scf::ForOp loop;
  int64_t trip_count;
  bool balanced;
};

// Analysis (static helpers, no class):
std::optional<int64_t> getStaticTripCount(mlir::scf::ForOp loop);
bool isParallelLoop(mlir::scf::ForOp loop);
bool allUnitsAreCoreletLocal(mlir::ktdf::PipelineOp pipeline,
                             const scheduler::SchedulerExtContext& ctx);
std::optional<mlir::scf::ForOp> selectLoop(
    llvm::ArrayRef<mlir::scf::ForOp> scope_loops_outermost_to_innermost,
    unsigned num_instances);
std::optional<Candidate> findCandidate(
    mlir::ktdf::PipelineOp pipeline,
    const scheduler::SchedulerExtContext& ctx);

// Transformation:
void rewrite(const Candidate& c);

// Pass entry:
struct ParallelizeLoopsAcrossInstancesPass : public PassWrapper<...> {
  const scheduler::SchedulerExtContext& ctx_;
  void runOnOperation() override;
};

}  // namespace
```

Each helper is small, single-purpose, and unit-testable in isolation if a C++ unit-test harness is added later. Today they are exercised integration-style through lit tests.

**Reuse note.** The pass uses `ktdf::common::computePipelineEnclosingScope` and `ktdf::common::computePipelineApplicableUnits`, both from `src/Transform/Common/`. The scope-walk utility was previously under `ReusePromotion/Common/` (namespace `ktdf::reuse`); this design moves it to a pass-agnostic location alongside the new `ApplicableUnits` utility. The existing scope-walk rules match this pass's needs exactly — no behavioral changes, only a relocation. Existing callers (e.g., `BroadcastPromotion`) update their include paths and namespace usage.

---

## 8. Testing

**Framework.** LLVM `lit` + `FileCheck` via `build/scheduler/bin/scheduler`. New tests live under `test/Transforms/ParallelizeLoopsAcrossInstances/`.

**Standard RUN line:**
```
// RUN: scheduler %s -parallelize-loops-across-instances | FileCheck %s
```

**Round-trip RUN line for output legality:**
```
// RUN: scheduler %s -parallelize-loops-across-instances | scheduler | FileCheck %s
```

### Test inventory

**Positive cases (rewrite happens):**

| File | Shape |
|---|---|
| `basic-1d-balanced.mlir` | One pipeline, corelet-local `applicable_units`, scope with one parallel loop, trip count `c4`, num_instances `2`. Verify `scf.for` becomes `ktdf.parallel (%iv, %inst) = (%c0) to (%c4) step (%c1) distribute(num_instances = 2)`. |
| `basic-1d-imbalanced-fallback.mlir` | Trip count `c5` (not divisible by 2). Pass 1 finds nothing; Pass 2 accepts. Output uses `to (%c5)`. |
| `outermost-balanced-wins.mlir` | Two parallel loops in scope, both divisible. Outer trip `c4`, inner trip `c8`. Outer becomes `ktdf.parallel`; inner remains `scf.for` inside the parallel body. |
| `inner-divisible-when-outer-not.mlir` | Outer trip `c5` (not divisible), inner trip `c8` (divisible). Pass 1 walks outermost-first, skips outer, accepts inner. Output: outer remains `scf.for`, inner becomes `ktdf.parallel`. |
| `tmp-mlir-shape.mlir` | Replays the `test/Transform/tmp.mlir` shape: 5-stage outer pipeline with applicable_units L3LU / [SFP,LXLU,LXSU] / L3SU. Only the corelet-local middle stage's contained pipeline gets transformed. |

**Negative cases (no rewrite, IR unchanged):**

| File | Cause |
|---|---|
| `pipeline-no-stages.mlir` | Pipeline contains only `ktdf.private`, no stages — effective applicable_units is empty. |
| `non-corelet-unit.mlir` | A stage's `applicable_units = [L3LU]` (or `[SFP, L3LU]`). The pipeline's union includes a non-corelet unit. |
| `no-enclosing-scope.mlir` | Pipeline at function root, no `scf.for` parents. |
| `no-parallel-loop.mlir` | Scope has only `reduction_loop`-annotated loops or no annotation. |
| `dynamic-trip-count.mlir` | Loop bounds depend on a `func.func` argument. |
| `trip-count-too-small.mlir` | Trip count `c1` (less than `num_instances`). |
| `loop-with-sibling-ops.mlir` | A loop in scope has a sibling op next to the chain to the pipeline; scope walk truncates there. Pipeline has no qualifying scope. |

**Fatal-error cases (pass aborts via `report_fatal_error`):**

| File | Cause |
|---|---|
| `fatal-stage-without-applicable-units.mlir` | A stage in a pipeline being analyzed lacks the `applicable_units` attribute. Run with `not scheduler ... 2>&1 \| FileCheck %s --check-prefix=FATAL` to assert the diagnostic message. |

**Output legality:**

| File | Purpose |
|---|---|
| `output-round-trip.mlir` | Round-trip via the second RUN line above. |
| `output-token-flow-preserved.mlir` | Verify `depends_in`/`depends_out` chains around the transformed loop are byte-identical to input. |

**Idempotence:**

| File | Purpose |
|---|---|
| `idempotent.mlir` | `// RUN: scheduler %s -parallelize-loops-across-instances -parallelize-loops-across-instances \| FileCheck %s` — second run finds no candidates (the new `ktdf.parallel` is not an `scf.for`); verifies no spurious additional rewrites. |

**Robustness:**

| File | Purpose |
|---|---|
| `robust-empty-pipeline.mlir` | Pipeline with only `ktdf.private`, no stages. Should not crash; should be skipped. |

**Debug log:**

| File | Purpose |
|---|---|
| `debug-balanced-candidate.mlir` | `// RUN: scheduler %s -parallelize-loops-across-instances -debug-only=parallelize-loops-across-instances 2>&1 \| FileCheck %s --check-prefix=DEBUG`. Verify "balanced candidate" log line. |
| `debug-imbalanced-fallback.mlir` | Same RUN, verify "imbalanced (fallback): trip=N, num_instances=M, chunks=A,B" log line. |

### Validation criteria for shipping
- All positive cases produce the expected `ktdf.parallel` shape.
- All negative cases leave the IR unchanged.
- `output-round-trip.mlir` passes (output is structurally legal).
- `idempotent.mlir` shows no second-pass changes.

---

## 9. Open Items

1. *PE / PT units.* Today the `ComponentType` enum only enumerates `SFP, LXLU, LXSU, L3LU, L3SU` — `PE` and `PT` (mentioned in `skills/HardwareModel.md`) are not yet defined. When they're added, `isPerCoreletUnit` should return `true` for them. This pass does not need to act today; it's a forward-compatibility note.
