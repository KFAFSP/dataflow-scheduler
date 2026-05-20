# BroadcastPromotion Pass — Design

**Status:** design, pre-implementation
**Date:** 2026-05-12
**Scope:** First of two reuse-promotion passes. Handles single-transfer broadcast-style hoists only. ReductionPromotion and FifoReusePromotion are separate follow-up passes that will share the same utility layer.

---

## 1. Architecture & Scope

BroadcastPromotion identifies `ktdf.data_transfer` ops inside `ktdf.stage` bodies whose source *and* destination indices/sizes are invariant with respect to at least one enclosing `scf.for` induction variable, and hoists them outward as a sibling `ktdf.pipeline` placed before the invariant loop. This eliminates redundant data movement across iterations of that loop.

### Scope boundary

BroadcastPromotion handles only single-transfer, hoist-up (broadcast-style) promotions. The following are **out of scope** for this pass and are handled by separate passes built on the same utility layer:

- **ReductionPromotion** — paired pre-load/post-store hoists (one hoists up, one hoists down) for reduction-style transfers where the *destination* buffer is invariant under the reduction IV. Must hoist the pair together to preserve semantics.
- **FifoReusePromotion** — broadcast-style hoists where the destination is a `!ktdf.fifo.slot<...>`. These require co-hoisting the matching `ktdf.fifo.allocate`, co-moving the consumer-side `ktdf.data_transfer` that reads the slot, and expanding compute-register-level memory (SFP_REG) to support value persistence across iterations — a consumer-side rewrite significantly larger than data-transfer hoisting alone.

No grouping of multiple hoisted transfers — each hoist emits its own sibling pipeline. Grouping can be added later as an optimization if profiling motivates it.

### Input assumptions

Any structurally-legal KTDF IR with `ktdf.pipeline`/`ktdf.stage`/`ktdf.data_transfer`, at any point in the scheduler pipeline (pre- or post-stage-coarsening, flat or hierarchical). Function arguments in DRAM are assumed non-aliasing.

*TODO: introduce a `ktdf.noalias` (or similar) function-argument attribute so the compiler can verify rather than assume.*

### Output

Semantically equivalent KTDF IR where redundant transfers have been hoisted. Structurally legal by construction:

- New sibling pipelines contain a single `ktdf.stage` with `depends_in(none) depends_out(none)`.
- No `ktdf.private` inside sibling pipelines (empty-deps → no tokens needed, so Legalizer Rule 3 does not fire).
- No stage dependencies cross pipeline boundaries.
- Destination buffers for hoisted transfers are allocated at the hoist-target scope.
- Consumer subscripts for the promoted value are byte-identical to the input — only the memref SSA value is remapped.

### Repository layout

```
src/Transform/
  Passes.td                             # + def BroadcastPromotionPass
  Passes.h                              # + createBroadcastPromotionPass()
  BroadcastPromotion.cpp                # pass entry + driver loop
  ReusePromotion/Common/
    PipelineScope.{cpp,hpp}
    InvarianceCheck.{cpp,hpp}
    WriteSetScan.{cpp,hpp}
    RegionClonePrune.{cpp,hpp}
    CandidateSearch.{cpp,hpp}

test/Transforms/
  BroadcastPromotion/
    basic-broadcast-single-iv.mlir
    basic-broadcast-multiple-ivs.mlir
    broadcast-add-dram-to-lx.mlir
    broadcast-with-ssa-deps.mlir
    broadcast-with-scf-if.mlir
    broadcast-multiple-transfers-separate-pipelines.mlir
    broadcast-nested-pipelines.mlir
    broadcast-fixed-point-iteration.mlir
    no-hoist-source-depends-on-iv.mlir
    no-hoist-destination-depends-on-iv.mlir
    no-hoist-source-written-in-scope.mlir
    no-hoist-destination-written-in-scope.mlir
    no-hoist-source-alloc-inside-target-loop.mlir
    no-hoist-fifo-destination.mlir
    no-hoist-non-scf-for-parent.mlir
    no-hoist-stage-parent-boundary.mlir
    output-legal-sibling-pipeline-shape.mlir
    output-consumer-rewiring.mlir
    output-round-trip.mlir
    idempotent-already-promoted.mlir
    robust-empty-stage.mlir
    robust-pipeline-no-stages.mlir
```

### Pass invocation

Run via the `-broadcast-promotion` flag. May be composed in the scheduler pipeline alongside other passes, or run standalone. Idempotent: a second invocation on already-promoted IR performs no further hoists.

### Pipeline enclosing scope

The "pipeline enclosing scope" of a `ktdf.pipeline` is the chain of enclosing ops in which hoisting is allowed. Walk outward from the pipeline through enclosing ops:

- If the parent is an `scf.for` whose body contains only the chain leading back to the pipeline (no sibling ops beside that chain), extend the scope up through that loop and continue.
- Stop at: a non-`scf.for` parent, a `ktdf.stage` ancestor, or `func.func`.

The result is a list of enclosing `scf.for`s from innermost to outermost. Hoisting can target any of these (bounded by additional per-transfer legality).

### Invariance predicate (strict)

A `ktdf.data_transfer` is *invariant with respect to IV `%iv`* iff **neither its source nor its destination indices/sizes transitively depend on `%iv`** via SSA use-def. This rule applies symmetrically to broadcast and reduction categories; the difference between the categories lies in the hoist *direction*, not the invariance predicate.

---

## 2. Components & Interfaces

### Pass entry (`src/Transform/BroadcastPromotion.cpp`)

```cpp
namespace ktdf {
std::unique_ptr<mlir::Pass> createBroadcastPromotionPass();
}  // namespace ktdf
```

The pass is registered as `Pass<"broadcast-promotion", "mlir::ModuleOp">` in `Passes.td`. `runOnOperation()` drives a fixed-point loop:

```cpp
void runOnOperation() override {
  mlir::ModuleOp module = getOperation();
  bool changed = true;
  while (changed) {
    changed = tryHoistOneBroadcast(module);
  }
}
```

`tryHoistOneBroadcast` walks pipelines in post-order, finds the first hoistable `ktdf.data_transfer`, performs one hoist, and returns whether work was done. one hoist per iteration, re-scan after each — every iteration of the fixed point sees fresh, consistent IR. Please note that hoisting a tranfer can influence another transfer as the pipeline scoping gets effected with the introduction of sibling pipelines. Another approach is to consider all hoistable transfers, collectively perform transformation for each transfer which can increase the complexity of the pass and is deferred to future as need arises.

### Shared utilities (namespace `ktdf::reuse`)

Each utility is a small, focused unit with a narrow interface. All live under `src/Transform/ReusePromotion/Common/` so ReductionPromotion and FifoReusePromotion can reuse them.

#### `PipelineScope`

```cpp
struct PipelineEnclosingScope {
  llvm::SmallVector<mlir::scf::ForOp> loops;  // innermost to outermost
  mlir::Operation *outermost_op;              // the outermost scf.for in the
                                              // chain, or the pipeline itself
                                              // if no qualifying scf.for.
};

PipelineEnclosingScope computePipelineEnclosingScope(ktdf::PipelineOp pipeline);
```

#### `InvarianceCheck`

```cpp
// True iff `v` transitively depends on `iv` through SSA use-def chains.
bool ssaDependsOn(mlir::Value v, mlir::Value iv);

// True iff neither source indices/sizes nor destination indices/sizes of
// `transfer` transitively depend on `iv`.
bool transferIsInvariantWrt(ktdf::DataTransferOp transfer, mlir::Value iv);
```

Implementation uses `DataTransferOp`'s existing accessors (`getMixedSourceSizes`, source/dest indices, etc.) to enumerate the SSA operands to check.

#### `WriteSetScan`

```cpp
// Returns true if any op inside `region` writes through `memref` (via
// memref.store, linalg ops with write outputs, ktdf.data_transfer with
// `memref` as destination, scf.if / scf.for bodies that contain such
// writes, etc.). Aliasing-capable ops (memref.view/subview/cast) are
// followed conservatively. Function-arg memrefs are assumed non-aliasing
// with each other. `ignore` is excluded from the scan (used to exclude
// the transfer being hoisted when checking its own destination).
bool regionWritesTo(mlir::Region &region,
                    mlir::Value memref,
                    mlir::Operation *ignore = nullptr);
```

#### `RegionClonePrune`

```cpp
// Clone `source_region` into `target_block` at `ip`, then erase every op
// that is not the image of `keep_anchor` and not needed (transitively or
// structurally) by it. Returns the cloned anchor op.
mlir::Operation *cloneRegionAndPruneToAnchor(
    mlir::Region &source_region,
    mlir::Block *target_block,
    mlir::Block::iterator ip,
    mlir::Operation *keep_anchor,
    mlir::IRMapping &value_map);
```

Implements the "clone everything, then prune" approach. The set of ops kept is: the cloned anchor, its transitive operand producers, and the enclosing `scf.for`/`scf.if` structural skeleton required to position the anchor. Empty `scf.for` bodies and empty `scf.if` branches retain their terminators (`scf.yield`) so the result is structurally legal.

#### `CandidateSearch`

```cpp
struct Candidate {
  ktdf::DataTransferOp transfer;
  ktdf::StageOp donor_stage;
  mlir::scf::ForOp target_loop;   // outermost legal hoist destination
};

std::optional<Candidate> findFirstCandidate(ktdf::PipelineOp pipeline);
```

Integrates invariance, write-set, and structural checks (see Section 3).

### Pass-internal helpers (in `BroadcastPromotion.cpp`)

```cpp
bool tryHoistOneBroadcast(mlir::ModuleOp module);
void performHoist(const ktdf::reuse::Candidate &c);
```

---

## 3. Data Flow

This section describes end-to-end execution on input IR.

### Step 1 — Walk + integrated legality (candidate search)

Post-order walk over `ktdf.pipeline` ops in the module, so innermost pipelines are visited before their enclosing pipelines. For each pipeline:

1. Compute `PipelineEnclosingScope scope`. If `scope.loops` is empty, no candidates are possible — continue.

2. For each `ktdf.stage` in the pipeline (source order), for each `ktdf.data_transfer` in the stage body (source order): determine the outermost scf.for `target_loop` in `scope.loops` such that *all* of the following hold. Checks proceed innermost to outermost and stop at the first IV that fails any condition — the previous IV (if any) is the outermost legal target.

   **Legality conditions (all must hold):**

   (a) **Invariance** — `transferIsInvariantWrt(transfer, iv)` is true.

   (b) **FIFO destination excluded** — `transfer.getDest().getType()` is not a `ktdf::FifoSlotType`. (FIFO-destination hoists are handled by `FifoReusePromotion` — see TODO.)

   (c) **Source not overwritten in hoist region** — `regionWritesTo(target_loop.getRegion(), transfer.getSource()) == false`.

   (d) **Destination not overwritten in hoist region** — `regionWritesTo(target_loop.getRegion(), transfer.getDest(), /*ignore=*/transfer) == false`.

   (e) **Source allocation dominates target position** — `transfer.getSource()`'s defining op is not located inside `target_loop`'s region. Trivially holds for function-arg sources and for values defined above `target_loop`.

   (f) **Destination allocation can be cloned above `target_loop`** — the original destination `memref.alloc`'s type (and any dynamic-size operands) allow a clone at the target position. Always true for the scalar-size memrefs used in KTDF.

3. Return the first candidate found (innermost pipeline, first stage, first transfer, outermost legal `target_loop`). Return `std::nullopt` if no candidate exists anywhere in the module.

### Step 2 — Emission via region cloning + pruning

With `Candidate {transfer, donor_stage, target_loop}` in hand, `performHoist` does:

1. **Position builder** immediately before `target_loop` in `target_loop`'s parent block. All subsequent emission happens in order at this position.

2. **Allocate fresh destination buffer.** Clone the original destination's `memref.alloc` op at the current builder position. The clone has the same memref type as the original. Record `{original_dest_value -> new_alloc_result}` in a fresh `IRMapping value_map`.

3. **Create sibling pipeline.** Emit `ktdf.pipeline { ktdf.stage depends_in(none) depends_out(none) { ... } }` at the current position. Position builder inside the new stage body.

4. **Clone donor stage body + prune.** Call `cloneRegionAndPruneToAnchor(donor_stage.getBody(), new_stage_body, ip, transfer, value_map)`. This:
   - Clones every op from the donor stage body into the new stage body in source order, populating `value_map` for all cloned SSA values.
   - Identifies the cloned image of `transfer` as the anchor.
   - Computes the "kept set": the anchor, its transitive operand producers (via SSA use-def), the enclosing `scf.for`/`scf.if` structural skeleton required to position the anchor, and the terminators of any region whose body remains (even if pruned to empty).
   - Erases every cloned op not in the kept set, in reverse-topological order (users before defs) so that erasures do not leave dangling uses.
   - Preserves terminators of emptied regions (bare `scf.yield` etc.).

5. **Rewire consumers.** For every use of `transfer.getDest()` in the module *outside* the donor stage body and *outside* the original transfer op, replace with the new alloc's result. In-donor uses stay pointed at the original alloc; if they become dead after the transfer is erased (Step 6), DCE cleans them up.

6. **Erase original transfer.** Remove the original `ktdf.data_transfer` op from the donor stage body. The donor stage's structural skeleton (enclosing loops/ifs inside the stage) is retained — DCE cleans up any newly-dead loops/ifs later.

### Step 3 — Return to driver, re-scan

`performHoist` returns; `tryHoistOneBroadcast` returns `true`; the outer fixed-point loop re-scans the mutated module for the next candidate. When no candidate is found, the pass exits.

### Worked example (pre-stage-coarsening, memref-destination only)

Starting from the shape of `test/Transforms/StageCoarsening/basic-fifo.mlir` (syntax reference only), consider a variant where tensor A's DRAM→LX transfer is invariant in the inner loop `%n`:

```mlir
scf.for %m = %c0 to %M step %c1 {
  scf.for %n = %c0 to %N step %c64 {
    ktdf.pipeline {
      %lxA, %t = ktdf.private -> (memref<64xf16, #ktdf.memory_space<LX>>, !ktdf.token) {
        %a = memref.alloc() : memref<64xf16, #ktdf.memory_space<LX>>
        %k = ktdf.create_token : !ktdf.token
        ktdf.private_yield %a, %k : memref<64xf16, #ktdf.memory_space<LX>>, !ktdf.token
      }
      ktdf.stage depends_in(none) depends_out(%t#1 : !ktdf.token) {
        ktdf.data_transfer from %A[%m] size [64] to %lxA[%c0] size [64]
          : memref<?xf16, #ktdf.memory_space<HBM>>, memref<64xf16, #ktdf.memory_space<LX>>
        // ...other transfers using %n
      }
      // ...downstream stages consuming %lxA[%c0]
    }
  } {loop_type = #ktdf.loop_type<parallel_loop>}
}
```

The A transfer depends on `%m` but not `%n`. The pass finds:

- Candidate: `{A-transfer, stage, target_loop=scf.for %n}`.
- Emits before `scf.for %n`: a fresh `memref.alloc()` (new `%lxA_promoted`), then a sibling `ktdf.pipeline { ktdf.stage depends_in(none) depends_out(none) { ktdf.data_transfer from %A[%m] size [64] to %lxA_promoted[%c0] size [64] : ... } }`.
- Rewires downstream consumers' reads of `%lxA` (outside the donor stage) to `%lxA_promoted`.
- Erases the original A transfer from the donor stage.

---

## 4. Legality & Error Handling

### Legality conditions (consolidated)

For a candidate `{transfer, donor_stage, target_loop}`, all of (a)–(f) from Section 3 Step 1 must hold. The pass searches for the *outermost* `target_loop` where all conditions hold simultaneously.

### Structural invariants preserved by construction

- Every `ktdf.stage` has a `ktdf.pipeline` parent (sibling pipeline is well-formed; donor stage is unchanged).
- Every new `ktdf.pipeline`'s children are `ktdf.stage` — no `ktdf.private` needed since the sibling stage has empty deps.
- Every new `ktdf.stage` has `depends_in(none) depends_out(none)`.
- `scf.for` / `scf.if` bodies retain their terminators after pruning.
- Memref types preserved: the fresh alloc's type matches the original exactly; consumer subscripts and the hoisted transfer's own subscripts are byte-identical to the input.

### Rejection behavior

Rejected candidates are not errors. The pass is a best-effort optimization. If no transfer is hoistable, the pass exits cleanly with no changes.

### Verifier and diagnostics

The pass produces structurally legal IR after each hoist. No internal verifier call is needed beyond well-formed construction; the MLIR verifier catches any structural regression at pass-pipeline end.

### Debug logging

Under `LLVM_DEBUG(...)` with debug tag `broadcast-promotion`, the pass emits:
- For each candidate considered: donor stage location, transfer source/dest descriptors, IV chain checked, outermost legal IV found (or "rejected: reason").
- For each hoist performed: donor stage location, target loop location, new sibling pipeline location.
- On fixed-point exit: total hoists performed.

Invoked via `-debug-only=broadcast-promotion`.

### Debug-build assertions

- `assert(target_loop && "candidate must have a valid target_loop")` in `performHoist`.
- `assert(value_map.contains(orig_dest))` after fresh-alloc step, before cloning.
- `assert(cloned_transfer && "pruning must preserve the anchor")` after `cloneRegionAndPruneToAnchor`.

### Edge cases

| Case | Behavior |
|---|---|
| Pipeline has no enclosing `scf.for` | `computePipelineEnclosingScope` returns empty loop list. No candidates. |
| Transfer has no enclosing loops inside donor stage | Allowed. Pruning reduces to "just the transfer" in the cloned stage. |
| Transfer's source is a function argument | `regionWritesTo` reports false (no writes to func args inside scope, by non-aliasing assumption). Legal to hoist. |
| Multiple hoistable transfers in one stage | Each becomes its own sibling pipeline (no grouping). |
| Same transfer could hoist past multiple IVs, but only some are legal | `findFirstCandidate` returns the outermost legal IV. Subsequent iterations do not revisit. |
| Destination is a `!ktdf.fifo.slot<...>` | Skipped at candidate search. Handled by `FifoReusePromotion` (TODO). |
| Pipeline sits inside `scf.if`, `scf.parallel`, `scf.while` | Scope walk caps at the non-`scf.for` parent. Hoisting beyond is not attempted. |
| Pipeline nested inside an outer stage body | Scope walk stops at the outer stage boundary. No escape into the outer pipeline. |
| Source allocation inside `ktdf.private` | Allowed, but hoist target cannot be above the donor pipeline itself (private is pipeline-scoped). Condition (e) enforces this. |

---

## 5. Testing Plan

### Framework

LLVM `lit` + `FileCheck`, consistent with the rest of the repo. Tests under `test/Transforms/BroadcastPromotion/`. Driver tool: `build/scheduler/bin/scheduler`. Syntax follows `test/Transforms/StageCoarsening/basic-fifo.mlir` as the authoritative reference — not the stale `docs/MatrixAdd.md`.

Typical RUN line:
```
// RUN: scheduler %s -broadcast-promotion | FileCheck %s
```

Debug-log inspection:
```
// RUN: scheduler %s -broadcast-promotion -debug-only=broadcast-promotion 2>&1 | FileCheck %s --check-prefix=DEBUG
```

### Syntax conventions (from `basic-fifo.mlir`)

- Empty token lists: `depends_in(none) depends_out(none)`.
- Token-typed deps: `depends_in(%t : !ktdf.token)`.
- Memory spaces: `#ktdf.memory_space<LX>`, `<HBM>`, `<SFP_REG>`.
- Loop attributes: `{loop_type = #ktdf.loop_type<parallel_loop>}`.
- Pipeline-private resources via `ktdf.private -> (...) { ... ktdf.private_yield ... }`.
- No `depth(...)`, no `modulo(...)`, no `stage_yield` — these are artifacts of the stale MatrixAdd.md document.
- Sibling pipelines emitted by BroadcastPromotion use `ktdf.pipeline { ktdf.stage depends_in(none) depends_out(none) { ... } }` with no `ktdf.private` inside.

### Test inventory

**Positive cases (hoist happens):**

1. `basic-broadcast-single-iv.mlir` — Minimal kernel: one pipeline, one stage, one transfer invariant w.r.t. the innermost enclosing loop. Verify sibling pipeline appears before the loop, original transfer removed, destination alloc cloned at target scope.

2. `basic-broadcast-multiple-ivs.mlir` — Transfer invariant w.r.t. two enclosing loops. Verify hoisted to the outermost legal scope.

3. `broadcast-add-dram-to-lx.mlir` — Canonical broadcast-add DRAM→LX hoist (tensor A invariant in N-dimension). Written in `basic-fifo.mlir` syntax. Verify sibling pipeline at the M-scope, donor stage cleaned up, consumers still reference `%lxA[...]` with unchanged subscripts.

4. `broadcast-with-ssa-deps.mlir` — Transfer depends on an intra-stage SSA value. Verify SSA dep is cloned into the sibling pipeline's stage body.

5. `broadcast-with-scf-if.mlir` — Transfer inside an `scf.if` branch. Verify the `scf.if` skeleton is cloned; transfer sits in the matching branch; other branch preserved (possibly empty) with terminator intact.

6. `broadcast-multiple-transfers-separate-pipelines.mlir` — Two independent hoistable transfers in one stage. Verify two separate sibling pipelines (no grouping).

7. `broadcast-nested-pipelines.mlir` — Hierarchical pipeline (post-stage-coarsening shape). Inner pipeline has hoistable transfer targeting a loop inside the outer stage. Verify sibling pipeline materialized inside the outer stage body, respecting the stage-as-scope-boundary rule.

8. `broadcast-fixed-point-iteration.mlir` — Two hoistable transfers where the second becomes visible only after the first is performed. Verify fixed-point iteration performs both hoists.

**Negative cases (no hoist performed):**

9. `no-hoist-source-depends-on-iv.mlir` — Source indices depend on every enclosing IV. IR unchanged.

10. `no-hoist-destination-depends-on-iv.mlir` — Source invariant, destination indices depend on IV. Under strict rule, both must be invariant. IR unchanged.

11. `no-hoist-source-written-in-scope.mlir` — Source invariant, but another op in the target loop's region writes the source memref. IR unchanged.

12. `no-hoist-destination-written-in-scope.mlir` — Destination is also written by another op (not the transfer itself) in the target loop's region. IR unchanged.

13. `no-hoist-source-alloc-inside-target-loop.mlir` — Source memref's defining alloc sits inside the loop we'd hoist past. IR unchanged.

14. `no-hoist-fifo-destination.mlir` — Transfer's destination is `!ktdf.fifo.slot<...>`. Candidate search skips. IR unchanged.

15. `no-hoist-non-scf-for-parent.mlir` — Pipeline inside `scf.if` or `scf.parallel`. Scope walk caps at the non-scf.for parent. IR unchanged.

16. `no-hoist-stage-parent-boundary.mlir` — Pipeline inside an outer stage. Hoist cannot escape the outer stage. IR unchanged beyond the outer stage boundary.

**Structural legality of output:**

17. `output-legal-sibling-pipeline-shape.mlir` — Verify sibling pipeline has exactly one stage with `depends_in(none) depends_out(none)`, no `ktdf.private`.

18. `output-consumer-rewiring.mlir` — Verify consumer reads the new outer-scope alloc; subscripts are unchanged.

19. `output-round-trip.mlir` — `// RUN: scheduler %s -broadcast-promotion | scheduler | FileCheck %s` — verifies the output parses cleanly and is structurally legal.

**Idempotence:**

20. `idempotent-already-promoted.mlir` — `// RUN: scheduler %s -broadcast-promotion -broadcast-promotion | FileCheck %s` — second invocation performs no additional work.

**Debug output:**

21. `debug-log-hoist.mlir` — Verify debug log contains candidate location, target loop, and "performed hoist" markers.

**Robustness:**

22. `robust-empty-stage.mlir` — Stage with only an empty body. No crash, no hoist.

23. `robust-pipeline-no-stages.mlir` — Pipeline with only `ktdf.private`. No crash, no hoist.

### Unit tests

The repo is lit-based; utility-level testing happens indirectly via integration tests. If a C++ unit test harness is added later, natural targets are `computePipelineEnclosingScope`, `transferIsInvariantWrt`, `regionWritesTo`, and `cloneRegionAndPruneToAnchor`.

### Validation for v1 completion

All positive cases (1–8) pass, all negative cases (9–16) verify no change, round-trip (19) succeeds, idempotence (20) holds. Debug log (21) and robustness (22–23) tests are nice-to-have but not blocking.

---

## 6. TODOs for Follow-up Passes

The following items are deliberately out of scope for BroadcastPromotion v1. They are parked for the passes noted, which will share the utility layer introduced here.

1. **Control-flow conditionals around pre-load/post-store pairs in reduction** — legality reasoning under `scf.if` guards like `%is_first`. *(For ReductionPromotion.)*

2. **Function-argument non-aliasing attribute** — introduce `ktdf.noalias` (or similar) so non-aliasing is expressible and checkable rather than assumed. *(Global TODO.)*

3. **Reduction slicing is more than transfer + loops + ifs + SSA deps** — includes pair identification, conditional rewriting across the pair, guard simplification when the pair hoist unconditionalizes the pre-load. *(For ReductionPromotion.)*

4. **Broadcast's "destination not overwritten in hoist scope" rule explicitly excludes reduction candidates** — ReductionPromotion's legality check generalizes to pairs, reasoning about the accumulator read+write pattern. *(For ReductionPromotion.)*

5. **FifoReusePromotion** — separate pass for broadcast-style hoists where the destination is a FIFO slot. Must also hoist `ktdf.fifo.allocate` along with the transfer, co-move the matching FIFO-read consumer, and expand SFP_REG memory at the compute-register level to support value persistence across iterations. Significantly larger than simple data-transfer hoisting. Shares `PipelineScope`, `InvarianceCheck`, `WriteSetScan`, `RegionClonePrune`, `CandidateSearch` utilities with BroadcastPromotion.
