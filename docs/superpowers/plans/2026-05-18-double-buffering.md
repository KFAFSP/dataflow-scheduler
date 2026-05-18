# `-double-buffering` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a new MLIR pass `-double-buffering` that detects scratchpad memrefs handed off between sibling stages of a `ktdf.pipeline` (one stage writes via `ktdf.data_transfer`, one or more sibling stages read via `ktdf.data_transfer`) and rewrites them into a ping-pong form: paired allocs hoisted just outside the pipeline scope, a shared `ktdf.buffer_phase` plus per-buffer `ktdf.select_memref` inserted before the pipeline, and `modulo(size: 2)` set on the pipeline.

**Architecture:** The pass walks pipelines pre-order. For each pipeline, it computes the enclosing scope (reusing `ktdf::common::computePipelineEnclosingScope`); rejects pipelines with no scope loops or with a pre-existing `modulo`; finds candidate buffers in the pipeline's `ktdf.private` region (eligibility = single-use `memref.alloc` of an eligible memory space, dynamic sizes dominate the hoist target, exactly one producer sibling stage, ≥1 consumer sibling stage, producer ≠ any consumer, producer transitively precedes every consumer); then applies the transformation per-pipeline (hoist+dealloc each candidate, insert one shared `buffer_phase`, insert per-candidate `select_memref`s, RAUW uses inside the pipeline, set `modulo`, prune the dropped slots from `ktdf.private` and erase orphaned allocs). Three new IR constructs are added to the KTDF dialect (`ktdf.buffer_phase`, `ktdf.select_memref`, optional `modulo` attribute on `ktdf.pipeline`) and one new method on `SchedulerExtContext` (`isEligibleForDoubleBuffering`).

**Tech Stack:** MLIR (TableGen ODS for dialect + pass), C++17, CMake + Ninja, lit + FileCheck.

**Spec reference:** `docs/superpowers/specs/2026-05-18-double-buffering-design.md`. Spec sections cited inline as `spec §N`.

**Pre-flight (read once before starting):**
- `docs/superpowers/specs/2026-05-18-double-buffering-design.md` — the spec.
- `docs/Pipeline.md` Step 9 (lines 524–670) — canonical end-state IR.
- `src/Dialects/KTDF/KTDFOps.td` — existing op patterns. `KTDF_PipelineOp` (around line 347) is where the optional `modulo` attribute is added; `KTDF_FifoAllocateOp`, `KTDF_DataTransferOp` show variadic-operand assemblyFormat patterns.
- `src/Dialects/KTDF/KTDFAttributes.td` — existing attribute patterns. `KTDF_MemorySpaceCustomAttr` and `KTDF_ApplicableUnitsCustomAttr` show the `KTDF_Attr` template style.
- `src/Dialects/KTDF/KTDFOps.cpp` — verifier patterns. `PipelineOp::verify` (line 175), `PrivateOp::verify` (line 367), `StageOp::verify` and helpers (lines 290–357) are the closest templates for what we'll add.
- `src/Transform/Common/PipelineScope.{hpp,cpp}` — the scope-walk utility (do not modify).
- `src/Transform/ParallelizeLoopsAcrossInstances.cpp` — the closest pass template (analysis + transform in one file, takes `SchedulerExtContext`, debug logging).
- `src/Transform/Passes.{td,h}` — pass registration patterns. `ParallelizeLoopsAcrossInstancesPass` (Passes.td line 129) is the template.
- `src/Utils/SchedulerExtContext.{h,cpp}` — context being extended.
- `tools/scheduler/scheduler-pipeline.cpp` — driver; new pass added to `buildKTDPToDFIRPipeline`.
- `test/Dialects/KTDF/private-ops.mlir`, `test/Dialects/KTDF/parallel-op.mlir` — round-trip test format.
- `test/Transform/ParallelizeLoopsAcrossInstances/` — transform test directory layout (positive, negative, idempotent, debug-log, round-trip).
- `docs/knowledge-base/project_coding_conventions.md` — Google C++ style; snake_case locals; trailing `_` for class members; camelBack methods/free functions; `kFoo` constants; brace short single-statement bodies.

**Build & test commands:**
- Build: `ninja -C build`. Configure first if CMake changed: `./scripts/buildKTIRScheduler.sh && ninja -C build`.
- Full lit suite: `ninja -C build check-scheduler`.
- Single test: `build/scheduler/bin/scheduler -allow-unregistered-dialect <flag> path/to/test.mlir | /Users/prasanth-work/Documents/code/external/llvm-standalone-3/build/bin/FileCheck path/to/test.mlir`.
- Inspect raw printer output: `build/scheduler/bin/scheduler <flag> path/to/test.mlir`.
- Verify-diagnostics file: `build/scheduler/bin/scheduler -allow-unregistered-dialect -split-input-file -verify-diagnostics path/to/test.mlir`.

**Conventions (apply throughout):**
- Local vars: snake_case (`scope_loops`, `num_phases`, `candidate_buffers`).
- Free functions and class methods: camelBack (`isEligibleForDoubleBuffering`, `findCandidates`, `rewritePipeline`).
- Class members: trailing `_` (`scheduler_ctx_`, `num_phases_`).
- Constants: `kFoo` (`kDefaultNumPhases = 2`).
- Brace short single-statement `if`/`for` bodies.
- Pass `#define DEBUG_TYPE "double-buffering"` for `LLVM_DEBUG` logging.
- All new code lives under `namespace ktdf` (pass + dialect) and `namespace scheduler` (context method); no `mlir::` prefix on `ktdf::createXxxPass`.

**Pre-state:**
- Branch `pc/doublebuffering`. HEAD: `615111e` ("Add design spec for -double-buffering pass") sitting on top of `pc/coreletparallelism`.
- `git status` may show pre-existing untracked items (`.idea/`, `docs/knowledge-base/`, `skills/`, `src/.DS_Store`, `test/.lit_test_times.txt`, `test/Transform/BroadcastPromotion/`, `test/Transform/tmp.mlir`) — leave them alone.

---

## File Structure

| Path | Action | Responsibility |
|---|---|---|
| `src/Dialects/KTDF/KTDFAttributes.td` | Modify | Add `KTDF_ModuloAttr` (struct attribute, `size : i64`). |
| `src/Dialects/KTDF/KTDFOps.td` | Modify | Add `OptionalAttr<KTDF_ModuloAttr>:$modulo` on `KTDF_PipelineOp`. Add `KTDF_BufferPhaseOp` and `KTDF_SelectMemrefOp`. |
| `src/Dialects/KTDF/KTDFOps.cpp` | Modify | Verifier extensions for `PipelineOp` (modulo size check). New verifier bodies for `BufferPhaseOp` and `SelectMemrefOp`. |
| `src/Utils/SchedulerExtContext.h` | Modify | Declare `virtual bool isEligibleForDoubleBuffering(mlir::ktdf::MemorySpace) const;`. |
| `src/Utils/SchedulerExtContext.cpp` | Modify | Implement default body returning `mspace == MemorySpace::LX`. |
| `src/Transform/Passes.td` | Modify | Add `def DoubleBufferingPass`. |
| `src/Transform/Passes.h` | Modify | Declare `createDoubleBufferingPass(const SchedulerExtContext&)` plus a no-arg overload. |
| `src/Transform/DoubleBuffering.cpp` | Create | Pass entry, detection, transformation, debug logging. |
| `src/Transform/CMakeLists.txt` | Modify | Add `DoubleBuffering.cpp` to source list. |
| `tools/scheduler/scheduler-pipeline.cpp` | Modify | Insert `createDoubleBufferingPass(scheduler_ctx)` after `createParallelizeLoopsAcrossInstancesPass` in `buildKTDPToDFIRPipeline`. |
| `test/Dialects/KTDF/buffer-phase-op.mlir` | Create | Round-trip + verifier-failure tests for `ktdf.buffer_phase`. |
| `test/Dialects/KTDF/select-memref-op.mlir` | Create | Round-trip + verifier-failure tests for `ktdf.select_memref`. |
| `test/Dialects/KTDF/pipeline-modulo-attr.mlir` | Create | Round-trip + verifier-failure tests for `ktdf.pipeline modulo(size: N)`. |
| `test/Transform/DoubleBuffering/*.mlir` | Create | Pass tests: positive, negative, idempotent, debug-log. |

---

## Task 1: Add `KTDF_ModuloAttr` (struct attribute) to the KTDF dialect

**Goal:** Define a custom struct attribute `#ktdf.modulo<size = N>` with one `i64` field, addable as an `OptionalAttr` on `ktdf.pipeline`. No op uses it yet — we just want the attribute to parse, print, and round-trip.

**Files:**
- Modify: `src/Dialects/KTDF/KTDFAttributes.td`
- Test: `test/Dialects/KTDF/pipeline-modulo-attr.mlir` (created in Task 2)

- [ ] **Step 1: Open `src/Dialects/KTDF/KTDFAttributes.td` and append the new attribute definition**

Append immediately before `#endif // KTDF_SRC_DIALECTS_KTDF_ATTRIBUTES_TD` (i.e., after `KTDF_ApplicableUnitsCustomAttr`):

```tablegen
//===----------------------------------------------------------------------===//
// ModuloAttr
//===----------------------------------------------------------------------===//

def KTDF_ModuloAttr : KTDF_Attr<"Modulo", "modulo"> {
  let summary = "KTDF pipeline rolling-window depth";
  let description = [{
    The `modulo` attribute declares the rolling-window depth of a
    `ktdf.pipeline`. A pipeline with `modulo(size: N)` rotates through `N`
    physical resource sets (e.g., double-buffered LX memrefs) per `N`
    consecutive iterations, allowing producer of iteration `k+1` and
    consumer of iteration `k` to reference different physical buffers and
    run concurrently.

    Syntax:
    ```
    modulo-attr ::= `modulo` `<` `size` `=` integer-literal `>`
    ```

    Example:
    ```mlir
    ktdf.pipeline modulo(size: 2) { ... }
    ```
  }];

  let parameters = (ins "int64_t":$size);

  let assemblyFormat = "`<` `size` `=` $size `>`";

  let genVerifyDecl = 1;
}
```

- [ ] **Step 2: Add the verifier body to `KTDFOps.cpp`**

Append at the end of `src/Dialects/KTDF/KTDFOps.cpp`, after the `ParallelOp::verify` body. Add the include for the attribute storage if needed (already included via the dialect's generated `.cpp.inc`).

```cpp
//===----------------------------------------------------------------------===//
// ModuloAttr
//===----------------------------------------------------------------------===//

::mlir::LogicalResult ModuloAttr::verify(
    ::llvm::function_ref<::mlir::InFlightDiagnostic()> emit_error,
    int64_t size) {
  if (size < 2) {
    return emit_error() << "modulo size must be >= 2, got " << size;
  }
  return ::mlir::success();
}
```

- [ ] **Step 3: Build to confirm it compiles**

```bash
ninja -C build
```

Expected: clean build. The attribute is generated under `mlir::ktdf::ModuloAttr` but is not yet attached to any op. Compilation success means the TableGen ran cleanly.

- [ ] **Step 4: Commit**

```bash
git add src/Dialects/KTDF/KTDFAttributes.td src/Dialects/KTDF/KTDFOps.cpp
git commit -m "$(cat <<'EOF'
Add ktdf.modulo attribute for pipeline rolling-window depth

Defines KTDF_ModuloAttr with a single int64 size field and a verifier
requiring size >= 2. Will be attached to ktdf.pipeline as an optional
attribute in a follow-up commit.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Attach `modulo` as an optional attribute on `ktdf.pipeline`

**Goal:** Add `OptionalAttr<KTDF_ModuloAttr>:$modulo` to `KTDF_PipelineOp`. Existing IR (no `modulo`) must round-trip identically; new IR (with `modulo`) must round-trip and reject `size < 2`.

**Files:**
- Modify: `src/Dialects/KTDF/KTDFOps.td`
- Test: `test/Dialects/KTDF/pipeline-modulo-attr.mlir`

- [ ] **Step 1: Modify `KTDF_PipelineOp` in `KTDFOps.td`**

Locate `def KTDF_PipelineOp` (around line 347 in `KTDFOps.td`). Replace the `let regions = ...` line and the `let assemblyFormat = "$body attr-dict";` line with the modulo-aware versions. The full op definition becomes:

```tablegen
def KTDF_PipelineOp : KTDF_Op<"pipeline", [
    NoTerminator,
    SingleBlock
  ]> {
  let summary = "Defines a pipeline containing stages";
  let description = [{
    The `ktdf.pipeline` operation represents a structured directed acyclic graph
    (DAG) of execution steps that process data in stages. Instead of executing one
    step completely before starting the next, pipelines allow different stages to
    operate concurrently on different iterations of work, thereby overlapping
    computation and data movement. In dataflow architectures, pipelines reflect
    how hardware engines operate: one unit may be loading data, another computing,
    and another storing results simultaneously.

    The immediate children of a pipeline must be `ktdf.stage` operations.

    The optional `modulo` attribute declares the pipeline's rolling-window depth.
    With `modulo(size: N)`, the pipeline rotates through `N` physical resource
    sets per `N` consecutive iterations.

    Example:
    ```mlir
    ktdf.pipeline modulo(size: 2) {
      ktdf.stage {
        // Stage 1 operations
      }
      ktdf.stage {
        // Stage 2 operations
      }
    }
    ```
  }];

  let arguments = (ins
    OptionalAttr<KTDF_ModuloAttr>:$modulo
  );

  let regions = (region SizedRegion<1>:$body);

  let assemblyFormat = [{
    (`modulo` `(` `size` `:` $modulo^ `)`)? $body attr-dict
  }];

  let hasVerifier = 1;

  let builders = [
    OpBuilder<(ins), [{
      Region* bodyRegion = $_state.addRegion();
      bodyRegion->emplaceBlock();
    }]>
  ];

  let extraClassDeclaration = [{
    using StageRange = llvm::iterator_range<Block::op_iterator<StageOp>>;

    /// Get the block containing the stages.
    Block *getBodyBlock() { return &getBody().front(); }

    StageRange getStages();
    unsigned getNumStages();
  }];
}
```

Note: the assembly form `modulo(size: <attr>)` uses TableGen's `^` to mark `$modulo` as the trigger for the optional group. The attribute prints/parses as just its `int64_t` value inside that wrapper, not the full `<size = N>` form (TableGen prints the parameter directly when the assembly format provides surrounding syntax).

If TableGen's optional-group printing requires the bare attribute (not its custom assembly), we may instead need a custom assembly format. As a fallback, replace the `assemblyFormat` line with:

```tablegen
  let hasCustomAssemblyFormat = 1;
```

…and add custom `parse`/`print` static members in the `extraClassDeclaration` plus their bodies in `KTDFOps.cpp`. **Try the declarative form first**; only fall back if the build fails with a TableGen error about the optional-group format.

- [ ] **Step 2: Build to confirm**

```bash
ninja -C build
```

Expected: clean build. If TableGen complains about the optional group, switch to `hasCustomAssemblyFormat = 1` and implement `PipelineOp::parse` / `PipelineOp::print` in `KTDFOps.cpp` mirroring the pattern used by `StageOp::parse` / `StageOp::print` (KTDFOps.cpp lines 194–288).

- [ ] **Step 3: Add a `getModuloSize()` helper**

Edit the `extraClassDeclaration` in `KTDF_PipelineOp` to add the helper. Replace:

```tablegen
    StageRange getStages();
    unsigned getNumStages();
```

with:

```tablegen
    StageRange getStages();
    unsigned getNumStages();

    /// Returns the modulo size if the modulo attribute is set, otherwise
    /// returns std::nullopt.
    std::optional<int64_t> getModuloSize() {
      if (auto m = getModuloAttr()) {
        return m.getSize();
      }
      return std::nullopt;
    }
```

- [ ] **Step 4: Update `PipelineOp::verify` to be unchanged**

The `PipelineOp::verify` body in `KTDFOps.cpp` (lines 175-188) already only checks immediate children. The optional `modulo` attribute's own verifier (Task 1, Step 2) handles `size >= 2`. No changes here.

- [ ] **Step 5: Write round-trip + negative test**

Create `test/Dialects/KTDF/pipeline-modulo-attr.mlir`:

```mlir
// RUN: scheduler %s | scheduler | FileCheck %s
// RUN: scheduler -split-input-file -verify-diagnostics %s

// Round-trip: pipeline without modulo (existing behavior).
// CHECK-LABEL:   func.func @pipeline_no_modulo
// CHECK:           ktdf.pipeline {
// CHECK-NEXT:        ktdf.stage depends_in(none) depends_out(none) {
// CHECK-NEXT:        }
// CHECK-NEXT:      }

func.func @pipeline_no_modulo() {
  ktdf.pipeline {
    ktdf.stage depends_in(none) depends_out(none) {
    }
  }
  return
}

// -----

// Round-trip: pipeline with modulo(size: 2).
// CHECK-LABEL:   func.func @pipeline_modulo_2
// CHECK:           ktdf.pipeline modulo(size : 2) {
// CHECK-NEXT:        ktdf.stage depends_in(none) depends_out(none) {
// CHECK-NEXT:        }
// CHECK-NEXT:      }

func.func @pipeline_modulo_2() {
  ktdf.pipeline modulo(size: 2) {
    ktdf.stage depends_in(none) depends_out(none) {
    }
  }
  return
}

// -----

// Round-trip: pipeline with modulo(size: 4) — N>2 still parses (the dialect
// op does not reject this; only the transformation pass hardcodes 2).
// CHECK-LABEL:   func.func @pipeline_modulo_4
// CHECK:           ktdf.pipeline modulo(size : 4) {

func.func @pipeline_modulo_4() {
  ktdf.pipeline modulo(size: 4) {
    ktdf.stage depends_in(none) depends_out(none) {
    }
  }
  return
}

// -----

// Negative: modulo size < 2.
func.func @pipeline_modulo_too_small() {
  // expected-error @+1 {{modulo size must be >= 2}}
  ktdf.pipeline modulo(size: 1) {
    ktdf.stage depends_in(none) depends_out(none) {
    }
  }
  return
}
```

If your `assemblyFormat` for the optional group prints `modulo(size : N)` with a space, the `CHECK` lines above are correct. If it prints `modulo(size: N)`, change those `CHECK` lines accordingly. Run the file once after build to discover the actual printer output, then commit the matching expectations.

- [ ] **Step 6: Run the test**

```bash
build/scheduler/bin/scheduler test/Dialects/KTDF/pipeline-modulo-attr.mlir | /Users/prasanth-work/Documents/code/external/llvm-standalone-3/build/bin/FileCheck test/Dialects/KTDF/pipeline-modulo-attr.mlir
build/scheduler/bin/scheduler -split-input-file -verify-diagnostics test/Dialects/KTDF/pipeline-modulo-attr.mlir
```

Expected: both pass. If the printer's spacing differs, regenerate the `CHECK` lines from `build/scheduler/bin/scheduler test/Dialects/KTDF/pipeline-modulo-attr.mlir` output and re-commit.

- [ ] **Step 7: Run the full lit suite to confirm no existing test broke**

```bash
ninja -C build check-scheduler
```

Expected: all green. Existing tests that don't set `modulo` on `ktdf.pipeline` must still parse and print identically (no `modulo(size: ...)` injected by the printer when the attribute is absent).

- [ ] **Step 8: Commit**

```bash
git add src/Dialects/KTDF/KTDFOps.td src/Dialects/KTDF/KTDFOps.cpp test/Dialects/KTDF/pipeline-modulo-attr.mlir
git commit -m "$(cat <<'EOF'
Attach optional modulo attribute to ktdf.pipeline

Existing pipelines without modulo continue to round-trip identically.
With modulo(size: N), the pipeline declares a rolling-window depth of
N. The dialect op accepts any N >= 2; the upcoming -double-buffering
pass uses N = 2.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Add `ktdf.buffer_phase` op

**Goal:** Define a new op `ktdf.buffer_phase(%iv1, %iv2, ...) {num_phases = N : i64} : index` with verifier rules from spec §2.1: variadic `index` operands (≥1), `num_phases >= 2`, each operand must be the IV of an `scf.for`, op is `Pure`.

**Files:**
- Modify: `src/Dialects/KTDF/KTDFOps.td`
- Modify: `src/Dialects/KTDF/KTDFOps.cpp`
- Test: `test/Dialects/KTDF/buffer-phase-op.mlir`

- [ ] **Step 1: Add the op definition to `KTDFOps.td`**

Append the op definition immediately before `#endif // KTDF_SRC_DIALECTS_KTDF_OPS_TD`. Add a new section header to keep the file organized:

```tablegen
//===----------------------------------------------------------------------===//
// Double-Buffering Operations
//===----------------------------------------------------------------------===//

def KTDF_BufferPhaseOp : KTDF_Op<"buffer_phase", [Pure]> {
  let summary = "Compute the rotation phase for double-buffered memrefs";
  let description = [{
    The `ktdf.buffer_phase` operation produces an `index` value that selects
    which physical buffer a double-buffered (or N-buffered) memref handoff uses
    on the current pipeline iteration. The result is the linearized pipeline
    iteration count modulo `num_phases`.

    The operation takes one or more `index` operands; each operand must be the
    induction variable of an enclosing `scf.for`. The bounds and step of those
    loops are obtained by walking each operand to its `scf.for` parent.

    Semantics:
    ```
    phase = linearized_pipeline_iteration_count % num_phases
    ```
    where the linearized count is the global iteration index of the pipeline
    over its enclosing loop nest (computed from each operand IV's
    `(iv - lb) / step` and the trip counts of inner loops).

    Example:
    ```mlir
    %phase = ktdf.buffer_phase(%m0, %n0) {num_phases = 2 : i64} : index
    ```
  }];

  let arguments = (ins
    Variadic<Index>:$ivs,
    I64Attr:$num_phases
  );

  let results = (outs Index:$result);

  let assemblyFormat = [{
    `(` $ivs `)` attr-dict `:` type($result)
  }];

  let hasVerifier = 1;
}
```

- [ ] **Step 2: Add the verifier body to `KTDFOps.cpp`**

Append at the end of `src/Dialects/KTDF/KTDFOps.cpp`, after the `ModuloAttr` verifier added in Task 1:

```cpp
//===----------------------------------------------------------------------===//
// BufferPhaseOp
//===----------------------------------------------------------------------===//

LogicalResult BufferPhaseOp::verify() {
  if (getNumPhases() < 2) {
    return emitOpError("num_phases must be >= 2, got ") << getNumPhases();
  }

  if (getIvs().empty()) {
    return emitOpError("requires at least one induction variable operand");
  }

  // Each operand must be the induction variable of an enclosing scf.for.
  for (auto [idx, iv] : llvm::enumerate(getIvs())) {
    auto block_arg = llvm::dyn_cast<BlockArgument>(iv);
    if (!block_arg) {
      return emitOpError("operand #")
             << idx << " is not an scf.for induction variable";
    }
    Block* owner_block = block_arg.getOwner();
    if (!owner_block) {
      return emitOpError("operand #") << idx << " has no owning block";
    }
    auto for_op =
        llvm::dyn_cast_or_null<scf::ForOp>(owner_block->getParentOp());
    if (!for_op || for_op.getInductionVar() != iv) {
      return emitOpError("operand #")
             << idx << " is not an scf.for induction variable";
    }
  }

  return success();
}
```

You'll also need to add `#include "mlir/Dialect/SCF/IR/SCF.h"` near the top of `KTDFOps.cpp` if not already present.

- [ ] **Step 3: Build to confirm**

```bash
ninja -C build
```

Expected: clean build.

- [ ] **Step 4: Write round-trip + verifier-failure tests**

Create `test/Dialects/KTDF/buffer-phase-op.mlir`:

```mlir
// RUN: scheduler %s | scheduler | FileCheck %s
// RUN: scheduler -split-input-file -verify-diagnostics %s

// Round-trip: 1 IV, num_phases = 2.
// CHECK-LABEL:   func.func @buffer_phase_1d
// CHECK:           %[[PHASE:.*]] = ktdf.buffer_phase(%[[IV:.*]]) {num_phases = 2 : i64} : index

func.func @buffer_phase_1d() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  scf.for %i = %c0 to %c8 step %c1 {
    %phase = ktdf.buffer_phase(%i) {num_phases = 2 : i64} : index
  }
  return
}

// -----

// Round-trip: 2 IVs, num_phases = 2.
// CHECK-LABEL:   func.func @buffer_phase_2d
// CHECK:           %[[PHASE:.*]] = ktdf.buffer_phase(%[[M:.*]], %[[N:.*]]) {num_phases = 2 : i64} : index

func.func @buffer_phase_2d() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  scf.for %m = %c0 to %c8 step %c1 {
    scf.for %n = %c0 to %c8 step %c1 {
      %phase = ktdf.buffer_phase(%m, %n) {num_phases = 2 : i64} : index
    }
  }
  return
}

// -----

// Round-trip: num_phases = 4 still parses.
// CHECK-LABEL:   func.func @buffer_phase_n4
// CHECK:           %{{.*}} = ktdf.buffer_phase(%{{.*}}) {num_phases = 4 : i64} : index

func.func @buffer_phase_n4() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  scf.for %i = %c0 to %c8 step %c1 {
    %phase = ktdf.buffer_phase(%i) {num_phases = 4 : i64} : index
  }
  return
}

// -----

// Negative: num_phases < 2.
func.func @buffer_phase_num_phases_too_small() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  scf.for %i = %c0 to %c8 step %c1 {
    // expected-error @+1 {{num_phases must be >= 2}}
    %phase = ktdf.buffer_phase(%i) {num_phases = 1 : i64} : index
  }
  return
}

// -----

// Negative: empty operand list.
func.func @buffer_phase_no_ivs() {
  // expected-error @+1 {{requires at least one induction variable operand}}
  %phase = ktdf.buffer_phase() {num_phases = 2 : i64} : index
  return
}

// -----

// Negative: operand is not an scf.for IV.
func.func @buffer_phase_not_an_iv() {
  %c0 = arith.constant 0 : index
  // expected-error @+1 {{operand #0 is not an scf.for induction variable}}
  %phase = ktdf.buffer_phase(%c0) {num_phases = 2 : i64} : index
  return
}
```

- [ ] **Step 5: Run tests**

```bash
build/scheduler/bin/scheduler test/Dialects/KTDF/buffer-phase-op.mlir | /Users/prasanth-work/Documents/code/external/llvm-standalone-3/build/bin/FileCheck test/Dialects/KTDF/buffer-phase-op.mlir
build/scheduler/bin/scheduler -split-input-file -verify-diagnostics test/Dialects/KTDF/buffer-phase-op.mlir
```

Expected: both pass.

- [ ] **Step 6: Run full lit suite**

```bash
ninja -C build check-scheduler
```

Expected: all green.

- [ ] **Step 7: Commit**

```bash
git add src/Dialects/KTDF/KTDFOps.td src/Dialects/KTDF/KTDFOps.cpp test/Dialects/KTDF/buffer-phase-op.mlir
git commit -m "$(cat <<'EOF'
Add ktdf.buffer_phase op

Computes the rotation phase (linearized pipeline iteration count mod
num_phases) for double-buffered memref handoffs. Operands are induction
variables of enclosing scf.for ops; the verifier requires num_phases
>= 2, at least one operand, and each operand to be a valid IV.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Add `ktdf.select_memref` op

**Goal:** Define a new op `%X_sel = ktdf.select_memref %phase[%X0, %X1] : index -> memref<...>` per spec §2.2: one `index` operand (phase) + variadic memrefs (≥2), all candidates with identical type, result type matches; phase must come from a `ktdf.buffer_phase` whose `num_phases` equals `num_candidates`. Op is `Pure`.

**Files:**
- Modify: `src/Dialects/KTDF/KTDFOps.td`
- Modify: `src/Dialects/KTDF/KTDFOps.cpp`
- Test: `test/Dialects/KTDF/select-memref-op.mlir`

- [ ] **Step 1: Add the op definition to `KTDFOps.td`**

Append immediately after `KTDF_BufferPhaseOp` (still inside the "Double-Buffering Operations" section), before `#endif`:

```tablegen
def KTDF_SelectMemrefOp : KTDF_Op<"select_memref", [Pure]> {
  let summary = "Select one of N candidate memrefs by index";
  let description = [{
    The `ktdf.select_memref` operation picks one of `N >= 2` candidate
    memrefs based on a phase index produced by `ktdf.buffer_phase`. All
    candidate memrefs must have identical type (element type, shape including
    dynamic dims, and memory space); the result has the same type.

    Semantics:
    ```
    result = candidates[phase]
    ```
    The phase operand is verifier-guaranteed to be in range `[0, N)`.

    Example:
    ```mlir
    %X_sel = ktdf.select_memref %phase[%X0, %X1] : index -> memref<?x?xf16, #ktdf.memory_space<LX>>
    ```
  }];

  let arguments = (ins
    Index:$phase,
    Variadic<AnyMemRef>:$candidates
  );

  let results = (outs AnyMemRef:$result);

  let assemblyFormat = [{
    $phase `[` $candidates `]` attr-dict `:` type($phase) `->` type($result)
  }];

  let hasVerifier = 1;
}
```

- [ ] **Step 2: Add the verifier body to `KTDFOps.cpp`**

Append after `BufferPhaseOp::verify`:

```cpp
//===----------------------------------------------------------------------===//
// SelectMemrefOp
//===----------------------------------------------------------------------===//

LogicalResult SelectMemrefOp::verify() {
  auto candidates = getCandidates();
  if (candidates.size() < 2) {
    return emitOpError("requires at least 2 candidate memrefs, got ")
           << candidates.size();
  }

  // All candidates must have identical type, and the result must match.
  Type expected = candidates.front().getType();
  for (auto [idx, c] : llvm::enumerate(candidates)) {
    if (c.getType() != expected) {
      return emitOpError("candidate #")
             << idx << " type " << c.getType()
             << " does not match candidate #0 type " << expected;
    }
  }
  if (getResult().getType() != expected) {
    return emitOpError("result type ")
           << getResult().getType()
           << " must match candidate type " << expected;
  }

  // The phase operand must come from a ktdf.buffer_phase whose num_phases
  // equals the number of candidates.
  Operation* defining_op = getPhase().getDefiningOp();
  if (!defining_op) {
    return emitOpError("phase operand must be defined by a ktdf.buffer_phase op");
  }
  auto phase_op = llvm::dyn_cast<BufferPhaseOp>(defining_op);
  if (!phase_op) {
    return emitOpError("phase operand must be defined by a ktdf.buffer_phase op, "
                       "got ")
           << defining_op->getName();
  }
  if (phase_op.getNumPhases() != static_cast<int64_t>(candidates.size())) {
    return emitOpError("phase op num_phases (")
           << phase_op.getNumPhases() << ") must equal candidate count ("
           << candidates.size() << ")";
  }

  return success();
}
```

- [ ] **Step 3: Build to confirm**

```bash
ninja -C build
```

Expected: clean build.

- [ ] **Step 4: Write round-trip + verifier-failure tests**

Create `test/Dialects/KTDF/select-memref-op.mlir`:

```mlir
// RUN: scheduler %s | scheduler | FileCheck %s
// RUN: scheduler -split-input-file -verify-diagnostics %s

// Round-trip: select between 2 LX memrefs.
// CHECK-LABEL:   func.func @select_memref_2_lx
// CHECK:           %[[X0:.*]] = memref.alloc() : memref<64xf16, #ktdf.memory_space<LX>>
// CHECK:           %[[X1:.*]] = memref.alloc() : memref<64xf16, #ktdf.memory_space<LX>>
// CHECK:           scf.for %[[I:.*]] = {{.*}} {
// CHECK:             %[[PHASE:.*]] = ktdf.buffer_phase(%[[I]]) {num_phases = 2 : i64} : index
// CHECK:             %[[SEL:.*]] = ktdf.select_memref %[[PHASE]][%[[X0]], %[[X1]]] : index -> memref<64xf16, #ktdf.memory_space<LX>>

func.func @select_memref_2_lx() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %x0 = memref.alloc() : memref<64xf16, #ktdf.memory_space<LX>>
  %x1 = memref.alloc() : memref<64xf16, #ktdf.memory_space<LX>>
  scf.for %i = %c0 to %c8 step %c1 {
    %phase = ktdf.buffer_phase(%i) {num_phases = 2 : i64} : index
    %sel = ktdf.select_memref %phase[%x0, %x1] : index -> memref<64xf16, #ktdf.memory_space<LX>>
  }
  memref.dealloc %x0 : memref<64xf16, #ktdf.memory_space<LX>>
  memref.dealloc %x1 : memref<64xf16, #ktdf.memory_space<LX>>
  return
}

// -----

// Round-trip: select among 4 candidates with num_phases=4.
// CHECK-LABEL:   func.func @select_memref_4_lx
// CHECK:           %[[SEL:.*]] = ktdf.select_memref %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}] : index -> memref<64xf16, #ktdf.memory_space<LX>>

func.func @select_memref_4_lx() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %x0 = memref.alloc() : memref<64xf16, #ktdf.memory_space<LX>>
  %x1 = memref.alloc() : memref<64xf16, #ktdf.memory_space<LX>>
  %x2 = memref.alloc() : memref<64xf16, #ktdf.memory_space<LX>>
  %x3 = memref.alloc() : memref<64xf16, #ktdf.memory_space<LX>>
  scf.for %i = %c0 to %c8 step %c1 {
    %phase = ktdf.buffer_phase(%i) {num_phases = 4 : i64} : index
    %sel = ktdf.select_memref %phase[%x0, %x1, %x2, %x3] : index -> memref<64xf16, #ktdf.memory_space<LX>>
  }
  memref.dealloc %x0 : memref<64xf16, #ktdf.memory_space<LX>>
  memref.dealloc %x1 : memref<64xf16, #ktdf.memory_space<LX>>
  memref.dealloc %x2 : memref<64xf16, #ktdf.memory_space<LX>>
  memref.dealloc %x3 : memref<64xf16, #ktdf.memory_space<LX>>
  return
}

// -----

// Negative: only 1 candidate.
func.func @select_memref_one_candidate() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %x0 = memref.alloc() : memref<64xf16, #ktdf.memory_space<LX>>
  scf.for %i = %c0 to %c8 step %c1 {
    %phase = ktdf.buffer_phase(%i) {num_phases = 2 : i64} : index
    // expected-error @+1 {{requires at least 2 candidate memrefs}}
    %sel = ktdf.select_memref %phase[%x0] : index -> memref<64xf16, #ktdf.memory_space<LX>>
  }
  memref.dealloc %x0 : memref<64xf16, #ktdf.memory_space<LX>>
  return
}

// -----

// Negative: candidates with different types.
func.func @select_memref_mismatched_types() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %x0 = memref.alloc() : memref<64xf16, #ktdf.memory_space<LX>>
  %x1 = memref.alloc() : memref<32xf16, #ktdf.memory_space<LX>>
  scf.for %i = %c0 to %c8 step %c1 {
    %phase = ktdf.buffer_phase(%i) {num_phases = 2 : i64} : index
    // expected-error @+1 {{candidate #1 type 'memref<32xf16, #ktdf.memory_space<LX>>' does not match}}
    %sel = ktdf.select_memref %phase[%x0, %x1] : index -> memref<64xf16, #ktdf.memory_space<LX>>
  }
  memref.dealloc %x0 : memref<64xf16, #ktdf.memory_space<LX>>
  memref.dealloc %x1 : memref<32xf16, #ktdf.memory_space<LX>>
  return
}

// -----

// Negative: phase not from buffer_phase.
func.func @select_memref_phase_from_constant() {
  %c0 = arith.constant 0 : index
  %x0 = memref.alloc() : memref<64xf16, #ktdf.memory_space<LX>>
  %x1 = memref.alloc() : memref<64xf16, #ktdf.memory_space<LX>>
  // expected-error @+1 {{phase operand must be defined by a ktdf.buffer_phase op}}
  %sel = ktdf.select_memref %c0[%x0, %x1] : index -> memref<64xf16, #ktdf.memory_space<LX>>
  memref.dealloc %x0 : memref<64xf16, #ktdf.memory_space<LX>>
  memref.dealloc %x1 : memref<64xf16, #ktdf.memory_space<LX>>
  return
}

// -----

// Negative: num_phases mismatches candidate count.
func.func @select_memref_num_phases_mismatch() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %x0 = memref.alloc() : memref<64xf16, #ktdf.memory_space<LX>>
  %x1 = memref.alloc() : memref<64xf16, #ktdf.memory_space<LX>>
  %x2 = memref.alloc() : memref<64xf16, #ktdf.memory_space<LX>>
  scf.for %i = %c0 to %c8 step %c1 {
    %phase = ktdf.buffer_phase(%i) {num_phases = 2 : i64} : index
    // expected-error @+1 {{phase op num_phases (2) must equal candidate count (3)}}
    %sel = ktdf.select_memref %phase[%x0, %x1, %x2] : index -> memref<64xf16, #ktdf.memory_space<LX>>
  }
  memref.dealloc %x0 : memref<64xf16, #ktdf.memory_space<LX>>
  memref.dealloc %x1 : memref<64xf16, #ktdf.memory_space<LX>>
  memref.dealloc %x2 : memref<64xf16, #ktdf.memory_space<LX>>
  return
}
```

- [ ] **Step 5: Run tests**

```bash
build/scheduler/bin/scheduler test/Dialects/KTDF/select-memref-op.mlir | /Users/prasanth-work/Documents/code/external/llvm-standalone-3/build/bin/FileCheck test/Dialects/KTDF/select-memref-op.mlir
build/scheduler/bin/scheduler -split-input-file -verify-diagnostics test/Dialects/KTDF/select-memref-op.mlir
```

Expected: both pass. If error message wording differs, update the expected-error text to match the actual emitted diagnostic.

- [ ] **Step 6: Run full lit suite**

```bash
ninja -C build check-scheduler
```

Expected: all green.

- [ ] **Step 7: Commit**

```bash
git add src/Dialects/KTDF/KTDFOps.td src/Dialects/KTDF/KTDFOps.cpp test/Dialects/KTDF/select-memref-op.mlir
git commit -m "$(cat <<'EOF'
Add ktdf.select_memref op

Picks one of N >= 2 same-typed candidate memrefs by an index produced
by a sibling ktdf.buffer_phase. Verifier requires identical candidate
types, result type to match, and the phase op's num_phases to equal
the candidate count.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Extend `SchedulerExtContext` with `isEligibleForDoubleBuffering`

**Goal:** Add a virtual method on `SchedulerExtContext` whose default returns `true` for `MemorySpace::LX` and `false` for everything else. `DummySchedulerExtContext` inherits the default.

**Files:**
- Modify: `src/Utils/SchedulerExtContext.h`
- Modify: `src/Utils/SchedulerExtContext.cpp`

- [ ] **Step 1: Declare the method in the header**

In `src/Utils/SchedulerExtContext.h`, immediately after the `isPerCoreletUnit` declaration (around line 67), add:

```cpp
  /// True iff `mspace` is eligible for double-buffering on this target.
  /// Default returns true only for LX scratchpad. Used by the
  /// -double-buffering pass to gate which memrefs it transforms.
  virtual bool isEligibleForDoubleBuffering(
      mlir::ktdf::MemorySpace mspace) const;
```

- [ ] **Step 2: Implement the default in the source**

In `src/Utils/SchedulerExtContext.cpp`, append after the `isPerCoreletUnit` body:

```cpp
bool SchedulerExtContext::isEligibleForDoubleBuffering(
    mlir::ktdf::MemorySpace mspace) const {
  return mspace == mlir::ktdf::MemorySpace::LX;
}
```

- [ ] **Step 3: Build**

```bash
ninja -C build
```

Expected: clean build.

- [ ] **Step 4: Run full lit suite**

```bash
ninja -C build check-scheduler
```

Expected: all green (no behavior change for existing passes).

- [ ] **Step 5: Commit**

```bash
git add src/Utils/SchedulerExtContext.h src/Utils/SchedulerExtContext.cpp
git commit -m "$(cat <<'EOF'
Add SchedulerExtContext::isEligibleForDoubleBuffering

Virtual method with default returning true only for LX. The upcoming
-double-buffering pass queries this to decide whether a buffer's
memory space is eligible for the transform; subclasses can override
to support additional spaces.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Register the `-double-buffering` pass (skeleton)

**Goal:** Add the pass declaration to `Passes.td`, the constructor declaration to `Passes.h`, and a stub implementation file `DoubleBuffering.cpp` whose `runOnOperation` is empty (or just `return`). Wire it into CMake. The pass becomes usable on the CLI but does nothing yet — establishing the plumbing before the algorithm.

**Files:**
- Modify: `src/Transform/Passes.td`
- Modify: `src/Transform/Passes.h`
- Create: `src/Transform/DoubleBuffering.cpp`
- Modify: `src/Transform/CMakeLists.txt`

- [ ] **Step 1: Add the pass declaration to `Passes.td`**

Append immediately before `#endif // SCHEDULER_TRANSFORM_PASSES` in `src/Transform/Passes.td`:

```tablegen
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
  let dependentDialects = ["mlir::ktdf::KTDFDialect", "mlir::scf::SCFDialect", "mlir::memref::MemRefDialect"];
}
```

- [ ] **Step 2: Add the constructor declarations to `Passes.h`**

In `src/Transform/Passes.h`, immediately after the `createParallelizeLoopsAcrossInstancesPass` declarations (around line 30), add:

```cpp
std::unique_ptr<mlir::Pass> createDoubleBufferingPass();
std::unique_ptr<mlir::Pass> createDoubleBufferingPass(
    const scheduler::SchedulerExtContext& scheduler_ctx);
```

Place these immediately before the `void buildKTDPToDFIRPipeline(...)` declaration so the file's grouping is preserved.

- [ ] **Step 3: Create the stub pass file**

Create `src/Transform/DoubleBuffering.cpp`:

```cpp
//===----------------------------------------------------------------------===//
//
// Copyright 2026 The KTIR Scheduler Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//
// Pass: -double-buffering
//
// Detects scratchpad memrefs handed off between sibling stages of a
// ktdf.pipeline (single producer, >=1 consumer via ktdf.data_transfer) and
// rewrites them into a ping-pong form: paired allocs hoisted outside the
// pipeline scope, ktdf.buffer_phase + per-buffer ktdf.select_memref
// inserted before the pipeline, and modulo(size: 2) set on the pipeline.
//===----------------------------------------------------------------------===//

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "src/Dialects/KTDF/KTDFOps.hpp"
#include "src/Transform/Common/PipelineScope.hpp"
#include "src/Transform/Passes.h"
#include "src/Utils/SchedulerExtContext.h"

#define PASS_NAME "double-buffering"
#define DEBUG_TYPE PASS_NAME

namespace ktdf {
#define GEN_PASS_DEF_DOUBLEBUFFERINGPASS
#include "src/Transform/Passes.h.inc"
}  // namespace ktdf

using namespace mlir;

namespace {

constexpr int64_t kDefaultNumPhases = 2;

struct DoubleBufferingPass
    : public ::ktdf::impl::DoubleBufferingPassBase<DoubleBufferingPass> {
  DoubleBufferingPass() : scheduler_ctx_(scheduler::SchedulerExtContext::dummyContext()) {}
  explicit DoubleBufferingPass(const scheduler::SchedulerExtContext& ctx)
      : scheduler_ctx_(ctx) {}

  void runOnOperation() override {
    // Algorithm added in subsequent tasks.
    LLVM_DEBUG(llvm::dbgs() << "[" PASS_NAME "] running (no-op skeleton)\n");
  }

 private:
  const scheduler::SchedulerExtContext& scheduler_ctx_;
};

}  // namespace

namespace ktdf {

std::unique_ptr<mlir::Pass> createDoubleBufferingPass() {
  return std::make_unique<DoubleBufferingPass>();
}

std::unique_ptr<mlir::Pass> createDoubleBufferingPass(
    const scheduler::SchedulerExtContext& scheduler_ctx) {
  return std::make_unique<DoubleBufferingPass>(scheduler_ctx);
}

}  // namespace ktdf
```

- [ ] **Step 4: Add to CMake**

In `src/Transform/CMakeLists.txt`, add `DoubleBuffering.cpp` to the `add_library(SchedulerTransforms ...)` source list. The list (alphabetically) becomes:

```cmake
add_library(SchedulerTransforms
    BroadcastPromotion.cpp
    Common/ApplicableUnits.cpp
    Common/PipelineScope.cpp
    ComputeGroupExtraction.cpp
    ConstructThreeStagePipeline.cpp
    DoubleBuffering.cpp
    ParallelizeLoopsAcrossInstances.cpp
    PipelineTree/Legalizer.cpp
    PipelineTree/Materializer.cpp
    ReusePromotion/Common/CandidateSearch.cpp
    ReusePromotion/Common/InvarianceCheck.cpp
    ReusePromotion/Common/RegionClonePrune.cpp
    ReusePromotion/Common/WriteSetScan.cpp
    StageCoarsening/BufferExpansion.cpp
    StageCoarsening/ScopeCorrection.cpp
    StageCoarsening.cpp
    StripMineSCFForLoops.cpp
    TileSCFForLoops.cpp
    Utils/CustomLinalgTiling.cpp
    Utils/Utils.cpp
)
```

- [ ] **Step 5: Build**

```bash
ninja -C build
```

Expected: clean build. The pass is now registered and invokable.

- [ ] **Step 6: Verify the pass is registered**

```bash
build/scheduler/bin/scheduler --help 2>&1 | grep -E "double-buffering"
```

Expected: one line mentioning `--double-buffering`.

- [ ] **Step 7: Run a quick smoke test on existing IR**

```bash
build/scheduler/bin/scheduler -double-buffering test/Dialects/KTDF/private-ops.mlir
```

Expected: prints the input IR unchanged (the pass is a no-op stub). No errors.

- [ ] **Step 8: Run full lit suite**

```bash
ninja -C build check-scheduler
```

Expected: all green.

- [ ] **Step 9: Commit**

```bash
git add src/Transform/Passes.td src/Transform/Passes.h src/Transform/DoubleBuffering.cpp src/Transform/CMakeLists.txt
git commit -m "$(cat <<'EOF'
Register -double-buffering pass skeleton

Adds the pass registration with both no-arg and SchedulerExtContext
constructors. runOnOperation is a no-op for now; algorithm follows in
subsequent commits.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: Implement detection (Phase A — pre-pipeline guards)

**Goal:** Implement the per-pipeline early skip checks: pre-existing `modulo` and empty enclosing scope. After this task, the pass walks all pipelines and logs which ones it would consider, but doesn't yet examine candidates.

**Files:**
- Modify: `src/Transform/DoubleBuffering.cpp`

- [ ] **Step 1: Add the per-pipeline guard helpers**

Inside the anonymous namespace in `src/Transform/DoubleBuffering.cpp`, before the `DoubleBufferingPass` struct, add:

```cpp
//===----------------------------------------------------------------------===//
// Per-pipeline guards
//===----------------------------------------------------------------------===//

/// True iff the pipeline is eligible for analysis (no pre-existing modulo
/// and a non-empty enclosing scope of scf.for loops). On rejection, logs
/// the reason via LLVM_DEBUG.
bool isPipelineEligible(
    mlir::ktdf::PipelineOp pipeline,
    const ::ktdf::common::PipelineEnclosingScope& scope) {
  if (pipeline.getModuloAttr()) {
    LLVM_DEBUG(llvm::dbgs()
               << "[" PASS_NAME "] skip pipeline at " << pipeline.getLoc()
               << ": modulo already set\n");
    return false;
  }
  if (scope.loops.empty()) {
    LLVM_DEBUG(llvm::dbgs()
               << "[" PASS_NAME "] skip pipeline at " << pipeline.getLoc()
               << ": no enclosing scf.for scope\n");
    return false;
  }
  return true;
}
```

- [ ] **Step 2: Wire the guards into `runOnOperation`**

Replace the `runOnOperation` body with:

```cpp
  void runOnOperation() override {
    ModuleOp module = getOperation();

    LLVM_DEBUG(llvm::dbgs() << "[" PASS_NAME "] starting\n");

    // Pre-order walk over pipelines.
    SmallVector<mlir::ktdf::PipelineOp> eligible;
    module.walk<WalkOrder::PreOrder>([&](mlir::ktdf::PipelineOp pipeline) {
      auto scope = ::ktdf::common::computePipelineEnclosingScope(pipeline);
      if (!isPipelineEligible(pipeline, scope)) {
        return;
      }
      LLVM_DEBUG(llvm::dbgs()
                 << "[" PASS_NAME "] eligible pipeline at " << pipeline.getLoc()
                 << "\n");
      eligible.push_back(pipeline);
    });

    // Candidate detection and transformation arrive in subsequent tasks.
  }
```

- [ ] **Step 3: Build and run smoke test**

```bash
ninja -C build
build/scheduler/bin/scheduler -double-buffering test/Dialects/KTDF/private-ops.mlir
```

Expected: input IR unchanged (no transformation yet).

```bash
build/scheduler/bin/scheduler --debug-only=double-buffering -double-buffering test/Dialects/KTDF/private-ops.mlir 2>&1 | grep "double-buffering"
```

Expected: log lines for `starting`, then either `skip pipeline ...: no enclosing scf.for scope` (because the pipeline in `private-ops.mlir` has no enclosing loop) or similar.

- [ ] **Step 4: Run full lit suite**

```bash
ninja -C build check-scheduler
```

Expected: all green.

- [ ] **Step 5: Commit**

```bash
git add src/Transform/DoubleBuffering.cpp
git commit -m "$(cat <<'EOF'
double-buffering: add per-pipeline early-skip guards

Skips pipelines that already have a modulo attribute or whose enclosing
scope of scf.for loops is empty. Both rejections are logged via
LLVM_DEBUG. No transformation yet.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: Implement detection (Phase B — per-buffer eligibility)

**Goal:** For each eligible pipeline, examine each value yielded by its `ktdf.private` region and apply the per-yielded-value checks from spec §3.3 a–f: type is memref, memory-space is eligible, defining op is `memref.alloc` with exactly one use that is the `private_yield` in the same block, and dynamic size operands dominate the hoist target. Build a per-pipeline list of "shape-eligible" candidates. No producer/consumer scan yet.

**Files:**
- Modify: `src/Transform/DoubleBuffering.cpp`

- [ ] **Step 1: Add shape-eligibility helpers**

Inside the anonymous namespace, before `isPipelineEligible`, add:

```cpp
//===----------------------------------------------------------------------===//
// Per-buffer shape-eligibility helpers
//===----------------------------------------------------------------------===//

struct CandidateShape {
  /// SSA value yielded by ktdf.private (a result of the PrivateOp).
  mlir::Value yielded;
  /// Index into PrivateOp's results / PrivateYieldOp's operands.
  unsigned slot;
  /// The single memref.alloc op inside the private region producing this value.
  mlir::memref::AllocOp alloc;
  /// Memref type carried by `yielded`.
  mlir::MemRefType memref_ty;
};

/// Extract the KTDF MemorySpace enum from a memref's memory space attribute.
/// Returns std::nullopt if the memory space is not a KTDF MemorySpaceCustomAttr.
std::optional<mlir::ktdf::MemorySpace> extractKtdfMemorySpace(
    mlir::MemRefType memref_ty) {
  Attribute mspace_attr = memref_ty.getMemorySpace();
  if (!mspace_attr) {
    return std::nullopt;
  }
  if (auto ktdf_attr =
          llvm::dyn_cast<mlir::ktdf::MemorySpaceCustomAttr>(mspace_attr)) {
    return ktdf_attr.getValue();
  }
  return std::nullopt;
}

/// True iff every dynamic-size SSA operand of `alloc` is defined at a location
/// that dominates `insertion_point_op` (using strict op-level dominance —
/// "defined before" `insertion_point_op` in any common enclosing block).
bool dynamicSizesDominate(mlir::memref::AllocOp alloc,
                          mlir::Operation* insertion_point_op,
                          mlir::DominanceInfo& dom) {
  for (Value sz : alloc.getDynamicSizes()) {
    if (!dom.properlyDominates(sz, insertion_point_op)) {
      return false;
    }
  }
  return true;
}

/// Apply spec §3.3 a–f checks to one yielded value of a pipeline's
/// ktdf.private region. Returns a populated CandidateShape on success;
/// std::nullopt on rejection (with a reason logged via LLVM_DEBUG).
std::optional<CandidateShape> checkBufferShape(
    mlir::ktdf::PipelineOp pipeline, mlir::ktdf::PrivateOp private_op,
    unsigned slot, mlir::Operation* hoist_target,
    const scheduler::SchedulerExtContext& ctx, mlir::DominanceInfo& dom) {
  Value yielded = private_op.getResults()[slot];

  // a. Type check: memref.
  auto memref_ty = llvm::dyn_cast<mlir::MemRefType>(yielded.getType());
  if (!memref_ty) {
    LLVM_DEBUG(llvm::dbgs()
               << "[" PASS_NAME "]   slot " << slot << " skipped: not a memref\n");
    return std::nullopt;
  }

  // b. Memory-space eligibility.
  auto mspace = extractKtdfMemorySpace(memref_ty);
  if (!mspace.has_value() || !ctx.isEligibleForDoubleBuffering(*mspace)) {
    LLVM_DEBUG(llvm::dbgs()
               << "[" PASS_NAME "]   slot " << slot
               << " skipped: memory space ineligible\n");
    return std::nullopt;
  }

  // c. Defining op check: memref.alloc inside the private region.
  Value yield_operand = private_op.getYieldOp().getOperands()[slot];
  auto alloc = yield_operand.getDefiningOp<mlir::memref::AllocOp>();
  if (!alloc) {
    LLVM_DEBUG(llvm::dbgs()
               << "[" PASS_NAME "]   slot " << slot
               << " skipped: defining op is not memref.alloc\n");
    return std::nullopt;
  }

  // d. Single-use check: alloc result has exactly one use, and that use is
  // the private_yield op.
  if (!alloc.getResult().hasOneUse()) {
    LLVM_DEBUG(llvm::dbgs()
               << "[" PASS_NAME "]   slot " << slot
               << " skipped: alloc has multiple uses\n");
    return std::nullopt;
  }
  Operation* sole_user = *alloc.getResult().getUsers().begin();
  if (sole_user != private_op.getYieldOp().getOperation()) {
    LLVM_DEBUG(llvm::dbgs()
               << "[" PASS_NAME "]   slot " << slot
               << " skipped: alloc's sole use is not private_yield\n");
    return std::nullopt;
  }

  // e. Same-block check: alloc and private_yield in the same block of the
  // private region.
  if (alloc->getBlock() != private_op.getYieldOp()->getBlock()) {
    LLVM_DEBUG(llvm::dbgs()
               << "[" PASS_NAME "]   slot " << slot
               << " skipped: alloc and yield are in different blocks\n");
    return std::nullopt;
  }

  // f. Dynamic-size domination check.
  if (!dynamicSizesDominate(alloc, hoist_target, dom)) {
    LLVM_DEBUG(llvm::dbgs()
               << "[" PASS_NAME "]   slot " << slot
               << " skipped: dynamic size operand does not dominate hoist target\n");
    return std::nullopt;
  }

  return CandidateShape{yielded, slot, alloc, memref_ty};
}

/// Find the PrivateOp that is the immediate child of `pipeline`. Returns
/// nullptr if the pipeline has no PrivateOp.
mlir::ktdf::PrivateOp getPipelinePrivateOp(mlir::ktdf::PipelineOp pipeline) {
  for (Operation& op : pipeline.getBodyBlock()->getOperations()) {
    if (auto p = llvm::dyn_cast<mlir::ktdf::PrivateOp>(op)) {
      return p;
    }
  }
  return nullptr;
}
```

You'll also need to add `#include "mlir/IR/Dominance.h"` (already added in Task 6) and ensure `mlir::memref::AllocOp` is reachable via the existing `MemRef.h` include.

- [ ] **Step 2: Wire shape checking into `runOnOperation`**

Replace the body of the pre-order walk with:

```cpp
    DominanceInfo dom(module);

    SmallVector<std::pair<mlir::ktdf::PipelineOp, SmallVector<CandidateShape>>>
        per_pipeline_shapes;
    module.walk<WalkOrder::PreOrder>([&](mlir::ktdf::PipelineOp pipeline) {
      auto scope = ::ktdf::common::computePipelineEnclosingScope(pipeline);
      if (!isPipelineEligible(pipeline, scope)) {
        return;
      }

      // Hoist target: the position immediately before scope.loops.back()
      // (the outermost loop in scope).
      Operation* hoist_target = scope.loops.back().getOperation();

      auto private_op = getPipelinePrivateOp(pipeline);
      if (!private_op) {
        LLVM_DEBUG(llvm::dbgs()
                   << "[" PASS_NAME "] skip pipeline at " << pipeline.getLoc()
                   << ": no ktdf.private region\n");
        return;
      }

      SmallVector<CandidateShape> shapes;
      for (unsigned slot = 0; slot < private_op.getNumResults(); ++slot) {
        auto shape = checkBufferShape(pipeline, private_op, slot, hoist_target,
                                       scheduler_ctx_, dom);
        if (shape.has_value()) {
          shapes.push_back(*shape);
        }
      }

      if (shapes.empty()) {
        LLVM_DEBUG(llvm::dbgs()
                   << "[" PASS_NAME "] skip pipeline at " << pipeline.getLoc()
                   << ": no shape-eligible buffers\n");
        return;
      }

      LLVM_DEBUG(llvm::dbgs()
                 << "[" PASS_NAME "] pipeline at " << pipeline.getLoc()
                 << " has " << shapes.size() << " shape-eligible buffer(s)\n");
      per_pipeline_shapes.emplace_back(pipeline, std::move(shapes));
    });

    // Producer/consumer scan + transformation arrive in subsequent tasks.
```

- [ ] **Step 3: Build and smoke test**

```bash
ninja -C build
build/scheduler/bin/scheduler --debug-only=double-buffering -double-buffering test/Dialects/KTDF/private-ops.mlir 2>&1 | grep "double-buffering"
```

Expected: log lines including `skip pipeline ...: no enclosing scf.for scope` (since `private-ops.mlir`'s pipeline has no enclosing loop).

- [ ] **Step 4: Run full lit suite**

```bash
ninja -C build check-scheduler
```

Expected: all green.

- [ ] **Step 5: Commit**

```bash
git add src/Transform/DoubleBuffering.cpp
git commit -m "$(cat <<'EOF'
double-buffering: detect shape-eligible buffers in ktdf.private

For each eligible pipeline, examines each yielded slot of its
ktdf.private region against the spec §3.3 a–f filters: memref type,
LX-eligible memory space, defining op exactly memref.alloc, single
use that is the private_yield, alloc and yield in the same block,
dynamic size operands dominate the hoist target. No producer/consumer
scan yet — that's the next commit.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 9: Implement detection (Phase C — producer/consumer scan + token precedence)

**Goal:** For each shape-eligible buffer, scan the pipeline's sibling stages recursively for `ktdf.data_transfer` ops referencing the buffer, classifying each containing sibling stage as producer or consumer. Apply the cardinality, disjointness, and token-precedence checks from spec §3.3 g–k. Output a final list of `(pipeline, scope, candidates)` tuples.

**Files:**
- Modify: `src/Transform/DoubleBuffering.cpp`

- [ ] **Step 1: Add the producer/consumer scan and final-candidate types**

Inside the anonymous namespace, after `getPipelinePrivateOp`, add:

```cpp
//===----------------------------------------------------------------------===//
// Producer/consumer scan and token precedence
//===----------------------------------------------------------------------===//

struct Candidate {
  CandidateShape shape;
  /// The unique sibling stage of the pipeline that writes the buffer.
  mlir::ktdf::StageOp producer;
  /// All sibling stages that read the buffer.
  SmallVector<mlir::ktdf::StageOp> consumers;
};

/// Find the immediate sibling stage of `pipeline` containing `op` (op may be
/// nested arbitrarily deep inside the stage's region). Returns nullptr if no
/// sibling-stage ancestor is found.
mlir::ktdf::StageOp findEnclosingSiblingStage(
    mlir::ktdf::PipelineOp pipeline, mlir::Operation* op) {
  Operation* cur = op;
  while (cur && cur->getParentOp() != pipeline.getOperation()) {
    if (auto stage = llvm::dyn_cast<mlir::ktdf::StageOp>(cur)) {
      if (stage->getParentOp() == pipeline.getOperation()) {
        return stage;
      }
    }
    cur = cur->getParentOp();
  }
  if (auto stage = llvm::dyn_cast_or_null<mlir::ktdf::StageOp>(cur)) {
    return stage;
  }
  return nullptr;
}

/// Walk the entire region of each sibling stage of `pipeline` recursively,
/// classifying every ktdf.data_transfer that references `buffer` as producer
/// (buffer is dest) or consumer (buffer is source). Stages are deduplicated.
/// Returns std::nullopt iff a sibling stage is both producer and consumer.
std::optional<std::pair<SmallVector<mlir::ktdf::StageOp>,
                         SmallVector<mlir::ktdf::StageOp>>>
scanProducersConsumers(mlir::ktdf::PipelineOp pipeline, mlir::Value buffer) {
  llvm::SmallSetVector<mlir::ktdf::StageOp, 4> producers;
  llvm::SmallSetVector<mlir::ktdf::StageOp, 4> consumers;

  for (Operation& sibling : pipeline.getBodyBlock()->getOperations()) {
    auto stage = llvm::dyn_cast<mlir::ktdf::StageOp>(sibling);
    if (!stage) {
      continue;
    }
    stage->walk([&](mlir::ktdf::DataTransferOp xfer) {
      if (xfer.getDestination() == buffer) {
        producers.insert(stage);
      }
      if (xfer.getSource() == buffer) {
        consumers.insert(stage);
      }
    });
  }

  // Disjointness: no sibling can be both producer and consumer.
  for (auto stage : producers) {
    if (consumers.contains(stage)) {
      return std::nullopt;
    }
  }

  return std::make_pair(SmallVector<mlir::ktdf::StageOp>(producers.begin(),
                                                          producers.end()),
                        SmallVector<mlir::ktdf::StageOp>(consumers.begin(),
                                                          consumers.end()));
}

/// True iff `producer` transitively precedes `target` in `pipeline`'s token
/// dependency graph. Edges are stage-to-stage: stage S' has an incoming edge
/// from stage S iff S produces (depends_out) a token that S' consumes
/// (depends_in).
bool producerTransitivelyPrecedes(mlir::ktdf::PipelineOp pipeline,
                                  mlir::ktdf::StageOp producer,
                                  mlir::ktdf::StageOp target) {
  if (producer == target) {
    return false;
  }

  // Build a map from a token SSA value → defining stage.
  llvm::DenseMap<mlir::Value, mlir::ktdf::StageOp> token_producer_stage;
  for (Operation& op : pipeline.getBodyBlock()->getOperations()) {
    auto stage = llvm::dyn_cast<mlir::ktdf::StageOp>(op);
    if (!stage) {
      continue;
    }
    for (mlir::Value t : stage.getDependsOut()) {
      token_producer_stage.try_emplace(t, stage);
    }
  }

  // BFS from producer over token edges.
  llvm::SmallSetVector<mlir::ktdf::StageOp, 8> visited;
  visited.insert(producer);
  SmallVector<mlir::ktdf::StageOp> worklist{producer};
  while (!worklist.empty()) {
    mlir::ktdf::StageOp s = worklist.pop_back_val();
    // For every other sibling stage S' that consumes a token produced by S,
    // S' is reachable from S.
    for (Operation& op : pipeline.getBodyBlock()->getOperations()) {
      auto succ = llvm::dyn_cast<mlir::ktdf::StageOp>(op);
      if (!succ || succ == s) {
        continue;
      }
      bool depends_on_s = false;
      for (mlir::Value t : succ.getDependsIn()) {
        auto it = token_producer_stage.find(t);
        if (it != token_producer_stage.end() && it->second == s) {
          depends_on_s = true;
          break;
        }
      }
      if (!depends_on_s) {
        continue;
      }
      if (succ == target) {
        return true;
      }
      if (visited.insert(succ)) {
        worklist.push_back(succ);
      }
    }
  }
  return false;
}

/// Apply spec §3.3 g–k checks to a shape-eligible buffer in `pipeline`. On
/// success, returns a Candidate. On rejection, logs the reason and returns
/// std::nullopt.
std::optional<Candidate> checkProducerConsumer(
    mlir::ktdf::PipelineOp pipeline, const CandidateShape& shape) {
  auto scan = scanProducersConsumers(pipeline, shape.yielded);
  if (!scan.has_value()) {
    LLVM_DEBUG(llvm::dbgs()
               << "[" PASS_NAME "]   slot " << shape.slot
               << " skipped: a sibling stage both writes and reads the buffer\n");
    return std::nullopt;
  }
  auto& [producers, consumers] = *scan;

  // h. Producer cardinality.
  if (producers.size() != 1) {
    LLVM_DEBUG(llvm::dbgs()
               << "[" PASS_NAME "]   slot " << shape.slot << " skipped: "
               << producers.size() << " producer stages (need exactly 1)\n");
    return std::nullopt;
  }

  // i. Consumer cardinality.
  if (consumers.empty()) {
    LLVM_DEBUG(llvm::dbgs()
               << "[" PASS_NAME "]   slot " << shape.slot
               << " skipped: no consumer stages\n");
    return std::nullopt;
  }

  // k. Producer must transitively precede every consumer.
  for (mlir::ktdf::StageOp consumer : consumers) {
    if (!producerTransitivelyPrecedes(pipeline, producers.front(), consumer)) {
      LLVM_DEBUG(llvm::dbgs()
                 << "[" PASS_NAME "]   slot " << shape.slot
                 << " skipped: producer does not transitively precede consumer\n");
      return std::nullopt;
    }
  }

  return Candidate{shape, producers.front(), consumers};
}
```

- [ ] **Step 2: Wire the producer/consumer scan into `runOnOperation`**

Replace the trailing `// Producer/consumer scan + transformation arrive in subsequent tasks.` line with:

```cpp
    // Per pipeline, filter shape-eligible buffers down to producer/consumer-
    // valid candidates.
    SmallVector<std::pair<mlir::ktdf::PipelineOp, SmallVector<Candidate>>>
        per_pipeline_candidates;
    for (auto& [pipeline, shapes] : per_pipeline_shapes) {
      SmallVector<Candidate> candidates;
      for (const CandidateShape& shape : shapes) {
        if (auto c = checkProducerConsumer(pipeline, shape)) {
          candidates.push_back(*c);
        }
      }
      if (candidates.empty()) {
        LLVM_DEBUG(llvm::dbgs()
                   << "[" PASS_NAME "] skip pipeline at " << pipeline.getLoc()
                   << ": no candidates passed producer/consumer scan\n");
        continue;
      }
      LLVM_DEBUG(llvm::dbgs()
                 << "[" PASS_NAME "] pipeline at " << pipeline.getLoc()
                 << " has " << candidates.size() << " candidate(s)\n");
      per_pipeline_candidates.emplace_back(pipeline, std::move(candidates));
    }

    // Transformation arrives in the next task.
```

- [ ] **Step 3: Build and run lit suite**

```bash
ninja -C build
ninja -C build check-scheduler
```

Expected: all green. The pass still doesn't transform anything; it now logs every candidate it would handle.

- [ ] **Step 4: Commit**

```bash
git add src/Transform/DoubleBuffering.cpp
git commit -m "$(cat <<'EOF'
double-buffering: scan producers/consumers and check token precedence

For each shape-eligible buffer, walks every sibling stage's region
(recursively) for ktdf.data_transfer ops referencing the buffer,
classifying each containing sibling stage as producer or consumer.
Applies the spec §3.3 g–k filters: single producer, ≥1 consumer,
disjointness (no sibling is both), and the producer transitively
precedes every consumer in the pipeline's token graph. No
transformation yet.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 10: Implement transformation (hoist allocs + insert phase + selectors)

**Goal:** Implement the first three transform steps from spec §4 — hoist each candidate's alloc into two allocs just outside `scope.loops.back()` with matching deallocs, insert one shared `ktdf.buffer_phase` immediately before the pipeline, and per-candidate `ktdf.select_memref` after the phase. Don't yet rewrite the pipeline's references or set `modulo` or clean up `ktdf.private` (that's Task 11).

**Files:**
- Modify: `src/Transform/DoubleBuffering.cpp`

- [ ] **Step 1: Add transform helpers**

Inside the anonymous namespace, before the `DoubleBufferingPass` struct, add:

```cpp
//===----------------------------------------------------------------------===//
// Transformation
//===----------------------------------------------------------------------===//

/// Clone `alloc` `n` times immediately before `outermost_loop`. Insert
/// matching deallocs immediately after `outermost_loop`. Returns the cloned
/// alloc results.
SmallVector<Value> hoistAllocPair(mlir::memref::AllocOp alloc,
                                  mlir::scf::ForOp outermost_loop) {
  OpBuilder builder(outermost_loop);

  SmallVector<Value> hoisted;
  for (int i = 0; i < kDefaultNumPhases; ++i) {
    auto cloned = llvm::cast<mlir::memref::AllocOp>(builder.clone(*alloc));
    hoisted.push_back(cloned.getResult());
  }

  // Deallocs go just after the loop.
  builder.setInsertionPointAfter(outermost_loop);
  for (Value h : hoisted) {
    mlir::memref::DeallocOp::create(builder, alloc.getLoc(), h);
  }

  return hoisted;
}

/// Build the operand list for ktdf.buffer_phase from `scope_loops`
/// (innermost-first per the convention) reversed to outermost-first.
SmallVector<Value> sceneIvsOutermostFirst(
    llvm::ArrayRef<mlir::scf::ForOp> scope_loops_innermost_first) {
  SmallVector<Value> result;
  for (mlir::scf::ForOp loop : llvm::reverse(scope_loops_innermost_first)) {
    result.push_back(loop.getInductionVar());
  }
  return result;
}
```

- [ ] **Step 2: Add the per-pipeline transform function**

Append after `checkProducerConsumer`:

```cpp
/// Apply the transformation to one pipeline: hoist alloc pairs, insert one
/// shared ktdf.buffer_phase, insert per-candidate ktdf.select_memref,
/// rewrite uses of each original yielded buffer inside the pipeline's
/// region to the corresponding select result, set modulo(size: 2), and
/// clean up the dropped slots from ktdf.private.
void rewritePipeline(mlir::ktdf::PipelineOp pipeline,
                     const ::ktdf::common::PipelineEnclosingScope& scope,
                     llvm::ArrayRef<Candidate> candidates) {
  assert(!candidates.empty());
  assert(!scope.loops.empty());
  mlir::scf::ForOp outermost = scope.loops.back();

  // Step 1: hoist alloc pairs per candidate.
  SmallVector<SmallVector<Value>> hoisted_pairs;
  hoisted_pairs.reserve(candidates.size());
  for (const Candidate& c : candidates) {
    hoisted_pairs.push_back(hoistAllocPair(c.shape.alloc, outermost));
  }

  // Step 2: insert one shared buffer_phase immediately before the pipeline.
  OpBuilder builder(pipeline);
  SmallVector<Value> phase_ivs = sceneIvsOutermostFirst(scope.loops);
  auto phase_op = mlir::ktdf::BufferPhaseOp::create(
      builder, pipeline.getLoc(), builder.getIndexType(), phase_ivs,
      builder.getI64IntegerAttr(kDefaultNumPhases));
  Value phase = phase_op.getResult();

  // Step 3: insert per-candidate select_memref immediately before the
  // pipeline (after the buffer_phase).
  SmallVector<Value> selected;
  selected.reserve(candidates.size());
  for (auto [c, hoisted] : llvm::zip(candidates, hoisted_pairs)) {
    auto sel = mlir::ktdf::SelectMemrefOp::create(
        builder, pipeline.getLoc(), c.shape.memref_ty, phase, hoisted);
    selected.push_back(sel.getResult());
  }

  // Steps 4–6 arrive in the next task.
  (void)selected;
}
```

- [ ] **Step 3: Wire the transform call into `runOnOperation`**

Replace the trailing `// Transformation arrives in the next task.` with:

```cpp
    for (auto& [pipeline, candidates] : per_pipeline_candidates) {
      auto scope = ::ktdf::common::computePipelineEnclosingScope(pipeline);
      rewritePipeline(pipeline, scope, candidates);
    }
```

- [ ] **Step 4: Build**

```bash
ninja -C build
```

Expected: clean build.

- [ ] **Step 5: Run lit suite**

```bash
ninja -C build check-scheduler
```

Expected: all green. The pass now hoists allocs and inserts phase + selectors, but the pipeline's stages still reference the original `ktdf.private`-yielded values, so the IR is currently inconsistent (verifier may complain in extreme cases — but no existing test triggers the path because no existing input matches the candidate criteria once the algorithm is fully wired). If a regression appears, revert and address in Task 11.

If you do see verifier failures from existing tests after this commit, comment out the `for (auto& [pipeline, candidates] : per_pipeline_candidates) { ... }` block in `runOnOperation`, leaving Task 11 to enable end-to-end transformation.

- [ ] **Step 6: Commit**

```bash
git add src/Transform/DoubleBuffering.cpp
git commit -m "$(cat <<'EOF'
double-buffering: hoist alloc pairs and insert phase + selectors

For each candidate buffer, clones its memref.alloc twice immediately
before the outermost loop in the pipeline scope and inserts matching
deallocs just after that loop. Inserts one shared ktdf.buffer_phase
immediately before the pipeline (operands = scope IVs, outermost-
first), and a ktdf.select_memref per candidate after the phase.
Pipeline-internal references still point at the original yields;
that rewrite + modulo + private cleanup are next.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 11: Implement transformation (rewrite uses, set modulo, clean up `ktdf.private`)

**Goal:** Complete the transformation by rewriting every use of each original yielded buffer inside the pipeline's region to the corresponding `select_memref` result, setting `modulo(size: 2)` on the pipeline, and pruning the dropped slots from `ktdf.private` / `ktdf.private_yield` / erasing the orphaned `memref.alloc` ops in the private region.

**Files:**
- Modify: `src/Transform/DoubleBuffering.cpp`

- [ ] **Step 1: Add the cleanup helpers**

Inside the anonymous namespace, after `rewritePipeline` (replacing its tail steps), add:

```cpp
/// Replace all uses of `from` with `to` that are inside `region` (recursively
/// — including nested ops). Uses outside the region are left alone (there
/// shouldn't be any for a ktdf.private result, but this keeps the helper
/// scoped).
void replaceUsesInRegion(Value from, Value to, mlir::Region& region) {
  from.replaceUsesWithIf(to, [&](OpOperand& use) {
    Operation* user = use.getOwner();
    return region.isAncestor(user->getParentRegion());
  });
}

/// Erase the slots in `slots_to_drop` from a ktdf.private op:
///   1. Erase the corresponding operands of ktdf.private_yield.
///   2. Recreate the ktdf.private op with the reduced result types.
///   3. RAUW the kept results of the old op with the new op's results.
///   4. Erase the orphaned memref.alloc ops in the private region.
void prunePrivateSlots(mlir::ktdf::PrivateOp private_op,
                       llvm::ArrayRef<unsigned> slots_to_drop_sorted) {
  // Build keep mask.
  unsigned n_results = private_op.getNumResults();
  llvm::BitVector drop_mask(n_results);
  for (unsigned slot : slots_to_drop_sorted) {
    drop_mask.set(slot);
  }

  // Erase yield operands at dropped slots, collecting the orphaned alloc ops
  // for later erase. Process slots in descending order so indices stay valid.
  mlir::ktdf::PrivateYieldOp yield_op = private_op.getYieldOp();
  SmallVector<mlir::memref::AllocOp> orphaned_allocs;
  for (unsigned slot :
       llvm::reverse(SmallVector<unsigned>(slots_to_drop_sorted.begin(),
                                            slots_to_drop_sorted.end()))) {
    Value yield_operand = yield_op.getOperands()[slot];
    if (auto alloc = yield_operand.getDefiningOp<mlir::memref::AllocOp>()) {
      orphaned_allocs.push_back(alloc);
    }
    yield_op->eraseOperand(slot);
  }

  // Compute the new result types for the recreated PrivateOp.
  SmallVector<Type> new_result_types;
  for (unsigned i = 0; i < n_results; ++i) {
    if (!drop_mask.test(i)) {
      new_result_types.push_back(private_op.getResult(i).getType());
    }
  }

  // Build the new PrivateOp at the same location and splice the body.
  OpBuilder builder(private_op);
  auto new_private = mlir::ktdf::PrivateOp::create(
      builder, private_op.getLoc(), TypeRange(new_result_types));
  // Erase the auto-created empty body block; replace by splicing the old
  // region in.
  new_private.getBody().takeBody(private_op.getBody());

  // RAUW kept results of the old op with corresponding results of the new op.
  unsigned new_idx = 0;
  for (unsigned i = 0; i < n_results; ++i) {
    if (drop_mask.test(i)) {
      continue;
    }
    private_op.getResult(i).replaceAllUsesWith(new_private.getResult(new_idx));
    ++new_idx;
  }

  // Erase the old PrivateOp.
  private_op.erase();

  // Erase the orphaned allocs (now uniqued via private_yield's operand
  // erasure).
  for (mlir::memref::AllocOp alloc : orphaned_allocs) {
    alloc.erase();
  }
}
```

- [ ] **Step 2: Replace the tail of `rewritePipeline` with the full sequence**

Locate `rewritePipeline` and replace its trailing `// Steps 4–6 arrive in the next task.\n  (void)selected;` block with:

```cpp
  // Step 4: rewrite uses of each original yielded buffer inside the
  // pipeline's region to the corresponding selected memref.
  for (auto [c, sel] : llvm::zip(candidates, selected)) {
    replaceUsesInRegion(c.shape.yielded, sel, pipeline.getBody());
  }

  // Step 5: set modulo(size: 2) on the pipeline.
  pipeline.setModuloAttr(mlir::ktdf::ModuloAttr::get(pipeline.getContext(),
                                                     kDefaultNumPhases));

  // Step 6: clean up ktdf.private — drop the slots we transferred out.
  SmallVector<unsigned> slots_to_drop;
  slots_to_drop.reserve(candidates.size());
  for (const Candidate& c : candidates) {
    slots_to_drop.push_back(c.shape.slot);
  }
  llvm::sort(slots_to_drop);
  if (auto private_op = getPipelinePrivateOp(pipeline)) {
    prunePrivateSlots(private_op, slots_to_drop);
  }
}
```

- [ ] **Step 3: Build**

```bash
ninja -C build
```

Expected: clean build.

- [ ] **Step 4: Run a smoke test against a stub MLIR**

Create a temporary input file `/tmp/db-smoke.mlir` to inspect output by hand:

```bash
cat > /tmp/db-smoke.mlir <<'MLIR'
func.func @smoke(%M: index, %N: index, %Cm: index, %Cn: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c64 = arith.constant 64 : index
  scf.for %m0 = %c0 to %M step %Cm {
    scf.for %n0 = %c0 to %N step %Cn {
      ktdf.pipeline {
        %lx_a, %lx_c, %t1, %t2 = ktdf.private -> (
            memref<?x?xf16, #ktdf.memory_space<LX>>,
            memref<?x?xf16, #ktdf.memory_space<LX>>,
            !ktdf.token, !ktdf.token) {
          %a = memref.alloc(%Cm, %Cn) : memref<?x?xf16, #ktdf.memory_space<LX>>
          %c = memref.alloc(%Cm, %Cn) : memref<?x?xf16, #ktdf.memory_space<LX>>
          %tk1 = ktdf.create_token : !ktdf.token
          %tk2 = ktdf.create_token : !ktdf.token
          ktdf.private_yield %a, %c, %tk1, %tk2
            : memref<?x?xf16, #ktdf.memory_space<LX>>,
              memref<?x?xf16, #ktdf.memory_space<LX>>,
              !ktdf.token, !ktdf.token
        }
        ktdf.stage depends_in(none) depends_out(%t1 : !ktdf.token) {
          ktdf.data_transfer from %lx_a[%c0, %c0] size [1, 64]
                              to %lx_a[%c0, %c0] size [1, 64]
            : memref<?x?xf16, #ktdf.memory_space<LX>>,
              memref<?x?xf16, #ktdf.memory_space<LX>>
        }
        ktdf.stage depends_in(%t1 : !ktdf.token) depends_out(%t2 : !ktdf.token) {
          ktdf.data_transfer from %lx_a[%c0, %c0] size [1, 64]
                              to %lx_c[%c0, %c0] size [1, 64]
            : memref<?x?xf16, #ktdf.memory_space<LX>>,
              memref<?x?xf16, #ktdf.memory_space<LX>>
        }
      }
    }
  }
  return
}
MLIR
build/scheduler/bin/scheduler -double-buffering /tmp/db-smoke.mlir
```

Expected: the printed IR shows two `memref.alloc` pairs (one for `%lx_a`, one for `%lx_c`) just before `scf.for %m0`, two `memref.dealloc` pairs after, a `ktdf.buffer_phase(%m0, %n0)` and two `ktdf.select_memref` ops just before `ktdf.pipeline`, the pipeline carrying `modulo(size : 2)`, and the `ktdf.private` region only yielding the two tokens.

(Note: this smoke test's "producer" is contrived — `%lx_a` is both source and dest in stage 0 — so the producer/consumer detection should reject `%lx_a`. To force a candidate, edit the inputs so stage 0 only writes `%lx_a` and stage 1 only reads it. Use this iteration to debug; the formal positive lit tests come in Task 13.)

- [ ] **Step 5: Run full lit suite**

```bash
ninja -C build check-scheduler
```

Expected: all green. Existing tests don't trigger this pass (it's only registered, not in the default pipeline yet — wiring happens in Task 12).

- [ ] **Step 6: Commit**

```bash
git add src/Transform/DoubleBuffering.cpp
git commit -m "$(cat <<'EOF'
double-buffering: rewrite uses, set modulo, prune ktdf.private

Replaces every use of each original yielded buffer inside the pipeline's
region with the corresponding ktdf.select_memref result. Sets
modulo(size: 2) on the pipeline. Erases the dropped slots from
ktdf.private and ktdf.private_yield (recreating the ktdf.private op
with the reduced result type list) and erases the orphaned memref.alloc
ops in the private region.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 12: Wire `-double-buffering` into the driver pipeline

**Goal:** Add the pass to `buildKTDPToDFIRPipeline` immediately after `createParallelizeLoopsAcrossInstancesPass`. Confirm the kEmitDFIR pipeline registration still works.

**Files:**
- Modify: `tools/scheduler/scheduler-pipeline.cpp`

- [ ] **Step 1: Add the pass to the driver**

In `tools/scheduler/scheduler-pipeline.cpp`, modify `buildKTDPToDFIRPipeline` (around line 43–51) so the body is:

```cpp
void buildKTDPToDFIRPipeline(
    mlir::OpPassManager& pm,
    const scheduler::SchedulerExtContext& scheduler_ctx) {
  pm.addPass(createComputeGroupExtractionPass());
  pm.addPass(mlir::createSymbolDCEPass());
  pm.addPass(createConstructThreeStagePipelinePass(scheduler_ctx));
  pm.addPass(createParallelizeLoopsAcrossInstancesPass(scheduler_ctx));
  pm.addPass(createDoubleBufferingPass(scheduler_ctx));
  // todo: pm.addPass(createTileSCFForLoopsPass());
}
```

- [ ] **Step 2: Build**

```bash
ninja -C build
```

Expected: clean build.

- [ ] **Step 3: Run full lit suite**

```bash
ninja -C build check-scheduler
```

Expected: all green. Any existing test that goes through `kEmitDFIR` will now also run the double-buffering pass; if a test breaks, the pass either should be a no-op (no eligible pipelines) or the test exposes a real bug. Inspect failures and fix in this task.

- [ ] **Step 4: Commit**

```bash
git add tools/scheduler/scheduler-pipeline.cpp
git commit -m "$(cat <<'EOF'
Wire -double-buffering into buildKTDPToDFIRPipeline

Inserts the pass immediately after parallelize-loops-across-instances.
Existing tests are unaffected for inputs that don't expose
double-buffering candidates.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 13: Add positive lit tests

**Goal:** Add positive lit tests covering the canonical cases per spec §6.

**Files:**
- Create: `test/Transform/DoubleBuffering/basic.mlir`
- Create: `test/Transform/DoubleBuffering/multiple-candidates.mlir`
- Create: `test/Transform/DoubleBuffering/producer-in-nested-pipeline.mlir`
- Create: `test/Transform/DoubleBuffering/consumer-in-nested-pipeline.mlir`
- Create: `test/Transform/DoubleBuffering/multiple-consumers.mlir`

- [ ] **Step 1: Create `test/Transform/DoubleBuffering/basic.mlir`**

```mlir
// RUN: scheduler -allow-unregistered-dialect %s -double-buffering | FileCheck %s

// Single LX buffer with one producer stage and one consumer stage in a
// pipeline whose scope is one scf.for. Expected: one alloc pair hoisted
// outside the loop, one buffer_phase + one select_memref before the
// pipeline, modulo(size: 2) set on the pipeline, ktdf.private now only
// yielding the tokens.

// CHECK-LABEL:   func.func @basic
// CHECK:           %[[X0:.+]] = memref.alloc() : memref<64xf16, #ktdf.memory_space<LX>>
// CHECK:           %[[X1:.+]] = memref.alloc() : memref<64xf16, #ktdf.memory_space<LX>>
// CHECK:           scf.for %[[I:.+]] = %{{.+}} to %{{.+}} step %{{.+}}
// CHECK:             %[[PHASE:.+]] = ktdf.buffer_phase(%[[I]]) {num_phases = 2 : i64} : index
// CHECK:             %[[SEL:.+]] = ktdf.select_memref %[[PHASE]][%[[X0]], %[[X1]]] : index -> memref<64xf16, #ktdf.memory_space<LX>>
// CHECK:             ktdf.pipeline modulo(size : 2)
// CHECK:               ktdf.private -> (!ktdf.token, !ktdf.token)
// CHECK:               ktdf.stage depends_in(none) depends_out(
// CHECK:                 ktdf.data_transfer from %{{.+}}[%{{.+}}] size {{.*}}to %[[SEL]][%{{.+}}] size
// CHECK:               ktdf.stage depends_in
// CHECK:                 ktdf.data_transfer from %[[SEL]][%{{.+}}] size {{.*}}to %{{.+}}[%{{.+}}] size
// CHECK:           memref.dealloc %[[X0]] : memref<64xf16, #ktdf.memory_space<LX>>
// CHECK:           memref.dealloc %[[X1]] : memref<64xf16, #ktdf.memory_space<LX>>

func.func @basic(%src: memref<64xf16, #ktdf.memory_space<HBM>>,
                 %dst: memref<64xf16, #ktdf.memory_space<HBM>>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  scf.for %i = %c0 to %c8 step %c1 {
    ktdf.pipeline {
      %lx, %t1, %t2 = ktdf.private -> (
          memref<64xf16, #ktdf.memory_space<LX>>,
          !ktdf.token, !ktdf.token) {
        %a = memref.alloc() : memref<64xf16, #ktdf.memory_space<LX>>
        %tk1 = ktdf.create_token : !ktdf.token
        %tk2 = ktdf.create_token : !ktdf.token
        ktdf.private_yield %a, %tk1, %tk2
          : memref<64xf16, #ktdf.memory_space<LX>>, !ktdf.token, !ktdf.token
      }
      ktdf.stage depends_in(none) depends_out(%t1 : !ktdf.token) {
        ktdf.data_transfer from %src[%c0] size [64] to %lx[%c0] size [64]
          : memref<64xf16, #ktdf.memory_space<HBM>>,
            memref<64xf16, #ktdf.memory_space<LX>>
      }
      ktdf.stage depends_in(%t1 : !ktdf.token) depends_out(%t2 : !ktdf.token) {
        ktdf.data_transfer from %lx[%c0] size [64] to %dst[%c0] size [64]
          : memref<64xf16, #ktdf.memory_space<LX>>,
            memref<64xf16, #ktdf.memory_space<HBM>>
      }
    }
  }
  return
}
```

- [ ] **Step 2: Create `test/Transform/DoubleBuffering/multiple-candidates.mlir`**

```mlir
// RUN: scheduler -allow-unregistered-dialect %s -double-buffering | FileCheck %s

// Two LX buffers in the same pipeline. Expected: two alloc pairs, ONE
// buffer_phase shared across two select_memref ops, modulo(size: 2)
// once on the pipeline.

// CHECK-LABEL:   func.func @two_buffers
// CHECK:           memref.alloc()
// CHECK:           memref.alloc()
// CHECK:           memref.alloc()
// CHECK:           memref.alloc()
// CHECK:           scf.for %[[I:.+]] =
// CHECK:             %[[PHASE:.+]] = ktdf.buffer_phase(%[[I]]) {num_phases = 2 : i64} : index
// CHECK:             ktdf.select_memref %[[PHASE]][
// CHECK:             ktdf.select_memref %[[PHASE]][
// CHECK:             ktdf.pipeline modulo(size : 2)
// CHECK-NOT:         ktdf.buffer_phase

func.func @two_buffers(%a_src: memref<64xf16, #ktdf.memory_space<HBM>>,
                       %b_src: memref<64xf16, #ktdf.memory_space<HBM>>,
                       %dst: memref<64xf16, #ktdf.memory_space<HBM>>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  scf.for %i = %c0 to %c8 step %c1 {
    ktdf.pipeline {
      %lxa, %lxb, %t1, %t2, %t3 = ktdf.private -> (
          memref<64xf16, #ktdf.memory_space<LX>>,
          memref<64xf16, #ktdf.memory_space<LX>>,
          !ktdf.token, !ktdf.token, !ktdf.token) {
        %a = memref.alloc() : memref<64xf16, #ktdf.memory_space<LX>>
        %b = memref.alloc() : memref<64xf16, #ktdf.memory_space<LX>>
        %tk1 = ktdf.create_token : !ktdf.token
        %tk2 = ktdf.create_token : !ktdf.token
        %tk3 = ktdf.create_token : !ktdf.token
        ktdf.private_yield %a, %b, %tk1, %tk2, %tk3
          : memref<64xf16, #ktdf.memory_space<LX>>,
            memref<64xf16, #ktdf.memory_space<LX>>,
            !ktdf.token, !ktdf.token, !ktdf.token
      }
      ktdf.stage depends_in(none) depends_out(%t1 : !ktdf.token) {
        ktdf.data_transfer from %a_src[%c0] size [64] to %lxa[%c0] size [64]
          : memref<64xf16, #ktdf.memory_space<HBM>>,
            memref<64xf16, #ktdf.memory_space<LX>>
        ktdf.data_transfer from %b_src[%c0] size [64] to %lxb[%c0] size [64]
          : memref<64xf16, #ktdf.memory_space<HBM>>,
            memref<64xf16, #ktdf.memory_space<LX>>
      }
      ktdf.stage depends_in(%t1 : !ktdf.token) depends_out(%t2 : !ktdf.token) {
        ktdf.data_transfer from %lxa[%c0] size [64] to %dst[%c0] size [64]
          : memref<64xf16, #ktdf.memory_space<LX>>,
            memref<64xf16, #ktdf.memory_space<HBM>>
      }
      ktdf.stage depends_in(%t2 : !ktdf.token) depends_out(%t3 : !ktdf.token) {
        ktdf.data_transfer from %lxb[%c0] size [64] to %dst[%c0] size [64]
          : memref<64xf16, #ktdf.memory_space<LX>>,
            memref<64xf16, #ktdf.memory_space<HBM>>
      }
    }
  }
  return
}
```

- [ ] **Step 3: Create `test/Transform/DoubleBuffering/producer-in-nested-pipeline.mlir`**

```mlir
// RUN: scheduler -allow-unregistered-dialect %s -double-buffering | FileCheck %s

// Producer transfer is deeply nested inside a sibling stage's nested pipeline.
// Detection must walk into nested ops; transformation is unchanged at the
// outer pipeline level.

// CHECK-LABEL:   func.func @producer_nested
// CHECK:           memref.alloc() : memref<64xf16, #ktdf.memory_space<LX>>
// CHECK:           memref.alloc() : memref<64xf16, #ktdf.memory_space<LX>>
// CHECK:           ktdf.buffer_phase
// CHECK:           %[[SEL:.+]] = ktdf.select_memref
// CHECK:           ktdf.pipeline modulo(size : 2)
// CHECK:             ktdf.stage depends_in(none) depends_out
// CHECK:               ktdf.pipeline
// CHECK:                 ktdf.stage
// CHECK:                   ktdf.data_transfer from %{{.+}}[%{{.+}}] size {{.*}}to %[[SEL]][%{{.+}}] size

func.func @producer_nested(%src: memref<64xf16, #ktdf.memory_space<HBM>>,
                           %dst: memref<64xf16, #ktdf.memory_space<HBM>>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  scf.for %i = %c0 to %c8 step %c1 {
    ktdf.pipeline {
      %lx, %t1, %t2 = ktdf.private -> (
          memref<64xf16, #ktdf.memory_space<LX>>,
          !ktdf.token, !ktdf.token) {
        %a = memref.alloc() : memref<64xf16, #ktdf.memory_space<LX>>
        %tk1 = ktdf.create_token : !ktdf.token
        %tk2 = ktdf.create_token : !ktdf.token
        ktdf.private_yield %a, %tk1, %tk2
          : memref<64xf16, #ktdf.memory_space<LX>>, !ktdf.token, !ktdf.token
      }
      ktdf.stage depends_in(none) depends_out(%t1 : !ktdf.token) {
        ktdf.pipeline {
          ktdf.stage depends_in(none) depends_out(none) {
            ktdf.data_transfer from %src[%c0] size [64] to %lx[%c0] size [64]
              : memref<64xf16, #ktdf.memory_space<HBM>>,
                memref<64xf16, #ktdf.memory_space<LX>>
          }
        }
      }
      ktdf.stage depends_in(%t1 : !ktdf.token) depends_out(%t2 : !ktdf.token) {
        ktdf.data_transfer from %lx[%c0] size [64] to %dst[%c0] size [64]
          : memref<64xf16, #ktdf.memory_space<LX>>,
            memref<64xf16, #ktdf.memory_space<HBM>>
      }
    }
  }
  return
}
```

- [ ] **Step 4: Create `test/Transform/DoubleBuffering/consumer-in-nested-pipeline.mlir`**

```mlir
// RUN: scheduler -allow-unregistered-dialect %s -double-buffering | FileCheck %s

// Consumer transfer is deeply nested inside a sibling stage's nested pipeline.

// CHECK-LABEL:   func.func @consumer_nested
// CHECK:           memref.alloc() : memref<64xf16, #ktdf.memory_space<LX>>
// CHECK:           memref.alloc() : memref<64xf16, #ktdf.memory_space<LX>>
// CHECK:           %[[SEL:.+]] = ktdf.select_memref
// CHECK:           ktdf.pipeline modulo(size : 2)
// CHECK:             ktdf.stage
// CHECK:               ktdf.data_transfer from %{{.+}}[%{{.+}}] size {{.*}}to %[[SEL]][%{{.+}}] size
// CHECK:             ktdf.stage
// CHECK:               ktdf.pipeline
// CHECK:                 ktdf.stage
// CHECK:                   ktdf.data_transfer from %[[SEL]][%{{.+}}] size {{.*}}to %{{.+}}[%{{.+}}] size

func.func @consumer_nested(%src: memref<64xf16, #ktdf.memory_space<HBM>>,
                           %dst: memref<64xf16, #ktdf.memory_space<HBM>>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  scf.for %i = %c0 to %c8 step %c1 {
    ktdf.pipeline {
      %lx, %t1, %t2 = ktdf.private -> (
          memref<64xf16, #ktdf.memory_space<LX>>,
          !ktdf.token, !ktdf.token) {
        %a = memref.alloc() : memref<64xf16, #ktdf.memory_space<LX>>
        %tk1 = ktdf.create_token : !ktdf.token
        %tk2 = ktdf.create_token : !ktdf.token
        ktdf.private_yield %a, %tk1, %tk2
          : memref<64xf16, #ktdf.memory_space<LX>>, !ktdf.token, !ktdf.token
      }
      ktdf.stage depends_in(none) depends_out(%t1 : !ktdf.token) {
        ktdf.data_transfer from %src[%c0] size [64] to %lx[%c0] size [64]
          : memref<64xf16, #ktdf.memory_space<HBM>>,
            memref<64xf16, #ktdf.memory_space<LX>>
      }
      ktdf.stage depends_in(%t1 : !ktdf.token) depends_out(%t2 : !ktdf.token) {
        ktdf.pipeline {
          ktdf.stage depends_in(none) depends_out(none) {
            ktdf.data_transfer from %lx[%c0] size [64] to %dst[%c0] size [64]
              : memref<64xf16, #ktdf.memory_space<LX>>,
                memref<64xf16, #ktdf.memory_space<HBM>>
          }
        }
      }
    }
  }
  return
}
```

- [ ] **Step 5: Create `test/Transform/DoubleBuffering/multiple-consumers.mlir`**

```mlir
// RUN: scheduler -allow-unregistered-dialect %s -double-buffering | FileCheck %s

// One producer stage, two consumer stages reading the same buffer.

// CHECK-LABEL:   func.func @two_consumers
// CHECK:           memref.alloc() : memref<64xf16, #ktdf.memory_space<LX>>
// CHECK:           memref.alloc() : memref<64xf16, #ktdf.memory_space<LX>>
// CHECK:           %[[SEL:.+]] = ktdf.select_memref
// CHECK:           ktdf.pipeline modulo(size : 2)
// CHECK:             ktdf.stage depends_in(none) depends_out
// CHECK:               ktdf.data_transfer {{.*}}to %[[SEL]]
// CHECK:             ktdf.stage
// CHECK:               ktdf.data_transfer from %[[SEL]]
// CHECK:             ktdf.stage
// CHECK:               ktdf.data_transfer from %[[SEL]]

func.func @two_consumers(%src: memref<64xf16, #ktdf.memory_space<HBM>>,
                         %dst1: memref<64xf16, #ktdf.memory_space<HBM>>,
                         %dst2: memref<64xf16, #ktdf.memory_space<HBM>>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  scf.for %i = %c0 to %c8 step %c1 {
    ktdf.pipeline {
      %lx, %t1, %t2, %t3 = ktdf.private -> (
          memref<64xf16, #ktdf.memory_space<LX>>,
          !ktdf.token, !ktdf.token, !ktdf.token) {
        %a = memref.alloc() : memref<64xf16, #ktdf.memory_space<LX>>
        %tk1 = ktdf.create_token : !ktdf.token
        %tk2 = ktdf.create_token : !ktdf.token
        %tk3 = ktdf.create_token : !ktdf.token
        ktdf.private_yield %a, %tk1, %tk2, %tk3
          : memref<64xf16, #ktdf.memory_space<LX>>, !ktdf.token, !ktdf.token, !ktdf.token
      }
      ktdf.stage depends_in(none) depends_out(%t1 : !ktdf.token) {
        ktdf.data_transfer from %src[%c0] size [64] to %lx[%c0] size [64]
          : memref<64xf16, #ktdf.memory_space<HBM>>,
            memref<64xf16, #ktdf.memory_space<LX>>
      }
      ktdf.stage depends_in(%t1 : !ktdf.token) depends_out(%t2 : !ktdf.token) {
        ktdf.data_transfer from %lx[%c0] size [64] to %dst1[%c0] size [64]
          : memref<64xf16, #ktdf.memory_space<LX>>,
            memref<64xf16, #ktdf.memory_space<HBM>>
      }
      ktdf.stage depends_in(%t1 : !ktdf.token) depends_out(%t3 : !ktdf.token) {
        ktdf.data_transfer from %lx[%c0] size [64] to %dst2[%c0] size [64]
          : memref<64xf16, #ktdf.memory_space<LX>>,
            memref<64xf16, #ktdf.memory_space<HBM>>
      }
    }
  }
  return
}
```

- [ ] **Step 6: Run all five new tests**

```bash
for f in test/Transform/DoubleBuffering/basic.mlir \
         test/Transform/DoubleBuffering/multiple-candidates.mlir \
         test/Transform/DoubleBuffering/producer-in-nested-pipeline.mlir \
         test/Transform/DoubleBuffering/consumer-in-nested-pipeline.mlir \
         test/Transform/DoubleBuffering/multiple-consumers.mlir; do
  build/scheduler/bin/scheduler -allow-unregistered-dialect "$f" -double-buffering | \
    /Users/prasanth-work/Documents/code/external/llvm-standalone-3/build/bin/FileCheck "$f" || \
    { echo "FAILED: $f"; break; }
done
```

Expected: each prints nothing (FileCheck silent on success). If the printer's output has different `memref.alloc` ordering or different `CHECK:` cadence, regenerate the `CHECK` lines from the actual output.

- [ ] **Step 7: Run full lit suite**

```bash
ninja -C build check-scheduler
```

Expected: all green.

- [ ] **Step 8: Commit**

```bash
git add test/Transform/DoubleBuffering/basic.mlir \
        test/Transform/DoubleBuffering/multiple-candidates.mlir \
        test/Transform/DoubleBuffering/producer-in-nested-pipeline.mlir \
        test/Transform/DoubleBuffering/consumer-in-nested-pipeline.mlir \
        test/Transform/DoubleBuffering/multiple-consumers.mlir
git commit -m "$(cat <<'EOF'
Add positive lit tests for -double-buffering

Covers the canonical 5 cases per spec §6 positives: single buffer,
multiple buffers in one pipeline (shared buffer_phase), producer in a
nested pipeline, consumer in a nested pipeline, and multiple consumers
of one buffer.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 14: Add negative lit tests

**Goal:** Add the spec §6 negative tests verifying the pass is a no-op when each guard fires.

**Files:**
- Create: `test/Transform/DoubleBuffering/negative-no-scope.mlir`
- Create: `test/Transform/DoubleBuffering/negative-ineligible-memory-space.mlir`
- Create: `test/Transform/DoubleBuffering/negative-producer-equals-consumer.mlir`
- Create: `test/Transform/DoubleBuffering/negative-multi-producer.mlir`
- Create: `test/Transform/DoubleBuffering/negative-non-alloc-defining-op.mlir`
- Create: `test/Transform/DoubleBuffering/negative-alloc-multi-use.mlir`
- Create: `test/Transform/DoubleBuffering/negative-dynamic-size-not-dominating.mlir`
- Create: `test/Transform/DoubleBuffering/negative-pre-existing-modulo.mlir`

For each negative test, the structure is identical: feed the input, check that no `ktdf.buffer_phase`, `ktdf.select_memref`, or `modulo` appears in the output. The general template is:

```mlir
// RUN: scheduler -allow-unregistered-dialect %s -double-buffering | FileCheck %s

// CHECK-NOT: ktdf.buffer_phase
// CHECK-NOT: ktdf.select_memref
// CHECK-NOT: modulo(size

<<input MLIR matching the rejection scenario>>
```

- [ ] **Step 1: Create `test/Transform/DoubleBuffering/negative-no-scope.mlir`**

A pipeline at function scope (no enclosing `scf.for`).

```mlir
// RUN: scheduler -allow-unregistered-dialect %s -double-buffering | FileCheck %s

// CHECK-NOT: ktdf.buffer_phase
// CHECK-NOT: ktdf.select_memref
// CHECK-NOT: modulo(size

func.func @no_scope(%src: memref<64xf16, #ktdf.memory_space<HBM>>,
                   %dst: memref<64xf16, #ktdf.memory_space<HBM>>) {
  %c0 = arith.constant 0 : index
  ktdf.pipeline {
    %lx, %t1, %t2 = ktdf.private -> (
        memref<64xf16, #ktdf.memory_space<LX>>,
        !ktdf.token, !ktdf.token) {
      %a = memref.alloc() : memref<64xf16, #ktdf.memory_space<LX>>
      %tk1 = ktdf.create_token : !ktdf.token
      %tk2 = ktdf.create_token : !ktdf.token
      ktdf.private_yield %a, %tk1, %tk2
        : memref<64xf16, #ktdf.memory_space<LX>>, !ktdf.token, !ktdf.token
    }
    ktdf.stage depends_in(none) depends_out(%t1 : !ktdf.token) {
      ktdf.data_transfer from %src[%c0] size [64] to %lx[%c0] size [64]
        : memref<64xf16, #ktdf.memory_space<HBM>>,
          memref<64xf16, #ktdf.memory_space<LX>>
    }
    ktdf.stage depends_in(%t1 : !ktdf.token) depends_out(%t2 : !ktdf.token) {
      ktdf.data_transfer from %lx[%c0] size [64] to %dst[%c0] size [64]
        : memref<64xf16, #ktdf.memory_space<LX>>,
          memref<64xf16, #ktdf.memory_space<HBM>>
    }
  }
  return
}
```

- [ ] **Step 2: Create `test/Transform/DoubleBuffering/negative-ineligible-memory-space.mlir`**

Buffer in HBM (not LX). The candidate fails `isEligibleForDoubleBuffering`.

```mlir
// RUN: scheduler -allow-unregistered-dialect %s -double-buffering | FileCheck %s

// CHECK-NOT: ktdf.buffer_phase
// CHECK-NOT: ktdf.select_memref
// CHECK-NOT: modulo(size

func.func @ineligible_memspace(%src: memref<64xf16, #ktdf.memory_space<HBM>>,
                               %dst: memref<64xf16, #ktdf.memory_space<HBM>>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  scf.for %i = %c0 to %c8 step %c1 {
    ktdf.pipeline {
      %hbm, %t1, %t2 = ktdf.private -> (
          memref<64xf16, #ktdf.memory_space<HBM>>,
          !ktdf.token, !ktdf.token) {
        %a = memref.alloc() : memref<64xf16, #ktdf.memory_space<HBM>>
        %tk1 = ktdf.create_token : !ktdf.token
        %tk2 = ktdf.create_token : !ktdf.token
        ktdf.private_yield %a, %tk1, %tk2
          : memref<64xf16, #ktdf.memory_space<HBM>>, !ktdf.token, !ktdf.token
      }
      ktdf.stage depends_in(none) depends_out(%t1 : !ktdf.token) {
        ktdf.data_transfer from %src[%c0] size [64] to %hbm[%c0] size [64]
          : memref<64xf16, #ktdf.memory_space<HBM>>,
            memref<64xf16, #ktdf.memory_space<HBM>>
      }
      ktdf.stage depends_in(%t1 : !ktdf.token) depends_out(%t2 : !ktdf.token) {
        ktdf.data_transfer from %hbm[%c0] size [64] to %dst[%c0] size [64]
          : memref<64xf16, #ktdf.memory_space<HBM>>,
            memref<64xf16, #ktdf.memory_space<HBM>>
      }
    }
  }
  return
}
```

- [ ] **Step 3: Create `test/Transform/DoubleBuffering/negative-producer-equals-consumer.mlir`**

A single sibling stage both writes and reads the buffer.

```mlir
// RUN: scheduler -allow-unregistered-dialect %s -double-buffering | FileCheck %s

// CHECK-NOT: ktdf.buffer_phase
// CHECK-NOT: ktdf.select_memref
// CHECK-NOT: modulo(size

func.func @producer_equals_consumer(%src: memref<64xf16, #ktdf.memory_space<HBM>>,
                                     %dst: memref<64xf16, #ktdf.memory_space<HBM>>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  scf.for %i = %c0 to %c8 step %c1 {
    ktdf.pipeline {
      %lx, %t1 = ktdf.private -> (
          memref<64xf16, #ktdf.memory_space<LX>>,
          !ktdf.token) {
        %a = memref.alloc() : memref<64xf16, #ktdf.memory_space<LX>>
        %tk1 = ktdf.create_token : !ktdf.token
        ktdf.private_yield %a, %tk1
          : memref<64xf16, #ktdf.memory_space<LX>>, !ktdf.token
      }
      ktdf.stage depends_in(none) depends_out(%t1 : !ktdf.token) {
        ktdf.data_transfer from %src[%c0] size [64] to %lx[%c0] size [64]
          : memref<64xf16, #ktdf.memory_space<HBM>>,
            memref<64xf16, #ktdf.memory_space<LX>>
        ktdf.data_transfer from %lx[%c0] size [64] to %dst[%c0] size [64]
          : memref<64xf16, #ktdf.memory_space<LX>>,
            memref<64xf16, #ktdf.memory_space<HBM>>
      }
    }
  }
  return
}
```

- [ ] **Step 4: Create `test/Transform/DoubleBuffering/negative-multi-producer.mlir`**

Two sibling stages each write the same buffer.

```mlir
// RUN: scheduler -allow-unregistered-dialect %s -double-buffering | FileCheck %s

// CHECK-NOT: ktdf.buffer_phase
// CHECK-NOT: ktdf.select_memref
// CHECK-NOT: modulo(size

func.func @multi_producer(%src1: memref<64xf16, #ktdf.memory_space<HBM>>,
                          %src2: memref<64xf16, #ktdf.memory_space<HBM>>,
                          %dst: memref<64xf16, #ktdf.memory_space<HBM>>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  scf.for %i = %c0 to %c8 step %c1 {
    ktdf.pipeline {
      %lx, %t1, %t2, %t3 = ktdf.private -> (
          memref<64xf16, #ktdf.memory_space<LX>>,
          !ktdf.token, !ktdf.token, !ktdf.token) {
        %a = memref.alloc() : memref<64xf16, #ktdf.memory_space<LX>>
        %tk1 = ktdf.create_token : !ktdf.token
        %tk2 = ktdf.create_token : !ktdf.token
        %tk3 = ktdf.create_token : !ktdf.token
        ktdf.private_yield %a, %tk1, %tk2, %tk3
          : memref<64xf16, #ktdf.memory_space<LX>>, !ktdf.token, !ktdf.token, !ktdf.token
      }
      ktdf.stage depends_in(none) depends_out(%t1 : !ktdf.token) {
        ktdf.data_transfer from %src1[%c0] size [64] to %lx[%c0] size [64]
          : memref<64xf16, #ktdf.memory_space<HBM>>,
            memref<64xf16, #ktdf.memory_space<LX>>
      }
      ktdf.stage depends_in(none) depends_out(%t2 : !ktdf.token) {
        ktdf.data_transfer from %src2[%c0] size [64] to %lx[%c0] size [64]
          : memref<64xf16, #ktdf.memory_space<HBM>>,
            memref<64xf16, #ktdf.memory_space<LX>>
      }
      ktdf.stage depends_in(%t1, %t2 : !ktdf.token, !ktdf.token) depends_out(%t3 : !ktdf.token) {
        ktdf.data_transfer from %lx[%c0] size [64] to %dst[%c0] size [64]
          : memref<64xf16, #ktdf.memory_space<LX>>,
            memref<64xf16, #ktdf.memory_space<HBM>>
      }
    }
  }
  return
}
```

- [ ] **Step 5: Create `test/Transform/DoubleBuffering/negative-non-alloc-defining-op.mlir`**

The yielded value's defining op is not `memref.alloc` (using `memref.cast` here).

```mlir
// RUN: scheduler -allow-unregistered-dialect %s -double-buffering | FileCheck %s

// CHECK-NOT: ktdf.buffer_phase
// CHECK-NOT: ktdf.select_memref
// CHECK-NOT: modulo(size

func.func @non_alloc_defining_op(%src: memref<64xf16, #ktdf.memory_space<HBM>>,
                                  %dst: memref<64xf16, #ktdf.memory_space<HBM>>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  scf.for %i = %c0 to %c8 step %c1 {
    ktdf.pipeline {
      %lx, %t1, %t2 = ktdf.private -> (
          memref<*xf16, #ktdf.memory_space<LX>>,
          !ktdf.token, !ktdf.token) {
        %a = memref.alloc() : memref<64xf16, #ktdf.memory_space<LX>>
        %a_cast = memref.cast %a : memref<64xf16, #ktdf.memory_space<LX>> to memref<*xf16, #ktdf.memory_space<LX>>
        %tk1 = ktdf.create_token : !ktdf.token
        %tk2 = ktdf.create_token : !ktdf.token
        ktdf.private_yield %a_cast, %tk1, %tk2
          : memref<*xf16, #ktdf.memory_space<LX>>, !ktdf.token, !ktdf.token
      }
      ktdf.stage depends_in(none) depends_out(%t1 : !ktdf.token) {
        "test.write"(%lx) : (memref<*xf16, #ktdf.memory_space<LX>>) -> ()
      }
      ktdf.stage depends_in(%t1 : !ktdf.token) depends_out(%t2 : !ktdf.token) {
        "test.read"(%lx) : (memref<*xf16, #ktdf.memory_space<LX>>) -> ()
      }
    }
  }
  return
}
```

(Note: this test uses `test.write` / `test.read` instead of `ktdf.data_transfer` because the cast result type is unranked. Since the defining-op check fires before producer/consumer scan, the `test.*` ops don't matter — they just satisfy the IR's structure. The `CHECK-NOT` lines guarantee no transformation.)

- [ ] **Step 6: Create `test/Transform/DoubleBuffering/negative-alloc-multi-use.mlir`**

The `memref.alloc` has another use inside the private region beyond the yield.

```mlir
// RUN: scheduler -allow-unregistered-dialect %s -double-buffering | FileCheck %s

// CHECK-NOT: ktdf.buffer_phase
// CHECK-NOT: ktdf.select_memref
// CHECK-NOT: modulo(size

func.func @alloc_multi_use(%src: memref<64xf16, #ktdf.memory_space<HBM>>,
                            %dst: memref<64xf16, #ktdf.memory_space<HBM>>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  scf.for %i = %c0 to %c8 step %c1 {
    ktdf.pipeline {
      %lx, %t1, %t2 = ktdf.private -> (
          memref<64xf16, #ktdf.memory_space<LX>>,
          !ktdf.token, !ktdf.token) {
        %a = memref.alloc() : memref<64xf16, #ktdf.memory_space<LX>>
        "test.touch"(%a) : (memref<64xf16, #ktdf.memory_space<LX>>) -> ()
        %tk1 = ktdf.create_token : !ktdf.token
        %tk2 = ktdf.create_token : !ktdf.token
        ktdf.private_yield %a, %tk1, %tk2
          : memref<64xf16, #ktdf.memory_space<LX>>, !ktdf.token, !ktdf.token
      }
      ktdf.stage depends_in(none) depends_out(%t1 : !ktdf.token) {
        ktdf.data_transfer from %src[%c0] size [64] to %lx[%c0] size [64]
          : memref<64xf16, #ktdf.memory_space<HBM>>,
            memref<64xf16, #ktdf.memory_space<LX>>
      }
      ktdf.stage depends_in(%t1 : !ktdf.token) depends_out(%t2 : !ktdf.token) {
        ktdf.data_transfer from %lx[%c0] size [64] to %dst[%c0] size [64]
          : memref<64xf16, #ktdf.memory_space<LX>>,
            memref<64xf16, #ktdf.memory_space<HBM>>
      }
    }
  }
  return
}
```

- [ ] **Step 7: Create `test/Transform/DoubleBuffering/negative-dynamic-size-not-dominating.mlir`**

A dynamic size operand is defined inside the scope's loops.

```mlir
// RUN: scheduler -allow-unregistered-dialect %s -double-buffering | FileCheck %s

// CHECK-NOT: ktdf.buffer_phase
// CHECK-NOT: ktdf.select_memref
// CHECK-NOT: modulo(size

func.func @dynamic_size_not_dominating(%src: memref<?xf16, #ktdf.memory_space<HBM>>,
                                        %dst: memref<?xf16, #ktdf.memory_space<HBM>>,
                                        %M: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  scf.for %i = %c0 to %c8 step %c1 {
    // Compute size INSIDE the loop — does not dominate the hoist target.
    %size = arith.addi %i, %M : index
    ktdf.pipeline {
      %lx, %t1, %t2 = ktdf.private -> (
          memref<?xf16, #ktdf.memory_space<LX>>,
          !ktdf.token, !ktdf.token) {
        %a = memref.alloc(%size) : memref<?xf16, #ktdf.memory_space<LX>>
        %tk1 = ktdf.create_token : !ktdf.token
        %tk2 = ktdf.create_token : !ktdf.token
        ktdf.private_yield %a, %tk1, %tk2
          : memref<?xf16, #ktdf.memory_space<LX>>, !ktdf.token, !ktdf.token
      }
      ktdf.stage depends_in(none) depends_out(%t1 : !ktdf.token) {
        ktdf.data_transfer from %src[%c0] size [%size] to %lx[%c0] size [%size]
          : memref<?xf16, #ktdf.memory_space<HBM>>,
            memref<?xf16, #ktdf.memory_space<LX>>
      }
      ktdf.stage depends_in(%t1 : !ktdf.token) depends_out(%t2 : !ktdf.token) {
        ktdf.data_transfer from %lx[%c0] size [%size] to %dst[%c0] size [%size]
          : memref<?xf16, #ktdf.memory_space<LX>>,
            memref<?xf16, #ktdf.memory_space<HBM>>
      }
    }
  }
  return
}
```

- [ ] **Step 8: Create `test/Transform/DoubleBuffering/negative-pre-existing-modulo.mlir`**

Pipeline already has `modulo(size: 2)`.

```mlir
// RUN: scheduler -allow-unregistered-dialect %s -double-buffering | FileCheck %s

// CHECK-COUNT-1: modulo(size
// CHECK-NOT: ktdf.buffer_phase
// CHECK-NOT: ktdf.select_memref

func.func @pre_existing_modulo(%src: memref<64xf16, #ktdf.memory_space<HBM>>,
                                %dst: memref<64xf16, #ktdf.memory_space<HBM>>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  scf.for %i = %c0 to %c8 step %c1 {
    ktdf.pipeline modulo(size: 2) {
      %lx, %t1, %t2 = ktdf.private -> (
          memref<64xf16, #ktdf.memory_space<LX>>,
          !ktdf.token, !ktdf.token) {
        %a = memref.alloc() : memref<64xf16, #ktdf.memory_space<LX>>
        %tk1 = ktdf.create_token : !ktdf.token
        %tk2 = ktdf.create_token : !ktdf.token
        ktdf.private_yield %a, %tk1, %tk2
          : memref<64xf16, #ktdf.memory_space<LX>>, !ktdf.token, !ktdf.token
      }
      ktdf.stage depends_in(none) depends_out(%t1 : !ktdf.token) {
        ktdf.data_transfer from %src[%c0] size [64] to %lx[%c0] size [64]
          : memref<64xf16, #ktdf.memory_space<HBM>>,
            memref<64xf16, #ktdf.memory_space<LX>>
      }
      ktdf.stage depends_in(%t1 : !ktdf.token) depends_out(%t2 : !ktdf.token) {
        ktdf.data_transfer from %lx[%c0] size [64] to %dst[%c0] size [64]
          : memref<64xf16, #ktdf.memory_space<LX>>,
            memref<64xf16, #ktdf.memory_space<HBM>>
      }
    }
  }
  return
}
```

`CHECK-COUNT-1: modulo(size` ensures the original `modulo(size: 2)` is preserved exactly once and no second one is added. `CHECK-NOT` ensures no `buffer_phase` / `select_memref` is added.

- [ ] **Step 9: Run all eight new tests + full lit suite**

```bash
ninja -C build check-scheduler
```

Expected: all green.

- [ ] **Step 10: Commit**

```bash
git add test/Transform/DoubleBuffering/negative-*.mlir
git commit -m "$(cat <<'EOF'
Add negative lit tests for -double-buffering

Verifies the pass is a no-op when each guard fires: empty enclosing
scope, ineligible memory space, producer = consumer in the same stage,
multi-producer, defining op is not memref.alloc, alloc has multiple
uses, dynamic size operand not dominating the hoist target, and
pre-existing modulo on the pipeline.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 15: Add idempotence and debug-log lit tests

**Goal:** Add the remaining tests from spec §6.

**Files:**
- Create: `test/Transform/DoubleBuffering/idempotent.mlir`
- Create: `test/Transform/DoubleBuffering/debug-log.mlir`

- [ ] **Step 1: Create `test/Transform/DoubleBuffering/idempotent.mlir`**

```mlir
// Running the pass twice produces the same output as running once. The
// second run sees the pre-existing modulo and skips.

// RUN: scheduler -allow-unregistered-dialect %s -double-buffering > %t.once.mlir
// RUN: scheduler -allow-unregistered-dialect %t.once.mlir -double-buffering > %t.twice.mlir
// RUN: diff %t.once.mlir %t.twice.mlir

func.func @idempotent(%src: memref<64xf16, #ktdf.memory_space<HBM>>,
                      %dst: memref<64xf16, #ktdf.memory_space<HBM>>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  scf.for %i = %c0 to %c8 step %c1 {
    ktdf.pipeline {
      %lx, %t1, %t2 = ktdf.private -> (
          memref<64xf16, #ktdf.memory_space<LX>>,
          !ktdf.token, !ktdf.token) {
        %a = memref.alloc() : memref<64xf16, #ktdf.memory_space<LX>>
        %tk1 = ktdf.create_token : !ktdf.token
        %tk2 = ktdf.create_token : !ktdf.token
        ktdf.private_yield %a, %tk1, %tk2
          : memref<64xf16, #ktdf.memory_space<LX>>, !ktdf.token, !ktdf.token
      }
      ktdf.stage depends_in(none) depends_out(%t1 : !ktdf.token) {
        ktdf.data_transfer from %src[%c0] size [64] to %lx[%c0] size [64]
          : memref<64xf16, #ktdf.memory_space<HBM>>,
            memref<64xf16, #ktdf.memory_space<LX>>
      }
      ktdf.stage depends_in(%t1 : !ktdf.token) depends_out(%t2 : !ktdf.token) {
        ktdf.data_transfer from %lx[%c0] size [64] to %dst[%c0] size [64]
          : memref<64xf16, #ktdf.memory_space<LX>>,
            memref<64xf16, #ktdf.memory_space<HBM>>
      }
    }
  }
  return
}
```

- [ ] **Step 2: Create `test/Transform/DoubleBuffering/debug-log.mlir`**

```mlir
// Verify the pass emits debug output via LLVM_DEBUG when --debug-only is
// supplied.

// RUN: scheduler -allow-unregistered-dialect --debug-only=double-buffering %s -double-buffering 2>&1 | FileCheck %s

// CHECK: [double-buffering] starting

func.func @debug_log() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  scf.for %i = %c0 to %c8 step %c1 {
    ktdf.pipeline {
      ktdf.stage depends_in(none) depends_out(none) {
      }
    }
  }
  return
}
```

- [ ] **Step 3: Run both tests**

```bash
ninja -C build check-scheduler
```

Expected: all green.

- [ ] **Step 4: Commit**

```bash
git add test/Transform/DoubleBuffering/idempotent.mlir test/Transform/DoubleBuffering/debug-log.mlir
git commit -m "$(cat <<'EOF'
Add idempotence and debug-log lit tests for -double-buffering

Idempotent: running the pass twice equals running once (second run
sees pre-existing modulo and skips). Debug-log: verify the
'starting' LLVM_DEBUG line is emitted under --debug-only.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Self-Review

Spec coverage walk:

| Spec section | Covered by |
|---|---|
| §1 Scope & Goals | Tasks 1–15 (the entire plan) |
| §2.1 `ktdf.buffer_phase` | Task 3 |
| §2.2 `ktdf.select_memref` | Task 4 |
| §2.3 `modulo` attribute | Tasks 1–2 |
| §2.4 `isEligibleForDoubleBuffering` | Task 5 |
| §3 Detection a–f (shape) | Task 8 |
| §3 Detection g–k (producer/consumer + tokens) | Task 9 |
| §3 pipeline-level guards (modulo + scope) | Task 7 |
| §4 Transform step 1 (hoist) | Task 10 |
| §4 Transform steps 2–3 (phase + selectors) | Task 10 |
| §4 Transform steps 4–6 (rewrite uses, modulo, prune private) | Task 11 |
| §5 Pass plumbing | Task 6 (skeleton) + Task 12 (driver wiring) |
| §6 Positive tests | Task 13 |
| §6 Negative tests | Task 14 |
| §6 Idempotence + debug-log | Task 15 |
| §6 Round-trip tests | Tasks 2 (modulo), 3 (buffer_phase), 4 (select_memref) |

Placeholder scan: no "TBD/TODO/handle edge cases" present — each step contains either (a) the exact code/MLIR/CHECK lines or (b) a precise instruction with the expected output. The two cases where the plan acknowledges runtime variability (printer spacing in Task 2 Step 5; expected-error wording in Tasks 3/4 Step 5; CHECK regeneration in Task 13 Step 6) are explicitly called out with the regeneration command.

Type/method consistency check: `kDefaultNumPhases = 2` defined in Task 6 and used in Task 10 (`hoistAllocPair`) and Task 11 (`pipeline.setModuloAttr(...)` in `rewritePipeline`). `Candidate`/`CandidateShape` types defined in Tasks 8/9. `getModuloAttr()` (auto-generated by ODS) used in Task 7. `setModuloAttr()` (auto-generated by ODS) used in Task 11. `getNumPhases()` (auto-generated by ODS) used in Task 4 verifier. `getDestination()` / `getSource()` on `DataTransferOp` used in Task 9 (already exist per `KTDFOps.cpp:407-408`). `getDependsOut()` / `getDependsIn()` on `StageOp` used in Task 9 (already exist per `KTDFOps.cpp:258, 273`). `isEligibleForDoubleBuffering` declared in Task 5, used in Task 8.

Plan complete and saved to `docs/superpowers/plans/2026-05-18-double-buffering.md`. Two execution options:

1. **Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.
2. **Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints.

Which approach?
