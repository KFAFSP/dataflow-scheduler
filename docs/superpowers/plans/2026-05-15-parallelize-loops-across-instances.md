# `-parallelize-loops-across-instances` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a new MLIR pass `-parallelize-loops-across-instances` that finds `parallel_loop`-annotated `scf.for` loops enclosing per-corelet `ktdf.pipeline`s and rewrites them into `ktdf.parallel` ops with `distribute(num_instances = N)` (where N comes from `SchedulerExtContext::getNumCorelets()`).

**Architecture:** The pass walks pipelines pre-order. For each pipeline, it computes the union of `applicable_units` across the pipeline's immediate child stages (a new utility); rejects pipelines that touch core-shared units (L3LU/L3SU per the context); finds the outermost parallel loop in the pipeline's enclosing scope (reusing the existing scope-walk utility) whose statically-known trip count is divisible by `num_instances` (Pass 1) or otherwise just ≥ `num_instances` (Pass 2 fallback); and rewrites the chosen `scf.for` into `ktdf.parallel` by splicing the body, RAUW-ing the IV, and appending `ktdf.parallel_yield`. Two new context methods (`getNumCorelets`, `isPerCoreletUnit`) provide target abstraction. Two utilities live under `src/Transform/Common/` (`PipelineScope` is moved there from `ReusePromotion/Common/`; `ApplicableUnits` is new).

**Tech Stack:** MLIR (TableGen ODS for pass registration), C++17, CMake + Ninja, lit + FileCheck.

**Spec reference:** `docs/superpowers/specs/2026-05-15-parallelize-loops-across-instances-design.md`. Spec sections cited inline as `spec §N`.

**Pre-flight (read once before starting):**
- `docs/superpowers/specs/2026-05-15-parallelize-loops-across-instances-design.md` — the spec.
- `docs/Representing Corelet Parallelism in KTDF.md` — semantics of `ktdf.parallel`.
- `test/Transform/tmp.mlir` — the worked input/output example.
- `src/Transform/ReusePromotion/Common/PipelineScope.{cpp,hpp}` — the utility being moved.
- `src/Transform/ConstructThreeStagePipeline.cpp` — existing pattern for a pass that takes `SchedulerExtContext&` (lines 64-80, 144-160 for class layout; tail of file for the two `createXxx` factories).
- `src/Transform/BroadcastPromotion.cpp` — caller of `PipelineScope` whose includes need updating (lines around `ReusePromotion/Common/PipelineScope` import).
- `src/Transform/Passes.{td,h}` — pass registration patterns (BroadcastPromotionPass at `Passes.td:109` is the closest template).
- `src/Utils/SchedulerExtContext.{h,cpp}` — context methods being extended.
- `tools/scheduler/scheduler-pipeline.cpp` — driver pipeline; the new pass is added to `buildKTDPToDFIRPipeline` here.
- `docs/knowledge-base/project_coding_conventions.md` — Google C++ style, snake_case locals, trailing `_` for class members, camelBack methods, brace short single-statement bodies.

**Build & test commands:**

- Build: `ninja -C build`. Configure first if CMake changed: `./scripts/buildKTIRScheduler.sh && ninja -C build`.
- Full lit suite: `ninja -C build check-scheduler`.
- Single test: `build/scheduler/bin/scheduler -allow-unregistered-dialect <flag> path/to/test.mlir | /Users/prasanth-work/Documents/code/external/llvm-standalone-3/build/bin/FileCheck path/to/test.mlir`.
- Inspect raw printer output (no FileCheck): `build/scheduler/bin/scheduler <flag> path/to/test.mlir`.
- Verify-diagnostics file: `build/scheduler/bin/scheduler -allow-unregistered-dialect -split-input-file -verify-diagnostics path/to/test.mlir`.

**Conventions (apply throughout):**
- Local vars: `lower_bounds`, `num_loops`, `scope_loops` — snake_case.
- Methods/free functions: `camelBack` (`getNumCorelets`, `findCandidate`, `computePipelineApplicableUnits`).
- Class members: trailing `_` (`scheduler_ctx_`).
- Constants: `kFoo` (none expected here).
- Brace short single-statement `if`/`for` bodies.
- Pass `#define DEBUG_TYPE "parallelize-loops-across-instances"` for `LLVM_DEBUG` logging.

**Pre-state on the branch:**
- Branch `pc/coreletparallelism`. Latest HEAD is `ca23924` ("Add design doc for -parallelize-loops-across-instances pass").
- The branch has the merged `ktdf.parallel` op from PR #5. The pass relies on it.
- `git status` may show pre-existing untracked items (`.idea/`, `docs/knowledge-base/`, `skills/`, `src/.DS_Store`, `test/.lit_test_times.txt`, `test/Transform/BroadcastPromotion/`) — leave them alone.

---

## File Structure

| Path | Action | Responsibility |
|---|---|---|
| `src/Transform/Common/PipelineScope.{cpp,hpp}` | Create (moved from ReusePromotion/Common) | Pipeline enclosing scope walk. Namespace `ktdf::common`. |
| `src/Transform/Common/ApplicableUnits.{cpp,hpp}` | Create | `computePipelineApplicableUnits` — union over immediate child stages. Namespace `ktdf::common`. |
| `src/Transform/ReusePromotion/Common/PipelineScope.{cpp,hpp}` | Delete | Old location of moved file. |
| `src/Transform/CMakeLists.txt` | Modify | Add `Common/PipelineScope.cpp`, `Common/ApplicableUnits.cpp`, and `ParallelizeLoopsAcrossInstances.cpp` to source list. Drop `ReusePromotion/Common/PipelineScope.cpp`. |
| `src/Transform/BroadcastPromotion.cpp` | Modify | Update `#include` from `ReusePromotion/Common/PipelineScope.hpp` to `Common/PipelineScope.hpp`; rename `ktdf::reuse::` → `ktdf::common::` for `PipelineScope` references. |
| `src/Transform/ReusePromotion/Common/CandidateSearch.cpp` | Modify | Same include + namespace updates as BroadcastPromotion. |
| `src/Transform/ReusePromotion/Common/CandidateSearch.hpp` | Modify | Update `#include` and any `ktdf::reuse::PipelineEnclosingScope` reference. |
| `src/Utils/SchedulerExtContext.h` | Modify | Declare `virtual unsigned getNumCorelets() const;` and `virtual bool isPerCoreletUnit(mlir::ktdf::ComponentType) const;`. |
| `src/Utils/SchedulerExtContext.cpp` | Modify | Implement Spyre defaults: `getNumCorelets()` → `2`; `isPerCoreletUnit(SFP|LXLU|LXSU)` → `true`, `(L3LU|L3SU)` → `false`. |
| `src/Transform/Passes.td` | Modify | Add `def ParallelizeLoopsAcrossInstancesPass`. |
| `src/Transform/Passes.h` | Modify | Declare `createParallelizeLoopsAcrossInstancesPass(const SchedulerExtContext&)` and a no-arg overload that uses `dummyContext()`. |
| `src/Transform/ParallelizeLoopsAcrossInstances.cpp` | Create | Pass entry, analysis (private static helpers), transformation, debug logging. |
| `tools/scheduler/scheduler-pipeline.cpp` | Modify | Insert the pass into `buildKTDPToDFIRPipeline` after the relevant existing pass. |
| `test/Transforms/ParallelizeLoopsAcrossInstances/*.mlir` | Create | Lit tests (positive transformations, rejection cases, round-trip, idempotence, debug, fatal-error). |

The `Common/` directory is new. `ReusePromotion/Common/` retains its other four utilities (`InvarianceCheck`, `WriteSetScan`, `RegionClonePrune`, `CandidateSearch`).

---

## Task 1: Move `PipelineScope` to `src/Transform/Common/`

**Goal:** Relocate the existing scope-walk utility, rename its namespace from `ktdf::reuse` to `ktdf::common`, update all callers, and verify nothing regresses. The behavior is unchanged.

**Files:**
- Create: `src/Transform/Common/PipelineScope.hpp`
- Create: `src/Transform/Common/PipelineScope.cpp`
- Delete: `src/Transform/ReusePromotion/Common/PipelineScope.hpp`
- Delete: `src/Transform/ReusePromotion/Common/PipelineScope.cpp`
- Modify: `src/Transform/CMakeLists.txt`
- Modify: `src/Transform/BroadcastPromotion.cpp`
- Modify: `src/Transform/ReusePromotion/Common/CandidateSearch.cpp`
- Modify: `src/Transform/ReusePromotion/Common/CandidateSearch.hpp`

- [ ] **Step 1: Create the new directory and copy the files**

```bash
mkdir -p /Users/prasanth-work/Documents/code/KTIR-Scheduler/src/Transform/Common
cp /Users/prasanth-work/Documents/code/KTIR-Scheduler/src/Transform/ReusePromotion/Common/PipelineScope.hpp \
   /Users/prasanth-work/Documents/code/KTIR-Scheduler/src/Transform/Common/PipelineScope.hpp
cp /Users/prasanth-work/Documents/code/KTIR-Scheduler/src/Transform/ReusePromotion/Common/PipelineScope.cpp \
   /Users/prasanth-work/Documents/code/KTIR-Scheduler/src/Transform/Common/PipelineScope.cpp
```

- [ ] **Step 2: Update the new header — namespace and include guard**

Edit `src/Transform/Common/PipelineScope.hpp`. Change exactly the following:

1. Include guard from `KTIR_SCHEDULER_TRANSFORM_REUSEPROMOTION_PIPELINESCOPE_HPP` to `KTIR_SCHEDULER_TRANSFORM_COMMON_PIPELINESCOPE_HPP` (both the `#ifndef`/`#define` and the `#endif` comment).
2. Outer namespace from `namespace reuse {` to `namespace common {` (and matching `}  // namespace reuse` to `}  // namespace common`).

Final header looks like this (full file shown for unambiguity):

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
// Pipeline Enclosing Scope computation, shared across transform passes.
//===----------------------------------------------------------------------===//

#ifndef KTIR_SCHEDULER_TRANSFORM_COMMON_PIPELINESCOPE_HPP
#define KTIR_SCHEDULER_TRANSFORM_COMMON_PIPELINESCOPE_HPP

#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "src/Dialects/KTDF/KTDFOps.hpp"

namespace ktdf {
namespace common {

/// The set of scf.for loops enclosing a pipeline, up to the scope boundary.
/// Elements are ordered innermost-to-outermost (index 0 is the loop directly
/// containing the pipeline, last element is the outermost loop still inside
/// the scope).
struct PipelineEnclosingScope {
  llvm::SmallVector<mlir::scf::ForOp> loops;
};

/// Walk outward from `pipeline` through enclosing ops. An scf.for qualifies
/// as part of the scope iff its body contains nothing except the chain that
/// leads back to the pipeline (i.e., the pipeline itself, or another
/// qualifying scf.for, plus that loop's terminator scf.yield).
///
/// The walk stops at (returns without extending) any of:
///   - a non-scf.for parent (e.g. scf.if, func.func, ktdf.stage)
///   - a parent whose body contains siblings beside the chain
///
/// Returns an empty scope if the pipeline has no enclosing scf.for that
/// qualifies.
PipelineEnclosingScope computePipelineEnclosingScope(
    mlir::ktdf::PipelineOp pipeline);

}  // namespace common
}  // namespace ktdf

#endif  // KTIR_SCHEDULER_TRANSFORM_COMMON_PIPELINESCOPE_HPP
```

- [ ] **Step 3: Update the new .cpp — namespace + include path**

Edit `src/Transform/Common/PipelineScope.cpp`. Change exactly:

1. `#include "src/Transform/ReusePromotion/Common/PipelineScope.hpp"` → `#include "src/Transform/Common/PipelineScope.hpp"`.
2. `namespace reuse {` → `namespace common {` and matching closing comment.

Final .cpp (full file):

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

#include "src/Transform/Common/PipelineScope.hpp"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"

namespace ktdf {
namespace common {

namespace {

/// True iff the body of `loop` contains nothing except `child` (and the
/// loop's required scf.yield terminator).
bool loopBodyContainsOnly(mlir::scf::ForOp loop, mlir::Operation* child) {
  mlir::Block& body = loop.getRegion().front();
  int non_terminator_count = 0;
  mlir::Operation* only_non_terminator = nullptr;
  for (mlir::Operation& op : body) {
    if (op.hasTrait<mlir::OpTrait::IsTerminator>()) {
      continue;
    }
    ++non_terminator_count;
    only_non_terminator = &op;
  }
  return non_terminator_count == 1 && only_non_terminator == child;
}

}  // namespace

PipelineEnclosingScope computePipelineEnclosingScope(
    mlir::ktdf::PipelineOp pipeline) {
  PipelineEnclosingScope scope;
  mlir::Operation* current = pipeline.getOperation();
  mlir::Operation* parent = current->getParentOp();
  while (parent) {
    auto for_op = mlir::dyn_cast<mlir::scf::ForOp>(parent);
    if (!for_op) {
      break;
    }
    if (!loopBodyContainsOnly(for_op, current)) {
      break;
    }
    scope.loops.push_back(for_op);
    current = for_op.getOperation();
    parent = current->getParentOp();
  }
  return scope;
}

}  // namespace common
}  // namespace ktdf
```

- [ ] **Step 4: Update `src/Transform/CMakeLists.txt`**

Find the `add_library(SchedulerTransforms ...)` source list and:
- Replace `ReusePromotion/Common/PipelineScope.cpp` with `Common/PipelineScope.cpp`.

Resulting source list (full block — only the one line changed):

```cmake
add_library(SchedulerTransforms
    BroadcastPromotion.cpp
    Common/PipelineScope.cpp
    ComputeGroupExtraction.cpp
    ConstructThreeStagePipeline.cpp
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

- [ ] **Step 5: Update callers — `BroadcastPromotion.cpp`**

Find `#include "src/Transform/ReusePromotion/Common/PipelineScope.hpp"` (around line 32 — but locate via grep, not by line number, since other reorderings may have shifted it). Replace with:

```cpp
#include "src/Transform/Common/PipelineScope.hpp"
```

Then in the same file, replace every `ktdf::reuse::computePipelineEnclosingScope` with `ktdf::common::computePipelineEnclosingScope`, and every `ktdf::reuse::PipelineEnclosingScope` with `ktdf::common::PipelineEnclosingScope`.

A quick sed alternative, run from the repo root:

```bash
grep -l "reuse::computePipelineEnclosingScope\|reuse::PipelineEnclosingScope" src/Transform/BroadcastPromotion.cpp
```

If that returns the file, the next command rewrites it in place:

```bash
sed -i.bak \
  -e 's|src/Transform/ReusePromotion/Common/PipelineScope.hpp|src/Transform/Common/PipelineScope.hpp|g' \
  -e 's|ktdf::reuse::computePipelineEnclosingScope|ktdf::common::computePipelineEnclosingScope|g' \
  -e 's|ktdf::reuse::PipelineEnclosingScope|ktdf::common::PipelineEnclosingScope|g' \
  src/Transform/BroadcastPromotion.cpp
rm src/Transform/BroadcastPromotion.cpp.bak
```

- [ ] **Step 6: Update callers — `CandidateSearch.cpp` and `CandidateSearch.hpp`**

Both files are in `src/Transform/ReusePromotion/Common/`. The grep finds which one(s) need updating; `.hpp` may or may not — only edit if grep matches:

```bash
grep -l "PipelineScope.hpp\|reuse::PipelineEnclosingScope\|PipelineEnclosingScope" \
  src/Transform/ReusePromotion/Common/CandidateSearch.cpp \
  src/Transform/ReusePromotion/Common/CandidateSearch.hpp
```

Then rewrite:

```bash
sed -i.bak \
  -e 's|src/Transform/ReusePromotion/Common/PipelineScope.hpp|src/Transform/Common/PipelineScope.hpp|g' \
  -e 's|reuse::PipelineEnclosingScope|common::PipelineEnclosingScope|g' \
  -e 's|reuse::computePipelineEnclosingScope|common::computePipelineEnclosingScope|g' \
  src/Transform/ReusePromotion/Common/CandidateSearch.cpp \
  src/Transform/ReusePromotion/Common/CandidateSearch.hpp
rm src/Transform/ReusePromotion/Common/CandidateSearch.cpp.bak \
   src/Transform/ReusePromotion/Common/CandidateSearch.hpp.bak
```

Caveat: if `CandidateSearch.{cpp,hpp}` reference the namespace as `reuse::` *qualified to the local `ktdf::reuse` they live in* (i.e., bare `PipelineEnclosingScope` because they themselves are inside `namespace reuse`), the sed may miss them. After running sed, manually inspect both files for any remaining bare `PipelineEnclosingScope` or `computePipelineEnclosingScope` references. If they appear inside a `namespace reuse { ... }` block in a way that previously resolved to `ktdf::reuse::PipelineEnclosingScope`, those must now be qualified explicitly as `ktdf::common::PipelineEnclosingScope`. Add the `ktdf::common::` qualifier anywhere needed.

- [ ] **Step 7: Delete the old files**

```bash
rm src/Transform/ReusePromotion/Common/PipelineScope.hpp \
   src/Transform/ReusePromotion/Common/PipelineScope.cpp
```

- [ ] **Step 8: Build**

```bash
ninja -C build
```

Expected: clean build. If a compile error mentions `ktdf::reuse::PipelineEnclosingScope` or `reuse::computePipelineEnclosingScope` not being found, search for the missing reference and replace it with `ktdf::common::...`.

- [ ] **Step 9: Run the full lit suite**

```bash
ninja -C build check-scheduler
```

Expected: 38/38 (or whatever the current baseline is — the move must not change test count or pass/fail status).

- [ ] **Step 10: Commit**

```bash
git add src/Transform/Common/PipelineScope.hpp \
        src/Transform/Common/PipelineScope.cpp \
        src/Transform/CMakeLists.txt \
        src/Transform/BroadcastPromotion.cpp \
        src/Transform/ReusePromotion/Common/CandidateSearch.hpp \
        src/Transform/ReusePromotion/Common/CandidateSearch.cpp
git rm src/Transform/ReusePromotion/Common/PipelineScope.hpp \
       src/Transform/ReusePromotion/Common/PipelineScope.cpp
git commit -m "$(cat <<'EOF'
Move PipelineScope to src/Transform/Common/, rename namespace

Promotes the pipeline-enclosing-scope walk from a reuse-pass-specific
utility to a transform-wide one in preparation for the corelet-parallelism
pass which also needs it. Behavior is unchanged; namespace renamed
ktdf::reuse → ktdf::common. Existing callers (BroadcastPromotion,
CandidateSearch) updated.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Add `ApplicableUnits` utility

**Goal:** Create `ktdf::common::computePipelineApplicableUnits` — the union of `applicable_units` over a pipeline's immediate child stages, with a fatal error on stages missing the attribute.

**Files:**
- Create: `src/Transform/Common/ApplicableUnits.hpp`
- Create: `src/Transform/Common/ApplicableUnits.cpp`
- Modify: `src/Transform/CMakeLists.txt`

- [ ] **Step 1: Create the header**

Create `src/Transform/Common/ApplicableUnits.hpp`:

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
// Compute the union of applicable_units over a pipeline's immediate stages.
//===----------------------------------------------------------------------===//

#ifndef KTIR_SCHEDULER_TRANSFORM_COMMON_APPLICABLEUNITS_HPP
#define KTIR_SCHEDULER_TRANSFORM_COMMON_APPLICABLEUNITS_HPP

#include "llvm/ADT/SetVector.h"
#include "src/Dialects/KTDF/KTDFOps.hpp"

namespace ktdf {
namespace common {

/// Union the `applicable_units` of every immediate `ktdf.stage` child of
/// `pipeline`. Does not descend into nested pipelines or stage bodies.
/// Skips non-stage children (e.g., `ktdf.private`).
///
/// Asserts that every immediate stage has its `applicable_units` attribute
/// set; absence is treated as an upstream-pass invariant violation and
/// triggers `llvm::report_fatal_error`.
///
/// Returns an empty SetVector when the pipeline has no immediate stages
/// (e.g., a pipeline containing only `ktdf.private`).
llvm::SmallSetVector<mlir::ktdf::ComponentType, 4>
computePipelineApplicableUnits(mlir::ktdf::PipelineOp pipeline);

}  // namespace common
}  // namespace ktdf

#endif  // KTIR_SCHEDULER_TRANSFORM_COMMON_APPLICABLEUNITS_HPP
```

- [ ] **Step 2: Create the .cpp**

Create `src/Transform/Common/ApplicableUnits.cpp`:

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

#include "src/Transform/Common/ApplicableUnits.hpp"

#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

namespace ktdf {
namespace common {

llvm::SmallSetVector<mlir::ktdf::ComponentType, 4>
computePipelineApplicableUnits(mlir::ktdf::PipelineOp pipeline) {
  llvm::SmallSetVector<mlir::ktdf::ComponentType, 4> result;

  for (mlir::ktdf::StageOp stage : pipeline.getStages()) {
    auto attr = stage.getApplicableUnitsAttr();
    if (!attr) {
      std::string msg;
      llvm::raw_string_ostream os(msg);
      os << "ktdf.stage at " << stage.getLoc()
         << " is missing required `applicable_units` attribute "
            "(needed by ktdf::common::computePipelineApplicableUnits)";
      llvm::report_fatal_error(llvm::Twine(msg));
    }
    for (mlir::ktdf::ComponentType unit : attr.getValue()) {
      result.insert(unit);
    }
  }

  return result;
}

}  // namespace common
}  // namespace ktdf
```

Note on the API used here:
- `pipeline.getStages()` is an iterator range exposed by `PipelineOp::extraClassDeclaration` (see `src/Dialects/KTDF/KTDFOps.td:390`).
- `stage.getApplicableUnitsAttr()` is the auto-generated TableGen accessor returning `mlir::ktdf::ApplicableUnitsCustomAttr` (or null when absent). The attribute's `.getValue()` returns `ArrayRef<mlir::ktdf::ComponentType>` per `KTDFAttributes.td:121`.

If the build complains that `getStages()` does not exist as expected, look at `KTDFOps.td:390` for the actual signature; it returns `iterator_range<Block::op_iterator<StageOp>>`, which is iterable in a range-for as shown.

- [ ] **Step 3: Add to `src/Transform/CMakeLists.txt`**

In the source list, add `Common/ApplicableUnits.cpp` next to `Common/PipelineScope.cpp` (kept alphabetical):

```cmake
add_library(SchedulerTransforms
    BroadcastPromotion.cpp
    Common/ApplicableUnits.cpp
    Common/PipelineScope.cpp
    ComputeGroupExtraction.cpp
    ...
```

(Only the one new line is added; rest of the source list unchanged.)

- [ ] **Step 4: Build**

```bash
ninja -C build
```

Expected: clean build. If `getStages()` or `getApplicableUnitsAttr()` is misnamed, fix per the ODS in `KTDFOps.td` and rebuild.

- [ ] **Step 5: Run the full lit suite**

```bash
ninja -C build check-scheduler
```

Expected: still passing baseline (no new tests yet).

- [ ] **Step 6: Commit**

```bash
git add src/Transform/Common/ApplicableUnits.hpp \
        src/Transform/Common/ApplicableUnits.cpp \
        src/Transform/CMakeLists.txt
git commit -m "$(cat <<'EOF'
Add ApplicableUnits utility under src/Transform/Common/

Provides ktdf::common::computePipelineApplicableUnits, which unions the
applicable_units attribute across a pipeline's immediate child stages.
Used by the upcoming corelet-parallelism pass to decide whether a
pipeline's work is corelet-local. Asserts (report_fatal_error) when a
stage is missing the attribute, treating it as an upstream-pass invariant
violation.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Extend `SchedulerExtContext` with `getNumCorelets` / `isPerCoreletUnit`

**Goal:** Add two new virtual methods to the context, with Spyre defaults.

**Files:**
- Modify: `src/Utils/SchedulerExtContext.h`
- Modify: `src/Utils/SchedulerExtContext.cpp`

- [ ] **Step 1: Update the header**

Edit `src/Utils/SchedulerExtContext.h`. Inside the `struct SchedulerExtContext` body (after `getFifoTypeStore()` on line 58), add the two new declarations. Also add `mlir::ktdf::ComponentType` forward usage; the struct already includes `KTDFOps.hpp`, so the type is available.

The relevant portion of the struct after the edit:

```cpp
struct SchedulerExtContext {
  SchedulerExtContext() {}

  static const SchedulerExtContext& dummyContext();

  virtual bool isDummy() const { return false; }

  // ... existing methods unchanged: getComputeUnit, getVectorLength,
  // getMaxFifoSlotSize, getInterfaceMemToComputeUnit, getFifoTypeLoad,
  // getFifoTypeStore ...

  /// Number of hardware instances iterations should be distributed across
  /// when emitting ktdf.parallel. Spyre = 2 (per corelet).
  virtual unsigned getNumCorelets() const;

  /// True iff `unit` is a per-corelet unit (LXLU, LXSU, SFP). False for
  /// core-shared units (L3LU, L3SU). Used to decide whether a pipeline's
  /// work is corelet-local.
  virtual bool isPerCoreletUnit(mlir::ktdf::ComponentType unit) const;
};
```

The placement follows the existing methods, separated by a blank line for readability. Both methods are `virtual` so the dummy (and any future real context) can override.

- [ ] **Step 2: Implement in the .cpp**

At the bottom of `src/Utils/SchedulerExtContext.cpp` (before the closing `}  // namespace scheduler`), add the two implementations:

```cpp
unsigned SchedulerExtContext::getNumCorelets() const {
  return 2;
}

bool SchedulerExtContext::isPerCoreletUnit(
    mlir::ktdf::ComponentType unit) const {
  switch (unit) {
    case mlir::ktdf::ComponentType::SFP:
    case mlir::ktdf::ComponentType::LXLU:
    case mlir::ktdf::ComponentType::LXSU:
      return true;
    case mlir::ktdf::ComponentType::L3LU:
    case mlir::ktdf::ComponentType::L3SU:
      return false;
  }
  llvm::report_fatal_error("Unknown ComponentType in isPerCoreletUnit");
}
```

The switch covers every case in the `ComponentType` enum (`SFP, LXLU, LXSU, L3LU, L3SU` per `KTDFEnums.td:67-71`). The fallthrough `report_fatal_error` is unreachable today but defends against future enum additions (e.g., when `PE`/`PT` are added) — without an explicit handler, those would silently default to `false`.

If `report_fatal_error` is not yet included, add at the top of the .cpp:

```cpp
#include "llvm/Support/ErrorHandling.h"
```

(Check first — the existing code already uses `llvm::report_fatal_error`, so the include is likely present.)

- [ ] **Step 3: Build**

```bash
ninja -C build
```

Expected: clean build.

- [ ] **Step 4: Run the full lit suite**

```bash
ninja -C build check-scheduler
```

Expected: still passing baseline.

- [ ] **Step 5: Commit**

```bash
git add src/Utils/SchedulerExtContext.h src/Utils/SchedulerExtContext.cpp
git commit -m "$(cat <<'EOF'
Add getNumCorelets / isPerCoreletUnit to SchedulerExtContext

Two new virtual methods used by the upcoming corelet-parallelism pass to
decide num_instances for ktdf.parallel and to filter pipelines whose work
is not corelet-local. Defaults are Spyre values (num_corelets=2; SFP/LXLU/
LXSU are per-corelet; L3LU/L3SU are core-shared) and are inherited by
DummySchedulerExtContext, matching the dummy's existing convention of
returning Spyre-flavored answers.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Register `ParallelizeLoopsAcrossInstancesPass` in TableGen + headers

**Goal:** Pass declaration and registration scaffolding, no implementation yet.

**Files:**
- Modify: `src/Transform/Passes.td`
- Modify: `src/Transform/Passes.h`

- [ ] **Step 1: Add to `Passes.td`**

Append after `BroadcastPromotionPass` (around line 127), before the `#endif`:

```tablegen
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
```

- [ ] **Step 2: Add factory declarations to `Passes.h`**

Insert next to `createBroadcastPromotionPass` (around line 27):

```cpp
std::unique_ptr<mlir::Pass> createParallelizeLoopsAcrossInstancesPass();
std::unique_ptr<mlir::Pass> createParallelizeLoopsAcrossInstancesPass(
    const scheduler::SchedulerExtContext& scheduler_ctx);
```

(The no-arg form returns a pass constructed with `dummyContext()`. Same pattern as `createConstructThreeStagePipelinePass`.)

- [ ] **Step 3: Build**

```bash
ninja -C build
```

Expected: TableGen step succeeds. C++ compile may fail because `createParallelizeLoopsAcrossInstancesPass` is declared but not defined; that is the expected state at the end of this task — the implementation lands in Task 5. If the build succeeds (because no caller exists yet), great; if it fails on a missing-symbol link error, that's expected and Task 5 fixes it.

If the linker has nothing to link (the no-op-pass-without-driver case), the build may pass. Either way, do not commit until Task 5 is done.

- [ ] **Step 4: Do not commit yet**

Commits land at the end of Task 5 (the implementation lands in the same commit as the declaration).

---

## Task 5: Implement `ParallelizeLoopsAcrossInstances.cpp`

**Goal:** The full pass implementation, including analysis helpers, the rewriter, and debug logging.

**Files:**
- Create: `src/Transform/ParallelizeLoopsAcrossInstances.cpp`
- Modify: `src/Transform/CMakeLists.txt`

- [ ] **Step 1: Create the source file**

Create `src/Transform/ParallelizeLoopsAcrossInstances.cpp`:

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
// Pass: -parallelize-loops-across-instances
//
// Rewrites parallel scf.for loops enclosing per-corelet ktdf.pipelines into
// ktdf.parallel ops that distribute iterations across hardware instances.
// Target abstraction is via SchedulerExtContext (getNumCorelets,
// isPerCoreletUnit); the pass code itself names no hardware specifics.
//===----------------------------------------------------------------------===//

#include <optional>

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "src/Dialects/KTDF/KTDFOps.hpp"
#include "src/Transform/Common/ApplicableUnits.hpp"
#include "src/Transform/Common/PipelineScope.hpp"
#include "src/Transform/Passes.h"
#include "src/Utils/SchedulerExtContext.h"

#define PASS_NAME "parallelize-loops-across-instances"
#define DEBUG_TYPE PASS_NAME

namespace ktdf {
#define GEN_PASS_DEF_PARALLELIZELOOPSACROSSINSTANCESPASS
#include "src/Transform/Passes.h.inc"
}  // namespace ktdf

using namespace mlir;

namespace {

//===----------------------------------------------------------------------===//
// Candidate
//===----------------------------------------------------------------------===//

struct Candidate {
  mlir::ktdf::PipelineOp pipeline;
  mlir::scf::ForOp loop;
  int64_t trip_count;
  bool balanced;
};

//===----------------------------------------------------------------------===//
// Analysis helpers
//===----------------------------------------------------------------------===//

/// Returns the static trip count of `loop` iff lb, ub, step are all
/// arith.constant index ops. step must be positive.
std::optional<int64_t> getStaticTripCount(scf::ForOp loop) {
  auto lb_const = loop.getLowerBound().getDefiningOp<arith::ConstantIndexOp>();
  auto ub_const = loop.getUpperBound().getDefiningOp<arith::ConstantIndexOp>();
  auto step_const = loop.getStep().getDefiningOp<arith::ConstantIndexOp>();
  if (!lb_const || !ub_const || !step_const) {
    return std::nullopt;
  }
  int64_t lb = lb_const.value();
  int64_t ub = ub_const.value();
  int64_t step = step_const.value();
  if (step <= 0) {
    return std::nullopt;
  }
  if (ub <= lb) {
    return 0;
  }
  return (ub - lb + step - 1) / step;
}

/// True iff `loop` carries `loop_type = #ktdf.loop_type<parallel_loop>`.
bool isParallelLoop(scf::ForOp loop) {
  auto attr = loop->getAttrOfType<mlir::ktdf::LoopTypeAttr>("loop_type");
  if (!attr) {
    return false;
  }
  return attr.getValue() == mlir::ktdf::LoopType::ParallelLoop;
}

/// True iff every unit in the pipeline's effective applicable_units is
/// per-corelet per `ctx`. Returns false if the effective set is empty.
/// Triggers report_fatal_error if any immediate stage lacks the attribute.
bool allUnitsAreCoreletLocal(mlir::ktdf::PipelineOp pipeline,
                             const scheduler::SchedulerExtContext& ctx) {
  auto units = ktdf::common::computePipelineApplicableUnits(pipeline);
  if (units.empty()) {
    return false;
  }
  for (mlir::ktdf::ComponentType unit : units) {
    if (!ctx.isPerCoreletUnit(unit)) {
      return false;
    }
  }
  return true;
}

/// Walk scope_loops outermost-to-innermost; return the first loop matching
/// the predicate `pred(loop, trip_count)`.
template <typename Pred>
std::optional<std::pair<scf::ForOp, int64_t>> firstMatching(
    llvm::ArrayRef<scf::ForOp> scope_loops_innermost_first, Pred pred) {
  for (scf::ForOp loop : llvm::reverse(scope_loops_innermost_first)) {
    if (!isParallelLoop(loop)) {
      continue;
    }
    auto trip = getStaticTripCount(loop);
    if (!trip.has_value()) {
      continue;
    }
    if (pred(loop, *trip)) {
      return std::make_pair(loop, *trip);
    }
  }
  return std::nullopt;
}

/// Two-pass loop selection: balanced first, imbalanced fallback.
std::optional<std::pair<scf::ForOp, int64_t>> selectLoop(
    llvm::ArrayRef<scf::ForOp> scope_loops_innermost_first,
    unsigned num_instances) {
  // Pass 1: trip >= N and trip % N == 0.
  auto balanced = firstMatching(
      scope_loops_innermost_first,
      [num_instances](scf::ForOp, int64_t trip) {
        return trip >= static_cast<int64_t>(num_instances) &&
               trip % static_cast<int64_t>(num_instances) == 0;
      });
  if (balanced.has_value()) {
    return balanced;
  }
  // Pass 2: trip >= N (drop divisibility).
  return firstMatching(
      scope_loops_innermost_first,
      [num_instances](scf::ForOp, int64_t trip) {
        return trip >= static_cast<int64_t>(num_instances);
      });
}

std::optional<Candidate> findCandidate(
    mlir::ktdf::PipelineOp pipeline,
    const scheduler::SchedulerExtContext& ctx) {
  // Step 1+2: applicable_units.
  if (!allUnitsAreCoreletLocal(pipeline, ctx)) {
    LLVM_DEBUG(llvm::dbgs() << "[" PASS_NAME "] skip pipeline at "
                            << pipeline.getLoc()
                            << ": empty or non-corelet applicable_units\n");
    return std::nullopt;
  }

  // Step 3: enclosing scope.
  auto scope = ktdf::common::computePipelineEnclosingScope(pipeline);
  if (scope.loops.empty()) {
    LLVM_DEBUG(llvm::dbgs() << "[" PASS_NAME "] skip pipeline at "
                            << pipeline.getLoc()
                            << ": no enclosing scf.for scope\n");
    return std::nullopt;
  }

  // Step 4: loop selection.
  unsigned num_instances = ctx.getNumCorelets();
  auto picked = selectLoop(scope.loops, num_instances);
  if (!picked.has_value()) {
    LLVM_DEBUG(llvm::dbgs() << "[" PASS_NAME "] skip pipeline at "
                            << pipeline.getLoc()
                            << ": no parallel loop in scope satisfies trip >= "
                            << num_instances << "\n");
    return std::nullopt;
  }
  scf::ForOp loop = picked->first;
  int64_t trip = picked->second;
  bool balanced = (trip % static_cast<int64_t>(num_instances)) == 0;

  if (!balanced) {
    int64_t chunk = (trip + num_instances - 1) / num_instances;
    int64_t leftover = trip - (num_instances - 1) * chunk;
    LLVM_DEBUG(llvm::dbgs()
               << "[" PASS_NAME "] imbalanced (fallback) at "
               << loop.getLoc() << ": trip=" << trip
               << ", num_instances=" << num_instances
               << ", chunks=" << chunk << "," << leftover << "\n");
  } else {
    LLVM_DEBUG(llvm::dbgs() << "[" PASS_NAME "] balanced candidate at "
                            << loop.getLoc() << ": trip=" << trip
                            << ", num_instances=" << num_instances << "\n");
  }

  return Candidate{pipeline, loop, trip, balanced};
}

//===----------------------------------------------------------------------===//
// Transformation
//===----------------------------------------------------------------------===//

void rewrite(const Candidate& c, unsigned num_instances) {
  scf::ForOp loop = c.loop;
  Location loc = loop.getLoc();
  Value lb = loop.getLowerBound();
  Value ub = loop.getUpperBound();
  Value step = loop.getStep();
  Value old_iv = loop.getInductionVar();
  Block* old_body = loop.getBody();

  // Build ktdf.parallel at the scf.for's position. The custom builder
  // creates the body block with two index args (IV + instance id).
  OpBuilder builder(loop);
  auto parallel = builder.create<mlir::ktdf::ParallelOp>(
      loc,
      /*lowerBounds=*/ValueRange{lb},
      /*upperBounds=*/ValueRange{ub},
      /*steps=*/ValueRange{step},
      /*numInstances=*/static_cast<int64_t>(num_instances));

  Block* new_body = &parallel.getBody().front();

  // Splice the old body (excluding the trailing scf.yield terminator) into
  // the new ktdf.parallel body.
  new_body->getOperations().splice(
      new_body->begin(), old_body->getOperations(), old_body->begin(),
      std::prev(old_body->end()));

  // RAUW the original IV with the new parallel's first block argument.
  Value new_iv = parallel.getInductionVar(0);
  old_iv.replaceAllUsesWith(new_iv);

  // Append ktdf.parallel_yield as the new body's terminator.
  builder.setInsertionPointToEnd(new_body);
  builder.create<mlir::ktdf::ParallelYieldOp>(loc);

  // Erase the original scf.for. Its body is now empty (only the orphaned
  // scf.yield remains, which is erased with the parent).
  loop.erase();

#ifndef NDEBUG
  if (failed(parallel.verify())) {
    llvm::report_fatal_error("ktdf.parallel verifier failed after rewrite");
  }
#endif
}

//===----------------------------------------------------------------------===//
// Pass entry
//===----------------------------------------------------------------------===//

struct ParallelizeLoopsAcrossInstancesPass
    : public ::ktdf::impl::ParallelizeLoopsAcrossInstancesPassBase<
          ParallelizeLoopsAcrossInstancesPass> {
  ParallelizeLoopsAcrossInstancesPass(
      const scheduler::SchedulerExtContext& scheduler_ctx)
      : scheduler_ctx_(scheduler_ctx) {}

  void runOnOperation() override {
    ModuleOp module = getOperation();
    unsigned num_instances = scheduler_ctx_.getNumCorelets();

    LLVM_DEBUG(llvm::dbgs() << "[" PASS_NAME "] num_instances="
                            << num_instances << "\n");

    if (num_instances < 2) {
      LLVM_DEBUG(llvm::dbgs() << "[" PASS_NAME "] num_instances < 2; "
                                 "no-op\n");
      return;
    }

    // Pre-order walk over pipelines. Collect candidates first, then rewrite,
    // so the walk's iterator is not invalidated by op erasure.
    SmallVector<Candidate> candidates;
    module.walk<WalkOrder::PreOrder>([&](mlir::ktdf::PipelineOp pipeline) {
      if (auto c = findCandidate(pipeline, scheduler_ctx_)) {
        candidates.push_back(*c);
      }
    });

    for (const Candidate& c : candidates) {
      rewrite(c, num_instances);
    }
  }

 private:
  const scheduler::SchedulerExtContext& scheduler_ctx_;
};

}  // namespace

namespace ktdf {

std::unique_ptr<mlir::Pass> createParallelizeLoopsAcrossInstancesPass(
    const scheduler::SchedulerExtContext& scheduler_ctx) {
  return std::make_unique<ParallelizeLoopsAcrossInstancesPass>(scheduler_ctx);
}

std::unique_ptr<mlir::Pass> createParallelizeLoopsAcrossInstancesPass() {
  return std::make_unique<ParallelizeLoopsAcrossInstancesPass>(
      scheduler::SchedulerExtContext::dummyContext());
}

}  // namespace ktdf
```

Notes on key API choices:
- `pipeline.walk<WalkOrder::PreOrder>(...)` — pre-order matches the design's "outermost first" intuition (spec §3).
- We **collect** candidates and **then** rewrite, rather than rewriting during the walk. Rewriting during a walk that's iterating through ops would risk invalidating iterators — the inner `scf.for` we replace might be the very loop the walk is currently visiting. Two-pass collect-then-rewrite is safer.
- `parallel.getInductionVar(0)` — the helper from PR #5's `extraClassDeclaration` (`KTDFOps.td:836`).
- `loop.erase()` is called only after all uses of `old_iv` have been RAUW'd; otherwise SSA uses become dangling.

- [ ] **Step 2: Add to `src/Transform/CMakeLists.txt`**

Add `ParallelizeLoopsAcrossInstances.cpp` to the source list, alphabetically between existing entries:

```cmake
add_library(SchedulerTransforms
    BroadcastPromotion.cpp
    Common/ApplicableUnits.cpp
    Common/PipelineScope.cpp
    ComputeGroupExtraction.cpp
    ConstructThreeStagePipeline.cpp
    ParallelizeLoopsAcrossInstances.cpp
    PipelineTree/Legalizer.cpp
    PipelineTree/Materializer.cpp
    ...
```

- [ ] **Step 3: Build**

```bash
ninja -C build
```

Expected: clean build. The TableGen-generated `impl::ParallelizeLoopsAcrossInstancesPassBase` exists because Task 4 added the def. Compile errors most likely come from: a wrong MLIR API call (missing include, wrong helper name), a typo in `getStages()` / `getApplicableUnitsAttr()`, or a mismatch with the `ktdf::common::` namespace in the headers. Fix per the actual error message.

- [ ] **Step 4: Run the full lit suite**

```bash
ninja -C build check-scheduler
```

Expected: still passing. The pass exists but has no tests yet, and it isn't invoked by any RUN line, so existing tests shouldn't be affected.

- [ ] **Step 5: Commit**

```bash
git add src/Transform/Passes.td \
        src/Transform/Passes.h \
        src/Transform/ParallelizeLoopsAcrossInstances.cpp \
        src/Transform/CMakeLists.txt
git commit -m "$(cat <<'EOF'
Add -parallelize-loops-across-instances pass

Walks ktdf.pipelines pre-order. For each pipeline whose immediate stages'
applicable_units are entirely per-corelet (per
SchedulerExtContext::isPerCoreletUnit), finds the outermost parallel
scf.for in the pipeline's enclosing scope whose statically-known trip
count permits a balanced split across num_instances (preferred) or any
trip >= num_instances (fallback), and rewrites that scf.for into
ktdf.parallel.

Target abstraction: getNumCorelets and isPerCoreletUnit on the context.
Reuses ktdf::common::{computePipelineEnclosingScope,
computePipelineApplicableUnits}.

Implements the design at
docs/superpowers/specs/2026-05-15-parallelize-loops-across-instances-design.md.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Wire into `buildKTDPToDFIRPipeline`

**Goal:** Add the pass to the scheduler driver pipeline so end-to-end runs include it.

**Files:**
- Modify: `tools/scheduler/scheduler-pipeline.cpp`

- [ ] **Step 1: Insert the pass call**

Edit `tools/scheduler/scheduler-pipeline.cpp`. Inside `buildKTDPToDFIRPipeline`, append after the existing `createConstructThreeStagePipelinePass` call (and after stage-coarsening if it appears here later — today the pipeline only has compute-group-extraction + symbolDCE + three-stage-pipeline):

```cpp
void buildKTDPToDFIRPipeline(
    mlir::OpPassManager& pm,
    const scheduler::SchedulerExtContext& scheduler_ctx) {
  pm.addPass(createComputeGroupExtractionPass());
  pm.addPass(mlir::createSymbolDCEPass());
  pm.addPass(createConstructThreeStagePipelinePass(scheduler_ctx));
  pm.addPass(createParallelizeLoopsAcrossInstancesPass(scheduler_ctx));
  // todo: pm.addPass(createTileSCFForLoopsPass());
}
```

(The existing `// todo:` comment stays unchanged.)

The placement is "after pipeline construction, before further lowering." If a future change inserts a stage-coarsening pass between the three-stage-pipeline pass and DFIR lowering, the corelet-parallelism pass should follow stage-coarsening (it expects the post-coarsening hierarchical shape). Today there's no stage-coarsening pass in this pipeline, so adding the corelet pass at the end is correct.

- [ ] **Step 2: Build**

```bash
ninja -C build
```

Expected: clean build.

- [ ] **Step 3: Run the full lit suite**

```bash
ninja -C build check-scheduler
```

Expected: passing. If end-to-end pipeline tests under `test/Scheduler_pipeline/` now produce different output (because the new pass kicks in on previously-untransformed loops), inspect the changes. **Likely scenarios:**
- Tests use input where the target loops aren't parallel-annotated → pass is a no-op → no change. *(Most likely.)*
- Tests do exercise parallel loops → pass transforms them → CHECK lines fail. If this happens, the output is the new desired output; update the CHECK lines accordingly.
- Tests have stages without `applicable_units` → fatal error. If this happens, that's a real upstream-pass bug surfaced by the new pass; STOP and report BLOCKED with the failing test name and the location.

- [ ] **Step 4: Commit**

```bash
git add tools/scheduler/scheduler-pipeline.cpp
git commit -m "$(cat <<'EOF'
Wire -parallelize-loops-across-instances into kEmitDFIR pipeline

Runs after construct-three-stage-pipeline so that ktdf.pipelines and
their enclosing parallel loops are visible. Threads scheduler_ctx
through.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: Lit tests — positive cases

**Goal:** Cover the rewrite paths with input → output FileCheck tests.

**Files:**
- Create: `test/Transforms/ParallelizeLoopsAcrossInstances/basic-1d-balanced.mlir`
- Create: `test/Transforms/ParallelizeLoopsAcrossInstances/basic-1d-imbalanced-fallback.mlir`
- Create: `test/Transforms/ParallelizeLoopsAcrossInstances/outermost-balanced-wins.mlir`
- Create: `test/Transforms/ParallelizeLoopsAcrossInstances/inner-divisible-when-outer-not.mlir`

(Skipping the larger `tmp-mlir-shape.mlir` reproduction — it adds little coverage beyond the smaller cases. The spec lists it as nice-to-have. Add later if regressions arise.)

The test pattern: input has a `ktdf.pipeline` with corelet-local stage `applicable_units` and a parallel-annotated `scf.for` in scope; output has `ktdf.parallel` replacing that loop. CHECK lines pin the surface form.

- [ ] **Step 1: Create `basic-1d-balanced.mlir`**

```mlir
// RUN: scheduler -allow-unregistered-dialect %s -parallelize-loops-across-instances | FileCheck %s

// One pipeline, corelet-local applicable_units, scope with one parallel
// loop, trip count 4 (balanced split for num_instances=2).

// CHECK-LABEL: func.func @basic_1d_balanced
// CHECK:         ktdf.parallel (%[[IV:[a-zA-Z0-9_]+]], %[[INST:[a-zA-Z0-9_]+]]) = (%{{.*}}) to (%{{.*}}) step (%{{.*}}) distribute(num_instances = 2) {
// CHECK:           "test.body"(%[[IV]]) : (index) -> ()
// CHECK:           ktdf.parallel_yield
// CHECK:         }

func.func @basic_1d_balanced() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  scf.for %i = %c0 to %c4 step %c1 {
    ktdf.pipeline {
      ktdf.stage depends_in(none) depends_out(none) {
        "test.body"(%i) : (index) -> ()
      } {applicable_units = #ktdf.applicable_units<[SFP]>}
    }
  } {loop_type = #ktdf.loop_type<parallel_loop>}
  return
}
```

- [ ] **Step 2: Create `basic-1d-imbalanced-fallback.mlir`**

```mlir
// RUN: scheduler -allow-unregistered-dialect %s -parallelize-loops-across-instances | FileCheck %s

// Trip count 5 (not divisible by 2). Pass 1 finds nothing; Pass 2 accepts.

// CHECK-LABEL: func.func @imbalanced_fallback
// CHECK:         ktdf.parallel (%[[IV:[a-zA-Z0-9_]+]], %[[INST:[a-zA-Z0-9_]+]]) = (%{{.*}}) to (%{{.*}}) step (%{{.*}}) distribute(num_instances = 2) {
// CHECK:           "test.body"(%[[IV]]) : (index) -> ()
// CHECK:           ktdf.parallel_yield
// CHECK:         }

func.func @imbalanced_fallback() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c5 = arith.constant 5 : index
  scf.for %i = %c0 to %c5 step %c1 {
    ktdf.pipeline {
      ktdf.stage depends_in(none) depends_out(none) {
        "test.body"(%i) : (index) -> ()
      } {applicable_units = #ktdf.applicable_units<[SFP]>}
    }
  } {loop_type = #ktdf.loop_type<parallel_loop>}
  return
}
```

- [ ] **Step 3: Create `outermost-balanced-wins.mlir`**

```mlir
// RUN: scheduler -allow-unregistered-dialect %s -parallelize-loops-across-instances | FileCheck %s

// Two parallel loops in scope, both divisible by 2. Outermost (trip 4) wins;
// inner (trip 8) remains an scf.for inside the parallel body.

// CHECK-LABEL: func.func @outermost_wins
// CHECK:         ktdf.parallel (%[[OUT:[a-zA-Z0-9_]+]], %[[INST:[a-zA-Z0-9_]+]]) = (%{{.*}}) to (%{{.*}}) step (%{{.*}}) distribute(num_instances = 2) {
// CHECK:           scf.for %[[IN:[a-zA-Z0-9_]+]] = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK:             "test.body"(%[[OUT]], %[[IN]]) : (index, index) -> ()
// CHECK:           } {loop_type = #ktdf.loop_type<parallel_loop>}
// CHECK:           ktdf.parallel_yield
// CHECK:         }

func.func @outermost_wins() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  scf.for %i = %c0 to %c4 step %c1 {
    scf.for %j = %c0 to %c8 step %c1 {
      ktdf.pipeline {
        ktdf.stage depends_in(none) depends_out(none) {
          "test.body"(%i, %j) : (index, index) -> ()
        } {applicable_units = #ktdf.applicable_units<[SFP]>}
      }
    } {loop_type = #ktdf.loop_type<parallel_loop>}
  } {loop_type = #ktdf.loop_type<parallel_loop>}
  return
}
```

- [ ] **Step 4: Create `inner-divisible-when-outer-not.mlir`**

```mlir
// RUN: scheduler -allow-unregistered-dialect %s -parallelize-loops-across-instances | FileCheck %s

// Outer trip 5 (not divisible), inner trip 8 (divisible). Pass 1 walks
// outermost-first, skips outer (rule 4 fails), accepts inner. The outer
// scf.for survives; the inner scf.for becomes ktdf.parallel.

// CHECK-LABEL: func.func @inner_divisible
// CHECK:         scf.for %[[OUT:[a-zA-Z0-9_]+]] = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK:           ktdf.parallel (%[[IN:[a-zA-Z0-9_]+]], %[[INST:[a-zA-Z0-9_]+]]) = (%{{.*}}) to (%{{.*}}) step (%{{.*}}) distribute(num_instances = 2) {
// CHECK:             "test.body"(%[[OUT]], %[[IN]]) : (index, index) -> ()
// CHECK:             ktdf.parallel_yield
// CHECK:           }
// CHECK:         } {loop_type = #ktdf.loop_type<parallel_loop>}

func.func @inner_divisible() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c5 = arith.constant 5 : index
  %c8 = arith.constant 8 : index
  scf.for %i = %c0 to %c5 step %c1 {
    scf.for %j = %c0 to %c8 step %c1 {
      ktdf.pipeline {
        ktdf.stage depends_in(none) depends_out(none) {
          "test.body"(%i, %j) : (index, index) -> ()
        } {applicable_units = #ktdf.applicable_units<[SFP]>}
      }
    } {loop_type = #ktdf.loop_type<parallel_loop>}
  } {loop_type = #ktdf.loop_type<parallel_loop>}
  return
}
```

- [ ] **Step 5: Run the lit suite**

```bash
ninja -C build check-scheduler
```

Expected: 4 new tests passing. If a CHECK fails, run the test directly to see actual printer output, then adjust the CHECK pattern to match. Do NOT modify the pass to make tests pass — adjust CHECK lines.

- [ ] **Step 6: Commit**

```bash
git add test/Transforms/ParallelizeLoopsAcrossInstances/
git commit -m "$(cat <<'EOF'
Add positive lit tests for -parallelize-loops-across-instances

Covers the four key transformation paths: balanced 1-D split, imbalanced
1-D fallback, outermost-balanced wins over inner, and inner-divisible
when the outer trip is not divisible.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: Lit tests — negative cases

**Goal:** Verify the IR is unchanged when the pipeline is not a candidate.

**Files:**
- Create: `test/Transforms/ParallelizeLoopsAcrossInstances/non-corelet-unit.mlir`
- Create: `test/Transforms/ParallelizeLoopsAcrossInstances/no-enclosing-scope.mlir`
- Create: `test/Transforms/ParallelizeLoopsAcrossInstances/no-parallel-loop.mlir`
- Create: `test/Transforms/ParallelizeLoopsAcrossInstances/dynamic-trip-count.mlir`
- Create: `test/Transforms/ParallelizeLoopsAcrossInstances/trip-count-too-small.mlir`
- Create: `test/Transforms/ParallelizeLoopsAcrossInstances/pipeline-no-stages.mlir`

For each, the CHECK lines assert the input shape survives — specifically the `scf.for` (no `ktdf.parallel`).

- [ ] **Step 1: Create `non-corelet-unit.mlir`**

```mlir
// RUN: scheduler -allow-unregistered-dialect %s -parallelize-loops-across-instances | FileCheck %s

// applicable_units includes L3LU (core-shared). Pipeline is rejected; loop
// stays as scf.for.

// CHECK-LABEL: func.func @non_corelet
// CHECK:         scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK:           ktdf.pipeline {
// CHECK-NOT:    ktdf.parallel
// CHECK:         } {loop_type = #ktdf.loop_type<parallel_loop>}

func.func @non_corelet() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  scf.for %i = %c0 to %c4 step %c1 {
    ktdf.pipeline {
      ktdf.stage depends_in(none) depends_out(none) {
        "test.body"(%i) : (index) -> ()
      } {applicable_units = #ktdf.applicable_units<[L3LU]>}
    }
  } {loop_type = #ktdf.loop_type<parallel_loop>}
  return
}
```

- [ ] **Step 2: Create `no-enclosing-scope.mlir`**

```mlir
// RUN: scheduler -allow-unregistered-dialect %s -parallelize-loops-across-instances | FileCheck %s

// Pipeline at function top-level, no scf.for parents. No candidate.

// CHECK-LABEL: func.func @no_scope
// CHECK:         ktdf.pipeline {
// CHECK-NOT:    ktdf.parallel
// CHECK:         }

func.func @no_scope() {
  ktdf.pipeline {
    ktdf.stage depends_in(none) depends_out(none) {
      "test.body"() : () -> ()
    } {applicable_units = #ktdf.applicable_units<[SFP]>}
  }
  return
}
```

- [ ] **Step 3: Create `no-parallel-loop.mlir`**

```mlir
// RUN: scheduler -allow-unregistered-dialect %s -parallelize-loops-across-instances | FileCheck %s

// Loop in scope is reduction, not parallel. No candidate.

// CHECK-LABEL: func.func @reduction_loop
// CHECK:         scf.for
// CHECK-NOT:    ktdf.parallel
// CHECK:         {loop_type = #ktdf.loop_type<reduction_loop>}

func.func @reduction_loop() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  scf.for %i = %c0 to %c4 step %c1 {
    ktdf.pipeline {
      ktdf.stage depends_in(none) depends_out(none) {
        "test.body"(%i) : (index) -> ()
      } {applicable_units = #ktdf.applicable_units<[SFP]>}
    }
  } {loop_type = #ktdf.loop_type<reduction_loop>}
  return
}
```

- [ ] **Step 4: Create `dynamic-trip-count.mlir`**

```mlir
// RUN: scheduler -allow-unregistered-dialect %s -parallelize-loops-across-instances | FileCheck %s

// Loop bounds depend on a function argument. Trip count not statically
// known. No candidate.

// CHECK-LABEL: func.func @dynamic_trip
// CHECK:         scf.for
// CHECK-NOT:    ktdf.parallel

func.func @dynamic_trip(%arg0: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %arg0 step %c1 {
    ktdf.pipeline {
      ktdf.stage depends_in(none) depends_out(none) {
        "test.body"(%i) : (index) -> ()
      } {applicable_units = #ktdf.applicable_units<[SFP]>}
    }
  } {loop_type = #ktdf.loop_type<parallel_loop>}
  return
}
```

- [ ] **Step 5: Create `trip-count-too-small.mlir`**

```mlir
// RUN: scheduler -allow-unregistered-dialect %s -parallelize-loops-across-instances | FileCheck %s

// Trip count 1 (less than num_instances=2). No candidate.

// CHECK-LABEL: func.func @trip_too_small
// CHECK:         scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK-NOT:    ktdf.parallel
// CHECK:         } {loop_type = #ktdf.loop_type<parallel_loop>}

func.func @trip_too_small() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %c1 step %c1 {
    ktdf.pipeline {
      ktdf.stage depends_in(none) depends_out(none) {
        "test.body"(%i) : (index) -> ()
      } {applicable_units = #ktdf.applicable_units<[SFP]>}
    }
  } {loop_type = #ktdf.loop_type<parallel_loop>}
  return
}
```

- [ ] **Step 6: Create `pipeline-no-stages.mlir`**

```mlir
// RUN: scheduler -allow-unregistered-dialect %s -parallelize-loops-across-instances | FileCheck %s

// Pipeline contains only a ktdf.private (no stages). Effective
// applicable_units is empty. Pipeline is rejected.

// CHECK-LABEL: func.func @no_stages
// CHECK:         scf.for
// CHECK-NOT:    ktdf.parallel

func.func @no_stages() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  scf.for %i = %c0 to %c4 step %c1 {
    ktdf.pipeline {
      %t = ktdf.private -> (!ktdf.token) {
        %tk = ktdf.create_token : !ktdf.token
        ktdf.private_yield %tk : !ktdf.token
      }
    }
  } {loop_type = #ktdf.loop_type<parallel_loop>}
  return
}
```

- [ ] **Step 7: Run the lit suite**

```bash
ninja -C build check-scheduler
```

Expected: 6 new tests passing. Each verifies that the IR's `scf.for` survives and no `ktdf.parallel` appears.

- [ ] **Step 8: Commit**

```bash
git add test/Transforms/ParallelizeLoopsAcrossInstances/
git commit -m "$(cat <<'EOF'
Add negative lit tests for -parallelize-loops-across-instances

Covers each rejection case from the design's §5 table: non-corelet unit,
no enclosing scope, reduction loop, dynamic trip count, trip too small,
and pipeline with no immediate stages. Each verifies IR shape is
unchanged.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 9: Lit tests — output legality, idempotence, debug

**Goal:** Confirm output round-trips cleanly, the pass is idempotent, and debug logging fires correctly.

**Files:**
- Create: `test/Transforms/ParallelizeLoopsAcrossInstances/output-round-trip.mlir`
- Create: `test/Transforms/ParallelizeLoopsAcrossInstances/idempotent.mlir`
- Create: `test/Transforms/ParallelizeLoopsAcrossInstances/debug-balanced-candidate.mlir`
- Create: `test/Transforms/ParallelizeLoopsAcrossInstances/debug-imbalanced-fallback.mlir`

- [ ] **Step 1: Create `output-round-trip.mlir`**

```mlir
// RUN: scheduler -allow-unregistered-dialect %s -parallelize-loops-across-instances | scheduler -allow-unregistered-dialect | FileCheck %s

// Verify the transformed output parses cleanly through scheduler again.

// CHECK-LABEL: func.func @round_trip
// CHECK:         ktdf.parallel
// CHECK:           ktdf.parallel_yield
// CHECK:         }

func.func @round_trip() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  scf.for %i = %c0 to %c4 step %c1 {
    ktdf.pipeline {
      ktdf.stage depends_in(none) depends_out(none) {
        "test.body"(%i) : (index) -> ()
      } {applicable_units = #ktdf.applicable_units<[SFP]>}
    }
  } {loop_type = #ktdf.loop_type<parallel_loop>}
  return
}
```

- [ ] **Step 2: Create `idempotent.mlir`**

```mlir
// RUN: scheduler -allow-unregistered-dialect %s -parallelize-loops-across-instances -parallelize-loops-across-instances | FileCheck %s

// Two runs back-to-back. Second run finds no scf.for to transform (the
// new ktdf.parallel is not an scf.for), so output is identical to a single
// run. Verify exactly one ktdf.parallel appears.

// CHECK-LABEL: func.func @idempotent
// CHECK:         ktdf.parallel
// CHECK-NOT:    ktdf.parallel
// CHECK:         return

func.func @idempotent() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  scf.for %i = %c0 to %c4 step %c1 {
    ktdf.pipeline {
      ktdf.stage depends_in(none) depends_out(none) {
        "test.body"(%i) : (index) -> ()
      } {applicable_units = #ktdf.applicable_units<[SFP]>}
    }
  } {loop_type = #ktdf.loop_type<parallel_loop>}
  return
}
```

- [ ] **Step 3: Create `debug-balanced-candidate.mlir`**

```mlir
// RUN: scheduler -allow-unregistered-dialect %s -parallelize-loops-across-instances -debug-only=parallelize-loops-across-instances 2>&1 | FileCheck %s --check-prefix=DEBUG

// DEBUG: balanced candidate
// DEBUG-SAME: trip=4
// DEBUG-SAME: num_instances=2

func.func @debug_balanced() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  scf.for %i = %c0 to %c4 step %c1 {
    ktdf.pipeline {
      ktdf.stage depends_in(none) depends_out(none) {
        "test.body"(%i) : (index) -> ()
      } {applicable_units = #ktdf.applicable_units<[SFP]>}
    }
  } {loop_type = #ktdf.loop_type<parallel_loop>}
  return
}
```

- [ ] **Step 4: Create `debug-imbalanced-fallback.mlir`**

```mlir
// RUN: scheduler -allow-unregistered-dialect %s -parallelize-loops-across-instances -debug-only=parallelize-loops-across-instances 2>&1 | FileCheck %s --check-prefix=DEBUG

// DEBUG: imbalanced (fallback)
// DEBUG-SAME: trip=5
// DEBUG-SAME: num_instances=2
// DEBUG-SAME: chunks=3,2

func.func @debug_imbalanced() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c5 = arith.constant 5 : index
  scf.for %i = %c0 to %c5 step %c1 {
    ktdf.pipeline {
      ktdf.stage depends_in(none) depends_out(none) {
        "test.body"(%i) : (index) -> ()
      } {applicable_units = #ktdf.applicable_units<[SFP]>}
    }
  } {loop_type = #ktdf.loop_type<parallel_loop>}
  return
}
```

Note: `-debug-only` requires a debug build. If `scheduler` is built in release mode, the `LLVM_DEBUG` macro expands to nothing and the test will fail. To make these tests robust across build configs, the project may need an `assert`-mode test directive or a config check. **If these two tests fail in your build because of release mode**, document the failure but do not block the rest of the work — these are diagnostic-style tests that follow the convention of `BroadcastPromotion` (see `docs/superpowers/specs/2026-05-12-broadcast-promotion-design.md` §4 "Debug logging").

- [ ] **Step 5: Run the lit suite**

```bash
ninja -C build check-scheduler
```

Expected: 4 new tests. The two debug tests may be skipped or pass depending on build mode; the round-trip and idempotent tests must pass.

- [ ] **Step 6: Commit**

```bash
git add test/Transforms/ParallelizeLoopsAcrossInstances/
git commit -m "$(cat <<'EOF'
Add round-trip, idempotence, and debug-log lit tests

Round-trip verifies the transformed output parses cleanly through
scheduler. Idempotence verifies a second pass run is a no-op. Debug-log
tests verify the LLVM_DEBUG output for both balanced and imbalanced
candidate paths.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 10: Final verification

- [ ] **Step 1: Clean rebuild**

```bash
ninja -C build
```

Expected: clean build (or no work to do).

- [ ] **Step 2: Full lit suite**

```bash
ninja -C build check-scheduler
```

Expected: all tests pass. Count is the prior baseline plus the new tests added across Tasks 7-9 (14 new tests in total: 4 positive + 6 negative + 4 round-trip/idempotent/debug).

- [ ] **Step 3: Confirm log of new commits matches plan**

```bash
git log --oneline ca23924..HEAD
```

Expected: 8 commits in order corresponding to Tasks 1, 2, 3, 5 (Task 4 has no commit), 6, 7, 8, 9.

If a Task is missing or out of order, that is a flag — re-run that Task before declaring done.

- [ ] **Step 4: Stop here**

The pass is implemented, wired into the driver pipeline, and tested. The fatal-error case (a stage missing `applicable_units`) is intentionally not tested at the lit level — `report_fatal_error` is hard to exercise from a lit test since it terminates the process, and the path is a defensive guard rather than a feature. If a regression test for it becomes necessary later, add a `// RUN: not scheduler ...` line per the design doc §8.

---

## Self-review (run by plan author)

**Spec coverage:**
- §1 Scope & Goals → Tasks 4 (pass registration), 5 (impl), 6 (driver wiring) collectively register the pass and its non-target-specific behavior.
- §2 Pre-conditions and target abstraction → Task 3 adds the context methods.
- §3 Analysis → Task 5 implements `findCandidate`, helpers, and pre-order walk.
- §4 Transformation → Task 5 implements `rewrite`.
- §5 Legality → Task 5 surfaces rejection paths with debug logs; Task 8 covers each rejection case in lit.
- §6 Prereqs (#1 ApplicableUnits utility, #2 SchedulerExtContext extensions, #3 pipeline integration) → Tasks 2, 3, 6.
- §7 File Structure → File table at top of plan and individual task headers map 1:1.
- §8 Tests → Tasks 7, 8, 9 cover positive, negative, round-trip, idempotent, and debug categories.
- §9 Open Items → PE/PT note carried in `isPerCoreletUnit`'s switch (the case for them isn't enumerated today, and the `report_fatal_error` fallback would surface them at runtime when the enum is extended). No task adds a stub for missing enum cases — that's appropriate (no PE/PT in the dialect today; adding a case would be premature).

**Placeholder scan:**
- No "TBD" / "TODO" steps. The release-mode debug-test caveat (Task 9 Step 4) is a real condition note, not a placeholder.

**Type consistency:**
- `ktdf::common::computePipelineApplicableUnits` (Task 2) is the same name used by Task 5's `allUnitsAreCoreletLocal`.
- `ktdf::common::computePipelineEnclosingScope` (Task 1) is the same name used by Task 5's `findCandidate`.
- `getNumCorelets` / `isPerCoreletUnit` (Task 3) are the same names used by Task 5.
- `createParallelizeLoopsAcrossInstancesPass` (Task 4) and the no-arg overload appear in Task 6.

No drift between tasks.
