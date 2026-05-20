# Path Expansion Design

## Purpose

This document describes the design for a new path expansion transformation that legalizes data movement in [`ktdf.pipeline`](../src/Dialects/KTDF/KTDFOps.td) IR using an explicit architecture graph as the source of truth for legal routing. 

For examples of the before/after IR see test files in [`test/Transform/PathExpansion/`] directory.
---

## Goals

- introduce an architecture graph abstraction that represents legal hardware routing
- use that graph as the ground truth for path finding and legalization
- infer stage endpoints and anchor points from existing IR
- identify legal source-to-destination paths through the architecture
- select the shortest path in phase 1 (in future the path choosen may be driven by workload balancing, bandwidth utilization, and other requirements).
- rewrite pipelines by inserting intermediate stages and resources where needed
- rebuild dependency and private resources consistently
- leave already-legal pipelines unchanged

The design should also set up future extensions for:

- weighted or policy-driven path selection
- more general DAG legalization
- reusable materialization abstractions shared with other transforms

---

## Non-goals for Phase 1

Phase 1 intentionally does not attempt to solve every possible case.

Out of scope for phase 1:

- configurable architecture descriptions from input json file
- support for branching stage DAG legalization. Phase 1 will still analyze the full stage DAG, but rewriting is only supported when the stage topology is a single linear chain.
- general weighted path search beyond shortest path
- path expansion in nested pipelines where a single stage represents multiple "applicable_units"

---

## High-Level Pass Scope

A new dedicated path expansion pass will be added under [`src/Transform`](../src/Transform/).

The pass operates on [`mlir::ModuleOp`](../src/Transform/Passes.td:6) similarly to the existing transform passes in [`src/Transform/Passes.td`](../src/Transform/Passes.td).

At a high level the pass performs these steps:

1. build stage dependency structure for each pipeline
2. identify anchor, source, and sink stages
3. infer resource endpoints for those stages
4. query the architecture graph for legal paths
5. choose the shortest path
6. compare the chosen path with the currently realized stage path
7. if already legal, do nothing
8. otherwise construct a rewritten pipeline plan
9. rematerialize the affected pipeline region with new private resources, stages, and dependencies

---

## Core Design Principles

### 1. Architecture graph is the ground truth

The architecture graph is the authoritative description of legal routing and transfer semantics.

It answers:

- what resources exist
- which resources can connect directly
- what transfer unit (ie load/store unit) is used on a given hop
- whether a hop requires FIFO-based streaming (eg. lxlu->to-sfp)
- what FIFO kind applies to that hop
- what shortest path exists between two resources

### 2. Plan first, materialize last

The transformation separates analysis from IR mutation:

- **Analyze**: inspect original stages, infer endpoints and paths
- **Plan**: update [`PipelineTree`](../src/Analysis/PipelineTree.hpp) structure and attach side information
- **Materialize**: create new IR from the modified tree

This is essential because [`ktdf.private`](../src/Dialects/KTDF/KTDFOps.td) cannot be expanded in place when result count and types change.

### 3. PipelineTree is the structural plan

[`PipelineTree`](../src/Analysis/PipelineTree.hpp) is both the analysis structure and the planning output:

- **Analysis phase**: discover stage DAG, topological ordering, source/sink identification
- **Planning phase**: directly modify the tree (add/remove nodes, update dependencies)
- **Materialization phase**: walk the modified tree to create new IR

Resource inference and FIFO validation inspect MLIR stage contents directly during analysis. The tree provides bidirectional mapping between IR operations and tree nodes.

### 4. Keep future refactoring easy

A new path-expansion-specific materializer will be created, but it should intentionally mirror the structure of the current materializer implementation used by stage coarsening (where possible) so that future comparison and extraction of common code is straightforward.

---

## Architecture Graph

### Overview

The architecture graph models hardware routing at the resource level.

The graph has two main categories of nodes:

- memory-resource nodes
- compute-resource nodes

Examples relevant to phase 1 include:

- `HBM`
- `LX`
- `SFP`
- `PE`

Transfer units such as `L3LU`, `L3SU`, `LXLU`, and `LXSU` are not represented as graph nodes. They are represented as metadata attached to graph edges.

### Why resource-only nodes

Using resource-only nodes keeps pathfinding simple and matches the conceptual question the pass must answer:

- route data from source resource to destination resource

Transfer semantics are still preserved by attaching them directly to traversable edges.

This supports:

- concise visualization and dumping
- efficient neighbor traversal
- efficient shortest-path search
- direct lookup of per-hop rewrite requirements after a path is selected

### Graph properties

The graph implementation must support:

- concise and readable dumping
- efficient path search from one node to another
- efficient shortest-path search between two nodes

An adjacency-list representation is the recommended implementation.

### Proposed location

The graph should live under analysis rather than transform code:

- [`src/Analysis/ArchitectureGraph.hpp`](../src/Analysis/ArchitectureGraph.hpp)
- [`src/Analysis/ArchitectureGraph.cpp`](../src/Analysis/ArchitectureGraph.cpp)

This is preferable because the graph is a reusable analysis concept, even if phase 1 only uses it from path expansion.

### Node model

Recommended conceptual node model:

- `ResourceNodeKind`
  - `Memory`
  - `Compute`
- `ResourceId`
  - phase-1 built-in identifiers for resources such as `HBM`, `LX`, `SFP`, and potentially `PE`

The resource vocabulary should align with or be compatible with the existing enums in [`src/Dialects/KTDF/KTDFEnums.td`](../src/Dialects/KTDF/KTDFEnums.td).

### Edge model

Each directed edge carries local metadata. Use a unified edge record with optional fields.

Recommended conceptual fields:

- source resource
- destination resource
- edge kind
- transfer unit
- optional FIFO kind
- optional future cost

The meaning is:

- memory-to-memory edges typically use transfer semantics and do not require FIFO information
- memory-to-compute or compute-to-memory edges may require FIFO information
- future cost is reserved for weighted path selection later

### Edge categories

The document should treat edges semantically as two categories even though the implementation uses one unified record:

- storage transfer hops
- streaming hops

Examples:

- `HBM -> LX`
  - storage transfer hop
  - transfer unit `L3LU`
  - no FIFO
- `LX -> HBM`
  - storage transfer hop
  - transfer unit `L3SU`
  - no FIFO
- `LX -> SFP`
  - streaming hop
  - transfer unit `LXLU`
  - FIFO kind `lxlu-to-sfp`
- `SFP -> LX`
  - streaming hop
  - transfer unit `LXSU`
  - FIFO kind `sfp-to-lxsu`

### Built-in phase-1 Spyre graph

Phase 1 uses a built-in hardcoded Spyre architecture graph.

At minimum it should include:

- memory resources:
  - `HBM`
  - `LX`
- compute resources:
  - `SFP`
  - `PE`

A likely built-in phase-1 graph is:

- `HBM -> LX`
- `LX -> HBM`
- `LX -> SFP`
- `SFP -> LX`

Additionally for multi-path selection testing:

- `SFP -> PE`
- `PE -> LX`
- and or the reverse directions as appropriate for the intended route vocabulary

The presence of a longer alternate route through `PE` is specifically useful to validate shortest-path selection.

### Path search policy

Phase 1 path selection policy:

- discover legal paths through the graph (in debug mode report all legal paths for debugging purposes)
- choose the shortest path
- shortest path should be implemented with BFS on the resource graph

Later phases may replace this with weighted path search or policy-driven cost models.

### Required graph APIs

Recommended public APIs:

- `dump(raw_ostream &)`
- `getNeighbors(resource)`
- `findPath(source, destination)` or `findAllPaths(source, destination)` if needed
- `findShortestPath(source, destination)`
- `getEdgeInfo(source, destination)`

The pass should be able to ask:

- whether a direct hop is legal
- what transfer unit and FIFO kind apply to a given direct hop
- what shortest path exists between endpoints

---

## Stage DAG and Structural Analysis

### Use of PipelineTree

[`PipelineTree`](../src/Analysis/PipelineTree.hpp) is used as the stage-structure abstraction for path expansion planning.

It is specifically responsible for:

- discovering stages and their dependency DAG
- topological ordering
- source-stage and sink-stage discovery
- representing the rewritten final stage DAG after inserted stages are added

This is a good fit because [`StageNode`](../src/Analysis/PipelineTree.hpp:179) already stores dependency edges via [`StageNode::addDependency`](../src/Analysis/PipelineTree.hpp:193) and [`StageNode::getDependencies`](../src/Analysis/PipelineTree.hpp:196).

### Phase-1 DAG scope

Phase 1 will:

- analyze the full stage DAG
- reject pipelines whose stage topology is not a single linear chain

This means path expansion has a general DAG analysis front-end, but a restricted legalization back-end.

Branching DAGs should produce a diagnostic and be left unmodified.

### Source and sink discovery

Source and sink stage discovery should come from topological analysis of the stage DAG.

This is an important principle of the design:

- anchor stages are important
- but the start and end stages are equally important
- source and sink determination must not be hardcoded to particular stage positions or resource assumptions

In a linear chain:

- source stages are those with no incoming dependencies
- sink stages are those with no outgoing dependencies

These boundary stages are then inspected to infer starting and ending resource endpoints.

---

## Endpoint and Anchor Inference

### Anchor stages

Any stage with a single [`applicable_units`](../src/Dialects/KTDF/KTDFAttributes.td:94) is treated as an anchor point. Any stage with multiple units listed will be flagged as unsupported.

In phase 1, these are the primary explicit compute anchors.

If a stage has [`applicable_units`](../src/Dialects/KTDF/KTDFAttributes.td:94), the pass should treat that as an explicit statement about the compute resource associated with that stage.

### Start and end points

Source and sink stages are discovered from the stage DAG and then inspected for their data endpoints.

For example:

- a source stage with [`ktdf.data_transfer`](../src/Dialects/KTDF/KTDFOps.td) from an `HBM` memref implies a start resource of `HBM`
- a sink stage with [`ktdf.data_transfer`](../src/Dialects/KTDF/KTDFOps.td) to an `LX` memref implies an end resource of `LX`

### Stages without applicable_units

A hybrid rule is used:

- if a stage has [`applicable_units`](../src/Dialects/KTDF/KTDFAttributes.td:94), that defines the anchor resource
- otherwise infer endpoints from stage contents

Phase 1 should support inference for stages without explicit [`applicable_units`](../src/Dialects/KTDF/KTDFAttributes.td:94), including the middle compute stage if needed.

### Primary inference sources

The primary endpoint inference mechanism should inspect:

- [`ktdf.data_transfer`](../src/Dialects/KTDF/KTDFOps.td)
- memref memory-space types
- stage position in the DAG
- stage role as source, sink, or anchored compute stage

Memory endpoints can be inferred from memref memory space attributes such as [`#ktdf.memory_space<HBM>`](../src/Dialects/KTDF/KTDFAttributes.td:45).

### FIFO-based validation, not primary path inference

[`ktdf.read_from_fifo`](../src/Dialects/KTDF/KTDFOps.td:146) and [`ktdf.write_to_fifo`](../src/Dialects/KTDF/KTDFOps.td:197) should be used after endpoint inference as a validation layer.

This is important.

FIFO information should not be the primary path-selection mechanism. Instead it should validate the consistency of the inferred path and stage semantics.

Examples:

- a [`ktdf.read_from_fifo`](../src/Dialects/KTDF/KTDFOps.td:146) with `<"hbm-to-sfp", ...>` suggests the current stage consumes data from a producer side associated with `HBM` and should be consistent with DAG order
- the same FIFO also suggests that the consuming stage should include `SFP` as an applicable unit or inferred compute anchor
- a [`ktdf.write_to_fifo`](../src/Dialects/KTDF/KTDFOps.td:197) should similarly be consistent with the stage endpoint and downstream path

In phase 1 this should be treated as a validation and assertion step once endpoints and chain order have been inferred.

---

## Original Path Versus Desired Path

### Original path

The pass must infer the currently original path represented by the existing pipeline stages.

This is not merely the source and sink resources. It is the sequence of realized resource hops implied by:

- stage structure
- transfer endpoints
- FIFO connections
- anchor stages

### Desired path

The desired path is the shortest legal path returned by the architecture graph between the inferred endpoints.

### Idempotency

If the currently original path already matches the selected shortest legal path, the pass must not make any changes.

This applies to already-legal inputs.

The pass should preserve IR unchanged in that case rather than rewriting it into a normalized alternative form.

---

## Legalization Strategy

### Overview

When the original path does not match the selected shortest legal path, the pass legalizes the pipeline by:

- rewriting existing transfer stages where appropriate
- preserving anchored compute stages where possible
- inserting intermediate transfer stages for missing hops
- allocating intermediate buffers if the chosen path introduces memory resources not represented in the current stages
- allocating FIFO slots for streaming hops
- rebuilding synchronization around the new stage DAG

### Example from basic.mlir

Input shape:

- stage 1 transfers directly from `HBM` to FIFO intended for `SFP`
- stage 2 computes on `SFP`
- stage 3 transfers directly from FIFO back to `HBM`

Architecture graph says legal path is:

- `HBM -> LX -> SFP`
- `SFP -> LX -> HBM`

Therefore legalization should:

- change first transfer stage into `HBM -> LX`
- insert intermediate `LX -> SFP` stage
- preserve the anchored compute stage but change its FIFO kinds if needed
- insert `SFP -> LX` stage
- change final transfer stage into `LX -> HBM`

Result:

- 3 stages become 5 stages

### Partial expansion

The pass must also support partially expanded inputs.

Example:

- one side is already legalized, such as `HBM -> LX -> SFP`
- the other side still contains a direct illegal `SFP -> HBM`

In this case the pass should only expand the side that still requires legalization.

---

## Rewrite Planning Model

The planning layer is intentionally lightweight and centered around the [`PipelineTree`](../src/Analysis/PipelineTree.hpp) as the single source of truth.

### Core Principle: PipelineTree is Central

The [`PipelineTree`](../src/Analysis/PipelineTree.hpp) is not just an analysis structure—it is the primary data structure that the planner modifies directly:

- The planner updates the [`PipelineTree`](../src/Analysis/PipelineTree.hpp) in-place by adding, removing, or modifying stage nodes
- Stage dependencies are tracked directly in the tree via [`StageNode::addDependency`](../src/Analysis/PipelineTree.hpp:193)
- The modified tree represents the final rewritten stage DAG
- No parallel or redundant abstractions duplicate this structural information

### Lightweight Side Information

The planner stores only minimal side information needed by the materializer:

1. **Stage Materialization Info** - attached to each [`StageNode`](../src/Analysis/PipelineTree.hpp:179):
   - Materialization kind (preserve original, adapt FIFO kinds, or synthetic transfer)
   - Transfer materialization details (template ops and architecture hops)
   - Applicable units if needed

2. **Private Resource Specs** - list of new resources to allocate:
   - Resource type (e.g., LX)
   - Element type and shape

This side information is stored in simple maps and vectors, not in a separate tree-like structure.

### Analysis-Only Structures

Some structures are used only during analysis and discarded after planning:

- **StageSummary** - temporary analysis of each original stage's endpoints and properties
- **Original path inference** - used to compare with architecture graph, then discarded

These are not part of the final plan output.

### Planning Algorithm

The planner operates in clear, progressive phases that separate concerns and build up the complete plan incrementally:

#### Initial Analysis Phase

1. **Analyze** - inspect original stages, infer endpoints and paths (creates temporary `StageSummary` records)
2. **Compare** - check original path against architecture graph's shortest path
3. **Decide** - determine if changes are needed

If no changes are needed, return early with `changed = false`.

#### Progressive Planning Phase (4 Steps)

The planning phase progressively completes the plan through four distinct steps, each building on the previous:

**Step 1: Connection Legalization**
- Walk through original sorted stages (e.g., 3 stages for basic.mlir)
- For each adjacent pair, check if their connection is legal via architecture graph
- If illegal, query `findShortestPath` to determine intermediate hops
- Insert unmaterialized intermediate stage nodes into PipelineTree
- Create `StageMaterializationInfo` for each stage with:
  - `kind` set appropriately (kPreserveOriginal, kAdaptFifoKinds, or kSyntheticTransfer)
  - `applicable_unit` filled in for compute stages
  - `transfers` list initially empty
- Update dependencies to maintain proper ordering
- Output: Expanded sorted stages (e.g., 5 stages for basic.mlir)

**Step 2: Original Stage Transfer Analysis**
- Walk through expanded stages, focusing on original stages (non-null operations)
- For each [`ktdf.data_transfer`](../src/Dialects/KTDF/KTDFOps.td) operation:
  - Check if adjacent stage (in sorted order) is an intermediate stage
  - If yes, create `PrivateResourceSpec` for intermediate memory resource
  - Create `TransferMaterializationInfo` and add to stage's transfers list
  - Set `source_private_resource` or `dest_private_resource` based on direction
- Output: Memory buffer specs created, transfers populated for original stages

**Step 3: FIFO Operation Analysis**
- Walk through expanded stages (original)
- For each [`ktdf.read_from_fifo`](../src/Dialects/KTDF/KTDFOps.td:146) or [`ktdf.write_to_fifo`](../src/Dialects/KTDF/KTDFOps.td:197):
  - Check if this stage faces an intermediate stage
  - Query architecture graph edge for correct FIFO type (not from illegal input IR)
  - Create or extend `PrivateResourceSpec` for FIFO
  - Assign slot indices incrementally per FIFO type
  - Create `TransferMaterializationInfo` linking FIFO to transfer
- Output: FIFO specs created, transfers updated with FIFO resources and slot indices

**Step 4: Intermediate Stage Transfer Population**
- Walk through intermediate stages (those with null operations)
- For each intermediate stage:
  - Examine predecessor stage's transfers targeting this intermediate
  - Examine successor stage's transfers sourcing from this intermediate
  - Create corresponding `TransferMaterializationInfo` entries
  - Reuse `PrivateResourceSpec` pointers and slot indices from neighboring stages
- Output: Complete transfer info for all stages

#### Why This Progressive Approach

This 4-step algorithm solves a critical problem in the original implementation:

**Problem**: When input IR contains illegal connections (e.g., HBM→SFP with `hbm-to-sfp` FIFO), the planner cannot correctly infer FIFO types without first determining what intermediate stages are needed.

**Solution**: By separating connection legalization (Step 1) from FIFO analysis (Step 3), the algorithm:
1. First establishes the complete legal stage structure
2. Then queries the architecture graph for correct FIFO types based on actual legal edges
3. Avoids trying to infer FIFO types from illegal connections in the input IR

**Benefits**:
- Clear separation of concerns: structure, memory, FIFOs, and intermediate stages
- Each step has complete information from previous steps
- Debug output at each step provides visibility into the transformation
- Easier to maintain and extend
- Correct FIFO type inference based on legal architecture edges

#### Debug Output

Each step produces clear debug output showing:
- Current stage list with classifications (original vs synthetic)
- Resources allocated in this step
- Transfers created or updated
- State of `StageMaterializationInfo` for affected stages

Example debug output for basic.mlir:
```
=== Step 1: Connection Legalization ===
Original stages: 3
Expanded stages: 5
  Stage 0: HBM (transfer-only, original)
  Stage 3: LX (transfer-only, synthetic)
  Stage 1: SFP (compute, original)
  Stage 4: LX (transfer-only, synthetic)
  Stage 2: HBM (transfer-only, original)

=== Step 2: Original Stage Transfer Analysis ===
Stage 0: 2 transfers to LX buffers
Stage 2: 1 transfer from LX buffer
Memory resources: 3 LX buffers

=== Step 3: FIFO Operation Analysis ===
Stage 1: 2 FIFO reads (lxlu-to-sfp), 1 FIFO write (sfp-to-lxsu)
FIFO resources: 2 FIFO types

=== Step 4: Intermediate Stage Transfer Population ===
Stage 3: 4 transfers (2 HBM→LX, 2 LX→SFP)
Stage 4: 2 transfers (1 SFP→LX, 1 LX→HBM)

=== Planning Complete ===
Total stages: 5
Total private resources: 5 (3 LX buffers, 2 FIFOs)
```

The output is:
- Modified [`PipelineTree`](../src/Analysis/PipelineTree.hpp) with final stage DAG
- Complete side information map for materialization
- All private resource specs with proper linkage

### No Redundant Abstractions

The design explicitly avoids:
- Separate "stage modification plan" trees parallel to PipelineTree
- Duplicate tracking of stage order or dependencies
- Complex planning-specific node hierarchies

The [`PipelineTree`](../src/Analysis/PipelineTree.hpp) already provides everything needed for structural representation. The planner simply updates it and attaches minimal semantic annotations.

---

## Private Region and Resource Allocation

### Why rematerialize private

[`ktdf.private`](../src/Dialects/KTDF/KTDFOps.td) cannot simply be expanded in place when new buffers, FIFO slots, or tokens are added because the result signature changes.

Therefore phase 1 should:

- create a new corresponding private region
- rematerialize the needed private resources
- update all stage uses to reference the new private results
- remove or replace the old pipeline region as part of full pipeline rematerialization

### Buffer allocation

If the chosen path includes additional memory resources not represented by the existing set of stages, the pass must allocate buffers in those intermediate resources.

For the baseline example, path expansion creates new `LX` buffers.

Expected pattern:

- one `LX` buffer per logical tensor flow that must be staged through `LX`
- for the basic example this likely means:
  - one intermediate `LX` buffer for input A
  - one intermediate `LX` buffer for input B
  - one intermediate `LX` buffer for output C

The type, shape and size should be derived from:

- the transferred slice shape
- the element type
- the intermediate memory space

For example the `HBM` memref in the `HBM` -> `SFP` data transfer specifies the size/shape/type of the `LX` buffer that is being introduced in the middle.

Planner records for such resources should be symbolic rather than SSA-based. A planned intermediate buffer should be referred to through a planner-owned identifier until materialization creates the corresponding [`ktdf.private`](../src/Dialects/KTDF/KTDFOps.td) result value.

### FIFO allocation

FIFO slots are allocated only for hops whose graph edge metadata requires them.

This choice should come from the architecture graph edge, not ad hoc pass logic.

Examples:

- `LX -> SFP` uses `lxlu-to-sfp`
- `SFP -> LX` uses `sfp-to-lxsu`

As with planned intermediate buffers, planned FIFO resources should be referred to symbolically by the planner until the new [`ktdf.private`](../src/Dialects/KTDF/KTDFOps.td) is materialized and concrete SSA values exist.

### Tokens

The planner should not preserve old token identities.

Instead:

- compute the final stage DAG
- identify dependency edges that require tokens
- materialize fresh tokens from the new [`ktdf.private`](../src/Dialects/KTDF/KTDFOps.td) based on that DAG

---

## Materialization Design

### Overall Decision

Phase 1 uses a new path-expansion-specific materializer that consumes the modified [`PipelineTree`](../src/Analysis/PipelineTree.hpp) and side information from the planner.

The materializer should not reuse the current stage-coarsening materializer directly. However, it should be intentionally shaped similarly to enable future refactoring and code sharing.

### Required Preparatory Rename

The current materializer should be renamed to clarify its purpose:

From:
- [`src/Transform/PipelineTree/Materializer.hpp`](../src/Transform/PipelineTree/Materializer.hpp)
- [`src/Transform/PipelineTree/Materializer.cpp`](../src/Transform/PipelineTree/Materializer.cpp)

To:
- `src/Transform/StageCoarsening/Materializer.hpp`
- `src/Transform/StageCoarsening/Materializer.cpp`

This clarifies that it's specific to stage coarsening, not a general-purpose materializer.

### Path Expansion Materializer Location

New files:
- [`src/Transform/PathExpansion/Materializer.hpp`](../src/Transform/PathExpansion/Materializer.hpp)
- [`src/Transform/PathExpansion/Materializer.cpp`](../src/Transform/PathExpansion/Materializer.cpp)

### Structural Similarity

The new materializer should mirror the stage-coarsening materializer structure:

- Similar function names: `materialize`, `materializeTreeNode`, `materializeLoopNode`, `materializePipelineNode`, `materializeStageNode`, `materializePrivateNode`
- Similar class member organization
- Similar tree-walking approach

This enables easier comparison and future extraction of common functionality.

### Materialization Algorithm

The materializer operates on the modified [`PipelineTree`](../src/Analysis/PipelineTree.hpp):

1. **Walk the tree** in pre-order
2. **Materialize private node** first:
   - Create new [`ktdf.private`](../src/Dialects/KTDF/KTDFOps.td) with all resources
   - Build mapping from `PrivateResourceSpec` to SSA values
3. **Materialize each stage**:
   - Look up `StageMaterializationInfo` from the side map
   - Based on the kind:
     - `kPreserveOriginal`: clone the original stage body
     - `kAdaptFifoKinds`: clone body but update FIFO types using transfer info
     - `kSyntheticTransfer`: generate new transfer stage from transfer info
   - Wire up dependencies using tokens from the tree's dependency edges
4. **Replace original pipeline** with the new one

### Input to Materializer

The materializer receives:

- Modified [`PipelineTree`](../src/Analysis/PipelineTree.hpp) (contains final stage DAG structure)
- `llvm::DenseMap<StageNode*, StageMaterializationInfo>` (how to materialize each stage)
- `llvm::SmallVector<PrivateResourceSpec>` (what resources to allocate)

No symbolic references or complex planning structures—just the tree and simple side information.

### Template Usage

The materializer reuses template operations where possible:

- `TransferMaterializationInfo` contains a `template_op` field
- For adapted or synthetic transfers, the materializer clones and modifies the template
- This preserves indexing, sizing, and typing information
- Supports dynamic SSA-defined operands (not just compile-time constants)

---

## Pass Algorithm

The phase-1 pass algorithm follows a clear analyze-plan-materialize flow.

### Step 1: Build PipelineTree

For each [`ktdf.pipeline`](../src/Dialects/KTDF/KTDFOps.td) in the module:

- build a [`PipelineTree`](../src/Analysis/PipelineTree.hpp) from the existing IR
- derive the stage dependency DAG
- reject non-linear topologies for rewriting (phase 1 limitation)

### Step 2: Analyze Stages (Pure Analysis)

Create temporary `StageSummary` records by analyzing each stage:

- identify source stages (no incoming dependencies)
- identify sink stages (no outgoing dependencies)
- identify anchor stages with [`applicable_units`](../src/Dialects/KTDF/KTDFAttributes.td:94)
- inspect [`ktdf.data_transfer`](../src/Dialects/KTDF/KTDFOps.td) endpoints
- inspect memref memory spaces
- classify as transfer-only or compute-containing
- infer anchor resource if no explicit [`applicable_units`](../src/Dialects/KTDF/KTDFAttributes.td:94)

### Step 3: Validate with FIFO Semantics

Use [`ktdf.read_from_fifo`](../src/Dialects/KTDF/KTDFOps.td:146) and [`ktdf.write_to_fifo`](../src/Dialects/KTDF/KTDFOps.td:197) to validate:

- endpoint consistency
- stage order consistency with the DAG
- compute anchor compatibility

### Step 4: Infer Original Path

From the stage summaries, derive the currently realized resource-hop sequence from source through anchors to sink.

### Step 5: Query Architecture Graph

For the inferred endpoints:

- query the architecture graph for legal paths
- choose shortest path using BFS
- fail if no legal path exists

### Step 6: Compare and Decide

Compare original path with selected architecture path:

- if they match: skip this pipeline (already legal)
- if they differ: proceed to rewrite

### Step 7: Update PipelineTree Directly

Modify the [`PipelineTree`](../src/Analysis/PipelineTree.hpp) in-place:

- **Preserve** stages that remain valid (mark with `kPreserveOriginal`)
- **Adapt** stages whose FIFO kinds must change (mark with `kAdaptFifoKinds`, attach transfer info)
- **Insert** new intermediate transfer stages:
  - Create unmaterialized [`StageNode`](../src/Analysis/PipelineTree.hpp:179) instances
  - Set template op if adapting from existing stage
  - Mark with `kSyntheticTransfer` and attach transfer info
- **Update dependencies** directly via [`StageNode::addDependency`](../src/Analysis/PipelineTree.hpp:193)

The modified tree now represents the final stage DAG.

### Step 8: Record Side Information

Store minimal materialization info:

- Map each modified/inserted [`StageNode`](../src/Analysis/PipelineTree.hpp:179) to its `StageMaterializationInfo`
- Record new private resources (intermediate buffers, FIFO slots)

Discard temporary `StageSummary` records—they're no longer needed.

### Step 9: Materialize New IR

Use the path-expansion-specific materializer:

- Walk the modified [`PipelineTree`](../src/Analysis/PipelineTree.hpp) in pre-order
- Materialize the private node first, creating new SSA values for resources
- For each stage node:
  - Look up its `StageMaterializationInfo`
  - Preserve, adapt, or synthesize the stage body accordingly
  - Wire up dependencies using tokens from the tree's dependency edges
- Replace the original pipeline with the new one

---

## Repository Layout

Recommended new and updated files.

### New analysis files

- [`src/Analysis/ArchitectureGraph.hpp`](../src/Analysis/ArchitectureGraph.hpp)
- [`src/Analysis/ArchitectureGraph.cpp`](../src/Analysis/ArchitectureGraph.cpp)

### New transform files

- [`src/Transform/PathExpansion.cpp`](../src/Transform/PathExpansion.cpp)
- [`src/Transform/PathExpansion/PathExpansionMaterializer.hpp`](../src/Transform/PathExpansion/PathExpansionMaterializer.hpp)
- [`src/Transform/PathExpansion/PathExpansionMaterializer.cpp`](../src/Transform/PathExpansion/PathExpansionMaterializer.cpp)

Optional helper files if implementation becomes cleaner with them:

- [`src/Transform/PathExpansion/Planner.hpp`](../src/Transform/PathExpansion/Planner.hpp)
- [`src/Transform/PathExpansion/Planner.cpp`](../src/Transform/PathExpansion/Planner.cpp)

### Existing files to update

- [`src/Transform/Passes.td`](../src/Transform/Passes.td)
- [`src/Transform/CMakeLists.txt`](../src/Transform/CMakeLists.txt)
- [`src/Analysis/CMakeLists.txt`](../src/Analysis/CMakeLists.txt)

### Existing files to rename

Rename the current stage-coarsening materializer files to clarify ownership:

- [`src/Transform/PipelineTree/Materializer.hpp`](../src/Transform/PipelineTree/Materializer.hpp)
- [`src/Transform/PipelineTree/Materializer.cpp`](../src/Transform/PipelineTree/Materializer.cpp)

to stage-coarsening-specific names.

---

## Testing Plan

The tests should focus primarily on intended path-expansion behavior rather than mostly invalid scenarios.

### Baseline golden test

- input: [`test/Transform/PathExpansion/basic.mlir`](../test/Transform/PathExpansion/basic.mlir)
- expected: [`test/Transform/PathExpansion/basic.out.mlir`](../test/Transform/PathExpansion/basic.out.mlir)

This validates:

- HBM to SFP expansion through LX
- SFP to HBM expansion through LX
- intermediate buffer allocation in LX
- FIFO kind changes
- stage insertion
- dependency rewiring

### Test A: LX boundary variant

Create a variant where:

- the first stage reads from `LX` instead of `HBM`
- the final stage writes back to `LX`

This validates:

- source and sink discovery from DAG boundaries
- endpoint inference is not hardcoded to HBM

### Test B: partially expanded case

Create a variant where the input is already partially legalized.

Examples:

- `HBM -> LX -> SFP` already exists, but `SFP -> HBM` still needs expansion
- or the reverse

This validates:

- selective expansion
- pass only inserts the missing legalization path
- already-legal subpath remains unchanged

### Test C: inferred middle-stage anchor

Create a variant where the middle compute stage has no [`applicable_units`](../src/Dialects/KTDF/KTDFAttributes.td:94).

The pass must:

- infer the stage anchor from stage contents
- use FIFO-based validation to confirm the inferred anchor is consistent

### Test D: already-legal no-op case

Create a variant whose original path already matches the shortest legal architecture path.

The pass should:

- leave the IR unchanged

### Test E: shortest path among multiple candidates

The built-in graph should include a longer alternative route, for example one involving `PE`.

A test should confirm:

- the pass chooses the shortest route
- not merely any legal route

This may be covered by a variant of [`basic.mlir`](../test/Transform/PathExpansion/basic.mlir) if the graph includes a longer `SFP -> PE -> LX` style alternative.

### Negative tests

Negative tests are still useful, but they are not the primary emphasis for phase 1.

A small number of negative tests may later be added for:

- branching DAG rejection
- no legal route
- inconsistent FIFO validation

---

## Diagnostics and Assertions

The pass should provide clear diagnostics for unsupported or inconsistent situations.

Examples include:

- branching stage DAG not supported in phase 1
- unable to infer source or sink resource
- unable to infer anchor resource for a compute-containing stage
- no legal architecture path between inferred endpoints
- FIFO direction inconsistent with inferred stage order
- FIFO semantics inconsistent with stage anchor
- partially inferred or contradictory stage endpoint information

---

## Future Follow-Ups

After phase 1 lands, likely follow-up work includes:

### 1. Shared materialization refactor

Once both stage coarsening and path expansion have sibling materializers with similar structure, compare them and extract common functionality into a shared base class or helper layer.

Likely shared areas:

- tree traversal materialization skeleton
- stage dependency to token mapping
- private rematerialization helpers
- value remapping helpers

### 2. Weighted path selection

Add cost-based path selection modes using the edge `cost` field.

Examples:

- shortest path
- load balancing
- congestion-aware routing
- policy-specific preference for certain compute paths

### 3. More general DAG legalization

Extend support from linear chains to branching stage DAGs.

### 4. Configurable architecture descriptions

Allow architecture graph construction from configuration or another representation instead of relying only on a hardcoded built-in model.

### 5. Broader validation coverage

Add more tests for invalid or ambiguous inputs once the core legalization path is stable.

---

## Mermaid Overview

```mermaid
flowchart LR
  A[Build PipelineTree] --> B[Analyze stage DAG]
  B --> C[Infer source sink and anchors]
  C --> D[Infer stage endpoints]
  D --> E[Validate with FIFO semantics]
  E --> F[Query ArchitectureGraph]
  F --> G[Choose shortest path]
  G --> H{Already legal}
  H -->|Yes| I[No-op]
  H -->|No| J[Rewrite PipelineTree DAG]
  J --> K[Plan private buffers fifos tokens]
  K --> L[Materialize new pipeline IR]
```

---

## Final Phase-1 Decisions Summary

- dedicated path-expansion pass
- hardcoded built-in Spyre architecture graph
- architecture graph under analysis in [`src/Analysis`](../src/Analysis/)
- resource-only graph nodes
- edge-local metadata with optional FIFO and cost fields
- full DAG analysis but linear-chain-only rewrite support
- [`PipelineTree`](../src/Analysis/PipelineTree.hpp) used for structural planning and rewritten dependency DAG
- direct MLIR inspection for endpoint inference
- FIFO ops used primarily for validation after inference
- shortest path policy only
- no-op for already-legal shortest-path pipelines
- new path-expansion-specific materializer
- rename current materializer to stage-coarsening-specific naming
- design the new materializer to mirror stage-coarsening materializer structure for future refactoring