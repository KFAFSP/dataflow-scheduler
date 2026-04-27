# Loop Sinking and Stage Coarsening Transformation

## Overview

This document describes a compiler transformation for creating hierarchical pipeline structures from flat pipelines after loop tiling. The transformation is called **stage coarsening** (or **loop sinking**) and operates on an internal tree abstraction representing pipelines, stages, and loops.

---

## Background: Stage Coarsening / Loop Sinking

### Problem
After loop tiling is applied to a pipeline IR:
- **Outer loops** (m0, n0) define tile-level synchronization scope
- **Inner loops** (m1, n1) define execution granularity (e.g., 1x64 sticks)
- **Pipeline remains flat** with all 5 stages at the same level
- **Synchronization is too fine-grained** - happens at every 1x64 iteration

### Goal
Transform the flat pipeline into a **hierarchical pipeline** where:
- **Outer pipeline** operates at tile granularity with coarse-grained synchronization
- **Inner nested pipelines** operate at hardware execution granularity
- **Loops are sunk** into appropriate stages
- **Related stages are coarsened** into single stages containing nested pipelines

---

## Internal Tree Abstraction

### Node Types

The abstraction represents the IR structure as a tree with these node types:

1. **RootNode** - Synthetic top-level container
2. **LoopNode** - Represents `scf.for` operations with induction variable
3. **PipelineNode** - Represents `ktdf.pipeline` operations
4. **StageNode** - Represents `ktdf.stage` operations with stage ID
5. **PrivateNode** - Represents `ktdf.private` operations (buffer allocations)

### Relationships

**Parent-Child Relationships (Blue Arrows)**:
- Represent structural nesting in the IR
- Form a tree structure
- Example: Pipeline → Stage → Loop → nested Pipeline → Stage

**Stage Dependencies (Red Arrows)**:
- Represent token-based synchronization between stages
- Producer-consumer relationships
- Must be preserved during transformations
- Cannot cross pipeline boundaries (enforced in Step 4)

### Key Properties

- Follows patterns from `OperationTree` and `RedundantDefinitionEliminationTree` in deeptools/dcc
- Uses sibling-linked tree structure (first_child, next_sibling pointers)
- Supports various tree traversal orders (pre-order, post-order, BFS, etc.)
- Provides `dump()` method to print tree structure with optional dependency information

---

## Stage Grouping for Coarsening

### Legality Rules

Stage coarsening legality is similar to **loop distribution** rules:
- Stages within loop distribution-preventing dependencies must be kept as a group together and cannot be coarsened independently of other stages within that group.
- Stages that share non-expandable resources (e.g., SFP registers) must be grouped together

### Example Grouping

For the matrix addition kernel with 5 stages:

**Group 1**: Stage 0 (Load from DRAM to LX)
- Single stage group
- Operates on DRAM ↔ LX transfers

**Group 2**: Stages 1, 2, 3 (LX ↔ SFP compute pipeline)
- Stage 1: LX → SFP registers
- Stage 2: SFP compute
- Stage 3: SFP registers → LX
- These share SFP register usage and can be coarsened together

**Group 3**: Stage 4 (Store from LX to DRAM)
- Single stage group
- Operates on LX ↔ DRAM transfers

**Note**: Stage grouping analysis is a future component. The transformation algorithm accepts groups as input.

---

## Four-Step Transformation Algorithm

### Input
- Tiled IR with flat pipeline structure
- Stage grouping information (which stages belong together)

### Step 1: Distribute and Sink Loops into Pipeline

**Actions**:

1. **Move loops into pipeline**:
   - Take loops (m1, n1) that are currently parents of the pipeline
   - Move them to become children of the pipeline
   - This inverts the nesting relationship

2. **Distribute loops by stage group**:
   - For each stage group identified by the grouping analysis
   - Create a copy of the loop nest
   - Insert the loop nest as a child of the pipeline (ie in between the parent pipeline and the stage group)

3. **Expand LX buffers**:
   - Identify which buffers need to be expanded to accommodate tiled access patterns
   - Record which data transfers and compute operations are affected by the expansion
   - Mark these buffers in the pipeline's private data as "expanded"
   - This information is needed when generating IR to properly size the buffers


**Result**:
```
Pipeline
├── Loop(m1, n1) [for Group 1]
├── Stage 0
├── Loop(m1, n1) [for Group 2 - shared by multiple stages]
├── Stage 1
├── Stage 2
├── Stage 3
├── Loop(m1, n1) [for Group 3]
└── Stage 4
```

**Note**: After this step, loops are sunk into the parent pipeline but not the stages. Step 2 will sink further into stages when appropriate.
---

### Step 2: Sink Loops to Stages (Conditional)

**Condition**: Only sink when `loopNest->numChildren() == 1`

**Actions**:
- For each loop nest with exactly one child stage:
  - Move the loop nest to become child of that stage
  - Stage becomes parent of the loop nest
- Loops shared by multiple stages remain as children of pipeline

**Result**:
```
Pipeline
├── Stage 0
│   └── Loop(m1, n1)
├── Loop(m1, n1) [shared by Stages 1, 2, 3]
├── Stage 1
├── Stage 2
├── Stage 3
└── Stage 4
    └── Loop(m1, n1)
```

**Problem Created**: Pipeline has a loop nest as direct child (violates structural legality rules)

---

### Step 3: Iterative Structure Legalization

**Legalization Rules**:
1. A pipeline can only have `ktdf.private` or `ktdf.stage` as immediate children
2. A stage must have a pipeline as immediate parent

**Algorithm** (iterative discovery and correction):

```
while (violations exist):
    traverse tree and find violations
    
    if (pipeline has non-stage, non-private child):
        // Violation: Missing coarsened stage
        child = the violating child (e.g., loop nest)
        create new coarsened stage (e.g., Stage 5)
        insert Stage 5 as child of pipeline
        move child to be child of Stage 5
        continue  // restart search
    
    if (stage has parent that is not a pipeline):
        // Violation: Missing nested pipeline
        stage = the violating stage
        parent = stage's parent (e.g., loop)
        create new nested pipeline
        insert nested pipeline as child of the parent
        move child stages to be children of the nested pipeline
        
        // Privatize buffers for nested pipeline
        identify buffers used only within nested scope
        move those buffers from outer private to nested private
        create new tokens for nested pipeline
        
        continue  // restart search
    
    // No violations found, exit loop
```

**Execution Example**:

**Iteration 1**:
- **Find**: Outer pipeline has loop nest as child (violation of rule 1)
- **Fix**: Create `Stage 5`, insert between pipeline and loop nest
- **Result**: Pipeline → Stage 5 → loop nest → Stages 1, 2, 3

**Iteration 2**:
- **Find**: `Stage 1, 2, 3` has loop nest as parent, not a pipeline (violation of rule 2)
- **Fix**: 
  - Create nested pipeline
  - Insert between loop nest and stages
  - Privatize SFP buffers to nested pipeline
  - Create new tokens for nested pipeline
- **Result**: Pipeline → Stage 5 → loop nest → nested pipeline → Stages 1, 2, 3

**Iteration 3**:
- **Find**: No violations
- **Exit**: Structure is now legal

**Final Structure**:
```
Pipeline (outer)
├── Stage 0
│   └── Loop(m1, n1)
├── Stage 5 (coarsened)
│   └── Loop(m1, n1)
│       └── Pipeline (nested)
│           ├── Private (SFP buffers, tokens)
│           ├── Stage 1
│           ├── Stage 2
│           └── Stage 3
└── Stage 4
    └── Loop(m1, n1)
```

---

### Step 4: Fix Cross-Boundary Token Dependencies

**Purpose**: Adjust stage dependencies to respect pipeline boundaries

**Actions**:

1. **Identify cross-boundary dependencies**:
   - Traverse all stage dependency edges (red arrows)
   - Find dependencies where source and target stages are in different pipelines
   - Example: Stage 0 → Stage 1 (crosses from outer to nested pipeline)
   - Example: Stage 3 → Stage 4 (crosses from nested to outer pipeline)

2. **Break and redirect dependencies**:
   - Remove direct token dependencies between stages in different pipelines
   - Coarsened stage acts as synchronization boundary
   - Create new tokens for each pipeline level

**Result**:
- **Outer pipeline dependencies**: Stage 0 → Stage 5 → Stage 4
- **Nested pipeline dependencies**: Stage 1 → Stage 2 → Stage 3
- Token dependencies respect pipeline boundaries
- Each pipeline has independent token flow

---

## Key Design Principles

### Separation of Concerns
- **Step 1-2**: Structural transformations (loop distribution and sinking)
- **Step 3**: Legalization (iterative discovery and correction)
- **Step 4**: Dependency adjustment (token scope management)

### Iterative Legalization
- Step 3 discovers violations rather than assuming them
- Handles arbitrary nesting structures
- Validates legality rules systematically
- Applies corrections only where needed

### Preservation of Semantics
- Stage dependencies (red arrows) preserved throughout Steps 1-3
- Only Step 4 modifies dependencies to respect pipeline boundaries
- Buffer privatization ensures correct scoping
- Token creation maintains synchronization semantics

### Future Extensibility
- Stage grouping analysis is pluggable (future component)
- Transformation accepts groups as input
- Can handle different grouping strategies
- Supports arbitrary number of groups

---

## Implementation Notes

### Tree Abstraction Design
- Follow patterns from `OperationTree` (dcc/src/Analysis/OperationTree.hpp)
- Follow patterns from `RedundantDefinitionEliminationTree` (dcc/src/Transform/Sentient/Analyses/RedundantDefinitionEliminationTree.hpp)
- Use sibling-linked tree structure
- Provide dump() method focusing on tree structure
- Stage dependencies printed as optional additional information

### Dump Method Behavior
- Default: Show tree structure only (parent-child relationships)
- Optional parameter: `showDependencies=true` to include stage dependencies
- Clear, indented output for debugging and visualization

---

## Benefits

### Performance
- **Reduced synchronization overhead**: Synchronization at tile boundaries instead of every iteration
- **Better hardware utilization**: Overlapping of tile-level operations
- **Improved memory hierarchy usage**: Explicit representation of data movement

### Compiler Analysis
- **Clear structure**: Hierarchical representation matches hardware organization
- **Analyzability**: Static structure enables performance estimation
- **Maintainability**: Separation of concerns in transformation steps

### Hardware Mapping
- **Outer pipeline**: Maps to memory engines (DMA units)
- **Inner pipeline**: Maps to compute units (SFP)
- **Explicit hierarchy**: Reflects memory hierarchy (DRAM → LX → SFP)

---

## Related Documentation

- [Hierarchical Pipeline Abstraction](Hierarchical%20Pipeline%20Abstraction.md) - Conceptual overview
- [Pipeline.md](Pipeline.md) - Scheduler pipeline details
- [MatrixAdd.md](MatrixAdd.md) - Complete transformation example

This comprehensive design enables systematic transformation of flat pipelines into hierarchical structures suitable for hardware mapping while maintaining correctness and extensibility.