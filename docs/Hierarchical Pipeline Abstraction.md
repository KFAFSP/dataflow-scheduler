# Hierarchical Pipeline Abstraction for Scheduling Dataflow Accelerators

## 1. Introduction

Modern accelerator architectures increasingly rely on **dataflow-style execution**, where computation and memory transfers are orchestrated as a pipeline of operations. Unlike traditional CPUs or GPUs—where scheduling is largely implicit and dynamically handled by hardware—many domain-specific accelerators rely on **static orchestration of compute and data movement** across multiple hardware engines such as DMA (load/store) units and compute engines.

Designing intermediate representations (IRs) that can capture such behavior in a **clear, analyzable, and hardware-faithful way** is a challenging problem. 
The **hierarchical pipeline abstraction** addresses this gap by explicitly modeling programs as **pipelines composed of stages**, where each stage represents a logical unit of execution that may correspond to hardware components such as load/store engines or compute engines. Pipelines can be nested within loops and within other pipelines, allowing the representation to naturally capture **data reuse hierarchies and multi-level scheduling decisions**.

---


## 2. Tensor Addition Kernel Example Using Pipelines and Stages


To illustrate the concepts of **pipelines**, and
**stages**, consider a simple 2-D tensor addition kernel:

C = A + B

In this example:

-   Tensors **A**, **B**, and **C** reside in **HBM (high-bandwidth
    memory)**.
-   Data is staged through **LX (local scratchpad memory)**.
-   Computation is performed at **SFP (special functional unit)**.

The goal is to move tiles of tensors from HBM into LX, perform the
computation at SFP, and then write the results back to HBM. 

------------------------------------------------------------------------

### Pseudo Kernel Structure

The following pseudo-code illustrates how a tensor addition kernel can
be structured using pipelines and stages.

```
kernel tensor_add(A, B, C)

for tile_i 
 for tile_j

      token tile_loaded, tile_computed, tile_stored
      pipeline {
        stage {                              // Reading a tile into LX
          depend(out: tile_loaded)
          move A[tile_i, tile_j] from HBM to LX
          move B[tile_i, tile_j] from HBM to LX
        }

        stage {                              // Computing on a tile at SFP
          depend(in: tile_loaded)
          depend(out: tile_computed)
          for ii
            for jj
              push A[ii, jj] from LX to SFP input
              push B[ii, jj] from LX to SFP input
              execute add at SFP
              move C[ii, jj] from SFP to LX
        }

        stage {                              // Writing back a tile into HBM
          depend(in: tile_computed)
          depend(out: tile_stored)
          move C[tile_i, tile_j] from LX to HBM
        }
      }
```

## 3. Pipeline

A pipeline represents a structured directed acyclic graph (DAG) of execution steps that process data in stages. Instead of executing one step completely before starting the next, pipelines allow different stages to operate concurrently on different iterations of work, thereby overlapping computation and data movement. In dataflow architectures, pipelines reflect how hardware engines operate: one unit may be loading data, another computing, and another storing results simultaneously.

In the example above, the pipeline operates at the tile level. Each iteration of the outer loops (tile_i, tile_j) processes one tile of tensors A, B, and C. The pipeline divides the work for that tile into three stages: loading the data into local memory, computing on the data, and writing the results back. As multiple tiles progress through the pipeline, the stages can execute in parallel across different tiles, improving throughput and hiding memory latency. As an example, the last stage of writing back computed tile into HBM can happen in parallel with the first stage of reading next tile from HBM.


## 4. Stage

A stage represents a logical step within the pipeline where a specific type of work is performed. Each stage "eventually" correspond to a particular hardware capability such as memory transfer or computation. Stages form the building blocks of the pipeline and define how work flows through the system.

In the example, there are three stages:

The first stage is responsible for loading data from global memory (HBM) into local memory (LX). This stage reads the tile of tensors A and B corresponding to the current (tile_i, tile_j) indices and moves them into the local memory where they can be accessed more efficiently by compute units.

The second stage performs the computation for the tile. It iterates over the tile using the inner loops ii and jj. For each micro-block, the stage pushes elements of A and B from LX into the SFP compute unit's FIFO, performs the addition operation, and then moves the resulting elements of C back into LX. This stage therefore represents the core arithmetic work of the kernel.

The third stage writes the computed results back to global memory. After the compute stage finishes processing the tile and populates C in LX, this stage transfers the tile of C from LX back to HBM so that the results become globally visible.

## 5. Tokens and Stage Connectivity

Stages in the pipeline are connected through tokens, which act as synchronization signals between stages. A token represents the completion of work in one stage and allows the next stage to begin its execution. Tokens therefore establish producer–consumer relationships between stages and enforce the correct ordering of operations. Each stage declares its dependencies using `depend(in:)` to consume a token and `depend(out:)` to produce one.

In the example, three tokens are declared: `tile_loaded`, `tile_computed`, and `tile_stored`.

The first stage has only `depend(out: tile_loaded)`. This means it does not depend on any previous stage and, once it completes loading the tile into LX, it produces `tile_loaded`. The presence of this token signals that the input data for the tile is ready.

The second stage specifies `depend(in: tile_loaded)` and `depend(out: tile_computed)`. This indicates that the compute stage must wait until `tile_loaded` is produced by the load stage before it can begin execution. Once computation on the tile is completed, it produces `tile_computed`, signaling that the results are available in LX.

The third stage has `depend(in: tile_computed)` and `depend(out: tile_stored)`. This stage waits for the compute stage to finish before transferring the result tile back to HBM. After the store operation completes, it produces `tile_stored`, indicating that the tile has been fully processed.

Through these token dependencies, the pipeline establishes the following execution flow:

Load Stage → Compute Stage → Store Stage


## 6. Nested (or) Hierarchical pipelines

In many accelerator workloads, a single flat pipeline is not sufficient to capture the full execution structure of the program. This is because accelerator programs often operate at multiple levels of granularity, such as tiles, micro-tiles, and vector operations. To represent these structures cleanly, the pipeline abstraction can be extended to support nested or hierarchical pipelines, where a stage of one pipeline can contain another pipeline.

A hierarchical pipeline allows the scheduling structure to mirror the natural hierarchy of the computation and the hardware. The outer pipeline typically orchestrates coarse-grained work, such as moving tiles between memory levels, while inner pipelines handle finer-grained operations, such as streaming micro-blocks of data through compute units. This layered structure makes it possible to represent both large-scale data movement and fine-grained compute pipelines within the same abstraction.

As a concrete example, take the tensor addition kernel introduced earlier. The outer pipeline works at tile granularity, with three stages: loading tiles from HBM into LX, performing the computation, and writing results back to HBM. The compute stage internally iterates over smaller blocks via the `ii` and `jj` loops, and a nested pipeline can be embedded within it to sequence the operations on each micro-block.


```
kernel tensor_add(A, B, C)

for tile_i
for tile_j

  token tile_loaded, tile_computed, tile_stored
  pipeline {

    stage depend(out: tile_loaded) {                              // Stage 1: Load tile from HBM into LX
      move A[tile_i, tile_j] from HBM to LX
      move B[tile_i, tile_j] from HBM to LX
    }

    stage depend(in: tile_loaded) depend(out: tile_computed) {                              // Stage 2: Compute tile

      for ii
        for jj

          token inputs_streamed, add_done, result_written
          pipeline {

            stage depend(out: inputs_streamed) {                      // Inner Stage 1: Stream inputs to SFP
              push A[ii, jj] from LX to SFP input
              push B[ii, jj] from LX to SFP input
            }

            stage depend(in: inputs_streamed) depend(out: add_done) {                      // Inner Stage 2: Compute
              execute add at SFP
            }

            stage depend(in: add_done) depend(out: result_written) {                      // Inner Stage 3: Write result back to LX
              move C[ii, jj] from SFP to LX
            }

          }
    }

    stage depend(in: tile_computed) depend(out: tile_stored) {                              // Stage 3: Write tile back to HBM
      move C[tile_i, tile_j] from LX to HBM
    }

  }
```
In such a hierarchical design, the outer pipeline would control the movement of tiles across the memory hierarchy, while the inner pipeline would manage the streaming of data through the compute unit. 




Hierarchical pipelines also help expose data reuse boundaries more clearly. Data that is reused across several micro-operations can be loaded once at a higher pipeline level and reused by inner pipelines. This aligns well with accelerator hardware designs, where data often moves through multiple memory levels before reaching compute units.

Finally, hierarchical pipelines improve the expressiveness of the scheduling abstraction. Instead of forcing all scheduling decisions into a single pipeline, the abstraction allows the schedule to be organized into layers that reflect both the structure of the algorithm and the structure of the hardware. This results in clearer programs, more analyzable execution structures, and a more direct mapping between the IR and the underlying accelerator architecture.



## Benefits/Summary/FAQ's of Nested Pipeline

### Spatio-temporal scheduling: 
Dataflow accelerators require scheduling decisions along two dimensions: space, i.e., the distribution of work across distributed, programmable hardware engines, and time, i.e., the orchestration of operations in the correct order across iterations. The pipeline abstraction captures the spatial dimension naturally: each stage eventually corresponds to a distinct hardware unit, and the pipeline structure expresses how work is partitioned across them. The temporal dimension is captured by loops — both the outer loops that drive iterations through the pipeline and the inner loops that sequence operations within a stage. Together, pipelines and loops provide a unified handle on both dimensions of the scheduling problem. 

### Synchronization:
Synchronization between stages is expressed through an elegant token-based construct: each stage declares depend(in:) and depend(out:) clauses that establish producer–consumer relationships, either within the same pipeline iteration or across iterations. Each dependency clause is annotated with an affine set that precisely specifies the set of iterations at which the synchronization is active. This makes it possible to express iteration-specific synchronization patterns — such as the spatial double buffering, the boundary conditions of double buffering where a stage must wait on a predecessor two iterations back, or asymmetric buffering scenarios where different stages have different reuse distances — all using the construct. 


### Token dependencies versus SSA def-use chains:
A natural alternative for expressing synchronization would be to rely on SSA-style def-use chains or named values associated with individual memory transfers. However, this approach has two fundamental limitations. First, def-use chains tie  synchronization to individual values or operations — a single load or store — whereas a token can span an entire stage, which may encompass multiple memory transfers, compute operations, or nested pipelines. This coarser, stage-level granularity matches the hardware reality: what matters for scheduling is not when a single transfer completes, but when a logical unit of work — a full tile load, a complete computation — is done. Second, SSA def-use chains are inherently intra-procedural and iteration-local; they have no natural mechanism to express that a stage in iteration i+1 depends on the completion of a stage in iteration i or i−2. Token dependencies annotated with affine sets directly capture these inter-iteration relationships, encoding both the distance between dependent iterations and the boundary conditions under which the dependency is active. This makes the token abstraction strictly more expressive than def-use chains for the class of cross-iteration, multi-stage synchronization patterns that arise in pipelined accelerator execution.


### Analyzability and performance estimation:

Because the pipeline structure, stage boundaries, synchronization dependencies, and buffer allocation strategies are all explicitly encoded in the IR, the abstraction is directly amenable to static analysis. A compiler or performance model can traverse  the pipeline graph, inspect token dependency distances, and reason about which stages are on the critical path of execution. Buffer liveness and reuse distances are readable directly from the affine sets on depend(in:) and depend(out:) clauses, making it straightforward to identify whether a    
given buffering strategy will stall the pipeline or allow full overlap. This enables early-stage bottleneck identification — for example, determining that a single-buffered store stage is throttling an otherwise double-buffered load, or that compute throughput is masked by memory latency given the current pipeline depth. Rather than requiring simulation or hardware profiling to understand execution behavior, the hierarchical pipeline IR provides a static, inspectable representation from which dataflow performance can be estimated and scheduling inefficiencies can be diagnosed directly.

