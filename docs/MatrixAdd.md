# Compilation of Matrix Add into Hierarchical Pipeline IR

We begin with a simple streaming pipeline representation of the matrix addition kernel and progressively transform it into a hierarchical, hardware-aware pipeline IR. Each compiler pass introduces one structural property while preserving the intent of the computation. The key idea is to avoid reconstructing structure via heavy analysis and instead incrementally refine explicit structure in the IR.

## Base IR: Logical Streaming Pipeline

The base IR expresses computation as a simple three-stage pipeline operating at 1x64 granularity. Each iteration streams a vector chunk of A and B from DRAM into the SFP input register, performs the computation, and writes the result back to DRAM.

At this level, the IR captures the logical dataflow, but it assumes unrealistic hardware capabilities—specifically, direct data movement between DRAM and SFP registers. It also exposes synchronization at the finest granularity, which is inefficient.

```mlir
C[M][N] = A[M][N] + B[M][N];

for m = M {
for n = N/64 {

    pipeline {

        stage depend(in: none) depend(out: t1) {
            Transfer A(m, n*64 : n*64+64) from DRAM to SFP input register
            Transfer B(m, n*64 : n*64+64) from DRAM to SFP input register
        }

        stage depend(in: t1) depend(out: t2) {
            SFP compute from SFP input registers and write into SFP output register
        }

        stage depend(in: t2) depend(out: none) {
            Transfer from SFP output register to C(m, n*64 : n*64+64) in DRAM
        }
    }
}}
```

## Pass 1: Path Expansion (Making Data Movement Legal)

The first transformation enforces hardware legality by expanding each end-to-end data movement into the required intermediate hops through the scratchpad (LX). Instead of allowing direct DRAM↔SFP transfers, the compiler inserts explicit stages for each hop.

Each hop becomes its own stage, making the memory hierarchy and movement explicit. This pass does not change the iteration structure or synchronization granularity—it simply rewrites the IR so that it matches the physical capabilities of the architecture.

```mlir
C[M][N] = A[M][N] + B[M][N];

for m = M {
for n = N/64 {

    pipeline {

        stage depend(in: none) depend(out: t1) {
            Transfer A(m, n*64 : n*64+64) from DRAM to LX
            Transfer B(m, n*64 : n*64+64) from DRAM to LX
        }

        stage depend(in: t1) depend(out: t2) {
            Transfer A(m, n*64 : n*64+64) from LX to SFP input register
            Transfer B(m, n*64 : n*64+64) from LX to SFP input register
        }

        stage depend(in: t2) depend(out: t3) {
            SFP compute from SFP input registers and write into SFP output register
        }

        stage depend(in: t3) depend(out: t4) {
            Transfer from SFP output register to C(m, n*64 : n*64+64) in LX
        }

        stage depend(in: t4) depend(out: none) {
            Transfer C(m, n*64 : n*64+64) from LX to DRAM
        }
    }
}}
```
At this point, the IR is hardware-correct but still too fine-grained in terms of synchronization.

## Pass 2: Tiling (Introducing Synchronization Structure)

The next pass introduces tiling. Unlike traditional compiler motivations, tiling here is used primarily to reduce synchronization overhead, not just to improve locality.

The compiler introduces outer tile loops m0 and n0, which define the synchronization scope, and inner loops m1 and n1, which correspond to the execution granularity. Importantly, the loops are introduced structurally around the pipeline, and no loops are sunk into stages yet.

This preserves a uniform structure where every stage still operates over the same inner iteration space.

```mlir
C[M][N] = A[M][N] + B[M][N];

for m0 = M/Cm {
for n0 = N/Cn {
for m1 = Cm {
for n1 = Cn/64 {

    pipeline {

        stage depend(in: none) depend(out: t1) {
            Transfer A(m0+m1, n0+n1*64 : n0+n1*64+64) from DRAM to LX
            Transfer B(m0+m1, n0+n1*64 : n0+n1*64+64) from DRAM to LX
        }

        stage depend(in: t1) depend(out: t2) {
            Transfer A(m0+m1, n0+n1*64 : n0+n1*64+64) from LX to SFP input register
            Transfer B(m0+m1, n0+n1*64 : n0+n1*64+64) from LX to SFP input register
        }

        stage depend(in: t2) depend(out: t3) {
            SFP compute from SFP input registers and write into SFP output register
        }

        stage depend(in: t3) depend(out: t4) {
            Transfer from SFP output register to C(m0+m1, n0+n1*64 : n0+n1*64+64) in LX
        }

        stage depend(in: t4) depend(out: none) {
            Transfer C(m0+m1, n0+n1*64 : n0+n1*64+64) from LX to DRAM
        }
    }

}}}}
```

At this stage, the IR exposes both tile-level structure and stick-level execution, but synchronization is still unnecessarily frequent because all five stages remain at the outer level.

## Pass 3: Loop Sinking, Stage Coarsening, and Nested Pipeline Formation

This is the key transformation that introduces hierarchy into the pipeline.

The compiler observes that the outer load and store stages naturally operate at tile granularity and benefit from reduced synchronization. Therefore, it sinks the inner loops m1 and n1 into these stages, effectively coarsening them.

At the same time, the three middle stages (LX→SFP, compute, SFP→LX) are grouped into a single larger outer stage. Inside this stage, the compiler introduces a nested pipeline that operates at 1x64 granularity.

This produces a clean hierarchical structure:
•	the outer pipeline operates at tile granularity
•	the inner pipeline operates at hardware execution granularity

```mlir
C[M][N] = A[M][N] + B[M][N];

for m0 = M/Cm {
for n0 = N/Cn {

    pipeline {

        stage depend(in: none) depend(out: t1) {
            for m1 = Cm {
            for n1 = Cn/64 {
                Transfer A(m0+m1, n0+n1*64 : n0+n1*64+64) from DRAM to LX
                Transfer B(m0+m1, n0+n1*64 : n0+n1*64+64) from DRAM to LX
            }}
        }

        stage depend(in: t1) depend(out: t2) {
            for m1 = Cm {
            for n1 = Cn/64 {

                pipeline {

                    stage depend(in: none) depend(out: t11) {
                        Transfer A from LX with indices (m1, n1*64) to SFP input register
                        Transfer B from LX with indices (m1, n1*64) to SFP input register
                    }

                    stage depend(in: t11) depend(out: t12) {
                        SFP compute from SFP input registers and write into SFP output register
                    }

                    stage depend(in: t12) depend(out: t13) {
                        Transfer from SFP output register to C in LX with indices (m1, n1*64)
                    }
                }

            }}
        }

        stage depend(in: t2) depend(out: none) {
            for m1 = Cm {
            for n1 = Cn/64 {
                Transfer C(m0+m1, n0+n1*64 : n0+n1*64+64) from LX to DRAM
            }}
        }
    }
}}
```
This IR is now well-structured for mapping to hardware.

### Note: Dependence Analysis for Loop Fusion and Stage Grouping

The loop sinking pass as described works cleanly here because each stage has a single, independent loop nest. In the general case, however, pulling loops from multiple stages into a shared outer loop nest—and grouping those stages into a single outer stage containing a nested pipeline—requires deeper dependence analysis.

Specifically, the compiler must verify that the iterations of the fused loop body are free of cross-iteration dependences that would prevent the grouping. If stage S1 produces a value at iteration (m1, n1) that stage S2 consumes at a different iteration (m1', n1'), the two cannot be merged into a single nested pipeline without introducing synchronization inside the inner loop body. Only when dependences are strictly within an iteration (i.e., S2 at (m1, n1) depends only on S1 at (m1, n1)) can the grouping proceed safely and yield the clean nested pipeline structure shown above.

## Pass 4: Reuse-Aware Hoisting (No Change for This Kernel)

The final pass looks for opportunities to hoist invariant data movement out of inner loops. This is useful for kernels where operands are reused across iterations (e.g., broadcast patterns).

However, in plain matrix addition, both A and B vary across (m1, n1), so there is no reuse to exploit. As a result, this pass does not modify the IR.


## Complete IR's after each of the above passes.

```mlir

// Original IR generated from KTIR.  
func.func @matrix_add_kernel(
      %A    : memref<?x?xf16, "DRAM">,
      %B    : memref<?x?xf16, "DRAM">,
      %C    : memref<?x?xf16, "DRAM">,
      %M    : index,
      %N    : index
  ) {

    scf.for %m = %c0 to %M step %c1 {
      scf.for %n = %c0 to %N step %c64 {

        %sfp_A   = memref.alloc() : memref<64xf16, "SFP_REG">
        %sfp_B   = memref.alloc() : memref<64xf16, "SFP_REG">
        %sfp_out = memref.alloc() : memref<64xf16, "SFP_REG">

        %t1 = ktdf.create_token : !ktdf.token
        %t2 = ktdf.create_token : !ktdf.token
        %t3 = ktdf.create_token : !ktdf.token

        ktdf.pipeline depth(1) {

          ktdf.stage depend_in() depend_out(%t1) {
            ktdf.data_transfer from %A[%m, %n] size [1, 64] to %sfp_A[%c0] size [64] : memref<?x?xf16, "DRAM">, memref<64xf16, "SFP_REG">
            ktdf.data_transfer from %B[%m, %n] size [1, 64] to %sfp_B[%c0] size [64] : memref<?x?xf16, "DRAM">, memref<64xf16, "SFP_REG">
            ktdf.stage_yield
          }

          ktdf.stage depend_in(%t1) depend_out(%t2) {
            linalg.add ins(%sfp_A, %sfp_B : memref<64xf16, "SFP_REG">, memref<64xf16, "SFP_REG">) outs(%sfp_out : memref<64xf16, "SFP_REG">)
            ktdf.stage_yield
          }

          ktdf.stage depend_in(%t2) depend_out(%t3) {
            ktdf.data_transfer from %sfp_out[%c0] size [64] to %C[%m, %n] size [1, 64] : memref<64xf16, "SFP_REG">, memref<?x?xf16, "DRAM">
            ktdf.stage_yield
          }

        } // ktdf.pipeline

        memref.dealloc %sfp_A   : memref<64xf16, "SFP_REG">
        memref.dealloc %sfp_B   : memref<64xf16, "SFP_REG">
        memref.dealloc %sfp_out : memref<64xf16, "SFP_REG">

      } // scf.for n
    } // scf.for m

    return
  }

// IR after path expansion.
func.func @matrix_add_kernel(
      %A    : memref<?x?xf16, "DRAM">,
      %B    : memref<?x?xf16, "DRAM">,
      %C    : memref<?x?xf16, "DRAM">,
      %M    : index,
      %N    : index
  ) {

    scf.for %m = %c0 to %M step %c1 {
      scf.for %n = %c0 to %N step %c64 {

        %lx_A    = memref.alloc() : memref<64xf16, "LX">
        %lx_B    = memref.alloc() : memref<64xf16, "LX">
        %lx_C    = memref.alloc() : memref<64xf16, "LX">
        %sfp_A   = memref.alloc() : memref<64xf16, "SFP_REG">
        %sfp_B   = memref.alloc() : memref<64xf16, "SFP_REG">
        %sfp_out = memref.alloc() : memref<64xf16, "SFP_REG">

        %fifo_slot_a, %fifo_slot_b = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"lxlu-to-sfp", 64xf16>,
                                                            !ktdf.fifo.slot<"lxlu-to-sfp", 64xf16>
        %fifo_slot_c = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"sfp-to-lxsu", 64xf16>

        %t1 = ktdf.create_token : !ktdf.token
        %t2 = ktdf.create_token : !ktdf.token
        %t3 = ktdf.create_token : !ktdf.token
        %t4 = ktdf.create_token : !ktdf.token
        %t5 = ktdf.create_token : !ktdf.token

        ktdf.pipeline depth(1) {

          // Stage 0: DRAM → LX
          ktdf.stage depend_in() depend_out(%t1) {
            ktdf.data_transfer from %A[%m, %n] size [1, 64] to %lx_A[%c0] size [64] : memref<?x?xf16, "DRAM">, memref<64xf16, "LX">
            ktdf.data_transfer from %B[%m, %n] size [1, 64] to %lx_B[%c0] size [64] : memref<?x?xf16, "DRAM">, memref<64xf16, "LX">
            ktdf.stage_yield
          }

          // Stage 1: LX → FIFO
          ktdf.stage depend_in(%t1) depend_out(%t2) {
            ktdf.data_transfer from %lx_A[%c0] size [64] to %fifo_slot_a size [64] : memref<64xf16, "LX">, !ktdf.fifo.slot<"lxlu-to-sfp", 64xf16>
            ktdf.data_transfer from %lx_B[%c0] size [64] to %fifo_slot_b size [64] : memref<64xf16, "LX">, !ktdf.fifo.slot<"lxlu-to-sfp", 64xf16>
            ktdf.stage_yield
          }

          // Stage 2: FIFO → SFP_REG and SFP compute
          ktdf.stage depend_in(%t2) depend_out(%t3) {
            ktdf.data_transfer from %fifo_slot_a to %sfp_A[%c0] size [64] : !ktdf.fifo.slot<"lxlu-to-sfp", 64xf16>, memref<64xf16, "SFP_REG">
            ktdf.data_transfer from %fifo_slot_b to %sfp_B[%c0] size [64] : !ktdf.fifo.slot<"lxlu-to-sfp", 64xf16>, memref<64xf16, "SFP_REG">
            linalg.add ins(%sfp_A, %sfp_B : memref<64xf16, "SFP_REG">, memref<64xf16, "SFP_REG">) outs(%sfp_out : memref<64xf16, "SFP_REG">)
            ktdf.data_transfer from %sfp_out[%c0] to %fifo_slot_c size [64] : memref<64xf16, "SFP_REG">, !ktdf.fifo.slot<"sfp-to-lxsu", 64xf16>
            ktdf.stage_yield
          }

          // Stage 3: FIFO → LX
          ktdf.stage depend_in(%t3) depend_out(%t4) {
            ktdf.data_transfer from %fifo_slot_c size [64] to %lx_C[%c0] size [64] : !ktdf.fifo.slot<"sfp-to-lxsu", 64xf16>, memref<64xf16, "LX">
            ktdf.stage_yield
          }

          // Stage 4: LX → DRAM
          ktdf.stage depend_in(%t4) depend_out(%t5) {
            ktdf.data_transfer from %lx_C[%c0] size [64] to %C[%m, %n] size [1, 64] : memref<64xf16, "LX">, memref<?x?xf16, "DRAM">
            ktdf.stage_yield
          }

        } // ktdf.pipeline

        memref.dealloc %lx_A    : memref<64xf16, "LX">
        memref.dealloc %lx_B    : memref<64xf16, "LX">
        memref.dealloc %lx_C    : memref<64xf16, "LX">
        memref.dealloc %sfp_A   : memref<64xf16, "SFP_REG">
        memref.dealloc %sfp_B   : memref<64xf16, "SFP_REG">
        memref.dealloc %sfp_out : memref<64xf16, "SFP_REG">

      } // scf.for n
    } // scf.for m

    return
  }


// IR after loop tiling.
func.func @matrix_add_kernel(
      %A    : memref<?x?xf16, "DRAM">,
      %B    : memref<?x?xf16, "DRAM">,
      %C    : memref<?x?xf16, "DRAM">,
      %M    : index,
      %N    : index,
      %tm   : index,  // tile size along M
      %tn   : index   // tile size along N (multiple of 64)
  ) {

    // Outer tile loops — define synchronization scope
    scf.for %m0 = %c0 to %M step %tm {
      scf.for %n0 = %c0 to %N step %tn {

        // Inner loops — define execution granularity, wrap the full pipeline
        scf.for %m1 = %m0 to %tm step %c1 {
          scf.for %n1 = %n0 to %tn step %c64 {

            %lx_A    = memref.alloc() : memref<64xf16, "LX">
            %lx_B    = memref.alloc() : memref<64xf16, "LX">
            %lx_C    = memref.alloc() : memref<64xf16, "LX">
            %sfp_A   = memref.alloc() : memref<64xf16, "SFP_REG">
            %sfp_B   = memref.alloc() : memref<64xf16, "SFP_REG">
            %sfp_out = memref.alloc() : memref<64xf16, "SFP_REG">

            %t1 = ktdf.create_token : !ktdf.token
            %t2 = ktdf.create_token : !ktdf.token
            %t3 = ktdf.create_token : !ktdf.token
            %t4 = ktdf.create_token : !ktdf.token
            %t5 = ktdf.create_token : !ktdf.token

            %fifo_slot_a, %fifo_slot_b = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"lxlu-to-sfp", 64xf16>,
                                                                !ktdf.fifo.slot<"lxlu-to-sfp", 64xf16>
            %fifo_slot_c = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"sfp-to-lxsu", 64xf16>

            ktdf.pipeline depth(1) {

              // Stage 0: DRAM → LX
              ktdf.stage depend_in() depend_out(%t1) {
                ktdf.data_transfer from %A[%m1, %n1] size [1, 64] to %lx_A[%c0] size [64] : memref<?x?xf16, "DRAM">, memref<64xf16, "LX">
                ktdf.data_transfer from %B[%m1, %n1] size [1, 64] to %lx_B[%c0] size [64] : memref<?x?xf16, "DRAM">, memref<64xf16, "LX">
                ktdf.stage_yield
              }

              // Stage 1: LX → FIFO
              ktdf.stage depend_in(%t1) depend_out(%t2) {
                ktdf.data_transfer from %lx_A[%c0] size [64] to %fifo_slot_a size [64] : memref<64xf16, "LX">, !ktdf.fifo.slot<"lxlu-to-sfp", 64xf16>
                ktdf.data_transfer from %lx_B[%c0] size [64] to %fifo_slot_b size [64] : memref<64xf16, "LX">, !ktdf.fifo.slot<"lxlu-to-sfp", 64xf16>
                ktdf.stage_yield
              }

              // Stage 2: FIFO → SFP_REG and SFP compute
              ktdf.stage depend_in(%t2) depend_out(%t3) {
                ktdf.data_transfer from %fifo_slot_a to %sfp_A[%c0] size [64] : !ktdf.fifo.slot<"lxlu-to-sfp", 64xf16>, memref<64xf16, "SFP_REG">
                ktdf.data_transfer from %fifo_slot_b to %sfp_B[%c0] size [64] : !ktdf.fifo.slot<"lxlu-to-sfp", 64xf16>, memref<64xf16, "SFP_REG">
                linalg.add ins(%sfp_A, %sfp_B : memref<64xf16, "SFP_REG">, memref<64xf16, "SFP_REG">) outs(%sfp_out : memref<64xf16, "SFP_REG">)
                ktdf.data_transfer from %sfp_out[%c0] to %fifo_slot_c size [64] : memref<64xf16, "SFP_REG">, !ktdf.fifo.slot<"sfp-to-lxsu", 64xf16>
                ktdf.stage_yield
              }

              // Stage 3: FIFO → LX
              ktdf.stage depend_in(%t3) depend_out(%t4) {
                ktdf.data_transfer from %fifo_slot_c size [64] to %lx_C[%c0] size [64] : !ktdf.fifo.slot<"sfp-to-lxsu", 64xf16>, memref<64xf16, "LX">
                ktdf.stage_yield
              }

              // Stage 4: LX → DRAM
              ktdf.stage depend_in(%t4) depend_out(%t5) {
                ktdf.data_transfer from %lx_C[%c0] size [64] to %C[%m1, %n1] size [1, 64] : memref<64xf16, "LX">, memref<?x?xf16, "DRAM">
                ktdf.stage_yield
              }

            } // ktdf.pipeline

            memref.dealloc %lx_A    : memref<64xf16, "LX">
            memref.dealloc %lx_B    : memref<64xf16, "LX">
            memref.dealloc %lx_C    : memref<64xf16, "LX">
            memref.dealloc %sfp_A   : memref<64xf16, "SFP_REG">
            memref.dealloc %sfp_B   : memref<64xf16, "SFP_REG">
            memref.dealloc %sfp_out : memref<64xf16, "SFP_REG">

          } // scf.for n1
        } // scf.for m1

      } // scf.for n0
    } // scf.for m0

    return
  }

// IR after loop sinking and coarsening the stages.
func.func @matrix_add_kernel(
      %A    : memref<?x?xf16, "DRAM">,
      %B    : memref<?x?xf16, "DRAM">,
      %C    : memref<?x?xf16, "DRAM">,
      %M    : index,
      %N    : index,
      %Cm   : index,  // tile size along M
      %Cn   : index   // tile size along N (multiple of 64)
  ) {

    // Outer tile loops — synchronization scope
    scf.for %m0 = %c0 to %M step %Cm {
      scf.for %n0 = %c0 to %N step %Cn {

        %lx_A    = memref.alloc() : memref<?x?xf16, "LX">(%Cm, %Cn)
        %lx_B    = memref.alloc() : memref<?x?xf16, "LX">(%Cm, %Cn)
        %lx_C    = memref.alloc() : memref<?x?xf16, "LX">(%Cm, %Cn)

        %t1 = ktdf.create_token : !ktdf.token
        %t2 = ktdf.create_token : !ktdf.token
        %t3 = ktdf.create_token : !ktdf.token

        ktdf.pipeline depth(1) {

          // Outer Stage 0: DRAM → LX (coarsened, loops sunk in)
          ktdf.stage depend_in() depend_out(%t1) {
            scf.for %m1 = %c0 to %Cm step %c1 {
              scf.for %n1 = %c0 to %Cn step %c64 {
                ktdf.data_transfer from %A[%m0+%m1, %n0+%n1] size [1, 64] to %lx_A[%m1, %n1] size [1, 64] : memref<?x?xf16, "DRAM">, memref<?x?xf16, "LX">
                ktdf.data_transfer from %B[%m0+%m1, %n0+%n1] size [1, 64] to %lx_B[%m1, %n1] size [1, 64] : memref<?x?xf16, "DRAM">, memref<?x?xf16, "LX">
              } // scf.for n1
            } // scf.for m1
            ktdf.stage_yield
          }

          // Outer Stage 1: inner pipeline over LX↔FIFO↔SFP + compute
          ktdf.stage depend_in(%t1) depend_out(%t2) {
            scf.for %m1 = %c0 to %Cm step %c1 {
              scf.for %n1 = %c0 to %Cn step %c64 {

                %sfp_A   = memref.alloc() : memref<64xf16, "SFP_REG">
                %sfp_B   = memref.alloc() : memref<64xf16, "SFP_REG">
                %sfp_out = memref.alloc() : memref<64xf16, "SFP_REG">
                %fifo_slot_a, %fifo_slot_b = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"lxlu-to-sfp", 64xf16>,
                                                                    !ktdf.fifo.slot<"lxlu-to-sfp", 64xf16>
                %fifo_slot_c = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"sfp-to-lxsu", 64xf16>

                %t11 = ktdf.create_token : !ktdf.token
                %t12 = ktdf.create_token : !ktdf.token
                %t13 = ktdf.create_token : !ktdf.token

                ktdf.pipeline depth(1) {

                  // Inner Stage 0: LX → FIFO
                  ktdf.stage depend_in() depend_out(%t11) {
                    ktdf.data_transfer from %lx_A[%m1, %n1] size [1, 64] to %fifo_slot_a size [64] : memref<?x?xf16, "LX">, !ktdf.fifo.slot<"lxlu-to-sfp", 64xf16>
                    ktdf.data_transfer from %lx_B[%m1, %n1] size [1, 64] to %fifo_slot_b size [64] : memref<?x?xf16, "LX">, !ktdf.fifo.slot<"lxlu-to-sfp", 64xf16>
                    ktdf.stage_yield
                  }

                  // Inner Stage 1: FIFO → SFP_REG and SFP compute
                  ktdf.stage depend_in(%t11) depend_out(%t12) {
                    ktdf.data_transfer from %fifo_slot_a to %sfp_A[%c0] size [64] : !ktdf.fifo.slot<"lxlu-to-sfp", 64xf16>, memref<64xf16, "SFP_REG">
                    ktdf.data_transfer from %fifo_slot_b to %sfp_B[%c0] size [64] : !ktdf.fifo.slot<"lxlu-to-sfp", 64xf16>, memref<64xf16, "SFP_REG">
                    linalg.add ins(%sfp_A, %sfp_B : memref<64xf16, "SFP_REG">, memref<64xf16, "SFP_REG">) outs(%sfp_out : memref<64xf16, "SFP_REG">)
                    ktdf.data_transfer from %sfp_out[%c0] to %fifo_slot_c size [64] : memref<64xf16, "SFP_REG">, !ktdf.fifo.slot<"sfp-to-lxsu", 64xf16>
                    ktdf.stage_yield
                  }

                  // Inner Stage 2: FIFO → LX
                  ktdf.stage depend_in(%t12) depend_out(%t13) {
                    ktdf.data_transfer from %fifo_slot_c size [64] to %lx_C[%m1, %n1] size [1, 64] : !ktdf.fifo.slot<"sfp-to-lxsu", 64xf16>, memref<?x?xf16, "LX">
                    ktdf.stage_yield
                  }

                } // ktdf.pipeline (inner)

                memref.dealloc %sfp_A   : memref<64xf16, "SFP_REG">
                memref.dealloc %sfp_B   : memref<64xf16, "SFP_REG">
                memref.dealloc %sfp_out : memref<64xf16, "SFP_REG">

              } // scf.for n1
            } // scf.for m1
            ktdf.stage_yield
          }

          // Outer Stage 2: LX → DRAM (coarsened, loops sunk in)
          ktdf.stage depend_in(%t2) depend_out(%t3) {
            scf.for %m1 = %c0 to %Cm step %c1 {
              scf.for %n1 = %c0 to %Cn step %c64 {
                ktdf.data_transfer from %lx_C[%m1, %n1] size [1, 64] to %C[%m0+%m1, %n0+%n1] size [1, 64] : memref<?x?xf16, "LX">, memref<?x?xf16, "DRAM">
              } // scf.for n1
            } // scf.for m1
            ktdf.stage_yield
          }

        } // ktdf.pipeline (outer)

        memref.dealloc %lx_A : memref<?x?xf16, "LX">
        memref.dealloc %lx_B : memref<?x?xf16, "LX">
        memref.dealloc %lx_C : memref<?x?xf16, "LX">

      } // scf.for n0
    } // scf.for m0

    return
  }

```
## Broadcast Version Kernel

### Base IR: 

C[M][N] = A[M] + B[M][N]

At the beginning, the IR is a flat streaming pipeline. Both A and B are treated as streamed operands, so A(m) is redundantly moved for every n chunk.

```mlir 
C[M][N] = A[M] + B[M][N];

for m = M {
for n = N/64 {

    pipeline {

        stage depend(in: none) depend(out: t1) {
            Transfer A(m) from DRAM to SFP input register
            Transfer B(m, n*64 : n*64+64) from DRAM to SFP input register
        }

        stage depend(in: t1) depend(out: t2) {
            SFP compute from SFP input registers and write into SFP output register
        }

        stage depend(in: t2) depend(out: none) {
            Transfer from SFP output register to C(m, n*64 : n*64+64) in DRAM
        }
    }
}}
```

### Pass 1: Path Expansion

This pass makes memory movement legal by expanding direct DRAM↔SFP transfers through LX. Each intermediate hop is represented explicitly as its own stage.

```mlir 
C[M][N] = A[M] + B[M][N];

for m = M {
for n = N/64 {

    pipeline {

        stage depend(in: none) depend(out: t1) {
            Transfer A(m) from DRAM to LX
            Transfer B(m, n*64 : n*64+64) from DRAM to LX
        }

        stage depend(in: t1) depend(out: t2) {
            Transfer A(m) from LX to SFP input register
            Transfer B(m, n*64 : n*64+64) from LX to SFP input register
        }

        stage depend(in: t2) depend(out: t3) {
            SFP compute from SFP input registers and write into SFP output register
        }

        stage depend(in: t3) depend(out: t4) {
            Transfer from SFP output register to C(m, n*64 : n*64+64) in LX
        }

        stage depend(in: t4) depend(out: none) {
            Transfer C(m, n*64 : n*64+64) from LX to DRAM
        }
    }
}}
```

At this point the IR is hardware-legal, but still too fine-grained and still misses reuse.

### Pass 2: Tiling

This pass introduces tiled iteration spaces. The point here is to create the scopes at which synchronization and reuse can later be managed. We introduce outer tile loops m0 and n0, and inner loops m1 and n1.

```mlir 
C[M][N] = A[M] + B[M][N];

for m0 = M/Cm {
for n0 = N/Cn {
for m1 = Cm {
for n1 = Cn/64 {

    pipeline {

        stage depend(in: none) depend(out: t1) {
            Transfer A(m0+m1) from DRAM to LX
            Transfer B(m0+m1, n0+n1*64 : n0+n1*64+64) from DRAM to LX
        }

        stage depend(in: t1) depend(out: t2) {
            Transfer A(m0+m1) from LX to SFP input register
            Transfer B(m0+m1, n0+n1*64 : n0+n1*64+64) from LX to SFP input register
        }

        stage depend(in: t2) depend(out: t3) {
            SFP compute from SFP input registers and write into SFP output register
        }

        stage depend(in: t3) depend(out: t4) {
            Transfer from SFP output register to C(m0+m1, n0+n1*64 : n0+n1*64+64) in LX
        }

        stage depend(in: t4) depend(out: none) {
            Transfer C(m0+m1, n0+n1*64 : n0+n1*64+64) from LX to DRAM
        }
    }

}}}}
```
This IR still redundantly moves A, but now the right reuse scopes are visible.

### After Pass 3: Loop Sinking + Stage Coarsening + Nested Pipeline Formation

This pass gives the IR its hierarchical structure before any reuse promotion is applied.

The outer DRAM→LX and LX→DRAM work becomes coarser by sinking m1,n1 loops into those stages. The middle three fine-grain stages are grouped into one larger stage, and inside that stage the compiler introduces a nested 1x64 pipeline.

```mlir

C[M][N] = A[M] + B[M][N];

for m0 = M/Cm {
for n0 = N/Cn {

    pipeline {

        stage depend(in: none) depend(out: t1) {
            for m1 = Cm {
            for n1 = Cn/64 {
                Transfer A(m0+m1) from DRAM to LX
                Transfer B(m0+m1, n0+n1*64 : n0+n1*64+64) from DRAM to LX
            }}
        }

        stage depend(in: t1) depend(out: t2) {
            for m1 = Cm {
            for n1 = Cn/64 {

                pipeline {

                    stage depend(in: none) depend(out: t11) {
                        Transfer A(m0+m1) from LX to SFP input register
                        Transfer B(m0+m1, n0+n1*64 : n0+n1*64+64) from LX to SFP input register
                    }

                    stage depend(in: t11) depend(out: t12) {
                        SFP compute from SFP input registers and write into SFP output register
                    }

                    stage depend(in: t12) depend(out: t13) {
                        Transfer from SFP output register to C in LX with indices (m1, n1*64)
                    }
                }

            }}
        }

        stage depend(in: t2) depend(out: none) {
            for m1 = Cm {
            for n1 = Cn/64 {
                Transfer C(m0+m1, n0+n1*64 : n0+n1*64+64) from LX to DRAM
            }}
        }
    }
}}
 
```

This is the right structural input to reuse promotion.

### After Pass 4: Reuse Promotion via Program Slicing Technique

Now reuse is promoted on top of the hierarchical structure.

At this point, the compiler can see two clear reuse scopes for A:
•	A(m0+m1) is reused across all n1 inside a fixed m1
•	A(m0:m0+Cm) is reused across all n0 for a fixed m0

A clean representation is to materialize hoisted transfers as sibling single-stage pipelines.

Reuse promotion is a transformation that eliminates redundant data movement by exploiting invariance in access patterns and restructuring the IR accordingly. In dataflow-style pipeline IRs, this often requires not only reordering operations but also extracting portions of a stage, relocating them, and reconnecting consumers. A natural and precise way to reason about this transformation is through program slicing.

At a high level, reuse promotion proceeds in three steps:
(1) identify a transfer that is redundantly executed across loop iterations,
(2) slice out the subprogram responsible for producing that value, and
(3) lift that slice to a dominating scope where the value can be reused safely.
The transformed slice is materialized as a sibling pipeline, while the original consumers are rewritten to use the promoted value.

```mlir 
C[M][N] = A[M] + B[M][N];

for m0 = M/Cm {

    pipeline {
        stage depend(in: none) depend(out: none) {
            for m1 = Cm {
                Transfer A(m0+m1) from DRAM to LX
            }
        }
    }

    for n0 = N/Cn {

        pipeline {

            stage depend(in: none) depend(out: t1) {
                for m1 = Cm {
                for n1 = Cn/64 {
                    Transfer B(m0+m1, n0+n1*64 : n0+n1*64+64) from DRAM to LX
                }}
            }

            stage depend(in: t1) depend(out: t2) {
                for m1 = Cm {

                    pipeline {
                        stage depend(in: none) depend(out: none) {
                            Transfer A(m0+m1) from LX to SFP register
                        }
                    }

                    for n1 = Cn/64 {

                        pipeline {

                            stage depend(in: none) depend(out: t11) {
                                Transfer B(m0+m1, n0+n1*64 : n0+n1*64+64) from LX to SFP input register
                            }

                            stage depend(in: t11) depend(out: t12) {
                                SFP compute using A from SFP register and B from SFP input register
                                Write result into SFP output register
                            }

                            stage depend(in: t12) depend(out: t13) {
                                Transfer from SFP output register to C in LX with indices (m1, n1*64)
                            }
                        }
                    }
                }
            }

            stage depend(in: t2) depend(out: none) {
                for m1 = Cm {
                for n1 = Cn/64 {
                    Transfer C(m0+m1, n0+n1*64 : n0+n1*64+64) from LX to DRAM
                }}
            }
        }
    }
}
```

This final IR captures the intended structure cleanly:
•	hoist A to LX once per m0
•	hoist A to SFP register once per m1
•	stream only B per n1
•	outer pipeline stays tile-granular
•	inner pipeline stays 1x64 granular

This is the right final ordering and result.

Reuse promotion is a transformation that eliminates redundant data movement by exploiting invariance in access patterns and restructuring the program accordingly. In a pipeline-based IR, this transformation is more involved than a simple reordering of statements, because transfers are embedded inside stages and tied to synchronization through tokens. As a result, promoting reuse requires extracting a portion of a stage, relocating it to a higher scope, and reconnecting the remaining computation. A natural way to formalize this process is through program slicing, where we isolate the minimal subprogram responsible for producing a value and then reposition it safely.

Consider the broadcast-add kernel C[M][N] = A[M] + B[M][N]. After loop sinking and nested pipeline formation, the IR contains stages that repeatedly transfer the same value A(m0+m1) across inner loop iterations. In particular, the transfer from DRAM to LX and the transfer from LX to the SFP FIFO both occur inside loops over n1, even though the value of A does not depend on n1. This redundancy is visible directly from the subscript expression A[m0+m1], which depends only on m0 and m1 and is invariant with respect to both n0 and n1. This invariance is the key observation that enables reuse promotion.

The first step in the transformation is to identify the slice corresponding to the production of A. In the original stage, the transfer of A is interleaved with the transfer of B. However, these are independent operations. Program slicing allows us to isolate just the part of the stage responsible for A, namely the transfer operation itself along with its source. Since there are no intermediate computations involved, the slice is straightforward and consists solely of the transfer instruction. Conceptually, we are carving out a minimal producer for A from within a larger mixed stage.

Once the slice is identified, the next step is to determine where it can be safely moved. This is guided by subscript-based invariance analysis. Because A(m0+m1) does not depend on n0 or n1, the DRAM-to-LX transfer can be lifted outside both loops. The closest dominating scope where this invariance still holds is the m0 loop. Moving the slice to this scope ensures that the value of A is computed once per m0 iteration and reused across all n0 tiles. After lifting, the IR introduces a sibling pipeline at the m0 level containing a single stage that performs the DRAM-to-LX transfer for A. The original stage is correspondingly simplified to only transfer B.

A similar transformation is applied to the LX-to-SFP transfer of A. Here, the invariance is with respect to n1, but not m1, so the correct promotion scope is the m1 loop. The compiler again slices out the transfer of A from the inner stage and lifts it to a new sibling pipeline at the m1 scope. This pipeline performs the transfer once per m1 iteration, and the inner compute pipeline is rewritten to consume A from a persistent SFP register instead of repeatedly reading it from the FIFO. This step not only removes redundant transfers but also changes the dataflow structure: A is no longer part of the streaming path but instead becomes a locally reused value.

After both promotions, the structure of the IR reflects the reuse hierarchy of the data. The outermost sibling pipeline at the m0 level ensures that A is brought into LX once per tile of rows. Within each tile, another sibling pipeline at the m1 level ensures that A is loaded into the SFP register once per row. The inner pipelines then operate only on B as a streamed operand, while reusing A across multiple iterations. This hierarchical restructuring aligns the program with the natural reuse pattern of the broadcast operation.

From a legality perspective, the transformation relies on ensuring that no intervening writes invalidate the promoted value and that the storage lifetime is sufficient to hold the value across the expanded scope. In this example, the absence of writes to A and the regular affine structure make the transformation straightforward. More generally, these conditions can be checked using dependence analysis, but for many structured kernels, subscript invariance alone provides a sufficient and efficient criterion.

Conceptually, reuse promotion in this framework can be viewed as a slice-and-lift transformation. A producer slice is extracted from within a stage, lifted to a higher loop scope where it becomes invariant, and re-materialized as a sibling pipeline. The consumers are then rewired to use the promoted value. This formulation integrates naturally with the pipeline abstraction, preserving explicit scheduling and synchronization while enabling aggressive elimination of redundant data movement.


```mlir 
  IR 1: Base IR
                                                                                                                                                                                
  func.func @broadcast_add_kernel(
      %A    : memref<?xf16, "DRAM">,
      %B    : memref<?x?xf16, "DRAM">,
      %C    : memref<?x?xf16, "DRAM">,
      %M    : index,
      %N    : index
  ) {

    scf.for %m = %c0 to %M step %c64 {
      scf.for %n = %c0 to %N step %c1 {

        %sfp_A   = memref.alloc() : memref<64xf16, "SFP_REG">
        %sfp_B   = memref.alloc() : memref<64xf16, "SFP_REG">
        %sfp_out = memref.alloc() : memref<64xf16, "SFP_REG">

        %t1 = ktdf.create_token : !ktdf.token
        %t2 = ktdf.create_token : !ktdf.token
        %t3 = ktdf.create_token : !ktdf.token

        ktdf.pipeline depth(1) {

          ktdf.stage depend_in() depend_out(%t1) {
            ktdf.data_transfer from %A[%m] size [64] to %sfp_A[%c0] size [64] : memref<?xf16, "DRAM">, memref<64xf16, "SFP_REG">
            ktdf.data_transfer from %B[%m, %n] size [64, 1] to %sfp_B[%c0] size [64] : memref<?x?xf16, "DRAM">, memref<64xf16, "SFP_REG">
            ktdf.stage_yield
          }

          ktdf.stage depend_in(%t1) depend_out(%t2) {
            linalg.add ins(%sfp_A, %sfp_B : memref<64xf16, "SFP_REG">, memref<64xf16, "SFP_REG">) outs(%sfp_out : memref<64xf16, "SFP_REG">)
            ktdf.stage_yield
          }

          ktdf.stage depend_in(%t2) depend_out(%t3) {
            ktdf.data_transfer from %sfp_out[%c0] size [64] to %C[%m, %n] size [64, 1] : memref<64xf16, "SFP_REG">, memref<?x?xf16, "DRAM">
            ktdf.stage_yield
          }

        } // ktdf.pipeline

        memref.dealloc %sfp_A   : memref<64xf16, "SFP_REG">
        memref.dealloc %sfp_B   : memref<64xf16, "SFP_REG">
        memref.dealloc %sfp_out : memref<64xf16, "SFP_REG">

      } // scf.for n                                                                                                                                                            
    } // scf.for m
                                                                                                                                                                                
    return        
  }

  ---
  IR 2: Path Expansion (DRAM↔SFP → DRAM↔LX↔SFP)
                                                                                                                                                                                
  func.func @broadcast_add_kernel(
      %A    : memref<?xf16, "DRAM">,                                                                                                                                            
      %B    : memref<?x?xf16, "DRAM">,                                                                                                                                          
      %C    : memref<?x?xf16, "DRAM">,                                                                                                                                          
      %M    : index,                                                                                                                                                            
      %N    : index
  ) {                                                                                                                                                                           
                  
    scf.for %m = %c0 to %M step %c64 {                                                                                                                                          
      scf.for %n = %c0 to %N step %c1 {
                                                                                                                                                                                
        %lx_A    = memref.alloc() : memref<64xf16, "LX">                                                                                                                        
        %lx_B    = memref.alloc() : memref<64xf16, "LX">                                                                                                                        
        %lx_C    = memref.alloc() : memref<64xf16, "LX">                                                                                                                        
        %sfp_A   = memref.alloc() : memref<64xf16, "SFP_REG">                                                                                                                   
        %sfp_B   = memref.alloc() : memref<64xf16, "SFP_REG">                                                                                                                   
        %sfp_out = memref.alloc() : memref<64xf16, "SFP_REG">                                                                                                                   
                                                                                                                                                                                
        %t1 = ktdf.create_token : !ktdf.token                                                                                                                                   
        %t2 = ktdf.create_token : !ktdf.token                                                                                                                                   
        %t3 = ktdf.create_token : !ktdf.token                                                                                                                                   
        %t4 = ktdf.create_token : !ktdf.token                                                                                                                                   
        %t5 = ktdf.create_token : !ktdf.token
                                                                                                                                                                                
        ktdf.pipeline depth(1) {
                                                                                                                                                                                
          // Stage 0: DRAM → LX                                                                                                                                                 
          ktdf.stage depend_in() depend_out(%t1) {
            ktdf.data_transfer from %A[%m] size [64] to %lx_A[%c0] size [64] : memref<?xf16, "DRAM">, memref<64xf16, "LX">
            ktdf.data_transfer from %B[%m, %n] size [64, 1] to %lx_B[%c0] size [64] : memref<?x?xf16, "DRAM">, memref<64xf16, "LX">
            ktdf.stage_yield                                                                                                                                                    
          }                                                                                                                                                                     
                                                                                                                                                                                
          // Stage 1: LX → FIFO
          ktdf.stage depend_in(%t1) depend_out(%t2) {
            ktdf.data_transfer from %lx_A[%c0] size [64] to %fifo_slot_a size [64] : memref<64xf16, "LX">, !ktdf.fifo.slot<"lxlu-to-sfp", 64xf16>
            ktdf.data_transfer from %lx_B[%c0] size [64] to %fifo_slot_b size [64] : memref<64xf16, "LX">, !ktdf.fifo.slot<"lxlu-to-sfp", 64xf16>
            ktdf.stage_yield
          }

          // Stage 2: FIFO → SFP_REG, compute, SFP_REG → FIFO
          ktdf.stage depend_in(%t2) depend_out(%t3) {
            ktdf.data_transfer from %fifo_slot_a to %sfp_A[%c0] size [64] : !ktdf.fifo.slot<"lxlu-to-sfp", 64xf16>, memref<64xf16, "SFP_REG">
            ktdf.data_transfer from %fifo_slot_b to %sfp_B[%c0] size [64] : !ktdf.fifo.slot<"lxlu-to-sfp", 64xf16>, memref<64xf16, "SFP_REG">
            linalg.add ins(%sfp_A, %sfp_B : memref<64xf16, "SFP_REG">, memref<64xf16, "SFP_REG">) outs(%sfp_out : memref<64xf16, "SFP_REG">)
            ktdf.data_transfer from %sfp_out[%c0] to %fifo_slot_c size [64] : memref<64xf16, "SFP_REG">, !ktdf.fifo.slot<"sfp-to-lxsu", 64xf16>
            ktdf.stage_yield
          }

          // Stage 3: FIFO → LX
          ktdf.stage depend_in(%t3) depend_out(%t4) {
            ktdf.data_transfer from %fifo_slot_c size [64] to %lx_C[%c0] size [64] : !ktdf.fifo.slot<"sfp-to-lxsu", 64xf16>, memref<64xf16, "LX">
            ktdf.stage_yield
          }

          // Stage 4: LX → DRAM
          ktdf.stage depend_in(%t4) depend_out(%t5) {
            ktdf.data_transfer from %lx_C[%c0] size [64] to %C[%m, %n] size [64, 1] : memref<64xf16, "LX">, memref<?x?xf16, "DRAM">
            ktdf.stage_yield
          }

        } // ktdf.pipeline

        memref.dealloc %lx_A    : memref<64xf16, "LX">
        memref.dealloc %lx_B    : memref<64xf16, "LX">
        memref.dealloc %lx_C    : memref<64xf16, "LX">
        memref.dealloc %sfp_A   : memref<64xf16, "SFP_REG">
        memref.dealloc %sfp_B   : memref<64xf16, "SFP_REG">
        memref.dealloc %sfp_out : memref<64xf16, "SFP_REG">
                                                                                                                                                                                
      } // scf.for n
    } // scf.for m                                                                                                                                                              
                  
    return
  }

  ---
  IR 3: Tiling (introduce outer m0/n0 loops)
                                                                                                                                                                                
  func.func @broadcast_add_kernel(
      %A    : memref<?xf16, "DRAM">,                                                                                                                                            
      %B    : memref<?x?xf16, "DRAM">,
      %C    : memref<?x?xf16, "DRAM">,                                                                                                                                          
      %M    : index,                                                                                                                                                            
      %N    : index,                                                                                                                                                            
      %Cm   : index,  // tile size along M (multiple of 64)                                                                                                                     
      %Cn   : index   // tile size along N                                                                                                                                      
  ) {                                                                                                                                                                           
                                                                                                                                                                                
    scf.for %m0 = %c0 to %M step %Cm {                                                                                                                                          
      scf.for %n0 = %c0 to %N step %Cn {
                                                                                                                                                                                
        scf.for %m1 = %c0 to %Cm step %c64 {                                                                                                                                    
          scf.for %n1 = %c0 to %Cn step %c1 {                                                                                                                                   
                                                                                                                                                                                
            %lx_A    = memref.alloc() : memref<64xf16, "LX">                                                                                                                    
            %lx_B    = memref.alloc() : memref<64xf16, "LX">                                                                                                                    
            %lx_C    = memref.alloc() : memref<64xf16, "LX">                                                                                                                    
            %sfp_A   = memref.alloc() : memref<64xf16, "SFP_REG">                                                                                                               
            %sfp_B   = memref.alloc() : memref<64xf16, "SFP_REG">                                                                                                               
            %sfp_out = memref.alloc() : memref<64xf16, "SFP_REG">                                                                                                               
                                                                                                                                                                                
            %t1 = ktdf.create_token : !ktdf.token                                                                                                                               
            %t2 = ktdf.create_token : !ktdf.token                                                                                                                               
            %t3 = ktdf.create_token : !ktdf.token                                                                                                                               
            %t4 = ktdf.create_token : !ktdf.token                                                                                                                               
            %t5 = ktdf.create_token : !ktdf.token                                                                                                                               
                                                                                                                                                                                
            ktdf.pipeline depth(1) {                                                                                                                                            
                  
              ktdf.stage depend_in() depend_out(%t1) {                                                                                                                          
                ktdf.data_transfer from %A[%m0+%m1] size [64] to %lx_A[%c0] size [64] : memref<?xf16, "DRAM">, memref<64xf16, "LX">
                ktdf.data_transfer from %B[%m0+%m1, %n0+%n1] size [64, 1] to %lx_B[%c0] size [64] : memref<?x?xf16, "DRAM">, memref<64xf16, "LX">
                ktdf.stage_yield                                                                                                                                                
              }                                                                                                                                                                 
                                                                                                                                                                                
              ktdf.stage depend_in(%t1) depend_out(%t2) {
                ktdf.data_transfer from %lx_A[%c0] size [64] to %fifo_slot_a size [64] : memref<64xf16, "LX">, !ktdf.fifo.slot<"lxlu-to-sfp", 64xf16>
                ktdf.data_transfer from %lx_B[%c0] size [64] to %fifo_slot_b size [64] : memref<64xf16, "LX">, !ktdf.fifo.slot<"lxlu-to-sfp", 64xf16>
                ktdf.stage_yield
              }

              ktdf.stage depend_in(%t2) depend_out(%t3) {
                ktdf.data_transfer from %fifo_slot_a to %sfp_A[%c0] size [64] : !ktdf.fifo.slot<"lxlu-to-sfp", 64xf16>, memref<64xf16, "SFP_REG">
                ktdf.data_transfer from %fifo_slot_b to %sfp_B[%c0] size [64] : !ktdf.fifo.slot<"lxlu-to-sfp", 64xf16>, memref<64xf16, "SFP_REG">
                linalg.add ins(%sfp_A, %sfp_B : memref<64xf16, "SFP_REG">, memref<64xf16, "SFP_REG">) outs(%sfp_out : memref<64xf16, "SFP_REG">)
                ktdf.data_transfer from %sfp_out[%c0] to %fifo_slot_c size [64] : memref<64xf16, "SFP_REG">, !ktdf.fifo.slot<"sfp-to-lxsu", 64xf16>
                ktdf.stage_yield
              }

              ktdf.stage depend_in(%t3) depend_out(%t4) {
                ktdf.data_transfer from %fifo_slot_c size [64] to %lx_C[%c0] size [64] : !ktdf.fifo.slot<"sfp-to-lxsu", 64xf16>, memref<64xf16, "LX">
                ktdf.stage_yield
              }
                                                                                                                                                                                
              ktdf.stage depend_in(%t4) depend_out(%t5) {                                                                                                                       
                ktdf.data_transfer from %lx_C[%c0] size [64] to %C[%m0+%m1, %n0+%n1] size [64, 1] : memref<64xf16, "LX">, memref<?x?xf16, "DRAM">
                ktdf.stage_yield                                                                                                                                                
              }   
                                                                                                                                                                                
            } // ktdf.pipeline                                                                                                                                                  
   
            memref.dealloc %lx_A    : memref<64xf16, "LX">                                                                                                                      
            memref.dealloc %lx_B    : memref<64xf16, "LX">
            memref.dealloc %lx_C    : memref<64xf16, "LX">                                                                                                                      
            memref.dealloc %sfp_A   : memref<64xf16, "SFP_REG">
            memref.dealloc %sfp_B   : memref<64xf16, "SFP_REG">                                                                                                                 
            memref.dealloc %sfp_out : memref<64xf16, "SFP_REG">                                                                                                                 
                                                                                                                                                                                
          } // scf.for n1                                                                                                                                                       
        } // scf.for m1                                                                                                                                                         
                                                                                                                                                                                
      } // scf.for n0
    } // scf.for m0                                                                                                                                                             
                  
    return
  }
                                                                                                                                                                                
  ---
  IR 4: Loop Sinking & Stage Coarsening (hierarchical pipeline)                                                                                                                 
                                                                                                                                                                                
  func.func @broadcast_add_kernel(
      %A    : memref<?xf16, "DRAM">,                                                                                                                                            
      %B    : memref<?x?xf16, "DRAM">,                                                                                                                                          
      %C    : memref<?x?xf16, "DRAM">,
      %M    : index,                                                                                                                                                            
      %N    : index,
      %Cm   : index,  // tile size along M (multiple of 64)                                                                                                                     
      %Cn   : index   // tile size along N
  ) {                                                                                                                                                                           
                  
    scf.for %m0 = %c0 to %M step %Cm {                                                                                                                                          
      scf.for %n0 = %c0 to %N step %Cn {
                                                                                                                                                                                
        %lx_A = memref.alloc() : memref<?xf16, "LX">(%Cm)                                                                                                                       
        %lx_B = memref.alloc() : memref<?x?xf16, "LX">(%Cm, %Cn)
        %lx_C = memref.alloc() : memref<?x?xf16, "LX">(%Cm, %Cn)                                                                                                                
                                                                                                                                                                                
        %t1 = ktdf.create_token : !ktdf.token                                                                                                                                   
        %t2 = ktdf.create_token : !ktdf.token                                                                                                                                   
        %t3 = ktdf.create_token : !ktdf.token                                                                                                                                   
   
        ktdf.pipeline depth(1) {                                                                                                                                                
                  
          // Outer Stage 0: DRAM → LX (loops sunk in)
          ktdf.stage depend_in() depend_out(%t1) {
            scf.for %m1 = %c0 to %Cm step %c64 {
              ktdf.data_transfer from %A[%m0+%m1] size [64] to %lx_A[%m1] size [64] : memref<?xf16, "DRAM">, memref<?xf16, "LX">
              scf.for %n1 = %c0 to %Cn step %c1 {
                ktdf.data_transfer from %B[%m0+%m1, %n0+%n1] size [64, 1] to %lx_B[%m1, %n1] size [64, 1] : memref<?x?xf16, "DRAM">, memref<?x?xf16, "LX">
              } // scf.for n1
            } // scf.for m1
            ktdf.stage_yield
          }

          // Outer Stage 1: nested pipeline over LX↔SFP + compute
          ktdf.stage depend_in(%t1) depend_out(%t2) {
            scf.for %m1 = %c0 to %Cm step %c64 {
              scf.for %n1 = %c0 to %Cn step %c1 {

                %sfp_A   = memref.alloc() : memref<64xf16, "SFP_REG">
                %sfp_B   = memref.alloc() : memref<64xf16, "SFP_REG">
                %sfp_out = memref.alloc() : memref<64xf16, "SFP_REG">

                %t11 = ktdf.create_token : !ktdf.token
                %t12 = ktdf.create_token : !ktdf.token
                %t13 = ktdf.create_token : !ktdf.token

                ktdf.pipeline depth(1) {

                  // Inner Stage 0: LX → FIFO
                  ktdf.stage depend_in() depend_out(%t11) {
                    ktdf.data_transfer from %lx_A[%m1] size [64] to %fifo_slot_a size [64] : memref<?xf16, "LX">, !ktdf.fifo.slot<"lxlu-to-sfp", 64xf16>
                    ktdf.data_transfer from %lx_B[%m1, %n1] size [64, 1] to %fifo_slot_b size [64] : memref<?x?xf16, "LX">, !ktdf.fifo.slot<"lxlu-to-sfp", 64xf16>
                    ktdf.stage_yield
                  }

                  // Inner Stage 1: FIFO → SFP_REG, compute, SFP_REG → FIFO
                  ktdf.stage depend_in(%t11) depend_out(%t12) {
                    ktdf.data_transfer from %fifo_slot_a to %sfp_A[%c0] size [64] : !ktdf.fifo.slot<"lxlu-to-sfp", 64xf16>, memref<64xf16, "SFP_REG">
                    ktdf.data_transfer from %fifo_slot_b to %sfp_B[%c0] size [64] : !ktdf.fifo.slot<"lxlu-to-sfp", 64xf16>, memref<64xf16, "SFP_REG">
                    linalg.add ins(%sfp_A, %sfp_B : memref<64xf16, "SFP_REG">, memref<64xf16, "SFP_REG">) outs(%sfp_out : memref<64xf16, "SFP_REG">)
                    ktdf.data_transfer from %sfp_out[%c0] to %fifo_slot_c size [64] : memref<64xf16, "SFP_REG">, !ktdf.fifo.slot<"sfp-to-lxsu", 64xf16>
                    ktdf.stage_yield
                  }

                  // Inner Stage 2: FIFO → LX
                  ktdf.stage depend_in(%t12) depend_out(%t13) {
                    ktdf.data_transfer from %fifo_slot_c size [64] to %lx_C[%m1, %n1] size [64, 1] : !ktdf.fifo.slot<"sfp-to-lxsu", 64xf16>, memref<?x?xf16, "LX">
                    ktdf.stage_yield
                  }

                } // ktdf.pipeline (inner)

                memref.dealloc %sfp_A   : memref<64xf16, "SFP_REG">
                memref.dealloc %sfp_B   : memref<64xf16, "SFP_REG">
                memref.dealloc %sfp_out : memref<64xf16, "SFP_REG">

              } // scf.for n1
            } // scf.for m1
            ktdf.stage_yield
          }

          // Outer Stage 2: LX → DRAM (loops sunk in)
          ktdf.stage depend_in(%t2) depend_out(%t3) {
            scf.for %m1 = %c0 to %Cm step %c64 {
              scf.for %n1 = %c0 to %Cn step %c1 {                                                                                                                               
                ktdf.data_transfer from %lx_C[%m1, %n1] size [64, 1] to %C[%m0+%m1, %n0+%n1] size [64, 1] : memref<?x?xf16, "LX">, memref<?x?xf16, "DRAM">
              } // scf.for n1                                                                                                                                                   
            } // scf.for m1                                                                                                                                                     
            ktdf.stage_yield                                                                                                                                                    
          }                                                                                                                                                                     
                  
        } // ktdf.pipeline (outer)                                                                                                                                              
   
        memref.dealloc %lx_A : memref<?xf16, "LX">                                                                                                                              
        memref.dealloc %lx_B : memref<?x?xf16, "LX">
        memref.dealloc %lx_C : memref<?x?xf16, "LX">                                                                                                                            
                                                                                                                                                                                
      } // scf.for n0                                                                                                                                                           
    } // scf.for m0                                                                                                                                                             
                                                                                                                                                                                
    return
  }                                                                                                                                                                             
                  
  ---
  IR 5: Reuse Promotion (hoist A via sibling pipelines)
                                                                                                                                                                                
  func.func @broadcast_add_kernel(
      %A    : memref<?xf16, "DRAM">,                                                                                                                                            
      %B    : memref<?x?xf16, "DRAM">,                                                                                                                                          
      %C    : memref<?x?xf16, "DRAM">,                                                                                                                                          
      %M    : index,                                                                                                                                                            
      %N    : index,
      %Cm   : index,  // tile size along M (multiple of 64)                                                                                                                     
      %Cn   : index   // tile size along N
  ) {                                                                                                                                                                           
                  
    scf.for %m0 = %c0 to %M step %Cm {                                                                                                                                          
                  
      // LX buffer for A — hoisted to m0 scope, reused across all n0                                                                                                            
      %lx_A = memref.alloc() : memref<?xf16, "LX">(%Cm)
                                                                                                                                                                                
      // Sibling pipeline: hoist A DRAM → LX once per m0                                                                                                                        
      ktdf.pipeline depth(1) {                                                                                                                                                  
        ktdf.stage depend_in() depend_out() {                                                                                                                                   
          scf.for %m1 = %c0 to %Cm step %c64 {
            ktdf.data_transfer from %A[%m0+%m1] size [64] to %lx_A[%m1] size [64] : memref<?xf16, "DRAM">, memref<?xf16, "LX">
          } // scf.for m1                                                                                                                                                       
          ktdf.stage_yield                                                                                                                                                      
        }                                                                                                                                                                       
      } // ktdf.pipeline (A hoist: DRAM → LX)

      scf.for %n0 = %c0 to %N step %Cn {

        %lx_B = memref.alloc() : memref<?x?xf16, "LX">(%Cm, %Cn)
        %lx_C = memref.alloc() : memref<?x?xf16, "LX">(%Cm, %Cn)

        %t1 = ktdf.create_token : !ktdf.token
        %t2 = ktdf.create_token : !ktdf.token
        %t3 = ktdf.create_token : !ktdf.token

        ktdf.pipeline depth(1) {

          // Outer Stage 0: B DRAM → LX
          ktdf.stage depend_in() depend_out(%t1) {
            scf.for %m1 = %c0 to %Cm step %c64 {
              scf.for %n1 = %c0 to %Cn step %c1 {
                ktdf.data_transfer from %B[%m0+%m1, %n0+%n1] size [64, 1] to %lx_B[%m1, %n1] size [64, 1] : memref<?x?xf16, "DRAM">, memref<?x?xf16, "LX">
              } // scf.for n1
            } // scf.for m1
            ktdf.stage_yield
          }

          // Outer Stage 1: compute (nested pipelines per m1)
          ktdf.stage depend_in(%t1) depend_out(%t2) {
            scf.for %m1 = %c0 to %Cm step %c64 {

              %fifo_slot_a = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"lxlu-to-sfp", 64xf16>

              // Sibling pipeline: hoist A LX → FIFO once per m1
              ktdf.pipeline depth(1) {
                ktdf.stage depend_in() depend_out() {
                  ktdf.data_transfer from %lx_A[%m1] size [64] to %fifo_slot_a size [64] : memref<?xf16, "LX">, !ktdf.fifo.slot<"lxlu-to-sfp", 64xf16>
                  ktdf.stage_yield
                }
              } // ktdf.pipeline (A hoist: LX → FIFO)

              scf.for %n1 = %c0 to %Cn step %c1 {

                %sfp_A   = memref.alloc() : memref<64xf16, "SFP_REG">
                %sfp_B   = memref.alloc() : memref<64xf16, "SFP_REG">
                %sfp_out = memref.alloc() : memref<64xf16, "SFP_REG">
                %fifo_slot_b = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"lxlu-to-sfp", 64xf16>
                %fifo_slot_c = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"sfp-to-lxsu", 64xf16>

                %t11 = ktdf.create_token : !ktdf.token
                %t12 = ktdf.create_token : !ktdf.token
                %t13 = ktdf.create_token : !ktdf.token
                                                                                                                                                                                 
                ktdf.pipeline depth(1) {
                                                                                                                                                                                 
                  // Inner Stage 0: B LX → FIFO
                  ktdf.stage depend_in() depend_out(%t11) {
                    ktdf.data_transfer from %lx_B[%m1, %n1] size [64, 1] to %fifo_slot_b size [64] : memref<?x?xf16, "LX">, !ktdf.fifo.slot<"lxlu-to-sfp", 64xf16>
                    ktdf.stage_yield
                  }

                  // Inner Stage 1: FIFO → SFP_REG, compute, SFP_REG → FIFO
                  ktdf.stage depend_in(%t11) depend_out(%t12) {
                    ktdf.data_transfer from %fifo_slot_a to %sfp_A[%c0] size [64] : !ktdf.fifo.slot<"lxlu-to-sfp", 64xf16>, memref<64xf16, "SFP_REG">
                    ktdf.data_transfer from %fifo_slot_b to %sfp_B[%c0] size [64] : !ktdf.fifo.slot<"lxlu-to-sfp", 64xf16>, memref<64xf16, "SFP_REG">
                    linalg.add ins(%sfp_A, %sfp_B : memref<64xf16, "SFP_REG">, memref<64xf16, "SFP_REG">) outs(%sfp_out : memref<64xf16, "SFP_REG">)
                    ktdf.data_transfer from %sfp_out[%c0] to %fifo_slot_c size [64] : memref<64xf16, "SFP_REG">, !ktdf.fifo.slot<"sfp-to-lxsu", 64xf16>
                    ktdf.stage_yield
                  }

                  // Inner Stage 2: FIFO → LX
                  ktdf.stage depend_in(%t12) depend_out(%t13) {
                    ktdf.data_transfer from %fifo_slot_c size [64] to %lx_C[%m1, %n1] size [64, 1] : !ktdf.fifo.slot<"sfp-to-lxsu", 64xf16>, memref<?x?xf16, "LX">
                    ktdf.stage_yield
                  }
                                                                                                                                                                                
                } // ktdf.pipeline (inner)                                                                                                                                      
                  
                memref.dealloc %sfp_B   : memref<64xf16, "SFP_REG">                                                                                                             
                memref.dealloc %sfp_out : memref<64xf16, "SFP_REG">
                                                                                                                                                                                
              } // scf.for n1
                                                                                                                                                                                
              memref.dealloc %sfp_A : memref<64xf16, "SFP_REG">                                                                                                                 
   
            } // scf.for m1                                                                                                                                                     
            ktdf.stage_yield
          }

          // Outer Stage 2: C LX → DRAM                                                                                                                                         
          ktdf.stage depend_in(%t2) depend_out(%t3) {
            scf.for %m1 = %c0 to %Cm step %c64 {                                                                                                                                
              scf.for %n1 = %c0 to %Cn step %c1 {                                                                                                                               
                ktdf.data_transfer from %lx_C[%m1, %n1] size [64, 1] to %C[%m0+%m1, %n0+%n1] size [64, 1] : memref<?x?xf16, "LX">, memref<?x?xf16, "DRAM">
              } // scf.for n1                                                                                                                                                   
            } // scf.for m1                                                                                                                                                     
            ktdf.stage_yield                                                                                                                                                    
          }       
                                                                                                                                                                                
        } // ktdf.pipeline (outer)                                                                                                                                              
   
        memref.dealloc %lx_B : memref<?x?xf16, "LX">                                                                                                                            
        memref.dealloc %lx_C : memref<?x?xf16, "LX">
                                                                                                                                                                                
      } // scf.for n0                                                                                                                                                           
                                                                                                                                                                                
      memref.dealloc %lx_A : memref<?xf16, "LX">                                                                                                                                
                  
    } // scf.for m0                                                                                                                                                             
                  
    return
  }


```