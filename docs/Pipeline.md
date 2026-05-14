# Scheduler Pipeline

## Overview

The pipeline takes a KTIR program — a single module containing a single function — and transforms it into a multi-module representation. The **main module** (analogous to an SDSC bundle) contains the program's control flow (loops and conditionals), which dispatches into **compute group modules** (analogous to SDSC local schedules). These compute group modules are eventually lowered to DFIR. This design intentionally mirrors the SDSC architecture so that the program-looping framework built for SDSC bundles can be reused in the new backend.

## Step 1 — Forming Compute Groups

The transformation from KTIR to the multi-module representation works by partitioning the computation into **compute groups**. Each group is a maximal sequence of operations bounded on both ends by loads from or stores to memory (HBM or LX). Using memory boundaries as group delimiters is a conservative but effective first cut to avoid IBUFF pressure issues. Groups can be refined into sub-groups in subsequent iterations as IBUFF pressure analysis improves.

```mlir 
module {
  func.func @add() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %tile_size = arith.constant 3 : index
    %A_start_address = arith.constant 1024 : index
    %B_start_address = arith.constant 12288 : index
    %C_start_address = arith.constant 18432 : index

    %id = ktdp.get_compute_tile_id : index
    %start_row = arith.muli %id, %tile_size : index

    // Construct a memory view of A from a given address
    %A_view = ktdp.construct_memory_view %A_start_address, sizes: [96, 64], strides: [64, 1] {
        coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 95 >= 0, d1 >= 0, -d1 + 63 >= 0)>,
        memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<96x64xf16>

    // Construct a memory view of B from a given address
    %B_view = ktdp.construct_memory_view %B_start_address, sizes: [96, 64], strides: [64, 1] {
        coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 95 >= 0, d1 >= 0, -d1 + 63 >= 0)>,
        memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<96x64xf16>

    // Construct a memory view of C from a given address
    %C_view = ktdp.construct_memory_view %C_start_address, sizes: [96, 64], strides: [64, 1] {
        coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 95 >= 0, d1 >= 0, -d1 + 63 >= 0)>,
        memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<96x64xf16>

    // Looping over tile size with each iteration working over 1x64 fp16
    scf.for %i = %c0 to %tile_size step %c1 {
        %start_row_i = arith.addi %start_row, %i : index
        // Construct an access tile from the memory view of A
        %A_access_tile = ktdp.construct_access_tile %A_view[%start_row_i, %c0] {
            access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 0 >= 0, d1 >= 0, -d1 + 63 >= 0)>,
            access_tile_order = #identity
        } : memref<96x64xf16> -> !ktdp.access_tile<1x64xindex>

        // Construct an access tile from the memory view of B
        %B_access_tile = ktdp.construct_access_tile %B_view[%start_row_i, %c0] {
            access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 0 >= 0, d1 >= 0, -d1 + 63 >= 0)>,
            access_tile_order = #identity
        } : memref<96x64xf16> -> !ktdp.access_tile<1x64xindex>

        // Load data from the corresponding access tile
        %A_data_tile = ktdp.load %A_access_tile : !ktdp.access_tile<1x64xindex> -> tensor<1x64xf16>

        %B_data_tile = ktdp.load %B_access_tile : !ktdp.access_tile<1x64xindex> -> tensor<1x64xf16>

        // C[start_row+i:start_row+i+2][c0:c0+63] = A[start_row+i:start_row+i+2][c0:c0+63] + B[start_row+i:start_row+i+2][c0:c0+63]
        // Perform add operation on the data tiles.
        %C_data_tile = tensor.empty() : tensor<1x64xf16>
        linalg.add ins(%A_data_tile, %B_data_tile : tensor<1x64xf16>, tensor<1x64xf16>)
                    outs(%C_data_tile: tensor<1x64xf16>) -> tensor<1x64xf16>

        // Construct an access tile from the memory view of C
        %C_access_tile = ktdp.construct_access_tile %C_view[%start_row_i, %c0] {
            access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 0 >= 0, d1 >= 0, -d1 + 63 >= 0)>,
            access_tile_order = #identity
        } : memref<96x64xf16> -> !ktdp.access_tile<1x64xindex>

        // Store data into the access tile.
        ktdp.store %C_data_tile, %C_access_tile : tensor<1x64xf16>, !ktdp.access_tile<1x64xindex>
    }

    return
  }
}
```

The overall main module (Global IR) looks as follows:

```mlir 
module {
  func.func @add() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %tile_size = arith.constant 3 : index

    // Looping over tile size with each iteration working over 1x64 fp16
    scf.for %i = %c0 to %tile_size step %c1 {
        func.func call @local-schedule-1(%i);
    }

    return
  }
}
```

The compute group module (Local IR) looks as follows, which is what goes into the scheduler.

```mlir
// Local schedule IR
module {
    func.func @local-schedule-1(%i: index) {
    
        %A_start_address = arith.constant 1024 : index
        %B_start_address = arith.constant 12288 : index
        %C_start_address = arith.constant 18432 : index
    
        %id = ktdp.get_compute_tile_id : index
        %start_row = arith.muli %id, %tile_size : index
    
        // Construct a memory view of A from a given address
        %A_view = ktdp.construct_memory_view %A_start_address, sizes: [96, 64], strides: [64, 1] {
            coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 95 >= 0, d1 >= 0, -d1 + 63 >= 0)>,
            memory_space = #ktdp.spyre_memory_space<HBM>
        } : memref<96x64xf16>
    
        // Construct a memory view of B from a given address
        %B_view = ktdp.construct_memory_view %B_start_address, sizes: [96, 64], strides: [64, 1] {
            coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 95 >= 0, d1 >= 0, -d1 + 63 >= 0)>,
            memory_space = #ktdp.spyre_memory_space<HBM>
        } : memref<96x64xf16>
    
        // Construct a memory view of C from a given address
        %C_view = ktdp.construct_memory_view %C_start_address, sizes: [96, 64], strides: [64, 1] {
            coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 95 >= 0, d1 >= 0, -d1 + 63 >= 0)>,
            memory_space = #ktdp.spyre_memory_space<HBM>
        } : memref<96x64xf16>
        
        %start_row_i = arith.addi %start_row, %i : index

        // Construct an access tile from the memory view of A
        %A_access_tile = ktdp.construct_access_tile %A_view[%start_row_i, %c0] {
            access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 0 >= 0, d1 >= 0, -d1 + 63 >= 0)>,
            access_tile_order = #identity
        } : memref<96x64xf16> -> !ktdp.access_tile<1x64xindex>

        // Construct an access tile from the memory view of B
        %B_access_tile = ktdp.construct_access_tile %B_view[%start_row_i, %c0] {
            access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 0 >= 0, d1 >= 0, -d1 + 63 >= 0)>,
            access_tile_order = #identity
        } : memref<96x64xf16> -> !ktdp.access_tile<1x64xindex>

        // Load data from the corresponding access tile
        %A_data_tile = ktdp.load %A_access_tile : !ktdp.access_tile<1x64xindex> -> tensor<1x64xf16>

        %B_data_tile = ktdp.load %B_access_tile : !ktdp.access_tile<1x64xindex> -> tensor<1x64xf16>

        // C[start_row+i:start_row+i+2][c0:c0+63] = A[start_row+i:start_row+i+2][c0:c0+63] + B[start_row+i:start_row+i+2][c0:c0+63]
        // Perform add operation on the data tiles.
        %C_data_tile = tensor.empty() : tensor<1x64xf16>
        linalg.add ins(%A_data_tile, %B_data_tile : tensor<1x64xf16>, tensor<1x64xf16>)
                    outs(%C_data_tile: tensor<1x64xf16>) -> tensor<1x64xf16>

        // Construct an access tile from the memory view of C
        %C_access_tile = ktdp.construct_access_tile %C_view[%start_row_i, %c0] {
            access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 0 >= 0, d1 >= 0, -d1 + 63 >= 0)>,
            access_tile_order = #identity
        } : memref<96x64xf16> -> !ktdp.access_tile<1x64xindex>

        // Store data into the access tile.
        ktdp.store %C_data_tile, %C_access_tile : tensor<1x64xf16>, !ktdp.access_tile<1x64xindex>    
    }
}
```

## Step 2 — Scheduler Input (Compute Group IR)

The scheduler operates independently over each compute group module, progressively lowering it to DFIR. The program-looping framework handles the control-flow structure in the main module.

The scheduler begins by mapping computations onto compute units. For the initial implementation, the following assumptions hold:

1. All computations are mapped to **SFP units**.
2. Input data resides in **DRAM or local LX**; outputs are written back to DRAM or local LX.
3. Data layout is amenable to **vectorization on SFP**.
4. Operations are **element-wise**, optionally with reductions or broadcasts.
5. A **single corelet** per core is assumed.

## Step 3 — 3-Stage Pipeline Construction

Given the above assumptions, the scheduler constructs a straightforward **3-stage pipeline** over each compute group, with each stage operating at the granularity of a **stick**.

Lowering `linalg.add` into `scf` loops has a key benefit: it exposes loop-type annotations (parallel, reduction, sequential) directly in the IR, eliminating the need for dependence analysis to recover this information.

The three pipeline stages are:
- **Stage 1 (Load):** Reads input tensors from their source memory (DRAM, as indicated by `construct_memory_view`) at stick granularity into SFP registers.
- **Stage 2 (Compute):** Executes the computation on the SFP engine.
- **Stage 3 (Store):** Writes the computed results back to the output tensor's destination memory (DRAM).

```mlir 
// transformed into a 3-stage pipeline
module {
    func.func @local-schedule-1(%i: index) {
        %A_start_address = arith.constant 1024 : index
        %B_start_address = arith.constant 12288 : index
        %C_start_address = arith.constant 18432 : index
                
        %id = ktdp.get_compute_tile_id : index
        %start_row = arith.muli %id, %tile_size : index
        %start_row_i = arith.addi %start_row, %i : index
                        
        // Construct a memory view of A from a given address
        %A_view = ktdp.construct_memory_view %A_start_address, sizes: [96, 64], strides: [64, 1] {
            coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 95 >= 0, d1 >= 0, -d1 + 63 >= 0)>,
            memory_space = #ktdp.spyre_memory_space<HBM>
        } : memref<96x64xf16>
    
        // Construct a memory view of B from a given address
        %B_view = ktdp.construct_memory_view %B_start_address, sizes: [96, 64], strides: [64, 1] {
            coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 95 >= 0, d1 >= 0, -d1 + 63 >= 0)>,
            memory_space = #ktdp.spyre_memory_space<HBM>
        } : memref<96x64xf16>
    
        // Construct a memory view of C from a given address
        %C_view = ktdp.construct_memory_view %C_start_address, sizes: [96, 64], strides: [64, 1] {
            coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 95 >= 0, d1 >= 0, -d1 + 63 >= 0)>,
            memory_space = #ktdp.spyre_memory_space<HBM>
        } : memref<96x64xf16>
       
        %offset = arith.muli %start_row_i, %c64 : index
        
        // Memory space cast: propagate ktdf memory space from construct_memory_view
        %A_cast = memref.memory_space_cast %A_view : memref<96x64xf16> to memref<96x64xf16, #ktdf.memory_space<HBM>>
        %B_cast = memref.memory_space_cast %B_view : memref<96x64xf16> to memref<96x64xf16, #ktdf.memory_space<HBM>>
        %C_cast = memref.memory_space_cast %C_view : memref<96x64xf16> to memref<96x64xf16, #ktdf.memory_space<HBM>>
        
        %A = memref.reinterpret_cast %A_cast to offset: [%offset], sizes: [1,64], strides: [64, 1] : memref<96x64xf16, #ktdf.memory_space<HBM>> to memref<1x64xf16, #ktdf.memory_space<HBM>>
        %B = memref.reinterpret_cast %B_cast to offset: [%offset], sizes: [1,64], strides: [64, 1] : memref<96x64xf16, #ktdf.memory_space<HBM>> to memref<1x64xf16, #ktdf.memory_space<HBM>>
        %C = memref.reinterpret_cast %C_cast to offset: [%offset], sizes: [1,64], strides: [64, 1] : memref<96x64xf16, #ktdf.memory_space<HBM>> to memref<1x64xf16, #ktdf.memory_space<HBM>>
        
        scf.for %m = %c0 to %c1 step %c1 {ktdf.loop parallel} {
          scf.for %n = %c0 to %c64 step %c64 {ktdf.loop parallel} {                                                                                                       
            ktdf.pipeline {                                                              
              %res_fifo_A, %res_fifo_B, %res_fifo_out, %res_t1, %res_t2, %res_t3 = ktdf.private {                       
                %fifo_slot_A = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"hbm-to-sfp", 64xf16>
                %fifo_slot_B = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"hbm-to-sfp", 64xf16>
                %fifo_slot_out = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"sfp-to-hbm", 64xf16>
                                                          
                %t1 = ktdf.create_token : !ktdf.token                                                              
                %t2 = ktdf.create_token : !ktdf.token                                                               
                %t3 = ktdf.create_token : !ktdf.token                                                               
                ktdf.private_yield %fifo_slot_A, %fifo_slot_B, %fifo_slot_out, %t1, %t2, %t3                                          
                    : !ktdf.fifo.slot<"hbm-to-sfp", 64xf16>, !ktdf.fifo.slot<"hbm-to-sfp", 64xf16>, !ktdf.fifo.slot<"hbm-to-sfp", 64xf16>,
                      !ktdf.token, !ktdf.token, !ktdf.token
              }                                                                                                                                                         
              ktdf.stage depend_in() depend_out(%res_t1) {                                                          
                ktdf.data_transfer from %A[%m, %n] size [1, 64] to %res_fifo_A size [64] : memref<1x64xf16, strided<[64, 1], offset: ?>>, !ktdf.fifo.slot<"hbm-to-sfp", 64xf16>
                ktdf.data_transfer from %B[%m, %n] size [1, 64] to %res_fifo_B size [64] : memref<1x64xf16, strided<[64, 1], offset: ?>>, !ktdf.fifo.slot<"hbm-to-sfp", 64xf16>
              }                                                                                                     
              
              ktdf.stage depend_in(%res_t1) depend_out(%res_t2) {                 
                %fifo_read_A = ktdf.read_from_fifo %res_fifo_A : <"hbm-to-sfp", 64xf16> -> tensor<1x64xf16>
                %fifo_read_B = ktdf.read_from_fifo %res_fifo_B : <"hbm-to-sfp", 64xf16> -> tensor<1x64xf16>
                %C_tensor = tensor.empty() : tensor<1x64xf16>
                %add = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%fifo_read_A, %fifo_read_B : tensor<1x64xf16>, tensor<1x64xf16>) outs(%C_tensor : tensor<1x64xf16>) {
                ^bb0(%in: f16, %in_10: f16, %out: f16):
                  %14 = arith.addf %in, %in_10 : f16
                  linalg.yield %14 : f16
                } -> tensor<1x64xf16>
                ktdf.write_to_fifo %add, %res_fifo_out : tensor<1x64xf16>, <"sfp-to-hbm", 64xf16>
              }                                                                                                     
              
              ktdf.stage depend_in(%res_t2) depend_out(%res_t3) {                                                   
                ktdf.data_transfer from %res_fifo_out size [64] to %reinterpret_cast_9[%arg1, %arg2] size [1, 64] : !ktdf.fifo.slot<"sfp-to-hbm", 64xf16>, memref<1x64xf16, strided<[64, 1], offset: ?>>
              }                                                                                                    
            } // ktdf.pipeline depth(0)                                                                            
          } // scf.for n parallel
        } // scf.for m parallel    
        return 
    }
}
```

## Step 4 - Path Expansion 

The first transformation enforces hardware legality by expanding each end-to-end data movement into the required intermediate hops through the scratchpad (LX). Instead of allowing direct DRAM↔SFP transfers, the compiler inserts explicit stages for each hop.

Each hop becomes its own stage, making the memory hierarchy and movement explicit. This pass does not change the iteration structure or synchronization granularity—it simply rewrites the IR so that it matches the physical capabilities of the architecture.

```mlir 
// After path expansion
module {
    func.func @local-schedule-1(%i: index) {
        %A_start_address = arith.constant 1024 : index
        %B_start_address = arith.constant 12288 : index
        %C_start_address = arith.constant 18432 : index
                
        %id = ktdp.get_compute_tile_id : index
        %start_row = arith.muli %id, %tile_size : index
        %start_row_i = arith.addi %start_row, %i : index
                        
        // Construct a memory view of A from a given address
        %A_view = ktdp.construct_memory_view %A_start_address, sizes: [96, 64], strides: [64, 1] {
            coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 95 >= 0, d1 >= 0, -d1 + 63 >= 0)>,
            memory_space = #ktdp.spyre_memory_space<HBM>
        } : memref<96x64xf16>
    
        // Construct a memory view of B from a given address
        %B_view = ktdp.construct_memory_view %B_start_address, sizes: [96, 64], strides: [64, 1] {
            coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 95 >= 0, d1 >= 0, -d1 + 63 >= 0)>,
            memory_space = #ktdp.spyre_memory_space<HBM>
        } : memref<96x64xf16>
    
        // Construct a memory view of C from a given address
        %C_view = ktdp.construct_memory_view %C_start_address, sizes: [96, 64], strides: [64, 1] {
            coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 95 >= 0, d1 >= 0, -d1 + 63 >= 0)>,
            memory_space = #ktdp.spyre_memory_space<HBM>
        } : memref<96x64xf16>
       
        %offset = arith.muli %start_row_i, %c64 : index
        
        // Memory space cast: propagate ktdf memory space from construct_memory_view
        %A_cast = memref.memory_space_cast %A_view : memref<96x64xf16> to memref<96x64xf16, #ktdf.memory_space<HBM>>
        %B_cast = memref.memory_space_cast %B_view : memref<96x64xf16> to memref<96x64xf16, #ktdf.memory_space<HBM>>
        %C_cast = memref.memory_space_cast %C_view : memref<96x64xf16> to memref<96x64xf16, #ktdf.memory_space<HBM>>
        
        %A = memref.reinterpret_cast %A_cast to offset: [%offset], sizes: [1,64], strides: [64, 1] : memref<96x64xf16, #ktdf.memory_space<HBM>> to memref<1x64xf16, #ktdf.memory_space<HBM>>
        %B = memref.reinterpret_cast %B_cast to offset: [%offset], sizes: [1,64], strides: [64, 1] : memref<96x64xf16, #ktdf.memory_space<HBM>> to memref<1x64xf16, #ktdf.memory_space<HBM>>
        %C = memref.reinterpret_cast %C_cast to offset: [%offset], sizes: [1,64], strides: [64, 1] : memref<96x64xf16, #ktdf.memory_space<HBM>> to memref<1x64xf16, #ktdf.memory_space<HBM>>
        
        scf.for %m = %c0 to %c1 step %c1 {ktdf.loop parallel} {
          scf.for %n = %c0 to %c64 step %c64 {ktdf.loop parallel} {                                                                                                       
           ktdf.pipeline depth(0)  {
              %res_lx_A, %res_lx_B, %res_lx_C, %res_sfp_A, %res_sfp_B, %res_sfp_out,
              %res_fifo_slot_a, %res_fifo_slot_b, %res_fifo_slot_c,
              %res_t1, %res_t2, %res_t3, %res_t4, %res_t5 = ktdf.private {
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
                ktdf.private_yield %lx_A, %lx_B, %lx_C, %sfp_A, %sfp_B, %sfp_out,
                                   %fifo_slot_a, %fifo_slot_b, %fifo_slot_c,
                                   %t1, %t2, %t3, %t4, %t5
                    : memref<64xf16, "LX">, memref<64xf16, "LX">, memref<64xf16, "LX">,
                      memref<64xf16, "SFP_REG">, memref<64xf16, "SFP_REG">, memref<64xf16, "SFP_REG">,
                      !ktdf.fifo.slot<"lxlu-to-sfp", 64xf16>, !ktdf.fifo.slot<"lxlu-to-sfp", 64xf16>,
                      !ktdf.fifo.slot<"sfp-to-lxsu", 64xf16>,
                      !ktdf.token, !ktdf.token, !ktdf.token, !ktdf.token, !ktdf.token
              }
    
              ktdf.stage depend_in() depend_out(%res_t1) {
                ktdf.data_transfer from %A[%m, %n] size [1, 64] to %res_lx_A[%c0] size [64] : memref<1x64xf16, "DRAM">, memref<64xf16, "LX">
                ktdf.data_transfer from %B[%m, %n] size [1, 64] to %res_lx_B[%c0] size [64] : memref<1x64xf16, "DRAM">, memref<64xf16, "LX">
                ktdf.stage_yield
              }
    
              ktdf.stage depend_in(%res_t1) depend_out(%res_t2) {
                ktdf.data_transfer from %res_lx_A[%c0] size [64] to %res_fifo_slot_a size [64] : memref<64xf16, "LX">, !ktdf.fifo.slot<"lxlu-to-sfp", 64xf16>
                ktdf.data_transfer from %res_lx_B[%c0] size [64] to %res_fifo_slot_b size [64] : memref<64xf16, "LX">, !ktdf.fifo.slot<"lxlu-to-sfp", 64xf16>
                ktdf.stage_yield
              }
    
              ktdf.stage depend_in(%res_t2) depend_out(%res_t3) {
                ktdf.data_transfer from %res_fifo_slot_a to %res_sfp_A[%c0] size [64] : !ktdf.fifo.slot<"lxlu-to-sfp", 64xf16>, memref<64xf16, "SFP_REG">
                ktdf.data_transfer from %res_fifo_slot_b to %res_sfp_B[%c0] size [64] : !ktdf.fifo.slot<"lxlu-to-sfp", 64xf16>, memref<64xf16, "SFP_REG">
                linalg.add ins(%res_sfp_A, %res_sfp_B : memref<64xf16, "SFP_REG">, memref<64xf16, "SFP_REG">) outs(%res_sfp_out : memref<64xf16, "SFP_REG">)
                ktdf.data_transfer from %res_sfp_out[%c0] to %res_fifo_slot_c size [64] : memref<64xf16, "SFP_REG">, !ktdf.fifo.slot<"sfp-to-lxsu", 64xf16>
                ktdf.stage_yield
              }
    
              ktdf.stage depend_in(%res_t3) depend_out(%res_t4) {
                ktdf.data_transfer from %res_fifo_slot_c size [64] to %res_lx_C[%c0] size [64] : !ktdf.fifo.slot<"sfp-to-lxsu", 64xf16>, memref<64xf16, "LX">
                ktdf.stage_yield
              }
    
              ktdf.stage depend_in(%res_t4) depend_out(%res_t5) {
                ktdf.data_transfer from %res_lx_C[%c0] size [64] to %C[%m, %n] size [1, 64] : memref<64xf16, "LX">, memref<1x64xf16, "DRAM">
                ktdf.stage_yield
              }
    
            } // ktdf.pipeline depth(0)                                                                           
          } // scf.for n parallel
        } // scf.for m parallel    
        return 
    }
}
```

## Step 5, 6, 7, 8 - Tiling, Loop Sinking/Stage coarsening, Permutation if needed, reuse promotion,

```mlir 
// After tiling, sinking, permutation, and reuse promotion.
// In this specific case of M=1, N=64: Cm = 1, Cn=64
module {
    func.func @local-schedule-1(%i: index) {
        %A_start_address = arith.constant 1024 : index
        %B_start_address = arith.constant 12288 : index
        %C_start_address = arith.constant 18432 : index
                
        %id = ktdp.get_compute_tile_id : index
        %start_row = arith.muli %id, %tile_size : index
        %start_row_i = arith.addi %start_row, %i : index
                        
        // Construct a memory view of A from a given address
        %A_view = ktdp.construct_memory_view %A_start_address, sizes: [96, 64], strides: [64, 1] {
            coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 95 >= 0, d1 >= 0, -d1 + 63 >= 0)>,
            memory_space = #ktdp.spyre_memory_space<HBM>
        } : memref<96x64xf16>
    
        // Construct a memory view of B from a given address
        %B_view = ktdp.construct_memory_view %B_start_address, sizes: [96, 64], strides: [64, 1] {
            coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 95 >= 0, d1 >= 0, -d1 + 63 >= 0)>,
            memory_space = #ktdp.spyre_memory_space<HBM>
        } : memref<96x64xf16>
    
        // Construct a memory view of C from a given address
        %C_view = ktdp.construct_memory_view %C_start_address, sizes: [96, 64], strides: [64, 1] {
            coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 95 >= 0, d1 >= 0, -d1 + 63 >= 0)>,
            memory_space = #ktdp.spyre_memory_space<HBM>
        } : memref<96x64xf16>
       
        %offset = arith.muli %start_row_i, %c64 : index
        
        // Memory space cast: propagate ktdf memory space from construct_memory_view
        %A_cast = memref.memory_space_cast %A_view : memref<96x64xf16> to memref<96x64xf16, #ktdf.memory_space<HBM>>
        %B_cast = memref.memory_space_cast %B_view : memref<96x64xf16> to memref<96x64xf16, #ktdf.memory_space<HBM>>
        %C_cast = memref.memory_space_cast %C_view : memref<96x64xf16> to memref<96x64xf16, #ktdf.memory_space<HBM>>
        
        %A = memref.reinterpret_cast %A_cast to offset: [%offset], sizes: [1,64], strides: [64, 1] : memref<96x64xf16, #ktdf.memory_space<HBM>> to memref<1x64xf16, #ktdf.memory_space<HBM>>
        %B = memref.reinterpret_cast %B_cast to offset: [%offset], sizes: [1,64], strides: [64, 1] : memref<96x64xf16, #ktdf.memory_space<HBM>> to memref<1x64xf16, #ktdf.memory_space<HBM>>
        %C = memref.reinterpret_cast %C_cast to offset: [%offset], sizes: [1,64], strides: [64, 1] : memref<96x64xf16, #ktdf.memory_space<HBM>> to memref<1x64xf16, #ktdf.memory_space<HBM>>
        
        scf.for %m0 = %c0 to %M step %Cm {ktdf.loop parallel} {
          scf.for %n0 = %c0 to %N step %Cn {ktdf.loop parallel} {
    
            ktdf.pipeline depth(0) modulo(all) {
    
              %res_lx_A, %res_lx_B, %res_lx_C, %res_t1, %res_t2, %res_t3 = ktdf.private {
                %lx_A = memref.alloc() : memref<?x?xf16, "LX">(%Cm, %Cn)
                %lx_B = memref.alloc() : memref<?x?xf16, "LX">(%Cm, %Cn)
                %lx_C = memref.alloc() : memref<?x?xf16, "LX">(%Cm, %Cn)    
                %t1 = ktdf.create_token : !ktdf.token
                %t2 = ktdf.create_token : !ktdf.token                       
                %t3 = ktdf.create_token : !ktdf.token                       
                ktdf.private_yield %lx_A, %lx_B, %lx_C, %t1, %t2, %t3
                    : memref<?x?xf16, "LX">, memref<?x?xf16, "LX">, memref<?x?xf16, "LX">,
                      !ktdf.token, !ktdf.token, !ktdf.token
              }
    
              // Outer Stage 0: DRAM → LX (tile-sized, loops sunk in)       
              ktdf.stage depend_in() depend_out(%res_t1) {
                scf.for %m1 = %c0 to %Cm step %c1 {ktdf.loop parallel} {    
                  scf.for %n1 = %c0 to %Cn step %c64 {ktdf.loop parallel} { 
                    ktdf.data_transfer from %A[%m0+%m1, %n0+%n1] size [1, 64] to %res_lx_A[%0] size [1, 64] : memref<?x?xf16, "DRAM">, memref<?x?xf16, "LX">       
                    ktdf.data_transfer from %B[%m0+%m1, %n0+%n1] size [1, 64] to %res_lx_B[%m1, %n1] size [1, 64] : memref<?x?xf16, "DRAM">, memref<?x?xf16, "LX">       
                  } // scf.for n1 parallel                                                                                                          
                } // scf.for m1 parallel                                                                                                            
                ktdf.stage_yield       
              }                                                                                                                                                         
              // Outer Stage 1: nested pipeline over (m1, n1)
              ktdf.stage depend_in(%res_t1) depend_out(%res_t2) {
                scf.for %m1 = %c0 to %Cm step %c1 {ktdf.loop parallel} {
                  scf.for %n1 = %c0 to %Cn step %c64 {ktdf.loop parallel} {
                    ktdf.pipeline depth(1)  {
                      %res_sfp_A, %res_sfp_B, %res_sfp_out,
                      %res_fifo_slot_a, %res_fifo_slot_b, %res_fifo_slot_c,
                      %res_t11, %res_t12, %res_t13 = ktdf.private {
                        %sfp_A   = memref.alloc() : memref<64xf16, "SFP_REG">
                        %sfp_B   = memref.alloc() : memref<64xf16, "SFP_REG">
                        %sfp_out = memref.alloc() : memref<64xf16, "SFP_REG">
                        %fifo_slot_a, %fifo_slot_b = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"lxlu-to-sfp", 64xf16>,
                                                                            !ktdf.fifo.slot<"lxlu-to-sfp", 64xf16>
                        %fifo_slot_c = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"sfp-to-lxsu", 64xf16>
                        %t11 = ktdf.create_token : !ktdf.token
                        %t12 = ktdf.create_token : !ktdf.token
                        %t13 = ktdf.create_token : !ktdf.token
                        ktdf.private_yield %sfp_A, %sfp_B, %sfp_out, %fifo_slot_a, %fifo_slot_b, %fifo_slot_c, %t11, %t12, %t13
                            : memref<64xf16, "SFP_REG">, memref<64xf16, "SFP_REG">, memref<64xf16, "SFP_REG">,
                              !ktdf.fifo.slot<"lxlu-to-sfp", 64xf16>, !ktdf.fifo.slot<"lxlu-to-sfp", 64xf16>, !ktdf.fifo.slot<"sfp-to-lxsu", 64xf16>,
                              !ktdf.token, !ktdf.token, !ktdf.token
                      }
                      ktdf.stage depend_in() depend_out(%res_t11) {
                        ktdf.data_transfer from %res_lx_A[%m1, %n1] size [1, 64] to %res_fifo_slot_a size [64] : memref<?x?xf16, "LX">, !ktdf.fifo.slot<"lxlu-to-sfp", 64xf16>
                        ktdf.data_transfer from %res_lx_B[%m1, %n1] size [1, 64] to %res_fifo_slot_b size [64] : memref<?x?xf16, "LX">, !ktdf.fifo.slot<"lxlu-to-sfp", 64xf16>
                        ktdf.stage_yield
                      }
                      ktdf.stage depend_in(%res_t11) depend_out(%res_t12) {
                        ktdf.data_transfer from %res_fifo_slot_a to %res_sfp_A[%c0] size [64] : !ktdf.fifo.slot<"lxlu-to-sfp", 64xf16>, memref<64xf16, "SFP_REG">
                        ktdf.data_transfer from %res_fifo_slot_b to %res_sfp_B[%c0] size [64] : !ktdf.fifo.slot<"lxlu-to-sfp", 64xf16>, memref<64xf16, "SFP_REG">
                        linalg.add ins(%res_sfp_A, %res_sfp_B : memref<64xf16, "SFP_REG">, memref<64xf16, "SFP_REG">) outs(%res_sfp_out : memref<64xf16, "SFP_REG">)
                        ktdf.data_transfer from %res_sfp_out[%c0] to %res_fifo_slot_c size [64] : memref<64xf16, "SFP_REG">, !ktdf.fifo.slot<"sfp-to-lxsu", 64xf16>
                        ktdf.stage_yield
                      }
                      ktdf.stage depend_in(%res_t12) depend_out(%res_t13) {
                        ktdf.data_transfer from %res_fifo_slot_c size [64] to %res_lx_C[%m1, %n1] size [1, 64] : !ktdf.fifo.slot<"sfp-to-lxsu", 64xf16>, memref<?x?xf16, "LX">
                        ktdf.stage_yield
                      }
                    } // ktdf.pipeline depth(1)
                  } // scf.for n1 parallel
                } // scf.for m1 parallel                                                                                                                                
                ktdf.stage_yield                             
              }
                                                                                                                                                    
              // Outer Stage 2: LX → DRAM (tile-sized, loops sunk in)
              ktdf.stage depend_in(%res_t2) depend_out(%res_t3) {
                scf.for %m1 = %c0 to %Cm step %c1 {ktdf.loop parallel} {
                  scf.for %n1 = %c0 to %Cn step %c64 {ktdf.loop parallel} {
                    ktdf.data_transfer from %res_lx_C[%m1, %n1] size [1, 64] to %C[%m0+%m1, %n0+%n1] size [1, 64] : memref<?x?xf16, "LX">, memref<?x?xf16, "DRAM">
                  } // scf.for n1 parallel                                 
                } // scf.for m1 parallel                                                                                                                                
                ktdf.stage_yield                                           
              }                                                            
            } // ktdf.pipeline depth(0)                                    
          } // scf.for n0 parallel                                         
        } // scf.for m0 parallel   
        return 
    }
}
```
## Step-9: Double buffering at LX-level

```mlir 
module {
  func.func @local_schedule_1_db(%i: index, %M: index, %N: index, %Cm: index, %Cn: index, %tile_size: index) {
    %A_start_address = arith.constant 1024 : index
    %B_start_address = arith.constant 12288 : index
    %C_start_address = arith.constant 18432 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index

    %id = ktdp.get_compute_tile_id : index
    %start_row = arith.muli %id, %tile_size : index
    %start_row_i = arith.addi %start_row, %i : index

    %A_view = ktdp.construct_memory_view %A_start_address, sizes: [96, 64], strides: [64, 1] {
      coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 95 >= 0, d1 >= 0, -d1 + 63 >= 0)>,
      memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<96x64xf16>

    %B_view = ktdp.construct_memory_view %B_start_address, sizes: [96, 64], strides: [64, 1] {
      coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 95 >= 0, d1 >= 0, -d1 + 63 >= 0)>,
      memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<96x64xf16>

    %C_view = ktdp.construct_memory_view %C_start_address, sizes: [96, 64], strides: [64, 1] {
      coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 95 >= 0, d1 >= 0, -d1 + 63 >= 0)>,
      memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<96x64xf16>

    %offset = arith.muli %start_row_i, %c64 : index
    
    // Memory space cast: propagate ktdf memory space from construct_memory_view
    %A_cast = memref.memory_space_cast %A_view : memref<96x64xf16> to memref<96x64xf16, #ktdf.memory_space<HBM>>
    %B_cast = memref.memory_space_cast %B_view : memref<96x64xf16> to memref<96x64xf16, #ktdf.memory_space<HBM>>
    %C_cast = memref.memory_space_cast %C_view : memref<96x64xf16> to memref<96x64xf16, #ktdf.memory_space<HBM>>
    
    %A = memref.reinterpret_cast %A_cast to offset: [%offset], sizes: [1, 64], strides: [64, 1] : memref<96x64xf16, #ktdf.memory_space<HBM>> to memref<1x64xf16, #ktdf.memory_space<HBM>>
    %B = memref.reinterpret_cast %B_cast to offset: [%offset], sizes: [1, 64], strides: [64, 1] : memref<96x64xf16, #ktdf.memory_space<HBM>> to memref<1x64xf16, #ktdf.memory_space<HBM>>
    %C = memref.reinterpret_cast %C_cast to offset: [%offset], sizes: [1, 64], strides: [64, 1] : memref<96x64xf16, #ktdf.memory_space<HBM>> to memref<1x64xf16, #ktdf.memory_space<HBM>>

    %lx_A0 = memref.alloc(%Cm, %Cn) : memref<?x?xf16, "LX">
    %lx_A1 = memref.alloc(%Cm, %Cn) : memref<?x?xf16, "LX">
    %lx_B0 = memref.alloc(%Cm, %Cn) : memref<?x?xf16, "LX">
    %lx_B1 = memref.alloc(%Cm, %Cn) : memref<?x?xf16, "LX">
    %lx_C0 = memref.alloc(%Cm, %Cn) : memref<?x?xf16, "LX">
    %lx_C1 = memref.alloc(%Cm, %Cn) : memref<?x?xf16, "LX">
        
    scf.for %m0 = %c0 to %M step %Cm {
      scf.for %n0 = %c0 to %N step %Cn {

        %phase = ktdf.buffer_phase (%m0, %n0) : index
        %res_lx_A = ktdf.select_memref %phase [%lx_A0, %lx_A1] : index -> memref<?x?xf16, "LX">
        %res_lx_B = ktdf.select_memref %phase [%lx_B0, %lx_B1] : index -> memref<?x?xf16, "LX">
        %res_lx_C = ktdf.select_memref %phase [%lx_C0, %lx_C1] : index -> memref<?x?xf16, "LX">

        ktdf.pipeline depth(0) modulo(size:2) {

          %t1, %t2, %t3 = ktdf.private {
            %tok1 = ktdf.create_token : !ktdf.token
            %tok2 = ktdf.create_token : !ktdf.token
            %tok3 = ktdf.create_token : !ktdf.token
            ktdf.private_yield %tok1, %tok2, %tok3 : !ktdf.token, !ktdf.token, !ktdf.token
          }

          ktdf.stage depend_in() depend_out(%t1) {
            scf.for %m1 = %c0 to %Cm step %c1 { ktdf.loop parallel } {
              scf.for %n1 = %c0 to %Cn step %c64 { ktdf.loop parallel } {
                ktdf.data_transfer from %A[%m0 + %m1, %n0 + %n1] size [1, 64] to %res_lx_A[%m1, %n1] size [1, 64] : memref<?x?xf16, "DRAM">, memref<?x?xf16, "LX">
                ktdf.data_transfer from %B[%m0 + %m1, %n0 + %n1] size [1, 64] to %res_lx_B[%m1, %n1] size [1, 64] : memref<?x?xf16, "DRAM">, memref<?x?xf16, "LX">
              }
            }
            ktdf.stage_yield
          }

          ktdf.stage depend_in(%t1) depend_out(%t2) {
            scf.for %m1 = %c0 to %Cm step %c1 { ktdf.loop parallel } {
              scf.for %n1 = %c0 to %Cn step %c64 { ktdf.loop parallel } {
                ktdf.pipeline depth(1) {
                  %res_sfp_A, %res_sfp_B, %res_sfp_out,
                  %res_fifo_slot_a, %res_fifo_slot_b, %res_fifo_slot_c,
                  %t11, %t12, %t13 = ktdf.private {
                    %sfp_A = memref.alloc() : memref<64xf16, "SFP_REG">
                    %sfp_B = memref.alloc() : memref<64xf16, "SFP_REG">
                    %sfp_out = memref.alloc() : memref<64xf16, "SFP_REG">
                    %fifo_slot_a, %fifo_slot_b = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"lxlu-to-sfp", 64xf16>,
                                                                        !ktdf.fifo.slot<"lxlu-to-sfp", 64xf16>
                    %fifo_slot_c = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"sfp-to-lxsu", 64xf16>
                    %tok11 = ktdf.create_token : !ktdf.token
                    %tok12 = ktdf.create_token : !ktdf.token
                    %tok13 = ktdf.create_token : !ktdf.token
                    ktdf.private_yield %sfp_A, %sfp_B, %sfp_out,
                                       %fifo_slot_a, %fifo_slot_b, %fifo_slot_c,
                                       %tok11, %tok12, %tok13
                      : memref<64xf16, "SFP_REG">, memref<64xf16, "SFP_REG">, memref<64xf16, "SFP_REG">,
                        !ktdf.fifo.slot<"lxlu-to-sfp", 64xf16>, !ktdf.fifo.slot<"lxlu-to-sfp", 64xf16>, !ktdf.fifo.slot<"sfp-to-lxsu", 64xf16>,
                        !ktdf.token, !ktdf.token, !ktdf.token
                  }

                  ktdf.stage depend_in() depend_out(%t11) {
                    ktdf.data_transfer from %res_lx_A[%m1, %n1] size [1, 64] to %res_fifo_slot_a size [64] : memref<?x?xf16, "LX">, !ktdf.fifo.slot<"lxlu-to-sfp", 64xf16>
                    ktdf.data_transfer from %res_lx_B[%m1, %n1] size [1, 64] to %res_fifo_slot_b size [64] : memref<?x?xf16, "LX">, !ktdf.fifo.slot<"lxlu-to-sfp", 64xf16>
                    ktdf.stage_yield
                  }

                  ktdf.stage depend_in(%t11) depend_out(%t12) {
                    ktdf.data_transfer from %res_fifo_slot_a to %res_sfp_A[%c0] size [64] : !ktdf.fifo.slot<"lxlu-to-sfp", 64xf16>, memref<64xf16, "SFP_REG">
                    ktdf.data_transfer from %res_fifo_slot_b to %res_sfp_B[%c0] size [64] : !ktdf.fifo.slot<"lxlu-to-sfp", 64xf16>, memref<64xf16, "SFP_REG">
                    linalg.add ins(%res_sfp_A, %res_sfp_B : memref<64xf16, "SFP_REG">, memref<64xf16, "SFP_REG">) outs(%res_sfp_out : memref<64xf16, "SFP_REG">)
                    ktdf.data_transfer from %res_sfp_out[%c0] to %res_fifo_slot_c size [64] : memref<64xf16, "SFP_REG">, !ktdf.fifo.slot<"sfp-to-lxsu", 64xf16>
                    ktdf.stage_yield
                  }

                  ktdf.stage depend_in(%t12) depend_out(%t13) {
                    ktdf.data_transfer from %res_fifo_slot_c size [64] to %res_lx_C[%m1, %n1] size [1, 64] : !ktdf.fifo.slot<"sfp-to-lxsu", 64xf16>, memref<?x?xf16, "LX">
                    ktdf.stage_yield
                  }
                }
              }
            }
            ktdf.stage_yield
          }

          ktdf.stage depend_in(%t2) depend_out(%t3) {
            scf.for %m1 = %c0 to %Cm step %c1 { ktdf.loop parallel } {
              scf.for %n1 = %c0 to %Cn step %c64 { ktdf.loop parallel } {
                ktdf.data_transfer from %res_lx_C[%m1, %n1] size [1, 64] to %C[%m0 + %m1, %n0 + %n1] size [1, 64] : memref<?x?xf16, "LX">, memref<?x?xf16, "DRAM">
              }
            }
            ktdf.stage_yield
          }
        }

        memref.dealloc %lx_A0 : memref<?x?xf16, "LX">
        memref.dealloc %lx_A1 : memref<?x?xf16, "LX">
        memref.dealloc %lx_B0 : memref<?x?xf16, "LX">
        memref.dealloc %lx_B1 : memref<?x?xf16, "LX">
        memref.dealloc %lx_C0 : memref<?x?xf16, "LX">
        memref.dealloc %lx_C1 : memref<?x?xf16, "LX">
      }
    }

    return
  }
}
```

## Step-10: Tile size selection and Memory allocation for intermediate tensors, vector registers.

```mlir 
module {
  func.func @local_schedule_1_db(%i: index, %M: index, %N: index, %Cm: index, %Cn: index, %tile_size: index) {
    %A_start_address = arith.constant 1024 : index
    %B_start_address = arith.constant 12288 : index
    %C_start_address = arith.constant 18432 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index

    %id = ktdp.get_compute_tile_id : index
    %start_row = arith.muli %id, %tile_size : index
    %start_row_i = arith.addi %start_row, %i : index

    %A_view = ktdp.construct_memory_view %A_start_address, sizes: [96, 64], strides: [64, 1] {
      coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 95 >= 0, d1 >= 0, -d1 + 63 >= 0)>,
      memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<96x64xf16>

    %B_view = ktdp.construct_memory_view %B_start_address, sizes: [96, 64], strides: [64, 1] {
      coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 95 >= 0, d1 >= 0, -d1 + 63 >= 0)>,
      memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<96x64xf16>

    %C_view = ktdp.construct_memory_view %C_start_address, sizes: [96, 64], strides: [64, 1] {
      coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 95 >= 0, d1 >= 0, -d1 + 63 >= 0)>,
      memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<96x64xf16>

    %offset = arith.muli %start_row_i, %c64 : index
    
    // Memory space cast: propagate ktdf memory space from construct_memory_view
    %A_cast = memref.memory_space_cast %A_view : memref<96x64xf16> to memref<96x64xf16, #ktdf.memory_space<HBM>>
    %B_cast = memref.memory_space_cast %B_view : memref<96x64xf16> to memref<96x64xf16, #ktdf.memory_space<HBM>>
    %C_cast = memref.memory_space_cast %C_view : memref<96x64xf16> to memref<96x64xf16, #ktdf.memory_space<HBM>>
    
    %A = memref.reinterpret_cast %A_cast to offset: [%offset], sizes: [1, 64], strides: [64, 1] : memref<96x64xf16, #ktdf.memory_space<HBM>> to memref<1x64xf16, #ktdf.memory_space<HBM>>
    %B = memref.reinterpret_cast %B_cast to offset: [%offset], sizes: [1, 64], strides: [64, 1] : memref<96x64xf16, #ktdf.memory_space<HBM>> to memref<1x64xf16, #ktdf.memory_space<HBM>>
    %C = memref.reinterpret_cast %C_cast to offset: [%offset], sizes: [1, 64], strides: [64, 1] : memref<96x64xf16, #ktdf.memory_space<HBM>> to memref<1x64xf16, #ktdf.memory_space<HBM>>

    %lx_A0 = memref.alloc(%Cm, %Cn) {Addr: A0} : memref<?x?xf16, "LX">
    %lx_A1 = memref.alloc(%Cm, %Cn) {Addr: A1} : memref<?x?xf16, "LX">
    %lx_B0 = memref.alloc(%Cm, %Cn) {Addr: B0} : memref<?x?xf16, "LX">
    %lx_B1 = memref.alloc(%Cm, %Cn) {Addr: B1} : memref<?x?xf16, "LX">
    %lx_C0 = memref.alloc(%Cm, %Cn) {Addr: C0} : memref<?x?xf16, "LX">
    %lx_C1 = memref.alloc(%Cm, %Cn) {Addr: C1} : memref<?x?xf16, "LX">
   
        
    scf.for %m0 = %c0 to %M step %Cm {
      scf.for %n0 = %c0 to %N step %Cn {

        %phase = ktdf.buffer_phase (%m0, %n0) : index
        %res_lx_A = ktdf.select_memref %phase [%lx_A0, %lx_A1] : index -> memref<?x?xf16, "LX">
        %res_lx_B = ktdf.select_memref %phase [%lx_B0, %lx_B1] : index -> memref<?x?xf16, "LX">
        %res_lx_C = ktdf.select_memref %phase [%lx_C0, %lx_C1] : index -> memref<?x?xf16, "LX">

        ktdf.pipeline depth(0) modulo(size:2) {

          %t1, %t2, %t3 = ktdf.private {
            %tok1 = ktdf.create_token : !ktdf.token
            %tok2 = ktdf.create_token : !ktdf.token
            %tok3 = ktdf.create_token : !ktdf.token
            ktdf.private_yield %tok1, %tok2, %tok3 : !ktdf.token, !ktdf.token, !ktdf.token
          }

          ktdf.stage depend_in() depend_out(%t1) {
            scf.for %m1 = %c0 to %Cm step %c1 { ktdf.loop parallel } {
              scf.for %n1 = %c0 to %Cn step %c64 { ktdf.loop parallel } {
                ktdf.data_transfer from %A[%m0 + %m1, %n0 + %n1] size [1, 64] to %res_lx_A[%m1, %n1] size [1, 64] : memref<?x?xf16, "DRAM">, memref<?x?xf16, "LX">
                ktdf.data_transfer from %B[%m0 + %m1, %n0 + %n1] size [1, 64] to %res_lx_B[%m1, %n1] size [1, 64] : memref<?x?xf16, "DRAM">, memref<?x?xf16, "LX">
              }
            }
            ktdf.stage_yield
          }

          ktdf.stage depend_in(%t1) depend_out(%t2) {
            %res_sfp_A = memref.alloc() {Addr = v0} : memref<64xf16, "SFP_REG">
            %res_sfp_B = memref.alloc() {Addr = v1} : memref<64xf16, "SFP_REG">
            %res_sfp_out = memref.alloc() {Addr = v2} : memref<64xf16, "SFP_REG">              
            scf.for %m1 = %c0 to %Cm step %c1 { ktdf.loop parallel } {
              scf.for %n1 = %c0 to %Cn step %c64 { ktdf.loop parallel } {
                ktdf.pipeline depth(1) {
                  %t11, %t12, %t13 = ktdf.private {
                    %tok11 = ktdf.create_token : !ktdf.token
                    %tok12 = ktdf.create_token : !ktdf.token
                    %tok13 = ktdf.create_token : !ktdf.token
                    ktdf.private_yield %tok11, %tok12, %tok13
                      : !ktdf.token, !ktdf.token, !ktdf.token
                  }

                  ktdf.stage depend_in() depend_out(%t11) {
                    ktdf.data_transfer from %res_lx_A[%m1, %n1] size [1, 64] to %fifo_slot_a size [64] : memref<?x?xf16, "LX">, !ktdf.fifo.slot<"lxlu-to-sfp", 64xf16>
                    ktdf.data_transfer from %res_lx_B[%m1, %n1] size [1, 64] to %fifo_slot_b size [64] : memref<?x?xf16, "LX">, !ktdf.fifo.slot<"lxlu-to-sfp", 64xf16>
                    ktdf.stage_yield
                  }

                  ktdf.stage depend_in(%t11) depend_out(%t12) {
                    ktdf.data_transfer from %fifo_slot_a to %res_sfp_A[%c0] size [64] : !ktdf.fifo.slot<"lxlu-to-sfp", 64xf16>, memref<64xf16, "SFP_REG">
                    ktdf.data_transfer from %fifo_slot_b to %res_sfp_B[%c0] size [64] : !ktdf.fifo.slot<"lxlu-to-sfp", 64xf16>, memref<64xf16, "SFP_REG">
                    linalg.add ins(%res_sfp_A, %res_sfp_B : memref<64xf16, "SFP_REG">, memref<64xf16, "SFP_REG">) outs(%res_sfp_out : memref<64xf16, "SFP_REG">)
                    ktdf.data_transfer from %res_sfp_out[%c0] to %fifo_slot_c size [64] : memref<64xf16, "SFP_REG">, !ktdf.fifo.slot<"sfp-to-lxsu", 64xf16>
                    ktdf.stage_yield
                  }

                  ktdf.stage depend_in(%t12) depend_out(%t13) {
                    ktdf.data_transfer from %fifo_slot_c size [64] to %res_lx_C[%m1, %n1] size [1, 64] : !ktdf.fifo.slot<"sfp-to-lxsu", 64xf16>, memref<?x?xf16, "LX">
                    ktdf.stage_yield
                  }
                }
              }
            }
            ktdf.stage_yield
          }

          ktdf.stage depend_in(%t2) depend_out(%t3) {
            scf.for %m1 = %c0 to %Cm step %c1 { ktdf.loop parallel } {
              scf.for %n1 = %c0 to %Cn step %c64 { ktdf.loop parallel } {
                ktdf.data_transfer from %res_lx_C[%m1, %n1] size [1, 64] to %C[%m0 + %m1, %n0 + %n1] size [1, 64] : memref<?x?xf16, "LX">, memref<?x?xf16, "DRAM">
              }
            }
            ktdf.stage_yield
          }
        }

        memref.dealloc %lx_A0 : memref<?x?xf16, "LX">
        memref.dealloc %lx_A1 : memref<?x?xf16, "LX">
        memref.dealloc %lx_B0 : memref<?x?xf16, "LX">
        memref.dealloc %lx_B1 : memref<?x?xf16, "LX">
        memref.dealloc %lx_C0 : memref<?x?xf16, "LX">
        memref.dealloc %lx_C1 : memref<?x?xf16, "LX">
      }
    }

    return
  }
}
```

## Step-11: Lowering from nested pipelines into schedule-tree like IR.

The goal of this pass is to lower pipelines and stage constructs appropriately into schedule-tree like IR encompassing explicit synchronization primitives in addition to data movement operations from agen dialect and compute operations from vectorchain dialect.

```mlir 
module {
  func.func @local_schedule_1_db(%i: index, %M: index, %N: index, %Cm: index, %Cn: index, %tile_size: index) {
    %A_start_address = arith.constant 1024 : index
    %B_start_address = arith.constant 12288 : index
    %C_start_address = arith.constant 18432 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index

    %id = ktdp.get_compute_tile_id : index
    %start_row = arith.muli %id, %tile_size : index
    %start_row_i = arith.addi %start_row, %i : index

    %A_view = ktdp.construct_memory_view %A_start_address, sizes: [96, 64], strides: [64, 1] {
      coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 95 >= 0, d1 >= 0, -d1 + 63 >= 0)>,
      memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<96x64xf16>

    %B_view = ktdp.construct_memory_view %B_start_address, sizes: [96, 64], strides: [64, 1] {
      coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 95 >= 0, d1 >= 0, -d1 + 63 >= 0)>,
      memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<96x64xf16>

    %C_view = ktdp.construct_memory_view %C_start_address, sizes: [96, 64], strides: [64, 1] {
      coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 95 >= 0, d1 >= 0, -d1 + 63 >= 0)>,
      memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<96x64xf16>

    %offset = arith.muli %start_row_i, %c64 : index
    
    // Memory space cast: propagate ktdf memory space from construct_memory_view
    %A_cast = memref.memory_space_cast %A_view : memref<96x64xf16> to memref<96x64xf16, #ktdf.memory_space<HBM>>
    %B_cast = memref.memory_space_cast %B_view : memref<96x64xf16> to memref<96x64xf16, #ktdf.memory_space<HBM>>
    %C_cast = memref.memory_space_cast %C_view : memref<96x64xf16> to memref<96x64xf16, #ktdf.memory_space<HBM>>
    
    %A = memref.reinterpret_cast %A_cast to offset: [%offset], sizes: [1, 64], strides: [64, 1] : memref<96x64xf16, #ktdf.memory_space<HBM>> to memref<1x64xf16, #ktdf.memory_space<HBM>>
    %B = memref.reinterpret_cast %B_cast to offset: [%offset], sizes: [1, 64], strides: [64, 1] : memref<96x64xf16, #ktdf.memory_space<HBM>> to memref<1x64xf16, #ktdf.memory_space<HBM>>
    %C = memref.reinterpret_cast %C_cast to offset: [%offset], sizes: [1, 64], strides: [64, 1] : memref<96x64xf16, #ktdf.memory_space<HBM>> to memref<1x64xf16, #ktdf.memory_space<HBM>>

    %lx_A0 = memref.alloc(%Cm, %Cn) {Addr: A0} : memref<?x?xf16, "LX">
    %lx_A1 = memref.alloc(%Cm, %Cn) {Addr: A1} : memref<?x?xf16, "LX">
    %lx_B0 = memref.alloc(%Cm, %Cn) {Addr: B0} : memref<?x?xf16, "LX">
    %lx_B1 = memref.alloc(%Cm, %Cn) {Addr: B1} : memref<?x?xf16, "LX">
    %lx_C0 = memref.alloc(%Cm, %Cn) {Addr: C0} : memref<?x?xf16, "LX">
    %lx_C1 = memref.alloc(%Cm, %Cn) {Addr: C1} : memref<?x?xf16, "LX">
   
        
    scf.for %m0 = %c0 to %M step %Cm {
      scf.for %n0 = %c0 to %N step %Cn {

        %phase = ktdf.buffer_phase (%m0, %n0) : index
        %res_lx_A = ktdf.select_memref %phase [%lx_A0, %lx_A1] : index -> memref<?x?xf16, "LX">
        %res_lx_B = ktdf.select_memref %phase [%lx_B0, %lx_B1] : index -> memref<?x?xf16, "LX">
        %res_lx_C = ktdf.select_memref %phase [%lx_C0, %lx_C1] : index -> memref<?x?xf16, "LX">

        scf.for %m1 = %c0 to %Cm step %c1 { ktdf.loop parallel } {
          scf.for %n1 = %c0 to %Cn step %c64 { ktdf.loop parallel } {
            ktdf.data_transfer from %A[%m0 + %m1, %n0 + %n1] size [1, 64] to %res_lx_A[%m1, %n1] size [1, 64] : memref<?x?xf16, "DRAM">, memref<?x?xf16, "LX">
            ktdf.data_transfer from %B[%m0 + %m1, %n0 + %n1] size [1, 64] to %res_lx_B[%m1, %n1] size [1, 64] : memref<?x?xf16, "DRAM">, memref<?x?xf16, "LX">
          }
        }

        // (alias) ktdf.sendrecv %l3lu <-> %lxlu
        ktdf.Signal %l3lu -> %lxlu //non-blocking at sender, blocking at receiver
        ktdf.Signal %lxlu -> %l3lu //non-blocking at sender, blocking at receiver

        %res_sfp_A = memref.alloc() {Addr = v0} : memref<64xf16, "SFP_REG">
        %res_sfp_B = memref.alloc() {Addr = v1} : memref<64xf16, "SFP_REG">
        %res_sfp_out = memref.alloc() {Addr = v2} : memref<64xf16, "SFP_REG">              
        scf.for %m1 = %c0 to %Cm step %c1 { ktdf.loop parallel } {
          scf.for %n1 = %c0 to %Cn step %c64 { ktdf.loop parallel } {
            %fifo_slot_a, %fifo_slot_b = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"lxlu-to-sfp", 64xf16>,
                                                                !ktdf.fifo.slot<"lxlu-to-sfp", 64xf16>
            %fifo_slot_c = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"sfp-to-lxsu", 64xf16>

            ktdf.data_transfer from %res_lx_A[%m1, %n1] size [1, 64] to %fifo_slot_a size [64] : memref<?x?xf16, "LX">, !ktdf.fifo.slot<"lxlu-to-sfp", 64xf16>
            ktdf.data_transfer from %res_lx_B[%m1, %n1] size [1, 64] to %fifo_slot_b size [64] : memref<?x?xf16, "LX">, !ktdf.fifo.slot<"lxlu-to-sfp", 64xf16>

            ktdf.data_transfer from %fifo_slot_a to %res_sfp_A[%c0] size [64] : !ktdf.fifo.slot<"lxlu-to-sfp", 64xf16>, memref<64xf16, "SFP_REG">
            ktdf.data_transfer from %fifo_slot_b to %res_sfp_B[%c0] size [64] : !ktdf.fifo.slot<"lxlu-to-sfp", 64xf16>, memref<64xf16, "SFP_REG">

            linalg.add ins(%res_sfp_A, %res_sfp_B : memref<64xf16, "SFP_REG">, memref<64xf16, "SFP_REG">) outs(%res_sfp_out : memref<64xf16, "SFP_REG">)

            ktdf.data_transfer from %res_sfp_out[%c0] to %fifo_slot_c size [64] : memref<64xf16, "SFP_REG">, !ktdf.fifo.slot<"sfp-to-lxsu", 64xf16>
            ktdf.data_transfer from %fifo_slot_c size [64] to %res_lx_C[%m1, %n1] size [1, 64] : !ktdf.fifo.slot<"sfp-to-lxsu", 64xf16>, memref<?x?xf16, "LX">
            }
          }
        }

        
        // (alias) ktdf.sendrecv %lxsu <-> %l3su
        ktdf.Signal %lxsu -> %l3su //non-blocking at sender, blocking at receiver
        ktdf.Signal %l3su -> %lxsu //non-blocking at sender, blocking at receiver
        
        scf.for %m1 = %c0 to %Cm step %c1 { ktdf.loop parallel } {
          scf.for %n1 = %c0 to %Cn step %c64 { ktdf.loop parallel } {
            ktdf.data_transfer from %res_lx_C[%m1, %n1] size [1, 64] to %C[%m0 + %m1, %n0 + %n1] size [1, 64] : memref<?x?xf16, "LX">, memref<?x?xf16, "DRAM">
          }
        }
        
        memref.dealloc %lx_A0 : memref<?x?xf16, "LX">
        memref.dealloc %lx_A1 : memref<?x?xf16, "LX">
        memref.dealloc %lx_B0 : memref<?x?xf16, "LX">
        memref.dealloc %lx_B1 : memref<?x?xf16, "LX">
        memref.dealloc %lx_C0 : memref<?x?xf16, "LX">
        memref.dealloc %lx_C1 : memref<?x?xf16, "LX">
      }
    }

    return
  }
}
```

## Step-11: Lowering remaining ktdf operations into agen/vectorchain.



## Step-12: Generating DFIR - per unit decoupled programs.
