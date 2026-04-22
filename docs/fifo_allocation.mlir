// After fifo allocation
module {
    func.func @local_schedule_1(%i: index) {
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
        %A = memref.reinterpret_cast %A_view to offset: [%offset], sizes: [1,64], strides: [64, 1] : memref<96x64xf16>, memref<1x64xf16>
        %B = memref.reinterpret_cast %B_view to offset: [%offset], sizes: [1,64], strides: [64, 1] : memref<96x64xf16>, memref<1x64xf16>
        %C = memref.reinterpret_cast %C_view to offset: [%offset], sizes: [1,64], strides: [64, 1] : memref<96x64xf16>, memref<1x64xf16>
        
        scf.for %m = %c0 to %c1 step %c1 {ktdf.loop parallel} {
          scf.for %n = %c0 to %c64 step %c64 {ktdf.loop parallel} {                                                                                                       
           ktdf.pipeline depth(0)  {
              %res_lx_A, %res_lx_B, %res_lx_C, %res_sfp_A, %res_sfp_B, %res_sfp_out,
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
                ktdf.private_yield %lx_A, %lx_B, %lx_C, %sfp_A, %sfp_B, %sfp_out, %fifo_slot_a, %fifo_slot_b, %fifo_slot_c
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
                ktdf.data_transfer from %res_lx_A[%c0] size [64] to %fifo_slot_a size [64] : memref<64xf16, "LX">, !ktdf.fifo.slot<"lxlu-to-sfp", 64xf16>
                ktdf.data_transfer from %res_lx_B[%c0] size [64] to %fifo_slot_b size [64] : memref<64xf16, "LX">, !ktdf.fifo.slot<"lxlu-to-sfp", 64xf16>
                ktdf.stage_yield
              }
    
              ktdf.stage depend_in(%res_t2) depend_out(%res_t3) {
                ktdf.data_transfer from %fifo_slot_a to %res_sfp_A[%c0] size [64] : !ktdf.fifo.slot<"lxlu-to-sfp", 64xf16>, memref<64xf16, "SFP_REG">
                ktdf.data_transfer from %fifo_slot_b to %res_sfp_B[%c0] size [64] : !ktdf.fifo.slot<"lxlu-to-sfp", 64xf16>, memref<64xf16, "SFP_REG">
                linalg.add ins(%res_sfp_A, %res_sfp_B : memref<64xf16, "SFP_REG">, memref<64xf16, "SFP_REG">) outs(%res_sfp_out : memref<64xf16, "SFP_REG">)
                ktdf.data_transfer from %res_sfp_out[%c0] to %fifo_slot_c size [64] : memref<64xf16, "SFP_REG">, !ktdf.fifo.slot<"sfp-to-lxsu", 64xf16>
                ktdf.stage_yield
              }
    
              ktdf.stage depend_in(%res_t3) depend_out(%res_t4) {
                ktdf.data_transfer from %fifo_slot_c size [64] to %res_lx_C[%c0] size [64] : !ktdf.fifo.slot<"sfp-to-lxsu", 64xf16>, memref<64xf16, "LX">
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
