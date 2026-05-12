module {
    func.func @local_schedule_1(%i: index) {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %c16 = arith.constant 16 : index
        %c64 = arith.constant 64 : index
        %c96 = arith.constant 96 : index
        %tile_size = arith.constant 32 : index
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
        } : memref<96x64xf16, #ktdf.memory_space<HBM>>

        // Construct a memory view of B from a given address
        %B_view = ktdp.construct_memory_view %B_start_address, sizes: [96, 64], strides: [64, 1] {
            coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 95 >= 0, d1 >= 0, -d1 + 63 >= 0)>,
            memory_space = #ktdp.spyre_memory_space<HBM>
        } : memref<96x64xf16, #ktdf.memory_space<HBM>>

        // Construct a memory view of C from a given address
        %C_view = ktdp.construct_memory_view %C_start_address, sizes: [96, 64], strides: [64, 1] {
            coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 95 >= 0, d1 >= 0, -d1 + 63 >= 0)>,
            memory_space = #ktdp.spyre_memory_space<HBM>
        } : memref<96x64xf16, #ktdf.memory_space<HBM>>

        %offset = arith.muli %start_row_i, %c64 : index
//        %A = memref.reinterpret_cast %A_view to offset: [%offset], sizes: [1,64], strides: [64, 1] : memref<96x64xf16, #ktdf.memory_space<HBM>> to memref<1x64xf16, strided<[64, 1], offset: ?>, #ktdf.memory_space<HBM>>
//        %B = memref.reinterpret_cast %B_view to offset: [%offset], sizes: [1,64], strides: [64, 1] : memref<96x64xf16, #ktdf.memory_space<HBM>> to memref<1x64xf16, strided<[64, 1], offset: ?>, #ktdf.memory_space<HBM>>
//        %C = memref.reinterpret_cast %C_view to offset: [%offset], sizes: [1,64], strides: [64, 1] : memref<96x64xf16, #ktdf.memory_space<HBM>> to memref<1x64xf16, strided<[64, 1], offset: ?>, #ktdf.memory_space<HBM>>

        scf.for %m = %c0 to %c96 step %c1 {
          scf.for %n = %c0 to %c64 step %c16 {
           ktdf.pipeline  {
              %res_lx_A, %res_lx_B, %res_lx_C, %res_sfp_A, %res_sfp_B, %res_sfp_out,
              %res_t1, %res_t2, %res_t3, %res_t4, %res_t5 = ktdf.private -> (
                memref<16xf16, #ktdf.memory_space<LX>>, memref<16xf16, #ktdf.memory_space<LX>>, memref<16xf16, #ktdf.memory_space<LX>>,
                memref<16xf16, #ktdf.memory_space<SFP_REG>>, memref<16xf16, #ktdf.memory_space<SFP_REG>>, memref<16xf16, #ktdf.memory_space<SFP_REG>>,
                !ktdf.token, !ktdf.token, !ktdf.token, !ktdf.token, !ktdf.token
              ) {
                %lx_A    = memref.alloc() : memref<16xf16, #ktdf.memory_space<LX>>
                %lx_B    = memref.alloc() : memref<16xf16, #ktdf.memory_space<LX>>
                %lx_C    = memref.alloc() : memref<16xf16, #ktdf.memory_space<LX>>
                %sfp_A   = memref.alloc() : memref<16xf16, #ktdf.memory_space<SFP_REG>>
                %sfp_B   = memref.alloc() : memref<16xf16, #ktdf.memory_space<SFP_REG>>
                %sfp_out = memref.alloc() : memref<16xf16, #ktdf.memory_space<SFP_REG>>
                %t1 = ktdf.create_token : !ktdf.token
                %t2 = ktdf.create_token : !ktdf.token
                %t3 = ktdf.create_token : !ktdf.token
                %t4 = ktdf.create_token : !ktdf.token
                %t5 = ktdf.create_token : !ktdf.token
                ktdf.private_yield %lx_A, %lx_B, %lx_C, %sfp_A, %sfp_B, %sfp_out,
                                   %t1, %t2, %t3, %t4, %t5
                    : memref<16xf16, #ktdf.memory_space<LX>>, memref<16xf16, #ktdf.memory_space<LX>>, memref<16xf16, #ktdf.memory_space<LX>>,
                      memref<16xf16, #ktdf.memory_space<SFP_REG>>, memref<16xf16, #ktdf.memory_space<SFP_REG>>, memref<16xf16, #ktdf.memory_space<SFP_REG>>,
                      !ktdf.token, !ktdf.token, !ktdf.token, !ktdf.token, !ktdf.token
              }

              ktdf.stage depends_in(none) depends_out(%res_t1 : !ktdf.token) {
                ktdf.data_transfer from %A_view[%m, %n] size [1, 16] to %res_lx_A[%c0] size [16] : memref<96x64xf16, #ktdf.memory_space<HBM>>, memref<16xf16, #ktdf.memory_space<LX>>
                ktdf.data_transfer from %B_view[%m, %n] size [1, 16] to %res_lx_B[%c0] size [16] : memref<96x64xf16, #ktdf.memory_space<HBM>>, memref<16xf16, #ktdf.memory_space<LX>>
              }

              ktdf.stage depends_in(%res_t1 : !ktdf.token) depends_out(%res_t2 : !ktdf.token) {
                ktdf.data_transfer from %res_lx_A[%c0] size [16] to %res_sfp_A[%c0] size [16] : memref<16xf16, #ktdf.memory_space<LX>>, memref<16xf16, #ktdf.memory_space<SFP_REG>>
                ktdf.data_transfer from %res_lx_B[%c0] size [16] to %res_sfp_B[%c0] size [16] : memref<16xf16, #ktdf.memory_space<LX>>, memref<16xf16, #ktdf.memory_space<SFP_REG>>
              }

              ktdf.stage depends_in(%res_t2 : !ktdf.token) depends_out(%res_t3 : !ktdf.token) {
                linalg.add ins(%res_sfp_A, %res_sfp_B : memref<16xf16, #ktdf.memory_space<SFP_REG>>, memref<16xf16, #ktdf.memory_space<SFP_REG>>) outs(%res_sfp_out : memref<16xf16, #ktdf.memory_space<SFP_REG>>)
              }

              ktdf.stage depends_in(%res_t3 : !ktdf.token) depends_out(%res_t4 : !ktdf.token) {
                ktdf.data_transfer from %res_sfp_out size [16] to %res_lx_C[%c0] size [16] : memref<16xf16, #ktdf.memory_space<SFP_REG>>, memref<16xf16, #ktdf.memory_space<LX>>
              }

              ktdf.stage depends_in(%res_t4 : !ktdf.token) depends_out(%res_t5 : !ktdf.token) {
                ktdf.data_transfer from %res_lx_C[%c0] size [16] to %C_view[%m, %n] size [1, 16] : memref<16xf16, #ktdf.memory_space<LX>>, memref<96x64xf16, #ktdf.memory_space<HBM>>
              }
            } // ktdf.pipeline
          } {loop_type = #ktdf.loop_type<parallel_loop>} // scf.for n parallel
        } {loop_type = #ktdf.loop_type<parallel_loop>} // scf.for m parallel
        return
    }
}
