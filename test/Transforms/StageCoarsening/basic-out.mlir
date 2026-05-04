module {
    func.func @local_schedule_1(%i: index) {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %c64 = arith.constant 64 : index
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
        %A = memref.reinterpret_cast %A_view to offset: [%offset], sizes: [1,64], strides: [64, 1] : memref<96x64xf16, #ktdf.memory_space<HBM>> to memref<1x64xf16, strided<[64, 1], offset: ?>, #ktdf.memory_space<HBM>>
        %B = memref.reinterpret_cast %B_view to offset: [%offset], sizes: [1,64], strides: [64, 1] : memref<96x64xf16, #ktdf.memory_space<HBM>> to memref<1x64xf16, strided<[64, 1], offset: ?>, #ktdf.memory_space<HBM>>
        %C = memref.reinterpret_cast %C_view to offset: [%offset], sizes: [1,64], strides: [64, 1] : memref<96x64xf16, #ktdf.memory_space<HBM>> to memref<1x64xf16, strided<[64, 1], offset: ?>, #ktdf.memory_space<HBM>>

        // Outer pipeline - operates at tile granularity
        ktdf.pipeline  {
          %res_lx_A, %res_lx_B, %res_lx_C,
          %res_t1, %res_t5, %res_t6 = ktdf.private -> (
            memref<64xf16, #ktdf.memory_space<LX>>, memref<64xf16, #ktdf.memory_space<LX>>, memref<64xf16, #ktdf.memory_space<LX>>,
            !ktdf.token, !ktdf.token, !ktdf.token
          ) {
            %lx_A    = memref.alloc() : memref<64xf16, #ktdf.memory_space<LX>>
            %lx_B    = memref.alloc() : memref<64xf16, #ktdf.memory_space<LX>>
            %lx_C    = memref.alloc() : memref<64xf16, #ktdf.memory_space<LX>>
            %t1 = ktdf.create_token : !ktdf.token
            %t5 = ktdf.create_token : !ktdf.token
            %t6 = ktdf.create_token : !ktdf.token
            ktdf.private_yield %lx_A, %lx_B, %lx_C,
                               %t1, %t5, %t6
                : memref<64xf16, #ktdf.memory_space<LX>>, memref<64xf16, #ktdf.memory_space<LX>>, memref<64xf16, #ktdf.memory_space<LX>>,
                  !ktdf.token, !ktdf.token, !ktdf.token
          }

          // Stage 0 - Load from HBM to LX (single-stage group, loops sunk into stage)
          ktdf.stage depends_in(none) depends_out(%res_t1 : !ktdf.token) {
            scf.for %m = %c0 to %c1 step %c1 {
              scf.for %n = %c0 to %c64 step %c64 {
                ktdf.data_transfer from %A[%m, %n] size [1, 64] to %res_lx_A[%c0] size [64] : memref<1x64xf16, strided<[64, 1], offset: ?>, #ktdf.memory_space<HBM>>, memref<64xf16, #ktdf.memory_space<LX>>
                ktdf.data_transfer from %B[%m, %n] size [1, 64] to %res_lx_B[%c0] size [64] : memref<1x64xf16, strided<[64, 1], offset: ?>, #ktdf.memory_space<HBM>>, memref<64xf16, #ktdf.memory_space<LX>>
              } {loop_type = #ktdf.loop_type<parallel_loop>}
            } {loop_type = #ktdf.loop_type<parallel_loop>}
          }

          // Stage 5 - Coarsened stage containing nested pipeline for SFP compute stages
          ktdf.stage depends_in(%res_t1 : !ktdf.token) depends_out(%res_t5 : !ktdf.token) {
            scf.for %m = %c0 to %c1 step %c1 {
              scf.for %n = %c0 to %c64 step %c64 {
                // Nested pipeline - operates at hardware execution granularity
                ktdf.pipeline {
                  %res_sfp_A, %res_sfp_B, %res_sfp_out,
                  %res_t2, %res_t3, %res_t4 = ktdf.private -> (
                    memref<64xf16, #ktdf.memory_space<SFP_REG>>, memref<64xf16, #ktdf.memory_space<SFP_REG>>, memref<64xf16, #ktdf.memory_space<SFP_REG>>,
                    !ktdf.token, !ktdf.token, !ktdf.token
                  ) {
                    %sfp_A   = memref.alloc() : memref<64xf16, #ktdf.memory_space<SFP_REG>>
                    %sfp_B   = memref.alloc() : memref<64xf16, #ktdf.memory_space<SFP_REG>>
                    %sfp_out = memref.alloc() : memref<64xf16, #ktdf.memory_space<SFP_REG>>
                    %t2 = ktdf.create_token : !ktdf.token
                    %t3 = ktdf.create_token : !ktdf.token
                    %t4 = ktdf.create_token : !ktdf.token
                    ktdf.private_yield %sfp_A, %sfp_B, %sfp_out,
                                       %t2, %t3, %t4
                        : memref<64xf16, #ktdf.memory_space<SFP_REG>>, memref<64xf16, #ktdf.memory_space<SFP_REG>>, memref<64xf16, #ktdf.memory_space<SFP_REG>>,
                          !ktdf.token, !ktdf.token, !ktdf.token
                  }

                  // Stage 1 - Transfer from LX to SFP
                  ktdf.stage depends_in(none) depends_out(%res_t2 : !ktdf.token) {
                    ktdf.data_transfer from %res_lx_A[%c0] size [64] to %res_sfp_A[%c0] size [64] : memref<64xf16, #ktdf.memory_space<LX>>, memref<64xf16, #ktdf.memory_space<SFP_REG>>
                    ktdf.data_transfer from %res_lx_B[%c0] size [64] to %res_sfp_B[%c0] size [64] : memref<64xf16, #ktdf.memory_space<LX>>, memref<64xf16, #ktdf.memory_space<SFP_REG>>
                  }

                  // Stage 2 - SFP Compute
                  ktdf.stage depends_in(%res_t2 : !ktdf.token) depends_out(%res_t3 : !ktdf.token) {
                    linalg.add ins(%res_sfp_A, %res_sfp_B : memref<64xf16, #ktdf.memory_space<SFP_REG>>, memref<64xf16, #ktdf.memory_space<SFP_REG>>) outs(%res_sfp_out : memref<64xf16, #ktdf.memory_space<SFP_REG>>)
                  }

                  // Stage 3 - Transfer from SFP to LX
                  ktdf.stage depends_in(%res_t3 : !ktdf.token) depends_out(none) {
                    ktdf.data_transfer from %res_sfp_out size [64] to %res_lx_C[%c0] size [64] : memref<64xf16, #ktdf.memory_space<SFP_REG>>, memref<64xf16, #ktdf.memory_space<LX>>
                  }
                } // nested ktdf.pipeline
              } {loop_type = #ktdf.loop_type<parallel_loop>}
            } {loop_type = #ktdf.loop_type<parallel_loop>}
          }

          // Stage 4 - Store from LX to HBM (single-stage group, loops sunk into stage)
          ktdf.stage depends_in(%res_t5 : !ktdf.token) depends_out(none) {
            scf.for %m = %c0 to %c1 step %c1 {
              scf.for %n = %c0 to %c64 step %c64 {
                ktdf.data_transfer from %res_lx_C[%c0] size [64] to %C[%m, %n] size [1, 64] : memref<64xf16, #ktdf.memory_space<LX>>, memref<1x64xf16, strided<[64, 1], offset: ?>, #ktdf.memory_space<HBM>>
              } {loop_type = #ktdf.loop_type<parallel_loop>}
            } {loop_type = #ktdf.loop_type<parallel_loop>}
          }
        } // outer ktdf.pipeline
        return
    }
}
