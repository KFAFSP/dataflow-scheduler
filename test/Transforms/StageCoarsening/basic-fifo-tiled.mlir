// CAUTION: Be careful when updating this test to make sure SSA results from ktdf.private are reusing the appropriate FileCheck matches.

// RUN: scheduler --stage-coarsening %s | FileCheck %s


// CHECK: #[[$ATTR_0:.+]] = affine_set<(d0, d1) : (d0 >= 0, -d0 + 95 >= 0, d1 >= 0, -d1 + 63 >= 0)>


// CHECK-LABEL:   func.func @local_schedule_1(
// CHECK-SAME:                                %[[VAL_0:.*]]: index) {
// CHECK-NEXT:     %[[VAL_1:.*]] = arith.constant 0 : index
// CHECK-NEXT:     %[[VAL_2:.*]] = arith.constant 1 : index
// CHECK-NEXT:     %[[VAL_3:.*]] = arith.constant 2 : index
// CHECK-NEXT:     %[[VAL_4:.*]] = arith.constant 64 : index
// CHECK-NEXT:     %[[VAL_5:.*]] = arith.constant 32 : index
// CHECK-NEXT:     %[[VAL_6:.*]] = arith.constant 1024 : index
// CHECK-NEXT:     %[[VAL_7:.*]] = arith.constant 12288 : index
// CHECK-NEXT:     %[[VAL_8:.*]] = arith.constant 18432 : index
// CHECK-NEXT:     %[[VAL_9:.*]] = ktdp.get_compute_tile_id : index
// CHECK-NEXT:     %[[VAL_10:.*]] = arith.muli %[[VAL_9]], %[[VAL_5]] : index
// CHECK-NEXT:     %[[VAL_11:.*]] = arith.addi %[[VAL_10]], %[[VAL_0]] : index
// CHECK-NEXT:     %[[VAL_12:.*]] = ktdp.construct_memory_view %[[VAL_6]], sizes: [96, 64], strides: [64, 1] {coordinate_set = #[[$ATTR_0]], memory_space = {{.*}}<HBM>} : memref<96x64xf16, {{.*}}<HBM>>
// CHECK-NEXT:     %[[VAL_13:.*]] = ktdp.construct_memory_view %[[VAL_7]], sizes: [96, 64], strides: [64, 1] {coordinate_set = #[[$ATTR_0]], memory_space = {{.*}}<HBM>} : memref<96x64xf16, {{.*}}<HBM>>
// CHECK-NEXT:     %[[VAL_14:.*]] = ktdp.construct_memory_view %[[VAL_8]], sizes: [96, 64], strides: [64, 1] {coordinate_set = #[[$ATTR_0]], memory_space = {{.*}}<HBM>} : memref<96x64xf16, {{.*}}<HBM>>
// CHECK-NEXT:     %[[VAL_15:.*]] = arith.muli %[[VAL_11]], %[[VAL_4]] : index
// CHECK-NEXT:     %[[VAL_16:.*]] = memref.reinterpret_cast %[[VAL_12]] to offset: {{\[}}%[[VAL_15]]], sizes: [1, 64], strides: [64, 1] : memref<96x64xf16, {{.*}}<HBM>> to memref<1x64xf16, strided<[64, 1], offset: ?>, {{.*}}<HBM>>
// CHECK-NEXT:     %[[VAL_17:.*]] = memref.reinterpret_cast %[[VAL_13]] to offset: {{\[}}%[[VAL_15]]], sizes: [1, 64], strides: [64, 1] : memref<96x64xf16, {{.*}}<HBM>> to memref<1x64xf16, strided<[64, 1], offset: ?>, {{.*}}<HBM>>
// CHECK-NEXT:     %[[VAL_18:.*]] = memref.reinterpret_cast %[[VAL_14]] to offset: {{\[}}%[[VAL_15]]], sizes: [1, 64], strides: [64, 1] : memref<96x64xf16, {{.*}}<HBM>> to memref<1x64xf16, strided<[64, 1], offset: ?>, {{.*}}<HBM>>
// CHECK-NEXT:     scf.for %[[VAL_19:.*]] = %[[VAL_1]] to %[[VAL_2]] step %[[VAL_3]] {
// CHECK-NEXT:       scf.for %[[VAL_20:.*]] = %[[VAL_1]] to %[[VAL_4]] step %[[VAL_4]] {
// CHECK-NEXT:         ktdf.pipeline {
// CHECK-NEXT:           %[[VAL_21:.*]]:5 = ktdf.private -> (memref<2x1x64xf16, {{.*}}<LX>>, memref<2x1x64xf16, {{.*}}<LX>>, memref<2x1x64xf16, {{.*}}<LX>>, !ktdf.token, !ktdf.token) {
// CHECK-NEXT:             %[[VAL_22:.*]] = memref.alloc() : memref<2x1x64xf16, {{.*}}<LX>>
// CHECK-NEXT:             %[[VAL_23:.*]] = memref.alloc() : memref<2x1x64xf16, {{.*}}<LX>>
// CHECK-NEXT:             %[[VAL_24:.*]] = memref.alloc() : memref<2x1x64xf16, {{.*}}<LX>>
// CHECK-NEXT:             %[[VAL_25:.*]] = ktdf.create_token : !ktdf.token
// CHECK-NEXT:             %[[VAL_26:.*]] = ktdf.create_token : !ktdf.token
// CHECK-NEXT:             ktdf.private_yield %[[VAL_22]], %[[VAL_23]], %[[VAL_24]], %[[VAL_25]], %[[VAL_26]] : memref<2x1x64xf16, {{.*}}<LX>>, memref<2x1x64xf16, {{.*}}<LX>>, memref<2x1x64xf16, {{.*}}<LX>>, !ktdf.token, !ktdf.token
// CHECK-NEXT:           }
// CHECK-NEXT:           ktdf.stage depends_in(none) depends_out(%[[VAL_21]]#3 : !ktdf.token) {
// CHECK-NEXT:             scf.for %[[VAL_28:.*]] = %[[VAL_1]] to %[[VAL_3]] step %[[VAL_2]] {
// CHECK-NEXT:               scf.for %[[VAL_29:.*]] = %[[VAL_1]] to %[[VAL_4]] step %[[VAL_4]] {
// CHECK-NEXT:                 %[[VAL_30:.*]] = arith.addi %[[VAL_19]], %[[VAL_28]] : index
// CHECK-NEXT:                 %[[VAL_31:.*]] = arith.addi %[[VAL_20]], %[[VAL_29]] : index
// CHECK-NEXT:                 %[[VAL_32:.*]] = arith.subi %[[VAL_28]], %[[VAL_1]] : index
// CHECK-NEXT:                 %[[VAL_33:.*]] = arith.divsi %[[VAL_32]], %[[VAL_2]] : index
// CHECK-NEXT:                 %[[VAL_34:.*]] = arith.subi %[[VAL_29]], %[[VAL_1]] : index
// CHECK-NEXT:                 %[[VAL_35:.*]] = arith.divsi %[[VAL_34]], %[[VAL_4]] : index
// CHECK-NEXT:                 ktdf.data_transfer from %[[VAL_16]]{{\[}}%[[VAL_30]], %[[VAL_31]]] size [1, 64] to %[[VAL_21]]#0{{\[}}%[[VAL_33]], %[[VAL_35]], %[[VAL_1]]] size [64] : memref<1x64xf16, strided<[64, 1], offset: ?>, {{.*}}<HBM>>, memref<2x1x64xf16, {{.*}}<LX>>
// CHECK-NEXT:                 %[[VAL_36:.*]] = arith.subi %[[VAL_28]], %[[VAL_1]] : index
// CHECK-NEXT:                 %[[VAL_37:.*]] = arith.divsi %[[VAL_36]], %[[VAL_2]] : index
// CHECK-NEXT:                 %[[VAL_38:.*]] = arith.subi %[[VAL_29]], %[[VAL_1]] : index
// CHECK-NEXT:                 %[[VAL_39:.*]] = arith.divsi %[[VAL_38]], %[[VAL_4]] : index
// CHECK-NEXT:                 ktdf.data_transfer from %[[VAL_17]]{{\[}}%[[VAL_30]], %[[VAL_31]]] size [1, 64] to %[[VAL_21]]#1{{\[}}%[[VAL_37]], %[[VAL_39]], %[[VAL_1]]] size [64] : memref<1x64xf16, strided<[64, 1], offset: ?>, {{.*}}<HBM>>, memref<2x1x64xf16, {{.*}}<LX>>
// CHECK-NEXT:               } {loop_type = {{.*}}<parallel_loop>}
// CHECK-NEXT:             } {loop_type = {{.*}}<parallel_loop>}
// CHECK-NEXT:           }
// CHECK-NEXT:           ktdf.stage depends_in(%[[VAL_21]]#3 : !ktdf.token) depends_out(%[[VAL_21]]#4 : !ktdf.token) {
// CHECK-NEXT:             scf.for %[[VAL_41:.*]] = %[[VAL_1]] to %[[VAL_3]] step %[[VAL_2]] {
// CHECK-NEXT:               scf.for %[[VAL_42:.*]] = %[[VAL_1]] to %[[VAL_4]] step %[[VAL_4]] {
// CHECK-NEXT:                 ktdf.pipeline {
// CHECK-NEXT:                   %[[VAL_43:.*]]:8 = ktdf.private -> (memref<64xf16, {{.*}}<SFP_REG>>, memref<64xf16, {{.*}}<SFP_REG>>, memref<64xf16, {{.*}}<SFP_REG>>, !ktdf.fifo.slot<"lxlu-to-sfp", 64xf16>, !ktdf.fifo.slot<"lxlu-to-sfp", 64xf16>, !ktdf.fifo.slot<"sfp-to-lxsu", 64xf16>, !ktdf.token, !ktdf.token) {
// CHECK-NEXT:                     %[[VAL_44:.*]] = memref.alloc() : memref<64xf16, {{.*}}<SFP_REG>>
// CHECK-NEXT:                     %[[VAL_45:.*]] = memref.alloc() : memref<64xf16, {{.*}}<SFP_REG>>
// CHECK-NEXT:                     %[[VAL_46:.*]] = memref.alloc() : memref<64xf16, {{.*}}<SFP_REG>>
// CHECK-NEXT:                     %[[VAL_47:.*]]:2 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"lxlu-to-sfp", 64xf16>, !ktdf.fifo.slot<"lxlu-to-sfp", 64xf16>
// CHECK-NEXT:                     %[[VAL_48:.*]] = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"sfp-to-lxsu", 64xf16>
// CHECK-NEXT:                     %[[VAL_49:.*]] = ktdf.create_token : !ktdf.token
// CHECK-NEXT:                     %[[VAL_50:.*]] = ktdf.create_token : !ktdf.token
// CHECK-NEXT:                     ktdf.private_yield %[[VAL_44]], %[[VAL_45]], %[[VAL_46]], %[[VAL_47]]#0, %[[VAL_47]]#1, %[[VAL_48]], %[[VAL_49]], %[[VAL_50]] : memref<64xf16, {{.*}}<SFP_REG>>, memref<64xf16, {{.*}}<SFP_REG>>, memref<64xf16, {{.*}}<SFP_REG>>, !ktdf.fifo.slot<"lxlu-to-sfp", 64xf16>, !ktdf.fifo.slot<"lxlu-to-sfp", 64xf16>, !ktdf.fifo.slot<"sfp-to-lxsu", 64xf16>, !ktdf.token, !ktdf.token
// CHECK-NEXT:                   }
// CHECK-NEXT:                   ktdf.stage depends_in(none) depends_out(%[[VAL_43]]#6 : !ktdf.token) {
// CHECK-NEXT:                     %[[VAL_52:.*]] = arith.subi %[[VAL_41]], %[[VAL_1]] : index
// CHECK-NEXT:                     %[[VAL_53:.*]] = arith.divsi %[[VAL_52]], %[[VAL_2]] : index
// CHECK-NEXT:                     %[[VAL_54:.*]] = arith.subi %[[VAL_42]], %[[VAL_1]] : index
// CHECK-NEXT:                     %[[VAL_55:.*]] = arith.divsi %[[VAL_54]], %[[VAL_4]] : index
// CHECK-NEXT:                     ktdf.data_transfer from %[[VAL_21]]#0{{\[}}%[[VAL_53]], %[[VAL_55]], %[[VAL_1]]] size [64] to %[[VAL_43]]#3 size [64] : memref<2x1x64xf16, {{.*}}<LX>>, !ktdf.fifo.slot<"lxlu-to-sfp", 64xf16>
// CHECK-NEXT:                     %[[VAL_56:.*]] = arith.subi %[[VAL_41]], %[[VAL_1]] : index
// CHECK-NEXT:                     %[[VAL_57:.*]] = arith.divsi %[[VAL_56]], %[[VAL_2]] : index
// CHECK-NEXT:                     %[[VAL_58:.*]] = arith.subi %[[VAL_42]], %[[VAL_1]] : index
// CHECK-NEXT:                     %[[VAL_59:.*]] = arith.divsi %[[VAL_58]], %[[VAL_4]] : index
// CHECK-NEXT:                     ktdf.data_transfer from %[[VAL_21]]#1{{\[}}%[[VAL_57]], %[[VAL_59]], %[[VAL_1]]] size [64] to %[[VAL_43]]#4 size [64] : memref<2x1x64xf16, {{.*}}<LX>>, !ktdf.fifo.slot<"lxlu-to-sfp", 64xf16>
// CHECK-NEXT:                   }
// CHECK-NEXT:                   ktdf.stage depends_in(%[[VAL_43]]#6 : !ktdf.token) depends_out(%[[VAL_43]]#7 : !ktdf.token) {
// CHECK-NEXT:                     ktdf.data_transfer from %[[VAL_43]]#3 to %[[VAL_43]]#0{{\[}}%[[VAL_1]]] size [64] : !ktdf.fifo.slot<"lxlu-to-sfp", 64xf16>, memref<64xf16, {{.*}}<SFP_REG>>
// CHECK-NEXT:                     ktdf.data_transfer from %[[VAL_43]]#4 to %[[VAL_43]]#1{{\[}}%[[VAL_1]]] size [64] : !ktdf.fifo.slot<"lxlu-to-sfp", 64xf16>, memref<64xf16, {{.*}}<SFP_REG>>
// CHECK-NEXT:                     linalg.add ins(%[[VAL_43]]#0, %[[VAL_43]]#1 : memref<64xf16, {{.*}}<SFP_REG>>, memref<64xf16, {{.*}}<SFP_REG>>) outs(%[[VAL_43]]#2 : memref<64xf16, {{.*}}<SFP_REG>>)
// CHECK-NEXT:                     ktdf.data_transfer from %[[VAL_43]]#2{{\[}}%[[VAL_1]]] to %[[VAL_43]]#5 size [64] : memref<64xf16, {{.*}}<SFP_REG>>, !ktdf.fifo.slot<"sfp-to-lxsu", 64xf16>
// CHECK-NEXT:                   }
// CHECK-NEXT:                   ktdf.stage depends_in(%[[VAL_43]]#7 : !ktdf.token) depends_out(none) {
// CHECK-NEXT:                     %[[VAL_62:.*]] = arith.subi %[[VAL_41]], %[[VAL_1]] : index
// CHECK-NEXT:                     %[[VAL_63:.*]] = arith.divsi %[[VAL_62]], %[[VAL_2]] : index
// CHECK-NEXT:                     %[[VAL_64:.*]] = arith.subi %[[VAL_42]], %[[VAL_1]] : index
// CHECK-NEXT:                     %[[VAL_65:.*]] = arith.divsi %[[VAL_64]], %[[VAL_4]] : index
// CHECK-NEXT:                     ktdf.data_transfer from %[[VAL_43]]#5 size [64] to %[[VAL_21]]#2{{\[}}%[[VAL_63]], %[[VAL_65]], %[[VAL_1]]] size [64] : !ktdf.fifo.slot<"sfp-to-lxsu", 64xf16>, memref<2x1x64xf16, {{.*}}<LX>>
// CHECK-NEXT:                   }
// CHECK-NEXT:                 }
// CHECK-NEXT:               } {loop_type = {{.*}}<parallel_loop>}
// CHECK-NEXT:             } {loop_type = {{.*}}<parallel_loop>}
// CHECK-NEXT:           }
// CHECK-NEXT:           ktdf.stage depends_in(%[[VAL_21]]#4 : !ktdf.token) depends_out(none) {
// CHECK-NEXT:             scf.for %[[VAL_67:.*]] = %[[VAL_1]] to %[[VAL_3]] step %[[VAL_2]] {
// CHECK-NEXT:               scf.for %[[VAL_68:.*]] = %[[VAL_1]] to %[[VAL_4]] step %[[VAL_4]] {
// CHECK-NEXT:                 %[[VAL_69:.*]] = arith.addi %[[VAL_19]], %[[VAL_67]] : index
// CHECK-NEXT:                 %[[VAL_70:.*]] = arith.addi %[[VAL_20]], %[[VAL_68]] : index
// CHECK-NEXT:                 %[[VAL_71:.*]] = arith.subi %[[VAL_67]], %[[VAL_1]] : index
// CHECK-NEXT:                 %[[VAL_72:.*]] = arith.divsi %[[VAL_71]], %[[VAL_2]] : index
// CHECK-NEXT:                 %[[VAL_73:.*]] = arith.subi %[[VAL_68]], %[[VAL_1]] : index
// CHECK-NEXT:                 %[[VAL_74:.*]] = arith.divsi %[[VAL_73]], %[[VAL_4]] : index
// CHECK-NEXT:                 ktdf.data_transfer from %[[VAL_21]]#2{{\[}}%[[VAL_72]], %[[VAL_74]], %[[VAL_1]]] size [64] to %[[VAL_18]]{{\[}}%[[VAL_69]], %[[VAL_70]]] size [1, 64] : memref<2x1x64xf16, {{.*}}<LX>>, memref<1x64xf16, strided<[64, 1], offset: ?>, {{.*}}<HBM>>
// CHECK-NEXT:               } {loop_type = {{.*}}<parallel_loop>}
// CHECK-NEXT:             } {loop_type = {{.*}}<parallel_loop>}
// CHECK-NEXT:           }
// CHECK-NEXT:         }
// CHECK-NEXT:       } {loop_type = {{.*}}<parallel_loop>}
// CHECK-NEXT:     } {loop_type = {{.*}}<parallel_loop>}
// CHECK-NEXT:     return
// CHECK-NEXT:   }



module {
    func.func @local_schedule_1(%i: index) {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %c2 = arith.constant 2 : index
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

        // Outer tile loops (m0, n0) - tile size 2x64
        scf.for %m0 = %c0 to %c1 step %c2 {
          scf.for %n0 = %c0 to %c64 step %c64 {
            // Inner loops (m1, n1) - execution granularity
            scf.for %m1 = %c0 to %c2 step %c1 {
              scf.for %n1 = %c0 to %c64 step %c64 {
               ktdf.pipeline  {
                  %res_lx_A, %res_lx_B, %res_lx_C, %res_sfp_A, %res_sfp_B, %res_sfp_out,
                  %res_fifo_slot_a, %res_fifo_slot_b, %res_fifo_slot_c,
                  %res_t1, %res_t2, %res_t3, %res_t4, %res_t5 = ktdf.private -> (
                    memref<64xf16, #ktdf.memory_space<LX>>, memref<64xf16, #ktdf.memory_space<LX>>, memref<64xf16, #ktdf.memory_space<LX>>,
                    memref<64xf16, #ktdf.memory_space<SFP_REG>>, memref<64xf16, #ktdf.memory_space<SFP_REG>>, memref<64xf16, #ktdf.memory_space<SFP_REG>>,
                    !ktdf.fifo.slot<"lxlu-to-sfp", 64xf16>, !ktdf.fifo.slot<"lxlu-to-sfp", 64xf16>,
                    !ktdf.fifo.slot<"sfp-to-lxsu", 64xf16>,
                    !ktdf.token, !ktdf.token, !ktdf.token, !ktdf.token, !ktdf.token
                  ) {
                    %lx_A    = memref.alloc() : memref<64xf16, #ktdf.memory_space<LX>>
                    %lx_B    = memref.alloc() : memref<64xf16, #ktdf.memory_space<LX>>
                    %lx_C    = memref.alloc() : memref<64xf16, #ktdf.memory_space<LX>>
                    %sfp_A   = memref.alloc() : memref<64xf16, #ktdf.memory_space<SFP_REG>>
                    %sfp_B   = memref.alloc() : memref<64xf16, #ktdf.memory_space<SFP_REG>>
                    %sfp_out = memref.alloc() : memref<64xf16, #ktdf.memory_space<SFP_REG>>
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
                        : memref<64xf16, #ktdf.memory_space<LX>>, memref<64xf16, #ktdf.memory_space<LX>>, memref<64xf16, #ktdf.memory_space<LX>>,
                          memref<64xf16, #ktdf.memory_space<SFP_REG>>, memref<64xf16, #ktdf.memory_space<SFP_REG>>, memref<64xf16, #ktdf.memory_space<SFP_REG>>,
                          !ktdf.fifo.slot<"lxlu-to-sfp", 64xf16>, !ktdf.fifo.slot<"lxlu-to-sfp", 64xf16>,
                          !ktdf.fifo.slot<"sfp-to-lxsu", 64xf16>,
                          !ktdf.token, !ktdf.token, !ktdf.token, !ktdf.token, !ktdf.token
                  }

                  ktdf.stage depends_in(none) depends_out(%res_t1 : !ktdf.token) {
                    %m_idx = arith.addi %m0, %m1 : index
                    %n_idx = arith.addi %n0, %n1 : index
                    ktdf.data_transfer from %A[%m_idx, %n_idx] size [1, 64] to %res_lx_A[%c0] size [64] : memref<1x64xf16, strided<[64, 1], offset: ?>, #ktdf.memory_space<HBM>>, memref<64xf16, #ktdf.memory_space<LX>>
                    ktdf.data_transfer from %B[%m_idx, %n_idx] size [1, 64] to %res_lx_B[%c0] size [64] : memref<1x64xf16, strided<[64, 1], offset: ?>, #ktdf.memory_space<HBM>>, memref<64xf16, #ktdf.memory_space<LX>>
                  }

                  ktdf.stage depends_in(%res_t1 : !ktdf.token) depends_out(%res_t2 : !ktdf.token) {
                    ktdf.data_transfer from %res_lx_A[%c0] size [64] to %res_fifo_slot_a size [64] : memref<64xf16, #ktdf.memory_space<LX>>, !ktdf.fifo.slot<"lxlu-to-sfp", 64xf16>
                    ktdf.data_transfer from %res_lx_B[%c0] size [64] to %res_fifo_slot_b size [64] : memref<64xf16, #ktdf.memory_space<LX>>, !ktdf.fifo.slot<"lxlu-to-sfp", 64xf16>
                  }

                  ktdf.stage depends_in(%res_t2 : !ktdf.token) depends_out(%res_t3 : !ktdf.token) {
                    ktdf.data_transfer from %res_fifo_slot_a to %res_sfp_A[%c0] size [64] : !ktdf.fifo.slot<"lxlu-to-sfp", 64xf16>, memref<64xf16, #ktdf.memory_space<SFP_REG>>
                    ktdf.data_transfer from %res_fifo_slot_b to %res_sfp_B[%c0] size [64] : !ktdf.fifo.slot<"lxlu-to-sfp", 64xf16>, memref<64xf16, #ktdf.memory_space<SFP_REG>>
                    linalg.add ins(%res_sfp_A, %res_sfp_B : memref<64xf16, #ktdf.memory_space<SFP_REG>>, memref<64xf16, #ktdf.memory_space<SFP_REG>>) outs(%res_sfp_out : memref<64xf16, #ktdf.memory_space<SFP_REG>>)
                    ktdf.data_transfer from %res_sfp_out[%c0] to %res_fifo_slot_c size [64] : memref<64xf16, #ktdf.memory_space<SFP_REG>>, !ktdf.fifo.slot<"sfp-to-lxsu", 64xf16>
                  }

                  ktdf.stage depends_in(%res_t3 : !ktdf.token) depends_out(%res_t4 : !ktdf.token) {
                    ktdf.data_transfer from %res_fifo_slot_c size [64] to %res_lx_C[%c0] size [64] : !ktdf.fifo.slot<"sfp-to-lxsu", 64xf16>, memref<64xf16, #ktdf.memory_space<LX>>
                  }

                  ktdf.stage depends_in(%res_t4 : !ktdf.token) depends_out(%res_t5 : !ktdf.token) {
                    %m_idx = arith.addi %m0, %m1 : index
                    %n_idx = arith.addi %n0, %n1 : index
                    ktdf.data_transfer from %res_lx_C[%c0] size [64] to %C[%m_idx, %n_idx] size [1, 64] : memref<64xf16, #ktdf.memory_space<LX>>, memref<1x64xf16, strided<[64, 1], offset: ?>, #ktdf.memory_space<HBM>>
                  }
                } // ktdf.pipeline
              } {loop_type = #ktdf.loop_type<parallel_loop>} // scf.for n1 parallel
            } {loop_type = #ktdf.loop_type<parallel_loop>} // scf.for m1 parallel
          } {loop_type = #ktdf.loop_type<parallel_loop>} // scf.for n0 parallel
        } {loop_type = #ktdf.loop_type<parallel_loop>} // scf.for m0 parallel
        return
    }
}