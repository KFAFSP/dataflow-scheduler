// CAUTION: Be careful when updating this test to make sure SSA results from ktdf.private are reusing the appropriate FileCheck matches.

// RUN: scheduler --stage-coarsening %s | FileCheck %s

// CHECK: #[[$ATTR_0:.+]] = affine_set<(d0, d1) : (d0 >= 0, -d0 + 95 >= 0, d1 >= 0, -d1 + 63 >= 0)>

// CHECK-LABEL:   func.func @local_schedule_1(
// CHECK-SAME:                                %[[VAL_0:.*]]: index) {
// CHECK-NEXT:     %[[VAL_1:.*]] = arith.constant 0 : index
// CHECK-NEXT:     %[[VAL_2:.*]] = arith.constant 1 : index
// CHECK-NEXT:     %[[VAL_3:.*]] = arith.constant 64 : index
// CHECK-NEXT:     %[[VAL_4:.*]] = arith.constant 32 : index
// CHECK-NEXT:     %[[VAL_5:.*]] = arith.constant 1024 : index
// CHECK-NEXT:     %[[VAL_6:.*]] = arith.constant 12288 : index
// CHECK-NEXT:     %[[VAL_7:.*]] = arith.constant 18432 : index
// CHECK-NEXT:     %[[VAL_8:.*]] = ktdp.get_compute_tile_id : index
// CHECK-NEXT:     %[[VAL_9:.*]] = arith.muli %[[VAL_8]], %[[VAL_4]] : index
// CHECK-NEXT:     %[[VAL_10:.*]] = arith.addi %[[VAL_9]], %[[VAL_0]] : index
// CHECK-NEXT:     %[[VAL_11:.*]] = ktdp.construct_memory_view %[[VAL_5]], sizes: [96, 64], strides: [64, 1] {coordinate_set = #[[$ATTR_0]], memory_space = {{.*}}<HBM>} : memref<96x64xf16, {{.*}}<HBM>>
// CHECK-NEXT:     %[[VAL_12:.*]] = ktdp.construct_memory_view %[[VAL_6]], sizes: [96, 64], strides: [64, 1] {coordinate_set = #[[$ATTR_0]], memory_space = {{.*}}<HBM>} : memref<96x64xf16, {{.*}}<HBM>>
// CHECK-NEXT:     %[[VAL_13:.*]] = ktdp.construct_memory_view %[[VAL_7]], sizes: [96, 64], strides: [64, 1] {coordinate_set = #[[$ATTR_0]], memory_space = {{.*}}<HBM>} : memref<96x64xf16, {{.*}}<HBM>>
// CHECK-NEXT:     %[[VAL_14:.*]] = arith.muli %[[VAL_10]], %[[VAL_3]] : index
// CHECK-NEXT:     %[[VAL_15:.*]] = memref.reinterpret_cast %[[VAL_11]] to offset: {{\[}}%[[VAL_14]]], sizes: [1, 64], strides: [64, 1] : memref<96x64xf16, {{.*}}<HBM>> to memref<1x64xf16, strided<[64, 1], offset: ?>, {{.*}}<HBM>>
// CHECK-NEXT:     %[[VAL_16:.*]] = memref.reinterpret_cast %[[VAL_12]] to offset: {{\[}}%[[VAL_14]]], sizes: [1, 64], strides: [64, 1] : memref<96x64xf16, {{.*}}<HBM>> to memref<1x64xf16, strided<[64, 1], offset: ?>, {{.*}}<HBM>>
// CHECK-NEXT:     %[[VAL_17:.*]] = memref.reinterpret_cast %[[VAL_13]] to offset: {{\[}}%[[VAL_14]]], sizes: [1, 64], strides: [64, 1] : memref<96x64xf16, {{.*}}<HBM>> to memref<1x64xf16, strided<[64, 1], offset: ?>, {{.*}}<HBM>>
// CHECK-NEXT:     ktdf.pipeline {
// CHECK-NEXT:       %[[VAL_18:.*]]:5 = ktdf.private -> (memref<1x1x64xf16, {{.*}}<LX>>, memref<1x1x64xf16, {{.*}}<LX>>, memref<1x1x64xf16, {{.*}}<LX>>, !ktdf.token, !ktdf.token) {
// CHECK-NEXT:         %[[VAL_19:.*]] = memref.alloc() : memref<1x1x64xf16, {{.*}}<LX>>
// CHECK-NEXT:         %[[VAL_20:.*]] = memref.alloc() : memref<1x1x64xf16, {{.*}}<LX>>
// CHECK-NEXT:         %[[VAL_21:.*]] = memref.alloc() : memref<1x1x64xf16, {{.*}}<LX>>
// CHECK-NEXT:         %[[VAL_22:.*]] = ktdf.create_token : !ktdf.token
// CHECK-NEXT:         %[[VAL_23:.*]] = ktdf.create_token : !ktdf.token
// CHECK-NEXT:         ktdf.private_yield %[[VAL_19]], %[[VAL_20]], %[[VAL_21]], %[[VAL_22]], %[[VAL_23]] : memref<1x1x64xf16, {{.*}}<LX>>, memref<1x1x64xf16, {{.*}}<LX>>, memref<1x1x64xf16, {{.*}}<LX>>, !ktdf.token, !ktdf.token
// CHECK-NEXT:       }
// CHECK-NEXT:       ktdf.stage depends_in(none) depends_out(%[[VAL_18]]#3 : !ktdf.token) {
// CHECK-NEXT:         scf.for %[[VAL_25:.*]] = %[[VAL_1]] to %[[VAL_2]] step %[[VAL_2]] {
// CHECK-NEXT:           scf.for %[[VAL_26:.*]] = %[[VAL_1]] to %[[VAL_3]] step %[[VAL_3]] {
// CHECK-NEXT:             %[[VAL_27:.*]] = arith.subi %[[VAL_25]], %[[VAL_1]] : index
// CHECK-NEXT:             %[[VAL_28:.*]] = arith.divsi %[[VAL_27]], %[[VAL_2]] : index
// CHECK-NEXT:             %[[VAL_29:.*]] = arith.subi %[[VAL_26]], %[[VAL_1]] : index
// CHECK-NEXT:             %[[VAL_30:.*]] = arith.divsi %[[VAL_29]], %[[VAL_3]] : index
// CHECK-NEXT:             ktdf.data_transfer from %[[VAL_15]]{{\[}}%[[VAL_25]], %[[VAL_26]]] size [1, 64] to %[[VAL_18]]#0{{\[}}%[[VAL_28]], %[[VAL_30]], %[[VAL_1]]] size [64] : memref<1x64xf16, strided<[64, 1], offset: ?>, {{.*}}<HBM>>, memref<1x1x64xf16, {{.*}}<LX>>
// CHECK-NEXT:             %[[VAL_31:.*]] = arith.subi %[[VAL_25]], %[[VAL_1]] : index
// CHECK-NEXT:             %[[VAL_32:.*]] = arith.divsi %[[VAL_31]], %[[VAL_2]] : index
// CHECK-NEXT:             %[[VAL_33:.*]] = arith.subi %[[VAL_26]], %[[VAL_1]] : index
// CHECK-NEXT:             %[[VAL_34:.*]] = arith.divsi %[[VAL_33]], %[[VAL_3]] : index
// CHECK-NEXT:             ktdf.data_transfer from %[[VAL_16]]{{\[}}%[[VAL_25]], %[[VAL_26]]] size [1, 64] to %[[VAL_18]]#1{{\[}}%[[VAL_32]], %[[VAL_34]], %[[VAL_1]]] size [64] : memref<1x64xf16, strided<[64, 1], offset: ?>, {{.*}}<HBM>>, memref<1x1x64xf16, {{.*}}<LX>>
// CHECK-NEXT:           } {loop_type = #ktdf.loop_type<parallel_loop>}
// CHECK-NEXT:         } {loop_type = #ktdf.loop_type<parallel_loop>}
// CHECK-NEXT:       }
// CHECK-NEXT:       ktdf.stage depends_in(%[[VAL_18]]#3 : !ktdf.token) depends_out(%[[VAL_18]]#4 : !ktdf.token) {
// CHECK-NEXT:         scf.for %[[VAL_36:.*]] = %[[VAL_1]] to %[[VAL_2]] step %[[VAL_2]] {
// CHECK-NEXT:           scf.for %[[VAL_37:.*]] = %[[VAL_1]] to %[[VAL_3]] step %[[VAL_3]] {
// CHECK-NEXT:             ktdf.pipeline {
// CHECK-NEXT:               %[[VAL_38:.*]]:5 = ktdf.private -> (memref<64xf16, {{.*}}<SFP_REG>>, memref<64xf16, {{.*}}<SFP_REG>>, memref<64xf16, {{.*}}<SFP_REG>>, !ktdf.token, !ktdf.token) {
// CHECK-NEXT:                 %[[VAL_39:.*]] = memref.alloc() : memref<64xf16, {{.*}}<SFP_REG>>
// CHECK-NEXT:                 %[[VAL_40:.*]] = memref.alloc() : memref<64xf16, {{.*}}<SFP_REG>>
// CHECK-NEXT:                 %[[VAL_41:.*]] = memref.alloc() : memref<64xf16, {{.*}}<SFP_REG>>
// CHECK-NEXT:                 %[[VAL_42:.*]] = ktdf.create_token : !ktdf.token
// CHECK-NEXT:                 %[[VAL_43:.*]] = ktdf.create_token : !ktdf.token
// CHECK-NEXT:                 ktdf.private_yield %[[VAL_39]], %[[VAL_40]], %[[VAL_41]], %[[VAL_42]], %[[VAL_43]] : memref<64xf16, {{.*}}<SFP_REG>>, memref<64xf16, {{.*}}<SFP_REG>>, memref<64xf16, {{.*}}<SFP_REG>>, !ktdf.token, !ktdf.token
// CHECK-NEXT:               }
// CHECK-NEXT:               ktdf.stage depends_in(none) depends_out(%[[VAL_38]]#3 : !ktdf.token) {
// CHECK-NEXT:                 %[[VAL_45:.*]] = arith.subi %[[VAL_36]], %[[VAL_1]] : index
// CHECK-NEXT:                 %[[VAL_46:.*]] = arith.divsi %[[VAL_45]], %[[VAL_2]] : index
// CHECK-NEXT:                 %[[VAL_47:.*]] = arith.subi %[[VAL_37]], %[[VAL_1]] : index
// CHECK-NEXT:                 %[[VAL_48:.*]] = arith.divsi %[[VAL_47]], %[[VAL_3]] : index
// CHECK-NEXT:                 ktdf.data_transfer from %[[VAL_18]]#0{{\[}}%[[VAL_46]], %[[VAL_48]], %[[VAL_1]]] size [64] to %[[VAL_38]]#0{{\[}}%[[VAL_1]]] size [64] : memref<1x1x64xf16, {{.*}}<LX>>, memref<64xf16, {{.*}}<SFP_REG>>
// CHECK-NEXT:                 %[[VAL_49:.*]] = arith.subi %[[VAL_36]], %[[VAL_1]] : index
// CHECK-NEXT:                 %[[VAL_50:.*]] = arith.divsi %[[VAL_49]], %[[VAL_2]] : index
// CHECK-NEXT:                 %[[VAL_51:.*]] = arith.subi %[[VAL_37]], %[[VAL_1]] : index
// CHECK-NEXT:                 %[[VAL_52:.*]] = arith.divsi %[[VAL_51]], %[[VAL_3]] : index
// CHECK-NEXT:                 ktdf.data_transfer from %[[VAL_18]]#1{{\[}}%[[VAL_50]], %[[VAL_52]], %[[VAL_1]]] size [64] to %[[VAL_38]]#1{{\[}}%[[VAL_1]]] size [64] : memref<1x1x64xf16, {{.*}}<LX>>, memref<64xf16, {{.*}}<SFP_REG>>
// CHECK-NEXT:               }
// CHECK-NEXT:               ktdf.stage depends_in(%[[VAL_38]]#3 : !ktdf.token) depends_out(%[[VAL_38]]#4 : !ktdf.token) {
// CHECK-NEXT:                 linalg.add ins(%[[VAL_38]]#0, %[[VAL_38]]#1 : memref<64xf16, {{.*}}<SFP_REG>>, memref<64xf16, {{.*}}<SFP_REG>>) outs(%[[VAL_38]]#2 : memref<64xf16, {{.*}}<SFP_REG>>)
// CHECK-NEXT:               }
// CHECK-NEXT:               ktdf.stage depends_in(%[[VAL_38]]#4 : !ktdf.token) depends_out(none) {
// CHECK-NEXT:                 %[[VAL_55:.*]] = arith.subi %[[VAL_36]], %[[VAL_1]] : index
// CHECK-NEXT:                 %[[VAL_56:.*]] = arith.divsi %[[VAL_55]], %[[VAL_2]] : index
// CHECK-NEXT:                 %[[VAL_57:.*]] = arith.subi %[[VAL_37]], %[[VAL_1]] : index
// CHECK-NEXT:                 %[[VAL_58:.*]] = arith.divsi %[[VAL_57]], %[[VAL_3]] : index
// CHECK-NEXT:                 ktdf.data_transfer from %[[VAL_38]]#2{{\[}}%[[VAL_1]]] size [64] to %[[VAL_18]]#2{{\[}}%[[VAL_56]], %[[VAL_58]], %[[VAL_1]]] size [64] : memref<64xf16, {{.*}}<SFP_REG>>, memref<1x1x64xf16, {{.*}}<LX>>
// CHECK-NEXT:               }
// CHECK-NEXT:             }
// CHECK-NEXT:           } {loop_type = #ktdf.loop_type<parallel_loop>}
// CHECK-NEXT:         } {loop_type = #ktdf.loop_type<parallel_loop>}
// CHECK-NEXT:       }
// CHECK-NEXT:       ktdf.stage depends_in(%[[VAL_18]]#4 : !ktdf.token) depends_out(none) {
// CHECK-NEXT:         scf.for %[[VAL_60:.*]] = %[[VAL_1]] to %[[VAL_2]] step %[[VAL_2]] {
// CHECK-NEXT:           scf.for %[[VAL_61:.*]] = %[[VAL_1]] to %[[VAL_3]] step %[[VAL_3]] {
// CHECK-NEXT:             %[[VAL_62:.*]] = arith.subi %[[VAL_60]], %[[VAL_1]] : index
// CHECK-NEXT:             %[[VAL_63:.*]] = arith.divsi %[[VAL_62]], %[[VAL_2]] : index
// CHECK-NEXT:             %[[VAL_64:.*]] = arith.subi %[[VAL_61]], %[[VAL_1]] : index
// CHECK-NEXT:             %[[VAL_65:.*]] = arith.divsi %[[VAL_64]], %[[VAL_3]] : index
// CHECK-NEXT:             ktdf.data_transfer from %[[VAL_18]]#2{{\[}}%[[VAL_63]], %[[VAL_65]], %[[VAL_1]]] size [64] to %[[VAL_17]]{{\[}}%[[VAL_60]], %[[VAL_61]]] size [1, 64] : memref<1x1x64xf16, {{.*}}<LX>>, memref<1x64xf16, strided<[64, 1], offset: ?>, {{.*}}<HBM>>
// CHECK-NEXT:           } {loop_type = #ktdf.loop_type<parallel_loop>}
// CHECK-NEXT:         } {loop_type = #ktdf.loop_type<parallel_loop>}
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }



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

        scf.for %m = %c0 to %c1 step %c1 {
          scf.for %n = %c0 to %c64 step %c64 {
           ktdf.pipeline  {
              %res_lx_A, %res_lx_B, %res_lx_C, %res_sfp_A, %res_sfp_B, %res_sfp_out,
              %res_t1, %res_t2, %res_t3, %res_t4, %res_t5 = ktdf.private -> (
                memref<64xf16, #ktdf.memory_space<LX>>, memref<64xf16, #ktdf.memory_space<LX>>, memref<64xf16, #ktdf.memory_space<LX>>,
                memref<64xf16, #ktdf.memory_space<SFP_REG>>, memref<64xf16, #ktdf.memory_space<SFP_REG>>, memref<64xf16, #ktdf.memory_space<SFP_REG>>,
                !ktdf.token, !ktdf.token, !ktdf.token, !ktdf.token, !ktdf.token
              ) {
                %lx_A    = memref.alloc() : memref<64xf16, #ktdf.memory_space<LX>>
                %lx_B    = memref.alloc() : memref<64xf16, #ktdf.memory_space<LX>>
                %lx_C    = memref.alloc() : memref<64xf16, #ktdf.memory_space<LX>>
                %sfp_A   = memref.alloc() : memref<64xf16, #ktdf.memory_space<SFP_REG>>
                %sfp_B   = memref.alloc() : memref<64xf16, #ktdf.memory_space<SFP_REG>>
                %sfp_out = memref.alloc() : memref<64xf16, #ktdf.memory_space<SFP_REG>>
                %t1 = ktdf.create_token : !ktdf.token
                %t2 = ktdf.create_token : !ktdf.token
                %t3 = ktdf.create_token : !ktdf.token
                %t4 = ktdf.create_token : !ktdf.token
                %t5 = ktdf.create_token : !ktdf.token
                ktdf.private_yield %lx_A, %lx_B, %lx_C, %sfp_A, %sfp_B, %sfp_out,
                                   %t1, %t2, %t3, %t4, %t5
                    : memref<64xf16, #ktdf.memory_space<LX>>, memref<64xf16, #ktdf.memory_space<LX>>, memref<64xf16, #ktdf.memory_space<LX>>,
                      memref<64xf16, #ktdf.memory_space<SFP_REG>>, memref<64xf16, #ktdf.memory_space<SFP_REG>>, memref<64xf16, #ktdf.memory_space<SFP_REG>>,
                      !ktdf.token, !ktdf.token, !ktdf.token, !ktdf.token, !ktdf.token
              }

              ktdf.stage depends_in(none) depends_out(%res_t1 : !ktdf.token) {
                ktdf.data_transfer from %A[%m, %n] size [1, 64] to %res_lx_A[%c0] size [64] : memref<1x64xf16, strided<[64, 1], offset: ?>, #ktdf.memory_space<HBM>>, memref<64xf16, #ktdf.memory_space<LX>>
                ktdf.data_transfer from %B[%m, %n] size [1, 64] to %res_lx_B[%c0] size [64] : memref<1x64xf16, strided<[64, 1], offset: ?>, #ktdf.memory_space<HBM>>, memref<64xf16, #ktdf.memory_space<LX>>
              }

              ktdf.stage depends_in(%res_t1 : !ktdf.token) depends_out(%res_t2 : !ktdf.token) {
                ktdf.data_transfer from %res_lx_A[%c0] size [64] to %res_sfp_A[%c0] size [64] : memref<64xf16, #ktdf.memory_space<LX>>, memref<64xf16, #ktdf.memory_space<SFP_REG>>
                ktdf.data_transfer from %res_lx_B[%c0] size [64] to %res_sfp_B[%c0] size [64] : memref<64xf16, #ktdf.memory_space<LX>>, memref<64xf16, #ktdf.memory_space<SFP_REG>>
              }

              ktdf.stage depends_in(%res_t2 : !ktdf.token) depends_out(%res_t3 : !ktdf.token) {
                linalg.add ins(%res_sfp_A, %res_sfp_B : memref<64xf16, #ktdf.memory_space<SFP_REG>>, memref<64xf16, #ktdf.memory_space<SFP_REG>>) outs(%res_sfp_out : memref<64xf16, #ktdf.memory_space<SFP_REG>>)
              }

              ktdf.stage depends_in(%res_t3 : !ktdf.token) depends_out(%res_t4 : !ktdf.token) {
                ktdf.data_transfer from %res_sfp_out[%c0] size [64] to %res_lx_C[%c0] size [64] : memref<64xf16, #ktdf.memory_space<SFP_REG>>, memref<64xf16, #ktdf.memory_space<LX>>
              }

              ktdf.stage depends_in(%res_t4 : !ktdf.token) depends_out(%res_t5 : !ktdf.token) {
                ktdf.data_transfer from %res_lx_C[%c0] size [64] to %C[%m, %n] size [1, 64] : memref<64xf16, #ktdf.memory_space<LX>>, memref<1x64xf16, strided<[64, 1], offset: ?>, #ktdf.memory_space<HBM>>
              }
            } // ktdf.pipeline
          } {loop_type = #ktdf.loop_type<parallel_loop>} // scf.for n parallel
        } {loop_type = #ktdf.loop_type<parallel_loop>} // scf.for m parallel
        return
    }
}
