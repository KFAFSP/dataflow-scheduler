// RUN: scheduler --stage-coarsening %s | FileCheck %s


// CHECK: #[[$ATTR_0:.+]] = affine_map<()[s0, s1] -> (-s0 + s1)>
// CHECK: #[[$ATTR_1:.+]] = affine_map<()[s0, s1] -> ((-s0 + s1) ceildiv 64)>
// CHECK: #[[$ATTR_2:.+]] = affine_map<(d0)[s0] -> (d0 * 64 + s0)>
// CHECK: #[[$ATTR_3:.+]] = affine_map<(d0)[s0] -> (d0 + s0)>
// CHECK: #[[$ATTR_4:.+]] = affine_set<(d0, d1) : (d0 >= 0, -d0 + 95 >= 0, d1 >= 0, -d1 + 63 >= 0)>


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
// CHECK-NEXT:     %[[VAL_11:.*]] = ktdp.construct_memory_view %[[VAL_5]], sizes: [96, 64], strides: [64, 1] {coordinate_set = #[[$ATTR_4]], memory_space = {{.*}}<HBM>} : memref<96x64xf16, {{.*}}<HBM>>
// CHECK-NEXT:     %[[VAL_12:.*]] = ktdp.construct_memory_view %[[VAL_6]], sizes: [96, 64], strides: [64, 1] {coordinate_set = #[[$ATTR_4]], memory_space = {{.*}}<HBM>} : memref<96x64xf16, {{.*}}<HBM>>
// CHECK-NEXT:     %[[VAL_13:.*]] = ktdp.construct_memory_view %[[VAL_7]], sizes: [96, 64], strides: [64, 1] {coordinate_set = #[[$ATTR_4]], memory_space = {{.*}}<HBM>} : memref<96x64xf16, {{.*}}<HBM>>
// CHECK-NEXT:     %[[VAL_14:.*]] = arith.muli %[[VAL_10]], %[[VAL_3]] : index
// CHECK-NEXT:     %[[VAL_15:.*]] = memref.reinterpret_cast %[[VAL_11]] to offset: {{\[}}%[[VAL_14]]], sizes: [1, 64], strides: [64, 1] : memref<96x64xf16, {{.*}}<HBM>> to memref<1x64xf16, strided<[64, 1], offset: ?>, {{.*}}<HBM>>
// CHECK-NEXT:     %[[VAL_16:.*]] = memref.reinterpret_cast %[[VAL_12]] to offset: {{\[}}%[[VAL_14]]], sizes: [1, 64], strides: [64, 1] : memref<96x64xf16, {{.*}}<HBM>> to memref<1x64xf16, strided<[64, 1], offset: ?>, {{.*}}<HBM>>
// CHECK-NEXT:     %[[VAL_17:.*]] = memref.reinterpret_cast %[[VAL_13]] to offset: {{\[}}%[[VAL_14]]], sizes: [1, 64], strides: [64, 1] : memref<96x64xf16, {{.*}}<HBM>> to memref<1x64xf16, strided<[64, 1], offset: ?>, {{.*}}<HBM>>
// CHECK-NEXT:     %[[VAL_18:.*]] = ktdf.create_tile_size {divisibility = 1 : index, min_value = 1 : index} : index
// CHECK-NEXT:     %[[VAL_19:.*]] = ktdf.create_tile_size {divisibility = 1 : index, min_value = 1 : index} : index
// CHECK-NEXT:     %[[VAL_20:.*]] = arith.muli %[[VAL_19]], %[[VAL_3]] : index
// CHECK-NEXT:     scf.for %[[VAL_21:.*]] = %[[VAL_1]] to %[[VAL_2]] step %[[VAL_18]] {
// CHECK-NEXT:       %[[VAL_22:.*]] = arith.addi %[[VAL_21]], %[[VAL_18]] : index
// CHECK-NEXT:       %[[VAL_23:.*]] = arith.minsi %[[VAL_22]], %[[VAL_2]] : index
// CHECK-NEXT:       %[[VAL_24:.*]] = affine.apply #[[$ATTR_0]](){{\[}}%[[VAL_21]], %[[VAL_23]]]
// CHECK-NEXT:       scf.for %[[VAL_25:.*]] = %[[VAL_1]] to %[[VAL_3]] step %[[VAL_20]] {
// CHECK-NEXT:         %[[VAL_26:.*]] = arith.addi %[[VAL_25]], %[[VAL_20]] : index
// CHECK-NEXT:         %[[VAL_27:.*]] = arith.minsi %[[VAL_26]], %[[VAL_3]] : index
// CHECK-NEXT:         %[[VAL_28:.*]] = affine.apply #[[$ATTR_1]](){{\[}}%[[VAL_25]], %[[VAL_27]]]
// CHECK-NEXT:         %[[VAL_29:.*]] = arith.subi %[[VAL_24]], %[[VAL_1]] : index
// CHECK-NEXT:         %[[VAL_30:.*]] = arith.ceildivsi %[[VAL_29]], %[[VAL_2]] : index
// CHECK-NEXT:         %[[VAL_31:.*]] = arith.subi %[[VAL_28]], %[[VAL_1]] : index
// CHECK-NEXT:         %[[VAL_32:.*]] = arith.ceildivsi %[[VAL_31]], %[[VAL_2]] : index
// CHECK-NEXT:         %[[VAL_33:.*]] = arith.subi %[[VAL_24]], %[[VAL_1]] : index
// CHECK-NEXT:         %[[VAL_34:.*]] = arith.ceildivsi %[[VAL_33]], %[[VAL_2]] : index
// CHECK-NEXT:         %[[VAL_35:.*]] = arith.subi %[[VAL_28]], %[[VAL_1]] : index
// CHECK-NEXT:         %[[VAL_36:.*]] = arith.ceildivsi %[[VAL_35]], %[[VAL_2]] : index
// CHECK-NEXT:         %[[VAL_37:.*]] = arith.subi %[[VAL_24]], %[[VAL_1]] : index
// CHECK-NEXT:         %[[VAL_38:.*]] = arith.ceildivsi %[[VAL_37]], %[[VAL_2]] : index
// CHECK-NEXT:         %[[VAL_39:.*]] = arith.subi %[[VAL_28]], %[[VAL_1]] : index
// CHECK-NEXT:         %[[VAL_40:.*]] = arith.ceildivsi %[[VAL_39]], %[[VAL_2]] : index
// CHECK-NEXT:         ktdf.pipeline {
// CHECK-NEXT:           %[[VAL_41:.*]]:5 = ktdf.private -> (memref<?x?x64xf16, {{.*}}<LX>>, memref<?x?x64xf16, {{.*}}<LX>>, memref<?x?x64xf16, {{.*}}<LX>>, !ktdf.token, !ktdf.token) {
// CHECK-NEXT:             %[[VAL_42:.*]] = memref.alloc(%[[VAL_30]], %[[VAL_32]]) : memref<?x?x64xf16, {{.*}}<LX>>
// CHECK-NEXT:             %[[VAL_43:.*]] = memref.alloc(%[[VAL_34]], %[[VAL_36]]) : memref<?x?x64xf16, {{.*}}<LX>>
// CHECK-NEXT:             %[[VAL_44:.*]] = memref.alloc(%[[VAL_38]], %[[VAL_40]]) : memref<?x?x64xf16, {{.*}}<LX>>
// CHECK-NEXT:             %[[VAL_45:.*]] = ktdf.create_token : !ktdf.token
// CHECK-NEXT:             %[[VAL_46:.*]] = ktdf.create_token : !ktdf.token
// CHECK-NEXT:             ktdf.private_yield %[[VAL_42]], %[[VAL_43]], %[[VAL_44]], %[[VAL_45]], %[[VAL_46]] : memref<?x?x64xf16, {{.*}}<LX>>, memref<?x?x64xf16, {{.*}}<LX>>, memref<?x?x64xf16, {{.*}}<LX>>, !ktdf.token, !ktdf.token
// CHECK-NEXT:           }
// CHECK-NEXT:           ktdf.stage depends_in(none) depends_out(%[[VAL_47:.*]]#3 : !ktdf.token) {
// CHECK-NEXT:             scf.for %[[VAL_48:.*]] = %[[VAL_1]] to %[[VAL_24]] step %[[VAL_2]] {
// CHECK-NEXT:               scf.for %[[VAL_49:.*]] = %[[VAL_1]] to %[[VAL_28]] step %[[VAL_2]] {
// CHECK-NEXT:                 %[[VAL_50:.*]] = affine.apply #[[$ATTR_2]](%[[VAL_49]]){{\[}}%[[VAL_25]]]
// CHECK-NEXT:                 %[[VAL_51:.*]] = affine.apply #[[$ATTR_3]](%[[VAL_48]]){{\[}}%[[VAL_21]]]
// CHECK-NEXT:                 %[[VAL_52:.*]] = arith.subi %[[VAL_48]], %[[VAL_1]] : index
// CHECK-NEXT:                 %[[VAL_53:.*]] = arith.divsi %[[VAL_52]], %[[VAL_2]] : index
// CHECK-NEXT:                 %[[VAL_54:.*]] = arith.subi %[[VAL_49]], %[[VAL_1]] : index
// CHECK-NEXT:                 %[[VAL_55:.*]] = arith.divsi %[[VAL_54]], %[[VAL_2]] : index
// CHECK-NEXT:                 ktdf.data_transfer from %[[VAL_15]]{{\[}}%[[VAL_51]], %[[VAL_50]]] size [1, 64] to %[[VAL_47]]#0{{\[}}%[[VAL_53]], %[[VAL_55]], %[[VAL_1]]] size [64] : memref<1x64xf16, strided<[64, 1], offset: ?>, {{.*}}<HBM>>, memref<?x?x64xf16, {{.*}}<LX>>
// CHECK-NEXT:                 %[[VAL_56:.*]] = affine.apply #[[$ATTR_2]](%[[VAL_49]]){{\[}}%[[VAL_25]]]
// CHECK-NEXT:                 %[[VAL_57:.*]] = affine.apply #[[$ATTR_3]](%[[VAL_48]]){{\[}}%[[VAL_21]]]
// CHECK-NEXT:                 %[[VAL_58:.*]] = arith.subi %[[VAL_48]], %[[VAL_1]] : index
// CHECK-NEXT:                 %[[VAL_59:.*]] = arith.divsi %[[VAL_58]], %[[VAL_2]] : index
// CHECK-NEXT:                 %[[VAL_60:.*]] = arith.subi %[[VAL_49]], %[[VAL_1]] : index
// CHECK-NEXT:                 %[[VAL_61:.*]] = arith.divsi %[[VAL_60]], %[[VAL_2]] : index
// CHECK-NEXT:                 ktdf.data_transfer from %[[VAL_16]]{{\[}}%[[VAL_57]], %[[VAL_56]]] size [1, 64] to %[[VAL_47]]#1{{\[}}%[[VAL_59]], %[[VAL_61]], %[[VAL_1]]] size [64] : memref<1x64xf16, strided<[64, 1], offset: ?>, {{.*}}<HBM>>, memref<?x?x64xf16, {{.*}}<LX>>
// CHECK-NEXT:               }
// CHECK-NEXT:             }
// CHECK-NEXT:           }
// CHECK-NEXT:           ktdf.stage depends_in(%[[VAL_62:.*]]#3 : !ktdf.token) depends_out(%[[VAL_62]]#4 : !ktdf.token) {
// CHECK-NEXT:             scf.for %[[VAL_63:.*]] = %[[VAL_1]] to %[[VAL_24]] step %[[VAL_2]] {
// CHECK-NEXT:               scf.for %[[VAL_64:.*]] = %[[VAL_1]] to %[[VAL_28]] step %[[VAL_2]] {
// CHECK-NEXT:                 ktdf.pipeline {
// CHECK-NEXT:                   %[[VAL_65:.*]]:5 = ktdf.private -> (memref<64xf16, {{.*}}<SFP_REG>>, memref<64xf16, {{.*}}<SFP_REG>>, memref<64xf16, {{.*}}<SFP_REG>>, !ktdf.token, !ktdf.token) {
// CHECK-NEXT:                     %[[VAL_66:.*]] = memref.alloc() : memref<64xf16, {{.*}}<SFP_REG>>
// CHECK-NEXT:                     %[[VAL_67:.*]] = memref.alloc() : memref<64xf16, {{.*}}<SFP_REG>>
// CHECK-NEXT:                     %[[VAL_68:.*]] = memref.alloc() : memref<64xf16, {{.*}}<SFP_REG>>
// CHECK-NEXT:                     %[[VAL_69:.*]] = ktdf.create_token : !ktdf.token
// CHECK-NEXT:                     %[[VAL_70:.*]] = ktdf.create_token : !ktdf.token
// CHECK-NEXT:                     ktdf.private_yield %[[VAL_66]], %[[VAL_67]], %[[VAL_68]], %[[VAL_69]], %[[VAL_70]] : memref<64xf16, {{.*}}<SFP_REG>>, memref<64xf16, {{.*}}<SFP_REG>>, memref<64xf16, {{.*}}<SFP_REG>>, !ktdf.token, !ktdf.token
// CHECK-NEXT:                   }
// CHECK-NEXT:                   ktdf.stage depends_in(none) depends_out(%[[VAL_71:.*]]#3 : !ktdf.token) {
// CHECK-NEXT:                     %[[VAL_72:.*]] = arith.subi %[[VAL_63]], %[[VAL_1]] : index
// CHECK-NEXT:                     %[[VAL_73:.*]] = arith.divsi %[[VAL_72]], %[[VAL_2]] : index
// CHECK-NEXT:                     %[[VAL_74:.*]] = arith.subi %[[VAL_64]], %[[VAL_1]] : index
// CHECK-NEXT:                     %[[VAL_75:.*]] = arith.divsi %[[VAL_74]], %[[VAL_2]] : index
// CHECK-NEXT:                     ktdf.data_transfer from %[[VAL_62]]#0{{\[}}%[[VAL_73]], %[[VAL_75]], %[[VAL_1]]] size [64] to %[[VAL_71]]#0{{\[}}%[[VAL_1]]] size [64] : memref<?x?x64xf16, {{.*}}<LX>>, memref<64xf16, {{.*}}<SFP_REG>>
// CHECK-NEXT:                     %[[VAL_76:.*]] = arith.subi %[[VAL_63]], %[[VAL_1]] : index
// CHECK-NEXT:                     %[[VAL_77:.*]] = arith.divsi %[[VAL_76]], %[[VAL_2]] : index
// CHECK-NEXT:                     %[[VAL_78:.*]] = arith.subi %[[VAL_64]], %[[VAL_1]] : index
// CHECK-NEXT:                     %[[VAL_79:.*]] = arith.divsi %[[VAL_78]], %[[VAL_2]] : index
// CHECK-NEXT:                     ktdf.data_transfer from %[[VAL_62]]#1{{\[}}%[[VAL_77]], %[[VAL_79]], %[[VAL_1]]] size [64] to %[[VAL_71]]#1{{\[}}%[[VAL_1]]] size [64] : memref<?x?x64xf16, {{.*}}<LX>>, memref<64xf16, {{.*}}<SFP_REG>>
// CHECK-NEXT:                   }
// CHECK-NEXT:                   ktdf.stage depends_in(%[[VAL_80:.*]]#3 : !ktdf.token) depends_out(%[[VAL_80]]#4 : !ktdf.token) {
// CHECK-NEXT:                     linalg.add ins(%[[VAL_80]]#0, %[[VAL_80]]#1 : memref<64xf16, {{.*}}<SFP_REG>>, memref<64xf16, {{.*}}<SFP_REG>>) outs(%[[VAL_80]]#2 : memref<64xf16, {{.*}}<SFP_REG>>)
// CHECK-NEXT:                   }
// CHECK-NEXT:                   ktdf.stage depends_in(%[[VAL_81:.*]]#4 : !ktdf.token) depends_out(none) {
// CHECK-NEXT:                     %[[VAL_82:.*]] = arith.subi %[[VAL_63]], %[[VAL_1]] : index
// CHECK-NEXT:                     %[[VAL_83:.*]] = arith.divsi %[[VAL_82]], %[[VAL_2]] : index
// CHECK-NEXT:                     %[[VAL_84:.*]] = arith.subi %[[VAL_64]], %[[VAL_1]] : index
// CHECK-NEXT:                     %[[VAL_85:.*]] = arith.divsi %[[VAL_84]], %[[VAL_2]] : index
// CHECK-NEXT:                     ktdf.data_transfer from %[[VAL_81]]#2{{\[}}%[[VAL_1]]] size [64] to %[[VAL_62]]#2{{\[}}%[[VAL_83]], %[[VAL_85]], %[[VAL_1]]] size [64] : memref<64xf16, {{.*}}<SFP_REG>>, memref<?x?x64xf16, {{.*}}<LX>>
// CHECK-NEXT:                   }
// CHECK-NEXT:                 }
// CHECK-NEXT:               }
// CHECK-NEXT:             }
// CHECK-NEXT:           }
// CHECK-NEXT:           ktdf.stage depends_in(%[[VAL_86:.*]]#4 : !ktdf.token) depends_out(none) {
// CHECK-NEXT:             scf.for %[[VAL_87:.*]] = %[[VAL_1]] to %[[VAL_24]] step %[[VAL_2]] {
// CHECK-NEXT:               scf.for %[[VAL_88:.*]] = %[[VAL_1]] to %[[VAL_28]] step %[[VAL_2]] {
// CHECK-NEXT:                 %[[VAL_89:.*]] = affine.apply #[[$ATTR_2]](%[[VAL_88]]){{\[}}%[[VAL_25]]]
// CHECK-NEXT:                 %[[VAL_90:.*]] = affine.apply #[[$ATTR_3]](%[[VAL_87]]){{\[}}%[[VAL_21]]]
// CHECK-NEXT:                 %[[VAL_91:.*]] = arith.subi %[[VAL_87]], %[[VAL_1]] : index
// CHECK-NEXT:                 %[[VAL_92:.*]] = arith.divsi %[[VAL_91]], %[[VAL_2]] : index
// CHECK-NEXT:                 %[[VAL_93:.*]] = arith.subi %[[VAL_88]], %[[VAL_1]] : index
// CHECK-NEXT:                 %[[VAL_94:.*]] = arith.divsi %[[VAL_93]], %[[VAL_2]] : index
// CHECK-NEXT:                 ktdf.data_transfer from %[[VAL_86]]#2{{\[}}%[[VAL_92]], %[[VAL_94]], %[[VAL_1]]] size [64] to %[[VAL_17]]{{\[}}%[[VAL_90]], %[[VAL_89]]] size [1, 64] : memref<?x?x64xf16, {{.*}}<LX>>, memref<1x64xf16, strided<[64, 1], offset: ?>, {{.*}}<HBM>>
// CHECK-NEXT:               }
// CHECK-NEXT:             }
// CHECK-NEXT:           }
// CHECK-NEXT:         }
// CHECK-NEXT:       } {loop_type = {{.*}}<parallel_loop>}
// CHECK-NEXT:     } {loop_type = {{.*}}<parallel_loop>}
// CHECK-NEXT:     return
// CHECK-NEXT:   }


#map = affine_map<()[s0, s1] -> (-s0 + s1)>
#map1 = affine_map<()[s0, s1] -> ((-s0 + s1) ceildiv 64)>
#map2 = affine_map<(d0)[s0] -> (d0 * 64 + s0)>
#map3 = affine_map<(d0)[s0] -> (d0 + s0)>
#set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 95 >= 0, d1 >= 0, -d1 + 63 >= 0)>
module {
  func.func @local_schedule_1(%arg0: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %c32 = arith.constant 32 : index
    %c1024 = arith.constant 1024 : index
    %c12288 = arith.constant 12288 : index
    %c18432 = arith.constant 18432 : index
    %0 = ktdp.get_compute_tile_id : index
    %1 = arith.muli %0, %c32 : index
    %2 = arith.addi %1, %arg0 : index
    %3 = ktdp.construct_memory_view %c1024, sizes: [96, 64], strides: [64, 1] {coordinate_set = #set, memory_space = #ktdp.spyre_memory_space<HBM>} : memref<96x64xf16, #ktdf.memory_space<HBM>>
    %4 = ktdp.construct_memory_view %c12288, sizes: [96, 64], strides: [64, 1] {coordinate_set = #set, memory_space = #ktdp.spyre_memory_space<HBM>} : memref<96x64xf16, #ktdf.memory_space<HBM>>
    %5 = ktdp.construct_memory_view %c18432, sizes: [96, 64], strides: [64, 1] {coordinate_set = #set, memory_space = #ktdp.spyre_memory_space<HBM>} : memref<96x64xf16, #ktdf.memory_space<HBM>>
    %6 = arith.muli %2, %c64 : index
    %reinterpret_cast = memref.reinterpret_cast %3 to offset: [%6], sizes: [1, 64], strides: [64, 1] : memref<96x64xf16, #ktdf.memory_space<HBM>> to memref<1x64xf16, strided<[64, 1], offset: ?>, #ktdf.memory_space<HBM>>
    %reinterpret_cast_0 = memref.reinterpret_cast %4 to offset: [%6], sizes: [1, 64], strides: [64, 1] : memref<96x64xf16, #ktdf.memory_space<HBM>> to memref<1x64xf16, strided<[64, 1], offset: ?>, #ktdf.memory_space<HBM>>
    %reinterpret_cast_1 = memref.reinterpret_cast %5 to offset: [%6], sizes: [1, 64], strides: [64, 1] : memref<96x64xf16, #ktdf.memory_space<HBM>> to memref<1x64xf16, strided<[64, 1], offset: ?>, #ktdf.memory_space<HBM>>
    %7 = ktdf.create_tile_size {divisibility = 1 : index, min_value = 1 : index} : index
    %8 = ktdf.create_tile_size {divisibility = 1 : index, min_value = 1 : index} : index
    %9 = arith.muli %8, %c64 : index
    scf.for %arg1 = %c0 to %c1 step %7 {
      %10 = arith.addi %arg1, %7 : index
      %11 = arith.minsi %10, %c1 : index
      %12 = affine.apply #map()[%arg1, %11]
      scf.for %arg2 = %c0 to %c64 step %9 {
        %13 = arith.addi %arg2, %9 : index
        %14 = arith.minsi %13, %c64 : index
        %15 = affine.apply #map1()[%arg2, %14]
        scf.for %arg3 = %c0 to %12 step %c1 {
          scf.for %arg4 = %c0 to %15 step %c1 {
            ktdf.pipeline {
              %16:11 = ktdf.private -> (memref<64xf16, #ktdf.memory_space<LX>>, memref<64xf16, #ktdf.memory_space<LX>>, memref<64xf16, #ktdf.memory_space<LX>>, memref<64xf16, #ktdf.memory_space<SFP_REG>>, memref<64xf16, #ktdf.memory_space<SFP_REG>>, memref<64xf16, #ktdf.memory_space<SFP_REG>>, !ktdf.token, !ktdf.token, !ktdf.token, !ktdf.token, !ktdf.token) {
                %alloc = memref.alloc() : memref<64xf16, #ktdf.memory_space<LX>>
                %alloc_2 = memref.alloc() : memref<64xf16, #ktdf.memory_space<LX>>
                %alloc_3 = memref.alloc() : memref<64xf16, #ktdf.memory_space<LX>>
                %alloc_4 = memref.alloc() : memref<64xf16, #ktdf.memory_space<SFP_REG>>
                %alloc_5 = memref.alloc() : memref<64xf16, #ktdf.memory_space<SFP_REG>>
                %alloc_6 = memref.alloc() : memref<64xf16, #ktdf.memory_space<SFP_REG>>
                %17 = ktdf.create_token : !ktdf.token
                %18 = ktdf.create_token : !ktdf.token
                %19 = ktdf.create_token : !ktdf.token
                %20 = ktdf.create_token : !ktdf.token
                %21 = ktdf.create_token : !ktdf.token
                ktdf.private_yield %alloc, %alloc_2, %alloc_3, %alloc_4, %alloc_5, %alloc_6, %17, %18, %19, %20, %21 : memref<64xf16, #ktdf.memory_space<LX>>, memref<64xf16, #ktdf.memory_space<LX>>, memref<64xf16, #ktdf.memory_space<LX>>, memref<64xf16, #ktdf.memory_space<SFP_REG>>, memref<64xf16, #ktdf.memory_space<SFP_REG>>, memref<64xf16, #ktdf.memory_space<SFP_REG>>, !ktdf.token, !ktdf.token, !ktdf.token, !ktdf.token, !ktdf.token
              }
              ktdf.stage depends_in(none) depends_out(%16#6 : !ktdf.token) {
                %17 = affine.apply #map2(%arg4)[%arg2]
                %18 = affine.apply #map3(%arg3)[%arg1]
                ktdf.data_transfer from %reinterpret_cast[%18, %17] size [1, 64] to %16#0[%c0] size [64] : memref<1x64xf16, strided<[64, 1], offset: ?>, #ktdf.memory_space<HBM>>, memref<64xf16, #ktdf.memory_space<LX>>
                %19 = affine.apply #map2(%arg4)[%arg2]
                %20 = affine.apply #map3(%arg3)[%arg1]
                ktdf.data_transfer from %reinterpret_cast_0[%20, %19] size [1, 64] to %16#1[%c0] size [64] : memref<1x64xf16, strided<[64, 1], offset: ?>, #ktdf.memory_space<HBM>>, memref<64xf16, #ktdf.memory_space<LX>>
              }
              ktdf.stage depends_in(%16#6 : !ktdf.token) depends_out(%16#7 : !ktdf.token) {
                ktdf.data_transfer from %16#0[%c0] size [64] to %16#3[%c0] size [64] : memref<64xf16, #ktdf.memory_space<LX>>, memref<64xf16, #ktdf.memory_space<SFP_REG>>
                ktdf.data_transfer from %16#1[%c0] size [64] to %16#4[%c0] size [64] : memref<64xf16, #ktdf.memory_space<LX>>, memref<64xf16, #ktdf.memory_space<SFP_REG>>
              }
              ktdf.stage depends_in(%16#7 : !ktdf.token) depends_out(%16#8 : !ktdf.token) {
                linalg.add ins(%16#3, %16#4 : memref<64xf16, #ktdf.memory_space<SFP_REG>>, memref<64xf16, #ktdf.memory_space<SFP_REG>>) outs(%16#5 : memref<64xf16, #ktdf.memory_space<SFP_REG>>)
              }
              ktdf.stage depends_in(%16#8 : !ktdf.token) depends_out(%16#9 : !ktdf.token) {
                ktdf.data_transfer from %16#5[%c0] size [64] to %16#2[%c0] size [64] : memref<64xf16, #ktdf.memory_space<SFP_REG>>, memref<64xf16, #ktdf.memory_space<LX>>
              }
              ktdf.stage depends_in(%16#9 : !ktdf.token) depends_out(%16#10 : !ktdf.token) {
                %17 = affine.apply #map2(%arg4)[%arg2]
                %18 = affine.apply #map3(%arg3)[%arg1]
                ktdf.data_transfer from %16#2[%c0] size [64] to %reinterpret_cast_1[%18, %17] size [1, 64] : memref<64xf16, #ktdf.memory_space<LX>>, memref<1x64xf16, strided<[64, 1], offset: ?>, #ktdf.memory_space<HBM>>
              }
            }
          }
        }
      } {loop_type = #ktdf.loop_type<parallel_loop>}
    } {loop_type = #ktdf.loop_type<parallel_loop>}
    return
  }
}

