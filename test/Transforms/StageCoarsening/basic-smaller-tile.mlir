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
    scf.for %arg1 = %c0 to %c1 step %7 {
      %9 = arith.muli %8, %c64 : index
      scf.for %arg2 = %c0 to %c64 step %9 {
        %10 = arith.addi %arg1, %7 : index
        %11 = arith.minsi %10, %c1 : index
        %12 = affine.apply #map()[%arg1, %11]
        scf.for %arg3 = %c0 to %12 step %c1 {
          %13 = arith.addi %arg2, %9 : index
          %14 = arith.minsi %13, %c64 : index
          %15 = affine.apply #map1()[%arg2, %14]
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
                ktdf.data_transfer from %16#5 size [64] to %16#2[%c0] size [64] : memref<64xf16, #ktdf.memory_space<SFP_REG>>, memref<64xf16, #ktdf.memory_space<LX>>
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

