#map = affine_map<(d0) -> (d0)>
#set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 95 >= 0, d1 >= 0, -d1 + 63 >= 0)>
module {
  func.func @partial_expansion_reverse_test(%arg0: index) {
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %c3 = arith.constant 3 : index
    %c1024 = arith.constant 1024 : index
    %c12288 = arith.constant 12288 : index
    %c18432 = arith.constant 18432 : index
    %0 = ktdp.get_compute_tile_id : index
    %1 = arith.muli %0, %c3 : index
    %2 = arith.addi %1, %arg0 : index
    %c0 = arith.constant 0 : index
    %c1_0 = arith.constant 1 : index
    %c1_1 = arith.constant 1 : index
    %c0_2 = arith.constant 0 : index
    %c64_3 = arith.constant 64 : index
    %c64_4 = arith.constant 64 : index
    %3 = ktdp.construct_memory_view %c1024, sizes: [96, 64], strides: [64, 1] {coordinate_set = #set, memory_space = #ktdp.spyre_memory_space<HBM>} : memref<96x64xf16, #ktdf.memory_space<HBM>>
    %c64_5 = arith.constant 64 : index
    %4 = arith.muli %2, %c64_5 : index
    %reinterpret_cast = memref.reinterpret_cast %3 to offset: [%4], sizes: [1, 64], strides: [64, 1] : memref<96x64xf16, #ktdf.memory_space<HBM>> to memref<1x64xf16, strided<[64, 1], offset: ?>, #ktdf.memory_space<HBM>>
    %5 = ktdp.construct_memory_view %c12288, sizes: [96, 64], strides: [64, 1] {coordinate_set = #set, memory_space = #ktdp.spyre_memory_space<HBM>} : memref<96x64xf16, #ktdf.memory_space<HBM>>
    %c64_6 = arith.constant 64 : index
    %6 = arith.muli %2, %c64_6 : index
    %reinterpret_cast_7 = memref.reinterpret_cast %5 to offset: [%6], sizes: [1, 64], strides: [64, 1] : memref<96x64xf16, #ktdf.memory_space<HBM>> to memref<1x64xf16, strided<[64, 1], offset: ?>, #ktdf.memory_space<HBM>>
    %7 = ktdp.construct_memory_view %c18432, sizes: [96, 64], strides: [64, 1] {coordinate_set = #set, memory_space = #ktdp.spyre_memory_space<HBM>} : memref<96x64xf16, #ktdf.memory_space<HBM>>
    %c64_8 = arith.constant 64 : index
    %8 = arith.muli %2, %c64_8 : index
    %reinterpret_cast_9 = memref.reinterpret_cast %7 to offset: [%8], sizes: [1, 64], strides: [64, 1] : memref<96x64xf16, #ktdf.memory_space<HBM>> to memref<1x64xf16, strided<[64, 1], offset: ?>, #ktdf.memory_space<HBM>>
    scf.for %arg1 = %c0 to %c1_0 step %c1_1 {
      scf.for %arg2 = %c0_2 to %c64_3 step %c64_4 {
        ktdf.pipeline {
          %9:10 = ktdf.private -> (memref<1x64xf16, #ktdf.memory_space<LX>>, memref<1x64xf16, #ktdf.memory_space<LX>>, memref<1x64xf16, #ktdf.memory_space<LX>>, !ktdf.fifo.slot<"lxlu-to-sfp", 64xf16>, !ktdf.fifo.slot<"lxlu-to-sfp", 64xf16>, !ktdf.fifo.slot<"sfp-to-lxsu", 64xf16>, !ktdf.token, !ktdf.token, !ktdf.token, !ktdf.token) {
            %LX_A = memref.alloc() : memref<1x64xf16, #ktdf.memory_space<LX>>
            %LX_B = memref.alloc() : memref<1x64xf16, #ktdf.memory_space<LX>>
            %LX_C = memref.alloc() : memref<1x64xf16, #ktdf.memory_space<LX>>
            %10:2 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"lxlu-to-sfp", 64xf16>, !ktdf.fifo.slot<"lxlu-to-sfp", 64xf16>
            %11 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"sfp-to-lxsu", 64xf16>
            %12 = ktdf.create_token : !ktdf.token
            %13 = ktdf.create_token : !ktdf.token
            %14 = ktdf.create_token : !ktdf.token
            %15 = ktdf.create_token : !ktdf.token
            ktdf.private_yield %LX_A, %LX_B, %LX_C, %10#0, %10#1, %11, %12, %13, %14, %15 : memref<1x64xf16, #ktdf.memory_space<LX>>, memref<1x64xf16, #ktdf.memory_space<LX>>, memref<1x64xf16, #ktdf.memory_space<LX>>, !ktdf.fifo.slot<"lxlu-to-sfp", 64xf16>, !ktdf.fifo.slot<"lxlu-to-sfp", 64xf16>, !ktdf.fifo.slot<"sfp-to-lxsu", 64xf16>, !ktdf.token, !ktdf.token, !ktdf.token, !ktdf.token
          }
          ktdf.stage depends_in(none) depends_out(%9#6 : !ktdf.token) {
            ktdf.data_transfer from %reinterpret_cast[%arg1, %arg2] size [1, 64] to %9#0 size [64] : memref<1x64xf16, strided<[64, 1], offset: ?>, #ktdf.memory_space<HBM>>, memref<1x64xf16, #ktdf.memory_space<LX>>
            ktdf.data_transfer from %reinterpret_cast_7[%arg1, %arg2] size [1, 64] to %9#1 size [64] : memref<1x64xf16, strided<[64, 1], offset: ?>, #ktdf.memory_space<HBM>>, memref<1x64xf16, #ktdf.memory_space<LX>>
          }
          ktdf.stage depends_in(%9#6 : !ktdf.token) depends_out(%9#7 : !ktdf.token) {
            ktdf.data_transfer from %9#0[%arg1, %arg2] size [1, 64] to %9#3 size [64] : memref<1x64xf16, #ktdf.memory_space<LX>>, !ktdf.fifo.slot<"lxlu-to-sfp", 64xf16>
            ktdf.data_transfer from %9#1[%arg1, %arg2] size [1, 64] to %9#4 size [64] : memref<1x64xf16, #ktdf.memory_space<LX>>, !ktdf.fifo.slot<"lxlu-to-sfp", 64xf16>
          }
          ktdf.stage depends_in(%9#7 : !ktdf.token) depends_out(%9#8 : !ktdf.token) {
            %10 = ktdf.read_from_fifo %9#3 : <"lxlu-to-sfp", 64xf16> -> tensor<64xf16>
            %11 = ktdf.read_from_fifo %9#4 : <"lxlu-to-sfp", 64xf16> -> tensor<64xf16>
            %12 = tensor.empty() : tensor<64xf16>
            %13 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%10, %11 : tensor<64xf16>, tensor<64xf16>) outs(%12 : tensor<64xf16>) {
            ^bb0(%in: f16, %in_10: f16, %out: f16):
              %14 = arith.addf %in, %in_10 : f16
              linalg.yield %14 : f16
            } -> tensor<64xf16>
            ktdf.write_to_fifo %13, %9#5 : tensor<64xf16>, <"sfp-to-lxsu", 64xf16>
          } {applicable_units = #ktdf.applicable_units<[SFP]>}
          ktdf.stage depends_in(%9#8 : !ktdf.token) depends_out(%9#9 : !ktdf.token) {
            ktdf.data_transfer from %9#5 size [64] to %9#2[%arg1, %arg2] size [1, 64] : !ktdf.fifo.slot<"sfp-to-lxsu", 64xf16>, memref<1x64xf16, #ktdf.memory_space<LX>>
          }
          ktdf.stage depends_in(%9#9 : !ktdf.token) depends_out(none) {
            ktdf.data_transfer from %9#2 size [64] to %reinterpret_cast_9[%arg1, %arg2] size [1, 64] : memref<1x64xf16, #ktdf.memory_space<LX>>, memref<1x64xf16, strided<[64, 1], offset: ?>, #ktdf.memory_space<HBM>>
          }
        }
      } {loop_type = #ktdf.loop_type<parallel_loop>}
    } {loop_type = #ktdf.loop_type<parallel_loop>}
    return
  }
}