#set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 95 >= 0, d1 >= 0, -d1 + 63 >= 0)>
module {
  func.func @local_schedule_1(%arg0: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    %c64 = arith.constant 64 : index
    %c96 = arith.constant 96 : index
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
    ktdf.pipeline {
      %7:5 = ktdf.private -> (memref<96x4x16xf16, #ktdf.memory_space<LX>>, memref<96x4x16xf16, #ktdf.memory_space<LX>>, memref<96x4x16xf16, #ktdf.memory_space<LX>>, !ktdf.token, !ktdf.token) {
        %alloc = memref.alloc() : memref<96x4x16xf16, #ktdf.memory_space<LX>>
        %alloc_0 = memref.alloc() : memref<96x4x16xf16, #ktdf.memory_space<LX>>
        %alloc_1 = memref.alloc() : memref<96x4x16xf16, #ktdf.memory_space<LX>>
        %8 = ktdf.create_token : !ktdf.token
        %9 = ktdf.create_token : !ktdf.token
        ktdf.private_yield %alloc, %alloc_0, %alloc_1, %8, %9 : memref<96x4x16xf16, #ktdf.memory_space<LX>>, memref<96x4x16xf16, #ktdf.memory_space<LX>>, memref<96x4x16xf16, #ktdf.memory_space<LX>>, !ktdf.token, !ktdf.token
      }
      ktdf.stage depends_in(none) depends_out(%7#3 : !ktdf.token) {
        scf.for %arg1 = %c0 to %c96 step %c1 {
          scf.for %arg2 = %c0 to %c64 step %c16 {
            %8 = arith.subi %arg1, %c0 : index
            %9 = arith.divsi %8, %c1 : index
            %10 = arith.subi %arg2, %c0 : index
            %11 = arith.divsi %10, %c16 : index
            ktdf.data_transfer from %3[%arg1, %arg2] size [1, 16] to %7#0[%9, %11, %c0] size [16] : memref<96x64xf16, #ktdf.memory_space<HBM>>, memref<96x4x16xf16, #ktdf.memory_space<LX>>
            %12 = arith.subi %arg1, %c0 : index
            %13 = arith.divsi %12, %c1 : index
            %14 = arith.subi %arg2, %c0 : index
            %15 = arith.divsi %14, %c16 : index
            ktdf.data_transfer from %4[%arg1, %arg2] size [1, 16] to %7#1[%13, %15, %c0] size [16] : memref<96x64xf16, #ktdf.memory_space<HBM>>, memref<96x4x16xf16, #ktdf.memory_space<LX>>
          } {loop_type = #ktdf.loop_type<parallel_loop>}
        } {loop_type = #ktdf.loop_type<parallel_loop>}
      }
      ktdf.stage depends_in(%7#3 : !ktdf.token) depends_out(%7#4 : !ktdf.token) {
        scf.for %arg1 = %c0 to %c96 step %c1 {
          scf.for %arg2 = %c0 to %c64 step %c16 {
            ktdf.pipeline {
              %8:5 = ktdf.private -> (memref<16xf16, #ktdf.memory_space<SFP_REG>>, memref<16xf16, #ktdf.memory_space<SFP_REG>>, memref<16xf16, #ktdf.memory_space<SFP_REG>>, !ktdf.token, !ktdf.token) {
                %alloc = memref.alloc() : memref<16xf16, #ktdf.memory_space<SFP_REG>>
                %alloc_0 = memref.alloc() : memref<16xf16, #ktdf.memory_space<SFP_REG>>
                %alloc_1 = memref.alloc() : memref<16xf16, #ktdf.memory_space<SFP_REG>>
                %9 = ktdf.create_token : !ktdf.token
                %10 = ktdf.create_token : !ktdf.token
                ktdf.private_yield %alloc, %alloc_0, %alloc_1, %9, %10 : memref<16xf16, #ktdf.memory_space<SFP_REG>>, memref<16xf16, #ktdf.memory_space<SFP_REG>>, memref<16xf16, #ktdf.memory_space<SFP_REG>>, !ktdf.token, !ktdf.token
              }
              ktdf.stage depends_in(none) depends_out(%8#3 : !ktdf.token) {
                %9 = arith.subi %arg1, %c0 : index
                %10 = arith.divsi %9, %c1 : index
                %11 = arith.subi %arg2, %c0 : index
                %12 = arith.divsi %11, %c16 : index
                ktdf.data_transfer from %7#0[%10, %12, %c0] size [16] to %8#0[%c0] size [16] : memref<96x4x16xf16, #ktdf.memory_space<LX>>, memref<16xf16, #ktdf.memory_space<SFP_REG>>
                %13 = arith.subi %arg1, %c0 : index
                %14 = arith.divsi %13, %c1 : index
                %15 = arith.subi %arg2, %c0 : index
                %16 = arith.divsi %15, %c16 : index
                ktdf.data_transfer from %7#1[%14, %16, %c0] size [16] to %8#1[%c0] size [16] : memref<96x4x16xf16, #ktdf.memory_space<LX>>, memref<16xf16, #ktdf.memory_space<SFP_REG>>
              }
              ktdf.stage depends_in(%8#3 : !ktdf.token) depends_out(%8#4 : !ktdf.token) {
                linalg.add ins(%8#0, %8#1 : memref<16xf16, #ktdf.memory_space<SFP_REG>>, memref<16xf16, #ktdf.memory_space<SFP_REG>>) outs(%8#2 : memref<16xf16, #ktdf.memory_space<SFP_REG>>)
              }
              ktdf.stage depends_in(%8#4 : !ktdf.token) depends_out(none) {
                %9 = arith.subi %arg1, %c0 : index
                %10 = arith.divsi %9, %c1 : index
                %11 = arith.subi %arg2, %c0 : index
                %12 = arith.divsi %11, %c16 : index
                ktdf.data_transfer from %8#2 size [16] to %7#2[%10, %12, %c0] size [16] : memref<16xf16, #ktdf.memory_space<SFP_REG>>, memref<96x4x16xf16, #ktdf.memory_space<LX>>
              }
            }
          } {loop_type = #ktdf.loop_type<parallel_loop>}
        } {loop_type = #ktdf.loop_type<parallel_loop>}
      }
      ktdf.stage depends_in(%7#4 : !ktdf.token) depends_out(none) {
        scf.for %arg1 = %c0 to %c96 step %c1 {
          scf.for %arg2 = %c0 to %c64 step %c16 {
            %8 = arith.subi %arg1, %c0 : index
            %9 = arith.divsi %8, %c1 : index
            %10 = arith.subi %arg2, %c0 : index
            %11 = arith.divsi %10, %c16 : index
            ktdf.data_transfer from %7#2[%9, %11, %c0] size [16] to %5[%arg1, %arg2] size [1, 16] : memref<96x4x16xf16, #ktdf.memory_space<LX>>, memref<96x64xf16, #ktdf.memory_space<HBM>>
          } {loop_type = #ktdf.loop_type<parallel_loop>}
        } {loop_type = #ktdf.loop_type<parallel_loop>}
      }
    }
    return
  }
}

