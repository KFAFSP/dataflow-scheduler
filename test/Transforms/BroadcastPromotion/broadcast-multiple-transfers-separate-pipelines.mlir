// RUN: scheduler --broadcast-promotion %s -allow-unregistered-dialect | FileCheck %s

// Two independent hoistable transfers in one stage should emit two
// separate sibling pipelines (no grouping).

// CHECK-LABEL: func.func @two_hoists
// CHECK: ktdf.pipeline {
// CHECK:   ktdf.stage depends_in(none) depends_out(none) {
// CHECK:     ktdf.data_transfer
// CHECK:   }
// CHECK: }
// CHECK: ktdf.pipeline {
// CHECK:   ktdf.stage depends_in(none) depends_out(none) {
// CHECK:     ktdf.data_transfer
// CHECK:   }
// CHECK: }
// CHECK: scf.for %{{.*}} =

module {
  func.func @two_hoists(%A: memref<?xf16, #ktdf.memory_space<HBM>>,
                        %B: memref<?xf16, #ktdf.memory_space<HBM>>,
                        %M: index, %N: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    scf.for %m = %c0 to %M step %c1 {
      scf.for %n = %c0 to %N step %c1 {
        ktdf.pipeline {
          %lx_A, %lx_B, %t = ktdf.private -> (
              memref<64xf16, #ktdf.memory_space<LX>>,
              memref<64xf16, #ktdf.memory_space<LX>>,
              !ktdf.token) {
            %a = memref.alloc() : memref<64xf16, #ktdf.memory_space<LX>>
            %b = memref.alloc() : memref<64xf16, #ktdf.memory_space<LX>>
            %k = ktdf.create_token : !ktdf.token
            ktdf.private_yield %a, %b, %k
              : memref<64xf16, #ktdf.memory_space<LX>>,
                memref<64xf16, #ktdf.memory_space<LX>>,
                !ktdf.token
          }
          ktdf.stage depends_in(none) depends_out(%t : !ktdf.token) {
            ktdf.data_transfer from %A[%m] size [64] to %lx_A[%c0] size [64]
              : memref<?xf16, #ktdf.memory_space<HBM>>, memref<64xf16, #ktdf.memory_space<LX>>
            ktdf.data_transfer from %B[%m] size [64] to %lx_B[%c0] size [64]
              : memref<?xf16, #ktdf.memory_space<HBM>>, memref<64xf16, #ktdf.memory_space<LX>>
          }
          ktdf.stage depends_in(%t : !ktdf.token) depends_out(none) {
            "test.use"(%lx_A) : (memref<64xf16, #ktdf.memory_space<LX>>) -> ()
            "test.use"(%lx_B) : (memref<64xf16, #ktdf.memory_space<LX>>) -> ()
          }
        }
      }
    }
    return
  }
}
