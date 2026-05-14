// RUN: scheduler --broadcast-promotion %s -allow-unregistered-dialect | FileCheck %s

// The transfer's index depends on an arith.addi computed inside the
// donor stage. The arith.addi must be cloned into the sibling pipeline's
// stage body alongside the transfer.

// CHECK-LABEL: func.func @ssa_dep_hoist
// CHECK: ktdf.pipeline {
// CHECK:   ktdf.stage depends_in(none) depends_out(none) {
// CHECK:     %[[IDX:.*]] = arith.addi
// CHECK:     ktdf.data_transfer from %{{.*}}[%[[IDX]]]
// CHECK:   }
// CHECK: }

module {
  func.func @ssa_dep_hoist(%A: memref<?xf16, #ktdf.memory_space<HBM>>,
                           %offset: index, %M: index, %N: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    scf.for %m = %c0 to %M step %c1 {
      scf.for %n = %c0 to %N step %c1 {
        ktdf.pipeline {
          %lx_A, %t = ktdf.private -> (memref<64xf16, #ktdf.memory_space<LX>>, !ktdf.token) {
            %a = memref.alloc() : memref<64xf16, #ktdf.memory_space<LX>>
            %k = ktdf.create_token : !ktdf.token
            ktdf.private_yield %a, %k : memref<64xf16, #ktdf.memory_space<LX>>, !ktdf.token
          }
          ktdf.stage depends_in(none) depends_out(%t : !ktdf.token) {
            %idx = arith.addi %m, %offset : index
            ktdf.data_transfer from %A[%idx] size [64] to %lx_A[%c0] size [64]
              : memref<?xf16, #ktdf.memory_space<HBM>>, memref<64xf16, #ktdf.memory_space<LX>>
          }
          ktdf.stage depends_in(%t : !ktdf.token) depends_out(none) {
            "test.use"(%lx_A) : (memref<64xf16, #ktdf.memory_space<LX>>) -> ()
          }
        }
      }
    }
    return
  }
}
