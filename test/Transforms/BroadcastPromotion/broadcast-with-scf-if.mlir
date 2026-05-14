// RUN: scheduler --broadcast-promotion %s -allow-unregistered-dialect | FileCheck %s

// An invariant transfer gated by an scf.if should hoist with the scf.if
// preserved around it.

// CHECK-LABEL: func.func @scf_if_hoist
// CHECK: ktdf.pipeline {
// CHECK:   ktdf.stage depends_in(none) depends_out(none) {
// CHECK:     scf.if %{{.*}} {
// CHECK:       ktdf.data_transfer
// CHECK:     }
// CHECK:   }
// CHECK: }

module {
  func.func @scf_if_hoist(%A: memref<?xf16, #ktdf.memory_space<HBM>>,
                          %M: index, %N: index, %cond: i1) {
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
            scf.if %cond {
              ktdf.data_transfer from %A[%m] size [64] to %lx_A[%c0] size [64]
                : memref<?xf16, #ktdf.memory_space<HBM>>, memref<64xf16, #ktdf.memory_space<LX>>
            }
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
