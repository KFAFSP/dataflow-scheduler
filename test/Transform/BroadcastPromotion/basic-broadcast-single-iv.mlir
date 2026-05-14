// RUN: scheduler --broadcast-promotion %s -allow-unregistered-dialect | FileCheck %s

// A single DRAM->LX transfer invariant w.r.t. the innermost %n loop.
// The hoist should lift the transfer into a sibling pipeline placed
// immediately before `scf.for %n`.

// CHECK-LABEL: func.func @single_iv_hoist
// CHECK: scf.for %[[M:.*]] = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK:   %[[NEW_LX:.*]] = memref.alloc() : memref<64xf16, #ktdf.memory_space<LX>>
// CHECK:   ktdf.pipeline {
// CHECK:     ktdf.stage depends_in(none) depends_out(none) {
// CHECK:       ktdf.data_transfer from %{{.*}}[%[[M]]] size [64] to %[[NEW_LX]][%{{.*}}] size [64]
// CHECK:     }
// CHECK:   }
// CHECK:   scf.for %[[N:.*]] = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK:     ktdf.pipeline {
// CHECK-NOT:   ktdf.data_transfer from %{{.*}}[%[[M]]]

module {
  func.func @single_iv_hoist(%A: memref<?xf16, #ktdf.memory_space<HBM>>,
                             %M: index, %N: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    scf.for %m = %c0 to %M step %c1 {
      scf.for %n = %c0 to %N step %c1 {
        ktdf.pipeline {
          %r:2 = ktdf.private -> (memref<64xf16, #ktdf.memory_space<LX>>, !ktdf.token) {
            %a = memref.alloc() : memref<64xf16, #ktdf.memory_space<LX>>
            %k = ktdf.create_token : !ktdf.token
            ktdf.private_yield %a, %k : memref<64xf16, #ktdf.memory_space<LX>>, !ktdf.token
          }
          ktdf.stage depends_in(none) depends_out(%r#1 : !ktdf.token) {
            ktdf.data_transfer from %A[%m] size [64] to %r#0[%c0] size [64]
              : memref<?xf16, #ktdf.memory_space<HBM>>, memref<64xf16, #ktdf.memory_space<LX>>
          }
          ktdf.stage depends_in(%r#1 : !ktdf.token) depends_out(none) {
            "test.use"(%r#0) : (memref<64xf16, #ktdf.memory_space<LX>>) -> ()
          }
        }
      }
    }
    return
  }
}
