// RUN: scheduler --broadcast-promotion %s -allow-unregistered-dialect | FileCheck %s

// Source is invariant in %n, but destination indices depend on %n.
// Under the strict "both source and destination must be invariant" rule,
// no hoist happens.

// CHECK-LABEL: func.func @no_hoist_dst
// CHECK: scf.for %[[M:.*]] =
// CHECK:   scf.for %[[N:.*]] =
// CHECK:     ktdf.pipeline {
// CHECK:       ktdf.stage depends_in(none) depends_out(%{{.*}} : !ktdf.token) {
// CHECK:         ktdf.data_transfer from %{{.*}}[%[[M]]] {{.*}} to %{{.*}}[%[[N]]]

module {
  func.func @no_hoist_dst(%A: memref<?xf16, #ktdf.memory_space<HBM>>,
                          %M: index, %N: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    scf.for %m = %c0 to %M step %c1 {
      scf.for %n = %c0 to %N step %c1 {
        ktdf.pipeline {
          %lx_A, %t = ktdf.private -> (memref<?xf16, #ktdf.memory_space<LX>>, !ktdf.token) {
            %a = memref.alloc(%N) : memref<?xf16, #ktdf.memory_space<LX>>
            %k = ktdf.create_token : !ktdf.token
            ktdf.private_yield %a, %k : memref<?xf16, #ktdf.memory_space<LX>>, !ktdf.token
          }
          ktdf.stage depends_in(none) depends_out(%t : !ktdf.token) {
            ktdf.data_transfer from %A[%m] size [64] to %lx_A[%n] size [64]
              : memref<?xf16, #ktdf.memory_space<HBM>>, memref<?xf16, #ktdf.memory_space<LX>>
          }
          ktdf.stage depends_in(%t : !ktdf.token) depends_out(none) {
            "test.use"(%lx_A) : (memref<?xf16, #ktdf.memory_space<LX>>) -> ()
          }
        }
      }
    }
    return
  }
}
