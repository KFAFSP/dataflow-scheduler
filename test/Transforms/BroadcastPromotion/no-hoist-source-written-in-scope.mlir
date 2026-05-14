// RUN: scheduler --broadcast-promotion %s -allow-unregistered-dialect | FileCheck %s

// Source is invariant in %n, but a memref.store inside the %n loop
// writes to the source. Pipeline enclosing scope caps at %n (because
// the loop body has the pipeline plus the store), so no hoist target
// qualifies.

// CHECK-LABEL: func.func @no_hoist_src_written
// CHECK: scf.for %{{.*}} =
// CHECK:   scf.for %{{.*}} =
// CHECK:     memref.store
// CHECK:     ktdf.pipeline {
// CHECK:       ktdf.stage depends_in(none) depends_out(%{{.*}} : !ktdf.token) {
// CHECK:         ktdf.data_transfer

module {
  func.func @no_hoist_src_written(%A: memref<?xf16, #ktdf.memory_space<HBM>>,
                                  %val: f16, %M: index, %N: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    scf.for %m = %c0 to %M step %c1 {
      scf.for %n = %c0 to %N step %c1 {
        memref.store %val, %A[%n] : memref<?xf16, #ktdf.memory_space<HBM>>
        ktdf.pipeline {
          %lx_A, %t = ktdf.private -> (memref<64xf16, #ktdf.memory_space<LX>>, !ktdf.token) {
            %a = memref.alloc() : memref<64xf16, #ktdf.memory_space<LX>>
            %k = ktdf.create_token : !ktdf.token
            ktdf.private_yield %a, %k : memref<64xf16, #ktdf.memory_space<LX>>, !ktdf.token
          }
          ktdf.stage depends_in(none) depends_out(%t : !ktdf.token) {
            ktdf.data_transfer from %A[%m] size [64] to %lx_A[%c0] size [64]
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
