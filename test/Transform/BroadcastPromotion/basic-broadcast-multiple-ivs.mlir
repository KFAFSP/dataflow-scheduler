// RUN: scheduler --broadcast-promotion %s -allow-unregistered-dialect | FileCheck %s

// A transfer whose source has no IV dependency (constant offset) should
// hoist past both %m and %n to the outermost scope.

// CHECK-LABEL: func.func @multi_iv_hoist
// CHECK: %[[NEW_LX:.*]] = memref.alloc() : memref<64xf16, #ktdf.memory_space<LX>>
// CHECK: ktdf.pipeline {
// CHECK:   ktdf.stage depends_in(none) depends_out(none) {
// CHECK:     ktdf.data_transfer
// CHECK:   }
// CHECK: }
// CHECK: scf.for %[[M:.*]] =
// CHECK:   scf.for %[[N:.*]] =

module {
  func.func @multi_iv_hoist(%A: memref<?xf16, #ktdf.memory_space<HBM>>,
                            %M: index, %N: index) {
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
            ktdf.data_transfer from %A[%c0] size [64] to %lx_A[%c0] size [64]
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
