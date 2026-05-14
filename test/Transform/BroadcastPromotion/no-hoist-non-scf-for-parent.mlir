// RUN: scheduler --broadcast-promotion %s -allow-unregistered-dialect | FileCheck %s

// Pipeline sits directly inside scf.if -- scope walk caps there. Even
// though the transfer's source/dest are IV-free, no enclosing scf.for
// exists in the scope, so no hoist candidate.

// CHECK-LABEL:   func.func @no_hoist_non_scf_for
// CHECK-SAME:        %[[A:[^,]+]]: memref<?xf16, #ktdf.memory_space<HBM>>, %[[COND:[^,]+]]: i1) {
// CHECK-NEXT:     %[[C0:.*]] = arith.constant 0 : index
// CHECK-NEXT:     scf.if %[[COND]] {
// CHECK-NEXT:       ktdf.pipeline {
// CHECK-NEXT:         %[[PRIV:.*]]:2 = ktdf.private -> (memref<64xf16, #ktdf.memory_space<LX>>, !ktdf.token) {
// CHECK-NEXT:           %[[ORIG_LX:.*]] = memref.alloc() : memref<64xf16, #ktdf.memory_space<LX>>
// CHECK-NEXT:           %[[TOK:.*]] = ktdf.create_token : !ktdf.token
// CHECK-NEXT:           ktdf.private_yield %[[ORIG_LX]], %[[TOK]] : memref<64xf16, #ktdf.memory_space<LX>>, !ktdf.token
// CHECK-NEXT:         }
// CHECK-NEXT:         ktdf.stage depends_in(none) depends_out(%[[PRIV]]#1 : !ktdf.token) {
// CHECK-NEXT:           ktdf.data_transfer from %[[A]][%[[C0]]] size [64] to %[[PRIV]]#0[%[[C0]]] size [64] : memref<?xf16, #ktdf.memory_space<HBM>>, memref<64xf16, #ktdf.memory_space<LX>>
// CHECK-NEXT:         }
// CHECK-NEXT:         ktdf.stage depends_in(%[[PRIV]]#1 : !ktdf.token) depends_out(none) {
// CHECK-NEXT:           "test.use"(%[[PRIV]]#0) : (memref<64xf16, #ktdf.memory_space<LX>>) -> ()
// CHECK-NEXT:         }
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }

module {
  func.func @no_hoist_non_scf_for(%A: memref<?xf16, #ktdf.memory_space<HBM>>,
                                  %cond: i1) {
    %c0 = arith.constant 0 : index
    scf.if %cond {
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
    return
  }
}
