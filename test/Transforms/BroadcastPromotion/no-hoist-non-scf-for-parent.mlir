// RUN: dataflow-scheduler-opt --broadcast-promotion %s -allow-unregistered-dialect | FileCheck %s

// Pipeline sits directly inside scf.if -- scope walk caps there. Even
// though the transfer's source/dest are IV-free, no enclosing scf.for
// exists in the scope, so no hoist candidate.

// CHECK-LABEL:   func.func @no_hoist_non_scf_for
// CHECK-SAME:        %[[A:[^,]+]]: memref<?xf16, "DDR">, %[[COND:[^,]+]]: i1) {
// CHECK-NEXT:     %[[C0:.*]] = arith.constant 0 : index
// CHECK-NEXT:     scf.if %[[COND]] {
// CHECK-NEXT:       ktdf.pipeline {
// CHECK-NEXT:         %[[PRIV:.*]]:2 = ktdf.private -> (memref<64xf16, "L1">, !ktdf.token) {
// CHECK-NEXT:           %[[ORIG_L1:.*]] = memref.alloc() : memref<64xf16, "L1">
// CHECK-NEXT:           %[[TOK:.*]] = ktdf.create_token : !ktdf.token
// CHECK-NEXT:           ktdf.private_yield %[[ORIG_L1]], %[[TOK]] : memref<64xf16, "L1">, !ktdf.token
// CHECK-NEXT:         }
// CHECK-NEXT:         ktdf.stage depends_in(none) depends_out(%[[PRIV]]#1) {
// CHECK-NEXT:           ktdf.data_transfer from %[[A]][%[[C0]]] size [64] to %[[PRIV]]#0[%[[C0]]] size [64] : memref<?xf16, "DDR">, memref<64xf16, "L1">
// CHECK-NEXT:         }
// CHECK-NEXT:         ktdf.stage depends_in(%[[PRIV]]#1) depends_out(none) {
// CHECK-NEXT:           "test.use"(%[[PRIV]]#0) : (memref<64xf16, "L1">) -> ()
// CHECK-NEXT:         }
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }

module {
  func.func @no_hoist_non_scf_for(%A: memref<?xf16, "DDR">,
                                  %cond: i1) {
    %c0 = arith.constant 0 : index
    scf.if %cond {
      ktdf.pipeline {
        %l1_A, %t = ktdf.private -> (memref<64xf16, "L1">, !ktdf.token) {
          %a = memref.alloc() : memref<64xf16, "L1">
          %k = ktdf.create_token : !ktdf.token
          ktdf.private_yield %a, %k : memref<64xf16, "L1">, !ktdf.token
        }
        ktdf.stage depends_in(none) depends_out(%t) {
          ktdf.data_transfer from %A[%c0] size [64] to %l1_A[%c0] size [64]
            : memref<?xf16, "DDR">, memref<64xf16, "L1">
        }
        ktdf.stage depends_in(%t) depends_out(none) {
          "test.use"(%l1_A) : (memref<64xf16, "L1">) -> ()
        }
      }
    }
    return
  }
}
