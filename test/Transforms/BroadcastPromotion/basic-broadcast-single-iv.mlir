// RUN: dataflow-scheduler-opt --broadcast-promotion %s -allow-unregistered-dialect | FileCheck %s

// A single DDR->L1 transfer invariant w.r.t. the innermost %n loop.
// The hoist lifts the transfer into a sibling pipeline placed
// immediately before `scf.for %n`. Full-output check: every line of
// the post-pass IR is matched by a CHECK directive in order.

// CHECK-LABEL:   func.func @single_iv_hoist
// CHECK-SAME:        %[[A:[^,]+]]: memref<?xf16, "DDR">, %[[M:[^,]+]]: index, %[[N:[^,]+]]: index) {
// CHECK-NEXT:     %[[C0:.*]] = arith.constant 0 : index
// CHECK-NEXT:     %[[C1:.*]] = arith.constant 1 : index
// CHECK-NEXT:     scf.for %[[IM:.*]] = %[[C0]] to %[[M]] step %[[C1]] {
// CHECK-NEXT:       %[[NEW_L1:.*]] = memref.alloc() : memref<64xf16, "L1">
// CHECK-NEXT:       ktdf.pipeline {
// CHECK-NEXT:         ktdf.stage depends_in(none) depends_out(none) {
// CHECK-NEXT:           ktdf.data_transfer from %[[A]][%[[IM]]] size [64] to %[[NEW_L1]][%[[C0]]] size [64] : memref<?xf16, "DDR">, memref<64xf16, "L1">
// CHECK-NEXT:         }
// CHECK-NEXT:       }
// CHECK-NEXT:       scf.for %[[IN:.*]] = %[[C0]] to %[[N]] step %[[C1]] {
// CHECK-NEXT:         ktdf.pipeline {
// CHECK-NEXT:           %[[PRIV:.*]]:2 = ktdf.private -> (memref<64xf16, "L1">, !ktdf.token) {
// CHECK-NEXT:             %[[ORIG_L1:.*]] = memref.alloc() : memref<64xf16, "L1">
// CHECK-NEXT:             %[[TOK:.*]] = ktdf.create_token : !ktdf.token
// CHECK-NEXT:             ktdf.private_yield %[[ORIG_L1]], %[[TOK]] : memref<64xf16, "L1">, !ktdf.token
// CHECK-NEXT:           }
// CHECK-NEXT:           ktdf.stage depends_in(none) depends_out(%[[PRIV]]#1) {
// CHECK-NEXT:           }
// CHECK-NEXT:           ktdf.stage depends_in(%[[PRIV]]#1) depends_out(none) {
// CHECK-NEXT:             "test.use"(%[[NEW_L1]]) : (memref<64xf16, "L1">) -> ()
// CHECK-NEXT:           }
// CHECK-NEXT:         }
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }

module {
  func.func @single_iv_hoist(%A: memref<?xf16, "DDR">,
                             %M: index, %N: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    scf.for %m = %c0 to %M step %c1 {
      scf.for %n = %c0 to %N step %c1 {
        ktdf.pipeline {
          %r:2 = ktdf.private -> (memref<64xf16, "L1">, !ktdf.token) {
            %a = memref.alloc() : memref<64xf16, "L1">
            %k = ktdf.create_token : !ktdf.token
            ktdf.private_yield %a, %k : memref<64xf16, "L1">, !ktdf.token
          }
          ktdf.stage depends_in(none) depends_out(%r#1) {
            ktdf.data_transfer from %A[%m] size [64] to %r#0[%c0] size [64]
              : memref<?xf16, "DDR">, memref<64xf16, "L1">
          }
          ktdf.stage depends_in(%r#1) depends_out(none) {
            "test.use"(%r#0) : (memref<64xf16, "L1">) -> ()
          }
        }
      }
    }
    return
  }
}
