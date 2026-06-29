// RUN: dataflow-scheduler-opt --broadcast-promotion %s -allow-unregistered-dialect | FileCheck %s

// Source is invariant in %n, but destination indices depend on %n.
// Under the strict "both source and destination must be invariant" rule,
// no hoist happens. IR after pass should be byte-equivalent to input.

// CHECK-LABEL:   func.func @no_hoist_dst
// CHECK-SAME:        %[[A:[^,]+]]: memref<?xf16, "DDR">, %[[M:[^,]+]]: index, %[[N:[^,]+]]: index) {
// CHECK-NEXT:     %[[C0:.*]] = arith.constant 0 : index
// CHECK-NEXT:     %[[C1:.*]] = arith.constant 1 : index
// CHECK-NEXT:     scf.for %[[IM:.*]] = %[[C0]] to %[[M]] step %[[C1]] {
// CHECK-NEXT:       scf.for %[[IN:.*]] = %[[C0]] to %[[N]] step %[[C1]] {
// CHECK-NEXT:         ktdf.pipeline {
// CHECK-NEXT:           %[[PRIV:.*]]:2 = ktdf.private -> (memref<?xf16, "L1">, !ktdf.token) {
// CHECK-NEXT:             %[[ORIG_L1:.*]] = memref.alloc(%[[N]]) : memref<?xf16, "L1">
// CHECK-NEXT:             %[[TOK:.*]] = ktdf.create_token : !ktdf.token
// CHECK-NEXT:             ktdf.private_yield %[[ORIG_L1]], %[[TOK]] : memref<?xf16, "L1">, !ktdf.token
// CHECK-NEXT:           }
// CHECK-NEXT:           ktdf.stage depends_in(none) depends_out(%[[PRIV]]#1) {
// CHECK-NEXT:             ktdf.data_transfer from %[[A]][%[[IM]]] size [64] to %[[PRIV]]#0[%[[IN]]] size [64] : memref<?xf16, "DDR">, memref<?xf16, "L1">
// CHECK-NEXT:           }
// CHECK-NEXT:           ktdf.stage depends_in(%[[PRIV]]#1) depends_out(none) {
// CHECK-NEXT:             "test.use"(%[[PRIV]]#0) : (memref<?xf16, "L1">) -> ()
// CHECK-NEXT:           }
// CHECK-NEXT:         }
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }

module {
  func.func @no_hoist_dst(%A: memref<?xf16, "DDR">,
                          %M: index, %N: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    scf.for %m = %c0 to %M step %c1 {
      scf.for %n = %c0 to %N step %c1 {
        ktdf.pipeline {
          %l1_A, %t = ktdf.private -> (memref<?xf16, "L1">, !ktdf.token) {
            %a = memref.alloc(%N) : memref<?xf16, "L1">
            %k = ktdf.create_token : !ktdf.token
            ktdf.private_yield %a, %k : memref<?xf16, "L1">, !ktdf.token
          }
          ktdf.stage depends_in(none) depends_out(%t) {
            ktdf.data_transfer from %A[%m] size [64] to %l1_A[%n] size [64]
              : memref<?xf16, "DDR">, memref<?xf16, "L1">
          }
          ktdf.stage depends_in(%t) depends_out(none) {
            "test.use"(%l1_A) : (memref<?xf16, "L1">) -> ()
          }
        }
      }
    }
    return
  }
}
