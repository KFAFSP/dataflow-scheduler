// RUN: dataflow-scheduler-opt --broadcast-promotion %s -allow-unregistered-dialect | FileCheck %s

// Source is invariant in %n, but a memref.store inside the %n loop
// writes to the source. Pipeline enclosing scope caps at %n (its body
// holds both the store and the pipeline), so no hoist target qualifies.

// CHECK-LABEL:   func.func @no_hoist_src_written
// CHECK-SAME:        %[[A:[^,]+]]: memref<?xf16, "DDR">, %[[VAL:[^,]+]]: f16, %[[M:[^,]+]]: index, %[[N:[^,]+]]: index) {
// CHECK-NEXT:     %[[C0:.*]] = arith.constant 0 : index
// CHECK-NEXT:     %[[C1:.*]] = arith.constant 1 : index
// CHECK-NEXT:     scf.for %[[IM:.*]] = %[[C0]] to %[[M]] step %[[C1]] {
// CHECK-NEXT:       scf.for %[[IN:.*]] = %[[C0]] to %[[N]] step %[[C1]] {
// CHECK-NEXT:         memref.store %[[VAL]], %[[A]][%[[IN]]] : memref<?xf16, "DDR">
// CHECK-NEXT:         ktdf.pipeline {
// CHECK-NEXT:           %[[PRIV:.*]]:2 = ktdf.private -> (memref<64xf16, "L1">, !ktdf.token) {
// CHECK-NEXT:             %[[ORIG_L1:.*]] = memref.alloc() : memref<64xf16, "L1">
// CHECK-NEXT:             %[[TOK:.*]] = ktdf.create_token : !ktdf.token
// CHECK-NEXT:             ktdf.private_yield %[[ORIG_L1]], %[[TOK]] : memref<64xf16, "L1">, !ktdf.token
// CHECK-NEXT:           }
// CHECK-NEXT:           ktdf.stage depends_in(none) depends_out(%[[PRIV]]#1) {
// CHECK-NEXT:             ktdf.data_transfer from %[[A]][%[[IM]]] size [64] to %[[PRIV]]#0[%[[C0]]] size [64] : memref<?xf16, "DDR">, memref<64xf16, "L1">
// CHECK-NEXT:           }
// CHECK-NEXT:           ktdf.stage depends_in(%[[PRIV]]#1) depends_out(none) {
// CHECK-NEXT:             "test.use"(%[[PRIV]]#0) : (memref<64xf16, "L1">) -> ()
// CHECK-NEXT:           }
// CHECK-NEXT:         }
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }

module {
  func.func @no_hoist_src_written(%A: memref<?xf16, "DDR">,
                                  %val: f16, %M: index, %N: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    scf.for %m = %c0 to %M step %c1 {
      scf.for %n = %c0 to %N step %c1 {
        memref.store %val, %A[%n] : memref<?xf16, "DDR">
        ktdf.pipeline {
          %l1_A, %t = ktdf.private -> (memref<64xf16, "L1">, !ktdf.token) {
            %a = memref.alloc() : memref<64xf16, "L1">
            %k = ktdf.create_token : !ktdf.token
            ktdf.private_yield %a, %k : memref<64xf16, "L1">, !ktdf.token
          }
          ktdf.stage depends_in(none) depends_out(%t) {
            ktdf.data_transfer from %A[%m] size [64] to %l1_A[%c0] size [64]
              : memref<?xf16, "DDR">, memref<64xf16, "L1">
          }
          ktdf.stage depends_in(%t) depends_out(none) {
            "test.use"(%l1_A) : (memref<64xf16, "L1">) -> ()
          }
        }
      }
    }
    return
  }
}
