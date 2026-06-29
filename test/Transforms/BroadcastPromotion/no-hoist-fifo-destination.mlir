// RUN: dataflow-scheduler-opt --broadcast-promotion %s -allow-unregistered-dialect | FileCheck %s

// Destination is a FIFO slot. BroadcastPromotion v1 does not handle FIFO
// destinations; the transfer remains in place. IR after pass should be
// byte-equivalent to input.

// CHECK-LABEL:   func.func @no_hoist_fifo
// CHECK-SAME:        %[[M:[^,]+]]: index, %[[N:[^,]+]]: index) {
// CHECK-NEXT:     %[[C0:.*]] = arith.constant 0 : index
// CHECK-NEXT:     %[[C1:.*]] = arith.constant 1 : index
// CHECK-NEXT:     scf.for %[[IM:.*]] = %[[C0]] to %[[M]] step %[[C1]] {
// CHECK-NEXT:       scf.for %[[IN:.*]] = %[[C0]] to %[[N]] step %[[C1]] {
// CHECK-NEXT:         ktdf.pipeline {
// CHECK-NEXT:           %[[PRIV:.*]]:3 = ktdf.private -> (memref<64xf16, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>, !ktdf.token) {
// CHECK-NEXT:             %[[ORIG_L1:.*]] = memref.alloc() : memref<64xf16, "L1">
// CHECK-NEXT:             %[[ORIG_FIFO:.*]] = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
// CHECK-NEXT:             %[[TOK:.*]] = ktdf.create_token : !ktdf.token
// CHECK-NEXT:             ktdf.private_yield %[[ORIG_L1]], %[[ORIG_FIFO]], %[[TOK]] : memref<64xf16, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>, !ktdf.token
// CHECK-NEXT:           }
// CHECK-NEXT:           ktdf.stage depends_in(none) depends_out(%[[PRIV]]#2) {
// CHECK-NEXT:             ktdf.data_transfer from %[[PRIV]]#0[%[[C0]]] size [64] to %[[PRIV]]#1 size [64] : memref<64xf16, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
// CHECK-NEXT:           }
// CHECK-NEXT:         }
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }

module {
  func.func @no_hoist_fifo(%M: index, %N: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    scf.for %m = %c0 to %M step %c1 {
      scf.for %n = %c0 to %N step %c1 {
        ktdf.pipeline {
          %l1_A, %fifo_a, %t = ktdf.private -> (
              memref<64xf16, "L1">,
              !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>,
              !ktdf.token) {
            %a = memref.alloc() : memref<64xf16, "L1">
            %f = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
            %k = ktdf.create_token : !ktdf.token
            ktdf.private_yield %a, %f, %k
              : memref<64xf16, "L1">,
                !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>,
                !ktdf.token
          }
          ktdf.stage depends_in(none) depends_out(%t) {
            // invariant-in-%n transfer but dest is FIFO -> skipped
            ktdf.data_transfer from %l1_A[%c0] size [64] to %fifo_a size [64]
              : memref<64xf16, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
          }
        }
      }
    }
    return
  }
}
