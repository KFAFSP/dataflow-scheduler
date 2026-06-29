// RUN: dataflow-scheduler-opt --broadcast-promotion %s | FileCheck %s

// Pipeline with only a ktdf.private and no stages. The pass completes
// without crashing; IR is unchanged.

// CHECK-LABEL:   func.func @no_stages() {
// CHECK-NEXT:     %[[C0:.*]] = arith.constant 0 : index
// CHECK-NEXT:     %[[C1:.*]] = arith.constant 1 : index
// CHECK-NEXT:     scf.for %[[IM:.*]] = %[[C0]] to %[[C1]] step %[[C1]] {
// CHECK-NEXT:       ktdf.pipeline {
// CHECK-NEXT:         %[[PRIV:.*]]:2 = ktdf.private -> (memref<64xf16, "L1">, !ktdf.token) {
// CHECK-NEXT:           %[[BUF:.*]] = memref.alloc() : memref<64xf16, "L1">
// CHECK-NEXT:           %[[TOK:.*]] = ktdf.create_token : !ktdf.token
// CHECK-NEXT:           ktdf.private_yield %[[BUF]], %[[TOK]] : memref<64xf16, "L1">, !ktdf.token
// CHECK-NEXT:         }
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }

module {
  func.func @no_stages() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    scf.for %m = %c0 to %c1 step %c1 {
      ktdf.pipeline {
        %a, %t = ktdf.private -> (memref<64xf16, "L1">, !ktdf.token) {
          %buf = memref.alloc() : memref<64xf16, "L1">
          %tok = ktdf.create_token : !ktdf.token
          ktdf.private_yield %buf, %tok : memref<64xf16, "L1">, !ktdf.token
        }
      }
    }
    return
  }
}
