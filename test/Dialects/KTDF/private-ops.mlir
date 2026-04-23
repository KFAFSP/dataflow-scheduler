// RUN: scheduler %s | scheduler | FileCheck %s

// Round trip test for ktdf.private and ktdf.private_yield operations

// CHECK-LABEL:   func.func @private_with_token_and_memref() {
// CHECK-NEXT:     ktdf.pipeline {
// CHECK-NEXT:       %[[VAL_0:.*]]:2 = ktdf.private -> (memref<64xf16, "SFP_REG">, !ktdf.token) {
// CHECK-NEXT:         %[[VAL_1:.*]] = memref.alloc() : memref<64xf16, "SFP_REG">
// CHECK-NEXT:         %[[VAL_2:.*]] = ktdf.create_token : !ktdf.token
// CHECK-NEXT:         ktdf.private_yield %[[VAL_1]], %[[VAL_2]] : memref<64xf16, "SFP_REG">, !ktdf.token
// CHECK-NEXT:       }
// CHECK-NEXT:       ktdf.stage depends_in(%[[VAL_0]]#1 : !ktdf.token) depends_out(none) {
// CHECK-NEXT:         %[[VAL_3:.*]] = arith.constant 0 : index
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }

module {
  func.func @private_with_token_and_memref() {
    ktdf.pipeline {
      %results:2 = ktdf.private -> (memref<64xf16, "SFP_REG">, !ktdf.token) {
        %sfp_A = memref.alloc() : memref<64xf16, "SFP_REG">
        %t = ktdf.create_token : !ktdf.token
        ktdf.private_yield %sfp_A, %t : memref<64xf16, "SFP_REG">, !ktdf.token
      }

      ktdf.stage depends_in(%results#1 : !ktdf.token) depends_out(none) {
        %c0 = arith.constant 0 : index
      }
    }
    return
  }
}