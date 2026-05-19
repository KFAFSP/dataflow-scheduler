// RUN: scheduler -allow-unregistered-dialect %s -double-buffering | FileCheck %s

// CHECK-LABEL:   func.func @no_scope(
// CHECK-SAME:        %[[SRC:[a-zA-Z0-9_]+]]: memref<64xf16, #ktdf.memory_space<HBM>>, %[[DST:[a-zA-Z0-9_]+]]: memref<64xf16, #ktdf.memory_space<HBM>>) {
// CHECK-NEXT:      %[[C0:.+]] = arith.constant 0 : index
// CHECK-NEXT:      ktdf.pipeline {
// CHECK-NEXT:        %[[TKS:.+]]:3 = ktdf.private -> (memref<64xf16, #ktdf.memory_space<LX>>, !ktdf.token, !ktdf.token) {
// CHECK-NEXT:          %[[ALLOC:.+]] = memref.alloc() : memref<64xf16, #ktdf.memory_space<LX>>
// CHECK-NEXT:          %[[T1:.+]] = ktdf.create_token : !ktdf.token
// CHECK-NEXT:          %[[T2:.+]] = ktdf.create_token : !ktdf.token
// CHECK-NEXT:          ktdf.private_yield %[[ALLOC]], %[[T1]], %[[T2]] : memref<64xf16, #ktdf.memory_space<LX>>, !ktdf.token, !ktdf.token
// CHECK-NEXT:        }
// CHECK-NEXT:        ktdf.stage depends_in(none) depends_out(%[[TKS]]#1 : !ktdf.token) {
// CHECK-NEXT:          ktdf.data_transfer from %[[SRC]][%[[C0]]] size [64] to %[[TKS]]#0[%[[C0]]] size [64] : memref<64xf16, #ktdf.memory_space<HBM>>, memref<64xf16, #ktdf.memory_space<LX>>
// CHECK-NEXT:        }
// CHECK-NEXT:        ktdf.stage depends_in(%[[TKS]]#1 : !ktdf.token) depends_out(%[[TKS]]#2 : !ktdf.token) {
// CHECK-NEXT:          ktdf.data_transfer from %[[TKS]]#0[%[[C0]]] size [64] to %[[DST]][%[[C0]]] size [64] : memref<64xf16, #ktdf.memory_space<LX>>, memref<64xf16, #ktdf.memory_space<HBM>>
// CHECK-NEXT:        }
// CHECK-NEXT:      }
// CHECK-NEXT:      return
// CHECK-NEXT:    }

func.func @no_scope(%src: memref<64xf16, #ktdf.memory_space<HBM>>,
                   %dst: memref<64xf16, #ktdf.memory_space<HBM>>) {
  %c0 = arith.constant 0 : index
  ktdf.pipeline {
    %lx, %t1, %t2 = ktdf.private -> (
        memref<64xf16, #ktdf.memory_space<LX>>,
        !ktdf.token, !ktdf.token) {
      %a = memref.alloc() : memref<64xf16, #ktdf.memory_space<LX>>
      %tk1 = ktdf.create_token : !ktdf.token
      %tk2 = ktdf.create_token : !ktdf.token
      ktdf.private_yield %a, %tk1, %tk2
        : memref<64xf16, #ktdf.memory_space<LX>>, !ktdf.token, !ktdf.token
    }
    ktdf.stage depends_in(none) depends_out(%t1 : !ktdf.token) {
      ktdf.data_transfer from %src[%c0] size [64] to %lx[%c0] size [64]
        : memref<64xf16, #ktdf.memory_space<HBM>>,
          memref<64xf16, #ktdf.memory_space<LX>>
    }
    ktdf.stage depends_in(%t1 : !ktdf.token) depends_out(%t2 : !ktdf.token) {
      ktdf.data_transfer from %lx[%c0] size [64] to %dst[%c0] size [64]
        : memref<64xf16, #ktdf.memory_space<LX>>,
          memref<64xf16, #ktdf.memory_space<HBM>>
    }
  }
  return
}
