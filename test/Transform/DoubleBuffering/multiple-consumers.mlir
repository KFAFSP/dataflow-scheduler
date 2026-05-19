// RUN: scheduler -allow-unregistered-dialect %s -double-buffering | FileCheck %s

// One producer stage, two consumer stages reading the same buffer.

// CHECK-LABEL:   func.func @two_consumers
// CHECK:           memref.alloc() : memref<64xf16, #ktdf.memory_space<LX>>
// CHECK:           memref.alloc() : memref<64xf16, #ktdf.memory_space<LX>>
// CHECK:           %[[SEL:.+]] = ktdf.select_memref
// CHECK:           ktdf.pipeline modulo(size : 2)
// CHECK:             ktdf.stage depends_in(none) depends_out
// CHECK:               ktdf.data_transfer {{.*}}to %[[SEL]]
// CHECK:             ktdf.stage
// CHECK:               ktdf.data_transfer from %[[SEL]]
// CHECK:             ktdf.stage
// CHECK:               ktdf.data_transfer from %[[SEL]]

func.func @two_consumers(%src: memref<64xf16, #ktdf.memory_space<HBM>>,
                         %dst1: memref<64xf16, #ktdf.memory_space<HBM>>,
                         %dst2: memref<64xf16, #ktdf.memory_space<HBM>>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  scf.for %i = %c0 to %c8 step %c1 {
    ktdf.pipeline {
      %lx, %t1, %t2, %t3 = ktdf.private -> (
          memref<64xf16, #ktdf.memory_space<LX>>,
          !ktdf.token, !ktdf.token, !ktdf.token) {
        %a = memref.alloc() : memref<64xf16, #ktdf.memory_space<LX>>
        %tk1 = ktdf.create_token : !ktdf.token
        %tk2 = ktdf.create_token : !ktdf.token
        %tk3 = ktdf.create_token : !ktdf.token
        ktdf.private_yield %a, %tk1, %tk2, %tk3
          : memref<64xf16, #ktdf.memory_space<LX>>, !ktdf.token, !ktdf.token, !ktdf.token
      }
      ktdf.stage depends_in(none) depends_out(%t1 : !ktdf.token) {
        ktdf.data_transfer from %src[%c0] size [64] to %lx[%c0] size [64]
          : memref<64xf16, #ktdf.memory_space<HBM>>,
            memref<64xf16, #ktdf.memory_space<LX>>
      }
      ktdf.stage depends_in(%t1 : !ktdf.token) depends_out(%t2 : !ktdf.token) {
        ktdf.data_transfer from %lx[%c0] size [64] to %dst1[%c0] size [64]
          : memref<64xf16, #ktdf.memory_space<LX>>,
            memref<64xf16, #ktdf.memory_space<HBM>>
      }
      ktdf.stage depends_in(%t1 : !ktdf.token) depends_out(%t3 : !ktdf.token) {
        ktdf.data_transfer from %lx[%c0] size [64] to %dst2[%c0] size [64]
          : memref<64xf16, #ktdf.memory_space<LX>>,
            memref<64xf16, #ktdf.memory_space<HBM>>
      }
    }
  }
  return
}
