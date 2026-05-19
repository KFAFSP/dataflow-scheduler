// RUN: scheduler -allow-unregistered-dialect %s -double-buffering | FileCheck %s

// Two LX buffers in the same pipeline. Expected: two alloc pairs, ONE
// buffer_phase shared across two select_memref ops, modulo(size: 2)
// once on the pipeline.

// CHECK-LABEL:   func.func @two_buffers
// CHECK:           memref.alloc()
// CHECK:           memref.alloc()
// CHECK:           memref.alloc()
// CHECK:           memref.alloc()
// CHECK:           scf.for %[[I:.+]] =
// CHECK:             %[[PHASE:.+]] = ktdf.buffer_phase(%[[I]]) {num_phases = 2 : i64} : index
// CHECK:             ktdf.select_memref %[[PHASE]][
// CHECK:             ktdf.select_memref %[[PHASE]][
// CHECK:             ktdf.pipeline modulo(size : 2)
// CHECK-NOT:         ktdf.buffer_phase

func.func @two_buffers(%a_src: memref<64xf16, #ktdf.memory_space<HBM>>,
                       %b_src: memref<64xf16, #ktdf.memory_space<HBM>>,
                       %dst: memref<64xf16, #ktdf.memory_space<HBM>>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  scf.for %i = %c0 to %c8 step %c1 {
    ktdf.pipeline {
      %lxa, %lxb, %t1, %t2, %t3 = ktdf.private -> (
          memref<64xf16, #ktdf.memory_space<LX>>,
          memref<64xf16, #ktdf.memory_space<LX>>,
          !ktdf.token, !ktdf.token, !ktdf.token) {
        %a = memref.alloc() : memref<64xf16, #ktdf.memory_space<LX>>
        %b = memref.alloc() : memref<64xf16, #ktdf.memory_space<LX>>
        %tk1 = ktdf.create_token : !ktdf.token
        %tk2 = ktdf.create_token : !ktdf.token
        %tk3 = ktdf.create_token : !ktdf.token
        ktdf.private_yield %a, %b, %tk1, %tk2, %tk3
          : memref<64xf16, #ktdf.memory_space<LX>>,
            memref<64xf16, #ktdf.memory_space<LX>>,
            !ktdf.token, !ktdf.token, !ktdf.token
      }
      ktdf.stage depends_in(none) depends_out(%t1 : !ktdf.token) {
        ktdf.data_transfer from %a_src[%c0] size [64] to %lxa[%c0] size [64]
          : memref<64xf16, #ktdf.memory_space<HBM>>,
            memref<64xf16, #ktdf.memory_space<LX>>
        ktdf.data_transfer from %b_src[%c0] size [64] to %lxb[%c0] size [64]
          : memref<64xf16, #ktdf.memory_space<HBM>>,
            memref<64xf16, #ktdf.memory_space<LX>>
      }
      ktdf.stage depends_in(%t1 : !ktdf.token) depends_out(%t2 : !ktdf.token) {
        ktdf.data_transfer from %lxa[%c0] size [64] to %dst[%c0] size [64]
          : memref<64xf16, #ktdf.memory_space<LX>>,
            memref<64xf16, #ktdf.memory_space<HBM>>
      }
      ktdf.stage depends_in(%t2 : !ktdf.token) depends_out(%t3 : !ktdf.token) {
        ktdf.data_transfer from %lxb[%c0] size [64] to %dst[%c0] size [64]
          : memref<64xf16, #ktdf.memory_space<LX>>,
            memref<64xf16, #ktdf.memory_space<HBM>>
      }
    }
  }
  return
}
