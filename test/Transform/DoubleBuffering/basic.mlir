// RUN: scheduler -allow-unregistered-dialect %s -double-buffering | FileCheck %s

// Single LX buffer with one producer stage and one consumer stage in a
// pipeline whose scope is one scf.for. Expected: one alloc pair hoisted
// outside the loop, one buffer_phase + one select_memref before the
// pipeline, modulo(size: 2) set on the pipeline, ktdf.private now only
// yielding the tokens.

// CHECK-LABEL:   func.func @basic
// CHECK:           %[[X0:.+]] = memref.alloc() : memref<64xf16, #ktdf.memory_space<LX>>
// CHECK:           %[[X1:.+]] = memref.alloc() : memref<64xf16, #ktdf.memory_space<LX>>
// CHECK:           scf.for %[[I:.+]] = %{{.+}} to %{{.+}} step %{{.+}}
// CHECK:             %[[PHASE:.+]] = ktdf.buffer_phase(%[[I]]) {num_phases = 2 : i64} : index
// CHECK:             %[[SEL:.+]] = ktdf.select_memref %[[PHASE]][%[[X0]], %[[X1]]] : index, memref<64xf16, #ktdf.memory_space<LX>>, memref<64xf16, #ktdf.memory_space<LX>> -> memref<64xf16, #ktdf.memory_space<LX>>
// CHECK:             ktdf.pipeline modulo(size : 2)
// CHECK:               ktdf.private -> (!ktdf.token, !ktdf.token)
// CHECK:               ktdf.stage depends_in(none) depends_out(
// CHECK:                 ktdf.data_transfer from %{{.+}}[%{{.+}}] size {{.*}}to %[[SEL]][%{{.+}}] size
// CHECK:               ktdf.stage depends_in
// CHECK:                 ktdf.data_transfer from %[[SEL]][%{{.+}}] size {{.*}}to %{{.+}}[%{{.+}}] size
// CHECK:           memref.dealloc %[[X0]] : memref<64xf16, #ktdf.memory_space<LX>>
// CHECK:           memref.dealloc %[[X1]] : memref<64xf16, #ktdf.memory_space<LX>>

func.func @basic(%src: memref<64xf16, #ktdf.memory_space<HBM>>,
                 %dst: memref<64xf16, #ktdf.memory_space<HBM>>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  scf.for %i = %c0 to %c8 step %c1 {
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
  }
  return
}
