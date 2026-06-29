// Running the pass twice produces the same output as running once. The
// second run sees the pre-existing modulo and skips.

// RUN: dataflow-scheduler-opt -allow-unregistered-dialect %s -double-buffering > %t.once.mlir
// RUN: dataflow-scheduler-opt -allow-unregistered-dialect %t.once.mlir -double-buffering > %t.twice.mlir
// RUN: diff %t.once.mlir %t.twice.mlir

func.func @idempotent(%src: memref<64xf16, "DDR">,
                      %dst: memref<64xf16, "DDR">) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  scf.for %i = %c0 to %c8 step %c1 {
    ktdf.pipeline {
      %l1, %t1, %t2 = ktdf.private -> (
          memref<64xf16, "L1">,
          !ktdf.token, !ktdf.token) {
        %a = memref.alloc() : memref<64xf16, "L1">
        %tk1 = ktdf.create_token : !ktdf.token
        %tk2 = ktdf.create_token : !ktdf.token
        ktdf.private_yield %a, %tk1, %tk2
          : memref<64xf16, "L1">, !ktdf.token, !ktdf.token
      }
      ktdf.stage depends_in(none) depends_out(%t1) {
        ktdf.data_transfer from %src[%c0] size [64] to %l1[%c0] size [64]
          : memref<64xf16, "DDR">,
            memref<64xf16, "L1">
      }
      ktdf.stage depends_in(%t1) depends_out(%t2) {
        ktdf.data_transfer from %l1[%c0] size [64] to %dst[%c0] size [64]
          : memref<64xf16, "L1">,
            memref<64xf16, "DDR">
      }
    }
  }
  return
}
