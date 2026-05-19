// RUN: scheduler -allow-unregistered-dialect %s -double-buffering | FileCheck %s

// CHECK-NOT: ktdf.buffer_phase
// CHECK-NOT: ktdf.select_memref
// CHECK-NOT: modulo(size

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
