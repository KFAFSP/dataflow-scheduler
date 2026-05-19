// RUN: scheduler -allow-unregistered-dialect %s -double-buffering | FileCheck %s

// CHECK-NOT: ktdf.buffer_phase
// CHECK-NOT: ktdf.select_memref
// CHECK-NOT: modulo(size

func.func @non_alloc_defining_op(%src: memref<64xf16, #ktdf.memory_space<HBM>>,
                                  %dst: memref<64xf16, #ktdf.memory_space<HBM>>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  scf.for %i = %c0 to %c8 step %c1 {
    ktdf.pipeline {
      %lx, %t1, %t2 = ktdf.private -> (
          memref<*xf16, #ktdf.memory_space<LX>>,
          !ktdf.token, !ktdf.token) {
        %a = memref.alloc() : memref<64xf16, #ktdf.memory_space<LX>>
        %a_cast = memref.cast %a : memref<64xf16, #ktdf.memory_space<LX>> to memref<*xf16, #ktdf.memory_space<LX>>
        %tk1 = ktdf.create_token : !ktdf.token
        %tk2 = ktdf.create_token : !ktdf.token
        ktdf.private_yield %a_cast, %tk1, %tk2
          : memref<*xf16, #ktdf.memory_space<LX>>, !ktdf.token, !ktdf.token
      }
      ktdf.stage depends_in(none) depends_out(%t1 : !ktdf.token) {
        "test.write"(%lx) : (memref<*xf16, #ktdf.memory_space<LX>>) -> ()
      }
      ktdf.stage depends_in(%t1 : !ktdf.token) depends_out(%t2 : !ktdf.token) {
        "test.read"(%lx) : (memref<*xf16, #ktdf.memory_space<LX>>) -> ()
      }
    }
  }
  return
}
