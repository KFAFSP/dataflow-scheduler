// RUN: scheduler -allow-unregistered-dialect %s -double-buffering | FileCheck %s

// CHECK-NOT: ktdf.buffer_phase
// CHECK-NOT: ktdf.select_memref
// CHECK-NOT: modulo(size

func.func @dynamic_size_not_dominating(%src: memref<?xf16, #ktdf.memory_space<HBM>>,
                                        %dst: memref<?xf16, #ktdf.memory_space<HBM>>,
                                        %M: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  scf.for %i = %c0 to %c8 step %c1 {
    // Compute size INSIDE the loop — does not dominate the hoist target.
    %size = arith.addi %i, %M : index
    ktdf.pipeline {
      %lx, %t1, %t2 = ktdf.private -> (
          memref<?xf16, #ktdf.memory_space<LX>>,
          !ktdf.token, !ktdf.token) {
        %a = memref.alloc(%size) : memref<?xf16, #ktdf.memory_space<LX>>
        %tk1 = ktdf.create_token : !ktdf.token
        %tk2 = ktdf.create_token : !ktdf.token
        ktdf.private_yield %a, %tk1, %tk2
          : memref<?xf16, #ktdf.memory_space<LX>>, !ktdf.token, !ktdf.token
      }
      ktdf.stage depends_in(none) depends_out(%t1 : !ktdf.token) {
        ktdf.data_transfer from %src[%c0] size [%size] to %lx[%c0] size [%size]
          : memref<?xf16, #ktdf.memory_space<HBM>>,
            memref<?xf16, #ktdf.memory_space<LX>>
      }
      ktdf.stage depends_in(%t1 : !ktdf.token) depends_out(%t2 : !ktdf.token) {
        ktdf.data_transfer from %lx[%c0] size [%size] to %dst[%c0] size [%size]
          : memref<?xf16, #ktdf.memory_space<LX>>,
            memref<?xf16, #ktdf.memory_space<HBM>>
      }
    }
  }
  return
}
