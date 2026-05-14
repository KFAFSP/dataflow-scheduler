// RUN: scheduler --broadcast-promotion %s -allow-unregistered-dialect | FileCheck %s

// Destination is a FIFO slot. BroadcastPromotion v1 does not handle FIFO
// destinations; the transfer remains in place.

// CHECK-LABEL: func.func @no_hoist_fifo
// CHECK: scf.for %{{.*}} =
// CHECK:   scf.for %{{.*}} =
// CHECK:     ktdf.pipeline {
// CHECK:       ktdf.stage
// CHECK:         ktdf.data_transfer from %{{.*}} size [64] to %{{.*}} size [64] : memref<64xf16, #ktdf.memory_space<LX>>, !ktdf.fifo.slot<"lxlu-to-sfp", 64xf16>

module {
  func.func @no_hoist_fifo(%M: index, %N: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    scf.for %m = %c0 to %M step %c1 {
      scf.for %n = %c0 to %N step %c1 {
        ktdf.pipeline {
          %lx_A, %fifo_a, %t = ktdf.private -> (
              memref<64xf16, #ktdf.memory_space<LX>>,
              !ktdf.fifo.slot<"lxlu-to-sfp", 64xf16>,
              !ktdf.token) {
            %a = memref.alloc() : memref<64xf16, #ktdf.memory_space<LX>>
            %f = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"lxlu-to-sfp", 64xf16>
            %k = ktdf.create_token : !ktdf.token
            ktdf.private_yield %a, %f, %k
              : memref<64xf16, #ktdf.memory_space<LX>>,
                !ktdf.fifo.slot<"lxlu-to-sfp", 64xf16>,
                !ktdf.token
          }
          ktdf.stage depends_in(none) depends_out(%t : !ktdf.token) {
            // invariant-in-%n transfer but dest is FIFO -> skipped
            ktdf.data_transfer from %lx_A[%c0] size [64] to %fifo_a size [64]
              : memref<64xf16, #ktdf.memory_space<LX>>, !ktdf.fifo.slot<"lxlu-to-sfp", 64xf16>
          }
        }
      }
    }
    return
  }
}
