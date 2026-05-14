// RUN: scheduler --broadcast-promotion %s -allow-unregistered-dialect | FileCheck %s

// Nested pipeline inside an outer stage (post-stage-coarsening shape).
// The outer stage is the scope boundary; hoist cannot escape it.
//
// The inner transfer is invariant in %n1 only, so it hoists to the block
// immediately inside the outer stage. Verify the sibling pipeline does
// NOT appear outside the outer stage.

// CHECK-LABEL: func.func @nested_boundary
// CHECK: ktdf.stage depends_in(none) depends_out(%{{.*}} : !ktdf.token) {
// CHECK:   scf.for %[[M1:.*]] = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK:     ktdf.pipeline {
// CHECK:     }
// CHECK:     scf.for %[[N1:.*]] = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK:       ktdf.pipeline {
// CHECK-NOT:     ktdf.data_transfer from %{{.*}}[%[[M1]]]
// CHECK:       }
// CHECK:     }
// CHECK:   }
// CHECK: }

module {
  func.func @nested_boundary(%A: memref<?xf16, #ktdf.memory_space<HBM>>,
                             %M: index, %N: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    ktdf.pipeline {
      %outer_t = ktdf.private -> (!ktdf.token) {
        %k = ktdf.create_token : !ktdf.token
        ktdf.private_yield %k : !ktdf.token
      }
      ktdf.stage depends_in(none) depends_out(%outer_t : !ktdf.token) {
        scf.for %m1 = %c0 to %M step %c1 {
          scf.for %n1 = %c0 to %N step %c1 {
            ktdf.pipeline {
              %inner_lx, %inner_t = ktdf.private -> (memref<64xf16, #ktdf.memory_space<LX>>, !ktdf.token) {
                %a = memref.alloc() : memref<64xf16, #ktdf.memory_space<LX>>
                %k = ktdf.create_token : !ktdf.token
                ktdf.private_yield %a, %k : memref<64xf16, #ktdf.memory_space<LX>>, !ktdf.token
              }
              ktdf.stage depends_in(none) depends_out(%inner_t : !ktdf.token) {
                ktdf.data_transfer from %A[%m1] size [64] to %inner_lx[%c0] size [64]
                  : memref<?xf16, #ktdf.memory_space<HBM>>, memref<64xf16, #ktdf.memory_space<LX>>
              }
              ktdf.stage depends_in(%inner_t : !ktdf.token) depends_out(none) {
                "test.use"(%inner_lx) : (memref<64xf16, #ktdf.memory_space<LX>>) -> ()
              }
            }
          }
        }
      }
    }
    return
  }
}
